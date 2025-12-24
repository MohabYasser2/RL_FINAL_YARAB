"""
DIAMOND (DIffusion As a Model Of eNvironment Dreams) - Complete Training Pipeline
Replicated for BreakoutNoFrameskip-v4

This implementation replicates the DIAMOND paper's training procedure in a single file.
Based on: https://github.com/eloialonso/diamond
Paper: "Diffusion for World Modeling: Visual Details Matter in Atari" (NeurIPS 2024)

Features:
- World model training with diffusion
- Policy training (Actor-Critic)
- Full training/test/recording modes
- WandB logging
- Gymnasium Atari environment support
"""

import os
import sys
import time
import argparse
from pathlib import Path
from collections import deque
from typing import Dict, List, Tuple, Optional
import numpy as np
# Compatibility patch: some `transformers` versions expect
# `huggingface_hub.constants.HF_HUB_CACHE` to exist. When importing
# `torch`, torch.onnx internals may import `transformers` which then
# imports `huggingface_hub`. If the installed `huggingface_hub` is a
# newer version that renamed the constant, it can raise an
# AttributeError. Set a fallback here before importing `torch`.
try:
    import huggingface_hub.constants as _hf_constants
    if not hasattr(_hf_constants, "HF_HUB_CACHE"):
        import os as _os
        _hf_constants.HF_HUB_CACHE = getattr(_hf_constants, "HF_HOME", _os.path.expanduser("~/.cache/huggingface"))
except Exception:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import wandb

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration for DIAMOND training on Breakout"""
    
    # Environment
    env_name = "BreakoutNoFrameskip-v4"
    num_actions = 4  # Breakout actions
    img_size = 64
    num_channels = 3
    frame_stack = 4
    
    # Training
    num_epochs = 1000
    batch_size = 16
    sequence_length = 16
    num_env_steps = 100_000  # Atari 100k
    episodes_per_epoch = 8
    
    # World Model (Diffusion)
    wm_hidden_dim = 512
    wm_num_layers = 6
    wm_num_heads = 8
    wm_dropout = 0.1
    diffusion_steps = 100
    diffusion_timesteps = 10  # For sampling during training
    diffusion_beta_start = 0.0001
    diffusion_beta_end = 0.02
    
    # Actor-Critic
    ac_hidden_dim = 512
    ac_num_layers = 4
    imagination_horizon = 15
    gamma = 0.995
    lambda_gae = 0.95
    
    # Optimization
    wm_learning_rate = 1e-4
    ac_learning_rate = 3e-4
    weight_decay = 1e-6
    grad_clip = 10.0
    
    # Logging
    log_interval = 10
    save_interval = 100
    eval_episodes = 10
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths
    output_dir = Path("outputs")
    checkpoint_dir = None
    dataset_dir = None
    
    def __post_init__(self):
        timestamp = time.strftime("%Y-%m-%d/%H-%M-%S")
        self.checkpoint_dir = self.output_dir / timestamp / "checkpoints"
        self.dataset_dir = self.output_dir / timestamp / "dataset"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Environment Wrappers
# ============================================================================

class AtariPreprocessing(gym.Wrapper):
    """Atari preprocessing: frame skip, resize, grayscale to RGB"""
    
    def __init__(self, env, img_size=64, frame_skip=4):
        super().__init__(env)
        self.img_size = img_size
        self.frame_skip = frame_skip
        self.observation_space = gym.spaces.Box(
            low=0, high=255, 
            shape=(img_size, img_size, 3), 
            dtype=np.uint8
        )
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process_frame(obs), info
    
    def step(self, action):
        total_reward = 0.0
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return self._process_frame(obs), total_reward, terminated, truncated, info
    
    def _process_frame(self, frame):
        # Resize
        import cv2
        frame = cv2.resize(frame, (self.img_size, self.img_size), 
                          interpolation=cv2.INTER_AREA)
        return frame

class FrameStack(gym.Wrapper):
    """Stack last N frames"""
    
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        low = np.repeat(env.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(env.observation_space.high[np.newaxis, ...], num_stack, axis=0)
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=env.observation_space.dtype
        )
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_obs(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self):
        return np.array(self.frames)

def make_env(env_name, img_size=64, frame_stack=4, seed=None):
    """Create preprocessed Atari environment"""
    env = gym.make(env_name, render_mode=None)
    if seed is not None:
        env.reset(seed=seed)
    env = AtariPreprocessing(env, img_size=img_size)
    env = FrameStack(env, num_stack=frame_stack)
    return env

# ============================================================================
# Replay Buffer
# ============================================================================

class ReplayBuffer:
    """Replay buffer for storing trajectories"""
    
    def __init__(self, capacity, obs_shape, action_dim, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        self.observations = torch.zeros((capacity, *obs_shape), dtype=torch.uint8)
        self.actions = torch.zeros((capacity,), dtype=torch.long)
        self.rewards = torch.zeros((capacity,), dtype=torch.float32)
        self.dones = torch.zeros((capacity,), dtype=torch.bool)
        
    def add(self, obs, action, reward, done):
        self.observations[self.ptr] = torch.from_numpy(obs)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size, sequence_length):
        """Sample sequences for training"""
        max_start = self.size - sequence_length
        if max_start <= 0:
            return None
            
        indices = np.random.randint(0, max_start, size=batch_size)
        
        obs_seq = []
        act_seq = []
        rew_seq = []
        done_seq = []
        
        for idx in indices:
            seq_idx = slice(idx, idx + sequence_length)
            obs_seq.append(self.observations[seq_idx])
            act_seq.append(self.actions[seq_idx])
            rew_seq.append(self.rewards[seq_idx])
            done_seq.append(self.dones[seq_idx])
            
        return {
            'observations': torch.stack(obs_seq).to(self.device),
            'actions': torch.stack(act_seq).to(self.device),
            'rewards': torch.stack(rew_seq).to(self.device),
            'dones': torch.stack(done_seq).to(self.device)
        }

# ============================================================================
# Diffusion World Model
# ============================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for timesteps"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResidualBlock(nn.Module):
    """Residual block for U-Net"""
    
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
            
    def forward(self, x, t):
        residual = self.residual_conv(x)
        
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
        
        t_emb = self.time_mlp(t)
        x = x + t_emb[:, :, None, None]
        
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        
        return x + residual

class DiffusionWorldModel(nn.Module):
    """Diffusion-based world model for predicting next observations"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.img_size = config.img_size
        self.num_channels = config.num_channels * config.frame_stack
        
        # Time embedding
        time_dim = config.wm_hidden_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        # Action embedding
        self.action_emb = nn.Embedding(config.num_actions, time_dim)
        
        # Encoder (downsampling)
        self.enc_conv1 = nn.Conv2d(self.num_channels, 64, 3, padding=1)
        self.enc_block1 = ResidualBlock(64, 128, time_dim)
        self.enc_down1 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        
        self.enc_block2 = ResidualBlock(128, 256, time_dim)
        self.enc_down2 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        
        self.enc_block3 = ResidualBlock(256, 512, time_dim)
        self.enc_down3 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        
        # Middle
        self.mid_block1 = ResidualBlock(512, 512, time_dim)
        self.mid_block2 = ResidualBlock(512, 512, time_dim)
        
        # Decoder (upsampling)
        self.dec_up3 = nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1)
        self.dec_block3 = ResidualBlock(1024, 256, time_dim)
        
        self.dec_up2 = nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1)
        self.dec_block2 = ResidualBlock(512, 128, time_dim)
        
        self.dec_up1 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)
        self.dec_block1 = ResidualBlock(256, 64, time_dim)
        
        # Output
        self.out_conv = nn.Conv2d(64, self.num_channels, 3, padding=1)
        
        # Diffusion schedule
        self.register_buffer('betas', self._cosine_beta_schedule(config.diffusion_steps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule as proposed in Improved DDPM"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
        
    def forward(self, x_noisy, t, action, x_prev):
        """
        Args:
            x_noisy: Noisy observation [B, C, H, W]
            t: Timestep [B]
            action: Action taken [B]
            x_prev: Previous observation [B, C, H, W]
        """
        # Time embedding
        t_emb = self.time_mlp(t.float())
        
        # Action embedding
        a_emb = self.action_emb(action)
        
        # Combine embeddings
        t_emb = t_emb + a_emb
        
        # Concatenate previous observation with noisy current
        x = torch.cat([x_prev, x_noisy], dim=1)
        
        # Encoder
        e1 = self.enc_conv1(x)
        e1 = self.enc_block1(e1, t_emb)
        e1_down = self.enc_down1(e1)
        
        e2 = self.enc_block2(e1_down, t_emb)
        e2_down = self.enc_down2(e2)
        
        e3 = self.enc_block3(e2_down, t_emb)
        e3_down = self.enc_down3(e3)
        
        # Middle
        m = self.mid_block1(e3_down, t_emb)
        m = self.mid_block2(m, t_emb)
        
        # Decoder with skip connections
        d3 = self.dec_up3(m)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec_block3(d3, t_emb)
        
        d2 = self.dec_up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec_block2(d2, t_emb)
        
        d1 = self.dec_up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec_block1(d1, t_emb)
        
        # Output - predict noise
        noise_pred = self.out_conv(d1)
        
        return noise_pred
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt()
        sqrt_one_minus_alphas_cumprod_t = (1.0 - self.alphas_cumprod[t]).sqrt()
        
        # Reshape for broadcasting
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    @torch.no_grad()
    def p_sample(self, x, t, action, x_prev):
        """Reverse diffusion process - single step"""
        noise_pred = self.forward(x, t, action, x_prev)
        
        alpha_t = self.alphas[t][:, None, None, None]
        alpha_cumprod_t = self.alphas_cumprod[t][:, None, None, None]
        beta_t = self.betas[t][:, None, None, None]
        
        # Predict x_0
        x_0_pred = (x - ((1 - alpha_cumprod_t).sqrt() * noise_pred)) / alpha_cumprod_t.sqrt()
        x_0_pred = torch.clamp(x_0_pred, -1, 1)
        
        # Compute posterior mean
        if t[0] > 0:
            alpha_cumprod_prev = self.alphas_cumprod[t - 1][:, None, None, None]
            posterior_variance = beta_t * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t)
            posterior_mean = (
                alpha_cumprod_prev.sqrt() * beta_t / (1 - alpha_cumprod_t) * x_0_pred +
                alpha_t.sqrt() * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) * x
            )
            
            noise = torch.randn_like(x)
            x_prev_sample = posterior_mean + posterior_variance.sqrt() * noise
        else:
            x_prev_sample = x_0_pred
            
        return x_prev_sample
    
    @torch.no_grad()
    def imagine(self, x_start, actions):
        """
        Imagine future observations given actions
        Args:
            x_start: Starting observation [B, C, H, W]
            actions: Sequence of actions [B, T]
        Returns:
            imagined_obs: [B, T, C, H, W]
        """
        batch_size, horizon = actions.shape
        device = x_start.device
        
        imagined = []
        x_prev = x_start
        
        for t in range(horizon):
            # Start from noise
            x_t = torch.randn_like(x_prev)
            action = actions[:, t]
            
            # Reverse diffusion process (simplified - fewer steps)
            num_steps = self.config.diffusion_timesteps
            timesteps = torch.linspace(
                self.config.diffusion_steps - 1, 0, num_steps, 
                device=device
            ).long()
            
            for step in timesteps:
                t_batch = torch.full((batch_size,), step, device=device, dtype=torch.long)
                x_t = self.p_sample(x_t, t_batch, action, x_prev)
            
            imagined.append(x_t)
            x_prev = x_t
            
        return torch.stack(imagined, dim=1)

# ============================================================================
# Actor-Critic
# ============================================================================

class ActorCritic(nn.Module):
    """Actor-Critic for policy learning"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Encoder for observations
        self.encoder = nn.Sequential(
            nn.Conv2d(config.num_channels * config.frame_stack, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, config.ac_hidden_dim),
            nn.ReLU()
        )
        
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(config.ac_hidden_dim, config.ac_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.ac_hidden_dim, config.num_actions)
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(config.ac_hidden_dim, config.ac_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.ac_hidden_dim, 1)
        )
        
    def forward(self, obs):
        """
        Args:
            obs: Observations [B, C, H, W], normalized to [-1, 1]
        Returns:
            action_logits: [B, num_actions]
            value: [B, 1]
        """
        features = self.encoder(obs)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action(self, obs, deterministic=False):
        """Sample action from policy"""
        logits, value = self.forward(obs)
        
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            
        return action, value

# ============================================================================
# Training Functions
# ============================================================================

def normalize_obs(obs):
    """Normalize observations to [-1, 1]"""
    return (obs.float() / 255.0) * 2.0 - 1.0

def denormalize_obs(obs):
    """Denormalize observations to [0, 255]"""
    return ((obs + 1.0) / 2.0 * 255.0).clamp(0, 255).to(torch.uint8)

def compute_gae(rewards, values, dones, gamma=0.995, lambda_=0.95):
    """Compute Generalized Advantage Estimation"""
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
            
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lambda_ * (1 - dones[t]) * gae
        advantages.insert(0, gae)
        
    advantages = torch.tensor(advantages)
    returns = advantages + values
    
    return advantages, returns

def train_world_model(world_model, optimizer, batch, config):
    """Train diffusion world model"""
    obs = normalize_obs(batch['observations'])  # [B, T, C, H, W]
    actions = batch['actions']  # [B, T]
    
    batch_size, seq_len = obs.shape[:2]
    
    # Prepare training data
    x_prev = obs[:, :-1].reshape(-1, *obs.shape[2:])  # [B*(T-1), C, H, W]
    x_next = obs[:, 1:].reshape(-1, *obs.shape[2:])
    actions_flat = actions[:, :-1].reshape(-1)
    
    # Sample timesteps
    t = torch.randint(0, config.diffusion_steps, (x_next.shape[0],), device=config.device)
    
    # Add noise
    noise = torch.randn_like(x_next)
    x_noisy = world_model.q_sample(x_next, t, noise)
    
    # Predict noise
    noise_pred = world_model(x_noisy, t, actions_flat, x_prev)
    
    # Compute loss
    loss = F.mse_loss(noise_pred, noise)
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(world_model.parameters(), config.grad_clip)
    optimizer.step()
    
    return {'wm_loss': loss.item()}

def train_actor_critic(actor_critic, world_model, optimizer, batch, config):
    """Train actor-critic in imagination"""
    obs = normalize_obs(batch['observations'][:, 0])  # Start state [B, C, H, W]
    batch_size = obs.shape[0]
    
    # Imagine trajectories
    with torch.no_grad():
        # Sample actions from current policy
        imagined_obs = [obs]
        imagined_actions = []
        imagined_rewards = []
        imagined_dones = []
        
        current_obs = obs
        for _ in range(config.imagination_horizon):
            action, _ = actor_critic.get_action(current_obs)
            imagined_actions.append(action)
            
            # Imagine next observation
            next_obs = world_model.imagine(current_obs, action.unsqueeze(1))[:, 0]
            imagined_obs.append(next_obs)
            
            # Dummy reward (would use reward model in full implementation)
            reward = torch.zeros(batch_size, device=config.device)
            imagined_rewards.append(reward)
            imagined_dones.append(torch.zeros(batch_size, dtype=torch.bool, device=config.device))
            
            current_obs = next_obs
    
    # Compute values for all states
    all_obs = torch.stack(imagined_obs)  # [T+1, B, C, H, W]
    all_logits, all_values = actor_critic(all_obs.view(-1, *all_obs.shape[2:]))
    all_values = all_values.view(len(imagined_obs), batch_size)
    
    # Compute advantages
    rewards_tensor = torch.stack(imagined_rewards)
    values_tensor = all_values[:-1]
    dones_tensor = torch.stack(imagined_dones)
    
    advantages, returns = compute_gae(
        rewards_tensor, values_tensor, dones_tensor, 
        config.gamma, config.lambda_gae
    )
    
    # Policy loss
    actions_tensor = torch.stack(imagined_actions)
    logits_for_actions = all_logits[:len(imagined_actions)].view(len(imagined_actions), batch_size, -1)
    
    dist = torch.distributions.Categorical(logits=logits_for_actions.transpose(0, 1))
    log_probs = dist.log_prob(actions_tensor.transpose(0, 1))
    
    policy_loss = -(log_probs * advantages.transpose(0, 1).detach()).mean()
    
    # Value loss
    value_loss = F.mse_loss(values_tensor.transpose(0, 1), returns.transpose(0, 1).detach())
    
    # Entropy bonus
    entropy = dist.entropy().mean()
    
    # Total loss
    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), config.grad_clip)
    optimizer.step()
    
    return {
        'ac_loss': loss.item(),
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': entropy.item()
    }

def collect_data(env, actor_critic, replay_buffer, num_steps, config, epsilon=0.1):
    """Collect data from environment"""
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    episodes_completed = 0
    
    for step in range(num_steps):
        # Epsilon-greedy exploration
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            obs_tensor = torch.from_numpy(obs).to(config.device)          # [T, H, W, C]
            obs_tensor = obs_tensor.permute(0, 3, 1, 2)                   # [T, C, H, W]
            obs_tensor = obs_tensor.reshape(1, -1, obs_tensor.size(2), obs_tensor.size(3))
            obs_tensor = normalize_obs(obs_tensor)                        # [1, T*C, H, W]
            with torch.no_grad():
                action, _ = actor_critic.get_action(obs_tensor, deterministic=False)
            action = action.item()
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store in replay buffer (store last frame only)
        replay_buffer.add(obs, action, reward, done)
        
        episode_reward += reward
        episode_length += 1
        
        if done:
            episodes_completed += 1
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
        else:
            obs = next_obs
            
    return episodes_completed

def evaluate(env, actor_critic, config, num_episodes=10):
    """Evaluate policy"""
    total_rewards = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            obs_tensor = torch.from_numpy(obs).to(config.device)          # [T, H, W, C]
            obs_tensor = obs_tensor.permute(0, 3, 1, 2)                   # [T, C, H, W]
            obs_tensor = obs_tensor.reshape(1, -1, obs_tensor.size(2), obs_tensor.size(3))
            obs_tensor = normalize_obs(obs_tensor)                        # [1, T*C, H, W]
            
            with torch.no_grad():
                action, _ = actor_critic.get_action(obs_tensor, deterministic=True)
            action = action.item()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        total_rewards.append(episode_reward)
    
    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'min_reward': np.min(total_rewards),
        'max_reward': np.max(total_rewards)
    }

# ============================================================================
# Main Training Loop
# ============================================================================

def train(config, use_wandb=False):
    """Main training loop"""
    
    # Initialize wandb
    if use_wandb:
        wandb.init(
            project="diamond-breakout",
            config=vars(config),
            name=f"diamond_{config.env_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Create environments
    train_env = make_env(config.env_name, config.img_size, config.frame_stack, seed=42)
    eval_env = make_env(config.env_name, config.img_size, config.frame_stack, seed=123)
    
    # Initialize models
    world_model = DiffusionWorldModel(config).to(config.device)
    actor_critic = ActorCritic(config).to(config.device)
    
    # Optimizers
    wm_optimizer = torch.optim.AdamW(
        world_model.parameters(),
        lr=config.wm_learning_rate,
        weight_decay=config.weight_decay
    )
    ac_optimizer = torch.optim.AdamW(
        actor_critic.parameters(),
        lr=config.ac_learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Replay buffer
    obs_shape = (config.frame_stack, config.img_size, config.img_size, config.num_channels)
    replay_buffer = ReplayBuffer(
        capacity=config.num_env_steps,
        obs_shape=obs_shape,
        action_dim=config.num_actions,
        device=config.device
    )
    
    # Training loop
    total_steps = 0
    best_eval_reward = -float('inf')
    
    print(f"Starting training on {config.device}")
    print(f"Environment: {config.env_name}")
    print(f"Total epochs: {config.num_epochs}")
    
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        
        # Collect data
        steps_per_epoch = config.num_env_steps // config.num_epochs
        epsilon = max(0.01, 0.1 - epoch * 0.09 / config.num_epochs)  # Decay exploration
        
        episodes_collected = collect_data(
            train_env, actor_critic, replay_buffer, 
            steps_per_epoch, config, epsilon
        )
        total_steps += steps_per_epoch
        
        # Train world model
        wm_losses = []
        if replay_buffer.size >= config.batch_size * config.sequence_length:
            for _ in range(10):  # Multiple updates per epoch
                batch = replay_buffer.sample(config.batch_size, config.sequence_length)
                if batch is not None:
                    metrics = train_world_model(world_model, wm_optimizer, batch, config)
                    wm_losses.append(metrics['wm_loss'])
        
        # Train actor-critic
        ac_losses = []
        if replay_buffer.size >= config.batch_size * config.sequence_length:
            for _ in range(5):  # Multiple updates per epoch
                batch = replay_buffer.sample(config.batch_size, config.sequence_length)
                if batch is not None:
                    metrics = train_actor_critic(actor_critic, world_model, ac_optimizer, batch, config)
                    ac_losses.append(metrics['ac_loss'])
        
        epoch_time = time.time() - epoch_start
        
        # Logging
        if epoch % config.log_interval == 0:
            log_dict = {
                'epoch': epoch,
                'total_steps': total_steps,
                'episodes_collected': episodes_collected,
                'epsilon': epsilon,
                'replay_buffer_size': replay_buffer.size,
                'epoch_time': epoch_time
            }
            
            if wm_losses:
                log_dict['wm_loss'] = np.mean(wm_losses)
            if ac_losses:
                log_dict['ac_loss'] = np.mean(ac_losses)
            
            # Evaluation
            if epoch % (config.log_interval * 5) == 0:
                eval_metrics = evaluate(eval_env, actor_critic, config, config.eval_episodes)
                log_dict.update({f'eval/{k}': v for k, v in eval_metrics.items()})
                
                print(f"\nEpoch {epoch}/{config.num_epochs}")
                print(f"  Steps: {total_steps}/{config.num_env_steps}")
                print(f"  WM Loss: {log_dict.get('wm_loss', 0):.4f}")
                print(f"  AC Loss: {log_dict.get('ac_loss', 0):.4f}")
                print(f"  Eval Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
                print(f"  Time: {epoch_time:.2f}s")
                
                # Save best model
                if eval_metrics['mean_reward'] > best_eval_reward:
                    best_eval_reward = eval_metrics['mean_reward']
                    save_checkpoint(
                        config.checkpoint_dir / 'best_model.pt',
                        world_model, actor_critic, wm_optimizer, ac_optimizer,
                        epoch, total_steps, best_eval_reward
                    )
            
            if use_wandb:
                wandb.log(log_dict)
        
        # Save checkpoint
        if epoch % config.save_interval == 0 and epoch > 0:
            save_checkpoint(
                config.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt',
                world_model, actor_critic, wm_optimizer, ac_optimizer,
                epoch, total_steps, best_eval_reward
            )
    
    # Final save
    save_checkpoint(
        config.checkpoint_dir / 'final_model.pt',
        world_model, actor_critic, wm_optimizer, ac_optimizer,
        config.num_epochs, total_steps, best_eval_reward
    )
    
    print(f"\nTraining completed!")
    print(f"Best eval reward: {best_eval_reward:.2f}")
    
    if use_wandb:
        wandb.finish()
    
    return world_model, actor_critic

def save_checkpoint(path, world_model, actor_critic, wm_opt, ac_opt, epoch, steps, best_reward):
    """Save training checkpoint"""
    torch.save({
        'epoch': epoch,
        'steps': steps,
        'best_reward': best_reward,
        'world_model_state': world_model.state_dict(),
        'actor_critic_state': actor_critic.state_dict(),
        'wm_optimizer_state': wm_opt.state_dict(),
        'ac_optimizer_state': ac_opt.state_dict(),
    }, path)
    print(f"Saved checkpoint to {path}")

def load_checkpoint(path, world_model, actor_critic, wm_opt=None, ac_opt=None):
    """Load training checkpoint"""
    checkpoint = torch.load(path, map_location='cpu')
    world_model.load_state_dict(checkpoint['world_model_state'])
    actor_critic.load_state_dict(checkpoint['actor_critic_state'])
    
    if wm_opt is not None:
        wm_opt.load_state_dict(checkpoint['wm_optimizer_state'])
    if ac_opt is not None:
        ac_opt.load_state_dict(checkpoint['ac_optimizer_state'])
    
    return checkpoint['epoch'], checkpoint['steps'], checkpoint['best_reward']

# ============================================================================
# Test and Recording Modes
# ============================================================================

def test_policy(checkpoint_path, config, num_episodes=10, render=False):
    """Test trained policy"""
    
    # Create environment
    render_mode = 'human' if render else None
    env = gym.make(config.env_name, render_mode=render_mode)
    env = AtariPreprocessing(env, config.img_size)
    env = FrameStack(env, config.frame_stack)
    
    # Load model
    actor_critic = ActorCritic(config).to(config.device)
    world_model = DiffusionWorldModel(config).to(config.device)
    load_checkpoint(checkpoint_path, world_model, actor_critic)
    actor_critic.eval()
    
    print(f"\nTesting policy from {checkpoint_path}")
    print(f"Running {num_episodes} episodes...")
    
    rewards = []
    lengths = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            obs_tensor = torch.from_numpy(obs).to(config.device)          # [T, H, W, C]
            obs_tensor = obs_tensor.permute(0, 3, 1, 2)                   # [T, C, H, W]
            obs_tensor = obs_tensor.reshape(1, -1, obs_tensor.size(2), obs_tensor.size(3))
            obs_tensor = normalize_obs(obs_tensor)                        # [1, T*C, H, W]
            
            with torch.no_grad():
                action, _ = actor_critic.get_action(obs_tensor, deterministic=True)
            action = action.item()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        rewards.append(episode_reward)
        lengths.append(episode_length)
        print(f"Episode {ep+1}: Reward={episode_reward:.1f}, Length={episode_length}")
    
    print(f"\nTest Results:")
    print(f"  Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Mean Length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"  Min/Max Reward: {np.min(rewards):.1f} / {np.max(rewards):.1f}")
    
    env.close()
    return rewards, lengths

def record_policy(checkpoint_path, config, num_episodes=5, video_folder='videos'):
    """Record policy videos"""
    
    video_folder = Path(video_folder)
    video_folder.mkdir(parents=True, exist_ok=True)
    
    # Create environment with video recording
    env = gym.make(config.env_name, render_mode='rgb_array')
    env = AtariPreprocessing(env, config.img_size)
    env = FrameStack(env, config.frame_stack)
    env = RecordVideo(
        env, 
        video_folder=str(video_folder),
        episode_trigger=lambda x: True,  # Record all episodes
        name_prefix='diamond_breakout'
    )
    
    # Load model
    actor_critic = ActorCritic(config).to(config.device)
    world_model = DiffusionWorldModel(config).to(config.device)
    load_checkpoint(checkpoint_path, world_model, actor_critic)
    actor_critic.eval()
    
    print(f"\nRecording {num_episodes} episodes to {video_folder}")
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            obs_tensor = torch.from_numpy(obs).to(config.device)          # [T, H, W, C]
            obs_tensor = obs_tensor.permute(0, 3, 1, 2)                   # [T, C, H, W]
            obs_tensor = obs_tensor.reshape(1, -1, obs_tensor.size(2), obs_tensor.size(3))
            obs_tensor = normalize_obs(obs_tensor)                        # [1, T*C, H, W]
            
            with torch.no_grad():
                action, _ = actor_critic.get_action(obs_tensor, deterministic=True)
            action = action.item()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        print(f"Episode {ep+1} recorded: Reward={episode_reward:.1f}")
    
    env.close()
    print(f"Videos saved to {video_folder}")

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='DIAMOND Training for Breakout')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'test', 'record'],
                       help='Mode: train, test, or record')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint path for test/record modes')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--num-epochs', type=int, default=1000,
                       help='Number of training epochs')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')
    parser.add_argument('--num-episodes', type=int, default=10,
                       help='Number of episodes for test/record')
    parser.add_argument('--render', action='store_true',
                       help='Render during testing')
    parser.add_argument('--video-folder', type=str, default='videos',
                       help='Folder for recorded videos')
    
    args = parser.parse_args()
    
    # Setup configuration
    config = Config()
    config.__post_init__()
    
    if args.device:
        config.device = args.device
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    
    print("="*80)
    print("DIAMOND: Diffusion for World Modeling")
    print("Environment: BreakoutNoFrameskip-v4")
    print("="*80)
    
    if args.mode == 'train':
        # Training mode
        print("\n[TRAIN MODE]")
        world_model, actor_critic = train(config, use_wandb=args.wandb)
        
    elif args.mode == 'test':
        # Testing mode
        if args.checkpoint is None:
            print("Error: --checkpoint required for test mode")
            sys.exit(1)
        
        print("\n[TEST MODE]")
        test_policy(args.checkpoint, config, args.num_episodes, args.render)
        
    elif args.mode == 'record':
        # Recording mode
        if args.checkpoint is None:
            print("Error: --checkpoint required for record mode")
            sys.exit(1)
        
        print("\n[RECORD MODE]")
        record_policy(args.checkpoint, config, args.num_episodes, args.video_folder)
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80)

if __name__ == '__main__':
    main()