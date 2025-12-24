# DIAMOND Implementation for Breakout

Complete implementation of DIAMOND (DIffusion As a Model Of eNvironment Dreams) for BreakoutNoFrameskip-v4 environment.

**Based on:** [DIAMOND GitHub Repository](https://github.com/eloialonso/diamond)  
**Paper:** "Diffusion for World Modeling: Visual Details Matter in Atari" (NeurIPS 2024 Spotlight)

## Overview

This implementation provides a single-file Python script that replicates the DIAMOND training procedure with:
- **Diffusion World Model**: U-Net based diffusion model for learning environment dynamics
- **Actor-Critic Policy**: Policy learning entirely in imagination
- **Full Pipeline**: Training, testing, and recording modes
- **WandB Integration**: Comprehensive logging and tracking

## Features

✅ Complete training pipeline in one file  
✅ Diffusion-based world model with cosine noise schedule  
✅ Imagination-based policy learning (Actor-Critic)  
✅ Atari 100k training regime  
✅ WandB logging and metrics tracking  
✅ Video recording of trained agents  
✅ Checkpoint saving and resuming  
✅ Test mode for evaluation  

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone or download the files
mkdir diamond_project
cd diamond_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Atari ROMs (required)
pip install "gym[accept-rom-license]"
```

## Requirements

Create a `requirements.txt` file with:

```
numpy>=1.21.0
torch>=2.0.0
gymnasium>=0.29.0
gymnasium[atari]>=0.29.0
opencv-python>=4.8.0
wandb>=0.15.0
ale-py>=0.8.0
autorom[accept-rom-license]>=0.4.2
```

## Usage

### 1. Training Mode

Train DIAMOND on Breakout with default hyperparameters:

```bash
python diamond_breakout.py --mode train --wandb
```

**Arguments:**
- `--mode train`: Training mode
- `--wandb`: Enable Weights & Biases logging (optional)
- `--num-epochs`: Number of training epochs (default: 1000)
- `--device`: Device to use (cuda/cpu)

**Example with custom settings:**
```bash
python diamond_breakout.py --mode train --wandb --num-epochs 500 --device cuda:0
```

### 2. Test Mode

Evaluate a trained model:

```bash
python diamond_breakout.py --mode test --checkpoint outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/best_model.pt --num-episodes 10 --render
```

**Arguments:**
- `--mode test`: Testing mode
- `--checkpoint`: Path to checkpoint file (required)
- `--num-episodes`: Number of episodes to run (default: 10)
- `--render`: Show visualization (optional)

### 3. Record Mode

Record videos of the trained agent:

```bash
python diamond_breakout.py --mode record --checkpoint outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/best_model.pt --num-episodes 5 --video-folder videos
```

**Arguments:**
- `--mode record`: Recording mode
- `--checkpoint`: Path to checkpoint file (required)
- `--num-episodes`: Number of episodes to record (default: 5)
- `--video-folder`: Output folder for videos (default: "videos")

## Project Structure

After training, your project will look like:

```
diamond_project/
├── diamond_breakout.py          # Main implementation file
├── requirements.txt             # Python dependencies
├── outputs/                     # Training outputs
│   └── YYYY-MM-DD/
│       └── HH-MM-SS/
│           ├── checkpoints/     # Model checkpoints
│           │   ├── best_model.pt
│           │   ├── final_model.pt
│           │   └── checkpoint_epoch_*.pt
│           └── dataset/         # Collected data
└── videos/                      # Recorded videos
```

## Model Architecture

### Diffusion World Model
- **Architecture**: U-Net with residual blocks
- **Input**: Previous observation + action
- **Output**: Next observation (denoised)
- **Diffusion Steps**: 100 (training), 10 (sampling)
- **Schedule**: Cosine beta schedule

### Actor-Critic
- **Encoder**: CNN (3 conv layers + flatten)
- **Actor Head**: 2-layer MLP → action logits
- **Critic Head**: 2-layer MLP → value estimate
- **Training**: GAE with imagination rollouts

## Hyperparameters

Key hyperparameters (as per DIAMOND paper for Atari 100k):

```python
# Environment
img_size = 64
frame_stack = 4

# Training
num_epochs = 1000
batch_size = 16
sequence_length = 16
num_env_steps = 100_000  # Atari 100k

# World Model
wm_hidden_dim = 512
diffusion_steps = 100
diffusion_timesteps = 10

# Actor-Critic
ac_hidden_dim = 512
imagination_horizon = 15
gamma = 0.995
lambda_gae = 0.95

# Optimization
wm_learning_rate = 1e-4
ac_learning_rate = 3e-4
```

## WandB Logging

The implementation tracks:
- Training losses (world model, actor-critic)
- Evaluation metrics (mean/std reward)
- Episode statistics
- Training progress
- Replay buffer size

To use WandB:
1. Create account at [wandb.ai](https://wandb.ai)
2. Run `wandb login`
3. Add `--wandb` flag when training

## Expected Results

On BreakoutNoFrameskip-v4 with Atari 100k:
- Training time: ~6-8 hours on RTX 3090
- Expected reward: 15-30+ (varies with seed)
- Episodes for convergence: ~200-400

## Checkpoints

Checkpoints are saved:
- `best_model.pt`: Best performing model on evaluation
- `final_model.pt`: Final model after all epochs
- `checkpoint_epoch_N.pt`: Periodic checkpoints every 100 epochs

Each checkpoint contains:
```python
{
    'epoch': int,
    'steps': int,
    'best_reward': float,
    'world_model_state': state_dict,
    'actor_critic_state': state_dict,
    'wm_optimizer_state': state_dict,
    'ac_optimizer_state': state_dict
}
```

## Paper Requirements

For your assignment paper, include:

1. **GitHub Link**: Link to your repository with this implementation
2. **Recorded Video**: Upload videos to WandB and include link
3. **Experiments**: Include training curves (WM loss, AC loss, eval rewards)
4. **Architecture**: Describe the diffusion world model and actor-critic
5. **Hyperparameters**: Document all settings used
6. **Results**: Report final performance metrics

### WandB Report
Create a WandB report with:
- Training curves
- Evaluation metrics
- Recorded videos
- Hyperparameter settings

Share the report link in your paper.

## Troubleshooting

### CUDA Out of Memory
Reduce batch size or sequence length:
```python
config.batch_size = 8
config.sequence_length = 12
```

### Slow Training
- Reduce `diffusion_timesteps` for faster sampling
- Use fewer imagination rollouts
- Reduce network sizes

### Poor Performance
- Train longer (more epochs)
- Increase replay buffer size
- Adjust learning rates
- Try different seeds

## Comparison with Model-Free Methods

For the bonus (2pt), compare with:
- **DDQN**: Deep Q-Network with double Q-learning
- **SAC**: Soft Actor-Critic
- **PPO**: Proximal Policy Optimization

Run model-free baselines and compare:
- Sample efficiency (reward vs. environment steps)
- Final performance
- Training time
- Computational cost

## Citation

If you use this implementation, cite the original DIAMOND paper:

```bibtex
@inproceedings{alonso2024diffusionworldmodelingvisual,
    title={Diffusion for World Modeling: Visual Details Matter in Atari},
    author={Eloi Alonso and Adam Jelley and Vincent Micheli and Anssi Kanervisto and Amos Storkey and Tim Pearce and François Fleuret},
    booktitle={Thirty-eighth Conference on Neural Information Processing Systems},
    year={2024},
    url={https://arxiv.org/abs/2405.12399}
}
```

## License

MIT License (following original DIAMOND repository)

## Additional Resources

- **Original Repo**: https://github.com/eloialonso/diamond
- **Paper**: https://arxiv.org/abs/2405.12399
- **Project Page**: https://diamond-wm.github.io
- **Atari Docs**: https://ale.farama.org/

## Support

For issues or questions:
1. Check the original DIAMOND repository
2. Review the paper for algorithmic details
3. Consult Atari Gym documentation for environment issues

---

**Note**: This implementation is designed for educational purposes to replicate the DIAMOND training procedure. For production use or research, refer to the official repository.