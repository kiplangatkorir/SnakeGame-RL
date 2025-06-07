# Snake Reinforcement Learning Project

## Overview
A Python implementation of Snake game with Deep Q-Learning (DQN) agent that learns to:
- Navigate around obstacles
- Distinguish between different food types
- Maximize score efficiently

## Features
- **Game Engine**: Custom Snake game with:
  - Obstacles
  - Multiple food types (normal, bonus, poison)
  - Realistic snake rendering
- **RL Agent**:
  - Deep Q-Network (DQN) with experience replay
  - Target network for stable training
  - Epsilon-greedy exploration
- **Visualization**: Real-time training metrics plotting

## Requirements
- Python 3.8+
- Pygame
- PyTorch
- NumPy
- Matplotlib

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. **Training**:
```bash
python train.py
```
   - Shows real-time training progress
   - Saves best model as `snake_rl.pth`

2. **Key Parameters**:
- State size: 14 features
- Actions: 3 (straight, right, left)
- Memory size: 100,000
- Batch size: 1024

## Project Structure
- `snake_game.py`: Game implementation
- `rl_agent.py`: DQN agent
- `train.py`: Training script
- `requirements.txt`: Dependencies

## Future Improvements
- Add more complex obstacles
- Implement Double DQN
- Add human vs AI comparison mode
- Optimize hyperparameters further
