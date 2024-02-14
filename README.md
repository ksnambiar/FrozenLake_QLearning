# FrozenLake Q-Learning Project

This project implements Q-Learning algorithms to solve the FrozenLake environment from OpenAI Gym. Two different approaches were used: a standard Q-Learning algorithm and a Deep Q-Network (DQN) implemented using PyTorch.

## Environment Description

The FrozenLake environment is a simple grid-world environment provided by OpenAI Gym. The agent must navigate through a grid of frozen and slippery tiles to reach a goal tile without falling into holes. The task is considered solved when the agent reaches the goal tile.
[https://www.gymlibrary.dev/environments/toy_text/frozen_lake/]

## Algorithms Implemented

1. **Q-Learning**: A standard Q-Learning algorithm was implemented to solve the FrozenLake environment. Q-Learning is a model-free reinforcement learning algorithm that learns to estimate the value of taking a particular action in a particular state.
2. **Deep Q-Network (DQN)**: A Deep Q-Learning algorithm was implemented using PyTorch. DQN is an extension of Q-Learning that utilizes deep neural networks to approximate the Q-values, allowing for better generalization and performance on complex environments.

## Files and Directory Structure

- `ql_agent.py`: Contains the implementation of the standard Q-Learning algorithm.
- `deepQonFrozenLake.py`: Contains the implementation of the Deep Q-Network using PyTorch.
- `README.md`: This file, provides an overview of the project.

## Requirements

- Python 3.x
- OpenAI Gym
- NumPy
- PyTorch (for DQN implementation)

## Usage

1. Clone this repository:

```bash
git clone https://github.com/ksnambiar/FrozenLake_QLearning.git
cd FrozenLake_QLearning
pip install -r requirements.txt
```
2. Run either ql_agent.py or deepQonFrozenLake.py
```bash
python ql_agent.py
```
or
```bash
python deepQonFrozenLake.py
```
