# MARL Formation Control & Collision Avoidance

A Multi-Agent Reinforcement Learning system where UAV (drone) agents learn to fly in V-formation, avoid collisions, and navigate around obstacles — built from scratch with PyTorch and PettingZoo.

## Architecture

**CTDE — Centralized Training, Decentralized Execution** using **MAPPO** (Multi-Agent PPO):

- **Training:** A centralized Critic sees all agents' observations to estimate state value
- **Execution:** Each agent runs its own Actor using only local observations (no communication needed)
- **Parameter Sharing:** All agents share the same neural network weights (they're identical drones)

```
Training:   Actor(obs_i) → action_i    +    Critic([obs_0, obs_1, ..., obs_N]) → V(s)
Execution:  Actor(obs_i) → action_i    (critic not needed)
```

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Verify everything works:
```bash
python -c "import torch; import pettingzoo; import pygame; print('Ready!')"
```

---

## Learning Path (run in order)

### Phase 1 — Physics Sandbox
```bash
python sandbox_physics.py
```
Interactive Pygame demo with no RL. Drive Agent 0 with arrow keys and see how Euler integration, velocity clamping, and collision detection work. This builds intuition for the physics model before any learning complexity.

### Phase 2 — Environment Validation
```bash
python test_env.py
```
Creates the PettingZoo environment, runs the API compliance test, executes 10 steps with random actions showing the full observation vector and reward breakdown per agent, then runs a full 500-step episode and plots agent trajectories.

### Phase 3 — Algorithm Walkthrough
```bash
python test_algo.py
```
Creates the Actor and Critic networks and prints their architectures, traces a forward pass through both, collects one episode, shows the GAE (Generalized Advantage Estimation) computation step by step, then runs one PPO update showing how values and policy change.

### Phase 4 — Training
```bash
python train.py --num-agents 3 --episodes 500 --render-every 100
```
Trains 3 agents to fly in V-formation. Prints a summary table every 50 episodes and opens a Pygame window every 100 episodes so you can watch the agents improve. Monitor detailed metrics with Tensorboard:
```bash
tensorboard --logdir runs/
```

### Phase 5 — Curriculum Training
```bash
python curriculum_train.py --stage-episodes 500
```
Trains in 3 progressive stages: 3 agents → 5 agents → 5 agents with obstacles. Each stage warm-starts from the previous one.

### Phase 6 — Evaluation & Interactive Explorer
```bash
python evaluate.py --checkpoint checkpoints/final.pt --num-agents 3
python interactive.py --checkpoint checkpoints/final.pt --num-agents 3
```
Evaluate generates statistics and plots. The interactive explorer lets you watch agents fly, place obstacles by clicking, take manual control of individual agents, and toggle between the trained and random policy.

---

## File-by-File Explanation

### `env/formation_config.py`
Single source of truth for every hyperparameter in the project. Contains world size, physics timestep, agent count, V-formation offsets, reward weights, PPO hyperparameters, and the observation vector layout. Every value has a comment explaining what it does and why it was chosen. Includes `print_config()` to dump all settings to the console.

### `env/formation_env.py`
The custom PettingZoo `ParallelEnv` — the core simulation. Implements a 100x100 2D world where agents are point masses with Euler integration physics. Each `step()` is broken into labeled sub-steps: apply actions, physics update, move obstacles, collision detection, compute observations, compute rewards, check termination. Has a `verbose` flag that prints detailed per-step information for learning. Key design choices: observations are normalized to [-1, 1], no episode termination on collision (just a penalty), and the observation vector is fixed at 20 values (padded with zeros for future obstacle slots so weights transfer during curriculum learning).

### `env/__init__.py`
Empty init file making `env/` a Python package.

### `algo/actor.py`
The policy neural network. Maps a 20-dimensional observation vector to a 2D action (acceleration in x and y) through two hidden layers of 128 neurons each with ReLU activations and a Tanh output layer that squashes actions to [-1, 1]. Outputs a Gaussian distribution (learnable mean + log standard deviation) so the agent can explore during training. Includes `explain()` which prints the full architecture and parameter count, `get_action()` for sampling during rollouts, and `evaluate_action()` for computing log probabilities during PPO updates.

### `algo/critic.py`
The value neural network. Takes the global state (concatenation of all agents' observations, so 60 values for 3 agents or 100 for 5) and outputs a single scalar V(s) estimating how good the current state is. Same hidden layer structure as the Actor. This is the "centralized" part of CTDE — it only runs during training, not during execution. Includes `explain()` and `get_value()`.

### `algo/buffer.py`
The rollout buffer that stores one episode's worth of transitions for all agents: observations, global states, actions, log probabilities, rewards, values, and done flags. After an episode, `compute_returns_and_advantages()` runs the GAE (Generalized Advantage Estimation) algorithm backwards through the episode to compute how much better or worse each timestep was compared to the critic's prediction. Has a `verbose` mode that prints the GAE computation step by step. `get_batches()` yields shuffled mini-batches for the PPO update, flattening all (timestep, agent) pairs into individual training samples.

### `algo/mappo.py`
The MAPPO coordinator that ties everything together. Holds the shared Actor, centralized Critic, and their Adam optimizers. `collect_episode()` runs one full episode: at each step, every agent queries the shared Actor for an action, the environment steps, and transitions are stored in the buffer. `update()` runs the PPO clipped objective for 10 epochs over mini-batches — it computes the probability ratio between the new and old policy, clips it to prevent destructive updates, adds an entropy bonus for exploration, and updates the Critic with MSE loss against computed returns. Includes `save()` and `load()` for checkpointing, with support for loading only actor weights when agent count changes during curriculum learning.

### `algo/__init__.py`
Empty init file making `algo/` a Python package.

### `sandbox_physics.py`
A standalone Pygame application (no RL) that lets you interact with the physics engine directly. Creates 3 agents in a 2D world — Agent 0 (gold) is controlled by arrow keys, others move randomly. Shows velocity arrows, collision detection (agents flash red), formation target positions, and prints physics state to the terminal every 10 steps. Controls: arrow keys to steer, SPACE to pause, R to reset, +/- to change simulation speed, Q to quit.

### `test_env.py`
Phase 2 checkpoint script. Runs three tests: (1) PettingZoo's `parallel_api_test` to verify the environment is API-compliant, (2) 10 steps with random actions printing the full observation vector with labeled indices and a per-agent reward breakdown table, (3) a full 500-step episode collecting metrics. Generates a matplotlib trajectory plot saved as `trajectories.png`.

### `test_algo.py`
Phase 3 checkpoint script. Runs four tests: (1) creates Actor and Critic and calls `explain()` on each, (2) traces a forward pass through both networks showing tensor shapes and output values, (3) creates the full MAPPO system, collects one episode showing the buffer summary and GAE computation, (4) runs one PPO update and prints before/after value estimates and action means to demonstrate that learning is happening.

### `train.py`
The main training script. Accepts command-line arguments for number of agents, episodes, render frequency, verbosity, checkpoint directory, and resume path. Runs a training loop that collects one episode then performs a PPO update, printing a summary table every 50 episodes with mean reward, formation error, collisions, distance traveled, and loss values. Saves checkpoints every 200 episodes and tracks the best model. Optionally logs all metrics to Tensorboard and renders episodes with Pygame at a specified interval.

### `visualize.py`
Pygame-based renderer used by `train.py` and `interactive.py`. Draws agents as colored circles (leader in gold, followers in blue), formation target positions as gray dashed circles, connecting lines color-coded by distance (green = close, orange = medium, red = far), agent trails as fading dots, velocity arrows, obstacles as red circles, and a HUD overlay with step count, reward, formation error, and collision count. Supports pause, speed control, and keyboard events.

### `curriculum_train.py`
Automates three-stage curriculum training by calling `train.py`'s `train()` function three times with increasing difficulty. Stage 1: 3 agents, no obstacles. Stage 2: 5 agents, no obstacles (loads Stage 1 actor weights; critic is re-initialized because its input size changes with agent count). Stage 3: 5 agents, 2 dynamic obstacles (loads Stage 2 weights). Saves separate checkpoints per stage and logs to separate Tensorboard directories.

### `evaluate.py`
Runs N episodes (default 100) with the trained policy in deterministic mode (using the action mean, no sampling noise). Computes mean and standard deviation of reward, formation error, collision rate, and distance traveled, plus per-agent reward breakdowns. Generates a 4-panel matplotlib figure: reward distribution histogram, formation error over episodes, collision count over episodes, and a sample trajectory plot. Saves everything to `results/`.

### `interactive.py`
The capstone interactive Pygame application. Loads a trained checkpoint and lets you experiment freely: watch agents fly in V-formation, left-click to place obstacles and see agents react, press 1-5 to take manual control of an agent (arrow keys to steer) while others use the trained policy, press T to toggle between trained and random policy for a dramatic before/after comparison, press 0 to release control. Auto-resets when episodes end.

### `requirements.txt`
Python dependencies: PyTorch (neural networks), PettingZoo (multi-agent environment API), Gymnasium (observation/action spaces), NumPy (math), Matplotlib (plotting), Pygame (visualization), Tensorboard (training logs).

---

## Key Concepts

| Concept | What it means | Where in the code |
|---------|--------------|-------------------|
| **Euler Integration** | `vel += accel * dt`, `pos += vel * dt` | `env/formation_env.py` step() |
| **Observation Normalization** | All values scaled to [-1, 1] | `env/formation_env.py` _get_obs() |
| **V-Formation Offsets** | Relative positions from centroid | `env/formation_config.py` |
| **Reward Shaping** | 5 weighted components per agent | `env/formation_env.py` _compute_reward() |
| **Parameter Sharing** | One Actor for all agents | `algo/mappo.py` |
| **CTDE** | Global Critic + Local Actors | `algo/mappo.py` collect_episode() |
| **GAE** | Advantage estimation with bias-variance tradeoff | `algo/buffer.py` compute_returns_and_advantages() |
| **PPO Clipping** | Limits policy update magnitude | `algo/mappo.py` update() |
| **Curriculum Learning** | Gradual difficulty increase | `curriculum_train.py` |

## Reward Function

Each agent receives per step:
```
reward = 1.0 * formation_reward      # exp(-error/5): close to target = good
       + 0.5 * direction_reward      # vx / max_speed: moving right = good
       - 5.0 * collision_penalty     # hit another agent = bad
       - 5.0 * obstacle_penalty      # hit an obstacle = bad
       + 0.1 * smoothness_reward     # smooth actions = good
```
