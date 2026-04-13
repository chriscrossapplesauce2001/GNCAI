# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-Agent Reinforcement Learning (MARL) system for UAV drone V-formation control and collision avoidance. Uses MAPPO (Multi-Agent PPO) with Centralized Training, Decentralized Execution (CTDE) — a shared centralized Critic sees all agents during training, while each agent runs a shared Actor using only local observations at execution time.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Common Commands

```bash
# Train (basic)
python train.py --num-agents 3 --episodes 500 --render-every 100

# Curriculum training (3 stages: 3 agents → 5 agents → 5 agents + obstacles)
python curriculum_train.py --stage-episodes 500

# Evaluate a checkpoint
python evaluate.py --checkpoint checkpoints/final.pt --num-agents 3

# Interactive explorer (watch agents, place obstacles, take manual control)
python interactive.py --checkpoint checkpoints/final.pt --num-agents 3

# Monitor training metrics
tensorboard --logdir runs/

# Run environment API compliance test + sanity checks
python test_env.py

# Run algorithm walkthrough (network architecture, forward pass, GAE, PPO update)
python test_algo.py
```

## Architecture

**Two packages:**
- `env/` — PettingZoo `ParallelEnv` environment and config
- `algo/` — MAPPO implementation (Actor, Critic, RolloutBuffer, MAPPO coordinator)

**Key design decisions:**
- All hyperparameters live in `env/formation_config.py` — this is the single source of truth for world params, reward weights, PPO hyperparams, and observation layout.
- Observation vector is fixed at 20 dimensions (padded with zeros) so weights transfer between curriculum stages when agent count changes.
- Only 3-agent and 5-agent formations are defined. Adding a new agent count requires defining offsets in `formation_config.py:get_formation_offsets()`.
- When loading checkpoints across different agent counts (curriculum), only Actor weights transfer — the Critic is re-initialized because its input size changes (it concatenates all agents' observations).
- The Actor outputs a Gaussian distribution (learned mean + log std) with Tanh squashing to [-1, 1].

**Reward function** (5 weighted components per agent, weights in `formation_config.py`):
- Formation distance keeping, target-directed movement, collision penalty, obstacle penalty, action smoothness, plus a soft proximity penalty.

**Data flow:** `MAPPO.collect_episode()` runs one full episode storing transitions in `RolloutBuffer`, then `MAPPO.update()` runs PPO clipped objective for 10 epochs over mini-batches with GAE advantages.
