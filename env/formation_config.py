"""
Formation Config — Single Source of Truth for All Hyperparameters

Every tunable value in the project lives here.
Change something here and it propagates everywhere.

Selected params can be overridden at import time via FORMATION_<NAME> env vars
(used by sweep_optuna.py to launch trials with different configs).
"""

import os as _os
import numpy as np


def _override(name, default, cast=float):
    v = _os.getenv(f"FORMATION_{name}")
    return cast(v) if v is not None else default

# =============================================================================
# World Parameters
# =============================================================================

WORLD_SIZE = 100.0       # 100x100 2D world
DT = 0.1                 # Physics timestep (seconds) — Euler integration step size
MAX_SPEED = 5.0           # Maximum agent velocity magnitude (units/s)
AGENT_RADIUS = 1.5        # Agent collision radius (units)
OBSTACLE_RADIUS = 3.0     # Obstacle collision radius (units)
NUM_AGENTS = 3            # Default number of agents
MAX_STEPS = 400           # Steps per episode before truncation
TARGET_THRESHOLD = 12.0   # Each agent must be within this distance to "arrive" at target
DESIRED_SPACING = 8.0     # Target inter-agent distance (units)
COMPLETION_BONUS = _override("COMPLETION_BONUS", 500.0)  # Per-agent bonus reward for reaching the target

# =============================================================================
# Observation & Action Dimensions
# =============================================================================

OBS_DIM = 20              # Fixed observation vector size (padded for curriculum transfer)
ACTION_DIM = 2            # 2D acceleration (ax, ay)
HIDDEN_DIM = 128          # Hidden layer size for Actor and Critic networks

# Observation vector layout (20 values, all normalized to [-1, 1]):
#  [0]  pos_x        — normalized x position
#  [1]  pos_y        — normalized y position
#  [2]  vel_x        — normalized x velocity
#  [3]  vel_y        — normalized y velocity
#  [4]  n1_dx        — delta x to nearest neighbor 1
#  [5]  n1_dy        — delta y to nearest neighbor 1
#  [6]  n2_dx        — delta x to nearest neighbor 2
#  [7]  n2_dy        — delta y to nearest neighbor 2
#  [8]  form_dx      — delta x to formation target
#  [9]  form_dy      — delta y to formation target
# [10]  tgt_dx       — normalized delta x to navigation target
# [11]  tgt_dy       — normalized delta y to navigation target
# [12]  obs1_dx      — delta x to obstacle 1
# [13]  obs1_dy      — delta y to obstacle 1
# [14]  obs1_vx      — velocity x of obstacle 1
# [15]  obs1_vy      — velocity y of obstacle 1
# [16]  obs2_dx      — delta x to obstacle 2
# [17]  obs2_dy      — delta y to obstacle 2
# [18]  obs2_vx      — velocity x of obstacle 2
# [19]  obs2_vy      — velocity y of obstacle 2

# =============================================================================
# Reward Weights
# =============================================================================

FORMATION_WEIGHT = _override("FORMATION_WEIGHT", 0.3)     # spacing penalty weight (capped)
TIME_PENALTY = 0.0         # disabled for now
APPROACH_WEIGHT = _override("APPROACH_WEIGHT", 2.0)       # approach reward scale (+2/step max)
COLLISION_PENALTY = _override("COLLISION_PENALTY", 2.0)   # proximity penalty scale
SAFE_DIST = _override("SAFE_DIST", 5.0)                   # ramp distance for proximity penalty
OBSTACLE_PENALTY = 0.0     # disabled for now
SMOOTHNESS_WEIGHT = 0.0    # disabled for now

# =============================================================================
# PPO Hyperparameters
# =============================================================================

LEARNING_RATE = 1e-4       # Lower for fine-tuning from navigation checkpoint
GAMMA = 0.99               # Discount factor (how much to care about future rewards)
GAE_LAMBDA = 0.95          # GAE lambda (bias-variance tradeoff for advantage estimation)
CLIP_EPSILON = 0.2         # PPO clipping range — limits policy update magnitude
PPO_EPOCHS = _override("PPO_EPOCHS", 10, int)            # Number of optimization passes over each episode's data
BATCH_SIZE = 0             # 0 = use full buffer as one batch (MAPPO SOTA)
EPISODES_PER_UPDATE = 10   # Collect N episodes before each PPO update
ENTROPY_COEF = _override("ENTROPY_COEF", 0.01)        # Entropy bonus coefficient — encourages exploration
VALUE_LOSS_COEF = 1.0      # Critic loss scaling factor

# =============================================================================
# V-Formation Offsets
# =============================================================================

def get_formation_offsets(num_agents):
    """
    Get V-formation target offsets relative to the swarm centroid.

    Returns array of shape (num_agents, 2) with [dx, dy] offsets.
    Agent 0 is the leader (front of the V), others trail behind.
    """
    if num_agents == 3:
        return np.array([
            [5.0, 0.0],     # Agent 0: leader, front center
            [-5.0, -5.0],   # Agent 1: left wing
            [-5.0, 5.0],    # Agent 2: right wing
        ])
    elif num_agents == 5:
        return np.array([
            [8.0, 0.0],     # Agent 0: leader
            [-2.0, -5.0],   # Agent 1: inner left
            [-2.0, 5.0],    # Agent 2: inner right
            [-8.0, -10.0],  # Agent 3: outer left
            [-8.0, 10.0],   # Agent 4: outer right
        ])
    else:
        # Fallback: arrange in a V pattern
        offsets = np.zeros((num_agents, 2))
        offsets[0] = [5.0, 0.0]
        for i in range(1, num_agents):
            side = 1 if i % 2 == 0 else -1
            rank = (i + 1) // 2
            offsets[i] = [-rank * 5.0, side * rank * 5.0]
        return offsets


# =============================================================================
# Print Configuration
# =============================================================================

def print_config():
    """Print all configuration values to the console."""
    print("\n" + "=" * 60)
    print("  CONFIGURATION")
    print("=" * 60)
    print(f"  World size:       {WORLD_SIZE} x {WORLD_SIZE}")
    print(f"  Physics dt:       {DT}")
    print(f"  Max speed:        {MAX_SPEED}")
    print(f"  Agent radius:     {AGENT_RADIUS}")
    print(f"  Obstacle radius:  {OBSTACLE_RADIUS}")
    print(f"  Max steps/ep:     {MAX_STEPS}")
    print(f"  Target threshold: {TARGET_THRESHOLD}")
    print(f"  Completion bonus: {COMPLETION_BONUS}")
    print(f"  Observation dim:  {OBS_DIM}")
    print(f"  Action dim:       {ACTION_DIM}")
    print(f"  Hidden dim:       {HIDDEN_DIM}")
    print(f"  ---")
    print(f"  Learning rate:    {LEARNING_RATE}")
    print(f"  Gamma:            {GAMMA}")
    print(f"  GAE lambda:       {GAE_LAMBDA}")
    print(f"  Clip epsilon:     {CLIP_EPSILON}")
    print(f"  PPO epochs:       {PPO_EPOCHS}")
    print(f"  Eps per update:   {EPISODES_PER_UPDATE}")
    print(f"  Batch size:       {BATCH_SIZE}")
    print(f"  Entropy coef:     {ENTROPY_COEF}")
    print(f"  Value loss coef:  {VALUE_LOSS_COEF}")
    print(f"  ---")
    print(f"  Reward weights:")
    print(f"    Formation:      {FORMATION_WEIGHT}")
    print(f"    Time penalty:   -{TIME_PENALTY}")
    print(f"    Approach:       {APPROACH_WEIGHT}")
    print(f"    Proximity:      -{COLLISION_PENALTY} (ramp over {SAFE_DIST} units)")
    print(f"    Obstacle:       -{OBSTACLE_PENALTY}")
    print(f"    Smoothness:     {SMOOTHNESS_WEIGHT}")
    print("=" * 60)
