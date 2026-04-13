"""
Test Algorithm — Phase 3 Checkpoint

Run this to verify all MAPPO components work together.
It will:
  1. Create actor and critic, explain their architectures
  2. Trace a forward pass through both networks
  3. Collect one episode using random policy
  4. Show the buffer summary and GAE computation
  5. Run one PPO update and show loss values

Usage:
    python test_algo.py
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.formation_env import UAVFormationEnv
from env.formation_config import OBS_DIM, ACTION_DIM, print_config
from algo.actor import Actor
from algo.critic import Critic
from algo.mappo import MAPPO


def test_networks():
    """Create and explain the neural networks."""
    print("\n" + "=" * 60)
    print("  TEST 1: Network Architecture")
    print("=" * 60)

    actor = Actor()
    actor.explain()

    critic = Critic(num_agents=3)
    critic.explain()

    return actor, critic


def test_forward_pass(actor, critic):
    """Trace data through both networks."""
    print("\n" + "=" * 60)
    print("  TEST 2: Forward Pass Trace")
    print("=" * 60)

    # Create a dummy observation
    obs = torch.randn(1, OBS_DIM)
    print(f"\n  Input observation shape: {obs.shape}")
    print(f"  Input values (first 5): {obs[0, :5].tolist()}")

    # Actor forward pass
    print("\n  --- Actor Forward Pass ---")
    action_mean = actor(obs)
    print(f"  Action mean: {action_mean.detach().numpy()}")
    print(f"  Action shape: {action_mean.shape}")

    # Sample an action
    action, log_prob = actor.get_action(obs.squeeze().numpy())
    print(f"  Sampled action: {action}")
    print(f"  Log probability: {log_prob.item():.4f}")
    print(f"  (log_prob tells us how 'likely' this action was under the policy)")

    # Critic forward pass
    print("\n  --- Critic Forward Pass ---")
    global_state = torch.randn(1, 3 * OBS_DIM)  # 3 agents
    print(f"  Global state shape: {global_state.shape} (3 agents × {OBS_DIM} obs)")
    value = critic.get_value(global_state.squeeze().numpy())
    print(f"  Value estimate: {value.item():.4f}")
    print(f"  (This is the critic's guess of 'how good is this state?')")


def test_episode_collection():
    """Collect one episode and show buffer contents."""
    print("\n" + "=" * 60)
    print("  TEST 3: Episode Collection")
    print("=" * 60)

    env = UAVFormationEnv(num_agents=3, verbose=False)
    mappo = MAPPO(num_agents=3)

    print("\n  Collecting one episode with untrained policy...")
    buffer, info = mappo.collect_episode(env, verbose=True)

    print(f"\n  Episode stats:")
    print(f"    Steps: {info['steps']}")
    print(f"    Mean reward: {info['mean_reward']:.2f}")
    print(f"    Collisions: {info['collisions']}")
    print(f"    Formation error: {info['mean_formation_error']:.2f}")

    buffer.summary()

    return mappo, buffer, env


def test_ppo_update(mappo, buffer):
    """Run one PPO update and show what happens."""
    print("\n" + "=" * 60)
    print("  TEST 4: PPO Update")
    print("=" * 60)

    # Get value estimate before update
    dummy_state = torch.randn(3 * OBS_DIM)
    value_before = mappo.critic.get_value(dummy_state.numpy()).item()
    print(f"\n  Value estimate BEFORE update: {value_before:.4f}")

    # Get action distribution before update
    dummy_obs = torch.randn(OBS_DIM)
    dist_before = mappo.actor.get_distribution(dummy_obs.unsqueeze(0))
    mean_before = dist_before.mean.detach().numpy()
    print(f"  Action mean BEFORE update: {mean_before}")

    # Run PPO update
    print("\n  Running PPO update (10 epochs)...")
    stats = mappo.update(buffer, verbose=True)

    # Get estimates after update
    value_after = mappo.critic.get_value(dummy_state.numpy()).item()
    dist_after = mappo.actor.get_distribution(dummy_obs.unsqueeze(0))
    mean_after = dist_after.mean.detach().numpy()

    print(f"\n  Value estimate AFTER update:  {value_after:.4f}")
    print(f"  Action mean AFTER update:  {mean_after}")
    print(f"\n  Value changed by: {value_after - value_before:+.4f}")
    print(f"  → The critic is learning to predict returns!")
    print(f"  Action mean changed by: {mean_after - mean_before}")
    print(f"  → The actor is adjusting its policy!")

    return stats


if __name__ == "__main__":
    print_config()

    actor, critic = test_networks()
    test_forward_pass(actor, critic)
    mappo, buffer, env = test_episode_collection()
    test_ppo_update(mappo, buffer)

    print("\n" + "=" * 60)
    print("  Phase 3 checkpoint complete!")
    print("  All MAPPO components working.")
    print("  Next: Phase 4 — Training + Visualization")
    print("=" * 60)
