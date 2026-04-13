"""
Test Environment — Phase 2 Checkpoint

Run this to verify the PettingZoo environment works correctly.
It will:
  1. Create the environment and print its configuration
  2. Run PettingZoo's API compliance test
  3. Run 10 steps with random actions, printing reward breakdowns
  4. Plot agent trajectories and save as PNG

Usage:
    python test_env.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.formation_env import UAVFormationEnv
from env.formation_config import print_config, MAX_STEPS


def test_api_compliance():
    """Run PettingZoo's built-in API test to make sure our env is valid."""
    print("\n" + "=" * 60)
    print("  TEST 1: PettingZoo API Compliance")
    print("=" * 60)
    try:
        from pettingzoo.test import parallel_api_test
        env = UAVFormationEnv(num_agents=3, verbose=False)
        parallel_api_test(env, num_cycles=50)
        print("  ✓ API test PASSED — environment is PettingZoo compliant!")
    except Exception as e:
        print(f"  ✗ API test FAILED: {e}")
        return False
    return True


def test_random_steps():
    """Run 10 steps with random actions and show what's happening."""
    print("\n" + "=" * 60)
    print("  TEST 2: Random Actions — 10 Steps")
    print("=" * 60)

    env = UAVFormationEnv(num_agents=3, verbose=True)
    obs, infos = env.reset(seed=42)

    print(f"\n  Observation space: {env.observation_space('uav_0')}")
    print(f"  Action space:     {env.action_space('uav_0')}")
    print(f"  Agents:           {env.agents}")

    print(f"\n  Sample observation (Agent 0):")
    o = obs["uav_0"]
    labels = ["pos_x", "pos_y", "vel_x", "vel_y",
              "n1_dx", "n1_dy", "n2_dx", "n2_dy",
              "form_dx", "form_dy", "dir_vx", "dir_dist",
              "obs1_dx", "obs1_dy", "obs1_vx", "obs1_vy",
              "obs2_dx", "obs2_dy", "obs2_vx", "obs2_vy"]
    for i, (val, label) in enumerate(zip(o, labels)):
        print(f"    [{i:2d}] {label:10s} = {val:+.4f}")

    # Run 10 steps
    total_rewards = {agent: 0.0 for agent in env.agents}
    trajectories = {agent: [env.positions[i].copy()] for i, agent in enumerate(env.agents)}

    for step in range(10):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)

        for i, agent in enumerate(env.possible_agents):
            if agent in rewards:
                total_rewards[agent] += rewards[agent]
            if agent in obs:
                trajectories[agent].append(env.positions[i].copy())

    print(f"\n  After 10 steps — total rewards:")
    for agent, r in total_rewards.items():
        print(f"    {agent}: {r:+.3f}")

    return trajectories


def test_full_episode():
    """Run a full episode and collect metrics."""
    print("\n" + "=" * 60)
    print("  TEST 3: Full Episode (500 steps)")
    print("=" * 60)

    env = UAVFormationEnv(num_agents=3, verbose=False)
    obs, _ = env.reset(seed=123)

    total_rewards = {agent: 0.0 for agent in env.agents}
    collisions = 0
    formation_errors = []
    trajectories = {agent: [] for agent in env.agents}

    step = 0
    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)

        for i, agent in enumerate(env.possible_agents):
            if agent in rewards:
                total_rewards[agent] += rewards[agent]
                trajectories[agent].append(env.positions[i].copy())
            if agent in infos:
                formation_errors.append(infos[agent].get("formation_error", 0))
                collisions += infos[agent].get("num_agent_collisions", 0)
        step += 1

    print(f"\n  Episode finished after {step} steps")
    print(f"  Total rewards:")
    for agent, r in total_rewards.items():
        print(f"    {agent}: {r:+.1f}")
    print(f"  Total collisions detected: {collisions}")
    if formation_errors:
        print(f"  Mean formation error: {np.mean(formation_errors):.2f}")
        print(f"  Final formation error: {np.mean(formation_errors[-3:]):.2f}")

    return trajectories


def plot_trajectories(trajectories, filename="trajectories.png"):
    """Plot agent trajectories and save as PNG."""
    print("\n" + "=" * 60)
    print("  Plotting trajectories...")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed, skipping plot.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    colors = ["gold", "steelblue", "coral", "mediumseagreen", "mediumpurple"]
    for i, (agent, traj) in enumerate(trajectories.items()):
        traj = np.array(traj)
        if len(traj) == 0:
            continue
        color = colors[i % len(colors)]
        ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.6, linewidth=1,
                label=agent)
        ax.scatter(traj[0, 0], traj[0, 1], color=color, marker="o", s=80,
                   edgecolors="black", zorder=5)  # Start
        ax.scatter(traj[-1, 0], traj[-1, 1], color=color, marker="*", s=120,
                   edgecolors="black", zorder=5)  # End

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_title("Agent Trajectories (Random Policy)\n○ = start, ★ = end")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    plt.savefig(filepath, dpi=150)
    print(f"  Saved trajectory plot to: {filepath}")
    plt.close()


if __name__ == "__main__":
    print_config()

    test_api_compliance()
    test_random_steps()
    trajectories = test_full_episode()
    plot_trajectories(trajectories)

    print("\n" + "=" * 60)
    print("  Phase 2 checkpoint complete!")
    print("  Next: Phase 3 — MAPPO algorithm")
    print("=" * 60)
