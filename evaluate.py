"""
Evaluate — Test Trained Policy and Generate Metrics

Runs 100 episodes with the trained policy (no exploration noise),
computes statistics, and generates visualization plots.

Usage:
    python evaluate.py --checkpoint checkpoints/final.pt
    python evaluate.py --checkpoint checkpoints/final.pt --num-agents 5 --num-obstacles 2
    python evaluate.py --checkpoint checkpoints/final.pt --episodes 50
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.formation_env import UAVFormationEnv
from env.formation_config import OBS_DIM, print_config
from algo.mappo import MAPPO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Trained UAV Swarm Policy")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--num-obstacles", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()


def evaluate(args):
    print_config()
    print("\n" + "=" * 60)
    print("  EVALUATION")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Agents:     {args.num_agents}")
    print(f"  Obstacles:  {args.num_obstacles}")
    print(f"  Episodes:   {args.episodes}")
    print("=" * 60)

    env = UAVFormationEnv(num_agents=args.num_agents,
                          num_obstacles=args.num_obstacles, verbose=False)
    mappo = MAPPO(num_agents=args.num_agents)
    mappo.load(args.checkpoint)

    # Set actor to eval mode (disables dropout, batch norm, etc.)
    mappo.actor.eval()

    all_rewards = []
    all_formation_errors = []
    all_collisions = []
    all_distances = []
    per_agent_rewards = {f"uav_{i}": [] for i in range(args.num_agents)}
    sample_trajectories = None  # Save one episode's trajectories for plotting

    for ep in range(args.episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_collisions = 0
        ep_errors = []
        trajectories = {f"uav_{i}": [] for i in range(args.num_agents)}
        agent_rewards = {f"uav_{i}": 0.0 for i in range(args.num_agents)}

        while env.agents:
            actions = {}
            for agent in env.agents:
                # Use mean action (no sampling noise) for deterministic evaluation
                import torch
                obs_t = torch.FloatTensor(obs[agent]).unsqueeze(0)
                action_mean = mappo.actor(obs_t).squeeze(0).detach().numpy()
                actions[agent] = np.clip(action_mean, -1, 1)

            obs, rewards, terms, truncs, infos = env.step(actions)

            for i, agent in enumerate(env.possible_agents):
                if agent in rewards:
                    ep_reward += rewards[agent]
                    agent_rewards[agent] += rewards[agent]
                    trajectories[agent].append(env.positions[i].copy())
                if agent in infos:
                    ep_errors.append(infos[agent].get("formation_error", 0))
                    ep_collisions += infos[agent].get("num_agent_collisions", 0)

        mean_reward = ep_reward / args.num_agents
        all_rewards.append(mean_reward)
        all_formation_errors.append(np.mean(ep_errors) if ep_errors else 0)
        all_collisions.append(ep_collisions // 2)
        all_distances.append(env.positions[:, 0].mean() if env.positions is not None else 0)

        for agent in per_agent_rewards:
            per_agent_rewards[agent].append(agent_rewards[agent])

        if sample_trajectories is None:
            sample_trajectories = trajectories

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep + 1}/{args.episodes}: "
                  f"reward={mean_reward:+.1f}, "
                  f"formation_err={all_formation_errors[-1]:.2f}, "
                  f"collisions={all_collisions[-1]}")

    # Print results
    print("\n" + "=" * 60)
    print("  RESULTS (over {} episodes)".format(args.episodes))
    print("=" * 60)
    print(f"\n  Mean Reward:         {np.mean(all_rewards):+.2f} ± {np.std(all_rewards):.2f}")
    print(f"  Mean Formation Error:{np.mean(all_formation_errors):8.2f} ± {np.std(all_formation_errors):.2f}")
    print(f"  Mean Collisions:     {np.mean(all_collisions):8.1f} ± {np.std(all_collisions):.1f}")
    print(f"  Mean Distance:       {np.mean(all_distances):8.1f} ± {np.std(all_distances):.1f}")

    print("\n  Per-agent mean rewards:")
    for agent, rewards in per_agent_rewards.items():
        print(f"    {agent}: {np.mean(rewards):+.2f} ± {np.std(rewards):.2f}")

    # Generate plots
    os.makedirs(args.output_dir, exist_ok=True)
    _plot_results(all_rewards, all_formation_errors, all_collisions,
                  all_distances, sample_trajectories, args)

    print(f"\n  Plots saved to {args.output_dir}/")
    print("=" * 60)


def _plot_results(rewards, errors, collisions, distances, trajectories, args):
    """Generate evaluation plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Reward distribution
    axes[0, 0].hist(rewards, bins=20, color="steelblue", alpha=0.7)
    axes[0, 0].axvline(np.mean(rewards), color="red", linestyle="--", label=f"Mean: {np.mean(rewards):.1f}")
    axes[0, 0].set_title("Reward Distribution")
    axes[0, 0].set_xlabel("Mean Episode Reward")
    axes[0, 0].legend()

    # Formation error over episodes
    axes[0, 1].plot(errors, color="coral", alpha=0.7)
    axes[0, 1].axhline(np.mean(errors), color="red", linestyle="--", label=f"Mean: {np.mean(errors):.2f}")
    axes[0, 1].set_title("Formation Error per Episode")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Mean Formation Error")
    axes[0, 1].legend()

    # Collisions over episodes
    axes[1, 0].plot(collisions, color="mediumpurple", alpha=0.7)
    axes[1, 0].axhline(np.mean(collisions), color="red", linestyle="--", label=f"Mean: {np.mean(collisions):.1f}")
    axes[1, 0].set_title("Collisions per Episode")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Collision Count")
    axes[1, 0].legend()

    # Sample trajectory
    colors = ["gold", "steelblue", "coral", "mediumseagreen", "mediumpurple"]
    if trajectories:
        for i, (agent, traj) in enumerate(trajectories.items()):
            traj = np.array(traj)
            if len(traj) > 0:
                axes[1, 1].plot(traj[:, 0], traj[:, 1], color=colors[i % len(colors)],
                                alpha=0.6, label=agent)
                axes[1, 1].scatter(traj[-1, 0], traj[-1, 1], color=colors[i % len(colors)],
                                   marker="*", s=100, zorder=5)
    axes[1, 1].set_xlim(0, 100)
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].set_aspect("equal")
    axes[1, 1].set_title("Sample Trajectory (★ = final position)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"Evaluation Results — {args.num_agents} agents, {args.num_obstacles} obstacles",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "evaluation.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
