"""
Training Script — Train UAV Agents with MAPPO

This script trains the swarm to fly in V-formation using MAPPO.
It provides rich console output so you can watch the learning progress,
and optionally renders episodes with Pygame.

Usage:
    # Basic training (3 agents, 500 episodes)
    python train.py --num-agents 3 --episodes 500

    # With visualization every 100 episodes
    python train.py --num-agents 3 --episodes 500 --render-every 100

    # Verbose mode (print per-step details for first few episodes)
    python train.py --num-agents 3 --episodes 100 --verbose

    # Resume from checkpoint
    python train.py --resume checkpoints/latest.pt --episodes 1000

    # Monitor with Tensorboard (in separate terminal):
    tensorboard --logdir runs/
"""

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.formation_env import UAVFormationEnv
from env.formation_config import print_config, MAX_STEPS, NUM_AGENTS
from algo.mappo import MAPPO

# Try to import optional dependencies
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False

try:
    from visualize import Renderer
    HAS_RENDERER = True
except ImportError:
    HAS_RENDERER = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train UAV Swarm Formation with MAPPO")
    parser.add_argument("--num-agents", type=int, default=3, help="Number of UAV agents (3 or 5)")
    parser.add_argument("--num-obstacles", type=int, default=0, help="Number of obstacles")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of training episodes")
    parser.add_argument("--render-every", type=int, default=0, help="Render every N episodes (0=off)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed per-step info")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Where to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint file")
    parser.add_argument("--log-dir", type=str, default="runs", help="Tensorboard log directory")
    return parser.parse_args()


def render_episode(mappo, env, renderer, episode_num):
    """Run one episode with rendering for visual inspection."""
    import pygame

    obs, _ = env.reset()
    renderer.reset_trails()

    total_reward = 0.0
    total_collisions = 0
    step = 0

    while env.agents:
        # Pump events every frame so the OS knows the window is alive
        if not renderer.handle_events():
            return False  # Window closed

        if renderer.paused:
            renderer.render(env, step, episode_num, total_reward, 0, total_collisions, "PAUSED")
            continue

        # Run sim_speed physics steps per rendered frame
        for _ in range(renderer.sim_speed):
            if not env.agents:
                break

            actions = {}
            for agent in env.agents:
                action, _ = mappo.actor.get_action(obs[agent])
                actions[agent] = action

            obs, rewards, terms, truncs, infos = env.step(actions)

            mean_reward = np.mean([r for r in rewards.values()])
            total_reward += mean_reward
            for info in infos.values():
                total_collisions += info.get("num_agent_collisions", 0)
            step += 1

        # Compute formation error for display
        errors = [env._get_formation_error(i) for i in range(env._num_agents)]
        mean_error = np.mean(errors) if errors else 0

        renderer.render(env, step, episode_num, total_reward, mean_error,
                        total_collisions // 2,
                        f"Trained policy | +/-=speed({renderer.sim_speed}x) SPACE=pause Q=close")

    # Show final frame for a moment
    for _ in range(60):
        pygame.event.pump()
        renderer.render(env, step, episode_num, total_reward,
                        0, total_collisions // 2, "Episode done — closing...")
    return True


def train(args):
    print_config()
    print("\n" + "=" * 60)
    print("  MAPPO TRAINING")
    print("=" * 60)
    print(f"  Agents:    {args.num_agents}")
    print(f"  Obstacles: {args.num_obstacles}")
    print(f"  Episodes:  {args.episodes}")
    print(f"  Render:    {'every ' + str(args.render_every) + ' episodes' if args.render_every else 'off'}")
    print(f"  Verbose:   {args.verbose}")
    print("=" * 60)

    # Create environment and MAPPO agent
    env = UAVFormationEnv(
        num_agents=args.num_agents,
        num_obstacles=args.num_obstacles,
        verbose=False,
    )
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device:    {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    mappo = MAPPO(num_agents=args.num_agents, device=device)

    # Resume from checkpoint if specified
    start_episode = 0
    if args.resume and os.path.exists(args.resume):
        mappo.load(args.resume)
        print(f"  Resumed from {args.resume}")

    # Setup Tensorboard logging
    writer = None
    if HAS_TB:
        writer = SummaryWriter(log_dir=args.log_dir)
        print(f"  Tensorboard: tensorboard --logdir {args.log_dir}")
    else:
        print("  Tensorboard not available (pip install tensorboard)")

    # Renderer is created on-demand per episode (no persistent window)
    use_renderer = args.render_every > 0 and HAS_RENDERER

    # Checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Training metrics (for printing summary table)
    recent_rewards = []
    recent_errors = []
    recent_collisions = []
    recent_distances = []
    best_reward = -float("inf")

    print("\n  Training started...\n")
    print(f"  {'Episode':>8} | {'Reward':>8} | {'FormErr':>8} | {'Collis':>6} | "
          f"{'Dist':>6} | {'PolLoss':>8} | {'ValLoss':>8} | {'Entropy':>8}")
    print("  " + "-" * 85)

    start_time = time.time()

    for episode in range(start_episode, args.episodes):
        # Collect one episode
        verbose_ep = args.verbose and episode < 3  # Verbose only for first few
        buffer, ep_info = mappo.collect_episode(env, verbose=verbose_ep)

        # PPO update
        update_stats = mappo.update(buffer, verbose=verbose_ep)

        # Track metrics
        recent_rewards.append(ep_info["mean_reward"])
        recent_errors.append(ep_info["mean_formation_error"])
        recent_collisions.append(ep_info["collisions"])
        recent_distances.append(ep_info["distance_traveled"])

        # Tensorboard logging
        if writer:
            writer.add_scalar("reward/mean", ep_info["mean_reward"], episode)
            writer.add_scalar("metrics/formation_error", ep_info["mean_formation_error"], episode)
            writer.add_scalar("metrics/collisions", ep_info["collisions"], episode)
            writer.add_scalar("metrics/distance", ep_info["distance_traveled"], episode)
            writer.add_scalar("loss/policy", update_stats["policy_loss"], episode)
            writer.add_scalar("loss/value", update_stats["value_loss"], episode)
            writer.add_scalar("loss/entropy", update_stats["entropy"], episode)
            writer.add_scalar("loss/clip_fraction", update_stats["clip_fraction"], episode)

        # Print summary every 50 episodes
        if (episode + 1) % 50 == 0 or episode == 0:
            avg_reward = np.mean(recent_rewards[-50:])
            avg_error = np.mean(recent_errors[-50:])
            avg_coll = np.mean(recent_collisions[-50:])
            avg_dist = np.mean(recent_distances[-50:])

            elapsed = time.time() - start_time
            eps_per_sec = (episode + 1) / elapsed

            print(f"  {episode + 1:>8} | {avg_reward:>+8.1f} | {avg_error:>8.2f} | "
                  f"{avg_coll:>6.1f} | {avg_dist:>6.1f} | "
                  f"{update_stats['policy_loss']:>8.4f} | "
                  f"{update_stats['value_loss']:>8.4f} | "
                  f"{update_stats['entropy']:>8.4f}")

            if (episode + 1) % 200 == 0:
                print(f"           [{eps_per_sec:.1f} episodes/sec, "
                      f"{elapsed:.0f}s elapsed]")

        # Save checkpoint every 200 episodes (and keep best)
        if (episode + 1) % 200 == 0:
            path = os.path.join(args.checkpoint_dir, f"checkpoint_{episode + 1}.pt")
            mappo.save(path)

            avg_reward = np.mean(recent_rewards[-200:])
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_path = os.path.join(args.checkpoint_dir, "best.pt")
                mappo.save(best_path)

        # Render episode if requested — window opens fresh each time, then closes
        if use_renderer and (episode + 1) % args.render_every == 0:
            print(f"\n  [RENDER] Visualizing episode {episode + 1}...")
            renderer = Renderer()
            render_env = UAVFormationEnv(
                num_agents=args.num_agents,
                num_obstacles=args.num_obstacles,
                verbose=False,
            )
            render_episode(mappo, render_env, renderer, episode + 1)
            renderer.close()
            print("  [RENDER] Window closed, continuing training...")

    # Final save
    final_path = os.path.join(args.checkpoint_dir, "final.pt")
    mappo.save(final_path)

    # Training summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Episodes:    {args.episodes}")
    print(f"  Time:        {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"  Final reward (avg last 50): {np.mean(recent_rewards[-50:]):+.2f}")
    print(f"  Final form error (avg 50):  {np.mean(recent_errors[-50:]):.2f}")
    print(f"  Final collisions (avg 50):  {np.mean(recent_collisions[-50:]):.1f}")
    print(f"  Best avg reward:            {best_reward:+.2f}")
    print(f"  Checkpoints in:             {args.checkpoint_dir}/")
    if HAS_TB:
        print(f"  View logs:  tensorboard --logdir {args.log_dir}")
    print("=" * 60)

    if writer:
        writer.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)
