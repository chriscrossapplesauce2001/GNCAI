"""
Curriculum Training — Progressive Difficulty

Trains the swarm in 3 stages, each building on the previous:

  Stage 1: 3 agents, no obstacles    (learn basic formation)
  Stage 2: 5 agents, no obstacles    (scale up formation)
  Stage 3: 5 agents, 2 obstacles     (add dynamic obstacles)

Each stage warm-starts from the previous stage's actor weights.
The critic is re-initialized when agent count changes (its input
size depends on num_agents × obs_dim).

This is "Curriculum Learning" — a technique where you start with
an easy version of the task and gradually increase difficulty.
Without it, agents often fail to learn anything because the full
task is too complex from the start.

Usage:
    python curriculum_train.py

    # Custom episodes per stage
    python curriculum_train.py --stage-episodes 500

    # With visualization
    python curriculum_train.py --render
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.formation_config import print_config
from train import train


def parse_args():
    parser = argparse.ArgumentParser(description="Curriculum Training for UAV Swarm")
    parser.add_argument("--stage-episodes", type=int, default=1000,
                        help="Episodes per stage (default: 1000)")
    parser.add_argument("--render", action="store_true",
                        help="Render last episode of each stage")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--log-dir", type=str, default="runs",
                        help="Tensorboard log directory")
    return parser.parse_args()


def main():
    args = parse_args()

    print_config()
    print("\n" + "=" * 60)
    print("  CURRICULUM TRAINING — 3 Stages")
    print("=" * 60)
    print(f"  Stage 1: 3 agents, 0 obstacles ({args.stage_episodes} episodes)")
    print(f"  Stage 2: 5 agents, 0 obstacles ({args.stage_episodes} episodes)")
    print(f"  Stage 3: 5 agents, 2 obstacles ({args.stage_episodes} episodes)")
    print(f"  Total:   {args.stage_episodes * 3} episodes")
    print("=" * 60)

    start_time = time.time()

    # Define curriculum stages
    stages = [
        {"name": "Stage 1: 3 agents, no obstacles",
         "num_agents": 3, "num_obstacles": 0, "resume": None},
        {"name": "Stage 2: 5 agents, no obstacles",
         "num_agents": 5, "num_obstacles": 0,
         "resume": os.path.join(args.checkpoint_dir, "stage1_final.pt")},
        {"name": "Stage 3: 5 agents, 2 obstacles",
         "num_agents": 5, "num_obstacles": 2,
         "resume": os.path.join(args.checkpoint_dir, "stage2_final.pt")},
    ]

    for i, stage in enumerate(stages):
        stage_num = i + 1
        print(f"\n{'#' * 60}")
        print(f"  CURRICULUM {stage['name']}")
        print(f"{'#' * 60}\n")

        # Build args namespace for train()
        train_args = argparse.Namespace(
            num_agents=stage["num_agents"],
            num_obstacles=stage["num_obstacles"],
            episodes=args.stage_episodes,
            render_every=args.stage_episodes if args.render else 0,
            verbose=False,
            checkpoint_dir=args.checkpoint_dir,
            resume=stage["resume"] if stage["resume"] and os.path.exists(stage["resume"]) else None,
            log_dir=os.path.join(args.log_dir, f"stage{stage_num}"),
        )

        train(train_args)

        # Rename final checkpoint for this stage
        src = os.path.join(args.checkpoint_dir, "final.pt")
        dst = os.path.join(args.checkpoint_dir, f"stage{stage_num}_final.pt")
        if os.path.exists(src):
            os.rename(src, dst)
            print(f"  Stage {stage_num} checkpoint: {dst}")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  CURRICULUM COMPLETE")
    print(f"  Total time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"  Final checkpoint: {os.path.join(args.checkpoint_dir, 'stage3_final.pt')}")
    print(f"  Tensorboard: tensorboard --logdir {args.log_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
