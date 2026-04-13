"""
Interactive Explorer — Play with the Trained Swarm

The capstone interactive experience. Load a trained checkpoint and:
  - Watch agents fly in V-formation
  - Click to place obstacles and watch agents react
  - Drag agents out of formation and see them return
  - Toggle between trained and random policy
  - Manually control individual agents

Controls:
    SPACE       — Pause / unpause
    +/-         — Speed up / slow down
    R           — Reset episode
    T           — Toggle trained vs random policy
    1-5         — Take control of agent 1-5 (arrow keys to move)
    0           — Release manual control (all agents use policy)
    LEFT CLICK  — Place/move obstacle at cursor position
    RIGHT CLICK — Remove placed obstacles
    D           — Drag mode: click and drag an agent
    Q/ESC       — Quit

Usage:
    python interactive.py --checkpoint checkpoints/final.pt
    python interactive.py --checkpoint checkpoints/final.pt --num-agents 5
"""

import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.formation_env import UAVFormationEnv
from env.formation_config import WORLD_SIZE, AGENT_RADIUS, OBS_DIM, MAX_SPEED, print_config
from algo.mappo import MAPPO

try:
    import pygame
except ImportError:
    print("pygame is required: pip install pygame")
    sys.exit(1)

from visualize import Renderer, WINDOW_SIZE, SCALE, world_to_screen

# Additional colors
YELLOW = (255, 255, 0)
CYAN = (0, 200, 200)


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive UAV Swarm Explorer")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--num-obstacles", type=int, default=0)
    return parser.parse_args()


def screen_to_world(screen_pos):
    """Convert screen pixel coordinates to world coordinates."""
    return np.array([screen_pos[0] / SCALE, screen_pos[1] / SCALE])


def main():
    args = parse_args()
    print_config()

    print("\n" + "=" * 60)
    print("  INTERACTIVE EXPLORER")
    print("=" * 60)
    print("  SPACE=pause  R=reset  T=toggle policy")
    print("  1-5=control agent  0=release  +/-=speed")
    print("  LEFT CLICK=place obstacle  RIGHT CLICK=remove")
    print("  Q=quit")
    print("=" * 60)

    # Load trained model
    mappo = MAPPO(num_agents=args.num_agents)
    mappo.load(args.checkpoint)
    mappo.actor.eval()

    # Create environment
    env = UAVFormationEnv(num_agents=args.num_agents,
                          num_obstacles=args.num_obstacles, verbose=False)
    obs, _ = env.reset()

    # Setup renderer
    renderer = Renderer()
    pygame.display.set_caption("UAV Swarm — Interactive Explorer")

    # Interactive state
    use_trained = True
    controlled_agent = -1  # -1 = no manual control
    placed_obstacles = []  # Extra obstacles placed by clicking
    total_reward = 0.0
    total_collisions = 0
    episode = 1
    step = 0

    running = True
    while running:
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    renderer.paused = not renderer.paused
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                    renderer.reset_trails()
                    total_reward = 0.0
                    total_collisions = 0
                    episode += 1
                    step = 0
                    print(f"\n[RESET] Episode {episode}")
                elif event.key == pygame.K_t:
                    use_trained = not use_trained
                    mode = "TRAINED policy" if use_trained else "RANDOM policy"
                    print(f"[TOGGLE] Now using {mode}")
                elif event.key in range(pygame.K_1, pygame.K_6):
                    agent_num = event.key - pygame.K_1
                    if agent_num < args.num_agents:
                        controlled_agent = agent_num
                        print(f"[CONTROL] You are now controlling Agent {agent_num}")
                        print(f"  Use arrow keys to steer, press 0 to release")
                elif event.key == pygame.K_0:
                    controlled_agent = -1
                    print("[CONTROL] Released — all agents using policy")
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    renderer.sim_speed = min(renderer.sim_speed + 1, 20)
                elif event.key == pygame.K_MINUS:
                    renderer.sim_speed = max(renderer.sim_speed - 1, 1)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                world_pos = screen_to_world(event.pos)
                if event.button == 1:  # Left click = place obstacle
                    placed_obstacles.append(world_pos.copy())
                    # Inject obstacle into environment
                    _update_env_obstacles(env, args.num_obstacles, placed_obstacles)
                    print(f"[OBSTACLE] Placed at ({world_pos[0]:.1f}, {world_pos[1]:.1f})")
                elif event.button == 3:  # Right click = remove obstacles
                    if placed_obstacles:
                        placed_obstacles.clear()
                        _update_env_obstacles(env, args.num_obstacles, placed_obstacles)
                        print("[OBSTACLE] All placed obstacles removed")

        if renderer.paused:
            renderer.render(env, step, episode, total_reward, 0, total_collisions,
                            "PAUSED | T=toggle policy")
            continue

        if not running:
            break

        # --- Step simulation ---
        for _ in range(renderer.sim_speed):
            if not env.agents:
                # Episode ended, auto-reset
                obs, _ = env.reset()
                renderer.reset_trails()
                total_reward = 0.0
                total_collisions = 0
                episode += 1
                step = 0

            if not env.agents:
                break

            actions = {}
            for i, agent in enumerate(env.agents):
                if i == controlled_agent:
                    # Manual control with arrow keys
                    keys = pygame.key.get_pressed()
                    accel = np.array([0.0, 0.0])
                    if keys[pygame.K_RIGHT]: accel[0] = 1.0
                    if keys[pygame.K_LEFT]: accel[0] = -1.0
                    if keys[pygame.K_DOWN]: accel[1] = 1.0
                    if keys[pygame.K_UP]: accel[1] = -1.0
                    actions[agent] = accel
                elif use_trained:
                    # Use trained policy (deterministic — mean action)
                    obs_t = torch.FloatTensor(obs[agent]).unsqueeze(0)
                    action_mean = mappo.actor(obs_t).squeeze(0).detach().numpy()
                    actions[agent] = np.clip(action_mean, -1, 1)
                else:
                    # Random policy
                    actions[agent] = env.action_space(agent).sample()

            obs, rewards, terms, truncs, infos = env.step(actions)

            total_reward += np.mean([r for r in rewards.values()])
            for info in infos.values():
                total_collisions += info.get("num_agent_collisions", 0)
            step += 1

        # --- Render ---
        errors = [env._get_formation_error(i) for i in range(env._num_agents)]
        mean_error = np.mean(errors) if errors else 0

        mode_str = "TRAINED" if use_trained else "RANDOM"
        ctrl_str = f" | Controlling Agent {controlled_agent}" if controlled_agent >= 0 else ""
        obs_str = f" | {len(placed_obstacles)} placed obs" if placed_obstacles else ""
        extra = f"Policy: {mode_str}{ctrl_str}{obs_str}"

        # Draw extra obstacles manually (they're already in env, renderer draws from env)
        renderer.render(env, step, episode, total_reward, mean_error,
                        total_collisions // 2, extra)

    renderer.close()
    print("\n[QUIT] Interactive explorer closed.")


def _update_env_obstacles(env, base_obstacles, placed):
    """Update environment with placed obstacles."""
    total = base_obstacles + len(placed)
    if total == 0:
        env.obstacle_positions = np.zeros((0, 2))
        env.obstacle_velocities = np.zeros((0, 2))
        env._num_obstacles = 0
        return

    positions = np.zeros((total, 2))
    velocities = np.zeros((total, 2))

    # Keep existing base obstacles
    if base_obstacles > 0 and env.obstacle_positions is not None and len(env.obstacle_positions) >= base_obstacles:
        positions[:base_obstacles] = env.obstacle_positions[:base_obstacles]
        velocities[:base_obstacles] = env.obstacle_velocities[:base_obstacles]

    # Add placed obstacles (stationary)
    for i, pos in enumerate(placed):
        positions[base_obstacles + i] = pos

    env.obstacle_positions = positions
    env.obstacle_velocities = velocities
    env._num_obstacles = total


if __name__ == "__main__":
    args = parse_args()
    main()
