"""
Physics Sandbox — Interactive Pygame Demo (Phase 1)

This is your first hands-on experience with the 2D world.
NO reinforcement learning here — just raw physics so you can
understand how agents move before adding any intelligence.

Controls:
    Arrow keys  — Accelerate Agent 0 (the gold one)
    SPACE       — Pause / unpause
    R           — Reset all agents
    Q / ESC     — Quit
    +/-         — Speed up / slow down simulation

What to observe:
    - Euler integration: velocity += acceleration * dt, position += velocity * dt
    - Velocity is clamped to MAX_SPEED — agents can't go infinitely fast
    - Agents bounce off world boundaries
    - Collisions are detected (agents flash red) but they pass through each other
    - The terminal prints physics state each step so you can see the numbers
"""

import sys
import os
import numpy as np

# Add project root to path so we can import config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.formation_config import (
    WORLD_SIZE, DT, MAX_SPEED, AGENT_RADIUS, NUM_AGENTS,
    get_formation_offsets, print_config,
)

# Try to import pygame
try:
    import pygame
except ImportError:
    print("Pygame not installed. Run: pip install pygame")
    sys.exit(1)


# =============================================================================
# Physics Engine (same as what the RL environment will use)
# =============================================================================

class PhysicsWorld:
    """
    Simple 2D physics with Euler integration.

    This is the SAME physics model the RL environment uses.
    By playing with it here, you'll understand exactly how
    actions (accelerations) translate to agent movement.
    """

    def __init__(self, num_agents=3):
        self.num_agents = num_agents
        self.reset()

    def reset(self):
        """Place agents in a cluster near the left side of the world."""
        self.positions = np.random.uniform(15, 30, size=(self.num_agents, 2))
        self.velocities = np.zeros((self.num_agents, 2))
        self.step_count = 0
        self.collisions = []  # List of colliding pairs this step
        print("\n[RESET] Agents placed at:")
        for i in range(self.num_agents):
            print(f"  Agent {i}: pos=({self.positions[i][0]:.1f}, {self.positions[i][1]:.1f})")

    def step(self, accelerations, verbose=True):
        """
        Advance physics by one timestep.

        This is Euler integration — the simplest numerical method:
            velocity_new = velocity_old + acceleration * dt
            position_new = position_old + velocity_new * dt

        It's not perfectly accurate, but good enough for our simulation.
        """
        self.step_count += 1
        self.collisions = []

        # 1. Apply accelerations (clamp to [-1, 1])
        accel = np.clip(accelerations, -1.0, 1.0)

        # 2. Euler integration: update velocity
        self.velocities += accel * DT

        # 3. Clamp velocity to MAX_SPEED
        speeds = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        mask = speeds > MAX_SPEED
        if mask.any():
            # Normalize and scale to MAX_SPEED where exceeded
            self.velocities = np.where(
                mask,
                self.velocities / (speeds + 1e-8) * MAX_SPEED,
                self.velocities
            )

        # 4. Update positions
        self.positions += self.velocities * DT

        # 5. Bounce off walls (reflect velocity when hitting boundary)
        for i in range(self.num_agents):
            for axis in range(2):
                if self.positions[i, axis] < AGENT_RADIUS:
                    self.positions[i, axis] = AGENT_RADIUS
                    self.velocities[i, axis] *= -0.5  # Bounce with energy loss
                elif self.positions[i, axis] > WORLD_SIZE - AGENT_RADIUS:
                    self.positions[i, axis] = WORLD_SIZE - AGENT_RADIUS
                    self.velocities[i, axis] *= -0.5

        # 6. Detect collisions (pairwise distance check)
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                if dist < 2 * AGENT_RADIUS:
                    self.collisions.append((i, j))

        # 7. Print state (educational output)
        if verbose and self.step_count % 10 == 0:
            print(f"\n--- Step {self.step_count} ---")
            for i in range(self.num_agents):
                speed = np.linalg.norm(self.velocities[i])
                print(f"  Agent {i}: "
                      f"pos=({self.positions[i][0]:6.1f}, {self.positions[i][1]:6.1f})  "
                      f"vel=({self.velocities[i][0]:+5.2f}, {self.velocities[i][1]:+5.2f})  "
                      f"speed={speed:.2f}")
            if self.collisions:
                print(f"  ⚠ COLLISIONS: {self.collisions}")


# =============================================================================
# Pygame Visualization
# =============================================================================

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GOLD = (255, 215, 0)
BLUE = (70, 130, 180)
RED = (220, 50, 50)
GRAY = (100, 100, 100)
GREEN = (50, 200, 50)
DARK_BG = (20, 20, 30)

WINDOW_SIZE = 700
SCALE = WINDOW_SIZE / WORLD_SIZE  # Convert world coords to pixel coords


def world_to_screen(pos):
    """Convert world coordinates to screen pixel coordinates."""
    return (int(pos[0] * SCALE), int(pos[1] * SCALE))


def run_sandbox():
    print_config()
    print("\n" + "=" * 60)
    print("  PHYSICS SANDBOX — Interactive Demo")
    print("=" * 60)
    print("  Arrow keys = accelerate Agent 0 (gold)")
    print("  SPACE = pause   R = reset   Q = quit")
    print("  +/- = speed up/slow down")
    print("=" * 60)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("UAV Physics Sandbox — Phase 1")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)
    big_font = pygame.font.SysFont("monospace", 20, bold=True)

    world = PhysicsWorld(num_agents=NUM_AGENTS)

    # Trail storage (last N positions per agent)
    trail_length = 100
    trails = [[] for _ in range(NUM_AGENTS)]

    paused = False
    sim_speed = 1  # Steps per frame
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
                    paused = not paused
                    print("[PAUSED]" if paused else "[RESUMED]")
                elif event.key == pygame.K_r:
                    world.reset()
                    trails = [[] for _ in range(NUM_AGENTS)]
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    sim_speed = min(sim_speed + 1, 10)
                    print(f"[SPEED] {sim_speed}x")
                elif event.key == pygame.K_MINUS:
                    sim_speed = max(sim_speed - 1, 1)
                    print(f"[SPEED] {sim_speed}x")

        if paused:
            # Still draw, just don't step physics
            clock.tick(30)
            # Draw "PAUSED" overlay
            pause_text = big_font.render("PAUSED", True, WHITE)
            screen.blit(pause_text, (WINDOW_SIZE // 2 - 50, WINDOW_SIZE // 2))
            pygame.display.flip()
            continue

        # --- Build accelerations ---
        accelerations = np.random.uniform(-0.3, 0.3, size=(NUM_AGENTS, 2))

        # Agent 0 controlled by arrow keys
        keys = pygame.key.get_pressed()
        player_accel = np.array([0.0, 0.0])
        if keys[pygame.K_RIGHT]:
            player_accel[0] = 1.0
        if keys[pygame.K_LEFT]:
            player_accel[0] = -1.0
        if keys[pygame.K_DOWN]:
            player_accel[1] = 1.0
        if keys[pygame.K_UP]:
            player_accel[1] = -1.0
        accelerations[0] = player_accel

        # --- Step physics ---
        for _ in range(sim_speed):
            world.step(accelerations, verbose=True)

        # --- Store trails ---
        for i in range(NUM_AGENTS):
            trails[i].append(world.positions[i].copy())
            if len(trails[i]) > trail_length:
                trails[i].pop(0)

        # --- Draw ---
        screen.fill(DARK_BG)

        # Grid lines
        for g in range(0, int(WORLD_SIZE) + 1, 10):
            px = int(g * SCALE)
            pygame.draw.line(screen, (30, 30, 40), (px, 0), (px, WINDOW_SIZE), 1)
            pygame.draw.line(screen, (30, 30, 40), (0, px), (WINDOW_SIZE, px), 1)

        # Formation target positions (dashed circles showing where agents should be)
        offsets = get_formation_offsets(NUM_AGENTS)
        centroid = np.mean(world.positions, axis=0)
        for i in range(NUM_AGENTS):
            target = centroid + offsets[i]
            tpos = world_to_screen(target)
            pygame.draw.circle(screen, GRAY, tpos, int(AGENT_RADIUS * SCALE * 2), 1)

        # Trails
        for i in range(NUM_AGENTS):
            color = GOLD if i == 0 else BLUE
            for t_idx, t_pos in enumerate(trails[i]):
                alpha = t_idx / len(trails[i])
                faded = tuple(int(c * alpha * 0.4) for c in color)
                spos = world_to_screen(t_pos)
                pygame.draw.circle(screen, faded, spos, 2)

        # Agents
        colliding_agents = set()
        for (a, b) in world.collisions:
            colliding_agents.add(a)
            colliding_agents.add(b)

        for i in range(NUM_AGENTS):
            pos = world_to_screen(world.positions[i])
            radius = int(AGENT_RADIUS * SCALE)

            if i in colliding_agents:
                # Flash red on collision
                pygame.draw.circle(screen, RED, pos, radius + 4)
                pygame.draw.circle(screen, RED, pos, radius)
            elif i == 0:
                pygame.draw.circle(screen, GOLD, pos, radius)
            else:
                pygame.draw.circle(screen, BLUE, pos, radius)

            # Agent label
            label = font.render(str(i), True, WHITE)
            screen.blit(label, (pos[0] - 4, pos[1] - 7))

            # Velocity arrow
            vel = world.velocities[i]
            arrow_end = (pos[0] + int(vel[0] * SCALE * 3),
                         pos[1] + int(vel[1] * SCALE * 3))
            pygame.draw.line(screen, GREEN, pos, arrow_end, 2)

        # HUD
        hud_lines = [
            f"Step: {world.step_count}",
            f"Speed: {sim_speed}x",
            f"Collisions this step: {len(world.collisions)}",
            f"Agent 0 speed: {np.linalg.norm(world.velocities[0]):.2f}",
            "",
            "Arrow keys = move Agent 0",
            "SPACE=pause  R=reset  Q=quit",
        ]
        for idx, line in enumerate(hud_lines):
            text = font.render(line, True, WHITE)
            screen.blit(text, (10, 10 + idx * 18))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    print("\n[QUIT] Sandbox closed.")


if __name__ == "__main__":
    run_sandbox()
