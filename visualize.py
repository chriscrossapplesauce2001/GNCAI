"""
Visualize — Pygame Renderer for UAV Formation

Renders the environment state to a Pygame window.
Can be used:
  - During training (called by train.py every N episodes)
  - Standalone to watch a saved checkpoint

Controls:
    SPACE   — Pause / unpause
    +/-     — Speed up / slow down
    Q/ESC   — Quit / close window
    R       — Reset episode

Color coding:
    Gold circle    — Agent 0 (leader)
    Blue circles   — Follower agents
    Gray dashed    — Formation target positions
    Green lines    — Connections to formation targets
    Red circles    — Obstacles (Phase 5)
    Green arrows   — Velocity vectors
"""

import numpy as np

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

from env.formation_config import WORLD_SIZE, AGENT_RADIUS, OBSTACLE_RADIUS

# Colors
DARK_BG = (20, 20, 30)
WHITE = (255, 255, 255)
GOLD = (255, 215, 0)
BLUE = (70, 130, 180)
RED = (220, 50, 50)
GRAY = (80, 80, 80)
GREEN = (50, 200, 50)
LIGHT_GREEN = (100, 255, 100)
ORANGE = (255, 165, 0)

WINDOW_SIZE = 700
SCALE = WINDOW_SIZE / WORLD_SIZE


def world_to_screen(pos):
    return (int(pos[0] * SCALE), int(pos[1] * SCALE))


class Renderer:
    """
    Pygame-based renderer for the UAV formation environment.

    Keeps track of agent trails and displays a HUD with live metrics.
    """

    def __init__(self):
        if not HAS_PYGAME:
            raise ImportError("pygame is required for visualization")
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("UAV Swarm Formation — MARL Visualization")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 13)
        self.big_font = pygame.font.SysFont("monospace", 18, bold=True)
        self.trails = {}
        self.trail_length = 80
        self.paused = False
        self.sim_speed = 1
        self.running = True

    def reset_trails(self):
        self.trails = {}

    def handle_events(self):
        """Process Pygame events. Returns False if window was closed."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    self.running = False
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    self.sim_speed = min(self.sim_speed + 1, 20)
                elif event.key == pygame.K_MINUS:
                    self.sim_speed = max(self.sim_speed - 1, 1)
        return True

    def render(self, env, step=0, episode=0, total_reward=0.0,
               formation_error=0.0, collisions=0, extra_info=""):
        """
        Render one frame of the environment.

        Args:
            env: UAVFormationEnv instance (reads positions, velocities, obstacles)
            step: Current step number
            episode: Current episode number
            total_reward: Cumulative reward this episode
            formation_error: Current mean formation error
            collisions: Total collisions this episode
            extra_info: Additional text to display
        """
        if not self.running:
            return

        self.screen.fill(DARK_BG)

        # Grid
        for g in range(0, int(WORLD_SIZE) + 1, 10):
            px = int(g * SCALE)
            pygame.draw.line(self.screen, (30, 30, 40), (px, 0), (px, WINDOW_SIZE), 1)
            pygame.draw.line(self.screen, (30, 30, 40), (0, px), (WINDOW_SIZE, px), 1)

        if env.positions is None:
            pygame.display.flip()
            self.clock.tick(60)
            return

        num_agents = env._num_agents

        # Store trails
        for i in range(num_agents):
            if i not in self.trails:
                self.trails[i] = []
            self.trails[i].append(env.positions[i].copy())
            if len(self.trails[i]) > self.trail_length:
                self.trails[i].pop(0)

        # Draw formation targets and connecting lines
        centroid = np.mean(env.positions, axis=0)
        for i in range(num_agents):
            target = centroid + env.formation_offsets[i]
            tpos = world_to_screen(target)
            apos = world_to_screen(env.positions[i])

            # Dashed target circle
            pygame.draw.circle(self.screen, GRAY, tpos, int(AGENT_RADIUS * SCALE * 2.5), 1)

            # Line from agent to its target (green = close, red = far)
            dist = np.linalg.norm(env.positions[i] - target)
            if dist < 3:
                line_color = GREEN
            elif dist < 8:
                line_color = ORANGE
            else:
                line_color = RED
            pygame.draw.line(self.screen, line_color, apos, tpos, 1)

        # Draw obstacles
        if env._num_obstacles > 0 and env.obstacle_positions is not None:
            for o in range(env._num_obstacles):
                opos = world_to_screen(env.obstacle_positions[o])
                radius = int(OBSTACLE_RADIUS * SCALE)
                pygame.draw.circle(self.screen, RED, opos, radius)
                pygame.draw.circle(self.screen, (255, 100, 100), opos, radius, 2)

        # Draw trails
        for i in range(num_agents):
            color = GOLD if i == 0 else BLUE
            for t_idx, t_pos in enumerate(self.trails.get(i, [])):
                alpha = t_idx / max(len(self.trails.get(i, [])), 1)
                faded = tuple(int(c * alpha * 0.5) for c in color)
                spos = world_to_screen(t_pos)
                pygame.draw.circle(self.screen, faded, spos, 2)

        # Draw agents
        for i in range(num_agents):
            pos = world_to_screen(env.positions[i])
            radius = int(AGENT_RADIUS * SCALE)

            # Agent body
            color = GOLD if i == 0 else BLUE
            pygame.draw.circle(self.screen, color, pos, radius)
            pygame.draw.circle(self.screen, WHITE, pos, radius, 1)

            # Agent label
            label = self.font.render(str(i), True, WHITE)
            self.screen.blit(label, (pos[0] - 4, pos[1] - 7))

            # Velocity arrow
            vel = env.velocities[i]
            arrow_end = (pos[0] + int(vel[0] * SCALE * 4),
                         pos[1] + int(vel[1] * SCALE * 4))
            pygame.draw.line(self.screen, LIGHT_GREEN, pos, arrow_end, 2)

        # HUD
        hud_lines = [
            f"Episode: {episode}  Step: {step}",
            f"Total Reward: {total_reward:.1f}",
            f"Formation Error: {formation_error:.2f}",
            f"Collisions: {collisions}",
            f"Speed: {self.sim_speed}x",
            extra_info if extra_info else "",
            "",
            "SPACE=pause  +/-=speed  Q=quit",
        ]
        for idx, line in enumerate(hud_lines):
            if line:
                text = self.font.render(line, True, WHITE)
                self.screen.blit(text, (10, 10 + idx * 16))

        if self.paused:
            text = self.big_font.render("PAUSED", True, WHITE)
            self.screen.blit(text, (WINDOW_SIZE // 2 - 40, WINDOW_SIZE // 2))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if HAS_PYGAME:
            pygame.quit()
