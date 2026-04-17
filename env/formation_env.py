"""
UAV Formation Environment — PettingZoo ParallelEnv

A 100x100 2D world where UAV agents learn to fly in V-formation,
avoid collisions with each other and obstacles, and navigate
toward a target.

Physics: Euler integration with velocity clamping and wall bouncing.
Observations: 20-dimensional vector normalized to [-1, 1].
Actions: 2D acceleration (ax, ay) in [-1, 1].
"""

import functools
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from env.formation_config import (
    WORLD_SIZE, DT, MAX_SPEED, AGENT_RADIUS, OBSTACLE_RADIUS,
    NUM_AGENTS, MAX_STEPS, OBS_DIM, ACTION_DIM,
    APPROACH_WEIGHT, COLLISION_PENALTY,
    TARGET_THRESHOLD, COMPLETION_BONUS,
    get_formation_offsets,
)


class UAVFormationEnv(ParallelEnv):
    """
    Multi-agent UAV formation environment.

    Agents are point masses in a 2D world using Euler integration.
    They must learn to fly in V-formation while avoiding collisions.
    """

    metadata = {"render_modes": [], "name": "uav_formation_v0"}

    def __init__(self, num_agents=NUM_AGENTS, num_obstacles=0, verbose=False):
        super().__init__()
        self._num_agents = num_agents
        self._num_obstacles = num_obstacles
        self.verbose = verbose

        self.possible_agents = [f"uav_{i}" for i in range(num_agents)]
        self.agents = list(self.possible_agents)

        self.formation_offsets = get_formation_offsets(num_agents)

        # State arrays
        self.positions = None
        self.velocities = None
        self.prev_actions = None
        self.obstacle_positions = None
        self.obstacle_velocities = None
        self.target_pos = None

        self._step_count = 0

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = list(self.possible_agents)
        self._step_count = 0

        # Spawn agents in a loose cluster in a random corner
        spawn_x = np.random.uniform(10, 30)
        spawn_y = np.random.uniform(10, 30)
        self.positions = np.random.uniform(-7, 7, size=(self._num_agents, 2)) + [spawn_x, spawn_y]
        self.velocities = np.zeros((self._num_agents, 2))
        self.prev_actions = np.zeros((self._num_agents, ACTION_DIM))

        # Random navigation target — at least 40 units from spawn centroid
        while True:
            self.target_pos = np.random.uniform(20, WORLD_SIZE - 20, size=2)
            spawn_centroid = np.mean(self.positions, axis=0)
            if np.linalg.norm(self.target_pos - spawn_centroid) > 40:
                break

        # Track distances to target for approach reward
        self.prev_distances_to_target = np.linalg.norm(
            self.positions - self.target_pos, axis=1
        )

        # Obstacles: random positions with random slow velocities
        if self._num_obstacles > 0:
            self.obstacle_positions = np.random.uniform(
                30, WORLD_SIZE - 10, size=(self._num_obstacles, 2)
            )
            self.obstacle_velocities = np.random.uniform(
                -0.3, 0.3, size=(self._num_obstacles, 2)
            )
        else:
            self.obstacle_positions = np.zeros((0, 2))
            self.obstacle_velocities = np.zeros((0, 2))

        observations = self._get_all_obs()
        infos = {agent: {} for agent in self.agents}

        if self.verbose:
            print(f"\n[ENV RESET] {self._num_agents} agents, {self._num_obstacles} obstacles")
            for i in range(self._num_agents):
                print(f"  Agent {i}: pos=({self.positions[i][0]:.1f}, {self.positions[i][1]:.1f})")

        return observations, infos

    def step(self, actions):
        if not self.agents:
            return {}, {}, {}, {}, {}

        self._step_count += 1

        # 1. Apply actions — convert dict to array
        accel = np.zeros((self._num_agents, ACTION_DIM))
        for i, agent in enumerate(self.possible_agents):
            if agent in actions:
                accel[i] = np.clip(actions[agent], -1.0, 1.0)

        # 2. Physics update — Euler integration
        self.velocities += accel * DT

        # Clamp velocity to MAX_SPEED
        speeds = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        mask = speeds > MAX_SPEED
        if mask.any():
            self.velocities = np.where(
                mask,
                self.velocities / (speeds + 1e-8) * MAX_SPEED,
                self.velocities
            )

        self.positions += self.velocities * DT

        # 3. Wall bouncing
        for i in range(self._num_agents):
            for axis in range(2):
                if self.positions[i, axis] < AGENT_RADIUS:
                    self.positions[i, axis] = AGENT_RADIUS
                    self.velocities[i, axis] *= -0.5
                elif self.positions[i, axis] > WORLD_SIZE - AGENT_RADIUS:
                    self.positions[i, axis] = WORLD_SIZE - AGENT_RADIUS
                    self.velocities[i, axis] *= -0.5

        # 4. Move obstacles
        if self._num_obstacles > 0 and self.obstacle_positions is not None:
            self.obstacle_positions += self.obstacle_velocities * DT
            # Bounce obstacles off walls
            for o in range(self._num_obstacles):
                for axis in range(2):
                    if self.obstacle_positions[o, axis] < OBSTACLE_RADIUS:
                        self.obstacle_positions[o, axis] = OBSTACLE_RADIUS
                        self.obstacle_velocities[o, axis] *= -1.0
                    elif self.obstacle_positions[o, axis] > WORLD_SIZE - OBSTACLE_RADIUS:
                        self.obstacle_positions[o, axis] = WORLD_SIZE - OBSTACLE_RADIUS
                        self.obstacle_velocities[o, axis] *= -1.0

        # 5. Collision detection
        agent_collisions = np.zeros(self._num_agents, dtype=int)
        for i in range(self._num_agents):
            for j in range(i + 1, self._num_agents):
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                if dist < 2 * AGENT_RADIUS:
                    agent_collisions[i] += 1
                    agent_collisions[j] += 1

        obstacle_collisions = np.zeros(self._num_agents, dtype=int)
        if self._num_obstacles > 0 and len(self.obstacle_positions) > 0:
            for i in range(self._num_agents):
                for o in range(self._num_obstacles):
                    dist = np.linalg.norm(self.positions[i] - self.obstacle_positions[o])
                    if dist < AGENT_RADIUS + OBSTACLE_RADIUS:
                        obstacle_collisions[i] += 1

        # 6. Compute observations
        observations = self._get_all_obs()

        # 7. Compute rewards
        rewards = {}
        infos = {}
        for i, agent in enumerate(self.possible_agents):
            reward, info = self._compute_reward(i, accel[i], agent_collisions[i],
                                                  obstacle_collisions[i])
            rewards[agent] = reward
            infos[agent] = info

        # Store previous actions and distances for next step's rewards
        self.prev_actions = accel.copy()
        self.prev_distances_to_target = np.linalg.norm(
            self.positions - self.target_pos, axis=1
        )

        # 8. Check termination — all agents must be within threshold
        dists_to_target = np.linalg.norm(self.positions - self.target_pos, axis=1)
        reached_target = bool(np.all(dists_to_target < TARGET_THRESHOLD))
        timed_out = self._step_count >= MAX_STEPS

        # Completion bonus for reaching the target
        if reached_target:
            for agent in self.possible_agents:
                rewards[agent] += COMPLETION_BONUS
                infos[agent]["completion_bonus"] = COMPLETION_BONUS

        terminations = {agent: reached_target for agent in self.agents}
        truncations = {agent: (timed_out and not reached_target) for agent in self.agents}

        if reached_target or timed_out:
            self.agents = []

        if self.verbose and self._step_count % 50 == 0:
            mean_reward = np.mean(list(rewards.values()))
            total_coll = sum(agent_collisions)
            print(f"  Step {self._step_count}: reward={mean_reward:+.3f}, "
                  f"collisions={total_coll}")

        return observations, rewards, terminations, truncations, infos

    def _get_all_obs(self):
        """Get observations for all agents."""
        return {agent: self._get_obs(i) for i, agent in enumerate(self.possible_agents)
                if agent in self.agents}

    def _get_obs(self, agent_idx):
        """
        Build the 20-dimensional observation vector for one agent.
        All values normalized to [-1, 1].
        """
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        pos = self.positions[agent_idx]
        vel = self.velocities[agent_idx]
        half_world = WORLD_SIZE / 2.0

        # [0-1] Normalized position (centered at world midpoint)
        obs[0] = (pos[0] - half_world) / half_world
        obs[1] = (pos[1] - half_world) / half_world

        # [2-3] Normalized velocity
        obs[2] = vel[0] / MAX_SPEED
        obs[3] = vel[1] / MAX_SPEED

        # [4-7] Relative positions of 2 nearest neighbors
        if self._num_agents > 1:
            deltas = self.positions - pos  # (N, 2)
            dists = np.linalg.norm(deltas, axis=1)
            dists[agent_idx] = np.inf  # exclude self
            sorted_idx = np.argsort(dists)

            # Nearest neighbor 1
            n1 = sorted_idx[0]
            obs[4] = np.clip(deltas[n1, 0] / (WORLD_SIZE * 0.2), -1, 1)
            obs[5] = np.clip(deltas[n1, 1] / (WORLD_SIZE * 0.2), -1, 1)

            # Nearest neighbor 2 (if exists)
            if self._num_agents > 2:
                n2 = sorted_idx[1]
                obs[6] = np.clip(deltas[n2, 0] / (WORLD_SIZE * 0.2), -1, 1)
                obs[7] = np.clip(deltas[n2, 1] / (WORLD_SIZE * 0.2), -1, 1)

        # [8-9] Delta to formation target
        centroid = np.mean(self.positions, axis=0)
        target = centroid + self.formation_offsets[agent_idx]
        form_delta = target - pos
        obs[8] = np.clip(form_delta[0] / (WORLD_SIZE * 0.2), -1, 1)
        obs[9] = np.clip(form_delta[1] / (WORLD_SIZE * 0.2), -1, 1)

        # [10-11] Delta to navigation target (direction + distance)
        if self.target_pos is not None:
            tgt_delta = self.target_pos - pos
            obs[10] = np.clip(tgt_delta[0] / (WORLD_SIZE * 0.5), -1, 1)
            obs[11] = np.clip(tgt_delta[1] / (WORLD_SIZE * 0.5), -1, 1)

        # [12-19] Obstacle information (up to 2 obstacles, 4 values each)
        if self._num_obstacles > 0 and self.obstacle_positions is not None and len(self.obstacle_positions) > 0:
            # Sort obstacles by distance
            obs_deltas = self.obstacle_positions - pos
            obs_dists = np.linalg.norm(obs_deltas, axis=1)
            obs_sorted = np.argsort(obs_dists)

            for slot in range(min(2, self._num_obstacles)):
                o_idx = obs_sorted[slot]
                base = 12 + slot * 4
                obs[base] = np.clip(obs_deltas[o_idx, 0] / (WORLD_SIZE * 0.2), -1, 1)
                obs[base + 1] = np.clip(obs_deltas[o_idx, 1] / (WORLD_SIZE * 0.2), -1, 1)
                obs[base + 2] = np.clip(self.obstacle_velocities[o_idx, 0] / MAX_SPEED, -1, 1)
                obs[base + 3] = np.clip(self.obstacle_velocities[o_idx, 1] / MAX_SPEED, -1, 1)

        return obs

    def get_global_state(self):
        """
        Concatenate all agents' observations into a global state vector.
        Used by the centralized critic during training.
        """
        all_obs = [self._get_obs(i) for i in range(self._num_agents)]
        return np.concatenate(all_obs)

    def _get_formation_error(self, agent_idx):
        """Compute formation error for a single agent (distance to target position)."""
        centroid = np.mean(self.positions, axis=0)
        target = centroid + self.formation_offsets[agent_idx]
        return np.linalg.norm(self.positions[agent_idx] - target)

    def _compute_reward(self, agent_idx, action, num_agent_collisions, num_obstacle_collisions):
        """
        Compute reward for one agent.
        - Approach: reward closing in on target (scaled by APPROACH_WEIGHT)
        - Collision: harsh penalty per frame of agent-agent overlap
        Plus one-time completion bonus (applied in step()).
        """
        dist_to_target = np.linalg.norm(self.target_pos - self.positions[agent_idx])

        approach_reward = APPROACH_WEIGHT * (self.prev_distances_to_target[agent_idx] - dist_to_target) / (MAX_SPEED * DT)

        collision_penalty = float(num_agent_collisions > 0)

        reward = approach_reward - COLLISION_PENALTY * collision_penalty

        info = {
            "collision_penalty": collision_penalty,
            "dist_to_target": dist_to_target,
            "num_agent_collisions": num_agent_collisions,
            "num_obstacle_collisions": num_obstacle_collisions,
        }

        return reward, info
