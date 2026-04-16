"""
MAPPO — Multi-Agent Proximal Policy Optimization

This is the main algorithm coordinator. It ties together:
  - Actor (shared policy network)
  - Critic (centralized value network)
  - Buffer (episode storage)
  - PPO update loop

CTDE Architecture (Centralized Training, Decentralized Execution):
  ┌──────────────────────────────────────────────────────┐
  │  TRAINING (centralized)                               │
  │                                                       │
  │  Actor (shared) ←── obs_i ──→ action_i               │
  │  Critic ←── [obs_0, obs_1, ..., obs_N] ──→ V(s)     │
  │                                                       │
  │  PPO update uses both actor and critic                │
  └──────────────────────────────────────────────────────┘
  ┌──────────────────────────────────────────────────────┐
  │  EXECUTION (decentralized)                            │
  │                                                       │
  │  Each agent only runs: Actor(obs_i) → action_i       │
  │  No critic, no communication between agents           │
  └──────────────────────────────��───────────────────────┘
"""

import os
import torch
import torch.nn as nn
import numpy as np

from algo.actor import Actor
from algo.critic import Critic
from algo.buffer import RolloutBuffer
from env.formation_config import (
    OBS_DIM, ACTION_DIM, HIDDEN_DIM, MAX_STEPS,
    LEARNING_RATE, GAMMA, GAE_LAMBDA, CLIP_EPSILON,
    PPO_EPOCHS, BATCH_SIZE, EPISODES_PER_UPDATE,
    ENTROPY_COEF, VALUE_LOSS_COEF,
)


class MAPPO:
    """
    Multi-Agent PPO coordinator.

    Manages the shared actor and centralized critic,
    collects episodes, and runs PPO updates.
    """

    def __init__(self, num_agents=3, obs_dim=OBS_DIM, action_dim=ACTION_DIM,
                 lr=LEARNING_RATE, device="cpu"):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        # Shared actor — all agents use the same policy
        self.actor = Actor(obs_dim, action_dim, HIDDEN_DIM).to(device)

        # Centralized critic — sees global state during training
        self.critic = Critic(num_agents, obs_dim, HIDDEN_DIM).to(device)

        # Separate optimizers (MAPPO uses Adam eps=1e-5 for stability)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr, eps=1e-5)

        # Value normalization (running mean/std of returns)
        self.value_normalizer = ValueNormalizer(device=device)

        # Training stats
        self.total_updates = 0

    def collect_episode(self, env, verbose=False):
        """
        Run one full episode and store all transitions in a buffer.

        Returns:
            buffer:       RolloutBuffer with the episode data
            episode_info: dict with episode statistics
        """
        buffer = RolloutBuffer()
        obs, _ = env.reset()
        agents = list(env.agents)

        total_rewards = {agent: 0.0 for agent in agents}
        total_collisions = 0
        formation_errors = []

        step = 0
        while env.agents:
            # Get global state for the critic
            global_state = env.get_global_state()

            # Get value estimate from critic
            value = self.critic.get_value(global_state).item()

            # Each agent picks an action using the shared actor
            actions = {}
            log_probs = {}
            for agent in env.agents:
                action, log_prob = self.actor.get_action(obs[agent])
                actions[agent] = action
                log_probs[agent] = log_prob

            # Step the environment
            next_obs, rewards, terms, truncs, infos = env.step(actions)

            # Check if episode is done
            done = any(terms.values()) or any(truncs.values())

            # Store transition
            buffer.store(obs, global_state, actions, log_probs, rewards, value, done)

            # Track statistics
            for agent in agents:
                if agent in rewards:
                    total_rewards[agent] += rewards[agent]
                if agent in infos:
                    formation_errors.append(infos[agent].get("formation_error", 0))
                    total_collisions += infos[agent].get("num_agent_collisions", 0)

            obs = next_obs
            step += 1

        # Compute returns and advantages using GAE
        # Last value is 0 because the episode ended
        buffer.compute_returns_and_advantages(GAMMA, GAE_LAMBDA, last_value=0.0,
                                               verbose=verbose)

        # Episode statistics
        mean_reward = np.mean(list(total_rewards.values()))
        mean_formation_error = np.mean(formation_errors) if formation_errors else 0.0
        reached_target = step < MAX_STEPS  # Early exit means target was reached

        episode_info = {
            "mean_reward": mean_reward,
            "total_rewards": total_rewards,
            "steps": step,
            "collisions": total_collisions // 2,  # Counted per agent, so halve
            "mean_formation_error": mean_formation_error,
            "distance_traveled": env.positions[:, 0].mean() if env.positions is not None else 0,
            "reached_target": reached_target,
        }

        if verbose:
            print(f"\n  [Episode] {step} steps, mean_reward={mean_reward:.2f}, "
                  f"collisions={episode_info['collisions']}, "
                  f"formation_error={mean_formation_error:.2f}")

        return buffer, episode_info

    def collect_episodes(self, env, n_episodes=EPISODES_PER_UPDATE, verbose=False):
        """
        Collect multiple episodes into a single merged buffer.

        This is the MAPPO SOTA approach: more data per PPO update = more
        stable gradients = less training variance.

        Returns:
            merged_buffer: RolloutBuffer with all episodes' data
            all_infos:     list of episode_info dicts
        """
        merged_buffer = None
        all_infos = []

        for i in range(n_episodes):
            buffer, ep_info = self.collect_episode(env, verbose=(verbose and i == 0))
            all_infos.append(ep_info)

            if merged_buffer is None:
                merged_buffer = buffer
            else:
                merged_buffer.merge(buffer)

        # Re-normalize advantages across all episodes
        merged_buffer.normalize_advantages()

        return merged_buffer, all_infos

    def update(self, buffer, verbose=False):
        """
        Run PPO update on the collected episode data.

        PPO's key idea: limit how much the policy can change per update.
        This prevents catastrophic policy updates that can ruin training.

        The "clipping" mechanism:
            ratio = new_prob / old_prob
            If ratio is too far from 1.0 (outside [1-eps, 1+eps]),
            the gradient is clipped — preventing large updates.

        Returns:
            dict with loss statistics
        """
        self.total_updates += 1
        agents = list(buffer.rewards[0].keys())

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_clip_fraction = 0
        num_batches = 0

        for epoch in range(PPO_EPOCHS):
            for batch in buffer.get_batches(BATCH_SIZE, agents, device=self.device):
                obs, global_states, actions, old_log_probs, returns, advantages = batch

                # --- Actor (Policy) Update ---
                # Evaluate the "new" probability of the "old" actions
                new_log_probs, entropy = self.actor.evaluate_action(obs, actions)

                # Probability ratio: how much has the policy changed?
                ratio = torch.exp(new_log_probs - old_log_probs)

                # PPO clipped objective (the core of PPO!)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus — encourages exploration
                entropy_loss = -entropy.mean()

                actor_loss = policy_loss + ENTROPY_COEF * entropy_loss

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                # Gradient clipping (MAPPO uses max_norm=10.0)
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
                self.actor_optimizer.step()

                # --- Critic (Value) Update ---
                values = self.critic(global_states).squeeze(-1)
                # Normalize returns for stable value learning
                normalized_returns = self.value_normalizer.normalize(returns)
                # Huber loss is more robust to outliers than MSE (e.g. completion bonus)
                value_loss = VALUE_LOSS_COEF * nn.HuberLoss(delta=10.0)(values, normalized_returns)

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
                self.critic_optimizer.step()

                # Update running stats for value normalization
                self.value_normalizer.update(returns)

                # Track stats
                with torch.no_grad():
                    clip_fraction = ((ratio - 1.0).abs() > CLIP_EPSILON).float().mean()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_clip_fraction += clip_fraction.item()
                num_batches += 1

        # Average over all batches and epochs
        n = max(num_batches, 1)
        stats = {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
            "clip_fraction": total_clip_fraction / n,
            "update_num": self.total_updates,
        }

        if verbose:
            print(f"\n  [PPO Update #{self.total_updates}]")
            print(f"    Policy loss:   {stats['policy_loss']:.4f}")
            print(f"    Value loss:    {stats['value_loss']:.4f}")
            print(f"    Entropy:       {stats['entropy']:.4f}")
            print(f"    Clip fraction: {stats['clip_fraction']:.4f}")

        return stats

    def save(self, path):
        """Save model checkpoints."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "total_updates": self.total_updates,
            "num_agents": self.num_agents,
        }, path)
        print(f"  [SAVE] Checkpoint saved to {path}")

    def load(self, path, load_critic=True):
        """
        Load model checkpoints.

        Args:
            path: Path to checkpoint file
            load_critic: If False, only load actor weights (useful for
                        curriculum learning when agent count changes)
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(checkpoint["actor"])
        if load_critic and checkpoint.get("num_agents") == self.num_agents:
            self.critic.load_state_dict(checkpoint["critic"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.total_updates = checkpoint.get("total_updates", 0)
        print(f"  [LOAD] Checkpoint loaded from {path}")
        if not load_critic or checkpoint.get("num_agents") != self.num_agents:
            print(f"  [LOAD] Critic NOT loaded (agent count changed: "
                  f"{checkpoint.get('num_agents')} → {self.num_agents})")


class ValueNormalizer:
    """
    Running mean/std normalization for value targets.

    This is the single most impactful trick from the MAPPO paper.
    It normalizes returns so the critic always regresses to targets
    near zero mean and unit variance, regardless of reward scale.
    """

    def __init__(self, device="cpu"):
        self.running_mean = torch.zeros(1, device=device)
        self.running_var = torch.ones(1, device=device)
        self.count = 1e-4

    def update(self, values):
        """Update running statistics with new values."""
        batch_mean = values.mean()
        batch_var = values.var()
        batch_count = values.numel()

        delta = batch_mean - self.running_mean
        total_count = self.count + batch_count

        self.running_mean = self.running_mean + delta * batch_count / total_count
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        self.running_var = m2 / total_count
        self.count = total_count

    def normalize(self, values):
        """Normalize values using running statistics."""
        return (values - self.running_mean) / (self.running_var.sqrt() + 1e-8)
