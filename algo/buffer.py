"""
Rollout Buffer — Stores One Episode's Experience for All Agents

During each episode, we collect transitions:
    (observation, action, log_prob, reward, value, done)

After the episode, we compute:
    1. Returns (discounted cumulative rewards)
    2. Advantages (using GAE — Generalized Advantage Estimation)

Then we feed mini-batches to PPO for the policy update.

GAE (Generalized Advantage Estimation) is a key concept:
    It balances bias vs. variance in advantage estimation.
    - lambda=0 → high bias, low variance (just one-step TD)
    - lambda=1 → low bias, high variance (full Monte Carlo return)
    - lambda=0.95 → sweet spot for most RL tasks
"""

import torch
import numpy as np


class RolloutBuffer:
    """
    Stores one episode's worth of data for all agents.

    Data layout: each list has one entry per timestep.
    Each entry is a dict mapping agent_id → value.
    """

    def __init__(self):
        self.observations = []     # Per-agent local observations
        self.global_states = []    # Global state (concat of all obs)
        self.actions = []          # Per-agent actions
        self.log_probs = []        # Per-agent action log probabilities
        self.rewards = []          # Per-agent rewards
        self.values = []           # Critic's value estimates
        self.dones = []            # Whether episode ended

        # Computed after episode
        self.returns = None
        self.advantages = None

    def store(self, obs_dict, global_state, actions_dict, log_probs_dict,
              rewards_dict, value, done):
        """
        Store one timestep of experience for all agents.

        Args:
            obs_dict:       {agent_id: obs_array}
            global_state:   numpy array (concat of all obs)
            actions_dict:   {agent_id: action_array}
            log_probs_dict: {agent_id: log_prob_tensor}
            rewards_dict:   {agent_id: float}
            value:          float (critic's estimate)
            done:           bool
        """
        self.observations.append(obs_dict)
        self.global_states.append(global_state)
        self.actions.append(actions_dict)
        self.log_probs.append(log_probs_dict)
        self.rewards.append(rewards_dict)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns_and_advantages(self, gamma, gae_lambda, last_value=0.0,
                                       verbose=False):
        """
        Compute discounted returns and GAE advantages.

        This is where the magic happens — GAE smoothly blends between
        Monte Carlo returns (high variance) and TD estimates (high bias).

        The formula (working backwards from the end of the episode):
            delta_t     = reward_t + gamma * V(s_{t+1}) - V(s_t)
            advantage_t = delta_t + gamma * lambda * advantage_{t+1}

        Args:
            gamma:      Discount factor (0.99 = care about future)
            gae_lambda: GAE lambda (0.95 = good bias-variance tradeoff)
            last_value: V(s_T) — value estimate of the final state
            verbose:    Print step-by-step GAE computation
        """
        T = len(self.rewards)
        if T == 0:
            return

        # Get agent list from first timestep
        agents = list(self.rewards[0].keys())

        # Compute mean reward across agents per step (shared value function)
        mean_rewards = []
        for t in range(T):
            r = np.mean([self.rewards[t][a] for a in agents])
            mean_rewards.append(r)

        # Values as tensor
        values = torch.tensor([self.values[t] for t in range(T)] + [last_value],
                              dtype=torch.float32)

        # GAE computation (working backwards)
        advantages = torch.zeros(T, dtype=torch.float32)
        returns = torch.zeros(T, dtype=torch.float32)
        gae = 0.0

        if verbose:
            print("\n  GAE Computation (last 5 steps shown):")
            print("  " + "-" * 60)

        for t in reversed(range(T)):
            reward = mean_rewards[t]
            next_value = values[t + 1]
            current_value = values[t]
            done = self.dones[t]

            # TD error: how much better/worse was this step than expected?
            delta = reward + gamma * next_value * (1 - done) - current_value
            # GAE: accumulated advantage with exponential decay
            gae = delta + gamma * gae_lambda * (1 - done) * gae
            advantages[t] = gae
            returns[t] = gae + current_value

            if verbose and t >= T - 5:
                print(f"    t={t:3d}: reward={reward:+.3f}, V(s)={current_value:.3f}, "
                      f"V(s')={next_value:.3f}, delta={delta:.3f}, advantage={gae:.3f}")

        if verbose:
            print("  " + "-" * 60)
            print(f"    Mean advantage: {advantages.mean():.3f}")
            print(f"    Std advantage:  {advantages.std():.3f}")

        # Normalize advantages (zero mean, unit variance)
        # This is crucial for stable PPO training
        self.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.returns = returns

    def get_batches(self, batch_size, agents, device="cpu"):
        """
        Yield mini-batches for PPO update.

        Each batch contains: observations, global_states, actions,
        old_log_probs, returns, advantages for a random subset of timesteps.

        We expand each timestep into one entry per agent (parameter sharing
        means we treat all agents' data as the same "type" of experience).
        """
        T = len(self.observations)

        # Flatten: each (timestep, agent) becomes one training sample
        all_obs = []
        all_global = []
        all_actions = []
        all_log_probs = []
        all_returns = []
        all_advantages = []

        for t in range(T):
            for agent in agents:
                if agent not in self.observations[t]:
                    continue
                all_obs.append(self.observations[t][agent])
                all_global.append(self.global_states[t])
                all_actions.append(self.actions[t][agent])
                all_log_probs.append(self.log_probs[t][agent])
                all_returns.append(self.returns[t])
                all_advantages.append(self.advantages[t])

        # Convert to tensors and move to device
        all_obs = torch.FloatTensor(np.array(all_obs)).to(device)
        all_global = torch.FloatTensor(np.array(all_global)).to(device)
        all_actions = torch.FloatTensor(np.array(all_actions)).to(device)
        all_log_probs = torch.stack(all_log_probs).to(device)
        all_returns = torch.stack(all_returns).to(device)
        all_advantages = torch.stack(all_advantages).to(device)

        # Shuffle and yield batches
        N = len(all_obs)
        indices = np.random.permutation(N)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            idx = indices[start:end]
            yield (
                all_obs[idx],
                all_global[idx],
                all_actions[idx],
                all_log_probs[idx],
                all_returns[idx],
                all_advantages[idx],
            )

    def summary(self):
        """Print a summary of what's in the buffer."""
        T = len(self.rewards)
        if T == 0:
            print("\n  [Buffer] Empty — no data collected yet.")
            return

        agents = list(self.rewards[0].keys())
        total_reward = sum(
            sum(self.rewards[t][a] for a in agents) / len(agents)
            for t in range(T)
        )

        print(f"\n  [Buffer] {T} timesteps, {len(agents)} agents")
        print(f"    Total entries: {T * len(agents)}")
        print(f"    Mean reward (across agents, across time): {total_reward / T:.3f}")
        if self.advantages is not None:
            print(f"    Advantage range: [{self.advantages.min():.3f}, {self.advantages.max():.3f}]")
            print(f"    Return range:    [{self.returns.min():.3f}, {self.returns.max():.3f}]")
