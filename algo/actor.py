"""
Actor Network — The Agent's "Brain" for Choosing Actions

The Actor maps an observation to an action distribution.
It's a simple MLP (Multi-Layer Perceptron):

    observation → [128 neurons] → [128 neurons] → action_mean
                                                 → action_std

The output is a Gaussian distribution, so the agent can:
  - Sample actions (for exploration during training)
  - Evaluate the probability of past actions (for PPO update)

All agents share the SAME actor (parameter sharing) — they're
identical drones that differ only in what they observe.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

from env.formation_config import OBS_DIM, ACTION_DIM, HIDDEN_DIM


class Actor(nn.Module):
    """
    Policy network: observation → action distribution.

    Architecture: obs_dim → 128 → 128 → action_dim (mean)
    Plus a learnable log_std parameter for the action standard deviation.
    """

    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # Squash output to [-1, 1] to match action space
        )

        # Learnable log standard deviation (starts at -0.5 ≈ std of 0.6)
        # This controls how much the agent explores:
        #   High std → lots of exploration (random-ish actions)
        #   Low std → exploitation (confident, deterministic actions)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

        # Initialize weights with small values for stable training
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization — a common trick for RL networks."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.5)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, obs):
        """
        Forward pass: observation → action mean.

        Args:
            obs: Tensor of shape (batch, obs_dim) or (obs_dim,)
        Returns:
            action_mean: Tensor of shape (batch, action_dim)
        """
        return self.network(obs)

    def get_distribution(self, obs):
        """
        Get the full action distribution for the given observation.

        Returns a Normal distribution you can sample from or evaluate.
        """
        action_mean = self.forward(obs)
        # Clamp log_std to prevent extreme values
        log_std = torch.clamp(self.log_std, -2.0, 0.5)
        action_std = log_std.exp()
        return Normal(action_mean, action_std)

    def get_action(self, obs):
        """
        Sample an action for the given observation.

        Used during episode collection (training rollouts).

        Returns:
            action:   numpy array, the sampled action
            log_prob: tensor, log probability of the action (for PPO)
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs)
        device = next(self.parameters()).device
        obs = obs.to(device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        dist = self.get_distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        # Clamp to action space [-1, 1]
        action = torch.clamp(action, -1.0, 1.0)

        return action.squeeze(0).detach().cpu().numpy(), log_prob.squeeze(0).detach().cpu()

    def evaluate_action(self, obs, action):
        """
        Evaluate the log probability and entropy of a given action.

        Used during PPO update — we need to compute the "new" probability
        of actions that were taken under the "old" policy.

        Returns:
            log_prob: tensor, log probability of the action under current policy
            entropy:  tensor, entropy of the distribution (measures exploration)
        """
        dist = self.get_distribution(obs)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

    def explain(self):
        """Print a human-readable description of the network architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("\n" + "=" * 50)
        print("  ACTOR NETWORK (Policy)")
        print("=" * 50)
        print(f"\n  Purpose: Maps observation → action distribution")
        print(f"  Shared:  Yes — all agents use the same weights")
        print(f"\n  Architecture:")
        print(f"    Input:  {OBS_DIM} (observation vector)")
        print(f"      ↓")
        print(f"    Linear({OBS_DIM} → {HIDDEN_DIM}) + ReLU")
        print(f"      ↓")
        print(f"    Linear({HIDDEN_DIM} → {HIDDEN_DIM}) + ReLU")
        print(f"      ↓")
        print(f"    Linear({HIDDEN_DIM} → {ACTION_DIM}) + Tanh")
        print(f"      ↓")
        print(f"    Output: {ACTION_DIM} (action mean, range [-1, 1])")
        print(f"    + learnable log_std ({ACTION_DIM} params)")
        print(f"\n  Parameters: {total_params:,} total ({trainable:,} trainable)")
        print(f"  Current exploration std: {self.log_std.exp().data.numpy()}")
        print("=" * 50)
