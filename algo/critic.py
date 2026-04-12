"""
Critic Network — The Value Estimator

The Critic estimates "how good is the current situation?"
It takes the GLOBAL state (all agents' observations concatenated)
and outputs a single value V(s).

This is the "centralized" part of CTDE:
  - During TRAINING: the critic sees everything (centralized)
  - During EXECUTION: only the actor runs (decentralized, local obs only)

The value estimate is used to compute advantages for PPO:
    advantage = actual_return - estimated_value
    If advantage > 0: "this was better than expected" → reinforce
    If advantage < 0: "this was worse than expected" → discourage
"""

import torch
import torch.nn as nn

from env.formation_config import OBS_DIM, HIDDEN_DIM


class Critic(nn.Module):
    """
    Value network: global_state → V(s).

    Architecture: global_state_dim → 128 → 128 → 1

    The global state is the concatenation of ALL agents' observations,
    so the critic has a complete picture of the world state.
    """

    def __init__(self, num_agents=3, obs_dim=OBS_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()

        self.num_agents = num_agents
        self.global_dim = num_agents * obs_dim

        self.network = nn.Sequential(
            nn.Linear(self.global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Single value output
        )

        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization for stable RL training."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, global_state):
        """
        Forward pass: global_state → value estimate.

        Args:
            global_state: Tensor of shape (batch, global_dim)
        Returns:
            value: Tensor of shape (batch, 1)
        """
        return self.network(global_state)

    def get_value(self, global_state):
        """
        Get value estimate for the given global state.

        Args:
            global_state: numpy array or tensor
        Returns:
            value: float
        """
        if not isinstance(global_state, torch.Tensor):
            global_state = torch.FloatTensor(global_state)
        if global_state.dim() == 1:
            global_state = global_state.unsqueeze(0)

        value = self.forward(global_state)
        return value.squeeze().detach()

    def explain(self):
        """Print a human-readable description of the network architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("\n" + "=" * 50)
        print("  CRITIC NETWORK (Value Estimator)")
        print("=" * 50)
        print(f"\n  Purpose: Estimates V(s) — how good is the current state?")
        print(f"  Scope:   GLOBAL — sees all agents' observations")
        print(f"  Used:    Only during training (CTDE)")
        print(f"\n  Architecture:")
        print(f"    Input:  {self.global_dim} ({self.num_agents} agents × {OBS_DIM} obs)")
        print(f"      ↓")
        print(f"    Linear({self.global_dim} → {HIDDEN_DIM}) + ReLU")
        print(f"      ↓")
        print(f"    Linear({HIDDEN_DIM} → {HIDDEN_DIM}) + ReLU")
        print(f"      ↓")
        print(f"    Linear({HIDDEN_DIM} → 1)")
        print(f"      ↓")
        print(f"    Output: 1 (scalar value estimate)")
        print(f"\n  Parameters: {total_params:,} total ({trainable:,} trainable)")
        print("=" * 50)
