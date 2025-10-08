"""Tabular Q-Learning agent (epsilon-greedy) with DQN-like API shape.

This module provides a small, dependency-light Q-Learning agent suitable for
discrete grid-like environments where observations can be encoded into fixed-
width numeric arrays. The agent exposes a minimal API similar to many DQN
implementations so it can be swapped into training loops that expect:
  - agent.act(state, eps) -> action
  - agent.train_step(batch, ...) -> loss/td-error

Notes
-----
- The agent stores action-values in a separate QTable helper (see q_table.py).
- Observations are treated as 1D numpy arrays and converted to keys by the
  QTable (rounded tuple form). This keeps the tabular representation simple
  while allowing floating-point observations that are discretized by rounding.
"""

from __future__ import annotations
import random
from typing import Iterable
import numpy as np

from .q_table import QTable


class QLearning:
    """Tabular Q-Learning agent with an API similar to Deep Q-Network (DQN).

    The implementation is intentionally small and focused — it supports:
      - epsilon-greedy action selection via `act`
      - single-step / batch Q-Learning updates via `train_step`

    The agent stores its Q-values in a QTable instance. Observations passed to
    this agent should be 1D numpy arrays (or convertible to them). The QTable
    will round/encode observations into hashable keys.

    Args:
        obs_dim: Dimensionality of the observation vector (not used directly by
            the tabular agent but kept for API parity with other agents).
        n_actions: Number of discrete actions available.
        alpha: Learning rate (step size) for Q updates.
        gamma: Discount factor for bootstrapping future rewards.
        seed: Seed for the agent's private RNG (used for epsilon-greedy draws).

    Attributes:
        obs_dim (int): observation dimensionality (int cast).
        n_actions (int): number of discrete actions (int cast).
        alpha (float): learning rate used in updates.
        gamma (float): discount factor used to compute TD target.
        rng (random.Random): RNG used for epsilon draws and exploration.
        table (QTable): underlying tabular action-value store.
    """

    def __init__(self, obs_dim: int, n_actions: int, alpha: float = 0.1, gamma: float = 0.97, seed: int = 0):
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.rng = random.Random(seed)

        # Underlying dictionary-backed Q-table storing arrays of length n_actions
        self.table = QTable(self.n_actions)

    def act(self, state: np.ndarray, eps: float) -> int:
        """Select an action using epsilon-greedy policy.

        With probability `eps` select a random action (exploration). Otherwise
        select the greedy action (argmax) from the tabular Q-values.

        Args:
            state: 1D numpy array representing the current observation.
            eps: exploration probability in [0, 1].

        Returns:
            int: The chosen action index in [0, n_actions-1].
        """
        # Exploration: choose uniformly at random with prob eps
        if self.rng.random() < float(eps):
            return self.rng.randrange(self.n_actions)

        # Exploitation: choose greedy action from Q-table
        return self.table.argmax(state)

    def train_step(self, batch: Iterable, return_loss: bool = True) -> float:
        """Perform Q-Learning updates for transitions in `batch`.

        The batch is an iterable of transitions (s, a, r, s2, done). For each
        transition we:
          1. lookup (and create if needed) the Q-values for state s (by ref)
          2. compute the TD target: r + gamma * max_a' Q(s2, a')  (0 if terminal)
          3. compute TD error = target - Q(s,a)
          4. apply the incremental update: Q(s,a) <- Q(s,a) + alpha * TD_error

        The method returns the mean absolute TD error across the batch (a single
        float). This mirrors many RL training loops which treat this as a loss
        signal for logging/diagnostics.

        Args:
            batch: iterable of tuples (s, a, r, s2, done).
                   `s` and `s2` are expected to be array-like (converted to 1D np arrays).
                   `done` can be boolean-like (True when episode terminated).
            return_loss: retained for API parity; currently always returns the
                         computed mean absolute TD error.

        Returns:
            float: mean absolute TD error across the processed batch. If the
                   batch is empty returns 0.0.
        """
        td_errors = []
        cnt = 0

        for (s, a, r, s2, done) in batch:
            # Ensure states are 1D numpy arrays (float32) used by QTable encoding
            s = np.asarray(s, dtype=np.float32).flatten()
            s2 = np.asarray(s2, dtype=np.float32).flatten()

            # q_s is a direct reference into the table (so we can update in-place).
            # q_sa = current value for the taken action (cast to float for arithmetic)
            q_s = self.table.get_ref(s)
            q_sa = float(q_s[int(a)])

            # For the bootstrap term, we need the values for next state s2.
            # Use .get() which returns a copy to avoid accidentally creating a
            # reference to s2 row if we only want to read its max.
            q_s2 = self.table.get(s2)

            # If the next-state has no stored actions (shouldn't happen because
            # QTable ensures existence), treat max as 0.0 to be conservative.
            max_q_s2 = float(np.max(q_s2)) if q_s2.size > 0 else 0.0

            # Compute TD target. If done==True then no bootstrap term is used.
            target = float(r) + self.gamma * max_q_s2 * (0.0 if float(done) else 1.0)
            td = target - q_sa

            # Standard incremental Q-Learning update (in-place on q_s reference).
            # This mutates the underlying QTable row for `s`.
            q_s[int(a)] = q_sa + self.alpha * td

            td_errors.append(abs(td))
            cnt += 1

        # Return mean absolute TD error (or 0.0 if no transitions were processed).
        if cnt == 0:
            return 0.0
        return float(sum(td_errors) / cnt)
