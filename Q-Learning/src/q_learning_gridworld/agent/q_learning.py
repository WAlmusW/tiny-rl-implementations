"""Tabular Q-Learning agent (epsilon-greedy) mirroring DQN API shape."""

from __future__ import annotations
import random
from typing import Iterable
import numpy as np

from .q_table import QTable


class QLearning:
    """Tabular Q-Learning agent with API similar to dqn.DQN."""

    def __init__(self, obs_dim: int, n_actions: int, alpha: float = 0.1, gamma: float = 0.97, seed: int = 0):
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.rng = random.Random(seed)
        self.table = QTable(self.n_actions)

    def act(self, state: np.ndarray, eps: float) -> int:
        """Epsilon-greedy action selection. `state` is 1D numpy array."""
        if self.rng.random() < float(eps):
            return self.rng.randrange(self.n_actions)
        return self.table.argmax(state)

    def train_step(self, batch: Iterable, return_loss: bool = True) -> float:
        """Perform Q-Learning updates for transitions in batch.

        batch: iterable of (s, a, r, s2, done)
        Returns: mean absolute TD error across batch (float).
        """
        td_errors = []
        cnt = 0

        for (s, a, r, s2, done) in batch:
            s = np.asarray(s, dtype=np.float32).flatten()
            s2 = np.asarray(s2, dtype=np.float32).flatten()

            q_s = self.table.get_ref(s)
            q_sa = float(q_s[int(a)])

            q_s2 = self.table.get(s2)
            max_q_s2 = float(np.max(q_s2)) if q_s2.size > 0 else 0.0

            target = float(r) + self.gamma * max_q_s2 * (0.0 if float(done) else 1.0)
            td = target - q_sa

            # update rule
            q_s[int(a)] = q_sa + self.alpha * td

            td_errors.append(abs(td))
            cnt += 1

        if cnt == 0:
            return 0.0
        return float(sum(td_errors) / cnt)
