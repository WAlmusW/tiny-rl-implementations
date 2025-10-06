"""DQN agent using NumPy MLPQ networks (online + target)."""
from __future__ import annotations
import random
from typing import Iterable

import numpy as np

from .model import MLPQ


class DQN:
    """Wrapper around MLPQ + target network with simple SGD training."""

    def __init__(self, obs_dim: int, n_actions: int,
                 hidden: int = 64, lr: float = 1e-3, gamma: float = 0.97,
                 seed: int = 0):
        self.q = MLPQ(obs_dim, hidden, n_actions, seed=seed)
        self.target = MLPQ(obs_dim, hidden, n_actions, seed=seed + 1)
        self.target.copy_from(self.q)

        self.gamma = float(gamma)
        self.lr = float(lr)
        self.n_actions = int(n_actions)

    def act(self, state: np.ndarray, eps: float) -> int:
        """Epsilon-greedy action selection. `state` is a 1D array (obs_dim,)."""
        if random.random() < float(eps):
            return random.randrange(self.n_actions)
        qvals = self.q.predict(state[None, :])[0]
        return int(np.argmax(qvals))

    def train_step(self, batch: Iterable, clip_grad: float | None = 1.0) -> float:
        """
        Train on a batch of transitions.

        batch: iterable of (s, a, r, s2, done)
        Returns: MSE loss (float)
        """
        states = np.stack([b[0] for b in batch]).astype(np.float32)   # (B, D)
        actions = np.array([b[1] for b in batch], dtype=np.int64)     # (B,)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)   # (B,)
        next_states = np.stack([b[3] for b in batch]).astype(np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.float32)

        # Forward pass
        Z1, A1, Q = self.q.forward(states)

        # Compute target values using target network
        with np.errstate(over='ignore'):
            Q_next = self.target.predict(next_states)
        max_next = np.max(Q_next, axis=1)
        y = rewards + self.gamma * max_next * (1.0 - dones)

        # Compute gradient of 0.5*(Q_sa - y)^2 w.r.t Q
        B = Q.shape[0]
        grad_Q = np.zeros_like(Q, dtype=np.float32)
        idx = (np.arange(B), actions)
        diff = (Q[idx] - y).astype(np.float32)  # (B,)
        grad_Q[idx] = diff  # dL/dQ

        # Backprop to compute parameter gradients
        dW2 = A1.T @ grad_Q            # (H, A)
        db2 = np.sum(grad_Q, axis=0)   # (A,)

        dA1 = grad_Q @ self.q.W2.T     # (B, H)
        dZ1 = dA1 * (Z1 > 0).astype(np.float32)  # ReLU derivative
        dW1 = states.T @ dZ1           # (D, H)
        db1 = np.sum(dZ1, axis=0)      # (H,)

        # Optional gradient clipping by global norm
        if clip_grad is not None:
            def clip(arr):
                norm = np.linalg.norm(arr)
                if norm > clip_grad:
                    arr *= (clip_grad / (norm + 1e-8))
                return arr
            dW1, db1 = clip(dW1), clip(db1)
            dW2, db2 = clip(dW2), clip(db2)

        # SGD update (mean over batch)
        bsz = float(B)
        self.q.W1 -= self.lr * (dW1 / bsz)
        self.q.b1 -= self.lr * (db1 / bsz)
        self.q.W2 -= self.lr * (dW2 / bsz)
        self.q.b2 -= self.lr * (db2 / bsz)

        loss = float(np.mean(0.5 * (diff ** 2)))
        return loss
