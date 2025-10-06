"""Minimal fully-connected Q-network (NumPy)."""
from __future__ import annotations
import math
import numpy as np


class MLPQ:
    """Simple 2-layer MLP with ReLU and linear outputs (Q-values)."""

    def __init__(self, input_dim: int, hidden: int, n_actions: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        # He initialization for ReLU hidden layer
        self.W1 = rng.normal(0, math.sqrt(2.0 / max(1, input_dim)), size=(input_dim, hidden)).astype(np.float32)
        self.b1 = np.zeros((hidden,), dtype=np.float32)
        self.W2 = rng.normal(0, math.sqrt(2.0 / max(1, hidden)), size=(hidden, n_actions)).astype(np.float32)
        self.b2 = np.zeros((n_actions,), dtype=np.float32)

    def forward(self, X: np.ndarray):
        """
        Forward pass.

        Args:
            X: (B, input_dim)
        Returns:
            Z1: pre-activation hidden (B, H)
            A1: post-ReLU hidden (B, H)
            Q: (B, n_actions)
        """
        Z1 = X @ self.W1 + self.b1  # (B, H)
        A1 = np.maximum(Z1, 0.0)    # ReLU
        Q = A1 @ self.W2 + self.b2  # (B, A)
        return Z1, A1, Q

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return Q-values for X (B, input_dim) -> (B, n_actions)."""
        return self.forward(X)[-1]

    def copy_from(self, other: "MLPQ"):
        """Copy parameters from another network (in-place)."""
        self.W1[...] = other.W1
        self.b1[...] = other.b1
        self.W2[...] = other.W2
        self.b2[...] = other.b2
