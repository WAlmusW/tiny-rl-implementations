"""Simple tabular Q-table helper used by QLearning agent."""

from __future__ import annotations
from typing import MutableMapping, Tuple
import numpy as np

StateKey = Tuple[float, ...]


def _state_to_key(state: np.ndarray, decimals: int = 4) -> StateKey:
    """Convert a 1D numpy observation to a hashable tuple key (rounded)."""
    if state.ndim != 1:
        state = state.flatten()
    return tuple(np.round(state.astype(np.float64), decimals=decimals).tolist())


class QTable:
    """Tabular action-value store (dict-backed)."""

    def __init__(self, n_actions: int):
        self.n_actions = int(n_actions)
        self._table: MutableMapping[StateKey, np.ndarray] = {}

    def _ensure_state(self, key: StateKey) -> np.ndarray:
        if key not in self._table:
            self._table[key] = np.zeros((self.n_actions,), dtype=np.float32)
        return self._table[key]

    def get(self, state: np.ndarray) -> np.ndarray:
        k = _state_to_key(state)
        return self._ensure_state(k).copy()

    def get_ref(self, state: np.ndarray) -> np.ndarray:
        k = _state_to_key(state)
        return self._ensure_state(k)

    def argmax(self, state: np.ndarray) -> int:
        q = self.get(state)
        return int(np.argmax(q))
