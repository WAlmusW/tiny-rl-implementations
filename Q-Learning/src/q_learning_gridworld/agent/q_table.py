"""Simple tabular Q-table helper used by the QLearning agent.

This module implements a tiny dictionary-backed action-value store optimized
for readability and parity with a small tabular RL demo. Observations are
converted to hashable tuple keys by `_state_to_key` which rounds floating
point values — this provides a simple discretization so continuous or
normalized observations can be used as table keys.

Design notes
------------
- The table stores, for each state key, a numpy array of length `n_actions`
  containing the action-values (dtype float32).
- `get_ref(state)` returns a direct reference to the internal numpy array so
  callers can update values in-place (this is used by QLearning.update).
- `get(state)` returns a copy to prevent accidental external mutation when
  only a read is intended.
- Rounding decimals default to 4 which is a pragmatic compromise for
  stability when observations are normalized floats (adjust if needed).
"""

from __future__ import annotations
from typing import MutableMapping, Tuple
import numpy as np

StateKey = Tuple[float, ...]


def _state_to_key(state: np.ndarray, decimals: int = 4) -> StateKey:
    """Convert a 1D numpy observation to a hashable tuple key.

    This function:
      - flattens non-1D arrays (defensive),
      - casts to float64 for stable rounding,
      - rounds to `decimals` places,
      - converts to a tuple to serve as a dict key.

    Rounding acts as a simple discretizer so that similar floating-point
    observations map to the same table row. Tune `decimals` if you need
    coarser/finer discretization.

    Args:
        state: Input observation (numpy array-like). Can be any shape,
               but is typically 1D for this project.
        decimals: Number of decimal places to round to (defaults to 4).

    Returns:
        StateKey: A tuple of rounded floats suitable for use as a dict key.
    """
    if state.ndim != 1:
        state = state.flatten()
    # Use float64 for rounding to minimize floating point surprises
    rounded = np.round(state.astype(np.float64), decimals=decimals)
    return tuple(rounded.tolist())


class QTable:
    """Tabular action-value store (dictionary-backed).

    The table maps `StateKey` -> numpy.ndarray shaped (n_actions,), dtype float32.

    Args:
        n_actions: Number of discrete actions. Each stored row is an array of
                   zeros of length `n_actions` when first created.

    Example:
        >>> qt = QTable(4)
        >>> s = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        >>> row = qt.get_ref(s)   # direct reference, can be updated in-place
        >>> row[1] = 2.0
        >>> qt.get(s)             # returns a copy containing the updated value
    """

    def __init__(self, n_actions: int):
        self.n_actions = int(n_actions)
        # Internal mapping from rounded state tuples to action-value arrays
        self._table: MutableMapping[StateKey, np.ndarray] = {}

    def _ensure_state(self, key: StateKey) -> np.ndarray:
        """Ensure a row exists for `key`, creating a zero-initialized array if needed.

        Returns the internal array (not a copy). Callers who receive this
        array may mutate it to update Q-values.
        """
        if key not in self._table:
            # initialize to zeros for each action
            self._table[key] = np.zeros((self.n_actions,), dtype=np.float32)
        return self._table[key]

    def get(self, state: np.ndarray) -> np.ndarray:
        """Return a copy of the Q-values for `state`.

        This is the safe read-only accessor — modifications to the returned
        array will not affect the table.

        Args:
            state: observation array to query.

        Returns:
            numpy.ndarray: 1D array of length `n_actions` (dtype float32).
        """
        k = _state_to_key(state)
        return self._ensure_state(k).copy()

    def get_ref(self, state: np.ndarray) -> np.ndarray:
        """Return a direct reference to the Q-values for `state`.

        Use this when you intend to update the table in-place (e.g., Q-Learning
        incremental updates). The returned array is the real storage and
        mutating it will mutate the table.

        Args:
            state: observation array to query.

        Returns:
            numpy.ndarray: internal 1D array of length `n_actions`.
        """
        k = _state_to_key(state)
        return self._ensure_state(k)

    def argmax(self, state: np.ndarray) -> int:
        """Return the index of the greedy action for `state`.

        If multiple actions are tied for maximum, `np.argmax` breaks ties by
        returning the first occurrence. For deterministic behavior the rounding
        function and dtype control how states map to keys.

        Args:
            state: observation array to query.

        Returns:
            int: index of the action with highest estimated value.
        """
        q = self.get(state)
        return int(np.argmax(q))
