"""Simple replay buffer used to keep API parity with the DQN project.

This module implements a minimal FIFO replay buffer backed by collections.deque.
It intentionally mirrors a tiny subset of the DQN project's buffer API:
  - append(transition)
  - sample(batch_size) -> list of transitions
  - __len__()

Design goals
------------
- Extremely small and dependency-light (only Python stdlib).
- Deterministic capacity via deque(maxlen=capacity) so old transitions are
  automatically dropped when capacity is reached.
- `sample` returns a uniform random sample without replacement when batch_size
  < buffer length; when batch_size >= current size it returns all entries.
- Transitions are stored as opaque tuples (e.g., (s, a, r, s2, done)) to keep
  the buffer generic.
"""

from __future__ import annotations
import random
from collections import deque
from typing import Iterable, List, Tuple, Any


class ReplayBuffer:
    """Very small wrapper providing append() and sample() semantics.

    Args:
        capacity: maximum number of transitions to keep. When full, older
                  transitions are discarded (FIFO).

    Methods:
        append(transition): add a transition tuple to the buffer.
        sample(batch_size): uniformly random sample of transitions.
        __len__(): current number of stored transitions.
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = int(capacity)
        # deque with maxlen automatically discards oldest entries when full
        self._buf = deque(maxlen=self.capacity)

    def append(self, transition: Tuple[Any, ...]) -> None:
        """Append a new transition to the buffer.

        The transition can be any tuple-like structure (e.g., (s, a, r, s2, done)).
        """
        self._buf.append(transition)

    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return len(self._buf)

    def sample(self, batch_size: int) -> List[Tuple[Any, ...]]:
        """Return a list of transitions sampled uniformly at random.

        Behavior:
          - If batch_size <= 0: return an empty list (no sampling).
          - If batch_size >= buffer length: return a shallow list copy of all entries.
          - Otherwise: return `batch_size` unique transitions sampled without replacement.

        Args:
            batch_size: desired number of sampled transitions.

        Returns:
            List of transition tuples.
        """
        if batch_size <= 0:
            return []
        if batch_size >= len(self._buf):
            # return all transitions in insertion order
            return list(self._buf)
        # random.sample performs sampling without replacement
        return random.sample(list(self._buf), batch_size)
