"""Small replay buffer wrapper around collections.deque.

Provides the minimal API used in training:
- append(transition)
- sample(batch_size) -> list
- __len__()
"""
from __future__ import annotations
import random
from collections import deque
from typing import Deque, Iterable, List, Tuple, Any


Transition = Tuple[Any, ...]  # (s, a, r, s2, done)


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self._buf: Deque[Transition] = deque(maxlen=int(capacity))

    def append(self, transition: Transition) -> None:
        """Append a single transition."""
        self._buf.append(transition)

    def extend(self, transitions: Iterable[Transition]) -> None:
        """Append multiple transitions."""
        for t in transitions:
            self._buf.append(t)

    def sample(self, batch_size: int) -> List[Transition]:
        """Uniform random sample without replacement (if possible)."""
        n = len(self._buf)
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if n == 0:
            return []
        # if requested batch larger than current buffer, sample with replacement
        if batch_size > n:
            return [random.choice(list(self._buf)) for _ in range(batch_size)]
        idxs = random.sample(range(n), batch_size)
        buf_list = list(self._buf)
        return [buf_list[i] for i in idxs]

    def __len__(self) -> int:
        return len(self._buf)
