"""Simple replay buffer used to keep API parity with the DQN project."""

from __future__ import annotations
import random
from collections import deque
from typing import Iterable, List, Tuple, Any

class ReplayBuffer:
    """Very small wrapper providing append() and sample() like in DQN code."""

    def __init__(self, capacity: int = 10000):
        self.capacity = int(capacity)
        self._buf = deque(maxlen=self.capacity)

    def append(self, transition: Tuple[Any, ...]) -> None:
        self._buf.append(transition)

    def __len__(self) -> int:
        return len(self._buf)

    def sample(self, batch_size: int) -> List[Tuple[Any, ...]]:
        if batch_size <= 0:
            return []
        if batch_size >= len(self._buf):
            # return all
            return list(self._buf)
        # uniform random sample
        return random.sample(list(self._buf), batch_size)
