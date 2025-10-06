"""GridWorld environment and StepResult dataclass.

Deterministic movement with pits and a single goal. Rewards:
- step: -0.01
- pit: -15.0
- goal: +15.0
Episode ends at pit/goal or when max_steps reached.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class StepResult:
    state: np.ndarray
    reward: float
    done: bool
    info: dict


class GridWorld:
    """Deterministic grid with holes (pits) and a single goal.

    Actions: 0=up, 1=right, 2=down, 3=left.
    """

    def __init__(
        self,
        size: int,
        pits: List[Tuple[int, int]] | None,
        goal: Tuple[int, int],
        start: Tuple[int, int],
        max_steps: int | None = None,
    ):
        self.size = int(size)
        self.pits = set(pits or [])
        self.goal = goal
        self.start = start
        self.pos = start
        self.t = 0
        self.max_steps = max_steps or (self.size * self.size)

        self.n_actions = 4  # up/right/down/left

    def reset(self) -> None:
        self.pos = self.start
        self.t = 0

    def step(self, action: int) -> StepResult:
        x, y = self.pos
        if action == 0:       # up
            y = max(0, y - 1)
        elif action == 1:     # right
            x = min(self.size - 1, x + 1)
        elif action == 2:     # down
            y = min(self.size - 1, y + 1)
        elif action == 3:     # left
            x = max(0, x - 1)
        else:
            raise ValueError("invalid action")

        self.pos = (x, y)
        self.t += 1

        reward = -0.01
        done = False
        info = {}

        if self.pos in self.pits:
            reward = -15.0
            done = True
            info["terminal"] = "pit"
        elif self.pos == self.goal:
            reward = +15.0
            done = True
            info["terminal"] = "goal"
        elif self.t >= self.max_steps:
            done = True
            info["terminal"] = "timeout"

        # state returned here is a placeholder; the external encoder produces the fixed-width observation
        return StepResult(state=np.array([], dtype=np.float32), reward=reward, done=done, info=info)
