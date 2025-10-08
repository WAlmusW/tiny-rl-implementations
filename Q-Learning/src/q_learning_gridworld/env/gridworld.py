"""GridWorld environment and StepResult dataclass.

A small deterministic grid environment used for tabular Q-Learning demos.

Environment dynamics
--------------------
- The grid is an `size x size` square with coordinates `(x, y)` where
  `0 <= x < size` and `0 <= y < size`. Coordinates follow (x=col, y=row).
- There are special cells:
    - `pits`: set of coordinates that terminate the episode with a large negative reward.
    - `goal`: coordinate that terminates the episode with a large positive reward.
    - `start`: initial agent position after `reset()`.
- Actions (discrete):
    - 0 = up    -> decrement y (clamped at 0)
    - 1 = right -> increment x (clamped at size-1)
    - 2 = down  -> increment y (clamped at size-1)
    - 3 = left  -> decrement x (clamped at 0)
- Each step yields a small negative step penalty (-0.01) to encourage faster solutions.
- If the agent enters a pit: reward = -15.0 and `done=True`.
- If the agent reaches the goal: reward = +15.0 and `done=True`.
- If the agent reaches `max_steps`: done=True with terminal="timeout".

Notes
-----
- This module intentionally keeps observations minimal (the StepResult.state is an
  empty numpy array). The training code uses a separate observation encoder
  (`encode_obs_fixed` in layout.py) to produce fixed-width observations for the agent.
- The environment is deterministic and has no stochastic transitions.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class StepResult:
    """Result returned by `GridWorld.step()`.

    Attributes:
        state (np.ndarray): placeholder for observation (empty here; encoder used externally).
        reward (float): scalar reward for the transition.
        done (bool): whether the episode terminated after this step.
        info (dict): optional extra information (e.g., terminal reason).
    """
    state: np.ndarray
    reward: float
    done: bool
    info: dict


class GridWorld:
    """Deterministic grid with pits and a single goal.

    The environment provides a tiny, focused interface compatible with the
    training loop in this project:

        env = GridWorld(size, pits, goal, start, max_steps=...)
        env.reset()
        res = env.step(action)          # returns StepResult
        pos = env.pos                   # current agent position (x, y)

    Args:
        size: Grid dimension (size x size).
        pits: Iterable of (x, y) tuples marking pit locations (episode ends if entered).
        goal: (x, y) coordinate of goal cell (episode ends if reached).
        start: (x, y) starting coordinate for the agent.
        max_steps: Optional maximum steps before timeout; defaults to size*size.

    Attributes:
        size (int): grid side length.
        pits (set): set of pit coordinate tuples for quick membership tests.
        goal (tuple): goal coordinate.
        start (tuple): starting coordinate.
        pos (tuple): current agent coordinate (mutable).
        t (int): current time step count since last reset.
        max_steps (int): allowed steps before forced timeout.
        n_actions (int): number of discrete actions (4).
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
        # store pits as a set for O(1) membership checks
        self.pits = set(pits or [])
        self.goal = goal
        self.start = start
        # current position and timestep
        self.pos = start
        self.t = 0
        # default max_steps scales with grid area if not provided
        self.max_steps = max_steps or (self.size * self.size)

        # actions: 0=up, 1=right, 2=down, 3=left
        self.n_actions = 4

    def reset(self) -> None:
        """Reset the environment to the starting state.

        After reset, `pos` is set to `start` and timestep `t` is zeroed.
        """
        self.pos = self.start
        self.t = 0

    def step(self, action: int) -> StepResult:
        """Apply `action`, update position and timestep, and return result.

        Args:
            action: integer in {0,1,2,3} mapping to up/right/down/left.

        Returns:
            StepResult: contains (state, reward, done, info). `state` is an
                        empty numpy array in this environment; a separate
                        encoder produces the actual observation used by agents.
        Raises:
            ValueError: if `action` is not a valid integer action.
        """
        x, y = self.pos

        # Compute candidate next position with clamping to grid bounds.
        # Note: grid coordinates use (x, y) with x across columns and y across rows.
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

        # Commit move and increment time step
        self.pos = (x, y)
        self.t += 1

        # Default small step penalty to encourage shorter solutions.
        reward = -0.01
        done = False
        info = {}

        # Terminal checks (pit, goal, then timeout)
        if self.pos in self.pits:
            # Fell in a pit: large negative reward and terminal.
            reward = -15.0
            done = True
            info["terminal"] = "pit"
        elif self.pos == self.goal:
            # Reached goal: large positive reward and terminal.
            reward = +15.0
            done = True
            info["terminal"] = "goal"
        elif self.t >= self.max_steps:
            # Reached max steps: terminal due to timeout (no additional reward change).
            done = True
            info["terminal"] = "timeout"

        # The environment itself does not produce a rich observation array here.
        # A separate encoder (layout.encode_obs_fixed) builds the observation used by the agent.
        return StepResult(state=np.array([], dtype=np.float32), reward=reward, done=done, info=info)
