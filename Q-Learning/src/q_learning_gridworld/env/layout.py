"""Layout sampling and fixed-width observation encoder.

This module provides:
- `Layout` dataclass: simple container describing a sampled grid layout.
- `LayoutSampler`: randomly samples variable-size layouts with start, goal, and pits.
- `encode_obs_fixed(env, max_size_canvas)`: encodes a GridWorld environment into a
  fixed-width 1D observation vector compatible with tabular agents.

Encoding format (returned shape = 4 + max_size_canvas*max_size_canvas):
  - head (4 values): [ax, ay, gx, gy] normalized to [0,1] using env.size
      ax, ay: agent position (x, y)
      gx, gy: goal position (x, y)
  - tail: flattened pit mask rasterized into a square canvas of side `max_size_canvas`.
    The pit mask uses nearest-cell scaling from the environment's coordinate grid.

Design notes
------------
- The encoder intentionally normalizes coordinates to be robust to different grid sizes.
- Pit rasterization uses nearest-cell mapping to the fixed canvas; this is a cheap,
  deterministic approach that preserves cell locations well for small grids.
- `LayoutSampler` provides reproducible randomness via a provided seed.
"""

from __future__ import annotations
from dataclasses import dataclass
import random
from typing import List, Tuple

import numpy as np


@dataclass
class Layout:
    """Container describing a sampled grid layout.

    Attributes:
        size: side length of the square grid.
        start: (x, y) starting coordinate for the agent.
        goal: (x, y) goal coordinate.
        pits: list of (x, y) coordinates marking pit cells.
    """
    size: int
    start: tuple[int, int]
    goal: tuple[int, int]
    pits: list[tuple[int, int]]


class LayoutSampler:
    """Random layout sampler producing variable-size grid configurations.

    Args:
        min_size: minimum grid side length (inclusive).
        max_size: maximum grid side length (inclusive).
        pit_frac: approximate fraction of non-start/goal cells to mark as pits.
        seed: random seed for reproducibility.

    Behavior:
        - Chooses a grid size uniformly between min_size and max_size.
        - Picks random start and goal cells (ensures they differ).
        - Selects ~pit_frac * (size*size) pit cells excluding start/goal.
    """

    def __init__(self, min_size: int = 4, max_size: int = 8, pit_frac: float = 0.12, seed: int = 0):
        assert 1 <= min_size <= max_size
        self.min_size = int(min_size)
        self.max_size = int(max_size)
        self.pit_frac = float(pit_frac)
        self.rng = random.Random(seed)

    def sample(self) -> Layout:
        """Sample and return a new Layout instance.

        Returns:
            Layout: sampled grid description.
        """
        s = self.rng.randint(self.min_size, self.max_size)

        # Random start & goal coordinates; ensure they are not the same.
        start = (self.rng.randrange(s), self.rng.randrange(s))
        goal = (self.rng.randrange(s), self.rng.randrange(s))
        while goal == start:
            goal = (self.rng.randrange(s), self.rng.randrange(s))

        # Determine number of pits (~pit_frac fraction), at least one.
        n_cells = s * s
        n_pits = max(1, int(self.pit_frac * n_cells))

        # Candidate cells exclude start and goal.
        candidates = [(x, y) for x in range(s) for y in range(s) if (x, y) not in (start, goal)]
        self.rng.shuffle(candidates)
        pits = candidates[:n_pits]

        return Layout(size=s, start=start, goal=goal, pits=pits)


def encode_obs_fixed(env, max_size_canvas: int) -> np.ndarray:
    """Encode the GridWorld environment into a fixed-width 1D observation.

    The returned observation is a float32 numpy array concatenating:
      - head: 4 normalized coordinates [ax, ay, gx, gy] in [0,1]
      - tail: flattened pit mask of shape (max_size_canvas, max_size_canvas)

    Args:
        env: GridWorld instance providing `.pos`, `.goal`, `.pits`, and `.size`.
        max_size_canvas: target canvas side length for pit mask rasterization.

    Returns:
        numpy.ndarray: 1D float32 array of length 4 + max_size_canvas**2.
    """
    ax, ay = env.pos
    gx, gy = env.goal
    s = env.size

    def norm(v):
        # Normalize coordinate to [0,1] using (s-1) so both endpoints map to 0 and 1.
        # For s==1 avoid division by zero and return 0.0.
        return 0.0 if s <= 1 else v / float(s - 1)

    # Head: normalized agent and goal coordinates.
    head = np.array([norm(ax), norm(ay), norm(gx), norm(gy)], dtype=np.float32)

    # Tail: rasterized pit mask on a fixed canvas.
    pit_mask = np.zeros((max_size_canvas, max_size_canvas), dtype=np.float32)
    if s > 1:
        # Map each pit cell to the nearest canvas cell using linear scaling.
        for (px, py) in env.pits:
            qx = int(round(px * (max_size_canvas - 1) / (s - 1)))
            qy = int(round(py * (max_size_canvas - 1) / (s - 1)))
            # Guard bounds; rounding may produce out-of-range indices at extremes.
            qx = max(0, min(max_size_canvas - 1, qx))
            qy = max(0, min(max_size_canvas - 1, qy))
            pit_mask[qy, qx] = 1.0

    # Flatten pit mask row-major and concatenate with head.
    return np.concatenate([head, pit_mask.flatten()], dtype=np.float32)
