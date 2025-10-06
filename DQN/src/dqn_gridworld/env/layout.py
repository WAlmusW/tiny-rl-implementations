"""Layout sampling and fixed-width observation encoder.

Provides:
- Layout dataclass
- LayoutSampler (random layouts with variable size, start, goal, pits)
- encode_obs_fixed(env, max_size_canvas) -> 1D numpy array observation
"""
from __future__ import annotations
from dataclasses import dataclass
import random
from typing import List, Tuple

import numpy as np

@dataclass
class Layout:
    size: int
    start: tuple[int, int]
    goal: tuple[int, int]
    pits: list[tuple[int, int]]


class LayoutSampler:
    """Random layout sampler producing variable-size grids."""
    def __init__(self, min_size: int = 4, max_size: int = 8, pit_frac: float = 0.12, seed: int = 0):
        assert 1 <= min_size <= max_size
        self.min_size = int(min_size)
        self.max_size = int(max_size)
        self.pit_frac = float(pit_frac)
        self.rng = random.Random(seed)

    def sample(self) -> Layout:
        s = self.rng.randint(self.min_size, self.max_size)

        # random start & goal, not overlapping
        start = (self.rng.randrange(s), self.rng.randrange(s))
        goal = (self.rng.randrange(s), self.rng.randrange(s))
        while goal == start:
            goal = (self.rng.randrange(s), self.rng.randrange(s))

        # pits: ~pit_frac of cells, excluding start/goal
        n_cells = s * s
        n_pits = max(1, int(self.pit_frac * n_cells))
        candidates = [(x, y) for x in range(s) for y in range(s) if (x, y) not in (start, goal)]
        self.rng.shuffle(candidates)
        pits = candidates[:n_pits]

        return Layout(size=s, start=start, goal=goal, pits=pits)


def encode_obs_fixed(env, max_size_canvas: int) -> np.ndarray:
    """
    Fixed-width observation:
      - head: [ax, ay, gx, gy] normalized to [0,1] using env.size
      - tail: flattened pit mask rasterized to max_size_canvas x max_size_canvas

    Returned shape: (4 + max_size_canvas*max_size_canvas,)
    """
    ax, ay = env.pos
    gx, gy = env.goal
    s = env.size

    def norm(v):
        return 0.0 if s <= 1 else v / float(s - 1)

    head = np.array([norm(ax), norm(ay), norm(gx), norm(gy)], dtype=np.float32)

    pit_mask = np.zeros((max_size_canvas, max_size_canvas), dtype=np.float32)
    if s > 1:
        # Nearest-cell scaling to canvas
        for (px, py) in env.pits:
            qx = int(round(px * (max_size_canvas - 1) / (s - 1)))
            qy = int(round(py * (max_size_canvas - 1) / (s - 1)))
            # guard bounds
            qx = max(0, min(max_size_canvas - 1, qx))
            qy = max(0, min(max_size_canvas - 1, qy))
            pit_mask[qy, qx] = 1.0

    return np.concatenate([head, pit_mask.flatten()], dtype=np.float32)
