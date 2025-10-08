"""Simple configuration container mirroring the DQN project's CFG shape.

This module defines lightweight dataclasses for configuring the GridWorld + Q-Learning
demo. The structure deliberately mirrors a small subset of the DQN project's config
shape so command-line defaults and code that expects `CFG.<group>.<field>` will work
with minimal changes.

Design notes
------------
- Dataclasses are used for clarity and easy instantiation; they serve as a typed
  configuration object rather than a full-featured configuration system.
- `ProjectConfig` exposes a convenience property `max_size_canvas` which returns
  either the explicitly configured canvas size or falls back to `env.max_size`.
- Fields intentionally use small sensible defaults suitable for quick demo runs.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnvConfig:
    """Environment / layout sampling configuration.

    Attributes:
        min_size: minimum grid side length (inclusive).
        max_size: maximum grid side length (inclusive).
        episodes_per_layout: number of episodes to run before resampling a new layout.
        pit_frac: approximate fraction of cells in a layout that will be pits.
        max_size_canvas: optional fixed canvas size used by observation encoder;
                         if None the encoder will default to `max_size`.
        max_steps_multiplier: multiplier applied to size*size to derive the
                              environment's max_steps default (int(max_steps_multiplier * size*size)).
    """
    min_size: int = 5
    max_size: int = 5
    episodes_per_layout: int = 999_999_999
    pit_frac: float = 0.12
    max_size_canvas: Optional[int] = None  # defaults to env.max_size when None
    max_steps_multiplier: float = 1.0  # multiplier for max_steps (default: size*size)


@dataclass
class AgentConfig:
    """Agent-specific hyperparameters.

    Attributes:
        seed: RNG seed for reproducibility across agent components.
        alpha: learning rate (step size) for Q-Learning updates.
        gamma: discount factor used when computing TD targets.
    """
    seed: int = 0
    alpha: float = 0.1  # learning rate
    gamma: float = 0.97  # discount factor


@dataclass
class EpsilonConfig:
    """Epsilon (exploration) schedule endpoints.

    Attributes:
        eps_start: starting epsilon (usually 1.0 for full exploration).
        eps_end: final epsilon value at the end of training.
    """
    eps_start: float = 1.0
    eps_end: float = 0.05


@dataclass
class RenderConfig:
    """Rendering / GUI configuration.

    Attributes:
        enable_gui: whether GUI rendering is allowed by default (CLI can override).
        render_every: render every N episodes (1 = render every episode).
        sleep: delay (seconds) between rendered frames (can be 0 for fast draws).
        cell_size_small: pixel size per cell for "small" grids (used to make UI larger).
        cell_size_large: pixel size per cell for "large" grids (smaller cell size).
        render_max_size_for_large_cells: threshold used to choose small vs large cell sizes.
                                         If env.size <= this threshold, `cell_size_small` is used.
    """
    enable_gui: bool = True
    render_every: int = 1
    sleep: float = 0.02
    cell_size_small: int = 64
    cell_size_large: int = 40
    render_max_size_for_large_cells: int = 8  # if env.size <= this -> use cell_size_small


@dataclass
class TrainConfig:
    """Training-run configuration.

    Attributes:
        episodes: total number of episodes to train for.
        save_every: placeholder for checkpoint frequency (0 = disabled).
        log_every: how often (in episodes) to flush logs and print summaries.
    """
    episodes: int = 600
    save_every: int = 0  # placeholder for future checkpointing
    log_every: int = 20


@dataclass
class ProjectConfig:
    """Top-level project configuration aggregating all config groups.

    Fields:
        env: environment/layout related settings.
        agent: agent hyperparameters.
        eps: epsilon schedule endpoints.
        render: rendering/UI settings.
        train: training loop settings.

    Properties:
        max_size_canvas: effective canvas size used by the encoder. If the user
                         doesn't explicitly set `env.max_size_canvas`, the property
                         falls back to `env.max_size`.
    """
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    eps: EpsilonConfig = field(default_factory=EpsilonConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def max_size_canvas(self) -> int:
        """Effective max_size_canvas used by the observation encoder.

        Returns:
            int: `env.max_size_canvas` if set; otherwise returns `env.max_size`.

        This convenience property centralizes the fallback logic so callers don't
        need to repeatedly check for None.
        """
        return self.env.max_size if self.env.max_size_canvas is None else self.env.max_size_canvas


# Global config instance used across the codebase (imported as `from config import CFG`).
CFG = ProjectConfig()
