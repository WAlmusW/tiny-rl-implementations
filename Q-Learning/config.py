"""Simple config object (mirror of DQN project's CFG shape)."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class EnvConfig:
    min_size: int = 5
    max_size: int = 5
    episodes_per_layout: int = 999_999_999
    pit_frac: float = 0.12
    max_size_canvas: Optional[int] = None  # defaults to max_size when None
    max_steps_multiplier: float = 1.0  # multiplier for max_steps (default: size*size)

@dataclass
class AgentConfig:
    seed: int = 0
    alpha: float = 0.1  # learning rate
    gamma: float = 0.97  # discount factor

@dataclass
class EpsilonConfig:
    eps_start: float = 1.0
    eps_end: float = 0.05

@dataclass
class RenderConfig:
    enable_gui: bool = True
    render_every: int = 1
    sleep: float = 0.02
    cell_size_small: int = 64
    cell_size_large: int = 40
    render_max_size_for_large_cells: int = 8  # if env.size <= this -> use cell_size_small

@dataclass
class TrainConfig:
    episodes: int = 600
    save_every: int = 0  # placeholder for future checkpointing
    log_every: int = 20


@dataclass
class ProjectConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    eps: EpsilonConfig = field(default_factory=EpsilonConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def max_size_canvas(self) -> int:
        """Effective max_size_canvas used by encoder (defaults to env.max_size)."""
        return self.env.max_size if self.env.max_size_canvas is None else self.env.max_size_canvas
    

# Global config instance
CFG = ProjectConfig()