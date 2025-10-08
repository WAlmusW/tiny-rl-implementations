"""Thin CLI that wires argparse to the training run function and project config.

This module provides a small command-line interface to run training with
convenient flags that mirror the project's configuration (`config.CFG`). The
CLI parses arguments, fills in defaults from `CFG` when flags are omitted, and
calls `train.run(...)` with the resolved parameters.

Design notes
------------
- Keeping CLI logic minimal keeps behavior consistent across entry points and
  avoids duplicating argparse defaults; `CFG` is the single source of truth.
- The CLI is designed for interactive use (development) and simple scriptable
  invocation from CI or shells.

Example
-------
From project root:
    python -m src.q_learning_gridworld.cli --episodes 50 --no-gui
"""
from __future__ import annotations
import argparse
import sys

from config import CFG
from .train import run


def build_parser() -> argparse.ArgumentParser:
    """Construct and return the argparse.ArgumentParser for the training CLI.

    The parser mirrors the main configuration options found in `config.CFG`.
    Defaults are populated from `CFG` so they remain synchronized with the
    programmatic configuration.

    Returns:
        argparse.ArgumentParser: configured parser ready to parse CLI args.
    """
    p = argparse.ArgumentParser(
        description="Tabular Q-Learning + Tkinter GridWorld (variable size + goal/pit-aware obs)"
    )

    # Training control
    p.add_argument("--episodes", type=int, default=CFG.train.episodes, help="Number of training episodes")
    p.add_argument("--seed", type=int, default=CFG.agent.seed, help="Random seed")

    # Rendering / GUI
    p.add_argument("--render-every", type=int, default=CFG.render.render_every, help="Render every N episodes (1=always)")
    p.add_argument("--sleep", type=float, default=CFG.render.sleep, help="Delay between rendered steps (seconds)")
    p.add_argument("--no-gui", action="store_true", help="Disable Tkinter GUI (headless training)")

    # Layout / environment sampling
    p.add_argument("--min-size", type=int, default=CFG.env.min_size, help="Minimum grid size (inclusive)")
    p.add_argument("--max-size", type=int, default=CFG.env.max_size, help="Maximum grid size (inclusive)")
    p.add_argument("--episodes-per-layout", type=int, default=CFG.env.episodes_per_layout, help="Resample layout every N episodes")
    p.add_argument("--pit-frac", type=float, default=CFG.env.pit_frac, help="Approx fraction of cells that are pits (0..1)")
    p.add_argument("--max-size-canvas", type=int, default=CFG.max_size_canvas, help="Canvas resolution for pit mask; defaults to --max-size if omitted")

    # Q-learning specific parameters
    p.add_argument("--alpha", type=float, default=CFG.agent.alpha, help="Learning rate for Q-Learning")
    p.add_argument("--gamma", type=float, default=CFG.agent.gamma, help="Discount factor for Q-Learning")

    # Misc / debug
    p.add_argument("--no-render-on-start", action="store_true", help="Don't render the initial episode immediately (quiet)")
    return p


def main(argv=None):
    """Parse CLI arguments and delegate to train.run().

    Args:
        argv: Optional list of command-line arguments (defaults to sys.argv[1:]).
              Providing `argv` makes this function easier to test programmatically.
    """
    argv = argv or sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)

    # Delegate to the run() function in train.py, mapping CLI args to parameters.
    run(
        episodes=args.episodes,
        seed=args.seed,
        render_every=args.render_every,
        sleep=args.sleep,
        no_gui=args.no_gui,
        min_size=args.min_size,
        max_size=args.max_size,
        episodes_per_layout=args.episodes_per_layout,
        pit_frac=args.pit_frac,
        max_size_canvas=args.max_size_canvas,
        alpha=args.alpha,
        gamma=args.gamma,
    )


if __name__ == "__main__":
    main()
