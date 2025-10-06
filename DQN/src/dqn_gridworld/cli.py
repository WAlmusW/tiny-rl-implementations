"""Thin CLI that wires argparse to the training run function and config."""
from __future__ import annotations
import argparse
import sys

from config import CFG
from .train import run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="NumPy DQN + Tkinter GridWorld (variable size + goal/pit-aware obs)"
    )

    # training
    p.add_argument("--episodes", type=int, default=CFG.train.episodes, help="Number of training episodes")
    p.add_argument("--seed", type=int, default=CFG.agent.seed, help="Random seed")

    # rendering
    p.add_argument("--render-every", type=int, default=CFG.render.render_every, help="Render every N episodes (1=always)")
    p.add_argument("--sleep", type=float, default=CFG.render.sleep, help="Delay between rendered steps (seconds)")
    p.add_argument("--no-gui", action="store_true", help="Disable Tkinter GUI (headless training)")

    # layout / env
    p.add_argument("--min-size", type=int, default=CFG.env.min_size, help="Minimum grid size (inclusive)")
    p.add_argument("--max-size", type=int, default=CFG.env.max_size, help="Maximum grid size (inclusive)")
    p.add_argument("--episodes-per-layout", type=int, default=CFG.env.episodes_per_layout, help="Resample layout every N episodes")
    p.add_argument("--pit-frac", type=float, default=CFG.env.pit_frac, help="Approx fraction of cells that are pits (0..1)")
    p.add_argument("--max-size-canvas", type=int, default=CFG.max_size_canvas, help="Canvas resolution for pit mask; defaults to --max-size if omitted")

    # debug / misc
    p.add_argument("--no-render-on-start", action="store_true", help="Don't render the initial episode immediately (quiet)")
    return p


def main(argv=None):
    argv = argv or sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)

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
    )


if __name__ == "__main__":
    main()
