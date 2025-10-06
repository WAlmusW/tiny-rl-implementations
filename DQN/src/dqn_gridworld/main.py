"""Simple package entrypoint for dqn_gridworld.

Usage from project root:
    python -m src.dqn_gridworld.main --episodes 10 --no-gui
or call the CLI module directly:
    python -m src.dqn_gridworld.cli --episodes 10 --no-gui
"""
from __future__ import annotations

# delegate to the CLI entrypoint so we don't duplicate argparse logic
from .cli import main as cli_main

if __name__ == "__main__":
    cli_main()
