"""Simple package entrypoint for q_learning_gridworld.

Usage from project root:
    python -m src.q_learning_gridworld.main --episodes 10 --no-gui
or call the CLI module directly:
    python -m src.q_learning_gridworld.cli --episodes 10 --no-gui
"""
from __future__ import annotations

# delegate to the CLI entrypoint so we don't duplicate argparse logic
from .cli import main as cli_main

if __name__ == "__main__":
    cli_main()
