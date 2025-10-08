"""Simple package entrypoint for the Q-Learning GridWorld project.

This module serves as the top-level entry point when executing the package with
`python -m`. It simply delegates to the command-line interface defined in
`cli.py` so that users can run the project without explicitly referencing the
CLI module.

Usage
-----
From project root:
    python -m src.q_learning_gridworld.main --episodes 10 --no-gui

Alternatively, you can directly call the CLI:
    python -m src.q_learning_gridworld.cli --episodes 10 --no-gui

Both forms are equivalent.
"""

from __future__ import annotations

# Delegate to CLI entrypoint so we don't duplicate argument parsing logic.
from .cli import main as cli_main


if __name__ == "__main__":
    # Forward command-line execution to CLI parser.
    cli_main()
