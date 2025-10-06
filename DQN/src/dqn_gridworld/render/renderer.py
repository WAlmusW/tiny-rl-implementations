"""Tkinter-based renderer for GridWorld.

This mirrors the single-file GridRenderer but is defensive when Tk isn't available
(e.g. headless CI). The training code should handle None renderer gracefully.
"""
from __future__ import annotations
import time
from typing import Optional

try:
    import tkinter as tk
except Exception:  # tkinter may not be installed on headless systems
    tk = None  # type: ignore

from ..env.gridworld import GridWorld


class GridRenderer:
    """Simple grid renderer using Tkinter.

    The renderer expects an env with attributes:
      - size, pits, goal, pos
    """
    def __init__(self, env: GridWorld, cell_size: int = 64, sleep: float = 0.02):
        if tk is None:
            raise RuntimeError("Tkinter isn't available. Initialize renderer only when GUI is desired.")
        self.env = env
        self.cell = int(cell_size)
        self.sleep = float(sleep)

        self.root = tk.Tk()
        self.root.title("DQN GridWorld (variable size)")
        w = env.size * self.cell
        h = env.size * self.cell + 28
        self.canvas = tk.Canvas(self.root, width=w, height=h)
        self.canvas.pack()

    def draw(self, episode: int, step: int, total_reward: float, eps: float) -> None:
        c = self.canvas
        s = self.env.size
        cell = self.cell
        c.delete("all")

        # Grid cells
        for i in range(s):
            for j in range(s):
                x0, y0 = i * cell, j * cell
                x1, y1 = x0 + cell, y0 + cell
                c.create_rectangle(x0, y0, x1, y1, outline="#888", width=1)

        # Pits
        for (px, py) in self.env.pits:
            x0, y0 = px * cell, py * cell
            c.create_rectangle(x0 + 3, y0 + 3, x0 + cell - 3, y0 + cell - 3, fill="#f88", outline="")

        # Goal
        gx, gy = self.env.goal
        c.create_rectangle(gx * cell + 3, gy * cell + 3, gx * cell + cell - 3, gy * cell + cell - 3, fill="#8f8", outline="")

        # Agent
        ax, ay = self.env.pos
        c.create_oval(ax * cell + 8, ay * cell + 8, ax * cell + cell - 8, ay * cell + cell - 8, fill="#88f", outline="")

        # Status line
        c.create_text(10, s * cell + 14, anchor='w',
                      text=f"Ep {episode} | step {step} | R={total_reward:.2f} | eps={eps:.3f} | size={s}",
                      font=("Segoe UI", 10))

        # push the UI and optionally sleep a bit (to slow down rendering)
        self.root.update_idletasks()
        self.root.update()
        if self.sleep > 0:
            time.sleep(self.sleep)

    def close(self) -> None:
        try:
            self.root.destroy()
        except Exception:
            pass
