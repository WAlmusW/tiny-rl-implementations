"""Minimal Tkinter grid renderer (kept minimal & compatible with the DQN renderer).

This module provides a tiny visualization utility for the GridWorld environment.
It's intentionally lightweight so training can run headless when tkinter is not
available (e.g., on CI or headless servers). When GUI rendering is desired, the
GridRenderer opens a simple Tkinter window and draws:
  - grid cells,
  - pits (red squares),
  - goal (green square),
  - agent (blue circle),
  - a status text line with episode/step/return/epsilon/size.

Behavioral notes
----------------
- The module tries to import tkinter at top-level. If tkinter is missing (common
  in headless environments), the `tk` symbol is set to None and attempting to
  instantiate GridRenderer raises a RuntimeError. This lets calling code choose
  whether to attempt GUI creation or fall back to headless behavior.
- The renderer's `draw()` method calls Tkinter's update loop (`update_idletasks` +
  `update`) to push frames to the window. It optionally sleeps between frames to
  slow down rendering for visualization.
"""

from __future__ import annotations
import time
from typing import Optional

try:
    import tkinter as tk
except Exception:  # tkinter may not be installed on headless systems
    tk = None  # type: ignore


class GridRenderer:
    """Simple Tkinter-based grid renderer for GridWorld.

    Args:
        env: GridWorld instance providing `size`, `pits`, `goal`, and `pos`.
        cell_size: Pixel size of each grid cell (square).
        sleep: Optional delay (seconds) to sleep after each `draw()` call; set
               to 0 for fastest non-blocking updates.

    Example:
        renderer = GridRenderer(env, cell_size=40, sleep=0.02)
        renderer.draw(episode, step, total_reward, eps)
        renderer.close()
    """

    def __init__(self, env, cell_size: int = 40, sleep: float = 0.02):
        # If tkinter isn't available, make the failure explicit early.
        if tk is None:
            raise RuntimeError("Tkinter isn't available. Initialize renderer only when GUI is desired.")

        self.env = env
        self.cell = int(cell_size)
        self.sleep = float(sleep)

        # Create the main window and canvas sized to the environment.
        self.root = tk.Tk()
        self.root.title("Q-Learning GridWorld (variable size)")
        # Width and height: grid area plus a small status line height.
        w = env.size * self.cell
        h = env.size * self.cell + 28
        self.canvas = tk.Canvas(self.root, width=w, height=h)
        self.canvas.pack()

    def draw(self, episode: int, step: int, total_reward: float, eps: float) -> None:
        """Redraw the entire grid and status line for the current environment state.

        Args:
            episode: current episode index (for display).
            step: current step within the episode (for display).
            total_reward: cumulative reward for the episode so far (for display).
            eps: current epsilon used by the agent (for display).

        Notes:
            - The method clears and redraws everything each frame which is simple
              and acceptable for small grids. For large/complex UIs incremental
              drawing would be more efficient.
            - Colors and margins are chosen for clear visual contrast; they can be
              adjusted easily if you want a different look.
        """
        c = self.canvas
        s = self.env.size
        cell = self.cell

        # Clear previous frame
        c.delete("all")

        # Draw grid cell outlines.
        for i in range(s):
            for j in range(s):
                x0, y0 = i * cell, j * cell
                x1, y1 = x0 + cell, y0 + cell
                c.create_rectangle(x0, y0, x1, y1, outline="#888", width=1)

        # Draw pits as slightly inset red rectangles.
        for (px, py) in self.env.pits:
            x0, y0 = px * cell, py * cell
            c.create_rectangle(x0 + 3, y0 + 3, x0 + cell - 3, y0 + cell - 3, fill="#f88", outline="")

        # Draw goal as inset green rectangle.
        gx, gy = self.env.goal
        c.create_rectangle(gx * cell + 3, gy * cell + 3, gx * cell + cell - 3, gy * cell + cell - 3, fill="#8f8", outline="")

        # Draw agent as an oval (circle) centered in its cell.
        ax, ay = self.env.pos
        c.create_oval(ax * cell + 8, ay * cell + 8, ax * cell + cell - 8, ay * cell + cell - 8, fill="#88f", outline="")

        # Draw status text line below the grid showing progress and parameters.
        c.create_text(10, s * cell + 14, anchor='w',
                      text=f"Ep {episode} | step {step} | R={total_reward:.2f} | eps={eps:.3f} | size={s}",
                      font=("Segoe UI", 10))

        # Push the UI event loop to display changes immediately. `update()` may
        # process pending events which keeps the window responsive.
        self.root.update_idletasks()
        self.root.update()

        # Optional small sleep to slow down visualization for human viewing.
        if self.sleep > 0:
            time.sleep(self.sleep)

    def close(self) -> None:
        """Close the renderer window.

        Guard against errors from the window manager by catching exceptions
        raised during `destroy()`.
        """
        try:
            self.root.destroy()
        except Exception:
            # Be robust — closing during interpreter shutdown or if the window
            # has already been destroyed may raise.
            pass
