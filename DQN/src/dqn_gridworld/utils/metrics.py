"""Lightweight metrics logger + plotter for training runs.

Usage:
    logger = MetricsLogger(output_dir="runs/exp1")
    logger.log_step(...)  # optional per-step metrics
    logger.log_episode(...)  # one call at episode end
    logger.flush()  # write CSV and create plots

The module only depends on numpy + matplotlib (install via pip if needed).
"""

from __future__ import annotations
import os
import csv
import time
from typing import Dict, List, Optional
import numpy as np

# matplotlib import only when plotting (so headless code can avoid import errors if not used)
import matplotlib
# prefer non-interactive backend for headless by default
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class MetricsLogger:
    def __init__(self, output_dir: str = "runs/run", auto_stamp: bool = True):
        stamp = f"-{int(time.time())}" if auto_stamp else ""
        self.output_dir = f"{output_dir}{stamp}"
        os.makedirs(self.output_dir, exist_ok=True)

        # Per-episode lists
        self.episodes: List[int] = []
        self.returns: List[float] = []
        self.epsilons: List[float] = []
        self.steps: List[int] = []
        self.avg_losses: List[float] = []
        self.success: List[int] = []   # 1 if goal reached
        self.pit: List[int] = []       # 1 if fell to pit
        self.timeout: List[int] = []   # 1 if timed out

        # optional per-step metrics (kept short)
        self.step_times: List[float] = []

    def log_episode(
        self,
        ep: int,
        total_reward: float,
        eps: float,
        n_steps: int,
        avg_loss: Optional[float] = None,
        terminal: Optional[str] = None,
    ):
        """Record summary for one episode."""
        self.episodes.append(ep)
        self.returns.append(float(total_reward))
        self.epsilons.append(float(eps))
        self.steps.append(int(n_steps))
        self.avg_losses.append(float(avg_loss) if avg_loss is not None else 0.0)

        # terminal: one of "goal", "pit", "timeout", None
        self.success.append(1 if terminal == "goal" else 0)
        self.pit.append(1 if terminal == "pit" else 0)
        self.timeout.append(1 if terminal == "timeout" else 0)

    def log_step_time(self, t: float):
        self.step_times.append(float(t))

    def to_csv(self, filename: Optional[str] = None):
        filename = filename or os.path.join(self.output_dir, "metrics.csv")
        headers = [
            "episode", "return", "eps", "steps", "avg_loss", "success", "pit", "timeout"
        ]
        with open(filename, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(headers)
            for i in range(len(self.episodes)):
                writer.writerow([
                    self.episodes[i],
                    self.returns[i],
                    self.epsilons[i],
                    self.steps[i],
                    self.avg_losses[i],
                    self.success[i],
                    self.pit[i],
                    self.timeout[i],
                ])
        return filename

    def plot(self, save_all: bool = True):
        """Create and save diagnostic plots (PNG files) into output_dir."""
        if len(self.episodes) == 0:
            return []

        out_files = []

        # rolling helper
        def rolling(x, window=20):
            if len(x) < window:
                return np.array(x)
            return np.convolve(x, np.ones(window) / window, mode="valid")

        # 1) Return per episode (+ rolling)
        plt.figure(figsize=(8, 4))
        plt.plot(self.episodes, self.returns, label="return")
        r = rolling(self.returns, window=20)
        if r.size > 0:
            plt.plot(self.episodes[len(self.episodes) - len(r):], r, label="rolling(20)")
        plt.xlabel("episode")
        plt.ylabel("return")
        plt.title("Episode return")
        plt.legend()
        f1 = os.path.join(self.output_dir, "return.png")
        plt.tight_layout()
        plt.savefig(f1)
        plt.close()
        out_files.append(f1)

        # 2) Epsilon
        plt.figure(figsize=(6, 3))
        plt.plot(self.episodes, self.epsilons)
        plt.xlabel("episode")
        plt.ylabel("epsilon")
        plt.title("Epsilon decay")
        f2 = os.path.join(self.output_dir, "epsilon.png")
        plt.tight_layout()
        plt.savefig(f2)
        plt.close()
        out_files.append(f2)

        # 3) Average loss
        plt.figure(figsize=(8, 4))
        plt.plot(self.episodes, self.avg_losses, label="avg_loss")
        r2 = rolling(self.avg_losses, window=50)
        if r2.size > 0:
            plt.plot(self.episodes[len(self.episodes) - len(r2):], r2, label="rolling(50)")
        plt.xlabel("episode")
        plt.ylabel("loss")
        plt.title("Avg training loss (per-episode)")
        plt.legend()
        f3 = os.path.join(self.output_dir, "loss.png")
        plt.tight_layout()
        plt.savefig(f3)
        plt.close()
        out_files.append(f3)

        # 4) Episode length
        plt.figure(figsize=(6, 3))
        plt.plot(self.episodes, self.steps)
        plt.xlabel("episode")
        plt.ylabel("steps")
        plt.title("Episode length (# steps)")
        f4 = os.path.join(self.output_dir, "steps.png")
        plt.tight_layout()
        plt.savefig(f4)
        plt.close()
        out_files.append(f4)

        # 5) Terminal distribution (stacked)
        plt.figure(figsize=(6, 3))
        arr = np.vstack([self.success, self.pit, self.timeout])
        # cumulative / rolling fraction
        cum = np.cumsum(arr, axis=1)
        n = np.arange(1, len(self.episodes) + 1)
        plt.plot(self.episodes, np.cumsum(self.success) / n, label="goal rate (cum)")
        plt.plot(self.episodes, np.cumsum(self.pit) / n, label="pit rate (cum)")
        plt.plot(self.episodes, np.cumsum(self.timeout) / n, label="timeout rate (cum)")
        plt.xlabel("episode")
        plt.ylabel("cumulative rate")
        plt.title("Terminal rates (cumulative)")
        plt.legend()
        f5 = os.path.join(self.output_dir, "terminals.png")
        plt.tight_layout()
        plt.savefig(f5)
        plt.close()
        out_files.append(f5)

        return out_files

    def flush(self):
        csv_file = self.to_csv()
        pngs = self.plot()
        return {"csv": csv_file, "plots": pngs}
