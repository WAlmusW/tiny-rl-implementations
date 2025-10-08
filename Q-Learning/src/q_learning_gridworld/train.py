"""Training loop for tabular Q-Learning.

This module implements the main training routine used by the CLI. The API and
behavior intentionally mirror a small DQN-style training loop so it is easy to
compare tabular Q-Learning runs with functionally similar DQN code.

High-level behavior
-------------------
- Samples grid layouts via `LayoutSampler` and builds `GridWorld` instances.
- Encodes observations into fixed-width vectors using `encode_obs_fixed`.
- Uses the `QLearning` agent for action selection and updates.
- Keeps a `ReplayBuffer` for API parity (tabular updates use batch size 1 by default).
- Optionally renders episodes using `GridRenderer` (Tkinter) when GUI is available.
- Logs episode metrics via `MetricsLogger` and periodically flushes to disk.

The training loop is intentionally straightforward and synchronous — it favors
clarity over micro-optimizations appropriate for small demos and educational use.
"""

from __future__ import annotations
import random
import time

import numpy as np

from config import CFG
from .agent.q_learning import QLearning
from .env.gridworld import GridWorld
from .env.layout import LayoutSampler, encode_obs_fixed
from .render.renderer import GridRenderer  # may raise if tkinter unavailable
from .utils.replay import ReplayBuffer
from .utils.metrics import MetricsLogger


def linear_decay(eps_start: float, eps_end: float, fraction: float) -> float:
    """Linearly decay epsilon between `eps_start` and `eps_end`.

    Args:
        eps_start: starting epsilon value.
        eps_end: final epsilon value.
        fraction: fraction in [0, 1] indicating progress (0=start, 1=end).

    Returns:
        float: interpolated epsilon.
    """
    return eps_end + (eps_start - eps_end) * max(0.0, 1.0 - fraction)


def run(
    episodes: int | None = None,
    seed: int | None = None,
    render_every: int | None = None,
    sleep: float | None = None,
    no_gui: bool = False,
    min_size: int | None = None,
    max_size: int | None = None,
    episodes_per_layout: int | None = None,
    pit_frac: float | None = None,
    max_size_canvas: int | None = None,
    alpha: float | None = None,
    gamma: float | None = None,
):
    """Run the Q-Learning training loop with configurable parameters.

    Args:
        episodes: number of training episodes (falls back to CFG if None).
        seed: RNG seed for reproducibility (falls back to CFG if None).
        render_every: render every N episodes (1=every episode). If None use CFG.
        sleep: seconds to sleep between rendered frames (None -> CFG default).
        no_gui: if True skip trying to create the Tkinter renderer.
        min_size, max_size: range of grid sizes to sample (None -> CFG defaults).
        episodes_per_layout: resample layout every N episodes (None -> CFG).
        pit_frac: approximate fraction of cells that are pits (None -> CFG).
        max_size_canvas: canvas resolution for the pit-mask observation encoding.
                         If None uses CFG.max_size_canvas.
        alpha: learning rate for Q-Learning (None -> CFG.agent.alpha).
        gamma: discount factor (None -> CFG.agent.gamma).

    Notes:
        - This function preserves the training behavior used elsewhere in the
          project; only documentation and comments are added here.
    """
    # Resolve arguments using CFG defaults when user didn't supply them.
    episodes = episodes if episodes is not None else CFG.train.episodes
    seed = seed if seed is not None else CFG.agent.seed
    render_every = render_every if render_every is not None else CFG.render.render_every
    sleep = sleep if sleep is not None else CFG.render.sleep
    min_size = min_size if min_size is not None else CFG.env.min_size
    max_size = max_size if max_size is not None else CFG.env.max_size
    episodes_per_layout = episodes_per_layout if episodes_per_layout is not None else CFG.env.episodes_per_layout
    pit_frac = pit_frac if pit_frac is not None else CFG.env.pit_frac
    max_size_canvas = CFG.max_size_canvas if max_size_canvas is None else max_size_canvas
    alpha = alpha if alpha is not None else CFG.agent.alpha
    gamma = gamma if gamma is not None else CFG.agent.gamma

    # Seed the top-level RNGs for reproducibility across components.
    random.seed(seed)
    np.random.seed(seed)

    # Safety check: canvas must be able to represent the largest grid.
    if max_size_canvas < max_size:
        raise ValueError("--max-size-canvas must be >= --max-size to safely encode all layouts.")

    # Observation dimensionality: 4 normalized coords + flattened pit mask.
    obs_dim = 4 + (max_size_canvas * max_size_canvas)
    n_actions = 4  # up/right/down/left

    # Instantiate agent. Note: obs_dim isn't used by the tabular QLearning instance
    # internally, but is provided to preserve API parity with other agents.
    agent = QLearning(obs_dim, n_actions, alpha=alpha, gamma=gamma, seed=seed)

    # Use ReplayBuffer wrapper (for parity with DQN training loop).
    # For tabular Q-Learning we'll use batch-size 1 updates by default, but the buffer
    # allows sampling/aggregation if you want to experiment with batch updates.
    replay = ReplayBuffer(capacity=CFG.train.episodes * 2)
    batch_size = 1
    warmup = 0  # no warmup for tabular agent

    logger = MetricsLogger(output_dir="runs/qlearning", auto_stamp=True)
    losses_this_ep = []

    eps_start, eps_end = CFG.eps.eps_start, CFG.eps.eps_end

    # Layout sampler controls the environments we generate for each layout.
    sampler = LayoutSampler(min_size=min_size, max_size=max_size, pit_frac=pit_frac, seed=seed)

    # Initial layout and environment creation.
    layout = sampler.sample()
    env = GridWorld(size=layout.size, pits=layout.pits, goal=layout.goal, start=layout.start,
                    max_steps=int(layout.size * layout.size * CFG.env.max_steps_multiplier))

    # Optionally create a renderer; on headless machines this may raise so we guard it.
    renderer = None
    if not no_gui:
        try:
            cell_size = CFG.render.cell_size_small if env.size <= CFG.render.render_max_size_for_large_cells else CFG.render.cell_size_large
            renderer = GridRenderer(env, cell_size=cell_size, sleep=sleep)
        except Exception:
            # Renderer creation failed (e.g., tkinter not available). Continue headless.
            renderer = None

    global_step = 0
    try:
        # Main episode loop
        for ep in range(1, episodes + 1):
            res = None  # ensure `res` exists even if episode ends immediately

            # Optionally resample layout at configured intervals (except before first).
            if (ep - 1) % episodes_per_layout == 0 and ep > 1:
                layout = sampler.sample()
                env = GridWorld(size=layout.size, pits=layout.pits, goal=layout.goal, start=layout.start,
                                max_steps=int(layout.size * layout.size * CFG.env.max_steps_multiplier))
                # If a renderer exists, close and recreate it for the new environment size.
                if renderer:
                    renderer.close()
                    cell_size = CFG.render.cell_size_small if env.size <= CFG.render.render_max_size_for_large_cells else CFG.render.cell_size_large
                    renderer = GridRenderer(env, cell_size=cell_size, sleep=sleep)

            # Reset environment and encode initial observation.
            env.reset()
            s = encode_obs_fixed(env, max_size_canvas)
            total_reward = 0.0
            done = False

            # Compute epsilon for this episode via linear schedule.
            frac = (ep - 1) / max(1, episodes - 1)
            eps = linear_decay(eps_start, eps_end, frac)

            step = 0
            while not done:
                step += 1

                # Select action and step the environment.
                a = agent.act(s, eps)
                res = env.step(a)

                # Encode next-state observation for agent + store transition.
                s2 = encode_obs_fixed(env, max_size_canvas)
                r, done = res.reward, res.done
                replay.append((s, a, r, s2, float(done)))
                s = s2
                total_reward += r

                # When we have enough transitions, sample a batch and update agent.
                if len(replay) >= max(batch_size, warmup):
                    batch = replay.sample(batch_size)
                    loss = agent.train_step(batch)  # QLearning.train_step returns a float
                    if loss is not None:
                        losses_this_ep.append(float(loss))

                # Render periodically based on render_every param.
                if renderer and (ep % render_every == 0):
                    renderer.draw(ep, step, total_reward, eps)

                global_step += 1

            # After episode ends, optionally render once more when not in the regular render cadence.
            if renderer and (ep % render_every != 0):
                renderer.draw(ep, step, total_reward, eps)

            # Compute average loss for the episode (useful for logging/diagnostics).
            avg_loss = float(np.mean(losses_this_ep)) if len(losses_this_ep) > 0 else 0.0

            # Determine terminal reason from last step's info if present.
            terminal = None
            if isinstance(res, object) and hasattr(res, "info") and isinstance(res.info, dict):
                terminal = res.info.get("terminal")

            # Record episode metrics
            logger.log_episode(ep=ep, total_reward=total_reward, eps=eps, n_steps=step, avg_loss=avg_loss, terminal=terminal)

            # Periodically flush metrics to disk and print a short summary.
            if ep % CFG.train.log_every == 0:
                logger.flush()
                print(
                    f"Episode {ep:4d}/{episodes} | eps={eps:.3f} | return={total_reward:+.2f} "
                    f"| size={env.size} pits={len(env.pits)} start={env.start} goal={env.goal}"
                )

            # Clear per-episode losses buffer for the next episode.
            losses_this_ep.clear()

    except KeyboardInterrupt:
        # Allow the user to stop training with Ctrl+C and still persist metrics.
        print("Training interrupted by user.")
    finally:
        # Ensure renderer is closed and metrics are flushed to disk when possible.
        if renderer:
            renderer.close()
        try:
            final = logger.flush()
            print(f"Final metrics saved to: {final['csv']}")
        except Exception:
            # Avoid raising on final flush issues (e.g., filesystem errors) during shutdown.
            pass
