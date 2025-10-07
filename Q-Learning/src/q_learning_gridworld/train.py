"""Training loop for tabular Q-Learning.

API mirrors the DQN training loop to enable side-by-side comparisons:
- uses LayoutSampler, encode_obs_fixed
- uses GridWorld
- agent implements act(state, eps) and train_step(batch)
- uses ReplayBuffer wrapper for consistent sampling interface (even though tabular doesn't require minibatches)
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
    # Resolve arguments with config defaults when None
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

    random.seed(seed)
    np.random.seed(seed)

    if max_size_canvas < max_size:
        raise ValueError("--max-size-canvas must be >= --max-size to safely encode all layouts.")

    obs_dim = 4 + (max_size_canvas * max_size_canvas)
    n_actions = 4

    agent = QLearning(obs_dim, n_actions, alpha=alpha, gamma=gamma, seed=seed)

    # Use ReplayBuffer wrapper (for parity with DQN training loop).
    # For tabular Q-Learning we'll use batch-size 1 updates by default, but the buffer
    # allows sampling/aggregation if you want to experiment with batch updates.
    replay = ReplayBuffer(capacity=CFG.train.episodes * 2)
    batch_size = 1
    warmup = 0

    logger = MetricsLogger(output_dir="runs/qlearning", auto_stamp=True)
    losses_this_ep = []

    eps_start, eps_end = CFG.eps.eps_start, CFG.eps.eps_end

    sampler = LayoutSampler(min_size=min_size, max_size=max_size, pit_frac=pit_frac, seed=seed)

    # Initial layout + env
    layout = sampler.sample()
    env = GridWorld(size=layout.size, pits=layout.pits, goal=layout.goal, start=layout.start,
                    max_steps=int(layout.size * layout.size * CFG.env.max_steps_multiplier))

    renderer = None
    if not no_gui:
        try:
            cell_size = CFG.render.cell_size_small if env.size <= CFG.render.render_max_size_for_large_cells else CFG.render.cell_size_large
            renderer = GridRenderer(env, cell_size=cell_size, sleep=sleep)
        except Exception:
            renderer = None

    global_step = 0
    try:
        for ep in range(1, episodes + 1):
            # Resample layout every N episodes (except before first)
            if (ep - 1) % episodes_per_layout == 0 and ep > 1:
                layout = sampler.sample()
                env = GridWorld(size=layout.size, pits=layout.pits, goal=layout.goal, start=layout.start,
                                max_steps=int(layout.size * layout.size * CFG.env.max_steps_multiplier))
                if renderer:
                    renderer.close()
                    cell_size = CFG.render.cell_size_small if env.size <= CFG.render.render_max_size_for_large_cells else CFG.render.cell_size_large
                    renderer = GridRenderer(env, cell_size=cell_size, sleep=sleep)

            env.reset()
            s = encode_obs_fixed(env, max_size_canvas)
            total_reward = 0.0
            done = False

            frac = (ep - 1) / max(1, episodes - 1)
            eps = linear_decay(eps_start, eps_end, frac)

            step = 0
            while not done:
                step += 1
                a = agent.act(s, eps)
                res = env.step(a)

                s2 = encode_obs_fixed(env, max_size_canvas)
                r, done = res.reward, res.done
                replay.append((s, a, r, s2, float(done)))
                s = s2
                total_reward += r

                if len(replay) >= max(batch_size, warmup):
                    batch = replay.sample(batch_size)
                    loss = agent.train_step(batch)  # QLearning.train_step returns a float
                    if loss is not None:
                        losses_this_ep.append(float(loss))

                if renderer and (ep % render_every == 0):
                    renderer.draw(ep, step, total_reward, eps)

                global_step += 1

            if renderer and (ep % render_every != 0):
                renderer.draw(ep, step, total_reward, eps)

            # compute average loss for this episode
            avg_loss = float(np.mean(losses_this_ep)) if len(losses_this_ep) > 0 else 0.0

            # last step result 'res' holds info about terminal cause ('pit', 'goal', or 'timeout')
            terminal = None
            if isinstance(res, object) and hasattr(res, "info") and isinstance(res.info, dict):
                terminal = res.info.get("terminal")

            # record episode metrics
            logger.log_episode(ep=ep, total_reward=total_reward, eps=eps, n_steps=step, avg_loss=avg_loss, terminal=terminal)

            if ep % CFG.train.log_every == 0:
                logger.flush()
                print(
                    f"Episode {ep:4d}/{episodes} | eps={eps:.3f} | return={total_reward:+.2f} "
                    f"| size={env.size} pits={len(env.pits)} start={env.start} goal={env.goal}"
                )

            losses_this_ep.clear()

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        if renderer:
            renderer.close()
        try:
            final = logger.flush()
            print(f"Final metrics saved to: {final['csv']}")
        except Exception:
            pass
