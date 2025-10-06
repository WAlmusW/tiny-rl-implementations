"""Training loop extracted from the single-file example.

Relies on:
- env.gridworld.GridWorld
- env.layout.LayoutSampler, encode_obs_fixed
- agent.dqn.DQN
- render.renderer.GridRenderer (optional)
- config.CFG
- utils.replay.ReplayBuffer
"""
from __future__ import annotations
import random
import time
from collections import deque  # kept for typing compatibility in some environments

import numpy as np

from config import CFG
from .agent.dqn import DQN
from .env.gridworld import GridWorld
from .env.layout import LayoutSampler, encode_obs_fixed
from .render.renderer import GridRenderer  # may raise if tkinter unavailable
from .utils.replay import ReplayBuffer


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

    random.seed(seed)
    np.random.seed(seed)

    if max_size_canvas < max_size:
        raise ValueError("--max-size-canvas must be >= --max-size to safely encode all layouts.")

    obs_dim = 4 + (max_size_canvas * max_size_canvas)
    n_actions = 4

    agent = DQN(obs_dim, n_actions,
                hidden=CFG.agent.hidden,
                lr=CFG.agent.lr,
                gamma=CFG.agent.gamma,
                seed=seed)

    # Use ReplayBuffer wrapper (uniform sampling)
    replay = ReplayBuffer(capacity=CFG.agent.replay_capacity)
    batch_size = CFG.agent.batch_size
    warmup = CFG.agent.warmup
    target_update = CFG.agent.target_update

    eps_start, eps_end = CFG.eps.eps_start, CFG.eps.eps_end

    sampler = LayoutSampler(min_size=min_size, max_size=max_size, pit_frac=pit_frac, seed=seed)

    # Initial layout + env
    layout = sampler.sample()
    env = GridWorld(size=layout.size, pits=layout.pits, goal=layout.goal, start=layout.start,
                    max_steps=int(layout.size * layout.size * CFG.env.max_steps_multiplier))

    renderer = None
    if not no_gui:
        try:
            # choose cell size heuristically
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
                    _ = agent.train_step(batch, clip_grad=CFG.agent.clip_grad)

                if global_step % target_update == 0:
                    agent.target.copy_from(agent.q)

                if renderer and (ep % render_every == 0):
                    renderer.draw(ep, step, total_reward, eps)

                global_step += 1

            if renderer and (ep % render_every != 0):
                renderer.draw(ep, step, total_reward, eps)

            if ep % CFG.train.log_every == 0:
                print(
                    f"Episode {ep:4d}/{episodes} | eps={eps:.3f} | return={total_reward:+.2f} "
                    f"| size={env.size} pits={len(env.pits)} start={env.start} goal={env.goal}"
                )

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        if renderer:
            renderer.close()
