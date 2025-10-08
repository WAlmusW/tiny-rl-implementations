# q-learning_gridworld_varsize

**Minimal NumPy Q-Learning + Tkinter GridWorld (variable map layouts)**

A tiny, well-commented educational project that demonstrates tabular Q-Learning on a small 2D GridWorld.  
It intentionally keeps dependencies minimal (NumPy + optional Tkinter + Matplotlib) and favors clarity over complexity so newcomers can follow every step.

---

## Table of contents

- [q-learning\_gridworld\_varsize](#q-learning_gridworld_varsize)
  - [Table of contents](#table-of-contents)
  - [Overview](#overview)
  - [What you'll learn](#what-youll-learn)
  - [Algorithm (intuition + formula)](#algorithm-intuition--formula)
  - [Observation encoding](#observation-encoding)
  - [Quick start](#quick-start)
  - [CLI / config summary](#cli--config-summary)
  - [Project layout (files)](#project-layout-files)
  - [Metrics \& visualization](#metrics--visualization)
  - [Tips for experimenting](#tips-for-experimenting)
  - [Troubleshooting](#troubleshooting)
  - [Contributing \& License](#contributing--license)

---

## Overview

This repository implements a simple Q-Learning agent that learns to navigate square GridWorlds that can vary in size and layout. Each sampled layout contains:

- a start cell,
- a goal cell (big positive reward),
- several pits (big negative reward),
- and a small step penalty to encourage shorter paths.

A separate encoder converts variable-sized grids into fixed-width observations so the same agent can train across multiple map sizes.

The Tkinter renderer is optional — you can run headless training or enable live visualization to watch the agent learn.

---

## What you'll learn

- The **tabular Q-Learning** algorithm (how values are stored and updated).
- Epsilon-greedy exploration and a simple linear decay schedule.
- How to discretize / encode variable-size environments into fixed-sized observations.
- How to log training metrics and visualize learning curves.
- How a tiny replay buffer and a Q-table work in practice.

Ideal for beginners who know Python and basic NumPy.

---

## Algorithm (intuition + formula)

**Goal:** estimate the action-value function $Q(s, a)$ that gives the expected return when starting in state $s$, taking action $a$, and following the learned policy thereafter.

**Core idea (one-step Q-Learning update):**

1. Take action $a$ in state $s$, observe reward $r$ and next state $s'$.
2. Compute a bootstrap target using the best action at $s'$:
   $$\text{target} = r + \gamma \max_{a'} Q(s', a')$$
   If $s'$ is terminal, we set the bootstrap term to 0.
3. Compute the TD error:
   $$\delta = \text{target} - Q(s, a)$$
4. Apply the incremental update (learning rate $\alpha$):
   $$Q(s,a) \leftarrow Q(s,a) + \alpha \, \delta$$

**Symbols**

- $Q(s,a)$ — current estimate of action value.
- $r$ — immediate reward.
- $\gamma$ — discount factor (0 ≤ γ ≤ 1).
- $\alpha$ — learning rate (0 < α ≤ 1).
- $\delta$ — TD (temporal-difference) error; used as a training "loss" for diagnostics.

**Exploration:** epsilon-greedy — with probability $\varepsilon$ choose a random action, otherwise take $\arg\max_a Q(s,a)$. Often $\varepsilon$ decays from `eps_start` to `eps_end` linearly across episodes.

**Worked numeric example**

- Suppose current $Q(s,a)=1.0$, observed reward $r=+15$ (reached goal), $\gamma=0.97$, and next-state is terminal so bootstrap is 0. Then:
  - target = 15 + 0 = 15
  - δ = 15 − 1 = 14
  - with α = 0.1, new Q = 1 + 0.1 × 14 = 2.4

---

## Observation encoding

Agents expect a fixed-length vector. To support variable-size grids we encode each environment into:

```markdown
obs_dim = 4 + max_size_canvas \* max_size_canvas

- **Head (4 values):** `[ax, ay, gx, gy]` — normalized agent and goal coordinates in [0, 1], where normalization uses `(size - 1)` so endpoints map to 0 and 1.
- **Tail (canvas):** flattened pit mask rasterized to a `max_size_canvas x max_size_canvas` grid (nearest-cell mapping).

This keeps the agent input consistent even when the actual grid size changes.
```

---

## Quick start

Requirements:

- Python 3.8+ (or compatible)
- numpy
- matplotlib (for plotting)
- tkinter (optional; skip if headless)

Install dependencies (example):

```bash
pip install numpy matplotlib
Run examples (from project root):
```

```bash
# run with GUI enabled (if tkinter present)
python -m src.q_learning_gridworld.cli --episodes 600 --render-every 1

# faster but less frequent render
python -m src.q_learning_gridworld.cli --episodes 800 --render-every 10 --sleep 0.01

# vary sizes 4..8 and resample layout every 20 episodes
python -m src.q_learning_gridworld.cli --min-size 4 --max-size 8 --episodes-per-layout 20

# headless mode
python -m src.q_learning_gridworld.cli --no-gui
```

---

## CLI / config summary

Most runtime defaults come from a central config.CFG. Key flags:

```
--episodes: number of episodes to run

--seed: random seed

--render-every: render every N episodes (1 = every episode)

--no-gui: run without Tkinter renderer

--min-size, --max-size: grid sizes to sample from

--episodes-per-layout: how often to resample a new layout

--pit-frac: fraction of cells that become pits

--max-size-canvas: canvas resolution for pit mask encoding

--alpha, --gamma: Q-Learning hyperparameters
```

---

## Project layout (files)

[src/agent/q_learning.py](src/q_learning_gridworld/agent/q_learning.py) — QLearning agent (epsilon-greedy + train_step performing tabular updates).

[src/agent/q_table.py](src/q_learning_gridworld/agent/q_table.py) — tiny Q-table helper (dict keyed by rounded state tuples).

[src/env/gridworld.py](src/q_learning_gridworld/env/gridworld.py) — deterministic GridWorld and StepResult.

[src/env/layout.py](src/q_learning_gridworld/env/layout.py) — LayoutSampler + encode_obs_fixed (observation encoder).

[src/render/renderer.py](src/q_learning_gridworld/render/renderer.py) — minimal Tkinter renderer (optional).

[src/utils/replay.py](src/q_learning_gridworld/utils/replay.py) — tiny FIFO replay buffer wrapper.

[src/utils/metrics.py](src/q_learning_gridworld/utils/metrics.py) — MetricsLogger which writes CSV and PNG diagnostics.

[src/train.py](src/q_learning_gridworld/train.py) — training loop wiring everything together.

[src/cli.py](src/q_learning_gridworld/cli.py) — CLI wrapper that reads flags and calls train.run.

[config.py](config.py) — dataclass-based configuration object (CFG).

---

## Metrics & visualization

MetricsLogger logs episode returns, epsilons, average losses, episode length, and terminal outcomes. Running logger.flush() writes:

metrics.csv — CSV of episode records.

PNG plots: return.png, epsilon.png, loss.png, steps.png, terminals.png.

These help you inspect training progress (e.g., goal reach rate vs episodes).

---

## Tips for experimenting

- Alpha (learning rate): start with 0.05–0.2. If learning is noisy, try smaller α.

- Gamma (discount): close to 1 (e.g., 0.95–0.99) for long-horizon tasks; smaller γ emphasizes immediate rewards.

- Epsilon schedule: keep eps_start high (1.0) and eps_end small (0.05). For small grids you can reduce exploration faster.

- Pit fraction: increasing pit_frac makes the task harder (more traps).

- max_size_canvas: larger canvas gives higher-resolution pit maps but increases observation size — for tiny agents you can keep it close to max_size.

- Replay / batch_size: for tabular Q-Learning, batch_size=1 works best; batching is more a DQN concept.

- Visualize: use render_every=1 for debugging, but for long runs render_every=50 or higher to avoid slowing training.

\
Internals (short)
QTable: converts float observations to hashable tuple keys by rounding (default 4 decimals). get_ref() returns a reference (for in-place updates); get() returns a copy (read-only).

ReplayBuffer: deque-backed buffer; sample() picks uniformly at random.

QLearning.train_step(batch): iterates transitions (s,a,r,s2,done) and applies the classic update. Returns mean absolute TD error for logging.

---

## Troubleshooting

No Tkinter: The renderer will raise if Tkinter is missing — run with --no-gui to avoid errors.

Training stalls / poor learning:

lower α or lower γ; increase episodes; ensure epsilon schedule doesn't decay too fast.

check observation encoding: obs_dim = 4 + max_size_canvas\*\*2 must match encoder and agent expectations.

Plots missing: ensure matplotlib is installed; the logger uses a non-interactive backend by default, so it works headless.

---

## Contributing & License

This project is intentionally small and educational. Contributions that improve documentation, add tests, or clarify examples are very welcome.

Please include clear explanations for any behavioral changes and keep the code simple and well-documented.

Final notes
This repository exists to make Q-Learning understandable. If something about the algorithm, encoding, or code is confusing, open an issue or submit a PR that improves the docs — even a short example calculation helps other beginners a lot.
