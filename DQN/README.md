# dqn_gridworld_varsize

Minimal NumPy DQN + Tkinter GridWorld with variable map layouts.

Features

- 2D GridWorld with resamplable layouts (variable size, start, goal, pits).
- Fixed-width observation encoder (works across sizes).
- NumPy-only DQN (MLP + ReLU), replay buffer, target network.
- Epsilon-greedy with linear decay across the entire training run.
- Live Tkinter rendering (can be disabled for headless training).

Quick examples

```bash
python -m src.dqn_gridworld.cli --episodes 600 --render-every 1
python -m src.dqn_gridworld.cli --episodes 800 --render-every 10 --sleep 0.01
# Vary sizes 4..8 and resample layout every 20 episodes:
python -m src.dqn_gridworld.cli --min-size 4 --max-size 8 --episodes-per-layout 20
```
