# Environment contract

- Task: MuJoCo Playground `Go1JoystickFlatTerrain`
- Robot: Unitree Go1 quadruped
- Control interval: 0.02 s (50 Hz)
- Native observation: 48 values: local linear velocity, angular velocity, projected gravity, 12 joint offsets, 12 joint velocities, previous 12 actions, and a 3D velocity command
- Policy observation: single native frame (48 values with `history_len: 1`); also supports multi-frame history when configured
- Action: 12 normalized joint-target offsets, bounded to `[-1, 1]`
- Reward: unmodified MuJoCo Playground `state.reward`; the task clips each step reward to `[0, 10000]`
- Natural termination: robot up-vector indicates a fall
- Time limit: 1,000 control steps (20 simulated seconds)
- Autoreset: each vector slot restores its cached randomized initial state while returning the terminal transition's flags and final policy observation
- Physics implementation: MuJoCo MJX's JAX backend is selected explicitly; no platform-dependent default backend is used

The wrapper is implemented in `ppo/env.py`. Environment changes relative to stock are limited to the training wrapper's episode/autoreset mechanics and configurable observation history. Reward scales are not modified.
