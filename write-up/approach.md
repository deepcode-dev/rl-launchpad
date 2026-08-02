# Approach

The submitted agent is a from-scratch clipped PPO implementation with
vectorized GAE, separate 512-256-128 SiLU actor/critic MLPs, a tanh-squashed
Gaussian action distribution (`initial_log_std: -1.0`), running observation
normalization, a single-frame 48-dim actor observation, 123-dim privileged
critic observations, bounded actions, and correct termination/truncation
bootstrapping. The reported 131k-v2 recipe uses eight PPO passes and batch size
16,384. Training uses the stock task reward; final evaluation reports a
separate shared command-tracking metric. Brax is baseline-only code. See
`submission.md` and `ppo/`.
