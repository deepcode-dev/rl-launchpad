# Approach

The submitted agent is a from-scratch clipped PPO implementation with vectorized GAE, separate actor/critic MLPs (512-256-128), a tanh-squashed Gaussian action distribution with tuned initial std (`initial_log_std: -1.9`), running observation normalization, single-frame 48-dim observation (`history_len: 1`), 123-dim privileged critic observations, native reward, bounded actions, and correct termination/truncation bootstrapping. Brax and SB3 are used as published baselines. See `submission.md` and the implementation under `ppo/`.
