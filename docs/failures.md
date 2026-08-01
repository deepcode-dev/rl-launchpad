# Failures and corrections

## 25 July 2026 — invalid vector GAE

The first implementation flattened `[time, environment]` rollouts before GAE. Adjacent entries were different robots, so value targets crossed trajectories and all environments shared one averaged bootstrap value. Existing checkpoints from that implementation are rejected by the evaluator. The corrected implementation computes GAE per environment before flattening and has a regression test with deliberately different trajectories.

## 25 July 2026 — terminal states were never reset

The raw vmapped MJX task does not autoreset. Fallen robots therefore remained terminal during collection. The production wrapper now applies a 1,000-step limit and restores completed slots using the same cached-initial-state pattern as MuJoCo Playground's Brax training wrapper.

## 25 July 2026 — evaluation used a different reward

The original training wrapper rebuilt reward from metric components, multiplied velocity tracking a second time, and omitted the stock nonnegative clip. Negative reported returns were consequently not MuJoCo Playground task returns. Training now uses `state.reward` unchanged. The final native evaluator intentionally reports a separate, shared command-tracking metric for the custom and Brax policies; it must not be described as the native task reward.

## 25 July 2026 — unbounded actions

A raw Normal policy sent arbitrarily large actions to normalized joint targets. It was replaced by a tanh-squashed Gaussian with the change-of-variables log-probability correction. The wrapper also validates shape and clips at the environment boundary as a final safety check.

## 25 July 2026 — synthetic evidence removed

An early plotting script drew hand-authored exponential curves and the draft write-up treated them as measured results. Those claims are invalid. The current plot uses the five pulled Slurm histories, sampled at the points actually printed by training, and measured baseline checkpoint evaluations; no synthetic training trajectory is used.

## Corrected pilot

A 327,680-step, one-seed pilot was used only to validate learning before the final run. With 10 PPO update epochs it reached 339.9 mean episode steps over 50 fixed evaluations, but roughly half of samples were clipped during updates. Repeating the pilot with four update epochs reduced final approximate KL from 0.0507 to 0.0206 and increased mean episode length to 805.3, with 34/50 episodes reaching the horizon. This selected the submission configuration; pilot numbers are not reported as final multi-seed evidence.
