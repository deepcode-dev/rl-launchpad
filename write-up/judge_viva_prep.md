# Judge walkthrough notes

These notes use the current canonical submission. Do not quote older SB3,
327,680-step, seed-2005, five-frame, or RTX-3070 draft values.

## R1: what was written from scratch

The submitted agent is clipped PPO in `ppo/ppo.py` and
`ppo/train_multi_seed.py`. The actor is a 48-input, 512-256-128 SiLU MLP;
the critic is a separate 123-input, 512-256-128 SiLU MLP. The actor samples a
tanh-squashed Normal action and includes the change-of-variables correction in
the stored log probability. The 12 outputs are normalized joint-target
offsets. The critic is privileged only during training.

GAE keeps the rollout tensor shaped `[time, environment]`:

```text
delta_t = r_t + gamma * V_bootstrap * (1 - terminated_t) - V_t
A_t = delta_t + gamma * lambda * (1 - terminated_t - truncated_t) * A_next
```

At a time-limit truncation, the value of the final terminal observation is
used for the current bootstrap, but the recursion does not cross into the
autoreset episode. Only after this calculation are samples flattened for
minibatch updates. PPO uses `gamma=0.97`, `lambda=0.95`, `clip_ratio=0.2`,
four update passes, batch size 5,120, Huber value loss, and entropy coefficient
0.01. If asked about early stopping, the implementation stops an update epoch
when the Schulman KL estimate exceeds 1.5 times `target_kl`.

## R2: baseline comparison

The baseline is the stock Brax PPO trainer, kept under `cluster/` and
`baselines/`; it is not used to train the custom policy. Both agents use the
Go1 task, 48-dimensional actor state, 12 actions, the same command seed block
20,000–20,049, 50 deterministic episodes per training seed, and the same
native-MuJoCo command-tracking evaluator.

The final custom result is five seeds: 9033, 9006, 9018, 9016, and 8009. It
achieves return 19.752 +/- 0.020 across seed means, LinErr 0.0851 m/s, and
YawErr 0.0676 rad/s. Brax seeds 10, 11, and 12 achieve return 19.821 +/-
0.016, LinErr 0.0668 m/s, and YawErr 0.0454 rad/s. The custom policy uses the
EMA action post-processing it was trained with; the stock Brax policy does
not. That difference is a limitation, not a hidden advantage.

The measured post-hoc EMA ablation is in
`baselines/brax_go1_200m_ema_ablation.json`: the unmodified Brax weights fall
to return 7.241 +/- 0.906 and mean length 380.6 when the EMA is added only at
evaluation. This is why the stock Brax result remains the established
reference and why a genuinely matched EMA baseline would need retraining.

## R3: reproduction

Dependencies are pinned in `pyproject.toml` and `uv.lock`. The canonical
checkpoint metadata records the environment, dimensions, seed, full config,
step count, reward source, and custom training wall time. A judge can run:

```powershell
uv sync --extra dev --locked
uv run pytest -q --basetemp=.pytest_tmp -p no:cacheprovider
uv run python eval/evaluate.py --checkpoint NEW_checkpoints/ppo_v2/ppo_seed9033.pt
```

The evaluator is native MuJoCo and takes substantially less than the training
run. `eval/validate_submission.py --require-demo` checks all committed final
JSON files and the submission limits.

## R4: evaluation protocol

The evaluation seed block is fixed and disjoint from training. Every final
seed has exactly 50 episode returns, lengths, LinErr values, and YawErr values
in JSON. The reported +/- on the aggregate is the standard deviation of
independent training-seed means; the per-seed episode standard deviations are
also retained. The benchmark plot contains measured Brax checkpoint points.
Five canonical custom Slurm logs are committed. Each contains 244 measured
training points from epoch 5 through epoch 1220, so the custom curve is a
five-seed mean with seed standard deviation. Do not claim that stdout logging
provides an every-epoch trace; it is sampled every five epochs.

## R5: environment and reward

The stock task is `Go1JoystickFlatTerrain`. Training uses its `state.reward`
unchanged. The custom wrapper adds vector-slot autoreset, explicit natural
termination versus time-limit truncation, optional history, bounded action
validation, and a 0.7/0.3 EMA filter. The final evaluator reports a separate
shared command-tracking metric so the custom and Brax results are directly
comparable. There is no sim-to-real claim.

## R6: compute honesty and failures

The custom H100 runs average 13,561 seconds for 199,884,800 steps per seed
(range 13,434–13,872 s). The documented Brax 200M reference takes 589.3 s.
The custom implementation crosses the JAX MJX and PyTorch boundary; the
JAX-native reference does not. The 131k-environment experiment reached roughly
19 minutes but learned poorly because it changed the number of optimizer
epochs per fixed sample budget, so it is not presented as a final result.

The main failed versions mixed vector trajectories during GAE, did not reset
completed slots, changed reward semantics, and allowed unbounded actions.
These are recorded in `docs/failures.md`. The answer to “what would you do
next?” is to log three custom checkpoint curves, profile the interop boundary,
run a matched EMA/no-EMA ablation, and test command perturbations.
