# Judge Guide: Reading, Evaluating, and Driving the Policies

This is an extra guide for judges and reviewers. It is intentionally separate
from the README: the README is the quickstart, while this document explains
how to inspect the evidence, reproduce the reported evaluations, compare the
custom policy with the Brax baseline, and drive both policies interactively.

## 1. Recommended review path

Use this order if you want to understand the project efficiently:

| Order | Read or run | What it answers |
| ---: | --- | --- |
| 1 | [README.md](../README.md) | What the project does, headline results, and quickstart |
| 2 | [write-up/submission.md](../write-up/submission.md) | Full method, reward, comparison, limitations, and claims |
| 3 | [write-up/rl_code_walkthrough.md](../write-up/rl_code_walkthrough.md) | File-by-file explanation of the implementation |
| 4 | [docs/submission-audit.md](submission-audit.md) | Requirement status, validation commands, and disclosed caveats |
| 5 | Run the validation command below | Whether the committed evidence satisfies the submission contract |
| 6 | Run one custom and one Brax evaluation | How the final numbers are produced |
| 7 | Launch a viewer | How to inspect the learned behavior interactively |

The three most important implementation files are:

- [ppo/agent.py](../ppo/agent.py): actor, critic, observation normalization,
  Gaussian policy, and bounded action distribution.
- [ppo/ppo.py](../ppo/ppo.py): GAE, clipped PPO policy loss, value loss,
  entropy bonus, gradient protection, and KL early stopping.
- [ppo/env.py](../ppo/env.py): observation construction, privileged state,
  episode boundaries, action bounds, and EMA action filtering.

The reported recipe is in
[configs/champion_v2.yaml](../configs/champion_v2.yaml). The canonical custom
checkpoints are in NEW_checkpoints/ppo_v2.

## 2. What is being compared

The project contains two different agents:

| Agent | Purpose | Training code | Final seeds |
| --- | --- | --- | ---: |
| Custom PyTorch PPO | The submitted from-scratch implementation | ppo/ and cluster/ | 13039, 13079, 13027 |
| Stock Brax PPO | The established baseline for R2 | cluster/ and baselines/ | 10, 11, 12 |

Both agents are evaluated on the same Go1 task and the same fixed command
episodes:

- Environment: Go1JoystickFlatTerrain.
- Evaluation seed: 20,000.
- Episode seeds: 20,000 through 20,049.
- Episodes per training seed: 50.
- Actions: deterministic at evaluation time.
- Episode horizon: 1,000 control steps.
- Reported metrics: return, linear-velocity error, yaw-rate error, episode
  length, and fall behavior.

The evaluator reports a shared command-tracking metric so the two agents can be
compared directly. Training itself uses the stock MuJoCo Playground
state.reward. These are simulation results; the repository does not claim
sim-to-real transfer.

The committed aggregate results are approximately:

| Agent | Mean return | Linear error | Yaw error | Mean episode length |
| --- | ---: | ---: | ---: | ---: |
| Custom PPO, 3 seeds | 19.795 +/- 0.033 | 0.0722 m/s | 0.0646 rad/s | 1,000 |
| Brax PPO, 3 seeds | 19.821 +/- 0.016 | 0.0668 m/s | 0.0454 rad/s | 1,000 |

The +/- value in this table is the standard deviation across independent
training-seed means. Per-episode values are retained in the JSON artifacts.

## 3. Install and verify the repository

Run these commands from the repository root. The pinned environment is
specified by pyproject.toml and uv.lock.

~~~powershell
uv sync --extra dev --locked
uv run pytest -q --basetemp=.pytest_tmp -p no:cacheprovider
uv run ruff check eval
uv run python eval/validate_submission.py --require-demo --strict
~~~

The validator checks the canonical custom and Brax JSON files, seed labels,
episode counts, fixed episode-seed block, deterministic-action flag, expected
step budget, checkpoint metadata, and the demo contract. It does not retrain
the agents.

If a machine does not have the required GPU, that does not prevent review of
the committed evaluation artifacts or the native evaluation scripts. Training
is a separate, GPU-intensive operation and is not needed to inspect the
submitted checkpoints.

## 4. Evaluate a custom checkpoint

The simplest reproducible custom evaluation is the top reported seed:

~~~powershell
uv run python eval/evaluate.py --checkpoint NEW_checkpoints/ppo_v2/ppo_seed13039.pt --config configs/champion_v2.yaml --eval-seed 20000 --num-episodes 50
~~~

The evaluator prints the aggregate metrics and writes the per-seed JSON next
to the checkpoint:

~~~text
NEW_checkpoints/ppo_v2/ppo_seed13039_eval.json
~~~

To evaluate every canonical custom seed with the identical protocol:

~~~powershell
$customSeeds = 13039, 13079, 13027
foreach ($seed in $customSeeds) {
  uv run python eval/evaluate.py --checkpoint "NEW_checkpoints/ppo_v2/ppo_seed$seed.pt" --config configs/champion_v2.yaml --eval-seed 20000 --num-episodes 50
}
~~~

The aggregate summary is already committed at
NEW_checkpoints/ppo_v2/ppo_v2_eval_summary.json. Compare the newly generated
per-seed files with that summary rather than averaging the three episode
standard deviations together.

## 5. Evaluate the Brax baseline

### 5.1 Evaluate one Brax seed

The Brax checkpoint directories are:

| Training seed | Directory | Final checkpoint |
| ---: | --- | ---: |
| 10 | baselines/brax_go1_200m | 000200540160 |
| 11 | baselines/brax_go1_200m_seed11 | 000200540160 |
| 12 | baselines/brax_go1_200m_seed12 | 000200540160 |

To evaluate the final checkpoint for seed 10:

~~~powershell
uv run python eval/eval_brax_seeds.py --checkpoint-dir baselines/brax_go1_200m --training-seed 10 --output baselines/brax_seed10_recheck.json
~~~

For seeds 11 and 12, substitute the corresponding directory and training seed:

~~~powershell
uv run python eval/eval_brax_seeds.py --checkpoint-dir baselines/brax_go1_200m_seed11 --training-seed 11 --output baselines/brax_seed11_recheck.json
uv run python eval/eval_brax_seeds.py --checkpoint-dir baselines/brax_go1_200m_seed12 --training-seed 12 --output baselines/brax_seed12_recheck.json
~~~

This script finds the latest numbered checkpoint in the selected directory and
evaluates it over 50 fixed episodes. It uses deterministic Brax actions and
does not apply the custom policy's EMA filter by default.

### 5.2 Evaluate all Brax seeds and checkpoints

To reproduce the complete measured baseline curve, including the intermediate
checkpoints:

~~~powershell
uv run python eval/build_brax_evidence.py --num-episodes 50 --output baselines/brax_go1_200m_eval_summary_recheck.json
~~~

This walks all four committed checkpoints for each of seeds 10, 11, and 12.
The committed version of the resulting evidence is
baselines/brax_go1_200m_eval_summary.json.

### 5.3 The EMA comparison is an ablation, not the main baseline

The custom policy was trained with a 0.7/0.3 EMA action filter. The stock Brax
policy was trained without that filter. Therefore the main comparison leaves
the stock Brax policy unfiltered, which is the fairest comparison to the
established stock Brax result.

The optional post-hoc ablation applies the custom EMA to already-trained Brax
weights:

~~~powershell
uv run python eval/build_brax_ema_ablation.py --num-episodes 50 --output baselines/brax_go1_200m_ema_ablation_recheck.json
~~~

Do not describe this ablation as a retrained matched baseline. It answers the
narrow question "what happens if the filter is added after Brax training?"
The repository discloses that a genuinely matched EMA comparison would require
retraining the Brax baseline with the filter included during training.

## 6. Compare the saved evidence

The main evidence files are:

- [NEW_checkpoints/ppo_v2/ppo_v2_eval_summary.json](../NEW_checkpoints/ppo_v2/ppo_v2_eval_summary.json)
- [baselines/brax_go1_200m_eval_summary.json](../baselines/brax_go1_200m_eval_summary.json)
- [baselines/brax_go1_200m_ema_ablation.json](../baselines/brax_go1_200m_ema_ablation.json)
- [write-up/benchmark_comparison.png](../write-up/benchmark_comparison.png)

For a quick structured view:

~~~powershell
Get-Content NEW_checkpoints/ppo_v2/ppo_v2_eval_summary.json
Get-Content baselines/brax_go1_200m_eval_summary.json
Get-Content baselines/brax_go1_200m_ema_ablation.json
~~~

The benchmark plot is generated from measured JSON evidence. If you regenerate
it after changing any input path:

~~~powershell
uv run python eval/plot_benchmark.py
~~~

Use the validator after regenerating evidence:

~~~powershell
uv run python eval/validate_submission.py --require-demo --strict
~~~

## 7. Drive the custom policy interactively

The custom viewer uses native MuJoCo for responsive visual inspection:

~~~powershell
uv run python eval/view_native_v2.py NEW_checkpoints/ppo_v2/ppo_seed13039.pt --config configs/champion_v2.yaml --command 0.8 0.0 0.0
~~~

The initial command is [forward velocity, sideways velocity, yaw rate].
Keyboard controls:

| Key | Effect |
| --- | --- |
| W or Up | Increase forward velocity |
| S or Down | Decrease forward velocity |
| A or Left | Increase yaw rate |
| D or Right | Decrease yaw rate |
| Q | Increase sideways velocity |
| E | Decrease sideways velocity |
| 1, 2, 3, 4 | Set forward speed to 0.5, 1.0, 1.5, or 2.0 m/s |
| Space | Set all commands to zero |

The viewer uses deterministic actor actions and the same 0.7/0.3 EMA action
smoothing used by the submitted custom policy. It is a visual control demo,
not a replacement for the fixed 50-episode benchmark.

## 8. Drive a Brax policy interactively

For the final seed-10 Brax checkpoint:

~~~powershell
uv run python eval/view_brax_native.py baselines/brax_go1_200m/000200540160 --command 0.8 0.0 0.0
~~~

For seeds 11 and 12, use:

~~~powershell
uv run python eval/view_brax_native.py baselines/brax_go1_200m_seed11/000200540160 --command 0.8 0.0 0.0
uv run python eval/view_brax_native.py baselines/brax_go1_200m_seed12/000200540160 --command 0.8 0.0 0.0
~~~

The Brax viewer controls are intentionally simpler:

| Key | Effect |
| --- | --- |
| W or Up | Increase forward velocity |
| S or Down | Decrease forward velocity |
| A or Left | Increase yaw rate |
| D or Right | Decrease yaw rate |
| Space | Set all commands to zero |

The stock Brax viewer does not apply the custom EMA filter. This makes the
interactive behavior consistent with the main unfiltered Brax benchmark.

Both viewers are simulation-only native MuJoCo previews. They do not connect
to, send commands to, or control a physical robot.

## 9. How to interpret the evaluation

### Return

Higher command-tracking return is better. The custom and Brax final summaries
use the same shared command-tracking evaluator, not the raw training reward
scale as a substitute.

### Linear-velocity error

This is the average distance between the commanded planar velocity and the
robot's measured planar velocity. Lower is better.

### Yaw-rate error

This is the average absolute difference between the commanded yaw rate and the
robot's measured yaw rate. Lower is better.

### Episode length and falls

An episode that reaches 1,000 steps without falling is evidence of stable
simulation behavior for that command sequence. It is not evidence of
real-world hardware robustness.

### Seed variation

Do not judge a policy from one lucky rollout. The canonical comparison uses
three independent training seeds for each agent and the same 50 evaluation
episodes for each seed. The aggregate +/- is variation between training-seed
means.

## 10. Important limitations and disclosures

- The custom and stock Brax policies use the same benchmark task and evaluator,
  but the custom policy includes EMA filtering during training and evaluation,
  while stock Brax does not.
- The post-hoc Brax EMA file is an ablation; it is not a retrained matched
  baseline.
- The custom trainer crosses a JAX/MJX and PyTorch boundary, while the Brax
  reference is JAX-native. Reported wall times should be read with that
  implementation difference in mind.
- The final results are simulation results. No sim-to-real claim is made.
- The documented Brax wall time is 589.3 seconds, but its raw training log is
  not committed. Custom checkpoint metadata contains custom training wall time.

For the full list of known failures and fixes, see
[docs/failures.md](failures.md). For compute accounting, see
[docs/compute-accounting.md](compute-accounting.md).
