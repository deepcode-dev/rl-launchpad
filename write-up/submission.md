# From-Scratch PPO for Command-Conditioned Go1 Locomotion

## Problem

Can a small, auditable PPO implementation written from scratch learn the stock MuJoCo Playground `Go1JoystickFlatTerrain` task? I evaluate that question against the established Brax PPO implementation on the same 48-dimensional actor observation, 12-dimensional normalized action space, 1,000-step episode limit, command distribution, deterministic evaluator, and held-out episode seeds.

## Approach and architecture

```mermaid
graph LR
  O[48-dim actor observation] --> A[MLP 512-256-128, SiLU]
  A --> D[tanh-squashed Gaussian]
  D --> F[0.7/0.3 EMA action filter]
  F --> U[12 joint-target offsets]
  P[123-dim privileged training observation] --> C[MLP 512-256-128, SiLU]
  C --> V[Value estimate]
```

The actor and critic are separate PyTorch networks. The actor sees only policy-available state; the critic additionally sees privileged simulator state during training. PPO, minibatch updates, clipped ratios, Adam, and GAE are implemented in `ppo/ppo.py` and `ppo/train_multi_seed.py`; no policy is trained with SB3, RSL-RL, RL-Games, SKRL, CleanRL, or a Brax trainer. Brax is a baseline only.

Rollouts remain shaped `[time, environment]` while GAE is computed, then are flattened for updates. The policy uses `gamma=0.97`, `lambda=0.95`, four PPO passes, batch size 5,120, `clip_ratio=0.2`, and a fixed entropy coefficient of 0.01. Checkpoints include the running observation statistics and the complete YAML configuration.

## Reward and environment contract

Training uses MuJoCo Playground's `state.reward` unchanged. Relative to the stock task, the custom training wrapper adds only vector-slot autoreset, explicit termination versus time-limit handling, a configurable observation-history wrapper, bounded normalized actions, and the 0.7/0.3 EMA post-processing used by the submitted policy. The evaluator uses native MuJoCo C physics and a shared command-tracking return, `max(0, 1 - ||v_xy-v_cmd||²) * control_dt`, plus linear-velocity and yaw-rate errors. This metric is deliberately stated separately from the stock training reward. The Brax baseline uses the same evaluator and observation/action dimensions, but its stock policy output has no custom EMA; that control-postprocessing difference is a declared limitation of this comparison.

## Evidence

All custom checkpoints were trained for 199,884,800 environment steps per seed and evaluated deterministically for 50 episodes using the fixed block 20,000–20,049. The table's `±` values on individual rows are within-episode standard deviations; the aggregate `±` is the standard deviation of the independent seed means.

| Agent | Seeds | Wall time/seed | Return | LinErr | YawErr | Length |
|---|---:|---:|---:|---:|---:|---:|
| Custom PyTorch PPO | 9033, 9006, 9018, 9016, 8009 | 13,434–13,872 s; mean 13,561 s | **19.752 ± 0.020** | **0.0851 m/s** | **0.0676 rad/s** | 1,000 |
| Brax PPO baseline | 10, 11, 12 | documented 589.3 s | **19.821 ± 0.016** | **0.0668 m/s** | **0.0454 rad/s** | 1,000 |

The custom agent reaches the horizon on every reported episode and is close to, but below, the faster Brax reference. The full per-episode JSON files, checkpoint metadata, seeds, and measured plot are present in the repository. I also ran a matched-postprocessing ablation: applying the custom EMA after training to stock Brax weights reduced Brax to **7.241 ± 0.906** return and **380.6** steps, so it is not substituted for the established reference. A fully controlled EMA comparison would require retraining Brax with the filter inside its environment. The five canonical custom Slurm logs provide 244 measured points per seed from epoch 5 through epoch 1220; the plot reports their mean ± seed standard deviation. The final benchmark statistics remain the primary multi-seed claim.

## Demo and constraints

[`demo.mp4`](demo.mp4) is an under-90-second narrated walkthrough using only the reported seed 9033 checkpoint; it shows three deterministic evaluation-style command episodes and `demo-narration.txt` is its transcript. It shows the architecture, failed approaches, the measured final comparison, and the limitation above. Training ran on H100 NVL hardware; the custom implementation transfers between JAX MJX and PyTorch, while Brax keeps simulation and learning in JAX. That bridge explains the large latency gap: the recorded custom mean is about 13,561 s for 200M steps versus about 589 s for the documented Brax run. The video is simulation-only and makes no sim-to-real claim.

## Honesty and trajectory

The first implementation mixed vector trajectories during GAE, failed to autoreset completed slots, changed reward semantics, and sent unbounded Gaussian actions. Those failures produced poor policies and are retained in `docs/failures.md`. Ten PPO passes also caused excessive clipping in the pilot; four passes were retained. The 131k-environment experiment reduced wall time to roughly 19 minutes but produced an awful policy because it reduced the number of optimizer epochs per fixed sample budget; it is not the submitted result.

The next two-week plan is: (1) profile and remove the JAX/PyTorch boundary while preserving the from-scratch PPO loss; (2) report a controlled EMA/no-EMA baseline ablation; (3) add command perturbations and push robustness; and (4) add a hardware-in-the-loop check before making any deployment claim.
