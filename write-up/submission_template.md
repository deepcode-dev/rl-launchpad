# From-Scratch PPO for Command-Conditioned Go1 Locomotion

## Problem

A quadruped deployed in a changing facility must track velocity commands without falling as contacts, joint state, and momentum evolve. A scripted gait can work inside its design envelope but is brittle when contact timing or commands change. I therefore ask a deliberately modest question: can a PPO implementation written from scratch learn the stock MuJoCo Playground Go1 joystick task, and how does its result compare with an established PPO implementation under one evaluation contract?

Success is defined before the final run: use the unmodified task reward; evaluate at least three training seeds for 50 fixed, disjoint episodes each; compare against Stable-Baselines3 with the same vector batch, rollout size, observations, actions, reward, step budget, and evaluation seeds; report environment steps and wall time; and show failure rather than selecting a highlight rollout.

## Approach

The submitted trainer is a PyTorch implementation of clipped PPO. The actor and critic are separate two-layer, 256-unit tanh MLPs. The actor produces a tanh-squashed Gaussian over 12 normalized joint-target offsets. Its log probability includes the squash Jacobian. A shared five-frame history turns the 48-value task state into a 240-value policy observation, and running observation moments are stored in the checkpoint.

Rollouts remain shaped `[time, environment]` while generalized advantage estimation runs, then flatten for minibatch updates. Natural termination prevents bootstrapping. A 1,000-step time-limit truncation bootstraps from the terminal observation but stops advantage propagation into the autoreset episode. Completed vector slots restore cached randomized initial states. Reward is exactly MuJoCo Playground's `state.reward`; no scale is changed.

I chose PPO because the task has continuous actions, dense reward, and fast parallel simulation. SAC or TD3 could reuse samples more efficiently, but would add replay-buffer and off-policy tuning complexity. I kept the critic on policy-visible state instead of privileged simulator state: this sacrifices a reference-trainer advantage but makes the implementation and deployment contract easier to explain. SB3 is baseline-only, as required by Rule R1.

## Evidence

Training seeds were **@@SEEDS@@**, with **@@STEPS@@ environment steps per seed**. Every checkpoint was evaluated deterministically on the same **@@EPISODES@@ untouched seeds starting at 20,000**.

| Agent | Mean return ± across-seed SD | Mean episode length | Mean training wall time/seed |
|---|---:|---:|---:|
| Custom PPO | **@@CUSTOM_MEAN@@ ± @@CUSTOM_STD@@** | **@@CUSTOM_LENGTH@@** | **@@CUSTOM_TIME@@ s** |
| SB3 PPO baseline | @@BASELINE_MEAN@@ ± @@BASELINE_STD@@ | @@BASELINE_LENGTH@@ | @@BASELINE_TIME@@ s |

![Measured training and evaluation comparison](benchmark_comparison.png)

The figure is built only from committed JSON histories. The plot command fails if results are missing or protocols differ. Error bars on evaluation are standard deviation across independently trained seed means; per-episode returns and lengths remain in the JSON artifacts.

## Constraints

The machine used an RTX 3070 for PyTorch. Native Windows JAX exposed MJX on CPU, so every control step crossed the JAX/PyTorch host boundary; wall time therefore reflects this mixed-device implementation rather than an optimized all-GPU trainer. The stock reference configuration uses far more simulation steps and a privileged critic. This project instead measures what a small, auditable implementation achieves within the available compute.

SB3 uses its standard two-layer, 64-unit `MlpPolicy`, whereas the custom actor and critic use 256-unit layers. The baseline controls the environment contract, rollout size, update count, step budget, and evaluation episodes; it is not a controlled architecture ablation.

The live viewer is display-only and runs at the task's 50 Hz control interval. Mouse forces applied to its copied `MjData` do not perturb the MJX policy state, so I do not claim push recovery or sim-to-real robustness.

## Honesty & Trajectory

The first checkpoints were invalid. I had flattened vector rollouts before GAE, never reset fallen slots, reconstructed a different reward, and sampled unbounded actions. A synthetic draft chart also presented hand-authored curves as evidence. I removed it, documented each failure, added regression tests, and made evaluation reject every legacy checkpoint.

A corrected one-seed pilot improved survival but showed excessive PPO clipping, motivating four rather than ten update epochs. Extending that setting to 2,457,600 steps per seed then degraded the 10,000-series validation result, so I archived the run, early-stopped at 327,680 steps, and moved final evaluation to the untouched 20,000-series block. Remaining failures are visible in the full episode-length distribution: reaching the time limit is not the same as tracking every command well. With two more weeks I would add command-conditioned success metrics, a privileged training-only critic ablation, Linux JAX-CUDA profiling, and controlled perturbations that affect the actual MJX state.
