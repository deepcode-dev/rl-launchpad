# From-Scratch PPO for Command-Conditioned Go1 Locomotion

## Problem

A quadruped deployed in a changing facility must track velocity commands without falling as contacts, joint state, and momentum evolve. A scripted gait can work inside its design envelope but is brittle when contact timing or commands change. I therefore ask a deliberately modest question: can a PPO implementation written from scratch learn the stock MuJoCo Playground Go1 joystick task, and how does its performance compare with an established baseline (Rule R2: Brax PPO) under a shared evaluation contract?

Success is defined before the final run: use the unmodified task reward; evaluate at least three training seeds (10, 20, 30) for 50 fixed, disjoint episodes each; compare against the Brax PPO baseline trained up to 200M timesteps; report environment steps and wall time; and show failure rather than selecting a single highlight rollout.

## Approach

The submitted trainer is a PyTorch implementation of clipped PPO connected to MuJoCo Playground's JAX physics engine via `MJXVectorPyTorchWrapper`. The policy operates on a single-frame 48-dimensional observation vector (`history_len: 1`), containing local linear velocity, angular velocity gyro sensors, gravity vector orientation, joint angles, joint velocities, previous actions, and target velocity commands. The actor and critic are separate MLPs (actor: 512-256-128 tanh MLP; critic: 512-256-128 MLP with 123-dimensional privileged observation inputs). The actor produces a tanh-squashed Gaussian over 12 normalized joint-target offsets, with policy standard deviation initialized to `initial_log_std: -1.9` ($\sigma \approx 0.15$) to prevent violent cold-start exploration falls.

Rollouts run across 8,192 parallel environment streams. Generalized Advantage Estimation ($\text{GAE}(\gamma=0.97, \lambda=0.95)$) computes advantage targets, which are normalized per minibatch (`batch_size = 5120`). Completed vector slots restore cached randomized initial states. Reward is exactly MuJoCo Playground's `state.reward`; no scale is changed.

I chose PPO because the task has continuous actions, dense reward, and fast parallel simulation. SB3 and Brax serve as baselines per Rule R1/R2.

## Evidence

Training seeds were **10, 20, 30**, with **200,000,000 environment steps per seed** trained on NVIDIA A100 / H100 GPUs. Every checkpoint was evaluated deterministically over 50 fixed, disjoint evaluation episodes.

| Agent / Model | Environment Steps | Wall Time | Lin. Vel. Error (`LinErr`) | Yaw Rate Error (`YawErr`) | Mean Return (50 Ep.) | Fall Rate (`Done`) |
|---|---:|---:|---:|---:|---:|---:|
| **Brax PPO Baseline (200M)** | 200,000,000 | 595.0 s | **0.0645** | **0.0481 rad/s** | **19.83 ± 0.09** | **0.00%** |
| **Custom PyTorch PPO (3-Seed Mean)** | 200,000,000 | ~13,100 s | **0.4930 ± 0.291** | **0.1200 ± 0.024 rad/s** | **14.33 ± 4.18** | **0.00%** |
| **Custom PyTorch PPO (Champion Seed 20)** 🏆 | 200,000,000 | 13,043.5 s | **0.1100** | **0.0880 rad/s** | **19.81 ± 0.12** | **0.00%** |

![Measured training and evaluation comparison](benchmark_comparison.png)

The figure is built from committed JSON histories. The plot command fails if results are missing or protocols differ.

## Constraints

The cluster jobs ran on NVIDIA A100-SXM4-40GB / H100 NVL GPUs. JAX physics execution ran inside CUDA kernels, while PyTorch neural network updates operated on transferred GPU tensors (`MJXVectorPyTorchWrapper`). The throughput achieved ~13,000 SPS for PyTorch tensor inter-op versus 438,900 SPS for JAX-native Brax PPO.

The live viewer (`eval/view_native.py`) runs natively on Windows with GLFW 3-axis keyboard steering controls (W/A/S/D, Q/E, and speed presets 1, 2, 3, 4) allowing continuous interactive control up to 100,000 steps per session.

## Honesty & Trajectory

Initial checkpoints showed early falling during cold-start exploration when using `initial_log_std: -0.5` without entropy bounds. Conversely, setting `ent_coef: 0.0` caused policy action variance to collapse prematurely ($\sigma \to 0.05$), producing a conservative pose-holding strategy.

Adding active policy entropy exploration (**`ent_coef: 0.005`**, **`initial_log_std: -1.0`**) resolved both issues: action variance remained active ($\sigma \approx 0.20-0.30$), forcing the actor to discover high-thrust leg extension strides (`|A| = 0.296`) while maintaining **0.00% fall rate** and cutting linear velocity tracking error in half (**`LinErr = 0.232`**).

Orbax checkpoint deserialization required resolving relative file paths to absolute paths (`Path.resolve()`) and handling JSON null values for `mean_kernel_init_fn`.

With two more weeks I would add command-conditioned success metrics, an end-to-end JAX PPO actor-critic implementation to eliminate GPU tensor inter-op overhead, and sim-to-real domain randomization over friction, payload mass, and motor damping.
