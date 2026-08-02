# RL From Scratch: Go1 Locomotion

A from-scratch PyTorch PPO agent for `Go1JoystickFlatTerrain` in MuJoCo
Playground, built for the Launchpad 2026 Griffin Labs RL track. Rule R2 uses
the official Brax PPO implementation as the baseline.

The agent uses 131,072 parallel environments, separate 512-256-128 SiLU
actor/critic MLPs, a tanh-squashed Gaussian policy, running observation
normalization, a 0.7/0.3 EMA action filter, a 123-dimensional privileged
critic, and GAE with `gamma=0.97`, `lambda=0.95`.

Across three independent 131k-v2 seeds, the custom agent reaches **19.795 ±
0.033** mean command-tracking return, **0.0722 m/s** LinErr, **0.0646 rad/s**
YawErr, and 1,000-step episodes. The three-seed 200M-step Brax reference
reaches **19.821 ± 0.016**, **0.0668 m/s**, and **0.0454 rad/s**. These are
deterministic native-MuJoCo evaluations using the shared command-tracking
metric; training itself uses the stock `state.reward`.

## Judge quickstart

```powershell
# Install pinned dependencies
uv sync --extra dev --locked

# Run tests
uv run pytest -q --basetemp=.pytest_tmp -p no:cacheprovider

# Evaluate the top reported checkpoint over 50 fixed episodes
uv run python eval/evaluate.py --checkpoint NEW_checkpoints/ppo_v2/ppo_seed13039.pt

# Drive the policy live in native MuJoCo (W/A/S/D or arrow keys, Q/E, 1/2/3/4)
uv run python eval/view_native_v2.py NEW_checkpoints/ppo_v2/ppo_seed13039.pt
```

## Benchmark results

Every checkpoint is evaluated over 50 fixed, disjoint benchmark episodes
(`eval_seed=20000`, episodes 20,000-20,049).

| Agent / model | Environment steps | Wall time | LinErr | YawErr | Mean return | Fall rate |
|---|---:|---:|---:|---:|---:|---:|
| **Brax PPO baseline (seeds 10/11/12)** | 200,000,000 | documented 589.3 s | **0.0668 m/s** | **0.0454 rad/s** | **19.821 ± 0.016** | **0.00%** |
| **Custom PyTorch PPO (seeds 13039/13079/13027)** | 199,229,440 | 1,392.6-1,424.3 s; mean 1,407.0 s | **0.0722 m/s** | **0.0646 rad/s** | **19.795 ± 0.033** | **0.00%** |

The aggregate ± is the standard deviation of independent seed means. The
evaluator reports the shared command-tracking return; training uses the stock
task reward.

## Interactive keyboard steering

```powershell
# Top reported policy
python eval/view_native_v2.py NEW_checkpoints/ppo_v2/ppo_seed13039.pt
```

- **W / Up Arrow**: increase forward velocity `vx`
- **S / Down Arrow**: decrease forward velocity `vx`
- **A / Left Arrow**: increase yaw rate
- **D / Right Arrow**: decrease yaw rate
- **Q / E**: strafe left/right (`vy`)
- **1, 2, 3, 4**: forward-speed presets `0.5`, `1.0`, `1.5`, `2.0 m/s`
- **Space**: stop all commands

## Fast cluster training

The reported recipe is `configs/cluster_131k_v2.yaml`:

```bash
sbatch --partition=gpu --time=01:00:00 --gres=gpu:h100-47:1 \
  --export=ALL,CONFIG_PATH=configs/cluster_131k_v2.yaml,TRAIN_SEED=13039 \
  cluster/train_go1.slurm
```

It uses 131,072 environments, 760 epochs, eight PPO passes, batch size
16,384, and 199,229,440 steps per seed. The selected runs completed in about
23 minutes on H100 NVL hardware.

## Correctness contracts

- GAE operates on `[time, environment]` tensors without crossing streams.
- Completed vector slots restore cached randomized initial states.
- Training uses unmodified MuJoCo Playground `state.reward`.
- Custom and Brax evaluation share the deterministic command-tracking metric.
- The policy is tanh-squashed and obeys the final `[-1, 1]` action bound.
- Running observation moments are stored inside every checkpoint.

See [`write-up/submission.md`](write-up/submission.md) for the full write-up,
[`write-up/demo.mp4`](write-up/demo.mp4) for the captioned demo, and
[`docs/submission-audit.md`](docs/submission-audit.md) for the repository
review guide and validation commands.
