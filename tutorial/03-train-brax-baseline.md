# 03 · Train the Brax PPO baseline (Rule R2)

The baseline uses the **stock Brax PPO** (from `brax.training.agents.ppo`) on the
same MuJoCo Playground task with a `tanh_normal` distribution and an asymmetric
critic (`state` for the policy, `privileged_state` for the value function).

| Entry point | Purpose |
| --- | --- |
| `cluster/train_brax_go1.py --config configs/brax_go1_200m.yaml` | Train one Brax PPO run and save step checkpoints |
| `cluster/train_brax_go1.slurm` | The same, but submitted to Slurm |

## CLI reference

```bash
# Local (not recommended on the login node; training belongs on Slurm)
uv run python cluster/train_brax_go1.py --config configs/brax_go1_200m.yaml
```

The trainer prints one JSON line per eval point (`step`, `elapsed_seconds`,
metrics) and saves Brax step checkpoints into `config["checkpoint_dir"]`.

## Brax config files

- `configs/brax_go1_20m.yaml` — 20M-step smoke gate.
- `configs/brax_go1_200m.yaml` — the reported 200M baseline.

Key fields:

| Key | Value | Meaning |
| --- | --- | --- |
| `num_timesteps` | `200000000` | total env steps |
| `num_envs` | `8192` | parallel environments |
| `hidden_sizes` | `[512, 256, 128]` | policy/value MLP widths |
| `learning_rate` / `entropy_cost` | `3e-4` / `0.01` | Brax PPO defaults |
| `discounting` / `gae_lambda` | `0.97` / `0.95` | must mirror the custom trainer |
| `unroll_length` | `20` | rollout length per update |
| `checkpoint_dir` | `/home/r/ravideep/.../brax_go1_200m` | where step dirs are written |
| `seed` | `10` | training seed (set a new one per run) |

> To train additional seeds, make a copy of the config with a **new `seed`** and a
> **new `checkpoint_dir`** so that runs never overwrite each other.

## Slurm

**Important:** `cluster/train_brax_go1.slurm` defaults to `--gres=gpu:a100-80:1`,
which never queues on this cluster. Always override the GPU type:

```bash
# 200M baseline, seed 10
sbatch --gres=gpu:a100-40:1 --time=03:00:00 \
  --export=ALL,CONFIG_PATH=configs/brax_go1_200m.yaml \
  cluster/train_brax_go1.slurm

# another seed: dedicated config + dedicated checkpoint_dir
sbatch --gres=gpu:a100-40:1 --time=03:00:00 \
  --export=ALL,CONFIG_PATH=configs/brax_go1_200m_seed11.yaml \
  cluster/train_brax_go1.slurm
```

Measured reference on `a100-40`: ~450k SPS; 200M steps ≈ 500–660 s per seed.

## Where the committed baselines live

The evaluated baselines are committed under `baselines/` (Brax checkpoints are
small, ~6 MB per seed):

- `baselines/brax_go1_200m/` — seed 10
- `baselines/brax_go1_200m_seed11/` — seed 11
- `baselines/brax_go1_200m_seed12/` — seed 12

Each contains step subdirs (`000200540160` is the final 200M checkpoint) plus
`brax_200m_seed<NN>_eval.json` with the 50-episode eval results.
