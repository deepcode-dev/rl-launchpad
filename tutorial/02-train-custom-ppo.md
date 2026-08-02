# 02 - Train the from-scratch PyTorch PPO

The reported recipe is the 131k-v2 configuration. It keeps the PPO loss and
GAE implementation in `ppo/`, while using 131,072 parallel environments to
finish a 199,229,440-step seed in about 23 minutes on the reported H100 NVL
jobs.

## Local CLI

```powershell
# Train the three reported seeds from the canonical config
uv run python ppo/train_multi_seed.py --config configs/cluster_131k_v2.yaml

# Train or resume one seed
uv run python ppo/train_multi_seed.py --config configs/cluster_131k_v2.yaml --seed 13039
uv run python ppo/train_multi_seed.py --config configs/cluster_131k_v2.yaml --seed 13039 --resume
```

The trainer rejects inconsistent `epochs * steps_per_epoch` budgets and
requires `steps_per_epoch` to be divisible by `num_envs`.

## Reported config

| Key | Value | Meaning |
|---|---:|---|
| `env_name` | `Go1JoystickFlatTerrain` | MuJoCo Playground task |
| `num_envs` | `131072` | parallel MJX environments |
| `history_len` | `1` | single-frame observation |
| `episode_length` | `1000` | truncation horizon |
| `steps_per_epoch` | `262144` | two rollout steps per environment |
| `epochs` | `760` | 199,229,440 steps per seed |
| `train_iters` | `8` | PPO passes per rollout |
| `batch_size` | `16384` | minibatch size |
| `gamma` / `lam` | `0.97` / `0.95` | GAE parameters |
| `clip_ratio` | `0.2` | PPO clipping |
| `hidden_sizes` | `[512, 256, 128]` | actor and critic widths |

## Outputs

For each seed, the checkpoint directory contains the model, its metadata
sidecar, and the measured history. The committed final evidence is kept under
`NEW_checkpoints/ppo_v2/`; cluster jobs write to `checkpoints/ppo_v2/` first.

## Slurm

```bash
sbatch --partition=gpu --time=01:00:00 --gres=gpu:h100-47:1 \
  --export=ALL,CONFIG_PATH=configs/cluster_131k_v2.yaml,TRAIN_SEED=13039 \
  cluster/train_go1.slurm
```

Submit one job per seed so each run has its own GPU:

```bash
for s in 13039 13079 13027; do
  sbatch --partition=gpu --time=01:00:00 --gres=gpu:h100-47:1 \
    --export=ALL,CONFIG_PATH=configs/cluster_131k_v2.yaml,TRAIN_SEED=$s \
    cluster/train_go1.slurm
done
```

The Slurm script sets the JAX cache and GPU memory variables. Pull completed
checkpoint files to the local repository before evaluating or committing them.
