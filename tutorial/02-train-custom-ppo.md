# 02 · Train the from-scratch PyTorch PPO

Two entry points live in `ppo/`:

| Script | Purpose | CLI |
| --- | --- | --- |
| `ppo/train_multi_seed.py` | **Use this.** Multi-seed training, per-seed checkpoints, auto-resume, progress JSON | `--config`, `--seed`, `--resume` |
| `ppo/train.py` | Single-seed reference trainer | `--config` |

## CLI reference

```bash
# Train every seed listed in the config
uv run python ppo/train_multi_seed.py --config configs/champion_v2.yaml

# Train only one seed (overrides the config's seed list)
uv run python ppo/train_multi_seed.py --config configs/champion_v2.yaml --seed 9033

# Resume the given seed from its existing ppo_seed<seed>.pt checkpoint
uv run python ppo/train_multi_seed.py --config configs/champion_v2.yaml --seed 9033 --resume
```

On the cluster the same script is launched by `cluster/train_go1.slurm`, which
passes config/seed/resume through environment variables (see below).

## Config files

All hyperparameters live in YAML under `configs/`. The two you will actually use:

- `configs/champion_v2.yaml` — the **champion recipe** (200M steps/seed, the numbers
  reported in the submission).
- `configs/default.yaml` — the submission-style config (3 seeds, 200M each).

Key fields (with the champion values):

| Key | Value | Meaning |
| --- | --- | --- |
| `env_name` | `Go1JoystickFlatTerrain` | MuJoCo Playground task |
| `num_envs` | `8192` | parallel vector environments (MJX) |
| `history_len` | `1` | single-frame observation stack |
| `episode_length` | `1000` | steps before truncation |
| `steps_per_epoch` | `163840` | rollout steps per epoch (must be divisible by `num_envs`) |
| `epochs` | `1220` | `163840 * 1220 = 199,884,800` ≈ 200M steps |
| `total_timesteps_per_seed` | `199884800` | must equal `epochs * steps_per_epoch` (validated) |
| `gamma` / `lam` | `0.97` / `0.95` | GAE discount / trace-decay |
| `clip_ratio` | `0.2` | PPO clip ε |
| `target_kl` | `0.02` | early-stop at `1.5 × 0.02` |
| `pi_lr` | `0.0003` | learning rate |
| `train_iters` | `4` | optimizer passes per epoch |
| `batch_size` | `5120` | minibatch size |
| `hidden_sizes` | `[512, 256, 128]` | actor & critic MLP widths |
| `initial_log_std` | `-1.0` | tanh-squashed Gaussian init |
| `ent_coef` / `ent_coef_final` | `0.01` | fixed policy entropy bonus |
| `checkpoint_dir` | `checkpoints/ppo_v2` | where `ppo_seed<seed>.pt` is written |

> `train_multi_seed.py` exits with an error if `total_timesteps_per_seed !=
> epochs * steps_per_epoch` and if `steps_per_epoch % num_envs != 0`. If you change
> one field, keep the others consistent.

## What the trainer writes

For each seed, in `checkpoint_dir`:

- `ppo_seed<seed>.pt` — raw `state_dict` (policy + critic + running-norm stats),
  saved every 10 epochs and at completion.
- `ppo_seed<seed>.pt.meta.json` — contract metadata (algorithm, dims, config,
  wall time, total steps). Evaluators read this to validate the checkpoint.
- `ppo_multi_seed_results.json` — per-epoch training history for all seeds,
  rewritten after every completed seed so an interrupted run keeps its data.

## Slurm (custom trainer)

`cluster/train_go1.slurm` is controlled entirely by environment variables, so you
can override them per-submission with `--export`:

```bash
# 1) smoke test (default CONFIG_PATH=configs/cluster_smoke.yaml)
sbatch cluster/train_go1.slurm

# 2) one 20M-step seed
sbatch --export=ALL,CONFIG_PATH=configs/cluster_20m.yaml,TRAIN_SEED=10 \
  cluster/train_go1.slurm

# 3) one 200M-step seed, longer allocation
sbatch --time=03:00:00 \
  --export=ALL,CONFIG_PATH=configs/cluster_200m.yaml,TRAIN_SEED=10 \
  cluster/train_go1.slurm

# 4) resume a seed (RESUME=true)
sbatch --export=ALL,CONFIG_PATH=configs/cluster_200m.yaml,TRAIN_SEED=10,RESUME=true \
  cluster/train_go1.slurm

# 5) multi-seed fan-out: one job per seed (TRAIN_SEED must be "none" for the
#    config's own seed list to be used, otherwise only that seed is trained)
for s in 9001 9002 9003; do
  sbatch --export=ALL,CONFIG_PATH=configs/champion_v2.yaml,TRAIN_SEED=$s \
    cluster/train_go1.slurm
done
```

Notes:

- The script already sets `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.65`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
  and a `$HOME/.cache/jax` compilation cache.
- Watch output with `tail -f slurm-go1-custom-ppo-<jobid>.out`.
- Throughput reference: ~13,500 SPS on an H100 NVL → ~13,100 s per 200M steps.
