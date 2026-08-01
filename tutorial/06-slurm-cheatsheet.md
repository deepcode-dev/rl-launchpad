# 06 · Slurm cheatsheet (NUS SoC)

All commands run on the `xlogin` login node, from the repo root. **Never run
training on the login node.**

## Setup (once per machine refresh)

```bash
bash cluster/setup.sh
sbatch cluster/check_gpu.slurm
```

## Submit

```bash
# Custom PyTorch PPO — config + seed via env vars
sbatch cluster/train_go1.slurm                                                    # smoke test
sbatch --export=ALL,CONFIG_PATH=configs/cluster_20m.yaml,TRAIN_SEED=10 \
  cluster/train_go1.slurm

# Custom PyTorch PPO — resume
sbatch --export=ALL,CONFIG_PATH=configs/cluster_200m.yaml,TRAIN_SEED=10,RESUME=true \
  cluster/train_go1.slurm

# Brax baseline — ALWAYS override the GPU type (a100-80 never queues)
sbatch --gres=gpu:a100-40:1 --time=03:00:00 \
  --export=ALL,CONFIG_PATH=configs/brax_go1_200m.yaml \
  cluster/train_brax_go1.slurm
```

Environment variables consumed by the slurm scripts:

| Var | Used by | Meaning |
| --- | --- | --- |
| `CONFIG_PATH` | both trainers | YAML config path |
| `TRAIN_SEED` | `train_go1.slurm` | single-seed override (omit/"none" → config seed list) |
| `RESUME` | `train_go1.slurm` | `true` → `--resume` |

## Multi-seed fan-out

Submit one job per seed so each has its own GPU:

```bash
for s in 9001 9002 9003; do
  sbatch --export=ALL,CONFIG_PATH=configs/champion_v2.yaml,TRAIN_SEED=$s \
    cluster/train_go1.slurm
done
```

For Brax, give each seed its own config with a distinct `checkpoint_dir` so runs
never overwrite each other.

## Monitor & cancel

```bash
squeue -u "$USER"                                              # queued / running
sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode          # finished job stats
scancel <jobid>                                                 # cancel one
scancel -u "$USER" -t RUNNING,PENDING                           # cancel all yours
tail -f slurm-go1-custom-ppo-<jobid>.out                        # stream a log
```

Logs are written next to the repo root as `slurm-<jobname>-<jobid>.out`.

## GPU types & resources

| Train script | Default `--gres` | Works? | Use instead |
| --- | --- | --- | --- |
| `cluster/train_go1.slurm` | `gpu:a100-40:1` | yes | — |
| `cluster/train_brax_go1.slurm` | `gpu:a100-80:1` | **no — never queues** | `--gres=gpu:a100-40:1` |

Both scripts request 8 CPUs, 32 GB RAM, and print their own `#SBATCH` timeout
(override with `--time=` if the run needs longer — e.g. 200M custom PPO needs
~04:00:00).

## Files & paths

- Cluster Python: `.venv-cluster` (created by `cluster/setup.sh`).
- JAX cache: `$HOME/.cache/jax` (keeps the compiled pipeline between jobs).
- `checkpoints/` is git-ignored — pull artifacts to your laptop and place
  committed copies under `baselines/` / `NEW_checkpoints/`.
- Pull from cluster:
  ```bash
  scp -J <jump> -r <user>@xlogin:~/rl-launchpad/checkpoints/ppo_v2 NEW_checkpoints/
  scp -J <jump> -r <user>@xlogin:~/rl-launchpad/checkpoints/brax_go1_200m_seed11 \
                   <user>@xlogin:~/rl-launchpad/checkpoints/brax_go1_200m_seed12 baselines/
  ```
