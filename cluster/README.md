# NUS SoC cluster workflow

Run all commands from the repository root on `xlogin`. Training itself is
submitted to Slurm and must never be run directly on the login node.

```bash
bash cluster/setup.sh
sbatch cluster/check_gpu.slurm
squeue -u "$USER"
```

The setup script first tries `/mnt/scratch/$USER/rl-launchpad/pip-tmp`, then
falls back to `$HOME/.tmp/rl-launchpad/pip-tmp` when scratch has not been
provisioned for the account. The login nodes limit `/tmp` to 50 MB. Pip's wheel
cache is disabled so CUDA packages are not stored twice.

The installer upgrades the shared cuDNN 9 runtime after resolving project
dependencies. PyTorch 2.6 otherwise downgrades it to 9.1, which is too old for
the JAX 0.11 CUDA wheel compiled against cuDNN 9.8.

After the GPU check succeeds, submit the compilation smoke test:

```bash
sbatch cluster/train_go1.slurm
```

Inspect `slurm-go1-custom-ppo-<jobid>.out`. If it succeeds, submit the 20M
single-seed learning gate:

```bash
sbatch --export=ALL,CONFIG_PATH=configs/cluster_20m.yaml,TRAIN_SEED=10 \
  cluster/train_go1.slurm
```

Only after linear-velocity error improves and the rendered policy visibly
moves should the 200M configuration be submitted. Request a longer allocation
if the measured 20M throughput requires it:

```bash
sbatch --time=03:00:00 \
  --export=ALL,CONFIG_PATH=configs/cluster_200m.yaml,TRAIN_SEED=10 \
  cluster/train_go1.slurm
```

Useful commands:

```bash
squeue -u "$USER"
sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode
scancel <jobid>
tail -f slurm-go1-custom-ppo-<jobid>.out
```
