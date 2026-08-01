# Tutorial

Step-by-step, function-by-function instructions for the `rl-launchpad` codebase:
from-scratch PyTorch PPO on the MuJoCo Playground `Go1JoystickFlatTerrain` task,
including how to run everything on the NUS SoC Slurm cluster.

Every command below is run from the repository root. Local Windows commands use
PowerShell; cluster commands use Bash on the `xlogin` login node.

## Map of topics

| Doc | What it covers | Key files |
| --- | --- | --- |
| [01-installation.md](01-installation.md) | Local env setup (`uv sync`) and cluster env setup | `pyproject.toml`, `cluster/setup.sh`, `cluster/check_gpu.slurm` |
| [02-train-custom-ppo.md](02-train-custom-ppo.md) | Train the from-scratch PyTorch PPO, single and multi-seed | `ppo/train_multi_seed.py`, `ppo/train.py`, `configs/*.yaml`, `cluster/train_go1.slurm` |
| [03-train-brax-baseline.md](03-train-brax-baseline.md) | Train the Brax PPO baseline (Rule R2) | `cluster/train_brax_go1.py`, `configs/brax_go1_*.yaml`, `cluster/train_brax_go1.slurm` |
| [04-evaluate.md](04-evaluate.md) | 50-fixed-episode evaluation of custom + Brax checkpoints | `eval/evaluate.py`, `eval/eval_brax_seeds.py`, `eval/eval_all_seeds.py` |
| [05-visualize-and-publish.md](05-visualize-and-publish.md) | Live viewers, video recording, benchmark plot, submission build | `eval/view_native_v2.py`, `eval/record_native_video.py`, `eval/plot_benchmark.py`, `eval/build_submission.py` |
| [06-slurm-cheatsheet.md](06-slurm-cheatsheet.md) | Slurm commands: submit, monitor, cancel, arrays, GPU types | `cluster/*.slurm` |
| [07-api-reference.md](07-api-reference.md) | Every function in `ppo/ppo.py`, `ppo/agent.py`, `ppo/env.py` | `ppo/*.py` |

## Golden rules

1. **Never train on the login node.** Submit everything with `sbatch`.
2. **Never run jobs on a GPU type that never queues** — use `--gres=gpu:a100-40:1`
   for both trainers (the `a100-80:1` default in `train_brax_go1.slurm` does not queue).
3. **The evaluation contract is fixed**: `eval_seed=20000`, 50 episodes, native C++
   MuJoCo. Do not change it between runs you intend to compare.
4. **Throughput / steps accounting is verified**: `epochs * steps_per_epoch` must equal
   `total_timesteps_per_seed`; the trainer refuses to run otherwise.
