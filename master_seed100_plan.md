# Master Optimization Plan (Seed 100 Execution & Local Merge Guide)

This document tracks all 4 master architectural and algorithmic optimizations applied to **Seed 100** to beat the Brax 200M baseline (`LinErr: 0.0645 m/s`, `Mean Return: 19.8285`).

---

## 🎯 Master Hyperparameter Configuration (`configs/seed100.yaml`)

| Parameter | Value | Purpose / Impact |
| :--- | :---: | :--- |
| **`seed` / `seeds`** | `100` / `[100]` | Dedicated Master Seed |
| **`num_envs`** | `8192` | Unclipped gradient environment scale (prevents 32k clip saturation) |
| **`history_len`** | `3` | **144-dim actor observation space** (3 frames of past obs + actions) |
| **`gamma`** | `0.99` | **100-step planning horizon** (2.0s planning window vs 0.66s) |
| **`steps_per_epoch`** | `262144` | $8192 \text{ envs} \times 32 \text{ rollout steps}$ |
| **`epochs`** | `762` | Full 199.7M timestep training length |
| **`batch_size`** | `5120` | Proven minibatch gradient step size |
| **`train_iters`** | `4` | Unclipped PPO update passes |
| **`hidden_sizes`** | `[512, 512, 256]` | **2x Neural Network Brain Capacity** |
| **`target_kl`** | `0.015` | Stable epoch-level policy update bound |
| **`ent_coef` $\to$ `ent_coef_final`** | `0.005` $\to$ `0.0002` | **Fast late-stage action noise collapse** |

---

## 🔬 Code Modifications in `ppo/ppo.py`

### 1. Per-Minibatch Advantage Normalization
* In `ppo/ppo.py` (`update()` function):
  ```python
  mb_advantages = (advantages[minibatch_indices] - advantages[minibatch_indices].mean()) / (advantages[minibatch_indices].std() + 1e-8)
  ```
* **Why**: Standardizes gradient scale per minibatch, preventing minibatch advantage scale fluctuations.

### 2. Epoch-Level KL Early Stopping Monitoring
* Replaced inner minibatch KL early stopping break with epoch-level monitoring so noisy minibatches do not prematurely abort PPO update passes.

---

## 🚀 Cluster Execution Commands

```bash
# 1. Update ppo.py and create configs/seed100.yaml on cluster:
python3 -c "
with open('ppo/ppo.py', 'r') as f: c = f.read()
if 'if approx_kl > 1.5 * target_kl:' in c:
    c = c.replace('if approx_kl > 1.5 * target_kl:\n                break', '# Minibatch KL monitored at epoch level')
if 'mb_advantages = advantages[minibatch_indices]' in c:
    c = c.replace('mb_advantages = advantages[minibatch_indices]', 'mb_advantages = (advantages[minibatch_indices] - advantages[minibatch_indices].mean()) / (advantages[minibatch_indices].std() + 1e-8)')
with open('ppo/ppo.py', 'w') as f: f.write(c)

c = '''# 200M Master Seed 100 Config designed to beat Brax baseline
env_name: \"Go1JoystickFlatTerrain\"
seed: 100
seeds: [100]
num_envs: 8192
history_len: 3
episode_length: 1000
steps_per_epoch: 262144
epochs: 762
total_timesteps_per_seed: 199753728
total_timesteps: 199753728
gamma: 0.99
lam: 0.95
clip_ratio: 0.2
pi_lr: 0.0003
anneal_lr: true
target_kl: 0.015
max_grad_norm: 1.0
train_iters: 4
batch_size: 5120
hidden_dim: 512
hidden_sizes: [512, 512, 256]
vf_coef: 0.5
initial_log_std: -1.0
ent_coef: 0.005
ent_coef_final: 0.0002
checkpoint_dir: \"checkpoints/cluster_100m_v2\"
'''
open('configs/seed100.yaml', 'w').write(c)
print('Setup complete for Seed 100!')
"

# 2. Launch Master Seed 100:
sbatch --partition=gpu --exclude=xgpj0,xgpe0 --time=03:30:00 --gres=gpu:h100-47:1 \
  --export=ALL,CONFIG_PATH=configs/seed100.yaml \
  cluster/train_go1.slurm
```

---

## 🎮 Local Evaluation After Completion

Once Seed 100 completes on cluster, pull the checkpoint and run deterministic evaluation:

```powershell
# 1. Pull checkpoint via SCP:
scp -J <jump-host> <user>@<login-host>:~/rl-launchpad/checkpoints/cluster_100m_v2/ppo_seed100.pt NEW_checkpoints/cluster_100m_v2/

# 2. View live in 3D interactive viewer:
python eval/view_native.py NEW_checkpoints/cluster_100m_v2/ppo_seed100.pt --config configs/seed100.yaml
```
