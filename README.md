# RL From Scratch: Go1 Locomotion

A from-scratch PyTorch PPO agent for `Go1JoystickFlatTerrain` in MuJoCo Playground (MJX), built for the Launchpad 2026 Griffin Labs RL track. Rule R2 baseline comparison uses official Brax PPO.

The implementation features 8,192 parallel vector environment integration via `MJXVectorPyTorchWrapper`, 512-256-128 separate Actor-Critic MLPs, active policy entropy exploration (`ent_coef: 0.005`, `initial_log_std: -1.0`), running observation normalization, 123-dim asymmetric privileged critic ($V(s)$ with terrain heightmaps & contact forces), and Generalized Advantage Estimation ($\text{GAE}(\gamma=0.97, \lambda=0.95)$).

Our 200M Custom PyTorch PPO agent outperforms the published 200M Brax baseline across every evaluation metric: **LinErr 0.232** (vs 0.480), **YawErr 0.096 rad/s** (vs 0.150), and **R>0 99.39%** (vs 97.8%).

---

## ⚡ 15-Minute Judge Quickstart (Rule R3: Clone-to-Eval)

From a fresh `git clone`, a judge can verify environment contracts, evaluate policy checkpoints, and launch 3D steering in **under 3 minutes**:

```powershell
# 1. Install pinned dependencies (takes ~1 minute)
uv sync --extra dev --locked

# 2. Run unit & contract regression tests (<15 seconds)
uv run pytest -q

# 3. Evaluate trained checkpoints & plot benchmark comparison (<2 minutes)
uv run python eval/plot_benchmark.py

# 4. Drive the trained policy live in 3D (W/A/S/D, Q/E, 1/2/3/4 speed presets):
uv run python eval/view_native.py checkpoints/cluster_100m_v2/ppo_seed30.pt --config configs/cluster_100m_v2.yaml
```

---

## 🚀 High-Throughput Cluster Training (200M Timesteps / Seed)
Train seeds (10, 20, 30) for 200,000,000 environment steps per seed on NVIDIA A100 / H100 GPUs via Slurm:

```bash
# Submit 200M Seed 30 Run (4-Hour Window, Auto-Saving Every 10 Epochs)
sbatch --partition=gpu --time=04:00:00 \
  --export=ALL,CONFIG_PATH=configs/cluster_100m_v2.yaml,TRAIN_SEED=30 \
  cluster/train_go1.slurm

# Submit Seed 10 & Seed 20
sbatch --partition=gpu --time=04:00:00 \
  --export=ALL,CONFIG_PATH=configs/cluster_100m_v2.yaml,TRAIN_SEED=10 \
  cluster/train_go1.slurm

sbatch --partition=gpu --time=04:00:00 \
  --export=ALL,CONFIG_PATH=configs/cluster_100m_v2.yaml,TRAIN_SEED=20 \
  cluster/train_go1.slurm
```

---

## 🎮 Interactive 3D Keyboard Steering (Drive the Robot)

Drive the quadruped live in native 100+ FPS C++ MuJoCo with full 3-axis joystick control:
* **W** / **S** or **Up** / **Down**: Forward / Reverse speed ($v_x$)
* **Q** / **E**: Strafe Left / Right ($v_y$)
* **A** / **D** or **Left** / **Right**: Yaw Turning ($\omega_z$)
* **Keys 1, 2, 3, 4**: Speed Presets ($0.5$, $1.0$, $1.5$, $2.0\text{ m/s}$)
* **Spacebar**: Emergency Stop (`[0, 0, 0]`)

Test and steer trained policies live in 3D using native C++ MuJoCo rendering with **W/A/S/D or Arrow keys**:

```powershell
# Drive Custom PyTorch PPO Policy:
python eval/view_native.py checkpoints/cluster_100m_v2/ppo_seed10.pt --config configs/cluster_100m_v2.yaml

# Drive Brax 200M Baseline Policy:
python eval/view_brax_native.py checkpoints/brax_go1_200m/000200540160
```

### 🕹️ Keyboard Controls:
* **W** / **`Up Arrow`**: Accelerate forward ($v_x \mathrel{+}= 0.2\text{ m/s}$)
* **S** / **`Down Arrow`**: Reverse ($v_x \mathrel{-}= 0.2\text{ m/s}$)
* **A** / **`Left Arrow`**: Steer / Turn Left ($\omega_z \mathrel{+}= 0.3\text{ rad/s}$)
* **D** / **`Right Arrow`**: Steer / Turn Right ($\omega_z \mathrel{-}= 0.3\text{ rad/s}$)
* **`Spacebar`**: Emergency Stop ($v_x=0, v_y=0, \omega_z=0$)

---

## 📊 Published Baselines & Benchmark Comparison

Rule R2 comparison uses the official Brax PPO baseline trained to 200M timesteps on NVIDIA H100 (438,907 SPS):

```powershell
# Plot comparison benchmark figure:
python eval/plot_benchmark.py

# Record offscreen MP4/GIF video footage:
python eval/record_video.py checkpoints/cluster_100m_v2/ppo_seed10.pt --output write-up/policy-footage.mp4
```

---

## 🛡️ Correctness Contracts

- **GAE**: Operates on `[time, environment]` tensors and never crosses vector boundaries.
- **Autoreset**: Completed vector slots restore cached randomized initial states.
- **Reward**: Uses unmodified MuJoCo Playground `state.reward`.
- **Policy Distribution**: Tanh-squashed Gaussian bounded strictly to `[-1, 1]`.
- **Observation Normalization**: Running observation moments stored inside every checkpoint.
- **Checkpoint Sidecars**: `.meta.json` sidecar files record exact contract metadata and cumulative timesteps.

See [docs/environment.md](docs/environment.md), [docs/failures.md](docs/failures.md), and [write-up/submission.md](write-up/submission.md) for full architecture and evaluation narratives.
