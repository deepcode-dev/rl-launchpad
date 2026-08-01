# RL From Scratch: Go1 Locomotion

A from-scratch PyTorch PPO agent for `Go1JoystickFlatTerrain` in MuJoCo Playground (MJX), built for the Launchpad 2026 Griffin Labs RL track. Rule R2 baseline comparison uses official Brax PPO.

The implementation features 8,192 parallel vector environment integration via `MJXVectorPyTorchWrapper`, 512-256-128 separate Actor-Critic MLPs with SiLU activations, active policy entropy exploration (`ent_coef: 0.01`, `initial_log_std: -1.0`), running observation normalization, 0.7/0.3 low-pass EMA action filtering, a 123-dim asymmetric privileged critic, and Generalized Advantage Estimation (\( \text{GAE}(\gamma=0.97, \lambda=0.95) \)).

Across five independent 200M-step seeds, the custom agent reaches **19.752 ± 0.020** mean command-tracking return, **0.0851 m/s** LinErr, **0.0676 rad/s** YawErr, and 1,000-step episodes. The three-seed 200M Brax reference reaches **19.821 ± 0.016**, **0.0668 m/s**, and **0.0454 rad/s**. These are deterministic native-MuJoCo evaluations using the shared command-tracking metric; training itself uses the stock `state.reward`.

---

## ⚡ 3-Minute Judge Quickstart (Rule R3: Clone-to-Eval)

From a fresh `git clone`, a judge can verify environment contracts, run unit tests, evaluate policy checkpoints, and launch 3D steering in **under 3 minutes**:

```powershell
# 1. Install pinned dependencies (takes ~1 minute)
uv sync --extra dev --locked

# 2. Run unit & contract regression tests (~1 minute)
uv run pytest -q --basetemp=.pytest_tmp -p no:cacheprovider

# 3. Evaluate top champion policy checkpoint (50 fixed benchmark episodes)
uv run python eval/evaluate.py --checkpoint NEW_checkpoints/ppo_v2/ppo_seed9033.pt

# 4. Drive the trained policy live in 3D (W/A/S/D, Q/E, 1/2/3/4 speed presets):
uv run python eval/view_native_v2.py NEW_checkpoints/ppo_v2/ppo_seed9033.pt
```

---

## 📊 Benchmark Evaluation Results (50 Fixed Episodes, `eval_seed=20000`)

Every checkpoint is evaluated over 50 fixed, disjoint benchmark evaluation episodes.

| Agent / Model | Environment Steps | Wall Time | Lin. Vel. Error (`LinErr`) | Yaw Rate Error (`YawErr`) | Mean Return (50 Ep.) | Fall Rate (`Done`) |
|---|---:|---:|---:|---:|---:|---:|
| **Brax PPO Baseline (200M, seeds 10/11/12)** | 200,000,000 | documented 589.3 s | **0.0668 m/s** | **0.0454 rad/s** | **19.821 ± 0.016** | **0.00%** |
| **Custom PyTorch PPO (seeds 9033/9006/9018/9016/8009)** | 199,884,800 | 13,434–13,872 s; mean 13,561 s | **0.0851 m/s** | **0.0676 rad/s** | **19.752 ± 0.020** | **0.00%** |

_Both rows summarize 50 fixed episodes per training seed at `eval_seed=20000`; aggregate ± is the standard deviation of independent seed means. The evaluator reports the shared command-tracking return, while training uses the stock task reward._

---

## 🎮 Interactive 3D Keyboard Steering (Drive the Robot)

Drive the quadruped live in native 100+ FPS C++ MuJoCo with full 3-axis joystick control:

```powershell
# Drive Top Champion Policy (Seed 9033):
python eval/view_native_v2.py NEW_checkpoints/ppo_v2/ppo_seed9033.pt

# Drive Seed 8009 Policy:
python eval/view_native_v2.py NEW_checkpoints/ppo_v2/ppo_seed8009.pt
```

### 🕹️ Keyboard Controls:
* **W** / **S**: Move Forward / Backward (\( v_x \))
* **A** / **D**: Strafe Left / Right (\( v_y \))
* **Q** / **E**: Yaw Turn Left / Right (\( \omega_z \))
* **Keys 1, 2, 3, 4**: Speed Presets (\( 0.3 \), \( 0.6 \), \( 0.9 \), \( 1.2\text{ m/s} \))

---

## 🚀 High-Throughput Cluster Training (200M Timesteps / Seed)

Submit multi-seed jobs on NVIDIA H100 GPUs via Slurm:

```bash
for s in {9001..9010}; do sbatch --partition=gpu --exclude=xgpj0,xgpe0 --time=04:15:00 --gres=gpu:h100-47:1 --job-name=v2-s$s --export=ALL,CONFIG_PATH=configs/champion_v2.yaml,TRAIN_SEED=$s cluster/train_go1.slurm; done
```

---

## 🛡️ Correctness Contracts

- **GAE**: Operates on `[time, environment]` tensors without crossing vector stream boundaries (\( \text{GAE}(\gamma=0.97, \lambda=0.95) \)).
- **Autoreset**: Completed vector slots restore cached randomized initial states.
- **Training reward**: Uses unmodified MuJoCo Playground `state.reward`.
- **Evaluation metric**: Custom and Brax use the same deterministic command-tracking return and velocity-error metrics; this is distinct from the training reward.
- **Policy Distribution**: Tanh-squashed Gaussian with the squash Jacobian in the log probability and a final `[-1, 1]` environment-bound check.
- **Low-Pass Action Filter**: 0.7/0.3 EMA filter on action output prevents high-frequency motor target jitter.
- **Observation Normalization**: Running observation moments stored inside every checkpoint.
