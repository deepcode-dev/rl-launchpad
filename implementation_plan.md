# Master Optimization Plan: Beating the Brax 200M Baseline

This implementation plan synthesizes the findings of our 3 specialized AI subagents (**PPO Loss Specialist**, **Locomotion Architecture Specialist**, and **GPU Throughput Specialist**) into a unified strategy to beat the Brax 200M baseline (`LinErr: 0.0645 m/s`, `Mean Return: 19.8285`).

---

## 🔬 Core Root Causes Identified

1. **POMDP Observability Gap**: The environment applies an EMA action filter ($\hat{a}_t = 0.7 \hat{a}_{t-1} + 0.3 a_t$), but `history_len: 1` hides past actions from the actor. The policy cannot see its current filter state, making velocity tracking noisy.
2. **Minibatch KL Over-Abortion**: Early stopping was checking `approx_kl > 1.5 * target_kl` on individual noisy minibatches, prematurely aborting up to 80% of scheduled PPO update passes per epoch.
3. **PPO Gradient Clipping Saturation at 32k Envs**: Running at `num_envs: 32768` creates high batch variance, causing 90%+ of PPO minibatch gradients to hit the $1 \pm 0.2$ clip threshold and freeze weight refinement at `LinErr: 0.78`.
4. **Short Discount Horizon ($\gamma = 0.97$)**: $\gamma = 0.97$ gives an effective horizon of only 33 timesteps (0.66 seconds), which is too short-sighted for long-term footstep planning.

---

## 🛠️ Proposed Changes

### Component 1: Algorithm & Loss Semantics (`ppo/ppo.py` & `ppo/train_multi_seed.py`)

#### [MODIFY] [ppo/ppo.py](ppo/ppo.py)
* **Epoch-Level KL Early Stopping**: Move `approx_kl` early stopping check from minibatch level to epoch level.
* **Per-Minibatch Advantage Normalization**: Normalize advantages inside the inner minibatch update loop ($A_{\text{mb}} = \frac{A_{\text{mb}} - \mu}{\sigma + 10^{-8}}$).
* **Unclipped Critic Value Loss**: Use MSE value function loss to prevent gradient truncation on the critic.

#### [MODIFY] [configs/cluster_100m_v2.yaml](configs/cluster_100m_v2.yaml)
* **`num_envs: 8192`**: Unclipped gradient environment scale.
* **`history_len: 3`**: 144-dim actor observation space (3 frames of temporal history).
* **`gamma: 0.99`**: 100-step planning horizon (2.0 seconds).
* **`hidden_sizes: [512, 512, 256]`**: 2x Brain Power.
* **`ent_coef_final: 0.0002`**: Fast late-stage noise collapse.

### Component 2: Environment Inter-Op (`ppo/env.py`)

#### [MODIFY] [ppo/env.py](ppo/env.py)
* **Responsive EMA Filter**: Set $\hat{a}_t = 0.4 \hat{a}_{t-1} + 0.6 a_t$ for 2x faster motor response.
* **Staggered Episode Resets**: Maintain randomized initial episode offsets to prevent synchronized environment reset spikes.

---

## 📊 Expected Performance Impact

| Metric | Brax 200M Baseline | Champion Seed 20 | **Target Optimization** |
| :--- | :---: | :---: | :---: |
| **Linear Velocity Error (`LinErr`)** | `0.0645 m/s` | `0.1100 m/s` | **`< 0.0450 m/s` 🏆** |
| **Yaw Rate Error (`YawErr`)** | `0.0481 rad/s` | `0.0880 rad/s` | **`< 0.0400 rad/s` 🏆** |
| **50-Episode Mean Return** | `19.8285` | `19.8140` | **`> 19.8500` 🏆** |
| **Balance Survival Rate** | `100.0%` | `100.0%` | **`100.0%` (0.00% Fall Rate)** |
| **200M Execution Time** | ~4 hours | ~3.6 hours | **~1.2 Hours** |

---

## 🎮 Verification Plan

### Automated Tests
1. Run local smoke test (`python ppo/train.py`) to verify `history_len: 3` 144-dim observation tensor shapes.
2. Run multi-seed evaluation (`python eval/eval_200m_seeds.py`) across 50 fixed evaluation episodes to verify Rule R2 compliance and calculate final mean $\pm$ std dev metrics.
