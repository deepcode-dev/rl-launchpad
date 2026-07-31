# RL Code Walkthrough — Line-by-Line

A complete annotation of the from-scratch PyTorch PPO training stack so any team member can answer *any* judge question about the algorithm, the network, or the training loop. Read together with `judge_explanation.md` (the 3 required talking points) and `write-up/judge_viva_prep.md` (the interview script).

## Quick map & data flow

```
train_multi_seed.py  ──config──▶  ActorCritic (agent.py)
        │                              │  get_action / get_value / evaluate
        │ obs/action/reward            ▼
        └──────────▶  env.py (MJX JAX physics) ◀── DLPack zero-copy ──▶ MuJoCo Playground
        │
        ▼
   rollout buffers (GPU, pre-allocated)
        │
        ▼
   compute_gae (ppo.py) ──▶ update (ppo.py) ──▶ optimizer.step
```

| File | Role | Key entry points |
| :--- | :--- | :--- |
| `ppo/env.py` | Vectorized JAX<->PyTorch environment wrapper | `reset`, `step` |
| `ppo/agent.py` | Actor-critic network + tanh-squashed Gaussian + normalization | `ActorCritic`, `get_action`, `evaluate`, `get_value` |
| `ppo/ppo.py` | Pure algorithm: GAE, losses, update loop | `compute_gae`, `ppo_clip_loss`, `value_loss`, `update` |
| `ppo/train_multi_seed.py` | Training orchestration: rollout + GAE + update + checkpointing | `save_checkpoint`, `train_single_seed`, `main` |

Champion hyperparameters live in `configs/champion_v2.yaml` (referenced throughout; see the table at the end).

---

# File 1 — `ppo/env.py`

**Role:** wraps MuJoCo Playground's JAX (`MJX`) Go1 joystick task behind a minimal vectorized environment, keeps the stock reward untouched, auto-resets finished slots, and moves data between JAX and PyTorch with **zero-copy DLPack** GPU exchanges.

## Imports & class contract (L1–28)

```python
1:  """A small PyTorch-facing, vectorized wrapper for MuJoCo Playground MJX tasks.
...
6:  """
7:
8:  from __future__ import annotations
9:  from typing import Any
10: import jax
11: import jax.numpy as jnp
12: import numpy as np
13: import torch
14: import mujoco_playground as mp
```

- **L8** `from __future__ import annotations` — stringized type hints (Python ≥3.10 syntax on older runtimes).
- **L10-11** JAX is the physics backend: `env.step` and `env.reset` are jitted functions over batches of environments.
- **L13-14** PyTorch for the policy tensors; MuJoCo Playground registry for the task definition.

```python
19: class MJXVectorPyTorchWrapper:
20:     """Vectorized MJX environment with a minimal Gymnasium-compatible API.
21:
22:     ``step`` auto-resets only completed slots.  The observation returned for a
23:     completed slot is therefore its reset observation, while ``done`` or
24:     ``truncated`` describes the transition that just ended.
25:     """
26:
27:     action_low = -1.0
28:     action_high = 1.0
```

- **L27-28** Class-level action bounds. The policy is tanh-squashed to this range and the wrapper clips again defensively.

## `__init__` (L30–63)

```python
30:     def __init__(
31:         self,
32:         env_name: str = "Go1JoystickFlatTerrain",
33:         num_envs: int = 32,
34:         seed: int = 42,
35:         history_len: int = 5,
36:         episode_length: int | None = None,
37:         config_overrides: dict[str, Any] | None = None,
38:     ):
39:         if num_envs < 1:
40:             raise ValueError("num_envs must be positive")
41:         if history_len < 1:
42:             raise ValueError("history_len must be positive")
```

- **L32** Default task. Champions use `Go1JoystickFlatTerrain` explicitly (config `env_name`).
- **L35** `history_len` = how many stacked frames the actor sees. **Champion uses `history_len: 1`** — a single-frame observation, so velocity is inferred from state, not from temporal stacking.
- **L39-42** Fail fast on impossible construction arguments.

```python
44:         self.env_name = env_name
45:         self.num_envs = int(num_envs)
46:         self.history_len = int(history_len)
47:         self.cfg, self.env = self._load_environment(env_name, config_overrides)
48:         self.episode_length = int(
49:             episode_length if episode_length is not None else getattr(self.cfg, "episode_length", 1000)
50:         )
51:         if self.episode_length < 1:
52:             raise ValueError("episode_length must be positive")
```

- **L47** Loads the environment (see `_load_environment` below). The config object is kept because it carries `episode_length` and the observation/action sizes.
- **L48-52** Default episode length 1000 steps; the champion config sets `episode_length: 1000`.

```python
54:         self._v_reset = jax.jit(jax.vmap(self.env.reset))
55:         self._v_step = jax.jit(jax.vmap(self.env.step))
56:         self._slot_keys: jax.Array | None = None
57:         self.states = None
58:         self._initial_states = None
59:         self.obs_history: jax.Array | None = None
60:         # Stagger initial episode steps to prevent synchronized resets
61:         self._episode_steps = jax.random.randint(jax.random.PRNGKey(42), (self.num_envs,), 0, self.episode_length, dtype=jnp.int32)
62:         self._jax_platform = jax.devices()[0].platform
63:         self.reset(seed=seed)
```

- **L54-55** JIT-compile the vectorized (`vmap`) reset/step. `vmap` batches the single-env MJX functions across the parallel environment dimension; `jit` compiles them into one fused CUDA kernel.
- **L56-59** Per-slot RNG keys (so each environment streams an independent random source), the current physics state tree, the memorized initial states (used for instant auto-reset), and the stacked observation history buffer.
- **L60-61** Attempt to stagger each env's phase so the 8192 robots don't reset in a synchronized wave. **Note (important for judges):** `reset()` at L168 overwrites this with zeros, so in practice episodes start synchronized. This is benign — time-limit truncations are bootstrapped with the value function (see `ppo.py` GAE) — but be accurate if asked about it.
- **L63** The constructor immediately resets the environment once.

## `_load_environment` (L65–91)

```python
65:     @staticmethod
66:     def _load_environment(
67:         env_name: str,
68:         config_overrides: dict[str, Any] | None = None,
69:     ):
70:         """Load through the appropriate registry rather than exception routing."""
71:         if env_name in mp.locomotion.ALL_ENVS:
72:             cfg = mp.locomotion.get_default_config(env_name)
73:             cfg.impl = "jax"
74:             return cfg, mp.locomotion.load(
75:                 env_name, config=cfg, config_overrides=config_overrides,
76:             )
77:         if env_name in mp.manipulation.ALL_ENVS:
78:             cfg = mp.manipulation.get_default_config(env_name)
79:             cfg.impl = "jax"
80:             return cfg, mp.manipulation.load(env_name, config=cfg, config_overrides=config_overrides,)
81:         raise ValueError(
82:             f"Unknown MuJoCo Playground environment {env_name!r}. "
83:             f"Known locomotion environments: {mp.locomotion.ALL_ENVS}; "
84:             f"known manipulation environments: {mp.manipulation.ALL_ENVS}"
85:         )
```

- **L71-76** Resolves through the official locomotion registry, forces the JAX implementation (`cfg.impl = "jax"`), and applies any `config_overrides` (the champion config passes no overrides — **stock task, stock reward**).
- **L77-80** Same path for manipulation tasks (unused here but makes the wrapper general).
- **L81-85** Explicit error instead of silently falling through.

## Observation dimension properties (L93–111)

```python
93:     @property
94:     def base_observation_dim(self) -> int:
95:         size = self.env.observation_size
96:         return int(size["state"][0] if isinstance(size, dict) else size)
97:
98:     @property
99:     def privileged_observation_dim(self) -> int:
100:         size = self.env.observation_size
101:         if isinstance(size, dict) and "privileged_state" in size:
102:             return int(size["privileged_state"][0])
103:         return self.base_observation_dim
104:
105:     @property
106:     def observation_dim(self) -> int:
107:         return self.base_observation_dim * self.history_len
108:
109:     @property
110:     def action_dim(self) -> int:
111:         return int(self.env.action_size)
```

- **L94-96** The stock MJX task reports `observation_size["state"]` = **48** (the hardware-available actor observation).
- **L99-103** `observation_size["privileged_state"]` = **123** — the simulator-only critic observation. If the task exposes no privileged state, it falls back to the actor dim.
- **L106-107** Actor input size = base (48) × history (1) = **48**.
- **L110-111** Action size = **12** joint targets.

## Observation extraction helpers (L113–122)

```python
113:     def _obs_array(self, obs_raw: Any) -> jax.Array:
114:         state = obs_raw["state"] if isinstance(obs_raw, dict) else obs_raw
115:         return jnp.asarray(state, dtype=jnp.float32)
116:
117:     def _privileged_obs_array(self, obs_raw: Any) -> jax.Array:
118:         if isinstance(obs_raw, dict):
119:             state = obs_raw.get("privileged_state", obs_raw["state"])
120:         else:
121:             state = obs_raw
122:         return jnp.asarray(state, dtype=jnp.float32)
```

- **L113-115** Extract the 48-dim actor observation as `float32` JAX array.
- **L117-122** Extract the 123-dim privileged observation, tolerating tasks that don't provide one.

## DLPack zero-copy tensor exchange (L124–138)

```python
124:     @staticmethod
125:     def _to_tensor(array) -> torch.Tensor:
126:         """Share JAX buffers with PyTorch via DLPack GPU zero-copy; fall back to numpy for CPU."""
127:         try:
128:             return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(array))
129:         except Exception:
130:             return torch.from_numpy(np.array(array, copy=True, order="C"))
131:
132:     def _to_jax_action(self, action) -> jax.Array:
133:         try:
134:             return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(action))
135:         except Exception:
136:             if isinstance(action, torch.Tensor):
137:                 return jnp.asarray(action.detach().cpu().numpy(), dtype=jnp.float32)
138:             return jnp.asarray(action, dtype=jnp.float32)
```

- **L124-130** The performance-critical path. `jax.dlpack.to_dlpack` exposes JAX's GPU buffer via the DLPack protocol; `torch.utils.dlpack.from_dlpack` wraps **the same memory** as a CUDA tensor — **0 bytes copied, no PCIe round-trip**. Falls back to a CPU numpy copy only when no CUDA device exists.
- **L132-138** Reverse exchange for actions. On failure (CPU-only), detach and copy via numpy.

## Reset-key bookkeeping & auto-reset helper (L140–157)

```python
140:     def _split_reset_keys(self) -> jax.Array:
141:         assert self._slot_keys is not None
142:         keys = jax.vmap(jax.random.split)(self._slot_keys)
143:         self._slot_keys = keys[:, 0]
144:         return keys[:, 1]
145:
146:     @staticmethod
147:     def _select_finished(current, reset, finished):
148:         """Select reset values for each completed batch slot in a JAX pytree."""
149:         def select_leaf(old, new):
150:             mask = finished.reshape((finished.shape[0],) + (1,) * (old.ndim - 1))
151:             return jnp.where(mask, new, old)
152:
153:         return jax.tree_util.tree_map(select_leaf, current, reset)
154:
155:     def _flat_observation(self) -> jax.Array:
156:         assert self.obs_history is not None
157:         return self.obs_history.reshape(self.num_envs, -1)
```

- **L140-144** Advances per-env RNG: each `reset` call splits every slot's key into (keep, use); the "use" halves feed that reset, guaranteeing disjoint randomness per environment *and* per reset.
- **L146-153** Generic pytree merge: for every leaf, replace entries where `finished` is True with the reset value (broadcasting the mask over trailing dims). This is how auto-reset replaces `states` per slot.
- **L155-157** Flattens `[num_envs, history_len, obs_dim]` → `[num_envs, obs_dim*history_len]` = the 48-dim actor input.

## `reset` (L159–172)

```python
159:     def reset(self, seed: int | None = None):
160:         if seed is not None or self._slot_keys is None:
161:             root_key = jax.random.PRNGKey(0 if seed is None else int(seed))
162:             self._slot_keys = jax.random.split(root_key, self.num_envs)
163:         self.states = self._v_reset(self._split_reset_keys())
164:         self._initial_states = jax.tree_util.tree_map(lambda value: value, self.states)
165:         self._last_ema_action = None
166:         obs = self._obs_array(self.states.obs)
167:         self.obs_history = jnp.repeat(obs[:, None, :], self.history_len, axis=1)
168:         self._episode_steps = jnp.zeros(self.num_envs, dtype=jnp.int32)
169:         privileged_obs = self._privileged_obs_array(self.states.obs)
170:         return self._to_tensor(self._flat_observation()), {
171:             "privileged_observation": self._to_tensor(privileged_obs),
172:         }
```

- **L160-162** Derive per-env keys from a single seed (or keep existing keys on a plain re-reset).
- **L163** Batch-reset all environments (the `vmap`+`jit` kernel).
- **L164** Deep-copy the reset states into `_initial_states` — the template used for fast auto-reset at every episode boundary.
- **L165** Clear the EMA action filter state (first action after reset is unfiltered).
- **L166-167** Fill the history stack with the current observation (all slots identical since it's the first frame).
- **L168** Zero the per-env step counters (this is what overrides the stagger from L61).
- **L170-172** Return actor obs (as a GPU tensor) plus the privileged observation in the info dict.

## `step` (L174–258)

```python
174:     def step(self, action):
175:         if self.states is None or self.obs_history is None:
176:             raise RuntimeError("Call reset() before step()")
177:         action_jax = self._to_jax_action(action)
178:         if action_jax.ndim == 1:
179:             action_jax = action_jax[None, :]
180:         if action_jax.shape != (self.num_envs, self.action_dim):
181:             raise ValueError(
182:                 f"Expected actions with shape {(self.num_envs, self.action_dim)}, got {action_jax.shape}"
183:             )
```

- **L175-176** Guard: never step a dead environment.
- **L177-179** Zero-copy action to JAX; promote a single vector to a batch.
- **L180-183** Shape contract check (batch, 12).

### EMA action filter (L184–190)

```python
184:         # Low-pass EMA Action Filtering for smooth joint targets
185:         action_jax = jnp.clip(action_jax, self.action_low, self.action_high)
186:         if self._last_ema_action is None or self._last_ema_action.shape != action_jax.shape:
187:             self._last_ema_action = action_jax
188:         else:
189:             action_jax = 0.7 * self._last_ema_action + 0.3 * action_jax
190:             self._last_ema_action = action_jax
```

- **L185** Defensive clip to $[-1, 1]$ (the policy is already tanh-bounded).
- **L186-190** First-order low-pass: `a_final = 0.7·a_prev + 0.3·a_raw`. This filters high-frequency joint-target jitter into smooth strides. The filter state persists across episodes within a slot (only cleared on `reset()`). **The evaluator applies the identical filter deterministically** (`eval/evaluate.py:226-233`) so reported numbers match deployed behavior.

### Physics step, termination vs truncation (L192–218)

```python
192:         terminal_states = self._v_step(self.states, action_jax)
193:         reward_array = jnp.asarray(terminal_states.reward, dtype=jnp.float32)
194:         natural_done = terminal_states.done.astype(jnp.bool_)
195:         self._episode_steps += 1
196:         time_limit = self._episode_steps >= self.episode_length
197:         truncated_np = time_limit & ~natural_done
198:         finished = natural_done | truncated_np
```

- **L192** Run physics: the vectorized MJX step kernel returns the **terminal** states for every slot.
- **L193** **Stock task reward** (`state.reward`) — never modified.
- **L194** True episode *terminations*: the robot fell (Go1's `done`).
- **L195-198** Time-limit detection: if a slot has run `episode_length` steps *without* falling it is **truncated**, not terminated. `finished` = either case (this slot needs auto-reset).

```python
200:         terminal_obs = self._obs_array(terminal_states.obs)
201:         terminal_privileged_obs = self._privileged_obs_array(terminal_states.obs)
202:         terminal_history = jnp.roll(self.obs_history, shift=-1, axis=1)
203:         terminal_history = terminal_history.at[:, -1, :].set(terminal_obs)
204:         assert self._initial_states is not None
205:         reset_states = self._initial_states
206:         self.states = self._select_finished(terminal_states, reset_states, finished)
207:         reset_obs = self._obs_array(reset_states.obs)
208:         reset_privileged_obs = self._privileged_obs_array(reset_states.obs)
209:         next_obs = jnp.where(finished[:, None], reset_obs, terminal_obs)
210:         next_privileged_obs = jnp.where(
211:             finished[:, None], reset_privileged_obs, terminal_privileged_obs
212:         )
213:         self._episode_steps = jnp.where(finished, 0, self._episode_steps)
214:         reset_history = jnp.repeat(next_obs[:, None, :], self.history_len, axis=1)
215:         self.obs_history = jnp.where(
216:             finished[:, None, None], reset_history, terminal_history
217:         )
```

- **L200-203** Build the *pre-reset* observation: roll the history stack and append the just-observed terminal obs. **This is the state seen at the moment of truncation** — used later to bootstrap the truncated value (via `final_observation`).
- **L204-206** Auto-reset: replace each finished slot's physics state with its memorized initial state.
- **L207-212** Next observations for finished slots are their reset observations; for live slots, the terminal ones.
- **L213** Reset episode-step counters for finished slots.
- **L214-217** Finished slots get a fresh history stack (all frames = reset obs); live slots keep the rolled history.

### Info dict (L219–251)

```python
219:         command = jnp.asarray(terminal_states.info["command"], dtype=jnp.float32)
220:         # The privileged observation layout is defined by the stock Go1 task:
221:         # actor state, then gyro, accelerometer, gravity and true local velocity.
222:         privileged_offset = self.base_observation_dim
223:         true_gyro = terminal_privileged_obs[:, privileged_offset : privileged_offset + 3]
224:         true_local_velocity = terminal_privileged_obs[
225:             :, privileged_offset + 9 : privileged_offset + 12
226:         ]
227:         info = {
228:             # Available to callers that need terminal transitions despite autoreset.
229:             # Full policy observation before autoreset.  PPO uses this to
230:             # bootstrap time-limit truncations without crossing episodes.
231:             "final_observation": self._to_tensor(
232:                 terminal_history.reshape(self.num_envs, -1)
233:             ),
234:             "final_observation_mask": self._to_tensor(finished),
235:             "privileged_observation": self._to_tensor(next_privileged_obs),
236:             "final_privileged_observation": self._to_tensor(terminal_privileged_obs),
237:             "command": self._to_tensor(command),
238:             "local_linear_velocity": self._to_tensor(true_local_velocity),
239:             "linear_velocity_error": self._to_tensor(
240:                 jnp.linalg.norm(command[:, :2] - true_local_velocity[:, :2], axis=1)
241:             ),
242:             "yaw_rate_error": self._to_tensor(
243:                 jnp.abs(command[:, 2] - true_gyro[:, 2])
244:             ),
245:             "mean_abs_action": self._to_tensor(
246:                 jnp.mean(jnp.abs(action_jax), axis=1)
247:             ),
248:             "base_position_xy": self._to_tensor(
249:                 jnp.asarray(terminal_states.data.qpos[:, :2], dtype=jnp.float32)
250:             ),
251:         }
252:         return (
253:             self._to_tensor(self._flat_observation()),
254:             self._to_tensor(reward_array),
255:             self._to_tensor(natural_done),
256:             self._to_tensor(truncated_np),
257:             info,
258:         )
```

- **L219** The commanded velocity (v_x, v_y, ω_z) from the stock task.
- **L222-226** Extract true gyro and true local linear velocity from the privileged observation (layout defined by the stock task), used for the tracking-error metrics.
- **L231-234** `final_observation` = the pre-autoreset obs for every slot (mask says which slots actually ended). **This is the linchpin of correct truncation bootstrapping** — see `ppo.py` GAE.
- **L239-244** `linear_velocity_error = ‖cmd[:2] − v_local[:2]‖`, `yaw_rate_error = |cmd[2] − ω_true[2]|` — the headline tracking metrics.
- **L252-258** Returns `(next_obs, reward, done, truncated, info)` — a standard 5-tuple. Critically, **`done` is only ever a true fall, never a time-limit**, so the learner can distinguish the two cases.

---

# File 2 — `ppo/agent.py`

**Role:** the neural network. An asymmetric actor-critic with a **tanh-squashed Gaussian** policy, orthogonal initialization, and running observation normalization.

## Initialization helpers (L1–21)

```python
7:  def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
8:      """Orthogonal initialization for neural network layers."""
9:      nn.init.orthogonal_(layer.weight, std)
10:     nn.init.constant_(layer.bias, bias_const)
11:     return layer
12:
13:
14: def _make_mlp(input_dim, hidden_sizes, output_dim, output_std):
15:     layers = []
16:     previous_dim = input_dim
17:     for hidden_size in hidden_sizes:
18:         layers.extend((layer_init(nn.Linear(previous_dim, hidden_size)), nn.SiLU()))
19:         previous_dim = hidden_size
20:     layers.append(layer_init(nn.Linear(previous_dim, output_dim), std=output_std))
21:     return nn.Sequential(*layers)
```

- **L7-11** Orthogonal weight init with gain `std` (default $\sqrt{2}$, matching PyTorch's ReLU-style convention) and zero bias.
- **L14-21** MLP builder: hidden blocks of `Linear + SiLU`, then one output `Linear` whose gain is `output_std` (small for the actor head, 1.0 for the critic head). Note **SiLU** activation (not ELU/ReLU) — smoother gradients, good for continuous control.

## `ActorCritic.__init__` (L24–54)

```python
24: class ActorCritic(nn.Module):
25:     """Actor-critic with a tanh-squashed Gaussian action distribution."""
26:
27:     _ACTION_EPS = 1e-6
28:     _LOG_STD_MIN = -3.0
29:     _LOG_STD_MAX = 0.5
30:
31:     def __init__(
32:         self,
33:         obs_dim,
34:         act_dim,
35:         hidden_dim=256,
36:         initial_log_std=-1.0,
37:         *,
38:         critic_obs_dim=None,
39:         hidden_sizes=None,
40:     ):
41:         super().__init__()
42:         self.obs_dim = obs_dim
43:         self.act_dim = act_dim
44:         self.critic_obs_dim = int(critic_obs_dim or obs_dim)
45:         self.hidden_sizes = tuple(int(size) for size in (hidden_sizes or (hidden_dim, hidden_dim)))
46:         self.actor = _make_mlp(obs_dim, self.hidden_sizes, act_dim, output_std=0.01)
47:         self.actor_log_std = nn.Parameter(torch.full((1, act_dim), float(initial_log_std)))
48:         self.critic = _make_mlp(self.critic_obs_dim, self.hidden_sizes, 1, output_std=1.0)
49:         self.register_buffer("obs_mean", torch.zeros(obs_dim))
50:         self.register_buffer("obs_var", torch.ones(obs_dim))
51:         self.register_buffer("obs_count", torch.tensor(1e-4))
52:         self.register_buffer("critic_obs_mean", torch.zeros(self.critic_obs_dim))
53:         self.register_buffer("critic_obs_var", torch.ones(self.critic_obs_dim))
54:         self.register_buffer("critic_obs_count", torch.tensor(1e-4))
```

- **L27-29** Constants: `ε=1e-6` prevents `log(0)` in the tanh Jacobian; `log_std` is clamped to `[−3.0, 0.5]` so the per-dimension standard deviation stays in `[0.05, 1.65]` (prevents both collapse and explosion).
- **L42-45** Dimension bookkeeping. Champion sizes: `obs_dim=48`, `critic_obs_dim=123`, `act_dim=12`, `hidden_sizes=(512, 256, 128)`.
- **L46-47** **Actor head gain 0.01** → near-zero initial action means near-zero initial actions = the robot starts near the standing pose; exploration is carried by the nonzero `log_std` (`initial_log_std = -1.0` → σ ≈ 0.37). A **single shared** `log_std` parameter per action dimension (not state-dependent) keeps the policy simple and stable.
- **L48** Critic head gain 1.0 (values are unconstrained).
- **L49-54** **Running observation statistics** stored as buffers (persist in checkpoints): mean, variance, and a pseudo-count per dimension, maintained separately for actor (48) and critic (123) observations. These drive online normalization at inference — a key sim-to-real-friendly design.

## `forward` + observation normalization (L56–80)

```python
56:     def forward(self, obs, critic_obs=None):
57:         critic_obs = obs if critic_obs is None else critic_obs
58:         return self.actor(self._normalize_observation(obs)), self.critic(
59:             self._normalize_critic_observation(critic_obs)
60:         )
61:
62:     def _normalize_observation(self, obs):
63:         clean_obs = torch.nan_to_num(
64:             obs.to(device=self.obs_mean.device, dtype=self.obs_mean.dtype),
65:             nan=0.0
66:         )
67:         clean_var = torch.nan_to_num(self.obs_var, nan=1.0).clamp(min=1e-4)
68:         clean_mean = torch.nan_to_num(self.obs_mean, nan=0.0)
69:         normalized = (clean_obs - clean_mean) / torch.sqrt(clean_var + 1e-8)
70:         return normalized.clamp(-10.0, 10.0)
71:
72:     def _normalize_critic_observation(self, obs):
73:         ...  # identical pattern for the 123-dim critic obs
80:         return normalized.clamp(-10.0, 10.0)
```

- **L56-60** `forward` returns `(actor_mean, V(s))` — both networks consume *normalized* inputs.
- **L62-70** Whitening: `(x − μ) / √(σ² + 1e-8)`, with NaN hardened to 0 everywhere, variance floored at `1e-4`, and the result clipped to `[−10, 10]`. Clipping caps the influence of rare extreme samples.
- **L72-80** Same routine for the privileged critic observation.

## Running-statistics merge (Welford) (L82–130)

```python
82:     @staticmethod
83:     def _merge_observation_stats(observations, mean, var, count):
84:         clean_obs = torch.nan_to_num(
85:             observations.to(device=mean.device, dtype=mean.dtype),
86:             nan=0.0
87:         )
88:         batch_mean = clean_obs.mean(dim=0)
89:         batch_var = clean_obs.var(dim=0, unbiased=False)
90:         batch_count = torch.tensor(clean_obs.shape[0], device=count.device, dtype=count.dtype)
91:
92:         safe_mean = torch.nan_to_num(mean, nan=0.0)
93:         safe_var = torch.nan_to_num(var, nan=1.0).clamp(min=1e-4)
94:         safe_count = torch.nan_to_num(count, nan=1e-4).clamp(min=1e-4)
95:
96:         delta = batch_mean - safe_mean
97:         total_count = safe_count + batch_count
98:         new_mean = torch.nan_to_num(safe_mean + delta * batch_count / total_count, nan=0.0)
99:         current_m2 = safe_var * safe_count
100:         batch_m2 = batch_var * batch_count
101:         correction = delta.square() * safe_count * batch_count / total_count
102:         new_var = torch.nan_to_num((current_m2 + batch_m2 + correction) / total_count, nan=1.0).clamp(min=1e-4)
103:
104:         mean.copy_(new_mean)
105:         var.copy_(new_var)
106:         count.copy_(total_count)
```

- **L82-106** The standard **parallel Welford combine** to merge a new batch's (mean, variance, count) into the persistent running moments **exactly** (no re-scanning history, no approximation):
  - `Δ = μ_batch − μ`
  - `μ_new = μ + Δ·n_batch / (n + n_batch)` (L98)
  - `M2_new = M2 + M2_batch + Δ²·n·n_batch/(n+n_batch)` (L99-102)
- This is how the normalization stays a true running statistic of all data seen so far, with NaN/bad-state hardening throughout.

```python
108:     @torch.no_grad()
109:     def update_observation_stats(self, observations, critic_observations=None):
110:         """Merge a raw observation batch into persistent running moments."""
111:         if observations.ndim != 2 or observations.shape[1] != self.obs_dim:
112:             raise ValueError(f"observations must have shape [batch, {self.obs_dim}]")
113:         critic_observations = observations if critic_observations is None else critic_observations
114:         if (
115:             critic_observations.ndim != 2
116:             or critic_observations.shape[1] != self.critic_obs_dim
117:         ):
118:             raise ValueError(
119:                 "critic_observations must have shape "
120:                 f"[batch, {self.critic_obs_dim}]"
121:             )
122:         self._merge_observation_stats(
123:             observations, self.obs_mean, self.obs_var, self.obs_count
124:         )
125:         self._merge_observation_stats(
126:             critic_observations,
127:             self.critic_obs_mean,
128:             self.critic_obs_var,
129:             self.critic_obs_count,
130:         )
```

- **L108-130** Public entry: validates shapes, then merges one batch into both the actor and critic running stats. Called with the first obs at reset (train_multi_seed.py:143) and with the full epoch batch after each update (train_multi_seed.py:281).

## The tanh-squashed Gaussian policy (L132–164)

```python
132:     def _distribution(self, obs):
133:         norm_obs = self._normalize_observation(obs)
134:         raw_mean = self.actor(norm_obs)
135:         mean = torch.nan_to_num(raw_mean, nan=0.0)
136:         log_std = torch.clamp(
137:             self.actor_log_std,
138:             min=self._LOG_STD_MIN,
139:             max=self._LOG_STD_MAX,
140:         )
141:         return Normal(mean, log_std.exp().expand_as(mean))
142:
143:     def _squashed_log_prob(self, dist, pre_tanh_action, action):
144:         correction = torch.log(1.0 - action.square() + self._ACTION_EPS)
145:         return (dist.log_prob(pre_tanh_action) - correction).sum(dim=-1)
```

- **L132-141** Build the pre-squash Gaussian `N(μ, σ²)` from the actor output and the shared (clamped) `log_std`.
- **L143-145** **Change-of-variables correction.** If `u ~ N(μ, σ²)` and `a = tanh(u)`, then

  $$\log \pi(a) = \log\mathcal{N}(u;\mu,\sigma^2) - \sum_i \log(1 - a_i^2 + \varepsilon)$$

  The `log(1−a²)` term is the log-determinant of the tanh Jacobian ($da/du = 1 - \tanh^2 u$). Without it, the importance ratio `exp(logπ_new − logπ_old)` in PPO is computed under a *different* distribution than the one actually sampled — silently corrupting every update. **This was the exact bug in the earlier `clipped_normal` implementation** and the reason seeds 7010/8001/8002 had to be regenerated.

```python
147:     def get_action(self, obs, deterministic=False):
148:         """Return an action in [-1, 1] and its squashed-distribution log-probability."""
149:         dist = self._distribution(obs)
150:         pre_tanh_action = dist.mean if deterministic else dist.sample()
151:         action = torch.tanh(pre_tanh_action)
152:         return action, self._squashed_log_prob(dist, pre_tanh_action, action)
153:
154:     def evaluate(self, obs, action, critic_obs=None):
155:         """Evaluate bounded actions under the same tanh-squashed distribution."""
156:         dist = self._distribution(obs)
157:         bounded_action = action.clamp(-1.0 + self._ACTION_EPS, 1.0 - self._ACTION_EPS)
158:         pre_tanh_action = torch.atanh(bounded_action)
159:         log_prob = self._squashed_log_prob(dist, pre_tanh_action, bounded_action)
160:         entropy = dist.entropy().sum(dim=-1)
161:         critic_obs = obs if critic_obs is None else critic_obs
162:         raw_value = self.critic(self._normalize_critic_observation(critic_obs)).squeeze(-1)
163:         value = torch.nan_to_num(raw_value, nan=0.0)
164:         return log_prob, entropy, value
```

- **L147-152** Rollout-time action: sample `u` (or use the mean deterministically at eval time), apply `tanh`, and return both the bounded action and its squashed log-prob. **`get_action` and `evaluate` compute log-probs under the identical density.**
- **L154-164** Update-time evaluation of *stored* actions:
  - L157-158 Invert the squash: `atanh(clamp(a, −1+ε, 1−ε))` recovers `u`. Clamping the stored action before `atanh` prevents `±∞` at the boundary.
  - L159 Squashed log-prob of the stored action (same formula as rollout).
  - L160 Gaussian entropy summed over the 12 dims — the exploration bonus term.
  - L162-163 Critic value from the privileged observation, NaN-hardened.

## Value-only inference (L166–169)

```python
166:     def get_value(self, obs, critic_obs=None):
167:         critic_obs = obs if critic_obs is None else critic_obs
168:         raw_value = self.critic(self._normalize_critic_observation(critic_obs)).squeeze(-1)
169:         return torch.nan_to_num(raw_value, nan=0.0)
```

- **L166-169** Critic-only forward used for `next_value` and for truncation bootstrapping (`V(s_final)`). Note the critic reads the **privileged** observation — at evaluation (native C++ MuJoCo) the evaluator instead calls `get_action` only, so the deployed policy never needs privileged data.

---

# File 3 — `ppo/ppo.py`

**Role:** the pure algorithm — no env, no network. GAE, the clipped policy loss, the clipped value loss, and the batched update loop with KL early stopping.

## Contract constant (L4)

```python
4: TRAINING_CONTRACT = "custom-ppo-privileged-critic-tracking-v3"
```

- **L4** A versioned string embedded in every checkpoint's metadata. The evaluator refuses checkpoints whose contract mismatches (`eval/evaluate.py:75-79`), guaranteeing "the numbers in the submission were produced by this exact algorithm."

## `compute_gae` (L7–60)

```python
7:  def compute_gae(
8:      rewards,
9:      values,
10:     terminated,
11:     next_value,
12:     gamma=0.99,
13:     lam=0.95,
14:     truncated=None,
15:     truncation_values=None,
16: ):
```

- **L7-16** Inputs (all shaped `[time, env]` except `next_value` = `[env]`):
  - `rewards` — one per transition.
  - `values` — `V(s_t)` for every stored state.
  - `terminated` — true only for falls.
  - `next_value` — `V(s_T)` for the last state of the rollout (no reward yet).
  - `truncated`, `truncation_values` — which slots hit the time limit, and their `V(s_final)` bootstrap.

```python
17:     """Compute GAE for a vectorized rollout without mixing environment streams."""
18:     rewards = torch.as_tensor(rewards)
19:     values = torch.as_tensor(values, device=rewards.device, dtype=rewards.dtype)
20:     terminated = torch.as_tensor(terminated, device=rewards.device, dtype=torch.bool)
21:     next_value = torch.as_tensor(next_value, device=rewards.device, dtype=rewards.dtype)
...
29:
30:     if truncated is None:
31:         truncated = torch.zeros_like(terminated)
32:     else:
33:         truncated = torch.as_tensor(truncated, device=rewards.device, dtype=torch.bool)
...
37:     if truncation_values is None:
38:         if truncated.any():
39:             raise ValueError("truncation_values are required for time-limit bootstrapping")
40:         truncation_values = torch.zeros_like(values)
41:     else:
42:         truncation_values = torch.as_tensor(...)
```

- **L18-28** Shape/device validation: everything `[time, env]`, `next_value` `[env]`. Vectorized streams are kept separate — one GAE pass spans all envs simultaneously but the recursion never mixes one env's trajectory into another's.
- **L30-46** If no truncation info is supplied, treat everything as terminations (and demand bootstrap values if any truncations exist).

```python
48:     continuation_mask = (~(terminated | truncated)).to(rewards.dtype)
49:     advantages = torch.zeros_like(rewards)
50:     gae = torch.zeros_like(next_value)
51:     next_values = torch.cat((values[1:], next_value.unsqueeze(0)), dim=0)
52:
53:     for t in range(rewards.shape[0] - 1, -1, -1):
54:         bootstrap_value = torch.where(truncated[t], truncation_values[t], next_values[t])
55:         bootstrap_mask = (~terminated[t]).to(rewards.dtype)
56:         delta = rewards[t] + gamma * bootstrap_value * bootstrap_mask - values[t]
57:         gae = delta + gamma * lam * continuation_mask[t] * gae
58:         advantages[t] = gae
59:
60:     return advantages, advantages + values
```

- **L48** Recursion continues only through slots that neither fell nor hit the time limit.
- **L51** Align `V(s_{t+1})`: for `t < T−1` it's `values[t+1]`; for the last stored step it's the freshly computed `next_value` (L247-248 in train_multi_seed.py).
- **L54** **The termination-vs-truncation fork:**
  - truncated slot → bootstrap with `V(s_final)` (the pre-autoreset state from `final_observation`);
  - otherwise → normal `V(s_{t+1})`.
- **L55** Terminal (fallen) slots get a **zero** bootstrap — no continuation after a fall.
- **L56-58** The two recursions:

  $$\delta_t = r_t + \gamma\,V_{\text{boot}}\,(1-d_t) - V(s_t) \qquad A_t = \delta_t + \gamma\lambda\,(1 - (d_t \lor t_t))\,A_{t+1}$$

- **L60** Returns `(A_t, A_t + V(s_t))` — advantages and value targets. `γ=0.97, λ=0.95` in the champion config.

## `ppo_clip_loss` (L63–68)

```python
63: def ppo_clip_loss(log_probs, old_log_probs, advantages, clip_ratio=0.2):
64:     """Compute the standard PPO clipped surrogate objective."""
65:     ratio = torch.exp(log_probs - old_log_probs)
66:     surr1 = ratio * advantages
67:     surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
68:     return -torch.min(surr1, surr2).mean()
```

- **L65** The probability ratio `r̂ = exp(log π_θ(a|s) − log π_old(a|s))`.
- **L66-68** Clipped surrogate, minimized: `−E[min(r̂·Â, clip(r̂, 0.8, 1.2)·Â)]`. The negative sign is the "minimize" convention. For a given minibatch the `Â` passed in has already been standardized (see `update`, L135).

## `value_loss` (L71–78)

```python
71: def value_loss(values, returns, old_values=None, clip_ratio=None):
72:     """Compute clipped or unclipped Huber value-function error."""
73:     if old_values is not None and clip_ratio is not None:
74:         v_clipped = old_values + torch.clamp(values - old_values, -clip_ratio, clip_ratio)
75:         v_loss1 = F.huber_loss(values, returns, reduction="none")
76:         v_loss2 = F.huber_loss(v_clipped, returns, reduction="none")
77:         return torch.max(v_loss1, v_loss2).mean()
78:     return F.huber_loss(values, returns)
```

- **L73-77** **Clipped value loss with Huber (Smooth L1) error.** Like PPO's policy clip, the value estimate is prevented from moving more than `±ε` from the old value: `V_clip = V_old + clip(V − V_old, ±ε)`. The loss takes the `max` of the two Huber errors (so the critic can't "game" the clip by moving the estimate). Huber (`0.5e²` for `|e|≤1`, `|e|−0.5` otherwise) is robust to reward outliers vs plain MSE.
- **L78** Unclipped fallback path (not used in the champion recipe).

## `update` (L81–184)

```python
81: def update(
82:     agent, optimizer, observations, actions, old_log_probs, returns, advantages,
89:     critic_observations=None, old_values=None, epochs=10, batch_size=256,
93:     clip_ratio=0.2, max_grad_norm=0.5, vf_coef=0.5, ent_coef=0.01, target_kl=None,
98: ):
99:     """Update PPO over an already-flattened ``[time * env, ...]`` rollout."""
100:     dataset_size = observations.shape[0]
```

- **L81-100** All rollout tensors arrive already flattened to `[time·env, dim]`. `critic_observations` may differ from `observations` (privileged critic). Shape checks follow (L104-114).

```python
121:     uncompiled_agent = getattr(agent, "_orig_mod", agent)
...
123:     for _ in range(epochs):
124:         epoch_kls = []
125:         indices = torch.randperm(dataset_size, device=observations.device)
126:         for start in range(0, dataset_size, batch_size):
127:             batch_idx = indices[start : start + batch_size]
128:             new_log_probs, entropy, values = agent.evaluate(
129:                 observations[batch_idx], actions[batch_idx], critic_observations[batch_idx],
130:             )
131:             batch_old_log_probs = old_log_probs[batch_idx]
132:             batch_advantages = advantages[batch_idx]
133:             batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std(unbiased=False) + 1e-8)
134:             batch_old_values = old_values[batch_idx] if old_values is not None else None
135:
136:             pol_loss = ppo_clip_loss(new_log_probs, batch_old_log_probs, batch_advantages, clip_ratio)
137:             val_loss = value_loss(values, returns[batch_idx], old_values=batch_old_values, clip_ratio=clip_ratio)
138:             entropy_bonus = entropy.mean()
139:             total_loss = pol_loss + vf_coef * val_loss - ent_coef * entropy_bonus
```

- **L121** `torch.compile` unwrap: if the agent was compiled, `_orig_mod` is the real module (needed for the `log_std` clamp below).
- **L123-127** Multiple passes (`train_iters: 4`) over the epoch's data; each pass reshuffles with `randperm` and iterates minibatches (`batch_size: 5120`).
- **L128-130** Re-evaluate stored actions under the *current* policy — this is where `log π_θ` (new) comes from.
- **L133** **Per-minibatch advantage re-standardization** — stabilizes the gradient scale regardless of the batch's empirical advantage distribution.
- **L136-139** The three-term loss with `vf_coef=0.5`, `ent_coef=0.01` (the champion config keeps entropy **fixed** at 0.01 — no annealing):

  $$L = L^{\text{CLIP}} + 0.5\,L^V - 0.01\,\bar{H}$$

```python
143:             optimizer.zero_grad(set_to_none=True)
144:             if torch.isfinite(total_loss):
145:                 total_loss.backward()
146:                 # Safeguard: verify all gradients are finite before optimizer step
147:                 grads_ok = True
148:                 for p in agent.parameters():
149:                     if p.grad is not None and not torch.isfinite(p.grad).all():
150:                         grads_ok = False
151:                         break
152:                 if grads_ok:
153:                     if max_grad_norm is not None:
154:                         torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
155:                     optimizer.step()
```

- **L143** `set_to_none=True` releases gradient memory each iteration (faster than zeroing).
- **L144** Skip the step entirely if the loss is non-finite.
- **L146-151** **Gradient-finiteness guard:** scan every parameter; if any gradient contains a NaN/Inf, *skip the optimizer step* rather than poisoning the weights.
- **L153-155** Global grad-norm clipping at `max_grad_norm=1.0` (champion), then `optimizer.step()`.

```python
157:             # Absolute Parameter Protection: Ensure actor_log_std parameter data stays valid
158:             with torch.no_grad():
159:                 uncompiled_agent.actor_log_std.nan_to_num_(nan=-1.0, posinf=0.5, neginf=-3.0).clamp_(-3.0, 0.5)
160:
161:             log_ratio = new_log_probs - batch_old_log_probs
162:             totals["policy_loss"] += pol_loss.item()
...
165:             ratio = log_ratio.exp()
166:             # Exact Seed 2001 Schulman KL estimator: ((ratio - 1.0) - log_ratio).mean()
167:             approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
168:             totals["approx_kl"] += approx_kl
169:             epoch_kls.append(approx_kl)
170:             totals["clip_fraction"] += ((ratio - 1.0).abs() > clip_ratio).float().mean().item()
171:             num_updates += 1
172:
173:             if target_kl is not None and approx_kl > 1.5 * float(target_kl):
174:                 early_stopped = True
175:                 break
176:
177:         epochs_completed += 1
178:         if early_stopped:
179:             break
```

- **L158-159** **Hard clamp on `actor_log_std`** after every step (NaN→−1.0, ±Inf→bounds, then `clamp_[−3, 0.5]`). The policy's variance can never leave the safe operating range, even under a pathological gradient.
- **L167** **Schulman's ratio KL estimator** `(r̂ − 1) − ln r̂`. Note this is *not* the second-moment approximation `½E[(Δlog π)²]`; the ratio form is an unbiased estimator of `KL(π_old ‖ π_new)` and was adopted after a first-moment `mean(log_old − log_new)` variant allowed cancellations across minibatches.
- **L170** Clip fraction — fraction of ratios outside `[0.8, 1.2]`, a useful "how hard did we push" telemetry.
- **L173-175** **Early stop** the whole update once a minibatch exceeds `1.5 × target_kl = 0.03` (`target_kl: 0.02`).
- **L177-179** Stop the outer epoch loop too.

```python
181:     result = {name: value / num_updates for name, value in totals.items()}
182:     result["update_epochs"] = epochs_completed
183:     result["early_stopped"] = early_stopped
184:     return result
```

- **L181-184** Averages the metrics across the minibatches actually applied and reports how many epochs ran + whether early stopping fired (this feeds the KL-based LR adaptation in the training loop).

---

# File 4 — `ppo/train_multi_seed.py`

**Role:** orchestration — build env + agent, roll out, call GAE/update, adapt the learning rate, and checkpoint with verifiable metadata. The champion config (`champion_v2.yaml`) drives it: `num_envs=8192`, `steps_per_epoch=163840` (20 steps × 8192 envs), `epochs=1220` → **199,884,800 steps/seed**, `train_iters=4`, `batch_size=5120`.

## Imports & preamble (L1–21)

```python
1:  # ruff: noqa: E402
2:  import argparse, json, os, sys, time, warnings
...
12: # Filter harmless JAX integer casting runtime warnings
13: warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*overflow encountered in cast.*")
14: warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
15:
16: # Ensure project root is in sys.path
17: sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
18:
19: from ppo.agent import ActorCritic
20: from ppo.ppo import TRAINING_CONTRACT, compute_gae, update
21: from ppo.env import MJXVectorPyTorchWrapper
```

- **L13-14** JAX emits a spurious "overflow encountered in cast" warning on some platforms; it's noise, not an error.
- **L16-17** Script can be launched from anywhere (`python ppo/train_multi_seed.py`).
- **L19-21** Imports the three modules just annotated.

## `save_checkpoint` (L24–54)

```python
24: def save_checkpoint(
25:     path, agent, *, seed, env_name, obs_dim, act_dim, hidden_dim, config,
26:     total_env_steps, wall_time_seconds, critic_obs_dim=None, hidden_sizes=None,
27: ):
28:     """Keep .pt compatible with existing evaluators and record its contract separately."""
29:     torch.save(agent.state_dict(), path)
30:     metadata = {
31:         "checkpoint_format": "raw_state_dict",
32:         "algorithm": "PPO",
33:         "training_contract": TRAINING_CONTRACT,
34:         "policy_distribution": "tanh_squashed_normal",
35:         "observation_normalization": "running_mean_variance",
36:         "reward_source": "mujoco_playground_state_reward",
37:         "physics_backend": "jax",
...
51:         "config": config,
52:     }
53:     with open(f"{path}.meta.json", "w") as f:
54:         json.dump(metadata, f, indent=2)
```

- **L29** Saves only the raw `state_dict` (weights + buffers + normalization stats).
- **L30-52** Sidecar `.meta.json` records the full provenance: **`training_contract`** (checked by the evaluator), **`policy_distribution: tanh_squashed_normal`** (self-documenting that the log-prob math is the tanh-squashed one), reward source, env, dims, `hidden_sizes`, seeds, step budget, wall time, and the whole training config. This metadata is what makes every checkpoint auditable.

## `train_single_seed` — config load & validation (L57–100)

```python
57: def train_single_seed(seed, config):
58:     env_name = config.get("env_name", "Go1JoystickFlatTerrain")
59:     num_envs = config.get("num_envs", 32)
60:     epochs = config.get("epochs", 50)
61:     steps_per_epoch = config.get("steps_per_epoch", 4096)
62:     batch_size = config.get("batch_size", 256)
63:     train_iters = config.get("train_iters", 10)
64:     gamma = config.get("gamma", 0.99)
65:     lam = config.get("lam", 0.95)
66:     clip_ratio = config.get("clip_ratio", 0.2)
67:     max_grad_norm = config.get("max_grad_norm", 0.5)
68:     hidden_dim = config.get("hidden_dim", 256)
69:     hidden_sizes = tuple(config.get("hidden_sizes", (hidden_dim, hidden_dim)))
70:     pi_lr = config.get("pi_lr", 0.0003)
71:     initial_log_std = config.get("initial_log_std", -0.5)
72:     ent_coef_start = config.get("ent_coef", 0.01)
73:     ent_coef_final = config.get("ent_coef_final", ent_coef_start)
```

- **L57-73** Pull every hyperparameter from the YAML. Champion values: `num_envs 8192`, `epochs 1220`, `steps_per_epoch 163840`, `batch_size 5120`, `train_iters 4`, `gamma 0.97`, `lam 0.95`, `clip_ratio 0.2`, `max_grad_norm 1.0`, `hidden_sizes [512,256,128]`, `pi_lr 3e-4`, `initial_log_std −1.0`, `ent_coef 0.01`, `ent_coef_final 0.01` (no annealing).

```python
75:     if epochs < 1 or num_envs < 1 or steps_per_epoch < 1:
76:         raise ValueError("epochs, num_envs, and steps_per_epoch must be positive")
77:     if steps_per_epoch % num_envs:
78:         raise ValueError("steps_per_epoch must be divisible by num_envs")
79:     configured_steps = config.get("total_timesteps_per_seed")
80:     actual_steps = epochs * steps_per_epoch
81:     if configured_steps is not None and int(configured_steps) != actual_steps:
82:         raise ValueError(
83:             "total_timesteps_per_seed must equal epochs * steps_per_epoch: "
84:             f"{configured_steps} != {actual_steps}"
85:         )
```

- **L75-85** **Reproducibility guards.** The config's declared step budget must equal `epochs × steps_per_epoch` exactly; otherwise training refuses to start. This makes "200M steps" a *checked* fact, not an aspiration.

## Seeding, device, env, agent (L87–135)

```python
87:     np.random.seed(seed)
88:     torch.manual_seed(seed)
89:     torch.cuda.manual_seed_all(seed)
90:
91:     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
...
96:     env = MJXVectorPyTorchWrapper(
97:         env_name, num_envs=num_envs, seed=seed,
98:         history_len=config.get("history_len", 5),
99:         episode_length=config.get("episode_length", 1000),
100:        config_overrides=config.get("environment_overrides"),
101:    )
102:    obs_dim = env.observation_dim          # 48
103:    critic_obs_dim = env.privileged_observation_dim   # 123
104:    act_dim = env.action_dim              # 12
105:
106:    agent = ActorCritic(
107:        obs_dim, act_dim, hidden_dim,
108:        initial_log_std=initial_log_std,
109:        critic_obs_dim=critic_obs_dim,
110:        hidden_sizes=hidden_sizes,
111:    ).to(device)
```

- **L87-89** Fully seed NumPy, CPU torch, and every CUDA device — deterministic per seed.
- **L96-101** Environment with the champion `history_len: 1`, `episode_length: 1000`, 8192 parallel robots.
- **L102-104** Dims come from the *environment*, not hardcoded — one source of truth.
- **L106-111** Build the asymmetric actor-critic on the target device.

```python
114:    initial_steps = 0
115:    resume_checkpoint = config.get("resume_checkpoint")
116:    if config.get("resume") and not resume_checkpoint:
117:        auto_ckpt = os.path.join(checkpoint_dir, f"ppo_seed{seed}.pt")
...
121:    if resume_checkpoint and os.path.exists(resume_checkpoint):
122:        state_dict = torch.load(resume_checkpoint, map_location=device, weights_only=True)
123:        agent.load_state_dict(state_dict, strict=True)
...
135:    optimizer = torch.optim.Adam(agent.parameters(), lr=pi_lr)
```

- **L114-134** Optional resume: loads weights with `strict=True` and recovers `total_env_steps` from the metadata sidecar so the step counter stays truthful across restarts.
- **L135** Adam at `lr=3e-4` (the classic PPO learning rate).

## Rollout buffer pre-allocation (L137–179)

```python
137:    num_steps = steps_per_epoch // num_envs        # 163840 / 8192 = 20
138:    total_steps = initial_steps
139:    start_time = time.time()
140:
141:    obs, reset_info = env.reset(seed=seed)
142:    critic_obs = reset_info["privileged_observation"]
143:    agent.update_observation_stats(obs.to(device), critic_obs.to(device))
...
168:    # Pre-allocate contiguous GPU rollout buffers to eliminate thousands of CUDA allocation kernels per epoch
169:    obs_buffer = torch.empty((num_steps, num_envs, obs_dim), device=device)
170:    critic_obs_buffer = torch.empty((num_steps, num_envs, critic_obs_dim), device=device)
171:    actions_buffer = torch.empty((num_steps, num_envs, act_dim), device=device)
172:    log_probs_buffer = torch.empty((num_steps, num_envs), device=device)
173:    values_buffer = torch.empty((num_steps, num_envs), device=device)
174:    rewards_buffer = torch.empty((num_steps, num_envs), device=device)
175:    terminated_buffer = torch.empty((num_steps, num_envs), device=device, dtype=torch.bool)
176:    truncated_buffer = torch.empty((num_steps, num_envs), device=device, dtype=torch.bool)
177:    truncation_values_buffer = torch.empty((num_steps, num_envs), device=device)
```

- **L137** 20 steps × 8192 envs = 163,840 transitions per epoch.
- **L141-143** First obs seeds the running normalization before anything is rolled out.
- **L169-177** **One-time contiguous GPU allocation** for every tensor the epoch needs (including separate `truncated` and `truncation_values` buffers for the GAE fork). No per-step allocation → no allocator churn.

## Per-epoch LR/entropy scheduling (L189–204)

```python
189:        if epoch > 1:
190:            last_kl = history["approx_kls"][-1]
191:            target_kl = config.get("target_kl", 0.02)
192:            if last_kl > target_kl * 2.0:
193:                learning_rate = max(learning_rate / 1.5, 1e-5)
194:            elif last_kl < target_kl / 2.0:
195:                learning_rate = min(learning_rate * 1.5, pi_lr)
196:
197:        for parameter_group in optimizer.param_groups:
198:            parameter_group["lr"] = learning_rate
199:
200:        schedule_fraction = (epoch - 1) / max(epochs - 1, 1)
201:        entropy_coefficient = (
202:            ent_coef_start
203:            + schedule_fraction * (ent_coef_final - ent_coef_start)
204:        )
```

- **L189-198** **KL-based LR adaptation:** if last epoch's KL exceeded `2×target_kl`, halve-ish the LR (`/1.5`, floored at `1e-5`); if it was under `target_kl/2`, grow it back (capped at `pi_lr`). This keeps policy drift bounded *between* epochs too, complementing the intra-epoch early stop.
- **L200-204** Linear entropy annealing slot; the champion pins `ent_coef_start == ent_coef_final = 0.01`, so this is a no-op there.

## Rollout loop (L206–245)

```python
206:        for step in range(num_steps):
207:            with torch.no_grad():
208:                obs_dev = obs.to(device)
209:                critic_obs_dev = critic_obs.to(device)
210:                action, log_prob = agent.get_action(obs_dev)
211:                value = agent.get_value(obs_dev, critic_obs_dev)
212:
213:            next_obs, reward, done, truncated_step, step_info = env.step(action)
214:            truncation_value = torch.zeros(num_envs, device=device)
215:            if bool(truncated_step.any()):
216:                with torch.no_grad():
217:                    final_values = agent.get_value(
218:                        step_info["final_observation"].to(device),
219:                        step_info["final_privileged_observation"].to(device),
220:                    )
221:                truncation_value = torch.where(
222:                    truncated_step.to(device), final_values, truncation_value
223:                )
224:
225:            obs_buffer[step] = obs_dev
226:            critic_obs_buffer[step] = critic_obs_dev
227:            actions_buffer[step] = action
228:            rewards_buffer[step] = reward.to(device)
229:            terminated_buffer[step] = done.to(device)
230:            truncated_buffer[step] = truncated_step.to(device)
231:            truncation_values_buffer[step] = truncation_value
232:            log_probs_buffer[step] = log_prob
233:            values_buffer[step] = value
```

- **L207-211** Inference is under `no_grad`: `get_action` yields the bounded action **and its squashed log-prob**; `get_value` yields `V(s_t)`.
- **L213** Physics step — the wrapper distinguishes `done` (fall) from `truncated_step` (time limit).
- **L214-223** **Truncation bootstrap computed at rollout time:** for every slot that just hit the time limit, evaluate `V(s_final)` on the *pre-autoreset* observation (`final_observation`/`final_privileged_observation`). Non-truncated slots get 0. This value flows into `compute_gae`'s `truncation_values` so time-limit returns are bootstrapped but falls are not.
- **L225-233** Everything lands in the pre-allocated GPU buffers, one row per step.

```python
235:            epoch_rewards.append(reward.mean().item())
236:            epoch_linear_velocity_errors.append(step_info["linear_velocity_error"].mean().item())
...
243:            total_steps += num_envs
244:            obs = next_obs
245:            critic_obs = step_info["privileged_observation"]
```

- **L235-243** Rolling telemetry (reward, LinErr, YawErr, |action|, % positive reward, % falls) plus the step counter.
- **L244-245** Advance the "current" observation for the next step.

## GAE + update + stats (L247–281)

```python
247:        with torch.no_grad():
248:            next_value = agent.get_value(obs.to(device), critic_obs.to(device))
249:
250:        advantages, returns = compute_gae(
251:            rewards_buffer, values_buffer, terminated_buffer, next_value,
252:            gamma=gamma, lam=lam, truncated=truncated_buffer,
253:            truncation_values=truncation_values_buffer,
254:        )
255:
256:        obs_tensor = obs_buffer.flatten(0, 1)
257:        critic_obs_tensor = critic_obs_buffer.flatten(0, 1)
258:        act_tensor = actions_buffer.flatten(0, 1)
259:        old_log_probs = log_probs_buffer.flatten(0, 1)
260:        adv_tensor = advantages.flatten(0, 1)
261:        ret_tensor = returns.flatten(0, 1)
262:
263:        update_metrics = update(
264:            agent=agent, optimizer=optimizer,
265:            observations=obs_tensor, actions=act_tensor,
266:            old_log_probs=old_log_probs, returns=ret_tensor,
267:            advantages=adv_tensor, critic_observations=critic_obs_tensor,
268:            old_values=values_buffer.flatten(0, 1),
269:            epochs=train_iters, batch_size=batch_size, clip_ratio=clip_ratio,
270:            max_grad_norm=max_grad_norm, vf_coef=config.get("vf_coef", 0.5),
271:            ent_coef=entropy_coefficient, target_kl=config.get("target_kl", 0.02),
272:        )
273:        agent.update_observation_stats(obs_tensor, critic_obs_tensor)
```

- **L247-248** Bootstrap value for the *last* stored state.
- **L250-254** One GAE pass over the full `[20, 8192]` rollout with truncation-aware bootstrapping (the termination/truncation fork from `ppo.py`).
- **L256-261** Flatten `[time, env] → [time·env]` (163,840 rows) for minibatching.
- **L263-272** The batched update: 4 epochs over 32 minibatches of 5120, clipped value loss with old values, gradient clipping at 1.0, and KL early stop at 0.03.
- **L273** After the update, fold the whole epoch's observations into the running normalization moments (so normalization and policy co-evolve).

## History, logging, checkpointing (L283–366)

```python
286:        history["epochs"].append(epoch)
...
314:        history["entropy_coefficients"].append(float(entropy_coefficient))
315:        history["learning_rates"].append(float(learning_rate))
...
338:        # Auto-save main seed checkpoint every 10 epochs and at completion
339:        if epoch % 10 == 0 or epoch == epochs:
340:            checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
341:            os.makedirs(checkpoint_dir, exist_ok=True)
342:            ckpt_path = os.path.join(checkpoint_dir, f"ppo_seed{seed}.pt")
343:            save_checkpoint(ckpt_path, agent, ...)
```

- **L286-318** A full measurement history per epoch: losses, KL, clip fraction, LinErr/YawErr, `action_stds` (the **actual** `σ = exp(clamp(log_std))` from the policy — proof the entropy coefficient did its job), LR, steps, wall time.
- **L320-335** Human-readable progress line every 5 epochs (and at the end).
- **L337-366** Checkpoint every 10 epochs and always at completion, each with auditable `.meta.json`.

## `main` + CLI (L371–423)

```python
371: def main(config_path="configs/default.yaml", seed_override=None, resume_override=False):
372:     with open(config_path, "r") as f:
373:         config = yaml.safe_load(f)
374:
375:     if resume_override:
376:         config["resume"] = True
377:
378:     seeds = [int(seed_override)] if seed_override is not None else config.get(
379:         "seeds", [10, 11, 21, 67, 96]
380:     )
381:     config = {**config, "seeds": seeds, "seed": seeds[0]}
382:     expected_total = int(config.get("total_timesteps_per_seed", 0)) * len(seeds)
...
396:    for s in seeds:
397:        hist = train_single_seed(s, config)
398:        all_histories.append(hist)
399:        # Persist after every seed so an interrupted multi-hour run keeps all
400:        # completed measurements.
401:        ...json.dump(all_histories, f, indent=2)
...
406:    with open(output_path, "w") as f:
407:        json.dump(all_histories, f, indent=2)
```

- **L378-381** Multi-seed driver: one config, N seeds. The `total_timesteps` cross-check (L382-390) ensures the declared budget matches `per_seed × seed_count`.
- **L396-404** Trains each seed sequentially and **persists results after every seed**, so a killed multi-hour run never loses completed seeds.
- **L406-414** Final combined JSON + summary.

---

# Hyperparameter reference (`configs/champion_v2.yaml`)

| Param | Value | Where it's used |
| :--- | :---: | :--- |
| `gamma` / `lam` | `0.97` / `0.95` | `compute_gae` (ppo.py:56-57) |
| `clip_ratio` | `0.2` | `ppo_clip_loss` + `value_loss` (ppo.py:67,74) |
| `target_kl` | `0.02` | early stop `1.5×` (ppo.py:173) + LR adaptation |
| `max_grad_norm` | `1.0` | gradient clip (ppo.py:154) |
| `train_iters` / `batch_size` | `4` / `5120` | minibatch loop (ppo.py:123-127) |
| `vf_coef` / `ent_coef` | `0.5` / `0.01` | total loss (ppo.py:139) |
| `initial_log_std` | `-1.0` | `ActorCritic` (agent.py:47) |
| `hidden_sizes` | `[512, 256, 128]` | `_make_mlp` (agent.py:46,48) |
| `num_envs` / `steps_per_epoch` | `8192` / `163840` | rollout (train_multi_seed.py:137) |
| `epochs` | `1220` | → **199,884,800 steps/seed** |
| `pi_lr` | `3e-4` | Adam (train_multi_seed.py:135) |
