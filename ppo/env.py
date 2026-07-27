"""A small PyTorch-facing, vectorized wrapper for MuJoCo Playground MJX tasks.

The wrapper deliberately keeps the task's reward untouched.  It exposes a
fixed ``[-1, 1]`` action contract, handles per-environment episode boundaries,
and returns CPU ``torch.float32`` tensors suitable for the PPO implementation.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
import mujoco_playground as mp


class MJXVectorPyTorchWrapper:
    """Vectorized MJX environment with a minimal Gymnasium-compatible API.

    ``step`` auto-resets only completed slots.  The observation returned for a
    completed slot is therefore its reset observation, while ``done`` or
    ``truncated`` describes the transition that just ended.
    """

    action_low = -1.0
    action_high = 1.0

    def __init__(
        self,
        env_name: str = "Go1JoystickFlatTerrain",
        num_envs: int = 32,
        seed: int = 42,
        history_len: int = 5,
        episode_length: int | None = None,
        config_overrides: dict[str, Any] | None = None,
    ):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if history_len < 1:
            raise ValueError("history_len must be positive")

        self.env_name = env_name
        self.num_envs = int(num_envs)
        self.history_len = int(history_len)
        self.cfg, self.env = self._load_environment(env_name, config_overrides)
        self.episode_length = int(
            episode_length if episode_length is not None else getattr(self.cfg, "episode_length", 1000)
        )
        if self.episode_length < 1:
            raise ValueError("episode_length must be positive")

        self._v_reset = jax.jit(jax.vmap(self.env.reset))
        self._v_step = jax.jit(jax.vmap(self.env.step))
        self._slot_keys: jax.Array | None = None
        self.states = None
        self._initial_states = None
        self.obs_history: jax.Array | None = None
        self._episode_steps = jnp.zeros(self.num_envs, dtype=jnp.int32)
        self._jax_platform = jax.devices()[0].platform
        self.reset(seed=seed)

    @staticmethod
    def _load_environment(
        env_name: str,
        config_overrides: dict[str, Any] | None = None,
    ):
        """Load through the appropriate registry rather than exception routing."""
        if env_name in mp.locomotion.ALL_ENVS:
            cfg = mp.locomotion.get_default_config(env_name)
            cfg.impl = "jax"
            return cfg, mp.locomotion.load(
                env_name,
                config=cfg,
                config_overrides=config_overrides,
            )
        if env_name in mp.manipulation.ALL_ENVS:
            cfg = mp.manipulation.get_default_config(env_name)
            cfg.impl = "jax"
            return cfg, mp.manipulation.load(
                env_name,
                config=cfg,
                config_overrides=config_overrides,
            )
        raise ValueError(
            f"Unknown MuJoCo Playground environment {env_name!r}. "
            f"Known locomotion environments: {mp.locomotion.ALL_ENVS}; "
            f"known manipulation environments: {mp.manipulation.ALL_ENVS}"
        )

    @property
    def base_observation_dim(self) -> int:
        size = self.env.observation_size
        return int(size["state"][0] if isinstance(size, dict) else size)

    @property
    def privileged_observation_dim(self) -> int:
        size = self.env.observation_size
        if isinstance(size, dict) and "privileged_state" in size:
            return int(size["privileged_state"][0])
        return self.base_observation_dim

    @property
    def observation_dim(self) -> int:
        return self.base_observation_dim * self.history_len

    @property
    def action_dim(self) -> int:
        return int(self.env.action_size)

    def _obs_array(self, obs_raw: Any) -> jax.Array:
        state = obs_raw["state"] if isinstance(obs_raw, dict) else obs_raw
        return jnp.asarray(state, dtype=jnp.float32)

    def _privileged_obs_array(self, obs_raw: Any) -> jax.Array:
        if isinstance(obs_raw, dict):
            state = obs_raw.get("privileged_state", obs_raw["state"])
        else:
            state = obs_raw
        return jnp.asarray(state, dtype=jnp.float32)

    @staticmethod
    def _to_tensor(array) -> torch.Tensor:
        """Share JAX buffers with PyTorch via DLPack GPU zero-copy; fall back to numpy for CPU."""
        try:
            return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(array))
        except Exception:
            return torch.from_numpy(np.array(array, copy=True, order="C"))

    def _to_jax_action(self, action) -> jax.Array:
        try:
            return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(action))
        except Exception:
            if isinstance(action, torch.Tensor):
                return jnp.asarray(action.detach().cpu().numpy(), dtype=jnp.float32)
            return jnp.asarray(action, dtype=jnp.float32)

    def _split_reset_keys(self) -> jax.Array:
        assert self._slot_keys is not None
        keys = jax.vmap(jax.random.split)(self._slot_keys)
        self._slot_keys = keys[:, 0]
        return keys[:, 1]

    @staticmethod
    def _select_finished(current, reset, finished):
        """Select reset values for each completed batch slot in a JAX pytree."""
        def select_leaf(old, new):
            mask = finished.reshape((finished.shape[0],) + (1,) * (old.ndim - 1))
            return jnp.where(mask, new, old)

        return jax.tree_util.tree_map(select_leaf, current, reset)

    def _flat_observation(self) -> jax.Array:
        assert self.obs_history is not None
        return self.obs_history.reshape(self.num_envs, -1)

    def reset(self, seed: int | None = None):
        if seed is not None or self._slot_keys is None:
            root_key = jax.random.PRNGKey(0 if seed is None else int(seed))
            self._slot_keys = jax.random.split(root_key, self.num_envs)
        self.states = self._v_reset(self._split_reset_keys())
        self._initial_states = jax.tree_util.tree_map(lambda value: value, self.states)
        self._last_ema_action = None
        obs = self._obs_array(self.states.obs)
        self.obs_history = jnp.repeat(obs[:, None, :], self.history_len, axis=1)
        self._episode_steps = jnp.zeros(self.num_envs, dtype=jnp.int32)
        privileged_obs = self._privileged_obs_array(self.states.obs)
        return self._to_tensor(self._flat_observation()), {
            "privileged_observation": self._to_tensor(privileged_obs),
        }

    def step(self, action):
        if self.states is None or self.obs_history is None:
            raise RuntimeError("Call reset() before step()")
        action_jax = self._to_jax_action(action)
        if action_jax.ndim == 1:
            action_jax = action_jax[None, :]
        if action_jax.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                f"Expected actions with shape {(self.num_envs, self.action_dim)}, got {action_jax.shape}"
            )
        # Low-pass EMA Action Filtering for smooth joint targets
        action_jax = jnp.clip(action_jax, self.action_low, self.action_high)
        if self._last_ema_action is None or self._last_ema_action.shape != action_jax.shape:
            self._last_ema_action = action_jax
        else:
            action_jax = 0.8 * self._last_ema_action + 0.2 * action_jax
            self._last_ema_action = action_jax

        terminal_states = self._v_step(self.states, action_jax)
        reward_array = jnp.asarray(terminal_states.reward, dtype=jnp.float32)
        natural_done = terminal_states.done.astype(jnp.bool_)
        self._episode_steps += 1
        time_limit = self._episode_steps >= self.episode_length
        truncated_np = time_limit & ~natural_done
        finished = natural_done | truncated_np

        terminal_obs = self._obs_array(terminal_states.obs)
        terminal_privileged_obs = self._privileged_obs_array(terminal_states.obs)
        terminal_history = jnp.roll(self.obs_history, shift=-1, axis=1)
        terminal_history = terminal_history.at[:, -1, :].set(terminal_obs)
        assert self._initial_states is not None
        reset_states = self._initial_states
        self.states = self._select_finished(terminal_states, reset_states, finished)
        reset_obs = self._obs_array(reset_states.obs)
        reset_privileged_obs = self._privileged_obs_array(reset_states.obs)
        next_obs = jnp.where(finished[:, None], reset_obs, terminal_obs)
        next_privileged_obs = jnp.where(
            finished[:, None], reset_privileged_obs, terminal_privileged_obs
        )
        self._episode_steps = jnp.where(finished, 0, self._episode_steps)
        reset_history = jnp.repeat(next_obs[:, None, :], self.history_len, axis=1)
        self.obs_history = jnp.where(
            finished[:, None, None], reset_history, terminal_history
        )

        command = jnp.asarray(terminal_states.info["command"], dtype=jnp.float32)
        # The privileged observation layout is defined by the stock Go1 task:
        # actor state, then gyro, accelerometer, gravity and true local velocity.
        privileged_offset = self.base_observation_dim
        true_gyro = terminal_privileged_obs[:, privileged_offset : privileged_offset + 3]
        true_local_velocity = terminal_privileged_obs[
            :, privileged_offset + 9 : privileged_offset + 12
        ]
        info = {
            # Available to callers that need terminal transitions despite autoreset.
            # Full policy observation before autoreset.  PPO uses this to
            # bootstrap time-limit truncations without crossing episodes.
            "final_observation": self._to_tensor(
                terminal_history.reshape(self.num_envs, -1)
            ),
            "final_observation_mask": self._to_tensor(finished),
            "privileged_observation": self._to_tensor(next_privileged_obs),
            "final_privileged_observation": self._to_tensor(terminal_privileged_obs),
            "command": self._to_tensor(command),
            "local_linear_velocity": self._to_tensor(true_local_velocity),
            "linear_velocity_error": self._to_tensor(
                jnp.linalg.norm(command[:, :2] - true_local_velocity[:, :2], axis=1)
            ),
            "yaw_rate_error": self._to_tensor(
                jnp.abs(command[:, 2] - true_gyro[:, 2])
            ),
            "mean_abs_action": self._to_tensor(
                jnp.mean(jnp.abs(action_jax), axis=1)
            ),
            "base_position_xy": self._to_tensor(
                jnp.asarray(terminal_states.data.qpos[:, :2], dtype=jnp.float32)
            ),
        }
        return (
            self._to_tensor(self._flat_observation()),
            self._to_tensor(reward_array),
            self._to_tensor(natural_done),
            self._to_tensor(truncated_np),
            info,
        )
