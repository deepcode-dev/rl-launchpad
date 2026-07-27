"""Smoke tests and backwards-compatible import for the production wrapper."""

import torch

from ppo.env import MJXVectorPyTorchWrapper


def test_env_contract_smoke():
    """Runs on CPU JAX too; one slot avoids requiring a GPU."""
    env = MJXVectorPyTorchWrapper("Go1JoystickFlatTerrain", num_envs=1, seed=42, history_len=2)
    obs, info = env.reset(seed=42)
    assert info["privileged_observation"].shape == (1, env.privileged_observation_dim)
    assert obs.shape == (1, env.observation_dim)
    assert obs.dtype == torch.float32

    next_obs, reward, done, truncated, info = env.step(torch.full((1, env.action_dim), 2.0))
    assert next_obs.shape == obs.shape
    assert reward.shape == done.shape == truncated.shape == (1,)
    assert done.dtype == truncated.dtype == torch.bool
    assert "final_observation" in info
    assert info["final_observation"].shape == (1, env.observation_dim)
    assert info["privileged_observation"].shape == (1, env.privileged_observation_dim)
    assert info["linear_velocity_error"].shape == (1,)
    assert info["base_position_xy"].shape == (1, 2)


def test_env_seed_is_repeatable():
    env = MJXVectorPyTorchWrapper("Go1JoystickFlatTerrain", num_envs=1, seed=7)
    first, _ = env.reset(seed=123)
    second, _ = env.reset(seed=123)
    torch.testing.assert_close(first, second)


def test_time_limit_autoresets_and_resets_history():
    env = MJXVectorPyTorchWrapper(
        "Go1JoystickFlatTerrain", num_envs=1, seed=9, history_len=3, episode_length=1
    )
    obs, _ = env.reset(seed=9)
    next_obs, _, done, truncated, info = env.step(torch.zeros(1, env.action_dim))
    assert bool(done.item()) or bool(truncated.item())
    if not bool(done.item()):
        assert bool(truncated.item())
    # A reset slot has no stale pre-terminal frames in its history.
    frames = next_obs.reshape(1, env.history_len, env.base_observation_dim)
    torch.testing.assert_close(frames[:, 0], frames[:, -1])
    assert bool(info["final_observation_mask"].item())


if __name__ == "__main__":
    test_env_contract_smoke()
    print("Vectorized environment contract smoke test passed.")
