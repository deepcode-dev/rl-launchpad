import numpy as np

from baselines.sb3_ppo import RolloutMetricsCallback, SB3MJXVecEnv


def test_sb3_vector_adapter_preserves_time_limit_terminal_observation():
    env = SB3MJXVecEnv(
        "Go1JoystickFlatTerrain", seed=17, num_envs=2,
        history_len=2, episode_length=1,
    )
    try:
        obs = env.reset()
        assert obs.shape == (2, 96)
        env.step_async(np.zeros((2, 12), dtype=np.float32))
        next_obs, rewards, dones, infos = env.step_wait()
        assert next_obs.shape == obs.shape
        assert rewards.shape == (2,)
        assert dones.tolist() == [True, True]
        for info in infos:
            assert info["TimeLimit.truncated"] is True
            assert info["terminal_observation"].shape == (96,)
    finally:
        env.close()


def test_sb3_callback_logs_raw_step_rewards():
    callback = RolloutMetricsCallback()
    callback._on_rollout_start()
    callback.locals = {"rewards": np.asarray([1.0, 3.0])}
    assert callback._on_step() is True
    callback.num_timesteps = 2
    callback._on_rollout_end()
    assert callback.history["total_steps"] == [2]
    assert callback.history["mean_step_rewards"] == [2.0]
