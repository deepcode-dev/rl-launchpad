import torch

from ppo.agent import ActorCritic


def test_actor_critic_shapes_bounded_actions_and_log_prob_round_trip():
    torch.manual_seed(0)
    obs_dim, act_dim, batch_size = 103, 29, 32
    agent = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=64)
    observations = torch.randn(batch_size, obs_dim)

    actions, rollout_log_prob = agent.get_action(observations)
    assert actions.shape == (batch_size, act_dim)
    assert rollout_log_prob.shape == (batch_size,)
    assert torch.all(actions >= -1.0) and torch.all(actions <= 1.0)

    evaluated_log_prob, entropy, value = agent.evaluate(observations, actions)
    torch.testing.assert_close(evaluated_log_prob, rollout_log_prob, atol=1e-5, rtol=1e-5)
    assert entropy.shape == (batch_size,)
    assert value.shape == (batch_size,)
    assert agent.get_value(observations).shape == (batch_size,)


def test_deterministic_action_is_bounded():
    agent = ActorCritic(obs_dim=4, act_dim=2, hidden_dim=16)
    with torch.no_grad():
        agent.actor[-1].bias.fill_(100.0)
    action, _ = agent.get_action(torch.zeros(1, 4), deterministic=True)
    assert torch.all(action <= 1.0)
    assert torch.all(action >= -1.0)


def test_low_initial_action_std_is_not_clamped_to_old_floor():
    agent = ActorCritic(
        obs_dim=4,
        act_dim=2,
        hidden_dim=16,
        initial_log_std=-1.9,
    )
    distribution = agent._distribution(torch.zeros(1, 4))
    torch.testing.assert_close(
        distribution.scale,
        torch.full((1, 2), torch.exp(torch.tensor(-1.9))),
    )


def test_observation_normalization_round_trips_in_state_dict():
    agent = ActorCritic(obs_dim=4, act_dim=2, hidden_dim=16)
    observations = torch.tensor([[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]])
    agent.update_observation_stats(observations)
    restored = ActorCritic(obs_dim=4, act_dim=2, hidden_dim=16)
    restored.load_state_dict(agent.state_dict())
    torch.testing.assert_close(restored.obs_mean, agent.obs_mean)
    torch.testing.assert_close(restored.obs_var, agent.obs_var)
    torch.testing.assert_close(restored.obs_count, agent.obs_count)


def test_asymmetric_critic_accepts_privileged_observations():
    agent = ActorCritic(
        obs_dim=48,
        act_dim=12,
        critic_obs_dim=123,
        hidden_sizes=(64, 32),
    )
    observations = torch.randn(8, 48)
    privileged = torch.randn(8, 123)
    actions, _ = agent.get_action(observations)
    _, _, values = agent.evaluate(observations, actions, privileged)
    assert values.shape == (8,)
    agent.update_observation_stats(observations, privileged)
    assert agent.critic_obs_count > 8
