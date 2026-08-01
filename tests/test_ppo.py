import pytest
import torch

from ppo.agent import ActorCritic
from ppo.ppo import compute_gae, ppo_clip_loss, update, value_loss


def test_vectorized_gae_keeps_environment_trajectories_separate():
    # Columns are independent environments.  A flattened implementation would
    # incorrectly use env 1's values as env 0's next-time-step values.
    rewards = torch.tensor([[1.0, 10.0], [2.0, 20.0]])
    values = torch.zeros_like(rewards)
    terminated = torch.tensor([[False, False], [True, False]])
    next_value = torch.tensor([3.0, 30.0])

    advantages, returns = compute_gae(
        rewards, values, terminated, next_value, gamma=1.0, lam=1.0
    )

    expected = torch.tensor([[3.0, 60.0], [2.0, 50.0]])
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, expected)


def test_gae_bootstraps_truncation_without_crossing_reset_and_rejects_flat_rollouts():
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.zeros_like(rewards)
    truncated = torch.tensor([[False], [True]])
    advantages, _ = compute_gae(
        rewards, values, torch.zeros_like(truncated), torch.tensor([99.0]),
        gamma=1.0, lam=1.0, truncated=truncated,
        truncation_values=torch.tensor([[0.0], [5.0]]),
    )
    torch.testing.assert_close(advantages, torch.tensor([[8.0], [7.0]]))

    with pytest.raises(ValueError, match="truncation_values"):
        compute_gae(
            rewards, values, torch.zeros_like(truncated), torch.tensor([99.0]),
            truncated=truncated,
        )

    with pytest.raises(ValueError, match=r"\[time, env\]"):
        compute_gae(torch.ones(3), torch.ones(3), torch.zeros(3), torch.tensor(0.0))


def test_losses_and_update_return_finite_metrics():
    torch.manual_seed(0)
    log_probs = torch.randn(64)
    old_log_probs = torch.randn(64)
    advantages = torch.randn(64)
    target_returns = torch.randn(64)
    current_values = torch.randn(64)
    assert torch.isfinite(ppo_clip_loss(log_probs, old_log_probs, advantages))
    assert torch.isfinite(value_loss(current_values, target_returns))

    agent = ActorCritic(obs_dim=8, act_dim=3, hidden_dim=32)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    observations = torch.randn(64, 8)
    with torch.no_grad():
        actions, rollout_log_probs = agent.get_action(observations)
        returns = agent.get_value(observations) + torch.randn(64)

    metrics = update(
        agent, optimizer, observations, actions, rollout_log_probs, returns,
        torch.randn(64), epochs=2, batch_size=16,
    )
    assert set(metrics) == {
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "update_epochs",
        "early_stopped",
    }
    numeric_metrics = {key: value for key, value in metrics.items() if key != "early_stopped"}
    assert all(torch.isfinite(torch.tensor(value)) for value in numeric_metrics.values())
