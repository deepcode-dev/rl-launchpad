import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization for neural network layers."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _make_mlp(input_dim, hidden_sizes, output_dim, output_std):
    layers = []
    previous_dim = input_dim
    for hidden_size in hidden_sizes:
        layers.extend((layer_init(nn.Linear(previous_dim, hidden_size)), nn.SiLU()))
        previous_dim = hidden_size
    layers.append(layer_init(nn.Linear(previous_dim, output_dim), std=output_std))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Actor-critic with a Gaussian with clipped actions."""

    _LOG_STD_MIN = -3.0
    _LOG_STD_MAX = 0.5

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_dim=256,
        initial_log_std=-0.5,
        *,
        critic_obs_dim=None,
        hidden_sizes=None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.critic_obs_dim = int(critic_obs_dim or obs_dim)
        self.hidden_sizes = tuple(int(size) for size in (hidden_sizes or (hidden_dim, hidden_dim)))
        self.actor = _make_mlp(obs_dim, self.hidden_sizes, act_dim, output_std=0.01)
        self.actor_log_std = nn.Parameter(torch.full((1, act_dim), float(initial_log_std)))
        self.critic = _make_mlp(self.critic_obs_dim, self.hidden_sizes, 1, output_std=1.0)
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_var", torch.ones(obs_dim))
        self.register_buffer("obs_count", torch.tensor(1e-4))
        self.register_buffer("critic_obs_mean", torch.zeros(self.critic_obs_dim))
        self.register_buffer("critic_obs_var", torch.ones(self.critic_obs_dim))
        self.register_buffer("critic_obs_count", torch.tensor(1e-4))

    def forward(self, obs, critic_obs=None):
        critic_obs = obs if critic_obs is None else critic_obs
        return self.actor(self._normalize_observation(obs)), self.critic(
            self._normalize_critic_observation(critic_obs)
        )

    def _normalize_observation(self, obs):
        normalized = (obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8)
        return normalized.clamp(-10.0, 10.0)

    def _normalize_critic_observation(self, obs):
        normalized = (obs - self.critic_obs_mean) / torch.sqrt(self.critic_obs_var + 1e-8)
        return normalized.clamp(-10.0, 10.0)

    @staticmethod
    def _merge_observation_stats(observations, mean, var, count):
        observations = observations.to(device=mean.device, dtype=mean.dtype)
        batch_mean = observations.mean(dim=0)
        batch_var = observations.var(dim=0, unbiased=False)
        batch_count = torch.tensor(observations.shape[0], device=count.device, dtype=count.dtype)
        delta = batch_mean - mean
        total_count = count + batch_count
        new_mean = mean + delta * batch_count / total_count
        current_m2 = var * count
        batch_m2 = batch_var * batch_count
        correction = delta.square() * count * batch_count / total_count
        mean.copy_(new_mean)
        var.copy_((current_m2 + batch_m2 + correction) / total_count)
        count.copy_(total_count)

    @torch.no_grad()
    def update_observation_stats(self, observations, critic_observations=None):
        """Merge a raw observation batch into persistent running moments."""
        if observations.ndim != 2 or observations.shape[1] != self.obs_dim:
            raise ValueError(f"observations must have shape [batch, {self.obs_dim}]")
        critic_observations = observations if critic_observations is None else critic_observations
        if (
            critic_observations.ndim != 2
            or critic_observations.shape[1] != self.critic_obs_dim
        ):
            raise ValueError(
                "critic_observations must have shape "
                f"[batch, {self.critic_obs_dim}]"
            )
        self._merge_observation_stats(
            observations, self.obs_mean, self.obs_var, self.obs_count
        )
        self._merge_observation_stats(
            critic_observations,
            self.critic_obs_mean,
            self.critic_obs_var,
            self.critic_obs_count,
        )

    def _distribution(self, obs):
        mean = self.actor(self._normalize_observation(obs))
        log_std = torch.clamp(
            self.actor_log_std,
            min=self._LOG_STD_MIN,
            max=self._LOG_STD_MAX,
        )
        std = torch.nan_to_num(log_std.exp(), nan=0.1, posinf=0.5, neginf=0.001).clamp(min=1e-3, max=1.0)
        return Normal(mean, std.expand_as(mean))

    def get_action(self, obs, deterministic=False):
        """Return an action in [-1, 1] and its log-probability."""
        dist = self._distribution(obs)
        unclamped_action = dist.mean if deterministic else dist.sample()
        action = unclamped_action.clamp(-1.0, 1.0)
        return action, dist.log_prob(action).sum(dim=-1)

    def evaluate(self, obs, action, critic_obs=None):
        """Evaluate bounded actions under the same Gaussian distribution."""
        dist = self._distribution(obs)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        critic_obs = obs if critic_obs is None else critic_obs
        value = self.critic(self._normalize_critic_observation(critic_obs)).squeeze(-1)
        return log_prob, entropy, value

    def get_value(self, obs, critic_obs=None):
        critic_obs = obs if critic_obs is None else critic_obs
        return self.critic(self._normalize_critic_observation(critic_obs)).squeeze(-1)
