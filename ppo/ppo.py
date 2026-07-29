import torch
import torch.nn.functional as F

TRAINING_CONTRACT = "custom-ppo-privileged-critic-tracking-v3"


def compute_gae(
    rewards,
    values,
    terminated,
    next_value,
    gamma=0.99,
    lam=0.95,
    truncated=None,
    truncation_values=None,
):
    """Compute GAE for a vectorized rollout without mixing environment streams."""
    rewards = torch.as_tensor(rewards)
    values = torch.as_tensor(values, device=rewards.device, dtype=rewards.dtype)
    terminated = torch.as_tensor(terminated, device=rewards.device, dtype=torch.bool)
    next_value = torch.as_tensor(next_value, device=rewards.device, dtype=rewards.dtype)

    if rewards.ndim != 2:
        raise ValueError(f"rewards must have shape [time, env], got {tuple(rewards.shape)}")
    if values.shape != rewards.shape or terminated.shape != rewards.shape:
        raise ValueError("values and terminated must match rewards shape [time, env]")
    if next_value.shape != rewards.shape[1:]:
        raise ValueError(f"next_value must have shape {tuple(rewards.shape[1:])}")

    if truncated is None:
        truncated = torch.zeros_like(terminated)
    else:
        truncated = torch.as_tensor(truncated, device=rewards.device, dtype=torch.bool)
        if truncated.shape != rewards.shape:
            raise ValueError("truncated must match rewards shape [time, env]")

    if truncation_values is None:
        if truncated.any():
            raise ValueError("truncation_values are required for time-limit bootstrapping")
        truncation_values = torch.zeros_like(values)
    else:
        truncation_values = torch.as_tensor(
            truncation_values, device=rewards.device, dtype=rewards.dtype
        )
        if truncation_values.shape != rewards.shape:
            raise ValueError("truncation_values must match rewards shape [time, env]")

    continuation_mask = (~(terminated | truncated)).to(rewards.dtype)
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(next_value)
    next_values = torch.cat((values[1:], next_value.unsqueeze(0)), dim=0)

    for t in range(rewards.shape[0] - 1, -1, -1):
        bootstrap_value = torch.where(truncated[t], truncation_values[t], next_values[t])
        bootstrap_mask = (~terminated[t]).to(rewards.dtype)
        delta = rewards[t] + gamma * bootstrap_value * bootstrap_mask - values[t]
        gae = delta + gamma * lam * continuation_mask[t] * gae
        advantages[t] = gae

    return advantages, advantages + values


def ppo_clip_loss(log_probs, old_log_probs, advantages, clip_ratio=0.2):
    """Compute the standard PPO clipped surrogate objective."""
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    return -torch.min(surr1, surr2).mean()


def value_loss(values, returns, old_values=None, clip_ratio=None):
    """Compute clipped or unclipped Huber value-function error."""
    if old_values is not None and clip_ratio is not None:
        v_clipped = old_values + torch.clamp(values - old_values, -clip_ratio, clip_ratio)
        v_loss1 = F.huber_loss(values, returns, reduction="none")
        v_loss2 = F.huber_loss(v_clipped, returns, reduction="none")
        return torch.max(v_loss1, v_loss2).mean()
    return F.huber_loss(values, returns)


def update(
    agent,
    optimizer,
    observations,
    actions,
    old_log_probs,
    returns,
    advantages,
    critic_observations=None,
    old_values=None,
    epochs=10,
    batch_size=256,
    clip_ratio=0.2,
    max_grad_norm=0.5,
    vf_coef=0.5,
    ent_coef=0.01,
    target_kl=None,
):
    """Update PPO over an already-flattened ``[time * env, ...]`` rollout."""
    dataset_size = observations.shape[0]
    if dataset_size == 0 or batch_size <= 0:
        raise ValueError("rollout and batch_size must be non-empty")
    critic_observations = observations if critic_observations is None else critic_observations
    if any(
        tensor.shape[0] != dataset_size
        for tensor in (
            actions,
            old_log_probs,
            returns,
            advantages,
            critic_observations,
        )
    ):
        raise ValueError("all PPO rollout tensors must have the same leading dimension")

    totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0}
    num_updates = 0
    epochs_completed = 0
    early_stopped = False

    uncompiled_agent = getattr(agent, "_orig_mod", agent)

    for _ in range(epochs):
        epoch_kls = []
        indices = torch.randperm(dataset_size, device=observations.device)
        for start in range(0, dataset_size, batch_size):
            batch_idx = indices[start : start + batch_size]
            new_log_probs, entropy, values = agent.evaluate(
                observations[batch_idx],
                actions[batch_idx],
                critic_observations[batch_idx],
            )
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_advantages = advantages[batch_idx]
            batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std(unbiased=False) + 1e-8)
            batch_old_values = old_values[batch_idx] if old_values is not None else None

            pol_loss = ppo_clip_loss(new_log_probs, batch_old_log_probs, batch_advantages, clip_ratio)
            val_loss = value_loss(values, returns[batch_idx], old_values=batch_old_values, clip_ratio=clip_ratio)
            entropy_bonus = entropy.mean()
            total_loss = pol_loss + vf_coef * val_loss - ent_coef * entropy_bonus

            optimizer.zero_grad(set_to_none=True)
            if torch.isfinite(total_loss):
                total_loss.backward()
                # Safeguard: verify all gradients are finite before optimizer step
                grads_ok = True
                for p in agent.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        grads_ok = False
                        break
                if grads_ok:
                    if max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                    optimizer.step()

            # Absolute Parameter Protection: Ensure actor_log_std parameter data never drops below -1.2 (sigma >= 0.301)
            with torch.no_grad():
                uncompiled_agent.actor_log_std.nan_to_num_(nan=-1.0, posinf=0.5, neginf=-1.2).clamp_(-1.2, 0.5)

            log_ratio = new_log_probs - batch_old_log_probs
            totals["policy_loss"] += pol_loss.item()
            totals["value_loss"] += val_loss.item()
            totals["entropy"] += entropy_bonus.item()
            ratio = log_ratio.exp()
            # Standard numerically stable Schulman KL estimate
            approx_kl = (batch_old_log_probs - new_log_probs).mean().abs().item()
            totals["approx_kl"] += approx_kl
            epoch_kls.append(approx_kl)
            totals["clip_fraction"] += ((ratio - 1.0).abs() > clip_ratio).float().mean().item()
            num_updates += 1

            if target_kl is not None and approx_kl > 1.5 * float(target_kl):
                early_stopped = True
                break

        epochs_completed += 1
        if early_stopped:
            break

    result = {name: value / num_updates for name, value in totals.items()}
    result["update_epochs"] = epochs_completed
    result["early_stopped"] = early_stopped
    return result
