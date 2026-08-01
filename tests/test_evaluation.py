import json

import pytest
import torch

from eval.evaluate import load_actor_critic_checkpoint
from eval.plot_benchmark import load_summary, load_training_histories
from ppo.agent import ActorCritic
from ppo.ppo import TRAINING_CONTRACT


def _write_checkpoint(tmp_path, metadata):
    path = tmp_path / "policy.pt"
    torch.save(ActorCritic(8, 3, 16).state_dict(), path)
    (tmp_path / "policy.pt.meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    return path


def test_checkpoint_loader_requires_current_training_contract(tmp_path):
    path = _write_checkpoint(tmp_path, {})
    with pytest.raises(ValueError, match="missing the verified training contract"):
        load_actor_critic_checkpoint(
            path, env_name="Go1JoystickFlatTerrain", obs_dim=8, act_dim=3, hidden_dim=16
        )


def test_checkpoint_loader_accepts_matching_metadata_and_rejects_env_mismatch(tmp_path):
    metadata = {
        "training_contract": TRAINING_CONTRACT,
        "env_name": "Go1JoystickFlatTerrain",
        "obs_dim": 8,
        "act_dim": 3,
        "hidden_dim": 16,
    }
    path = _write_checkpoint(tmp_path, metadata)
    agent, loaded = load_actor_critic_checkpoint(
        path, env_name="Go1JoystickFlatTerrain", obs_dim=8, act_dim=3, hidden_dim=16
    )
    assert isinstance(agent, ActorCritic)
    assert loaded == metadata

    with pytest.raises(ValueError, match="environment_name|env_name"):
        load_actor_critic_checkpoint(
            path, env_name="G1JoystickFlatTerrain", obs_dim=8, act_dim=3, hidden_dim=16
        )


def test_plot_inputs_must_be_recorded_and_complete(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_summary(tmp_path / "missing.json", "custom-ppo-native-reward-v1")
    with pytest.raises(FileNotFoundError):
        load_training_histories(tmp_path / "missing-training.json")

    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({"protocol": "custom-ppo-native-reward-v1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_summary(incomplete, "custom-ppo-native-reward-v1")
