# RL Launchpad — Griffin Labs RL Track

From-scratch PPO implementation on MuJoCo Playground.

## Team
- Person A: PPO implementation
- Person B: Baselines & evaluation
- Person C: Documentation & write-up

## Environment
- Simulator: MuJoCo Playground (MJX)
- Algorithm: PPO-Clip (from scratch, PyTorch)
- Baseline: Stable-Baselines3 PPO

## Getting Started
```bash
uv sync
python ppo/train.py

# running eval
python eval/evaluate.py --checkpoint checkpoints/ppo_final.pt

# running baseline
python baselines/sb3_ppo.py
