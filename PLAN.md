Here's the complete, properly formatted `.md` file. Copy this into a file called `PLAN.md`:

```markdown
# RL Launchpad — 5-Day Sprint Plan

Three people, three independent workstreams, zero blocking.

## Issue Order

| Issue | Person | Day | Title |
|-------|--------|-----|-------|
| A0 | A | 1 | Read docs + pick MJX environment |
| B0 | B | 1 | Read docs + install deps + test env |
| C0 | C | 1 | Read docs + create repo + project structure |
| A1 | A | 2 | Implement actor-critic network |
| A2 | A | 2 | Implement PPO algorithm |
| A3 | A | 3 | Implement training loop |
| A4 | A | 3 | Debug and fix training |
| A5 | A | 4 | Run 3 training seeds |
| A6 | A | 5 | Code review + pin deps + test reproducibility |
| B1 | B | 2 | Write SB3 baseline |
| B2 | B | 3 | Run SB3 baseline 3 seeds |
| B3 | B | 2 | Write evaluation script |
| B4 | B | 4 | Run evaluation on all 6 checkpoints |
| B5 | B | 5 | Record demo video |
| B6 | B | 5 | Practice live walkthrough |
| C1 | C | 2 | Write pyproject.toml |
| C2 | C | 2 | Write configs/default.yaml |
| C3 | C | 2 | Write initial README.md |
| C4 | C | 2 | Write docs/environment.md |
| C5 | C | 3-4 | Write all write-up sections |
| C6 | C | 4 | Set up notebooks/results.ipynb |
| C7 | C | 3-4 | Write docs/failures.md |
| C8 | C | 5 | Assemble final write-up |
| C9 | C | 5 | Finalize README |
| C10 | C | 5 | Submit to challenge platform |

## Dependency Chain

```
A0 -> A1 -> A2 -> A3 -> A4 -> A5 -> A6
B0 -> B1 -> B2         -> B3 -> B4 -> B5 -> B6
C0 -> C1 -> C2 -> C3 -> C4 -> C5 -> C6 -> C7 -> C8 -> C9 -> C10


Only 2 real blockers:
- B4 needs A5 (A's 3 checkpoints)
- C8 needs B4 (eval numbers)

Everything else runs in parallel.

## A0: Read docs + pick MJX environment

### Task
Read docs and pick your MJX environment. Do NOT start coding yet.

### What to do

#### 1. Read Spinning Up PPO (30 min)

- Go to https://spinningup.openai.com/en/latest/algorithms/ppo.html
- Read: Background, Key Equations, Pseudocode
- Skip: Documentation section (that is their code, not yours)
- Write 3-sentence summary in `docs/learning-notes.md`

#### 2. Read CleanRL PPO (1 hour)

- Go to https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
- Read the entire file (about 200 lines)
- For each section, write a 1-line comment in `docs/learning-notes.md`
- DO NOT copy any code into your submission

#### 3. Run MJX locomotion notebook (15 min)

- Open https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/locomotion.ipynb
- Click Copy to Drive
- Run every cell from top to bottom
- Watch the robot learn
- Note down: environment name used, what training looks like

#### 4. Pick your environment

Choose ONE:
- `G1JoystickFlatTerrain` (humanoid, medium difficulty)
- `CheetahRun` (quadruped, easier)
- `AntMove` (quadruped, easier)

Write in `docs/learning-notes.md`:
- Which environment you picked
- Observation space (shape, what each dim means)
- Action space (shape, bounds)
- Why you picked it

### Done when

- [ ] Read Spinning Up PPO page
- [ ] Read CleanRL `ppo_continuous_action.py`
- [ ] Ran MJX locomotion notebook on Colab
- [ ] Picked environment and documented it in `docs/learning-notes.md`
---


## A1: Implement actor-critic network

### Task
Write the `ActorCritic` network in `ppo/agent.py`.

### File: `ppo/agent.py`

```python
import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_dim, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        return self.shared(obs)

    def get_action(self, obs):
        shared_out = self.shared(obs)
        mean = self.actor_mean(shared_out)
        std = self.actor_log_std.exp()
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        return action, log_prob

    def evaluate(self, obs, action):
        shared_out = self.shared(obs)
        mean = self.actor_mean(shared_out)
        std = self.actor_log_std.exp()
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        value = self.critic(shared_out).squeeze(-1)
        return log_prob, entropy, value

    def get_value(self, obs):
        shared_out = self.shared(obs)
        return self.critic(shared_out).squeeze(-1)
```

### Test it

Create `tests/test_agent.py`:

```python
import torch
from ppo.agent import ActorCritic

agent = ActorCritic(obs_dim=20, act_dim=6)
dummy_obs = torch.randn(1, 20)

action, log_prob = agent.get_action(dummy_obs)
value = agent.get_value(dummy_obs)

print(f"Action shape: {action.shape}")
print(f"Log prob shape: {log_prob.shape}")
print(f"Value shape: {value.shape}")

action2, log_prob2, entropy, value2 = agent.evaluate(dummy_obs, action)
print(f"Entropy shape: {entropy.shape}")
print("Agent network works!")
```

Run:

```bash
python tests/test_agent.py
```

Expected output:

```
Action shape: torch.Size([1, 6])
Log prob shape: torch.Size([1])
Value shape: torch.Size([1])
Entropy shape: torch.Size([1])
Agent network works!
```

### Done when

- [ ] `ppo/agent.py` contains `ActorCritic` class
- [ ] `tests/test_agent.py` passes

---

## A2: Implement PPO algorithm

### Task
Write PPO-clip algorithm in `ppo/ppo.py`.

### File: `ppo/ppo.py`

```python
import numpy as np
import torch


def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values = values + [next_value]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns


def ppo_clip_loss(log_probs, old_log_probs, advantages, clip_ratio=0.2):
    ratio = (log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
    return -torch.min(surr1, surr2).mean()


def value_loss(values, returns):
    return ((values - returns) ** 2).mean()


def update(agent, optimizer, observations, actions, old_log_probs, returns,
           advantages, epochs=10, batch_size=64, clip_ratio=0.2):
    total_pol_loss = 0
    total_val_loss = 0
    total_entropy = 0
    num_updates = 0

    for _ in range(epochs):
        indices = np.arange(len(observations))
        np.random.shuffle(indices)

        for start in range(0, len(observations), batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            batch_obs = observations[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_returns = returns[batch_idx]
            batch_advantages = advantages[batch_idx]

            batch_advantages = (batch_advantages - batch_advantages.mean()) / (
                batch_advantages.std() + 1e-8
            )

            new_log_probs, entropy, values = agent.evaluate(batch_obs, batch_actions)

            pol_loss = ppo_clip_loss(
                new_log_probs, batch_old_log_probs, batch_advantages, clip_ratio
            )
            val_loss = value_loss(values, batch_returns)
            ent_bonus = entropy.mean()

            loss = pol_loss + 0.5 * val_loss - 0.01 * ent_bonus

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_pol_loss += pol_loss.item()
            total_val_loss += val_loss.item()
            total_entropy += ent_bonus.item()
            num_updates += 1

    return (
        total_pol_loss / num_updates,
        total_val_loss / num_updates,
        total_entropy / num_updates,
    )
```

### Test it

Create `tests/test_ppo.py`:

```python
import torch
from ppo.agent import ActorCritic
from ppo.ppo import compute_gae, ppo_clip_loss, update

rewards = [1.0, 1.0, 1.0, 1.0, 1.0]
values = [0.5, 0.5, 0.5, 0.5, 0.5]
dones = [0.0, 0.0, 0.0, 0.0, 0.0]
next_value = 0.5

advantages, returns = compute_gae(rewards, values, dones, next_value)
print(f"GAE advantages: {[f'{a:.3f}' for a in advantages]}")
print(f"GAE returns: {[f'{r:.3f}' for r in returns]}")

log_probs = torch.randn(32)
old_log_probs = torch.randn(32)
advantages_tensor = torch.randn(32)
loss = ppo_clip_loss(log_probs, old_log_probs, advantages_tensor)
print(f"PPO loss: {loss.item():.4f}")

print("PPO functions work!")
```

Run:

```bash
python tests/test_ppo.py
```

Expected output:

```
GAE advantages: ['0.500', '0.500', '0.500', '0.500', '0.500']
GAE returns: ['1.000', '1.000', '1.000', '1.000', '1.000']
PPO loss: X.XXXX
PPO functions work!
```

### Done when

- [ ] All 4 functions work correctly
- [ ] `tests/test_ppo.py` passes

---

## A3: Implement training loop

### Task
Write the full training loop in `ppo/train.py`.

### File: `ppo/train.py`

```python
import os
import time
import numpy as np
import torch
import yaml


def main():
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    from mujoco_playground import locomotion
    env = locomotion.load(config["env_name"])
    obs_dim = env.observation_size
    act_dim = env.action_size
    print(f"Environment: {config['env_name']}")
    print(f"Observation dim: {obs_dim}, Action dim: {act_dim}")

    from ppo.agent import ActorCritic
    from ppo.ppo import compute_gae, update

    agent = ActorCritic(obs_dim, act_dim, config["hidden_dim"])
    optimizer = torch.optim.Adam(agent.parameters(), lr=config["pi_lr"])

    os.makedirs("checkpoints", exist_ok=True)

    total_steps = 0
    start_time = time.time()

    for epoch in range(config["epochs"]):
        observations, actions, rewards, dones, log_probs, values = (
            [], [], [], [], [], []
        )

        obs, _ = env.reset()
        obs = np.array(obs)

        for step in range(config["steps_per_epoch"]):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                action, log_prob = agent.get_action(obs_t)
                value = agent.get_value(obs_t)

            action_np = np.array(action.squeeze(0))
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            observations.append(obs_t.squeeze(0))
            actions.append(action.squeeze(0))
            rewards.append(reward)
            dones.append(float(done))
            log_probs.append(log_prob.squeeze(0))
            values.append(value.squeeze(0))

            obs = np.array(next_obs)
            total_steps += 1

            if done:
                obs, _ = env.reset()
                obs = np.array(obs)

        with torch.no_grad():
            next_obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            next_value = agent.get_value(next_obs_t).item()

        advantages, returns = compute_gae(
            rewards,
            [v.item() for v in values],
            dones,
            next_value,
            config["gamma"],
            config["lam"],
        )

        observations = torch.stack(observations)
        actions = torch.stack(actions)
        old_log_probs = torch.stack(log_probs)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        pol_loss, val_loss, entropy = update(
            agent,
            optimizer,
            observations,
            actions,
            old_log_probs,
            returns,
            advantages,
            epochs=config["train_iters"],
            batch_size=config["batch_size"],
            clip_ratio=config["clip_ratio"],
        )

        elapsed = time.time() - start_time
        avg_reward = np.mean(rewards)
        print(
            f"Epoch {epoch+1}/{config['epochs']} | "
            f"Avg Reward: {avg_reward:.2f} | "
            f"Pol Loss: {pol_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Entropy: {entropy:.4f} | "
            f"Steps: {total_steps} | "
            f"Time: {elapsed:.1f}s"
        )

        if (epoch + 1) % 10 == 0:
            torch.save(
                agent.state_dict(), f"checkpoints/ppo_epoch_{epoch+1}.pt"
            )
            print(f"  -> Saved checkpoint: checkpoints/ppo_epoch_{epoch+1}.pt")

    total_time = time.time() - start_time
    print(
        f"\nTraining complete! Total steps: {total_steps}, "
        f"Wall time: {total_time:.1f}s"
    )

    torch.save(agent.state_dict(), "checkpoints/ppo_final.pt")
    print("Saved final checkpoint: checkpoints/ppo_final.pt")


if __name__ == "__main__":
    main()
```

### Run it

```bash
python ppo/train.py
```

### Common debugging tips

- Reward is NaN: decrease learning rate, try `0.0001` instead of `0.0003`
- Reward goes negative: check action bounds, check reward function
- Crashes with shape mismatch: observation/action dimension mismatch
- Crashes with JAX error: set `JAX_DEFAULT_MATMUL_PRECISION=highest`

### Done when

- [ ] Training runs without crashing
- [ ] Reward increases over epochs
- [ ] Checkpoints saved to `checkpoints/`

---

## A4: Debug and fix training

### Task
Run training, observe, fix issues, get working learning curves.

### What to check

- Reward goes up over epochs
- No NaN values
- No crashes

### Common fixes

- Reward flat: increase learning rate (`pi_lr: 0.0003` to `pi_lr: 0.001`)
- Reward NaN: decrease learning rate
- Crashes: read error, fix code

### Log results

Write results in `docs/training-log.md`:

```markdown
## First Training Run
- Environment: [X]
- Seed: [X]
- Total steps: [X]
- Wall-clock time: [X] seconds
- Final average reward: [X]
- Did it learn? Yes/No
- Notes: [any issues encountered]
```

### Done when

- [ ] Training shows clear learning (reward going up)
- [ ] No crashes over full 50 epochs

---

## A5: Run 3 training seeds

### Task
Train with 3 different seeds for reproducibility.

### What to do

1. Set seed: 42, train, save `ppo_seed42.pt`
2. Set seed: 123, train, save `ppo_seed123.pt`
3. Set seed: 777, train, save `ppo_seed777.pt`
4. Record results in `docs/training-log.md`

### Commands

```bash
# Seed 42
# Edit configs/default.yaml -> seed: 42
python ppo/train.py
cp checkpoints/ppo_final.pt checkpoints/ppo_seed42.pt

# Seed 123
# Edit configs/default.yaml -> seed: 123
python ppo/train.py
cp checkpoints/ppo_final.pt checkpoints/ppo_seed123.pt

# Seed 777
# Edit configs/default.yaml -> seed: 777
python ppo/train.py
cp checkpoints/ppo_final.pt checkpoints/ppo_seed777.pt
```

### Record in `docs/training-log.md`

```markdown
## Training Results (3 Seeds)

| Seed | Final Avg Reward | Total Steps | Wall Time |
|------|-----------------|-------------|-----------|
| 42   | [X]             | [X]         | [X]s      |
| 123  | [X]             | [X]         | [X]s      |
| 777  | [X]             | [X]         | [X]s      |
```

### Done when

- [ ] 3 checkpoints exist: `ppo_seed42.pt`, `ppo_seed123.pt`, `ppo_seed777.pt`
- [ ] Results logged in `docs/training-log.md`

---

## A6: Code review + pin deps + test reproducibility

### Task
Review code, pin versions, test reproducibility (R1, R3).

### What to do

1. Read through `ppo/agent.py`, `ppo/ppo.py`, `ppo/train.py`
2. Confirm all PPO code is YOUR code, not copied from CleanRL (R1)
3. Run `pip freeze > requirements_pinned.txt`
4. Update `pyproject.toml` with pinned versions
5. Test reproducibility: fresh clone, `pip install -e .`, `python ppo/train.py`
6. Should work in less than 15 min

### Done when

- [ ] All PPO code is original (not copied from CleanRL)
- [ ] `pyproject.toml` has pinned versions
- [ ] Fresh clone-to-eval works in less than 15 min

---

# Person B: Baselines + Evaluation + Demo Video

## B0: Read docs + install deps + test env

### Task
Read docs, install dependencies, and test the MJX environment. Do NOT start coding yet.

### What to do

#### 1. Read Spinning Up PPO (20 min)

- Go to https://spinningup.openai.com/en/latest/algorithms/ppo.html
- Read ONLY: Quick Facts, Exploration vs Exploitation, Pseudocode
- Skip all math equations
- Write 3-sentence summary in `docs/learning-notes.md`

#### 2. Run MJX starter notebook (15 min)

- Open https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/dm_control_suite.ipynb
- Click Copy to Drive
- Run every cell
- Just see it work, do not worry about understanding the code

#### 3. Install dependencies locally (10 min)

Run these commands one by one in a terminal:

```bash
python -m venv venv
source venv/bin/activate
pip install playground stable-baselines3 torch numpy matplotlib pyyaml
```

Verify GPU works:

```bash
python -c "import jax; print(jax.default_backend())"
```

Should print: `gpu`

If it prints `cpu`, ask Person A for help.

#### 4. Test environment interaction (10 min)

Create `tests/test_env.py` with this content:

```python
import numpy as np
import jax.numpy as jnp
from mujoco_playground import locomotion

env = locomotion.load("CheetahRun")
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")
print(f"Action shape: {env.action_size}")

for i in range(100):
    action = jnp.zeros(env.action_size)
    obs, reward, terminated, truncated, info = env.step(action)
    if i % 20 == 0:
        print(f"Step {i}: reward={reward:.4f}")

print("Environment works!")
```

Run it:

```bash
python tests/test_env.py
```

### Done when

- [ ] Read Spinning Up PPO page
- [ ] Ran MJX starter notebook on Colab
- [ ] Installed all dependencies locally
- [ ] Verified GPU works (prints `gpu`)
- [ ] `tests/test_env.py` runs and prints "Environment works!"

---

## B1: Write SB3 baseline

### Task
Write Stable-Baselines3 PPO baseline for comparison.

### File: `baselines/sb3_ppo.py`

```python
import time
import numpy as np
import yaml
from stable_baselines3 import PPO as SB3PPO


class MJXWrapper:
    def __init__(self, env_name):
        from mujoco_playground import locomotion
        self.env = locomotion.load(env_name)

    def reset(self):
        obs, _ = self.env.reset()
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return np.array(obs, dtype=np.float32), float(reward), done, info

    @property
    def observation_space(self):
        from gymnasium import spaces
        return spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.env.observation_size,), dtype=np.float32,
        )

    @property
    def action_space(self):
        from gymnasium import spaces
        return spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.env.action_size,), dtype=np.float32,
        )


def main():
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    print(f"Training SB3 PPO on {config['env_name']} with seed {config['seed']}")

    env = MJXWrapper(config["env_name"])

    start_time = time.time()
    model = SB3PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=config["pi_lr"],
        n_steps=config["steps_per_epoch"],
        batch_size=config["batch_size"],
        n_epochs=config["train_iters"],
        gamma=config["gamma"],
        gae_lambda=config["lam"],
        clip_range=config["clip_ratio"],
        seed=config["seed"],
    )
    model.learn(total_timesteps=config["total_timesteps"])
    elapsed = time.time() - start_time

    model.save("baselines/sb3_ppo_checkpoint")
    print(f"\nSB3 training complete! Wall time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
```

### Test it

```bash
python baselines/sb3_ppo.py
```

### Done when

- [ ] SB3 baseline trains successfully
- [ ] Model saved to `baselines/sb3_ppo_checkpoint.zip`

---

## B2: Run SB3 baseline 3 seeds

### Task
Run SB3 baseline with 3 different seeds.

### What to do

1. Set seed 42, train, save `sb3_seed42.zip`
2. Set seed 123, train, save `sb3_seed123.zip`
3. Set seed 777, train, save `sb3_seed777.zip`
4. Record results in `docs/training-log.md`

### Commands

```bash
# Seed 42
# Edit configs/default.yaml -> seed: 42
python baselines/sb3_ppo.py
cp baselines/sb3_ppo_checkpoint.zip baselines/sb3_seed42.zip

# Seed 123
# Edit configs/default.yaml -> seed: 123
python baselines/sb3_ppo.py
cp baselines/sb3_ppo_checkpoint.zip baselines/sb3_seed123.zip

# Seed 777
# Edit configs/default.yaml -> seed: 777
python baselines/sb3_ppo.py
cp baselines/sb3_ppo_checkpoint.zip baselines/sb3_seed777.zip
```

### Done when

- [ ] 3 SB3 checkpoints exist
- [ ] Results logged in `docs/training-log.md`

---

## B3: Write evaluation script

### Task
Write evaluation script that tests trained agents.

### File: `eval/evaluate.py`

```python
import argparse
import numpy as np
import torch
import yaml
from ppo.agent import ActorCritic


def evaluate(checkpoint_path, env_name, num_episodes=50, seed=999):
    from mujoco_playground import locomotion
    env = locomotion.load(env_name)

    obs_dim = env.observation_size
    act_dim = env.action_size
    agent = ActorCritic(obs_dim, act_dim)
    agent.load_state_dict(torch.load(checkpoint_path))
    agent.eval()

    np.random.seed(seed)
    all_rewards = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            obs_t = torch.tensor(np.array(obs), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                mean = agent.actor_mean(agent.shared(obs_t))
            action = mean.squeeze(0).numpy()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        all_rewards.append(total_reward)

    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    return mean_reward, std_reward, all_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--env_name", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=999)
    args = parser.parse_args()

    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    env_name = args.env_name or config["env_name"]

    print(f"Evaluating {args.checkpoint} on {env_name}...")
    mean_r, std_r, rewards = evaluate(
        args.checkpoint, env_name, args.num_episodes, args.seed
    )
    print(f"Result: {mean_r:.2f} +/- {std_r:.2f}")
    print(f"Min: {min(rewards):.2f}, Max: {max(rewards):.2f}")
```

### Test it

```bash
python eval/evaluate.py --checkpoint checkpoints/ppo_final.pt
```

### Done when

- [ ] `eval/evaluate.py` works
- [ ] Can evaluate any checkpoint

---

## B4: Run evaluation on all 6 checkpoints

### Task
Evaluate all 6 trained models (3 ours + 3 SB3).

### What to do

Run for each of A's 3 checkpoints:

```bash
python eval/evaluate.py --checkpoint checkpoints/ppo_seed42.pt
python eval/evaluate.py --checkpoint checkpoints/ppo_seed123.pt
python eval/evaluate.py --checkpoint checkpoints/ppo_seed777.pt
```

Run for each of your 3 SB3 checkpoints:

```bash
python eval/evaluate.py --checkpoint baselines/sb3_seed42.zip
python eval/evaluate.py --checkpoint baselines/sb3_seed123.zip
python eval/evaluate.py --checkpoint baselines/sb3_seed777.zip
```

### Record in `docs/eval-results.md`

```markdown
## Evaluation Results

### Our PPO (from scratch)

| Seed | Mean Reward | Std | Min | Max |
|------|------------|-----|-----|-----|
| 42   | [X]        | [X] | [X] | [X] |
| 123  | [X]        | [X] | [X] | [X] |
| 777  | [X]        | [X] | [X] | [X] |

### SB3 PPO (baseline)

| Seed | Mean Reward | Std | Min | Max |
|------|------------|-----|-----|-----|
| 42   | [X]        | [X] | [X] | [X] |
| 123  | [X]        | [X] | [X] | [X] |
| 777  | [X]        | [X] | [X] | [X] |

## Comparison Table

| Agent         | Mean Reward   | Std  | Training Time | Env Steps |
|---------------|--------------|------|---------------|-----------|
| Our PPO       | [X] +/- [Y] |      |               |           |
| SB3 PPO       | [X] +/- [Y] |      |               |           |
```

### Depends on

- A5 (A's 3 trained checkpoints must exist)

### Done when

- [ ] 6 eval results recorded
- [ ] Comparison table in `docs/eval-results.md`

---

## B5: Record demo video

### Task
Record 2-minute demo video of the project.

### Script

```
0:00-0:10  Title slide (team name, challenge name)
0:10-0:30  Problem statement
0:30-1:00  Show trained policy running
1:00-1:20  Show training curves
1:20-1:40  Explain architecture choice
1:40-2:00  Show comparison results
```

### Tools

- OBS Studio (free) for screen recording

### Done when

- [ ] Video recorded, 2 min or less
- [ ] Uploaded to YouTube (unlisted) or Drive
- [ ] Share link ready

---

## B6: Practice live walkthrough

### Task
Practice explaining the project to judges.

### Things to explain

#### 1. Loss function

> PPO uses a clipped surrogate objective. We compute the ratio of the new policy's probability to the old policy's probability for each action. If this ratio is between 0.8 and 1.2 (controlled by our clip ratio of 0.2), we use it directly. If it goes outside that range, we clip it. This prevents the policy from changing too much in one update, which prevents catastrophic learning collapse.

#### 2. Advantage estimation

> GAE computes how much better each action was than expected. It blends one-step returns with multi-step returns using lambda=0.95. This means we mostly trust multi-step estimates (lower variance, higher bias) while adding just enough one-step signal to reduce bias. The gamma=0.99 discount factor means we value future rewards almost as much as immediate ones.

#### 3. One architecture choice

> We used a 2-layer MLP with 256 hidden units and Tanh activations. We chose Tanh over ReLU because [your reason]. We chose 256 units because [your reason]. We experimented with [alternative] but found [result].

### Done when

- [ ] Can explain all 3 without notes

---

# Person C: Documentation, Config, Write-up, Submission

## C0: Read docs + create repo + project structure

### Task
Read docs, create the GitHub repo, set up project structure, and write initial config files.

### What to do

#### 1. Read Spinning Up PPO (15 min)

- Go to https://spinningup.openai.com/en/latest/algorithms/ppo.html
- Read ONLY: Quick Facts and Pseudocode
- Write 3-sentence summary in `docs/learning-notes.md`

#### 2. Create the GitHub repo (5 min)

- Go to https://github.com/new
- Repository name: `rl-launchpad`
- Description: Launchpad 2026 RL Track - From-scratch PPO on MuJoCo Playground
- Set to Public
- Click Create repository
- Clone it:

```bash
git clone https://github.com/YOUR_USERNAME/rl-launchpad.git
cd rl-launchpad
```

#### 3. Create project structure (5 min)

Run these commands:

```bash
mkdir -p ppo eval baselines configs notebooks tests docs checkpoints write-up
touch ppo/__init__.py eval/__init__.py baselines/__init__.py
touch ppo/agent.py ppo/ppo.py ppo/train.py
touch eval/evaluate.py baselines/sb3_ppo.py
touch configs/default.yaml notebooks/results.ipynb tests/test_env.py tests/test_agent.py tests/test_ppo.py
touch docs/learning-notes.md docs/training-log.md docs/eval-results.md docs/failures.md docs/environment.md
touch write-up/problem.md write-up/approach.md write-up/evidence.md write-up/constraints.md write-up/honesty.md write-up/submission.md
```

#### 4. Write `.gitignore`

```
__pycache__/
*.pyc
venv/
.venv/
*.pt
wandb/
runs/
outputs/
*.mp4
.DS_Store
requirements_pinned.txt
```

#### 5. Write `pyproject.toml`

```toml
[project]
name = "rl-launchpad"
version = "0.1.0"
description = "Launchpad 2026 RL Track - From-scratch PPO on MuJoCo Playground"
requires-python = ">=3.10"
dependencies = [
    "playground",
    "stable-baselines3",
    "torch",
    "numpy",
    "matplotlib",
    "pyyaml",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "ruff",
]

[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.ruff]
line-length = 100
```

#### 6. Write `configs/default.yaml`

```yaml
env_name: "CheetahRun"
seed: 42
total_timesteps: 1000000
steps_per_epoch: 4096
epochs: 50
gamma: 0.99
lam: 0.95
clip_ratio: 0.2
pi_lr: 0.0003
vf_lr: 0.001
train_iters: 80
batch_size: 256
hidden_dim: 256
```

#### 7. Write initial `README.md`

```markdown
# RL Launchpad - Griffin Labs RL Track

From-scratch PPO implementation on MuJoCo Playground.

## Team
- Person A: PPO implementation
- Person B: Baselines and evaluation
- Person C: Documentation and write-up

## Environment
- Simulator: MuJoCo Playground (MJX)
- Algorithm: PPO-Clip (from scratch, PyTorch)
- Baseline: Stable-Baselines3 PPO

## Getting Started
```bash
pip install -e .
python ppo/train.py
```

## Running Evaluation
```bash
python eval/evaluate.py --checkpoint checkpoints/ppo_final.pt
```

## Running Baseline
```bash
python baselines/sb3_ppo.py
```
```

#### 8. Write Problem section of write-up

Create `write-up/problem.md` with this content (fill in actual details):

```markdown
## Problem

[1-2 paragraphs explaining the problem]
- What task is the robot trying to solve?
- Why is this hard?
- What do existing approaches do and why are they insufficient?
- What would success look like? Define success criteria BEFORE you built anything.
```

#### 9. Push to GitHub

```bash
git add .
git commit -m "Initial project structure"
git push origin main
```

### Done when

- [ ] Read Spinning Up PPO page
- [ ] GitHub repo exists and is cloned locally
- [ ] All directories and empty files created
- [ ] `.gitignore` written
- [ ] `pyproject.toml` written
- [ ] `configs/default.yaml` written
- [ ] `README.md` written
- [ ] `write-up/problem.md` drafted
- [ ] Initial commit pushed to GitHub

---

## C1: Write pyproject.toml

### Task
Write PEP 518/621 compliant `pyproject.toml`.

### What to include

- Project metadata (name, version, description)
- Dependencies: playground, stable-baselines3, torch, numpy, matplotlib, pyyaml
- Build system: setuptools
- Ruff config

### Done when

- [ ] `pyproject.toml` exists and is valid
- [ ] `pip install -e .` works

---

## C2: Write configs/default.yaml

### Task
Write hyperparameter config file.

### Contents

- `env_name`, `seed`, `total_timesteps`, `steps_per_epoch`
- `epochs`, `gamma`, `lam`, `clip_ratio`
- `pi_lr`, `vf_lr`, `train_iters`, `batch_size`, `hidden_dim`

### Done when

- [ ] `configs/default.yaml` exists with all hyperparameters

---

## C3: Write initial README.md

### Task
Write README with getting started instructions.

### Sections

- Project description
- Team info
- Getting started (`pip install -e .` and `python ppo/train.py`)
- Running evaluation
- Running baseline
- Environment info

### Done when

- [ ] `README.md` exists with all sections

---

## C4: Write docs/environment.md

### Task
Document the MJX environment details.

### What to fill in

- Observation shape, action shape
- Action bounds, reward range
- Max episode length
- What observations contain
- What actions control

### Done when

- [ ] `docs/environment.md` exists with all details

---

## C5: Write all write-up sections

### Task
Write all 5 sections of the write-up.

### Sections to write

#### 1. `write-up/problem.md`

```markdown
## Problem

[1-2 paragraphs explaining the problem]
- What task is the robot trying to solve?
- Why is this hard?
- What do existing approaches do and why are they insufficient?
- What would success look like? Define success criteria BEFORE you built anything.
```

#### 2. `write-up/approach.md`

```markdown
## Approach

### Algorithm Choice
- Why PPO over SAC/TD3?
- What did you rule out and why?

### Network Architecture
- [Description or diagram]
- Why this size?

### Reward Design
- What reward function did you use?
- Did you modify the default reward?

### Environment Modifications
- What did you change from the stock task?
- Why?
```

#### 3. `write-up/evidence.md`

```markdown
## Evidence

### Results
- [Insert comparison table]
- [Insert training curves plot]
- [Insert evaluation bar chart]

### Analysis
- How does our PPO compare to SB3?
- What does mean +/- std tell us about consistency?

### Statistical Honesty
- Number of evaluation episodes: 50
- Number of training seeds: 3
- Fixed eval seeds (disjoint from training): Yes (seed 999)
```

#### 4. `write-up/constraints.md`

```markdown
## Constraints

### Compute
- GPU: [X]
- Training time for our PPO: [X] seconds
- Training time for SB3: [X] seconds
- Total env steps: [X]

### Sample Efficiency
- Our PPO: [X] steps to reach [Y] reward
- SB3: [X] steps to reach [Y] reward

### Trade-offs
- [What trade-offs did you observe?]
```

#### 5. `write-up/honesty.md`

```markdown
## Honesty and Trajectory

### Known Failure Modes
- [Where does the policy fail?]
- [What conditions cause degradation?]

### Negative Results
- [What did you try that didn't work?]

### What We Would Do With Two More Weeks
- [Concrete next step 1]
- [Concrete next step 2]
- [Concrete next step 3]

### Lessons Learned
- [What surprised you?]
- [What would you do differently?]
```

### Done when

- [ ] All 5 section files exist
- [ ] Total is 4 pages or less (~1000 words)

---

## C6: Set up notebooks/results.ipynb

### Task
Set up Jupyter notebook for plots.

### What to create

1. Training curves plot: our PPO vs SB3, mean +/- std shading
2. Evaluation bar chart: final scores comparison
3. Save as PNG files

### Notebook cell 1

```python
import matplotlib.pyplot as plt
import numpy as np

epochs = list(range(1, 51))
our_rewards_seed42 = []    # fill in
our_rewards_seed123 = []   # fill in
our_rewards_seed777 = []   # fill in
sb3_rewards = []           # fill in

our_mean = np.mean([our_rewards_seed42, our_rewards_seed123, our_rewards_seed777], axis=0)
our_std = np.std([our_rewards_seed42, our_rewards_seed123, our_rewards_seed777], axis=0)

plt.figure(figsize=(10, 6))
plt.plot(epochs, our_mean, label="Our PPO (from scratch)", color="blue")
plt.fill_between(epochs, our_mean - our_std, our_mean + our_std, alpha=0.3, color="blue")
plt.plot(epochs, sb3_rewards, label="SB3 PPO (baseline)", color="red")
plt.xlabel("Epoch")
plt.ylabel("Average Reward")
plt.title("PPO Training: Our Implementation vs SB3 Baseline")
plt.legend()
plt.grid(True)
plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
```

### Notebook cell 2

```python
agents = ["Our PPO", "SB3 PPO"]
means = [our_final_mean, sb3_final_mean]  # fill in
stds = [our_final_std, sb3_final_std]      # fill in

plt.figure(figsize=(6, 5))
plt.bar(agents, means, yerr=stds, capsize=10, color=["blue", "red"])
plt.ylabel("Final Mean Reward")
plt.title("Evaluation: Our PPO vs SB3 Baseline")
plt.savefig("eval_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
```

### Depends on

- B4 (eval results for actual numbers)

### Done when

- [ ] Notebook exists with plotting code
- [ ] After B4: plots generated and saved as PNG

---

## C7: Write docs/failures.md

### Task
Document failures and issues as they happen.

### What to record

- Date, what happened
- What we tried
- What went wrong
- What we learned

### Template

```markdown
## Failures and Issues

### [Date] - [What happened]
- What we tried
- What went wrong
- What we learned
```

### Done when

- [ ] `docs/failures.md` exists
- [ ] Updated as issues come up

---

## C8: Assemble final write-up

### Task
Combine all sections into final submission.

### What to do

1. Combine all sections into `write-up/submission.md`
2. Fill in actual numbers from eval results
3. Insert plots
4. Polish language
5. Ensure 4 pages or less

### Template

```markdown
# [Project Title]

## Problem
[Paste from write-up/problem.md, polish]

## Approach
[Paste from write-up/approach.md, polish]

## Evidence
[Paste from write-up/evidence.md, polish]
[Insert training curves plot]
[Insert evaluation comparison plot]

## Constraints
[Paste from write-up/constraints.md, polish]

## Honesty and Trajectory
[Paste from write-up/honesty.md, polish]
```

### Depends on

- B4 (eval results)
- C6 (plots)

### Done when

- [ ] `write-up/submission.md` is complete
- [ ] All numbers filled in
- [ ] All plots inserted
- [ ] 4 pages or less

---

## C9: Finalize README

### Task
Update README with complete clone-to-eval instructions.

### Template

```markdown
# RL Launchpad - Griffin Labs RL Track

## Quick Start (clone to running in less than 15 min)
```bash
git clone https://github.com/YOUR_USERNAME/rl-launchpad.git
cd rl-launchpad
python -m venv venv
source venv/bin/activate
pip install -e .
python ppo/train.py
```

## Running Evaluation
```bash
python eval/evaluate.py --checkpoint checkpoints/ppo_final.pt
```

## Running Baseline
```bash
python baselines/sb3_ppo.py
```

## Training Configs
All hyperparameters are in `configs/default.yaml`.

## Reproducibility
- All seeds are pinned in `configs/default.yaml`
- All dependency versions are in `pyproject.toml`
- To reproduce: clone, install, run


### Done when

- [ ] README has clone-to-eval in less than 15 min
- [ ] All sections complete

---

## C10: Submit to challenge platform

### Task
Upload everything to boardingpass.work.

### What to submit

- Repository link (public or judge access)
- Demo video link (YouTube/Drive)
- Write-up link (Google Doc/PDF)

### Final checklist

- [ ] Repo is public
- [ ] All links work
- [ ] Write-up is 4 pages or less
- [ ] Video is 2 min or less
- [ ] Configs and seeds committed
- [ ] `pyproject.toml` pinned
- [ ] README has clone-to-eval
- [ ] Every claim has evidence

### Done when

- [ ] All 3 links submitted on boardingpass.work

