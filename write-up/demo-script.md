# RL-track demo capture script

## Capture contract

Use the same checkpoint reported in the submission:

```powershell
.venv\Scripts\python.exe eval\record_native_video.py NEW_checkpoints\ppo_v2\ppo_seed9033.pt --output write-up\policy-footage.mp4 --command 1.0 0 0 --max-steps 500 --fps 50
```

The final video is under 90 seconds and uses only the reported
`ppo_seed9033.pt` checkpoint. It shows three labeled deterministic command
episodes using command vectors sampled from evaluation seeds 20,000–20,002.
It uses the human-readable `demo-narration.wav` generated from
`demo-narration.txt`. Do not use the old 327,680-step pilot or seed 2005 clip.

## Spoken and visual order

1. **Title:** from-scratch PyTorch PPO, Go1 locomotion, five custom seeds.
2. **Architecture:** 48-dim actor, 123-dim privileged critic, 512-256-128
   SiLU MLPs, tanh-squashed actions, and the declared EMA action filter.
3. **Correctness:** time-by-environment GAE, termination/truncation handling,
   per-slot autoreset, bounded actions, and stored observation statistics.
4. **Loss:** clipped PPO objective, Huber value loss, entropy bonus, and four
   minibatch passes.
5. **Evidence:** 200M steps per seed, 50 fixed episodes, eval seeds
   20,000–20,049, custom return 19.752 +/- 0.020, Brax return
   19.821 +/- 0.016.
6. **Limitations:** the custom curve is built from stdout metrics sampled every
   five epochs, the custom/Brax postprocessing differs by the disclosed EMA,
   and the JAX/MJX to PyTorch bridge is much slower than the JAX-native
   baseline.

## Final checks

- Show the updated `benchmark_comparison.png`, not the retired synthetic plot.
- Say “stock reward for training” and “shared command-tracking metric for
  evaluation”; do not call the evaluator return the native task reward.
- State that the demo is simulation-only and do not imply sim-to-real transfer.
