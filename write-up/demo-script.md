# RL-track demo capture script

## Capture contract

The current demo is a narration-free montage of three reported custom PPO
checkpoints. Each seed contributes two held-out evaluation episodes, for six
clips total. Each clip is recorded with deterministic native MuJoCo C physics
using the same evaluator-style controller and is labeled on screen with its
training seed, evaluation episode, and command.

```powershell
.venv\Scripts\python.exe eval\record_native_video.py NEW_checkpoints\ppo_v2\ppo_seed9033.pt --output write-up\demo-seed9033.mp4 --command 0.316973567 0.482398257 -0.762142644 --max-steps 1000 --fps 50
.venv\Scripts\python.exe eval\record_native_video.py NEW_checkpoints\ppo_v2\ppo_seed9033.pt --output write-up\demo-seed9033-ep20003.mp4 --command -0.131128497 -0.104267239 0.583250722 --max-steps 1000 --fps 50
.venv\Scripts\python.exe eval\record_native_video.py NEW_checkpoints\ppo_v2\ppo_seed9016.pt --output write-up\demo-seed9016.mp4 --command -0.827028589 0.353608204 0.736460695 --max-steps 1000 --fps 50
.venv\Scripts\python.exe eval\record_native_video.py NEW_checkpoints\ppo_v2\ppo_seed9016.pt --output write-up\demo-seed9016-ep20004.mp4 --command 0.632708849 0.169197491 -0.773160679 --max-steps 1000 --fps 50
.venv\Scripts\python.exe eval\record_native_video.py NEW_checkpoints\ppo_v2\ppo_seed8009.pt --output write-up\demo-seed8009.mp4 --command 0.960403828 -0.573840980 0.454944087 --max-steps 1000 --fps 50
.venv\Scripts\python.exe eval\record_native_video.py NEW_checkpoints\ppo_v2\ppo_seed8009.pt --output write-up\demo-seed8009-ep20005.mp4 --command -0.040002457 -0.250343880 0.664421158 --max-steps 1000 --fps 50
```

The final `demo.mp4` is a 120-second, captioned montage of those six clips:
two 1,000-step held-out evaluation episodes for each seed. It uses only
reported checkpoints, contains no sim-to-real claim, and does not require
spoken narration.

## Clip order

1. Seed 9033, evaluation episode 20000: forward command.
2. Seed 9033, evaluation episode 20003: forward, lateral, and yaw command.
3. Seed 9016, evaluation episode 20001: forward, lateral, and yaw command.
4. Seed 9016, evaluation episode 20004: forward, lateral, and yaw command.
5. Seed 8009, evaluation episode 20002: forward, lateral, and yaw command.
6. Seed 8009, evaluation episode 20005: forward, lateral, and yaw command.

## Final checks

- Keep `demo.mp4` at or under two minutes and playable when embedded.
- Keep the seed labels and command labels visible in every clip.
- Do not present the 131k-v2 candidates as reported checkpoints until their
  final evaluation artifacts are selected and added to the evidence summary.
- State that the benchmark and video are simulation-only.
