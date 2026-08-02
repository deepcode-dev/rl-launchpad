# RL-track demo capture script

## Capture contract

The current demo is a narration-free montage of the three reported custom PPO
checkpoints. Each seed contributes two held-out evaluation episodes, for six
clips total. Each clip uses deterministic native MuJoCo C physics, runs for
1,000 control steps, and is labeled on screen with its training seed,
evaluation episode, command vector, and the seed's 50-episode metrics.

```powershell
.venv\Scripts\python.exe eval\record_native_video.py NEW_checkpoints\ppo_v2\ppo_seed13039.pt --output write-up\demo-seed13039.mp4 --command 0.316973567 0.482398257 -0.762142644 --max-steps 1000 --fps 50
.venv\Scripts\python.exe eval\record_native_video.py NEW_checkpoints\ppo_v2\ppo_seed13039.pt --output write-up\demo-seed13039-ep20003.mp4 --command -0.131128497 -0.104267239 0.583250722 --max-steps 1000 --fps 50
.venv\Scripts\python.exe eval\record_native_video.py NEW_checkpoints\ppo_v2\ppo_seed13079.pt --output write-up\demo-seed13079.mp4 --command -0.827028589 0.353608204 0.736460695 --max-steps 1000 --fps 50
.venv\Scripts\python.exe eval\record_native_video.py NEW_checkpoints\ppo_v2\ppo_seed13079.pt --output write-up\demo-seed13079-ep20004.mp4 --command 0.632708849 0.169197491 -0.773160679 --max-steps 1000 --fps 50
.venv\Scripts\python.exe eval\record_native_video.py NEW_checkpoints\ppo_v2\ppo_seed13027.pt --output write-up\demo-seed13027.mp4 --command 0.960403828 -0.573840980 0.454944087 --max-steps 1000 --fps 50
.venv\Scripts\python.exe eval\record_native_video.py NEW_checkpoints\ppo_v2\ppo_seed13027.pt --output write-up\demo-seed13027-ep20005.mp4 --command -0.040002457 -0.250343880 0.664421158 --max-steps 1000 --fps 50
```

The final `demo.mp4` is a captioned montage of these six clips, kept at or
under the two-minute submission limit. Spoken narration is optional. The
video is an evaluation-style simulation demonstration, not a sim-to-real
claim.

## Clip order

1. Seed 13039, evaluation episode 20000.
2. Seed 13039, evaluation episode 20003.
3. Seed 13079, evaluation episode 20001.
4. Seed 13079, evaluation episode 20004.
5. Seed 13027, evaluation episode 20002.
6. Seed 13027, evaluation episode 20005.

## Final checks

- Keep `demo.mp4` at or under two minutes and playable when embedded.
- Keep seed, command, return, LinErr, and YawErr captions visible in every
  clip.
- Use only the three reported checkpoints in the final montage.
- State that the benchmark and video are simulation-only.
