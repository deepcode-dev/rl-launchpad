# Two-minute demo script

## 0:00–0:20 — Task and policy

Show a deterministic Go1 rollout. Explain that the submitted policy is a from-scratch PyTorch PPO implementation controlling 12 normalized joint targets from a five-frame history of the stock 48-value observation. Reward is the unmodified MuJoCo Playground task reward.

## 0:20–0:55 — What was broken

Show the failure notes or one legacy bad rollout. The original implementation flattened vector environments before GAE, never reset fallen slots, reconstructed a different reward, and emitted unbounded Gaussian actions. State plainly that those checkpoints and the earlier synthetic chart are excluded from all evidence.

## 0:55–1:25 — Correctness changes

Show the contract bullets in the README: time-by-environment GAE, separate termination/truncation handling, per-slot autoreset, tanh-squashed actions, and stored observation-normalization statistics. Mention that regression tests cover trajectory separation, truncation bootstrap, action bounds, reset/history behavior, and checkpoint rejection.

## 1:25–1:50 — Measured comparison

Show `benchmark_comparison.png` and the final table in `submission.md`. Read the custom PPO and SB3 across-seed mean returns, episode lengths, equal environment-step budget, and 50 fixed evaluation episodes per training seed. Identify SB3 as a baseline only, not the submitted trainer.

## 1:50–2:00 — Honest limitation

Return to the rollout. Explain that native Windows ran JAX simulation on CPU while PyTorch used the RTX 3070, and that the passive viewer's mouse forces do not perturb MJX state. Conclude with the strongest remaining failure visible in the episode-length distribution.

## Capture checklist

- Use a final-contract checkpoint and the deterministic recorder.
- Keep the policy footage, result table, and benchmark plot readable at 1080p.
- Do not demonstrate viewer mouse pushes as policy disturbances.
- Keep the exported video at or below two minutes.

## Rebuild the local video

Record deterministic policy footage first:

```powershell
uv run python eval/record_video.py checkpoints/ppo_seed10.pt --output write-up/policy-footage.mp4 --seed 20000 --max-steps 500
```

On Windows, synthesize the tracked narration at the measured rate:

```powershell
Add-Type -AssemblyName System.Speech
$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
$speaker.SelectVoice('Microsoft Zira Desktop')
$speaker.Rate = 2
$speaker.SetOutputToWaveFile((Join-Path (Resolve-Path write-up).Path 'demo-narration.wav'))
$speaker.Speak((Get-Content write-up/demo-narration.txt -Raw -Encoding UTF8))
$speaker.Dispose()
```

Then assemble the final narrated video:

```powershell
uv run python eval/build_demo_video.py
```
