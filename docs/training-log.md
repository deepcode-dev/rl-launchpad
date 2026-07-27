# Training log

## Invalid legacy runs

All checkpoints created before the `custom-ppo-native-reward-autoreset-v2` contract are excluded. Their rollout GAE, reset, reward, and action semantics were invalid; decreasing value loss was not evidence of task learning.

## Corrected pilot A — seed 10, 327,680 steps

- Purpose: verify that corrected rollouts produce a learnable policy before the multi-seed budget.
- PPO update epochs: 10.
- Fixed evaluation: 50 episodes, seeds 10,000–10,049.
- Mean native return: 2.881; episode standard deviation: 4.689.
- Mean episode length: 339.9; five episodes reached 1,000 steps.
- Diagnostic failure: final approximate KL 0.0507 and clip fraction 0.504 indicated excessive reuse of each rollout.
- Decision: reduce update epochs from 10 to 4 for the submission run. Pilot A is not final evidence.

## Corrected pilot B — seed 10, 327,680 steps

- PPO update epochs: 4.
- Fixed evaluation: 50 episodes, seeds 10,000–10,049.
- Mean native return: 10.084; episode standard deviation: 6.737.
- Mean episode length: 805.3; 34 episodes reached 1,000 steps.
- Final approximate KL: 0.0206, less than half of pilot A's 0.0507.
- Decision: use four update epochs for the three-seed submission run. Pilot B selected the configuration but remains excluded from final comparative evidence.

## Long-horizon validation — three seeds, 2,457,600 steps/seed

- Purpose: test whether continuing the corrected optimizer improved the policy.
- Selection/evaluation block: 50 episodes per checkpoint, seeds 10,000–10,049; these episodes are validation data, not final evidence.
- Across-seed mean native return: 1.872; across-seed standard deviation: 1.252.
- Mean episode length: 280.6 steps. Seeds 10, 11, and 21 reached the horizon in 5, 10, and 14 of 50 episodes, respectively.
- Diagnosis: prolonged constant-rate updates increased clipping and degraded deterministic behavior despite finite value loss. The result is archived under `legacy_artifacts/overtrained-2457600-step-run/` as negative evidence.
- Decision: use the pilot-supported 20-epoch early stop (327,680 steps/seed) and reserve seeds 20,000–20,049 as the untouched final evaluation block.

## Final early-stopped run — three seeds, 327,680 steps/seed

- Training seeds: 10, 11, 21.
- Total environment steps: 983,040.
- Per-seed wall times: 132.9 s, 147.5 s, 143.6 s (mean 141.4 s).
- Final approximate KL: 0.0206, 0.0203, 0.0207.
- Final evaluation uses only the untouched 20,000–20,049 block; results are in `docs/eval-results.md`.
