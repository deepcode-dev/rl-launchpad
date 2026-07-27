# Evaluation results

Final numbers were generated only after all configured custom and SB3 seeds completed. The authoritative artifacts are:

- `checkpoints/ppo_eval_summary.json`
- `baselines/sb3_ppo_eval_summary.json`
- `write-up/benchmark_comparison.png`

Each final model is evaluated deterministically on the same 50 untouched episode seeds, 20,000–20,049. The 10,000-series block became validation data during training-horizon selection and is excluded from final evidence. Legacy negative-return JSON files use a changed reward and are not comparable.

| Agent | Across-seed mean return ± SD | Mean episode length |
|---|---:|---:|
| Custom PPO | 10.913 ± 3.785 | 746.1 |
| SB3 baseline | 1.310 ± 1.267 | 281.9 |

Custom per-seed mean return / mean length / horizon completions were:

- seed 10: 8.118 / 738.1 / 30 of 50
- seed 11: 8.358 / 557.3 / 24 of 50
- seed 21: 16.264 / 942.8 / 47 of 50

The baseline is a comparison only. The submitted policy is the from-scratch custom PPO.
