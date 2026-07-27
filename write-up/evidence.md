# Evidence

Final evidence is generated from `checkpoints/ppo_eval_summary.json`, `baselines/sb3_ppo_eval_summary.json`, and their measured training histories. `eval/plot_benchmark.py` rejects missing or incompatible protocols; `eval/build_submission.py` rejects fewer than three complete seeds. Pilot and legacy results are not final evidence.
