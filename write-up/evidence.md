# Evidence

Final evidence is generated from
`NEW_checkpoints/ppo_v2/ppo_v2_eval_summary.json`, the three per-seed custom
evaluation JSON files, the three pulled 131k-v2 Slurm training logs,
`baselines/brax_go1_200m_eval_summary.json`, and the post-hoc EMA ablation
`baselines/brax_go1_200m_ema_ablation.json`. `eval/validate_submission.py`
checks the fixed 20,000-20,049 episode block, 50 episodes per seed,
deterministic actions, step budgets, observation/action dimensions, and the
measured Brax checkpoint curve. The benchmark is three custom seeds versus
three Brax seeds. The custom curve is a measured mean +/- seed standard
deviation over 152 logged points per seed.
