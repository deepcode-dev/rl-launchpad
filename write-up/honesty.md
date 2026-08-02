# Honesty and trajectory

Legacy checkpoints are invalid because the initial implementation crossed
environment trajectories during GAE, failed to autoreset terminal slots,
changed task reward, and used unbounded actions. The early benchmark chart was
synthetic and has been removed from the evidence pipeline. The final result is
the three-seed 131k-v2 custom set in `NEW_checkpoints/ppo_v2`, trained for
199,229,440 steps per seed. The final evaluator reports a shared command-
tracking metric, not the stock training reward, and the custom EMA versus
stock Brax post-processing difference is disclosed. `docs/failures.md` records
the defects and corrections.
