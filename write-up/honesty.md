# Honesty & trajectory

Legacy checkpoints are invalid because the initial implementation crossed environment trajectories during GAE, failed to autoreset terminal slots, changed task reward, and used unbounded actions. The early benchmark chart was synthetic and has been removed from the evidence pipeline. A corrected but overtrained three-seed run is also retained as negative validation evidence. `docs/failures.md` and `docs/training-log.md` record the defects, corrections, tests, and horizon-selection result.
