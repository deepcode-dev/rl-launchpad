# Constraints

The final 131k-v2 jobs ran on NVIDIA H100 NVL GPUs. The custom trainer still
crosses the JAX MJX and PyTorch boundary, so its 1,392.6-1,424.3 second
per-seed wall times include interop overhead and are not an all-GPU claim. The
documented Brax reference takes 589.3 seconds for its 200M run. Interactive
viewers are for visualization only and are not used for benchmark scoring or
video generation; both run headlessly. See `submission.md` and checkpoint
metadata for the step budgets and wall times.
