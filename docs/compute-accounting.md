# Compute accounting

The final custom checkpoints record these per-seed wall times in their sidecar
metadata:

| Seed | Environment steps | Wall time |
|---:|---:|---:|
| 13039 | 199,229,440 | 1,403.9 s |
| 13079 | 199,229,440 | 1,424.3 s |
| 13027 | 199,229,440 | 1,392.6 s |

Mean custom time is 1,407.0 seconds per seed, about 23.4 minutes. The Brax
baseline summary records the documented 589.3-second 200M run and identifies
that the raw training log was not committed. Its final checkpoint is
200,540,160 steps, the nearest saved checkpoint after the 200,000,000-step
training budget.

The selected 131k-v2 runs use 131,072 environments, 760 epochs, eight PPO
passes, and batch size 16,384. The earlier fast-but-poor candidates are not
part of the final benchmark.
