# Compute accounting

The final custom checkpoints record these per-seed wall times in their sidecar
metadata:

| Seed | Environment steps | Wall time |
|---:|---:|---:|
| 9033 | 199,884,800 | 13,502.3 s |
| 9006 | 199,884,800 | 13,434.3 s |
| 9018 | 199,884,800 | 13,872.3 s |
| 9016 | 199,884,800 | 13,552.3 s |
| 8009 | 199,884,800 | 13,443.6 s |

Mean custom time is 13,561.0 seconds per seed. The Brax baseline summary
records the documented 589.3-second 200M run and identifies that the raw
training log was not committed. Its final checkpoint is 200,540,160 steps,
the nearest saved checkpoint after the 200,000,000-step training budget.

The 131k-environment experiment reached 199,229,440 steps in approximately
1,134.7 seconds, but its policy quality was poor. That result demonstrates
that wall-clock throughput and learning quality are separate axes; it is not
used in the final benchmark.
