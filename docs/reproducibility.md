# Reproducibility

## Scope

The release contains the recommendation model and its training, evaluation,
and serving workflows. Model weights, prepared user histories, and teacher
score caches are external artifacts; they are not bundled with the code.
The paper's rating experiments and every historical baseline are outside
this focused code release.

## Evaluation protocol

Training uses mixed sets of eight candidates. Evaluation uses one observed
target and 99 fixed, randomly sampled unobserved candidates. Negative sampling,
candidate order, item content, visible feedback, and split membership are
bound to preparation manifests and checkpoint identities. Do not regenerate
negative candidates separately for competing methods.

Checkpoint selection uses 2,000 development users. The paper's Books table
reports all 5,000 development users; it therefore includes the selection
subset. The Video Games table reports the remaining 3,000 development users.
Users who cannot be scored stay in the denominator with zero contribution.
The paper's main recommendation table does not report the sealed test split.

Training/configuration presets and matched ablation presets serve different
comparisons. In particular, the main Books result and the later matched
readout/distillation runs should not be combined as if they were a single run.
Follow [the ablation guide](ablations.md) for the correct matched settings.

## Serving measurements

The recommendation serving comparison uses a frozen model and trace on one
H100, 100 candidates per request, and request concurrencies 1, 8, 32, and 64.
Reported throughput is candidate scores per second, not requests per second
or generated tokens per second. End-to-end timing includes request preparation,
process communication, queueing, model execution, and returned scores.

| Concurrency | Independent scoring | Shared history | Shared history + CUDA Graph |
| ---: | ---: | ---: | ---: |
| 1 | 520.53 | 2,503.55 | 8,870.36 |
| 8 | 921.45 | 9,480.19 | 20,740.56 |
| 32 | 881.61 | 19,883.01 | 25,506.11 |
| 64 | 851.48 | 24,953.95 | 26,403.98 |

These values come from the paper's execution-ablation table. The three modes
hold the item-vector representation and trained model fixed. Independent
scoring also changes request scheduling and input preparation; the difference
does not isolate attention arithmetic alone. Cross-request prefix caching is
disabled. CUDA Graph gains are workload-dependent.

Use the pinned environment and source-checked overlay in [serving.md](serving.md).
Run numerical checks before measuring speed and retain the actual software
versions, weights, precision, candidate sets, concurrency, timing scope,
latency distribution, and warmup policy with new measurements.

## Checkpoint compatibility

Checkpoints save tensor state dictionaries. Exact training resume binds the
source, configuration, data, teacher cache, optimizer state, and random state.
Source changes can invalidate an old resume identity even when model weights
have the same shapes. Do not disable those checks to resume an unrelated run.
Protocol identifiers inside files are retained for format compatibility.
