# Small, controlled ablations

Keep the same prepared candidate groups, candidate seed, public model revisions,
teacher cache, effective query batch and fixed checkpoint selector within each
comparison. Each configuration is a complete JSON recipe, not a patch applied
by a private launcher. Run it with the same training command and a new output
directory.

## Main recipe and matched fast recipes

`configs/books/student.json` is the original Books main recipe: V3,
LastToken, microbatch 1, and legacy input assembly. Its reported paper score
belongs to that experiment. `configs/ablations/default.json` is the separate
fast matched readout reference: LastToken, microbatch 8, batched input assembly,
fused optimizer, pretokenized items and the same scientific objective.
Readout comparisons use the fast reference, not the main result as a substitute.

| Config under `configs/ablations/` | Change from its reference | Comparison scope |
| --- | --- | --- |
| `default.json` | Fast V3 LastToken reference | Matched readout reference |
| `mem1.json` | A learned memory-token readout, one vector | Compare with `default.json` |
| `perceiver1.json` | One-query Perceiver readout | Compare with `default.json` |
| `ce-only.json` | CE without teacher KL | Compare with `default.json` |
| `beta09.json` | Final KD interpolation weight 0.9 instead of 0.5 | Compare with `default.json` |
| `tau2.json` | KD temperature 2 instead of 1 | Compare with `default.json` |
| `v1.json` | Feedback-conditioned item encoding (V1/FIE) | Compare with `configs/books/student.json` |

V1 retains the original legacy assembly and microbatch 1. It is not represented
as a matched fast V3 configuration because canonical item-token precomputation
is V3-specific. `mem1.json` and `perceiver1.json` come from the original matched
fast readout configurations; their readout-specific dimensions are retained.
CE-only, beta 0.9 and temperature 2 are runnable configuration variants. Their
presence does not establish that a new public-release run or result has been
completed. Self-alignment experiments are not included in this release.

For a KD experiment:

```bash
python -m duolm.train_next_item \
  --config configs/ablations/mem1.json --teacher-cache runs/books-cache \
  --data-dir data/books/candidates --output runs/books-mem1 --device cuda
```

For CE-only, omit the teacher cache:

```bash
python -m duolm.train_next_item \
  --config configs/ablations/ce-only.json \
  --data-dir data/books/candidates --output runs/books-ce-only --device cuda
```

Evaluate the checkpoint chosen by the fixed selector on the same disjoint
`report` users. Do not select the ablation's checkpoint on the report complement
or sealed test data. Keep the generated `resolved-config.json`, run manifest,
checkpoint manifest, and prediction files with each experiment so objective,
readout, input identity and selection procedure remain auditable.
