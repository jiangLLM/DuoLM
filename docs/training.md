# Training and evaluation

Install the package and prepare the frozen inputs described in [data.md](data.md).
The public training commands use pinned public Qwen revisions, local atomic
checkpoints, and no external backup or launch service. Full training uses CUDA
and FlexAttention; the offline CPU tests use tiny randomly initialized Qwen
models and do not represent the paper's quality or speed.

## Books

The teacher and student use the same eight-candidate mixed-history training
protocol. The original teacher preparation (`data/books/mixed8`) contains
retrieved development candidates. The student preparation
(`data/books/candidates`) preserves the mixed8 training groups and uses the
frozen target-guaranteed random100 evaluation groups. These evaluation protocols
must not be compared as if they were the same task.

1. Train the full-text teacher with the paper's Books teacher recipe:

   ```bash
   python -m duolm.train_next_item \
     --config configs/books/teacher.json \
     --data-dir data/books/mixed8 --output runs/books-teacher --device cuda
   ```

   `configs/books/teacher.json` derives from the original mixed8 teacher recipe.
   It is not the separate random100 teacher-training experiment. The paper's
   frozen Books teacher is the step-4000 checkpoint. A new run writes
   `checkpoints/step-00004000` at that boundary; keep its accompanying manifest
   and `resolved-config.json` together. Automatic checkpoint selection is
   separately recorded in `best_checkpoint.json`.

2. Produce scalar teacher logits on the student's exact training groups:

   ```bash
   python -m duolm.candidate_teacher cache \
     --checkpoint-dir runs/books-teacher/checkpoints/step-00004000 \
     --checkpoint-config runs/books-teacher/resolved-config.json \
     --config configs/books/teacher-cache.json \
     --data-dir data/books/candidates --output runs/books-cache --device cuda
   ```

   Cache generation loads weights only and verifies the source checkpoint,
   model revision, ordered candidates, query semantics, and source inputs.
   Shards are restartable: rerun exactly the same command after interruption.
   Only a completed cache with its completion manifest and
   `teacher_identity.json` can be bound by the student CLI.

3. Train the main student:

   ```bash
   python -m duolm.train_next_item \
     --config configs/books/student.json --teacher-cache runs/books-cache \
     --data-dir data/books/candidates --output runs/books-student --device cuda
   ```

   `--teacher-cache` resolves the cache file, records its teacher identity and
   SHA256 in the effective configuration, and writes `resolved-config.json` in
   the run directory. Row-level candidate order and semantic input hashes are
   checked again before training. An already-bound config cannot silently
   accept a different teacher or cache file.

The main Books recipe retains microbatch 1 and the original input assembly. The
matched fast readout experiments have their own configurations under
`configs/ablations`; their results should not be substituted for the main paper
row. See [ablations.md](ablations.md).

For a matched teacher development score, use the student's candidate groups:

```bash
python -m duolm.candidate_teacher evaluate \
  --checkpoint-dir runs/books-teacher/checkpoints/step-00004000 \
  --checkpoint-config runs/books-teacher/resolved-config.json \
  --config configs/books/teacher-cache.json \
  --data-dir data/books/candidates --output runs/books-teacher-report \
  --subset full-dev --device cuda
```

## Games

Games uses its own domain prompts, training groups and teacher; no Books teacher
cache is accepted for Games distillation. Both teacher and student use
`data/games/mixed8`, prepared with eight mixed-history training candidates and
100 target-guaranteed evaluation candidates. The Games main configurations use
the final branch's BF16 evaluation and teacher-cache profile. The explicit
teacher tree profile disables TF32; FP32 is reserved for precision diagnostics.

```bash
python -m duolm.train_next_item \
  --config configs/games/teacher.json \
  --data-dir data/games/mixed8 --output runs/games-teacher --device cuda
```

Use the checkpoint path recorded in `runs/games-teacher/best_checkpoint.json`
for the cache command below. `SELECTED_STEP` means that fixed selector's chosen
step, not a step chosen using held-out evaluation results.

```bash
python -m duolm.candidate_teacher cache \
  --checkpoint-dir runs/games-teacher/checkpoints/step-SELECTED_STEP \
  --checkpoint-config runs/games-teacher/resolved-config.json \
  --config configs/games/teacher-cache.json \
  --data-dir data/games/mixed8 --output runs/games-cache --device cuda

python -m duolm.train_next_item \
  --config configs/games/student.json --teacher-cache runs/games-cache \
  --data-dir data/games/mixed8 --output runs/games-student --device cuda
```

Prepared Games cohorts may be smaller than the caps when legal examples are
unavailable. Use the actual sizes recorded by preparation in a copy of the
Games config. Training checks them against the prepared cohort identities;
it does not manufacture or silently skip users to fit a requested size.

## Selecting and reporting checkpoints

Training opens only train and development records. The hash-defined selector
is a fixed subset of development users; its NDCG@10 selects checkpoints. Full
development results are reporting only and cannot replace that selection.

Use `duolm.evaluate` with the run's resolved config and a checkpoint directory:

```bash
python -m duolm.evaluate \
  --checkpoint-dir runs/books-student/checkpoints/step-SELECTED_STEP \
  --config runs/books-student/resolved-config.json \
  --data-dir data/books/candidates --output runs/books-student-report \
  --subset dev --device cuda
```

The three explicit subsets are:

| Subset | Users | Purpose |
| --- | --- | --- |
| `selector` | Frozen checkpoint-selector users | Inspect selection metric |
| `dev` | All development users | Full development reporting |
| `report` | Development users outside the selector | Disjoint development reporting |

The Books paper table uses all 5,000 development users (`--subset dev`), as
in the command above; matched teacher evaluation uses `--subset full-dev`.
The Games reported development comparison uses the 3,000 users outside the
selector (`--subset report`). `report` is the CLI default for an unspecified
new reporting run, but pass the intended subset explicitly for reproduction.
Every output records the subset, checkpoint hash,
prediction file hash, metrics, and `selects_checkpoint: false`. This command
restores model weights only, does not modify a training run, and never reads
test labels. The report complement is development data; do not label it test.
To evaluate a genuinely sealed test set, define and audit a separate reporting
protocol after checkpoint selection is frozen.

## Local interruption and resume

A run directory has a single-writer lock. Checkpoints are written to a temporary
directory, flushed, and atomically renamed. Existing checkpoint steps are
immutable. Model weights, optimizer moments, Python/Torch/CUDA RNG state,
query cursor and best-selection state are saved together.

```bash
python -m duolm.train_next_item \
  --config runs/books-student/resolved-config.json \
  --data-dir data/books/candidates --output runs/books-student \
  --resume runs/books-student/checkpoints/step-00001000 --device cuda
```

Resume uses the same effective configuration, prepared input bytes, cache
identity, source files and CUDA RNG device count. To stop cleanly, launch with
`--stop-request-file runs/stop-books`; create that file when a stop is needed.
Training finishes the current optimizer update, saves a checkpoint and exits.
Remove the request file before an explicit resume.

When moving a run, preserve the original best checkpoint as well as the latest.
The loader validates or restores that best checkpoint before any further
updates. Full resume from an original research checkout is intentionally not
silently compatible with the cleaned public source: source and configuration
hashes have changed. Do not edit manifests or disable identity checks to force
it. Use the explicit weights-only export path with its documented checks, or start a
fresh public run. The standalone student evaluator also requires matching
public training source. Model state-dict keys and scientific schema
and hash namespaces remain unchanged by the package rename.

## Offline verification

```bash
python -m unittest discover -s tests -p 'test_core_*.py' -v
```

The CPU pipeline test exercises real tiny Qwen modules through teacher training,
cache creation, student KD, exact interrupted/resumed model and optimizer
states, and all three evaluation subsets. Separate tests check packed causal
masks, candidate isolation, losses, bridge calibration, readouts, cache integrity
and corrupted-checkpoint rejection. CUDA-only numerical tests are skipped on a
CPU-only machine; a GPU training or throughput claim requires an actual GPU run.
