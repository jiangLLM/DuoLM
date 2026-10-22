# Export and serving

The compact runtime scores a candidate set with the trained scalar head. It
loads a frozen reader and one offline vector per catalog item, compiles a
shared history prefix, and isolates candidate branches with Multi-Item Scoring
(MIS). The public execution comparisons are:

| Configuration | Score mode | Prefill graph | Purpose |
| --- | --- | --- | --- |
| Main | `shared` | `breakable` | Shared MIS with prefill graph replay |
| Graph ablation | `shared` | `disabled` | Same exported model, without prefill graphs |
| Sharing ablation | `independent` | `disabled` | Ordinary causal sequence per candidate |

All three use FP16 reader computation and an FP32 scalar head. They disable
radix caching. Independent execution submits candidates asynchronously; it does
not replace the shared method with a serial candidate loop. The vocabulary
head in the Hugging Face reader directory is a loader placeholder, never the
trained score head.

Use `--arithmetic-mode sglang` for these rows. The retained `reference_fp16`
option is a bounded numerical diagnostic for the 0.6B/N=100 layout: it forces
one uncached physical request and does not support graph replay. It is not a
throughput configuration.

## Two separate environments

Use the training environment (`.venv`, Transformers **4.57.1**) for exports,
workloads, reference scores and replay. Use Linux x86_64, Python 3.12, glibc
2.34 or newer, a CUDA 13 compatible NVIDIA driver, and a separate environment
for SGLang. The measured hardware was one H100 80 GB. macOS CPU tests cannot
establish GPU compatibility or throughput.

The serving versions are SGLang **0.5.21**, PyTorch **2.13.0** with CUDA 13,
Transformers **5.18.0**, tokenizers **0.23.2**, and FlashInfer **0.6.18**.
SGLang's [published package metadata](https://pypi.org/pypi/sglang/0.5.21/json)
pins older Transformers/tokenizers. `requirements-serving-overrides.txt`
explicitly records the two overrides used by the measured runtime. Do not
install the training requirements into this environment.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv venv --python 3.12 .venv-serving
uv pip sync --python .venv-serving/bin/python requirements-serving.lock
uv pip install --python .venv-serving/bin/python --no-deps -e .
.venv-serving/bin/python -c 'import torch, transformers, tokenizers, sglang; print(torch.__version__, torch.version.cuda, transformers.__version__, tokenizers.__version__); assert torch.cuda.is_available()'
```

The lock contains a complete dependency resolution for Linux x86_64/Python
3.12. It was resolved from public PyPI; the transitive dependency closure is a
release-time resolution, not a claim that every dependency matches the
historical environment. To regenerate it with the same deliberate overrides:

```bash
uv pip compile requirements-serving.txt \
  --override requirements-serving-overrides.txt \
  --python-version 3.12 --python-platform x86_64-manylinux_2_34 \
  --default-index https://pypi.org/simple --no-config --prerelease=allow \
  --output-file requirements-serving.lock
```

`--prerelease=allow` is required by upstream's pinned CUDA dependencies. A
plain `pip install -r requirements-serving.txt` cannot resolve the two
conflicting metadata pins. `pip check` can report these deliberate upstream
pin mismatches; it does not replace the source checks and numerical pilot below.
Actual CUDA execution is still required on the target host after installation.

## Export the trained checkpoint

First complete [data preparation](data.md) and [training](training.md). Keep
the exact training source checkout, effective `resolved-config.json`,
checkpoint/manifest, prepared data, and pinned public model snapshots. Exports
hash-check these identities before loading checkpoint weights. Edited source
cannot silently export a checkpoint trained by different source bytes.

Run from the repository root in Bash. These Books paths match the training
commands. Select the checkpoint recorded by that run, rather than hard-coding
a historical step:

```bash
export REF_PY=.venv/bin/python
export SG_PY=.venv-serving/bin/python
export DUOLM_RUN=runs/books-student
export DUOLM_DATA=data/books/candidates
export DUOLM_EXPORT=artifacts/books-serving
export DUOLM_GPU=0
export DUOLM_CHECKPOINT="$($REF_PY -c 'import json,os; print(json.load(open(os.environ["DUOLM_RUN"] + "/best_checkpoint.json"))["path"])')"

"$REF_PY" -m duolm.nextbook_export \
  --source . --checkpoint "$DUOLM_CHECKPOINT" \
  --config "$DUOLM_RUN/resolved-config.json" --data "$DUOLM_DATA" \
  --a-model data/models/encoder --b-model data/models/reader \
  --output "$DUOLM_EXPORT" --vector-device cuda --vector-dtype bfloat16 \
  --batch-size 32
```

The export contains `reader/`, `scalar_head.safetensors`, `vectors.npy`,
`index.jsonl`, and a checksummed inventory `manifest.json`. It does not copy training
rows into the serving directory. Output directories must be new. Full-catalog
vector construction can be expensive; it happens once, outside serving timing.

For a new Games checkpoint, use its own run, `data/games/mixed8`, and a separate
export directory. The exporter and compiler preserve the Games instruction and
checkpoint identity. That compatibility extension is CPU-tested; the Books
throughput result does not establish Games serving throughput.

## Workload and independent score reference

The replay workload contains exactly 1,000 distinct development users with ten
history items and 100 candidates each. Selection ignores labels and model
scores. The ten-candidate condition is a nested subset of each 100-candidate
set. It may omit the positive item and is for timing, not ranking quality.

```bash
export DUOLM_BENCH=results/serving-new-run
mkdir -p "$DUOLM_BENCH" run/serving
"$REF_PY" -m duolm.nextbook_benchmark \
  --dev "$DUOLM_DATA/dev.jsonl" --users 1000 --candidates 10 100 \
  --seed 17 --output "$DUOLM_BENCH/workload.json"
"$REF_PY" -m duolm.nextbook_benchmark \
  --dev "$DUOLM_DATA/dev.jsonl" --users 10 --candidates 10 100 \
  --seed 17 --output "$DUOLM_BENCH/pilot.json"
CUDA_VISIBLE_DEVICES="$DUOLM_GPU" "$REF_PY" scripts/build_nextbook_reference.py \
  --export "$DUOLM_EXPORT" --workload "$DUOLM_BENCH/pilot.json" \
  --output "$DUOLM_BENCH/reference.json"
```

Insufficient eligible users causes a failure, not an unreported sample-size
reduction. The reference uses independent causal PyTorch sequences on the
same FP16 offline vectors, with both FP16 and FP32 reader diagnostics and the
original FP32 head. Build it before starting a serving process on that GPU.

## Install and verify the SGLang overlay

The overlay preserves the installed SGLang package and creates a separate
copy with narrowly scoped hooks. It checks SHA256 identities for fourteen
upstream source files and refuses an unsupported package before writing the
completed overlay. A matching version string alone is insufficient. Runtime
source files are bound into the overlay receipt too; rebuild the overlay after
changing this repository's runtime code.

```bash
export DUOLM_OVERLAY=artifacts/sglang-overlay
"$SG_PY" -m duolm.nextbook_sglang_overlay install --output "$DUOLM_OVERLAY"
"$SG_PY" -m duolm.nextbook_sglang_overlay verify --output "$DUOLM_OVERLAY"
```

Do not bypass a source mismatch or substitute a nearby SGLang release. The
public SGLang 0.5.21 CPython 3.12 Linux x86_64 wheel was checked against all
fourteen source pins. SGLang remains Apache-2.0 licensed; see the
[third-party notices](../THIRD_PARTY_NOTICES.md).

## Serve and replay the three configurations

Launch one resident service at a time. Start with the main row below in a
terminal carrying the variables above:

```bash
export DUOLM_ROW=shared-graph
export DUOLM_MODE=shared
export DUOLM_GRAPH=breakable
export DUOLM_SOCKET="run/serving/$DUOLM_ROW.sock"
CUDA_VISIBLE_DEVICES="$DUOLM_GPU" "$SG_PY" -m duolm.nextbook_server \
  --variant V3 --score-mode "$DUOLM_MODE" --prefill-graph "$DUOLM_GRAPH" \
  --export "$DUOLM_EXPORT" --overlay "$DUOLM_OVERLAY" \
  --arithmetic-mode sglang --chunk-concurrency 100 --kv-tokens 262144 \
  --prefill-graph-token-buckets 369,738,1476,2952,5904,11808,17712,23616 \
  --socket "$DUOLM_SOCKET" --ready "$DUOLM_BENCH/$DUOLM_ROW-ready.json"
```

In a second terminal, set the same variables and wait for the ready file. A
ready file is written only after the bound scalar head and item table have
been installed by the worker. Check the live service before replay:

```bash
"$REF_PY" - <<'PY'
import asyncio, json, os
from duolm.nextbook_comparison_benchmark import control
print(json.dumps(asyncio.run(control(os.environ["DUOLM_SOCKET"], "info")), indent=2))
PY
"$REF_PY" -m duolm.nextbook_comparison_benchmark \
  --socket "$DUOLM_SOCKET" --workload "$DUOLM_BENCH/pilot.json" \
  --reference "$DUOLM_BENCH/reference.json" --mode closed \
  --candidates 100 --concurrency 1 --seconds 1 --warmup 10 \
  --flush-after-warmup --timeout 180 \
  --output "$DUOLM_BENCH/pilot-$DUOLM_ROW"
```

Inspect `numerical.json`: all ten N=100 pilot queries must be compared. Review
maximum scalar error, rank flips, top-1 equality and top-10 overlap. These are
numerical diagnostics; a written `summary.json` alone is not proof of score
agreement. Do not mix references from different exports. Once the pilot is
understood, run the complete closed-loop grid for that resident service:

```bash
for n in 100 10; do
  concurrencies="1 32"
  if [ "$n" = 100 ]; then concurrencies="1 8 32 64"; fi
  for c in $concurrencies; do
    for repeat in 1 2 3; do
      "$REF_PY" -m duolm.nextbook_comparison_benchmark \
        --socket "$DUOLM_SOCKET" --workload "$DUOLM_BENCH/workload.json" \
        --reference "$DUOLM_BENCH/reference.json" --mode closed \
        --candidates "$n" --concurrency "$c" --seconds 30 --warmup 10 \
        --flush-after-warmup --timeout 180 --seed "$((16 + repeat))" \
        --gpu "$DUOLM_GPU" \
        --output "$DUOLM_BENCH/$DUOLM_ROW-n$n-c$c-r$repeat" || exit 1
    done
  done
done
"$REF_PY" - <<'PY'
import asyncio, os
from duolm.nextbook_comparison_benchmark import control
async def stop():
    socket = os.environ["DUOLM_SOCKET"]
    state = await control(socket, "info")
    if state.get("active") != 0:
        raise RuntimeError("Service still has active requests")
    await control(socket, "stop")
asyncio.run(stop())
PY
```

Repeat the launch, pilot, matrix, and stop sequence for `shared-eager` with
`DUOLM_MODE=shared`, `DUOLM_GRAPH=disabled`, then `independent-eager` with
`DUOLM_MODE=independent`, `DUOLM_GRAPH=disabled`. Update `DUOLM_SOCKET` in both
terminals after each row change. Use the same export, trace, precision and GPU.
These three rows produce 54 formal runs (18 per row). A timeout does not prove
GPU cancellation; inspect service activity before any subsequent replay.

For the graph row, inspect before/after worker graph counters, captured buckets
and `debug_eager_replays`. Real `graph_requests` increments and zero debug-eager
replays are required to establish that graph execution happened. Capture or
`--prefill-graph breakable` alone is not evidence of replay. Counter deltas also
include warmup, so do not label them measurement-only request counts.

## Optional full-text teacher comparison

The same runtime supports a full-text teacher, with graphs disabled. Export
the specific teacher used to generate the student's cache (Books step 4000 in
the documented recipe), using its own prepared data and resolved config:

```bash
export DUOLM_T_CHECKPOINT=runs/books-teacher/checkpoints/step-00004000
export DUOLM_T_EXPORT=artifacts/books-teacher-serving
export DUOLM_T_SHA="$($REF_PY -c 'import json,os; print(json.load(open(os.environ["DUOLM_T_CHECKPOINT"]+"/manifest.json"))["files"]["training.pt"]["sha256"])')"
"$REF_PY" -m duolm.nextbook_teacher_export \
  --source . --checkpoint "$DUOLM_T_CHECKPOINT" \
  --config runs/books-teacher/resolved-config.json --data data/books/mixed8 \
  --b-model data/models/reader --expected-checkpoint-sha256 "$DUOLM_T_SHA" \
  --output "$DUOLM_T_EXPORT"
CUDA_VISIBLE_DEVICES="$DUOLM_GPU" "$REF_PY" scripts/build_nextbook_teacher_reference.py \
  --export "$DUOLM_T_EXPORT" --workload-export "$DUOLM_EXPORT" \
  --workload "$DUOLM_BENCH/pilot.json" --device cuda \
  --output "$DUOLM_BENCH/reference-teacher.json"
```

Launch `duolm.nextbook_server` with `--variant T --score-mode shared
--prefill-graph disabled --pretokenize-catalog --export "$DUOLM_T_EXPORT"`
and the same overlay, socket/ready, concurrency and KV options. Replay with
`reference-teacher.json`. Catalog tokenization happens at startup, outside the
timing window, and is not caching complete query scores. This is the SGLang
teacher baseline; it does not reproduce the separate vLLM comparison row.

## Read the measurements correctly

One request is a complete user query containing all N candidates. Concurrency
counts these queries, not candidate sequences. `summary.json` reports validated
candidates divided by actual elapsed time including the final drain. The client
timer includes compilation, queueing, inference, sorting and local Unix IPC;
startup, export, catalog preparation and warmup are excluded. `--seconds` is a
minimum: each invocation completes whole passes over the trace. Aggregate the
median rate of the three complete runs at each matched setting. Do not pool
latencies and call that a median of run-level p99 values.

The paper's **26,403.98 candidates/s** is the historical Books result at N=100,
concurrency 64 on one H100, from the 1,000-user/90-run campaign. That complete
campaign also included two teacher systems, whereas the compact three-row
replay above has 54 runs. The measured main runtime algorithms are retained;
package imports, checkpoint portability and Games input handling were adapted
for this release. CPU tests and public-wheel source verification do not rerun
that GPU result. Shared/independent and graph-on/off ratios include the whole
serving path, not just attention kernels. No closed-loop result establishes
SLA-qualified capacity; the optional open-loop benchmark mode is a separate
measurement protocol.
