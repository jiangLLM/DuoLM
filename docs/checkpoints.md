# Portable inference checkpoints

`duolm-inference-v1` packages the complete trained ranker for new item text. V3
includes the trained encoder A, reader B, ranking head and calibrated bridge; T
includes B and its head. Model tensors retain their original FP32 values and
state keys. No optimizer, RNG, training cursor, raw reviews, user IDs, catalog,
private paths or full training configuration is included.

This format is separate from a resumable training checkpoint. Existing training
resume and `duolm.evaluate --checkpoint-dir` retain their exact configuration,
data and source checks. An old checkpoint cannot become a valid public full
resume by replacing its source hashes. Its original hashes are provenance only.

## Load and score new items

Install DuoLM from this repository, then download a published artifact at an
explicit revision. Use `local_dir` so the files are regular files, rather than
symlinks to the Hub's shared blob cache:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="OWNER/ARTIFACT",              # use the model card's repository
    revision="FULL_HUB_COMMIT_SHA",         # pin the reviewed artifact revision
    local_dir="checkpoints/duolm",
)
```

The loader ignores Hub `.cache` bookkeeping and a regular root `.gitattributes`.
All model files, optional README/license/notice files, and their checksums must
match the manifest; other files and symlinks are rejected. Hashes detect a changed
artifact, but are not a publisher signature. Pin and trust the repository and
revision you choose.

```python
from types import SimpleNamespace
import torch
from duolm.inference_checkpoint import load_ranker

ranker = load_ranker("checkpoints/duolm", device="cpu")
items = {
    "history": "A historical novel about a family crossing generations.",
    "a": "A character-driven historical novel.",
    "b": "An introductory guide to computer networking.",
}
query = SimpleNamespace(
    history=(SimpleNamespace(item_id="history", rating=5),),
    candidate_ids=("a", "b"),
)
with torch.inference_mode():
    scores = ranker([query], items)["scores"][0]
order = scores.argsort(descending=True).tolist()
print([(query.candidate_ids[i], scores[i].item()) for i in order])
```

Use game descriptions with a Games checkpoint. Item IDs are local lookup keys;
no training catalog or ground-truth label is needed. Ratings and history order
must follow the training interface. These are scalar ranking scores, not
probabilities, generated recommendations or calibrated scores across users.

Loading is offline. Native Qwen3 modules are constructed on the meta device from
bundled, hash-pinned public configs. Trained tensors are then assigned strictly;
no original base-model weights are downloaded or allocated. Deterministic
nonpersistent rotary buffers are reconstructed from those same configs.
Tokenizers load locally with remote code disabled. Keep weights FP32; select
computation precision with autocast if required by the evaluation protocol.

The default CPU SDPA path is useful for a small example. Full candidate groups
can exceed its 4,096-token dense reference limit. On supported CUDA hardware use
`load_ranker(path, device="cuda", attention_backend="flex_attention")`, the same
shared-tree backend as training. The checkpoint manifest records the recommended
compute dtype; loading itself does not change precision. A successful CPU example
does not establish GPU throughput or reproduce benchmark metrics.

## Evaluate prepared development data

The returned object is the ordinary `GroupRanker`. For labeled queries use its
`loss` interface, or `duolm.train_next_item.evaluate` with prepared development
queries, an explicit user cohort, item text, precision and packing limits. No
historical private training manifest is required for these ordinary model calls.

For example, after preparing the public Books data as in [data.md](data.md):

```python
import json
from pathlib import Path
from duolm.inference_checkpoint import load_ranker, verify_checkpoint
from duolm.train_next_item import load_dataset, evaluate

artifact = Path("checkpoints/duolm")
manifest = verify_checkpoint(artifact, load_tensors=False)
assert manifest["architecture"]["domain"] == "book"
config = json.loads(Path("configs/books/student.json").read_text())
_, dev, items, cohorts, data_hashes = load_dataset("data/books/candidates", config)
users = tuple(cohorts["dev"])  # Books reported table uses full 5,000-user dev
wanted = set(users)
queries = [q for q in dev if q.user_id in wanted]
ranker = load_ranker(artifact, device="cuda", attention_backend="flex_attention")
metrics = evaluate(
    ranker, queries, items, expected_users=users, device="cuda",
    compute_dtype=manifest["recommended_compute_dtype"],
    max_reader_packed_tokens=config["evaluation"]["max_reader_packed_tokens"],
    microbatch_queries=1,
    unavailable_users=[u for u in cohorts["unavailable_dev_users"] if u in wanted],
    evaluation_scope=config["schema_version"],
)
print(json.dumps({"artifact": manifest["manifest_sha256"], "subset": "dev",
                  "data_sha256": data_hashes, "metrics": metrics}, indent=2))
```

For Games, use its artifact, `configs/games/student.json` and `data/games/mixed8`.
Its reported table uses the disjoint report cohort: select users in
`cohorts["dev"]` absent from `cohorts["checkpoint_selector"]`, label the output
`report`, and retain BF16 inference. For a teacher evaluation, also preserve the
teacher's specified shared-tree execution and strict math protocol. Recomputing
metrics on new prepared inputs creates a new evaluation; preserve its artifact,
data, subset and execution identities. Do not use test labels to select models.

## Export an owned training checkpoint

The exporter accepts `training.pt` (or its step directory), adjacent original
`manifest.json`, and the exact original effective config JSON. A run manifest
containing `config` and `identity` is also accepted. Only convert checkpoints you
own or trust. Input deserialization uses PyTorch's restricted `weights_only=True`
loader and never imports or executes the original project's source.

Supply local directories containing the hash-pinned public `config.json`,
`tokenizer.json` and `tokenizer_config.json` for each backbone. `vocab.json` and
`merges.txt` are copied if present. The supported immutable revisions are listed
in `duolm.assets.MODEL_ASSETS`; metadata-only Hub downloads are enough. Neither
original base-weight files nor the original local model paths are needed.

```bash
python -m duolm.inference_checkpoint export \
  --checkpoint runs/original/checkpoints/step-00007424 \
  --config runs/original/config.json \
  --a-model metadata/encoder \
  --b-model metadata/reader \
  --expected-checkpoint-sha256 ORIGINAL_TRAINING_PT_SHA256 \
  --model-card release/README.md \
  --license-file release/LICENSE \
  --notice-file release/NOTICE \
  --output release/duolm-books

python -m duolm.inference_checkpoint verify --artifact release/duolm-books
```

Omit `--a-model` for T. README, LICENSE and NOTICE are optional export arguments
and become checksum-bound files; include them before export rather than editing
a verified artifact afterward. The exporter never overwrites an existing output.
It validates original config/state/checkpoint checksums, the domain and public
base-model identities, then verifies every exported tensor and bridge field by
exact equality after loading through the current public architecture.

Initially supported recipes are T and V3 with LastToken, one item vector, shared
candidate header and candidate-position scoring. Other ablations fail explicitly.

`provenance.json` contains the original checkpoint/manifest/config/identity/data
identity hashes, step, source basenames and hashes, and conversion-source hash.
It does not rewrite original source identities. `full_resume_supported` and
`historical_behavior_verified` remain false: exact serialized-state preservation
and tiny synthetic round-trip tests do not establish historical trained-model
behavior or reproduce paper scores. Record any actual-checkpoint evaluation
separately, with its explicit data and runtime protocol.
