# Data preparation

Run commands from the repository root after installing the project. The commands
below create data locally; this repository contains no real user histories,
catalog dumps, model weights, or prepared evaluation labels. `data/` is ignored
by Git. Downloading the complete official metadata takes several GB, and the
catalog/index stages require substantial RAM, disk, and compute.

The source is [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/).
Use its official 5-core rating-only CSVs and category metadata. Games additionally
needs the original 0-core review JSONL to establish when each item was first
observed. Existing copies can be passed directly to the corresponding commands;
the preparation manifests bind their bytes with SHA-256.

## Pinned models

Both domains use these exact public revisions:

| Role | Repository | Revision |
| --- | --- | --- |
| Ranker/tokenizer B | `Qwen/Qwen3-0.6B-Base` | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` |
| Item encoder/tokenizer A | `Qwen/Qwen3-Embedding-0.6B` | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |

```bash
python -m duolm.assets --workspace data --models-only
```

This creates `data/models/reader` (the **Base** model),
`data/models/encoder`, and a `snapshot-manifest.json` in each directory.
The downloader checks pinned per-file SHA-256 values and writes a manifest of
file sizes and hashes. Preparation verifies local snapshot identities before
loading tokenizers. The instruction-tuned `Qwen/Qwen3-0.6B` is a different model
and must not be substituted.

## Books

### Download and freeze item membership

```bash
python -m duolm.assets --workspace data/books --domain books --data-only
python -m duolm.inventory \
  --source data/books/raw/Books.csv.gz \
  --output data/books/inventory
```

The original Books inventory is the sorted unique `parent_asin` column of the
official **5-core rating-only** `Books.csv.gz`. It is not every record in the
metadata file. The recovered original preparation has 9,488,297 source rows and
495,063 distinct item IDs. The inventory generator preserves its compact sorted
JSON representation and records the input and output hashes. It reads no rating,
timestamp, or user association for membership decisions; the upstream 5-core
selection is already based on interactions, so this is not an independent
metadata-only catalog. Metadata availability is checked afterward.

The downloader pins the original source bytes:

| Source | SHA-256 |
| --- | --- |
| `Books.csv.gz` | `28c5104d0c2f7a7e842da3431606e21a245d8fb317e3434cd20b5c5a7b521615` |
| `meta_Books.jsonl.gz` | `d41f2348d3e0f3099d2a7ee45b304970282199f92a7ccdfe8ba4b1f753009a10` |

### Catalog, cohorts, and chronological contexts

```bash
python -m duolm.build_next_item_assets catalog \
  --metadata data/books/raw/meta_Books.jsonl.gz \
  --catalog-item-ids data/books/inventory/catalog-item-ids.json \
  --inventory-manifest data/books/inventory/manifest.json \
  --a-tokenizer-path data/models/encoder \
  --b-tokenizer-path data/models/reader \
  --output data/books/assets

python -m duolm.build_next_item_assets cohorts \
  --ratings data/books/raw/Books.csv.gz \
  --catalog data/books/assets/catalog.json \
  --output data/books/assets

python -m duolm.build_next_item_assets stage \
  --ratings data/books/raw/Books.csv.gz \
  --assets data/books/assets \
  --a-tokenizer-path data/models/encoder \
  --b-tokenizer-path data/models/reader \
  --output data/books/contexts
```

The catalog uses title, author, and description only. Rating averages, review
text, and user associations are excluded from item text. One literal text
prefix must fit both tokenizers' 512-token limits. Conflicting metadata records
for the same parent ID fail rather than choosing one silently.

The original 80/10/10 user split and deterministic cap ordering are preserved.
Caps are 50,000 qualified train users, 5,000 dev identities, and 10,000 sealed
test identities, with a fixed 2,000-user dev checkpoint selector. Only train
users are qualified from labels. Dev/test identities are frozen before measuring
availability. Books does not read test labels during this pipeline.

Chronology is constructed before metadata checks; missing content never joins
nonadjacent events into a new transition. Histories contain at most 10 events,
while candidate exclusions cover **all** earlier items. At most five temporal
quantile queries are selected per train user. Only the final original time group
is considered for each dev user; an invalid final transition is not replaced by
an earlier valid one. Unavailable dev users
remain explicitly recorded and count in the original evaluation denominator.

### Encode the catalog and retrieve using history only

```bash
python -m duolm.next_item_retrieval encode \
  --catalog data/books/contexts/catalog.json \
  --model-path data/models/encoder \
  --output data/books/index --domain book --device cuda

python -m duolm.next_item_retrieval retrieve \
  --catalog data/books/contexts/catalog.json \
  --inputs data/books/contexts/retrieval-inputs.jsonl \
  --index data/books/index \
  --output data/books/retrieval --domain book --device cuda

python -m duolm.prepare_next_item assemble \
  --stage data/books/contexts \
  --retrieval data/books/retrieval/retrieval.jsonl \
  --retrieval-manifest data/books/retrieval/retrieval-manifest.json \
  --output data/books/mixed8
```

The encoder uses last-valid-token pooling and L2 normalization. Embeddings are
stored in FP16; retrieval uses FP32 inner products without TF32. The query is an
equally weighted aggregate of visible item vectors, with no rating weighting or
target text. Top-200 train and Top-100 dev retrieval results exclude all past
items. Index and retrieval shards can resume only with the same frozen
configuration, code, and input hashes.

`mixed8/train.jsonl` contains the target, four random unobserved items, and three
items sampled from the legal history-retrieval pool. Candidate order is shuffled
deterministically. `mixed8/dev.jsonl` is the original **retrieved-100** protocol:
the target can be absent and those misses are retained. This source is needed
for the original teacher-training protocol; do not report it as random-100.

### Freeze the student/evaluation candidate protocol

```bash
python -m duolm.prepare_candidate_ranking \
  --source-dir data/books/mixed8 \
  --train-candidates 100 --train-policy random-unobserved \
  --output data/books/random100

python -m duolm.prepare_candidate_ranking \
  --source-dir data/books/mixed8 \
  --train-candidates 8 --train-policy mixed-history \
  --eval-source-dir data/books/random100 \
  --output data/books/candidates
```

`candidates/train.jsonl` is a byte-for-byte copy of the frozen mixed-8 training
groups. `candidates/dev.jsonl` is a byte-for-byte copy of the frozen random-100
evaluation reference: one target plus 99 uniformly sampled unobserved items.
The sampler excludes every previous item, uses independent per-query sampling
and ordering RNGs, and binds the complete policy and source manifests. Evaluate
models on the same `candidates` directory when comparing fixed-candidate ranking
quality. Random-100 and retrieved-100 metrics answer different questions.

## Video Games

```bash
python -m duolm.assets --workspace data/games --domain games --data-only

python -m duolm.build_games_assets stage \
  --ratings data/games/raw/Video_Games.csv.gz \
  --raw-reviews data/games/raw/Video_Games.jsonl.gz \
  --metadata data/games/raw/meta_Video_Games.jsonl.gz \
  --a-tokenizer-path data/models/encoder \
  --b-tokenizer-path data/models/reader \
  --output data/games/stage

python -m duolm.build_games_assets contexts \
  --stage data/games/stage \
  --ratings data/games/raw/Video_Games.csv.gz \
  --output data/games/contexts

python -m duolm.next_item_retrieval encode \
  --catalog data/games/contexts/catalog.json \
  --model-path data/models/encoder \
  --output data/games/index --domain game --device cuda

python -m duolm.next_item_retrieval retrieve \
  --catalog data/games/contexts/catalog.json \
  --inputs data/games/contexts/retrieval-inputs.jsonl \
  --first-seen data/games/contexts/first-seen.json \
  --index data/games/index \
  --output data/games/retrieval --domain game --device cuda

python -m duolm.build_games_assets materialize \
  --contexts data/games/contexts \
  --retrieval data/games/retrieval/retrieval.jsonl \
  --retrieval-manifest data/games/retrieval/retrieval-manifest.json \
  --output data/games/mixed8
```

Games catalog membership comes from the official 5-core item identities.
Canonical content contains title, categories, features, and description.
The first-observation sidecar is the earliest **raw 0-core** timestamp for each
item; the 5-core timestamp is not substituted. Every candidate must have been
observed by the query cutoff. Timestamps must be Unix milliseconds. The same
split/cap namespaces and chronological rules used for Books are preserved.

Games retrieves only for training. Training groups contain one target, four
random time-legal contrasts, and three retrieved time-legal contrasts; dev/test
groups contain the target and 99 random time-legal contrasts. Users without a
legal final target or sufficient candidate pool remain in the dev/test
denominator, with an explicit unavailability reason. No easier query replaces
their fixed final query.

The Games context/materialization stage prepares and seals test labels in
`sealed-test.jsonl`. Normal training/development loading never opens that file.
Final test loading is an explicit operation bound to the preparation identity;
it is not a development-time model-selection input.

The Games downloader checks the following original source hashes recovered from
the experiment's verified source-byte archive record. The builders bind the
source bytes into the frozen preparation manifests, and subsequent runs reject
changes. A fresh public download must match these values before it is accepted.

| Source | SHA-256 |
| --- | --- |
| `Video_Games.csv.gz` | `a2bde5f3b945960d161538c200dd87845e6ee471b46da96410dde61613c6901c` |
| `Video_Games.jsonl.gz` | `52d8762c94a5b99d3bccdb31c15cc6c99e05c89f9042cd3753b3b623a44bc193` |
| `meta_Video_Games.jsonl.gz` | `ced9696b0874121327a438bfe9d3d01aba45da158e58f94f4920352d75fc1fc9` |

## Verification and recovery

Run the offline contracts and tiny end-to-end preparation smoke:

```bash
python -m pytest -q \
  tests/test_inventory.py tests/test_assets.py \
  tests/test_build_next_item_assets.py tests/test_games_assets.py \
  tests/test_next_item_data.py tests/test_next_item_retrieval.py \
  tests/test_prepare_next_item.py tests/test_prepare_candidate_ranking.py \
  tests/test_public_data_pipeline.py
```

The smoke creates synthetic rows, builds item inventories/catalogs/cohorts,
encodes with a tiny injected CPU model, performs actual retrieval, and freezes
candidate sets. It verifies missing-user accounting and sealed-label separation.
It validates execution and data contracts; it does not reproduce paper scores.
For a small real CPU run, the encoder CLI supports `--device cpu
--compute-dtype float32`; changing dtype changes the recorded preparation
identity and must not be called an exact BF16 replay.

Keep completed manifests with their data. Catalog/cohort and index/retrieval
phases verify completed outputs on rerun; Books context assembly uses a fresh
output directory. A checksum/configuration mismatch is an error, not a request
to silently regenerate data. Use a new output directory for changed inputs or
protocols. The historical protocol strings, hash namespaces, and seed 17 are
intentional scientific identities even though the public Python package is
named `duolm`.
