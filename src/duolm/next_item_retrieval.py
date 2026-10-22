"""Frozen public-embedding retrieval over target-free next-item inputs.

Item exports and query results have resumable, checksummed shards. No training
labels, chronological context file, future records or teacher model are opened.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from .build_next_item_assets import ENCODER_REPO, ENCODER_REVISION, _atomic_json, verify_public_snapshot
from .next_item_data import HistoryEntry, user_split
from .prepare_next_item import file_hash

ENCODER_IDENTITY = {"repo": ENCODER_REPO, "revision": ENCODER_REVISION}
INDEX_SCHEMA = "books-next-item-embedding-index-v1"


def _domain_schema(domain, kind):
    if domain not in {"book", "game"}:
        raise ValueError("retrieval domain must be book or game")
    return ("books" if domain == "book" else "games") + "-next-item-" + kind + "-v1"


def _numpy_file(path, array):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix="." + path.name, delete=False) as stream:
            temporary = Path(stream.name)
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _catalog(path):
    items = json.loads(Path(path).read_text())
    if not isinstance(items, dict) or not items or any(
            not isinstance(item, str) or not item or not isinstance(text, str) or not text.strip()
            for item, text in items.items()):
        raise ValueError("catalog must map item IDs to canonical nonempty text")
    return items


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _autocast(device, dtype):
    import torch
    return torch.autocast(torch.device(device).type, dtype=torch.bfloat16) if dtype == "bfloat16" else nullcontext()


def last_valid_unit_vectors(hidden, attention_mask):
    """Handle left or right padding; normalize last-valid states in FP32."""
    import torch
    if hidden.ndim != 3 or tuple(hidden.shape[:2]) != tuple(attention_mask.shape):
        raise ValueError("hidden states and attention mask have incompatible shapes")
    mask = attention_mask.bool()
    if not bool(mask.any(dim=1).all()):
        raise ValueError("embedding input cannot have an empty token sequence")
    positions = torch.arange(hidden.shape[1], device=hidden.device).expand_as(mask)
    last = positions.masked_fill(~mask, -1).max(dim=1).values
    selected = hidden[torch.arange(len(hidden), device=hidden.device), last].float()
    norms = torch.linalg.vector_norm(selected, dim=-1, keepdim=True)
    if not bool(torch.isfinite(selected).all()) or not bool((norms > 1e-12).all()):
        raise ValueError("public encoder produced nonfinite or zero vectors")
    return selected / norms


def _check_index_shard(root, record, *, expected_start, identity, dimension=None):
    import numpy as np
    if record.get("start") != expected_start or not isinstance(record.get("end"), int) or record["end"] <= expected_start:
        raise ValueError("index shard ranges must form a contiguous ordered prefix")
    path = root / record["file"]
    if not path.is_file() or file_hash(path) != record.get("sha256"):
        raise ValueError("index shard checksum mismatch")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.float16 or array.ndim != 2 or array.shape[0] != record["end"] - record["start"]:
        raise ValueError("index shard must contain the full FP16 item-vector range")
    if dimension is not None and array.shape[1] != dimension:
        raise ValueError("index embedding dimensions changed between shards")
    if not np.isfinite(array).all() or np.any(np.linalg.norm(array.astype(np.float32), axis=1) < 1e-6):
        raise ValueError("index shard contains invalid vectors")
    receipt = json.loads((root / record["receipt"]).read_text())
    if receipt.get("identity") != identity or receipt.get("shard") != record:
        raise ValueError("index shard receipt identity differs from the export")
    return array.shape[1]


def export_index(catalog, output, *, tokenizer, model, device="cuda", batch_size=64,
                 shard_items=10_000, compute_dtype="bfloat16", model_identity=None,
                 max_item_tokens=512, max_new_shards=None, domain="book"):
    """Encode each item exactly once, resuming only verified complete shards.

    Batch size/dtype/shard boundaries are frozen on resume. An OOM propagates;
    there is no hidden token truncation, dtype change, or automatic batch shrink.
    """
    import numpy as np
    import torch
    index_schema = _domain_schema(domain, "embedding-index")
    for value, name in ((batch_size, "batch size"), (shard_items, "shard items"), (max_item_tokens, "item tokens")):
        _positive(value, name)
    if compute_dtype not in ("float32", "bfloat16"):
        raise ValueError("export compute dtype must be float32 or bfloat16")
    if max_new_shards is not None:
        _positive(max_new_shards, "max new shards")
    items = _catalog(catalog)
    ids = sorted(items)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = {"catalog_sha256": file_hash(catalog), "model": model_identity or ENCODER_IDENTITY,
                "parameters": {"batch_size": batch_size, "shard_items": shard_items,
                               "compute_dtype": compute_dtype, "max_item_tokens": max_item_tokens,
                               "pooling": "last_valid_token", "normalize": "fp32_l2", "storage": "float16",
                               "padding": "batch_only_no_truncation", "item_order": "lexicographic_id"},
                "code_sha256": file_hash(__file__), "device_type": torch.device(device).type}
    if domain == "game":
        identity.update(domain=domain, schema=index_schema)
    state_path, complete = output / "index-state.json", output / "index-manifest.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("schema") != index_schema or state.get("identity") != identity:
            raise ValueError("existing index differs from requested catalog/model/configuration/code")
        if file_hash(output / "item_ids.json") != state.get("item_ids_sha256"):
            raise ValueError("frozen item-ID index checksum mismatch")
        if json.loads((output / "item_ids.json").read_text()) != ids:
            raise ValueError("frozen item-ID index order differs from the catalog")
    else:
        if any(output.iterdir()):
            raise ValueError("untracked index output is not overwritten")
        _atomic_json(output / "item_ids.json", ids)
        state = {"schema": index_schema, "identity": identity, "items": len(ids), "shards": [],
                 "item_ids_sha256": file_hash(output / "item_ids.json"), "complete": False}
        _atomic_json(state_path, state)
    position, dimension = 0, None
    for record in state["shards"]:
        dimension = _check_index_shard(output, record, expected_start=position, identity=identity, dimension=dimension)
        position = record["end"]
    if position > len(ids):
        raise ValueError("index shard extends beyond the frozen catalog")
    if complete.exists():
        report = json.loads(complete.read_text())
        if report != state or not report.get("complete") or position != len(ids):
            raise ValueError("completed index manifest differs from its verified shards")
        return report
    was_training = model.training
    model.eval()
    newly_written = 0
    try:
        for start in range(position, len(ids), shard_items):
            if max_new_shards is not None and newly_written == max_new_shards:
                return state
            end = min(start + shard_items, len(ids))
            filename = f"shards/items-{start:09d}-{end:09d}.npy"
            receipt_name = filename + ".json"
            if (output / receipt_name).exists():
                receipt = json.loads((output / receipt_name).read_text())
                record = receipt.get("shard", {})
                if record.get("file") != filename or record.get("end") != end:
                    raise ValueError("orphan index receipt does not match the expected shard")
                dimension = _check_index_shard(output, record, expected_start=start, identity=identity, dimension=dimension)
            else:
                lengths = []
                for index in range(start, end):
                    length = len(tokenizer.encode(items[ids[index]], add_special_tokens=False))
                    if not 0 < length <= max_item_tokens:
                        raise ValueError("canonical item exceeds encoder budget; no implicit truncation")
                    lengths.append((length, index))
                ordered = [index for _, index in sorted(lengths)]
                vectors = None
                with torch.inference_mode():
                    for offset in range(0, len(ordered), batch_size):
                        positions = ordered[offset:offset + batch_size]
                        tokens = tokenizer([items[ids[index]] for index in positions], padding=True,
                                           truncation=False, add_special_tokens=False, return_tensors="pt")
                        tokens = {name: value.to(device) for name, value in tokens.items()}
                        if not bool((tokens["attention_mask"].sum(dim=1) <= max_item_tokens).all()):
                            raise ValueError("batched tokenizer changed the frozen item budget")
                        with _autocast(device, compute_dtype):
                            result = model(**tokens)
                        hidden = result.last_hidden_state if hasattr(result, "last_hidden_state") else result["last_hidden_state"]
                        encoded = last_valid_unit_vectors(hidden, tokens["attention_mask"]).cpu().numpy().astype(np.float16)
                        if dimension is not None and encoded.shape[1] != dimension:
                            raise ValueError("public encoder output dimension changed")
                        dimension = encoded.shape[1]
                        if vectors is None:
                            vectors = np.empty((end - start, dimension), dtype=np.float16)
                        vectors[np.array(positions) - start] = encoded
                destination = output / filename
                if destination.exists():
                    previous = np.load(destination, allow_pickle=False)
                    if not np.array_equal(previous, vectors):
                        raise ValueError("unreceipted index shard differs from recomputed values; not overwritten")
                else:
                    _numpy_file(destination, vectors)
                record = {"file": filename, "receipt": receipt_name, "start": start, "end": end,
                          "dimension": dimension, "sha256": file_hash(destination)}
                _atomic_json(output / receipt_name, {"identity": identity, "shard": record})
            state["shards"].append(record)
            state["dimension"] = dimension
            _atomic_json(state_path, state)
            newly_written += 1
    finally:
        model.train(was_training)
    state["complete"] = True
    _atomic_json(state_path, state)
    _atomic_json(complete, state)
    return state


def load_index(index, catalog, *, device="cuda", domain="book"):
    import numpy as np
    import torch
    root = Path(index)
    report = json.loads((root / "index-manifest.json").read_text())
    if (report.get("schema") != _domain_schema(domain, "embedding-index") or report.get("complete") is not True
            or report["identity"]["catalog_sha256"] != file_hash(catalog)):
        raise ValueError("retrieval requires a completed matching catalog index")
    ids = json.loads((root / "item_ids.json").read_text())
    if file_hash(root / "item_ids.json") != report.get("item_ids_sha256") or ids != sorted(set(ids)):
        raise ValueError("index item IDs are not the frozen sorted unique identities")
    if set(ids) != set(_catalog(catalog)) or len(ids) != report.get("items"):
        raise ValueError("index does not cover exactly the frozen catalog")
    chunks, position, dimension = [], 0, None
    for record in report["shards"]:
        dimension = _check_index_shard(root, record, expected_start=position, identity=report["identity"], dimension=dimension)
        array = np.load(root / record["file"], mmap_mode="r", allow_pickle=False)
        # Restore unit length after FP16 storage. Both history and candidate
        # vectors use exactly this FP32 readback normalization.
        vectors = torch.tensor(array, device=device, dtype=torch.float32)
        chunks.append(torch.nn.functional.normalize(vectors, dim=-1))
        position = record["end"]
    if position != len(ids) or not chunks:
        raise ValueError("completed index is missing item vectors")
    return ids, torch.cat(chunks, dim=0), report


def _stable_topk(scores, k):
    """Exact score ordering with smallest column ID winning boundary ties."""
    import torch
    k = min(k, scores.shape[1])
    threshold = torch.topk(scores, k, dim=1, sorted=False).values.min(dim=1).values[:, None]
    strict = scores > threshold
    ties = scores == threshold
    needed = k - strict.sum(dim=1, keepdim=True)
    keep = strict | (ties & (ties.cumsum(dim=1, dtype=torch.int32) <= needed))
    columns = torch.arange(scores.shape[1], device=scores.device)[None, :].expand_as(scores)
    choices = torch.topk(columns.masked_fill(~keep, scores.shape[1]), k,
                        dim=1, largest=False, sorted=True).values
    selected = scores.gather(1, choices)
    order = torch.argsort(selected, dim=1, descending=True, stable=True)
    return selected.gather(1, order), choices.gather(1, order)


def retrieve_topk(ids, vectors, histories, all_seen, *, k=200, catalog_chunk_size=65_536,
                  item_index=None, first_seen_times=None, cutoff_timestamps=None, allow_short=False):
    """Equal-weight visible history means; no ratings or target arguments exist."""
    import torch
    _positive(k, "retrieval k")
    _positive(catalog_chunk_size, "catalog chunk size")
    if len(histories) != len(all_seen) or not histories:
        raise ValueError("retrieval requires a nonempty aligned history/seen batch")
    if len(ids) != len(vectors):
        raise ValueError("retrieval vector rows must follow sorted unique item IDs")
    if (first_seen_times is None) != (cutoff_timestamps is None):
        raise ValueError("time filtering requires both first-seen times and query cutoffs")
    device = vectors.device
    observed, cutoffs = None, None
    if first_seen_times is not None:
        observed = torch.as_tensor(first_seen_times, device=device, dtype=torch.long)
        cutoffs = torch.as_tensor(cutoff_timestamps, device=device, dtype=torch.long)
        if observed.shape != (len(ids),) or cutoffs.shape != (len(histories),) or bool((observed <= 0).any()) or bool((cutoffs <= 0).any()):
            raise ValueError("time filters must align every catalog row and query with positive timestamps")
    if item_index is None:
        if ids != sorted(set(ids)):
            raise ValueError("retrieval vector rows must follow sorted unique item IDs")
        item_index = {item: position for position, item in enumerate(ids)}
    elif len(item_index) != len(ids):
        raise ValueError("reusable item index size differs from the vector catalog")
    history_indices, history_masks, seen_partition = [], [], {}
    for row, (history, seen) in enumerate(zip(histories, all_seen)):
        if not 1 <= len(history) <= 10 or any(item not in item_index for item in history):
            raise ValueError("retrieval history needs 1 through 10 catalog books")
        if not set(history) <= set(seen):
            raise ValueError("seen exclusion must contain the visible history")
        excluded = {item_index[item] for item in seen if item in item_index}
        if not allow_short and observed is None and len(ids) - len(excluded) < k:
            raise ValueError("catalog cannot supply the fixed number of unseen candidates")
        # Every row has the same reduction shape. A scatter/index_add would use
        # CUDA atomic additions for repeated row indices, whose order can change
        # on replay. A fixed ten-slot gather and row reduction has no such race.
        history_indices.append([item_index[item] for item in history] + [0] * (10 - len(history)))
        history_masks.append([True] * len(history) + [False] * (10 - len(history)))
        for position in excluded:
            partition = seen_partition.setdefault(position // catalog_chunk_size, ([], []))
            partition[0].append(row)
            partition[1].append(position % catalog_chunk_size)
    gathered = vectors[torch.tensor(history_indices, device=device)].float()
    present = torch.tensor(history_masks, device=device)
    queries = gathered.masked_fill(~present[..., None], 0).sum(dim=1, dtype=torch.float32)
    norms = torch.linalg.vector_norm(queries, dim=-1, keepdim=True)
    if not bool(torch.isfinite(queries).all()) or not bool((norms > 1e-12).all()):
        raise ValueError("equal-weight history mean has a nonfinite or zero norm")
    queries = queries / norms  # Sum normalization equals mean normalization.
    best_scores, best_ids = None, None
    for start in range(0, len(ids), catalog_chunk_size):
        end = min(start + catalog_chunk_size, len(ids))
        scores = queries @ vectors[start:end].float().T
        if observed is not None:
            # Eligibility belongs in the score matrix before local TopK. A
            # post-TopK filter would lose valid lower-scoring past products.
            scores.masked_fill_(observed[None, start:end] > cutoffs[:, None], -torch.inf)
        masked = seen_partition.get(start // catalog_chunk_size)
        if masked:
            scores[torch.tensor(masked[0], device=device), torch.tensor(masked[1], device=device)] = -torch.inf
        local_scores, local_ids = _stable_topk(scores, k)
        local_ids = local_ids + start
        if best_scores is None:
            best_scores, best_ids = local_scores, local_ids
        else:
            merged_scores = torch.cat((best_scores, local_scores), dim=1)
            merged_ids = torch.cat((best_ids, local_ids), dim=1)
            by_id = torch.argsort(merged_ids, dim=1, stable=True)
            merged_ids, merged_scores = merged_ids.gather(1, by_id), merged_scores.gather(1, by_id)
            by_score = torch.argsort(merged_scores, dim=1, descending=True, stable=True)[:, :k]
            best_scores, best_ids = merged_scores.gather(1, by_score), merged_ids.gather(1, by_score)
    if not allow_short and not bool(torch.isfinite(best_scores).all()):
        raise ValueError("retrieval could not produce the fixed unseen candidate count")
    return [[ids[position] for position, score in zip(row, values) if math.isfinite(score)]
            for row, values in zip(best_ids.cpu().tolist(), best_scores.cpu().tolist())]


def read_retrieval_inputs(path, items, *, domain="book"):
    """Reject target-bearing or sealed-test rows before creating embeddings."""
    seen_queries = set()
    fields = {"query_id", "user_id", "split", "history", "seen_item_ids", "history_hash"}
    _domain_schema(domain, "retrieval")
    if domain == "game":
        fields.add("cutoff_timestamp")
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if set(row) != fields:
                raise ValueError("retrieval inputs must be exactly target-free history schema")
            if row["split"] not in ("train", "dev") or user_split(row["user_id"]) != row["split"]:
                raise ValueError("retrieval inputs may contain original train/dev users only")
            if not isinstance(row["query_id"], str) or not row["query_id"] or row["query_id"] in seen_queries:
                raise ValueError("retrieval query IDs must be nonempty and unique")
            seen_queries.add(row["query_id"])
            if not isinstance(row["history"], list) or not 1 <= len(row["history"]) <= 10:
                raise ValueError("retrieval history must contain 1 through 10 books")
            history = tuple(HistoryEntry(**entry) for entry in row["history"])
            if [entry.timestamp for entry in history] != sorted(set(entry.timestamp for entry in history)):
                raise ValueError("retrieval history timestamps must be strictly increasing")
            if domain == "game" and (isinstance(row["cutoff_timestamp"], bool)
                    or not isinstance(row["cutoff_timestamp"], int)
                    or row["cutoff_timestamp"] <= history[-1].timestamp):
                raise ValueError("Games cutoff must follow the complete visible history")
            seen = row["seen_item_ids"]
            if (not isinstance(seen, list) or len(set(seen)) != len(seen)
                    or any(not isinstance(item, str) or not item for item in seen)
                    or not {entry.item_id for entry in history} <= set(seen)):
                raise ValueError("retrieval seen IDs must contain the complete visible history")
            if any(entry.item_id not in items for entry in history):
                raise ValueError("retrieval history book is absent from the catalog")
            content = {"history": [(entry.item_id, entry.rating, entry.timestamp, items[entry.item_id])
                                   for entry in history], "seen_item_ids": seen}
            if domain == "game":
                content.update(protocol="games-candidate-ranking-v1", cutoff_timestamp=row["cutoff_timestamp"])
            expected = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if row["history_hash"] != expected:
                raise ValueError("retrieval input history/content checksum mismatch")
            yield row


def _result_shard(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name, delete=False) as stream:
            temporary = Path(stream.name)
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def produce_retrieval(catalog, inputs, index, output, *, device="cuda", batch_size=128,
                      catalog_chunk_size=65_536, max_new_batches=None, domain="book", first_seen=None):
    import torch
    _positive(batch_size, "query batch size")
    _positive(catalog_chunk_size, "catalog chunk size")
    if max_new_batches is not None:
        _positive(max_new_batches, "max new batches")
    items = _catalog(catalog)
    ids, vectors, index_report = load_index(index, catalog, device=device, domain=domain)
    item_index = {item: position for position, item in enumerate(ids)}
    observed = None
    if domain == "game":
        if first_seen is None:
            raise ValueError("Games retrieval requires the frozen first-seen sidecar")
        times = json.loads(Path(first_seen).read_text())
        if (not isinstance(times, dict) or set(times) != set(items) or any(
                not isinstance(row, dict) or set(row) != {"timestamp", "source"}
                or isinstance(row["timestamp"], bool) or not isinstance(row["timestamp"], int) or row["timestamp"] <= 0
                or row["source"] != "amazon-reviews-2023-video-games-raw-0core" for row in times.values())):
            raise ValueError("first-seen sidecar must cover the frozen Games catalog with raw-0core provenance")
        observed = torch.tensor([times[item]["timestamp"] for item in ids], device=vectors.device, dtype=torch.long)
    elif first_seen is not None:
        raise ValueError("Books retrieval does not accept a Games time sidecar")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = {"catalog_sha256": file_hash(catalog), "inputs_sha256": file_hash(inputs),
                "index_manifest_sha256": file_hash(Path(index) / "index-manifest.json"),
                "code_sha256": file_hash(__file__), "batch_size": batch_size,
                "catalog_chunk_size": catalog_chunk_size, "device_type": torch.device(device).type,
                "query": "equal_weight_visible_books_l2_no_rating",
                "history_reduction": "fixed_10_slot_fp32_sum", "score": "fp32_inner_product_no_tf32",
                "storage_readback": "fp16_to_fp32_l2", "tie_break": "lexicographic_item_id",
                "train_top_k": 200, "dev_top_k": 100}
    if domain == "game":
        identity.update(domain=domain, schema="games-next-item-retrieval-v1", query="equal_weight_visible_games_l2_no_rating",
                        time_filter="first-observed<=cutoff", first_seen_sha256=file_hash(first_seen),
                        candidate_count="min(top_k,eligible_unseen_catalog)")
    state_path, complete = output / "retrieval-state.json", output / "retrieval-manifest.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("identity") != identity:
            raise ValueError("retrieval resume differs from frozen source/index/configuration/code")
    else:
        if any(output.iterdir()):
            raise ValueError("untracked retrieval output is not overwritten")
        state = {"identity": identity, "shards": [], "queries_written": 0, "complete": False}
        _atomic_json(state_path, state)
    cursor = 0
    for record in state["shards"]:
        if record["start"] != cursor or record["end"] <= cursor or file_hash(output / record["file"]) != record["sha256"]:
            raise ValueError("retrieval shard range or checksum mismatch")
        cursor = record["end"]
    if cursor != state["queries_written"]:
        raise ValueError("retrieval resume cursor differs from verified results")
    if complete.exists():
        report = json.loads(complete.read_text())
        if not state["complete"] or report.get("results_sha256") != file_hash(output / "retrieval.jsonl"):
            raise ValueError("completed retrieval checksum mismatch")
        return report
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    written = 0
    try:
        pending = []
        for ordinal, row in enumerate(read_retrieval_inputs(inputs, items, domain=domain)):
            if ordinal < cursor:
                continue
            pending.append(row)
            if len(pending) == batch_size:
                _write_retrieval_batch(output, state, pending, ids, vectors, catalog_chunk_size, item_index, first_seen_times=observed)
                pending = []
                written += 1
                if max_new_batches is not None and written == max_new_batches:
                    return state
        if pending:
            _write_retrieval_batch(output, state, pending, ids, vectors, catalog_chunk_size, item_index, first_seen_times=observed)
        if state["queries_written"] == 0:
            raise ValueError("target-free retrieval input is empty")
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    # Copy only immutable, verified batch results into the final JSONL file.
    temporary = output / ".retrieval.jsonl.tmp"
    with temporary.open("wb") as destination:
        for record in state["shards"]:
            with (output / record["file"]).open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    destination.write(block)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, output / "retrieval.jsonl")
    state["complete"] = True
    _atomic_json(state_path, state)
    report = {"schema": _domain_schema(domain, "retrieval"), "history_only": True,
              "inputs_sha256": identity["inputs_sha256"], "catalog_sha256": identity["catalog_sha256"],
              "results_sha256": file_hash(output / "retrieval.jsonl"), "queries": state["queries_written"],
              "retriever": {"model": index_report["identity"]["model"], "index": identity["index_manifest_sha256"],
                            "producer": identity, "index_identity": index_report["identity"]},
              "test_labels_accessed": False}
    if domain == "game":
        report.update(time_filter="first-observed<=cutoff", first_seen_sha256=identity["first_seen_sha256"])
    _atomic_json(complete, report)
    return report


def _write_retrieval_batch(output, state, rows, ids, vectors, chunk_size, item_index, *, first_seen_times=None):
    # Train/dev can share a batch; compute Top200 once, retain dev's Top100.
    k = 200 if any(row["split"] == "train" for row in rows) else 100
    retrieved = retrieve_topk(ids, vectors, [[entry["item_id"] for entry in row["history"]] for row in rows],
                              [row["seen_item_ids"] for row in rows], k=k, catalog_chunk_size=chunk_size,
                              item_index=item_index, first_seen_times=first_seen_times,
                              cutoff_timestamps=[row["cutoff_timestamp"] for row in rows] if first_seen_times is not None else None,
                              allow_short=first_seen_times is not None)
    results = [{"query_id": row["query_id"], "history_hash": row["history_hash"],
                "retrieved_ids": candidates[:200 if row["split"] == "train" else 100]}
               for row, candidates in zip(rows, retrieved)]
    start, end = state["queries_written"], state["queries_written"] + len(rows)
    filename = f"shards/queries-{start:09d}-{end:09d}.jsonl"
    destination = output / filename
    if destination.exists():
        existing = [json.loads(line) for line in destination.read_text().splitlines()]
        if existing != results:
            raise ValueError("untracked result shard differs from recomputed results; not overwritten")
    else:
        _result_shard(destination, results)
    state["shards"].append({"file": filename, "start": start, "end": end, "sha256": file_hash(destination)})
    state["queries_written"] = end
    _atomic_json(output / "retrieval-state.json", state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    encode = sub.add_parser("encode")
    encode.add_argument("--catalog", type=Path, required=True)
    encode.add_argument("--output", type=Path, required=True)
    encode.add_argument("--device", default="cuda")
    encode.add_argument("--compute-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    encode.add_argument("--batch-size", type=int, default=64)
    encode.add_argument("--shard-items", type=int, default=10000)
    encode.add_argument("--cache-dir", type=Path)
    encode.add_argument("--model-path", type=Path)
    encode.add_argument("--max-new-shards", type=int)
    encode.add_argument("--domain", choices=("book", "game"), default="book")
    retrieve = sub.add_parser("retrieve")
    retrieve.add_argument("--catalog", type=Path, required=True)
    retrieve.add_argument("--inputs", type=Path, required=True)
    retrieve.add_argument("--index", type=Path, required=True)
    retrieve.add_argument("--output", type=Path, required=True)
    retrieve.add_argument("--device", default="cuda")
    retrieve.add_argument("--batch-size", type=int, default=128)
    retrieve.add_argument("--catalog-chunk-size", type=int, default=65536)
    retrieve.add_argument("--max-new-batches", type=int)
    retrieve.add_argument("--domain", choices=("book", "game"), default="book")
    retrieve.add_argument("--first-seen", type=Path)
    args = parser.parse_args(argv)
    if args.phase == "encode":
        import torch
        from transformers import AutoModel, AutoTokenizer
        model_identity = dict(ENCODER_IDENTITY)
        compute_dtype = getattr(torch, args.compute_dtype)
        if args.model_path:
            model_identity["snapshot_manifest_sha256"] = verify_public_snapshot(
                args.model_path, repo=ENCODER_REPO, revision=ENCODER_REVISION)
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
            model = AutoModel.from_pretrained(args.model_path, local_files_only=True,
                                             torch_dtype=compute_dtype, attn_implementation="sdpa").to(args.device)
        else:
            tokenizer = AutoTokenizer.from_pretrained(ENCODER_REPO, revision=ENCODER_REVISION, cache_dir=args.cache_dir)
            model = AutoModel.from_pretrained(ENCODER_REPO, revision=ENCODER_REVISION, cache_dir=args.cache_dir,
                                             torch_dtype=compute_dtype, attn_implementation="sdpa").to(args.device)
        report = export_index(args.catalog, args.output, tokenizer=tokenizer, model=model, device=args.device,
                              batch_size=args.batch_size, shard_items=args.shard_items,
                              max_new_shards=args.max_new_shards, model_identity=model_identity, domain=args.domain,
                              compute_dtype=args.compute_dtype)
    else:
        report = produce_retrieval(args.catalog, args.inputs, args.index, args.output, device=args.device,
                                   batch_size=args.batch_size, catalog_chunk_size=args.catalog_chunk_size,
                                   max_new_batches=args.max_new_batches, domain=args.domain, first_seen=args.first_seen)
    print(json.dumps({"output": str(args.output), "complete": report.get("complete", "queries" in report),
                      "items": report.get("items"), "queries": report.get("queries", report.get("queries_written")),
                      "test_labels_accessed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
