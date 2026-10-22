"""Full-parameter shared-history recommendation training.

Prepared candidate groups are atomic softmax denominators; gradient
accumulation normalizes by query count. Training uses group CE or CE+KD,
selects checkpoints on fixed development users, and never opens test labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
import hashlib
import fcntl
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from typing import Mapping

import torch

from .next_item_data import SELECTOR_NAMESPACE, read_queries, user_split
from .next_item_metrics import ranking_metrics
from .next_item_model import GroupRanker, domain_template, query_input_hash


PROTOCOL = "books-next-item-training-v1"
CANDIDATE_PROTOCOL = "books-candidate-ranking-v1"
GAMES_CANDIDATE_PROTOCOL = "games-candidate-ranking-v1"
GAMES_TRAINING_PROTOCOL = "games-next-item-training-v1"
def config_domain(config):
    return "game" if config.get("schema_version") == GAMES_CANDIDATE_PROTOCOL else "book"


def training_protocol(identity):
    return GAMES_TRAINING_PROTOCOL if identity.get("domain") == "game" else PROTOCOL


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hex_digest(value, length=64):
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def validate_local_snapshot(spec):
    """A local path must prove the declared public revision, including weights."""
    if "local_path" not in spec:
        return
    root = Path(spec["local_path"])
    manifest = root / "snapshot-manifest.json"
    if _file_hash(manifest) != spec.get("snapshot_manifest_sha256"):
        raise ValueError("local model snapshot manifest hash mismatch")
    report = _read_bound_json(manifest, spec["snapshot_manifest_sha256"])
    if report.get("repo_id") != spec["repo"] or report.get("revision") != spec["revision"]:
        raise ValueError("local model snapshot public repository/revision mismatch")
    expected = report.get("files", {})
    actual = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in {".cache", "__pycache__"} for part in relative.parts) or path == manifest:
            continue
        if path.is_symlink():
            raise ValueError("local model snapshot must contain regular owned files")
        if path.is_file():
            actual[relative.as_posix()] = {"bytes": path.stat().st_size, "sha256": _file_hash(path)}
    if actual != expected or not any(name.endswith((".safetensors", ".bin")) for name in actual):
        raise ValueError("local model snapshot inventory/checksum mismatch")


def _positive(value, name, *, zero=False, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if (integer and not isinstance(value, int)) or value < 0 or (not zero and value == 0):
        raise ValueError(f"invalid {name}")


def validate_config(config: Mapping, *, formal=True, require_teacher_cache=True):
    if formal and config.get("schema_version") not in {"books-next-item-v1", CANDIDATE_PROTOCOL, GAMES_CANDIDATE_PROTOCOL}:
        raise ValueError("unsupported formal recommendation schema_version")
    for section in ("model", "data", "training", "loss", "evaluation", "checkpoint"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"missing configuration section {section}")
    model, data, train, loss = (config[key] for key in ("model", "data", "training", "loss"))
    domain = config_domain(config)
    if model.get("domain", domain) != domain:
        raise ValueError("model domain differs from the recommendation schema")
    if model.get("variant") not in {"T", "V1", "V3"}:
        raise ValueError("model variant must be T, V1 or V3")
    for key in (["b_model"] if model["variant"] == "T" else ["a_model", "b_model"]):
        spec = model.get(key)
        if not isinstance(spec, dict) or not isinstance(spec.get("repo"), str) or not spec["repo"]:
            raise ValueError(f"invalid {key}")
        revision = spec.get("revision")
        if formal and (not isinstance(revision, str) or len(revision) != 40
                       or any(c not in "0123456789abcdef" for c in revision)):
            raise ValueError(f"{key} requires an immutable 40-character model revision")
    if train.get("epochs") != 1:
        raise ValueError("this protocol freezes one epoch")
    for key in ("effective_query_batch", "microbatch_queries", "max_reader_packed_tokens",
                "max_encoder_packed_tokens", "max_logical_tokens"):
        _positive(train.get(key), key, integer=True)
    if train["microbatch_queries"] > train["effective_query_batch"]:
        raise ValueError("microbatch queries exceed effective query batch")
    for key in ("backbone_lr", "head_lr", "eps", "clip"):
        _positive(train.get(key), key)
    for key in ("interface_lr",):
        if key in train:
            _positive(train[key], key)
    if model.get("a_readout", "perceiver") not in {"perceiver", "last_token", "memory"}:
        raise ValueError("invalid A readout")
    if model.get("score_readout", "legacy") not in {"legacy", "candidate", "score_token"}:
        raise ValueError("invalid score readout")
    if not isinstance(model.get("shared_candidate_header", False), bool):
        raise ValueError("shared_candidate_header must be boolean")
    if not isinstance(model.get("input_assembly", "legacy"), str) or model.get("input_assembly", "legacy") not in {"legacy", "batched"}:
        raise ValueError("input_assembly must be legacy or batched")
    if model.get("a_readout") in {"last_token", "memory"} and model.get("m") != 1:
        raise ValueError("last-token/MEM readout requires m=1")
    if model.get("score_readout") == "candidate" and model.get("m") != 1:
        raise ValueError("candidate readout requires m=1")
    if model["variant"] == "T" and (model.get("a_readout", "perceiver") != "perceiver"
            or model.get("score_readout", "legacy") != "legacy" or model.get("shared_candidate_header", False)):
        raise ValueError("teacher retains the original text template")
    for key in ("weight_decay", "warmup_ratio", "min_lr_ratio"):
        _positive(train.get(key), key, zero=True)
    if not 0 <= train["warmup_ratio"] < 1 or not 0 <= train["min_lr_ratio"] <= 1:
        raise ValueError("invalid learning-rate schedule ratios")
    betas = train.get("betas")
    if not isinstance(betas, list) or len(betas) != 2 or any(
            isinstance(b, bool) or not isinstance(b, (int, float)) or not 0 <= b < 1 for b in betas):
        raise ValueError("Adam betas must be two numbers in [0,1)")
    if train.get("compute_dtype") not in {"bfloat16", "float32"}:
        raise ValueError("only BF16 compute or FP32 reference is supported")
    if train.get("attention_backend") not in {"flex_attention", "sdpa"}:
        raise ValueError("unsupported attention backend")
    if not isinstance(train.get("gradient_checkpointing"), bool):
        raise ValueError("gradient_checkpointing must be explicit")
    for key in ("reader_gradient_checkpointing", "optimizer_fused", "teacher_cache_on_device", "cuda_event_timing", "pretokenize_items"):
        if key in train and not isinstance(train[key], bool):
            raise ValueError(f"{key} must be boolean when specified")
    if train.get("pretokenize_items", False) and model["variant"] != "V3":
        raise ValueError("pretokenize_items currently requires V3 canonical item inputs")
    if formal and train["attention_backend"] != "flex_attention":
        raise ValueError("formal runs require sparse FlexAttention packing")
    if loss.get("recipe") not in {"ce", "ce_kd"}:
        raise ValueError("only group CE and CE+KD are supported; full/raw/alignment are absent")
    for key in ("alignment_weight", "raw_ce_weight"):
        if key not in loss or isinstance(loss[key], bool) or loss[key] != 0:
            raise ValueError(f"{key} must explicitly be zero")
    _positive(loss.get("temperature"), "temperature")
    for key in ("beta_final", "beta_warmup_ratio"):
        _positive(loss.get(key), key, zero=True)
        if loss[key] > 1:
            raise ValueError(f"{key} must lie in [0,1]")
    if model["variant"] == "T" and loss["recipe"] != "ce":
        raise ValueError("T trains group CE, never KD")
    if require_teacher_cache and loss["recipe"] == "ce_kd" and not loss.get("teacher_cache"):
        raise ValueError("CE+KD requires an explicit scalar teacher cache path")
    if train.get("teacher_cache_on_device", False) and loss["recipe"] != "ce_kd":
        raise ValueError("teacher_cache_on_device requires CE+KD")
    if formal and require_teacher_cache and loss["recipe"] == "ce_kd":
        if not isinstance(loss.get("teacher_identity"), dict) or not _hex_digest(loss.get("teacher_cache_sha256")):
            raise ValueError("formal KD requires frozen teacher identity and cache hash; use --teacher-cache")
    for key in ("train_users", "dev_users", "selection_users", "test_users", "query_cap", "history_max"):
        _positive(data.get(key), key, integer=True)
    if data["history_max"] > 10 or data["selection_users"] > data["dev_users"]:
        raise ValueError("invalid history/cohort selection limits")
    if config.get("schema_version") in {CANDIDATE_PROTOCOL, GAMES_CANDIDATE_PROTOCOL}:
        if (data.get("train_candidates") not in (8, 100) or isinstance(data.get("train_candidates"), bool)
                or data.get("eval_candidates") != 100 or data.get("candidate_policy") != "random-unobserved"):
            raise ValueError("candidate ranking requires train_candidates=8 or 100, eval_candidates=100 and random-unobserved")
        if data.get("train_policy", "random-unobserved") not in {"mixed-history", "random-unobserved"}:
            raise ValueError("unsupported training candidate policy")
        if data.get("train_policy") == "mixed-history" and data["train_candidates"] != 8:
            raise ValueError("mixed-history training uses eight candidates")
    if formal and domain == "game":
        caps = {"train_users": 50000, "dev_users": 5000, "selection_users": 2000, "test_users": 10000}
        if any(data[key] > cap for key, cap in caps.items()):
            raise ValueError("Games actual cohorts cannot exceed the frozen 50K/5K/2K/10K caps")
        if (data.get("train_candidates") != 8 or data.get("train_policy") != "mixed-history"
                or data.get("history_max") != 10 or data.get("query_cap") != 5
                or train.get("effective_query_batch") != 32):
            raise ValueError("Games freezes mixed8 groups, history 10, query cap 5 and query batch 32")
        if model["variant"] == "V1":
            raise ValueError("the Games transfer uses T and V3")
    if formal and domain == "book" and {key: data[key] for key in ("train_users", "dev_users", "selection_users", "test_users")} != {
            "train_users": 50000, "dev_users": 5000, "selection_users": 2000, "test_users": 10000}:
        raise ValueError("formal first-round cohort is frozen at 50K/5K/2K/10K")
    evaluate, checkpoint = config["evaluation"], config["checkpoint"]
    inference_compute_dtype(config)
    scoring_fields = {"scoring_backend", "candidate_chunk_size", "input_assembly"}
    if scoring_fields & evaluate.keys():
        tree_bf16 = (evaluate.get("scoring_backend") == "tree"
                     and evaluate.get("input_assembly") == "legacy"
                     and "candidate_chunk_size" not in evaluate and model["variant"] == "T"
                     and model.get("input_assembly", "legacy") == "legacy"
                     and inference_compute_dtype(config) == "bfloat16")
        if not tree_bf16 and (not scoring_fields <= evaluate.keys() or evaluate["scoring_backend"] != "prefix-kv"
                or type(evaluate["candidate_chunk_size"]) is not int or evaluate["candidate_chunk_size"] != 8
                or evaluate["input_assembly"] != "batched" or model["variant"] != "T"
                or inference_compute_dtype(config) != "float32"):
            raise ValueError("evaluation prefix-kv requires T, FP32, batched assembly and candidate_chunk_size=8; "
                             "tree requires T, BF16 and legacy assembly without a candidate chunk override")
    for key in ("selection_every", "monitor_every", "monitor_users", "max_reader_packed_tokens"):
        _positive(evaluate.get(key), key, integer=True)
    if "microbatch_queries" in evaluate:
        _positive(evaluate["microbatch_queries"], "evaluation.microbatch_queries", integer=True)
    if evaluate["monitor_users"] > data["selection_users"] or evaluate.get("full_epoch_end") is not True:
        raise ValueError("fixed monitor must fit selector; full epoch-end reporting is required")
    for key in ("first_step", "every"):
        _positive(checkpoint.get(key), key, integer=True)
    if isinstance(config.get("seed"), bool) or not isinstance(config.get("seed"), int):
        raise ValueError("seed must be an integer")
    if isinstance(config.get("candidate_seed", config["seed"]), bool) or not isinstance(config.get("candidate_seed", config["seed"]), int):
        raise ValueError("candidate_seed must be an integer independent of model seed")
    return config


def fixed_selector(dev_users, count, *, namespace=SELECTOR_NAMESPACE):
    if isinstance(count, bool) or not isinstance(count, int) or not 0 < count <= len(dev_users) or len(set(dev_users)) != len(dev_users):
        raise ValueError("invalid fixed selector count/cohort")
    return tuple(sorted(dev_users, key=lambda user: (hashlib.sha256((namespace + user).encode()).hexdigest(), user))[:count])


def _read_bound_json(path, expected_sha256):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("parsed JSON bytes differ from frozen input hash")
    return json.loads(raw)


def _load_catalog(path, expected_sha256=None):
    path = Path(path)
    if path.suffix == ".jsonl":
        items = {}
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for line in stream:
                digest.update(line)
                row = json.loads(line)
                if set(row) != {"item_id", "text"} or row["item_id"] in items:
                    raise ValueError("catalog JSONL requires unique item_id/text rows")
                items[row["item_id"]] = row["text"]
        if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
            raise ValueError("parsed catalog bytes differ from frozen input hash")
    else:
        items = _read_bound_json(path, expected_sha256) if expected_sha256 else json.loads(path.read_text())
    if not isinstance(items, dict) or not items or any(
            not isinstance(key, str) or not key or not isinstance(text, str) or not text.strip()
            for key, text in items.items()):
        raise ValueError("catalog must map unique item identities to nonempty text")
    return items


def load_dataset(data_dir, config, *, formal=True):
    """Open train/dev only. Test cohort IDs are declarations, not label access."""
    if config_domain(config) == "game":
        from .games_data import load_dataset as load_games_dataset
        return load_games_dataset(data_dir, config, formal=formal)
    data_dir = Path(data_dir)
    candidate_ranking = config.get("schema_version") == CANDIDATE_PROTOCOL
    names = {"train": "train.jsonl", "dev": "dev.jsonl", "cohorts": "cohorts.json"}
    catalog_path = data_dir / "catalog.json"
    if not catalog_path.exists():
        catalog_path = data_dir / "catalog.jsonl"
    initial_hashes = {name: _file_hash(data_dir / filename) for name, filename in names.items()}
    initial_hashes["catalog"] = _file_hash(catalog_path)
    initial_manifest_hash = _file_hash(data_dir / "preparation-manifest.json")
    cohorts = _read_bound_json(data_dir / names["cohorts"], initial_hashes["cohorts"])
    for split in ("train", "dev", "test"):
        users = cohorts.get(split)
        if not isinstance(users, list) or len(users) != config["data"][split + "_users"] or len(set(users)) != len(users):
            raise ValueError(f"{split} declared cohort size/identity mismatch")
        if any(user_split(user) != split for user in users):
            raise ValueError(f"{split} does not preserve the original user boundary")
    expected_selector = fixed_selector(cohorts["dev"], config["data"]["selection_users"])
    if tuple(cohorts.get("checkpoint_selector", ())) != expected_selector:
        raise ValueError("checkpoint selector differs from the fixed hash-defined dev subset")
    train = read_queries(data_dir / names["train"], split="train", expected_sha256=initial_hashes["train"])
    preparation = _read_bound_json(data_dir / "preparation-manifest.json", initial_manifest_hash)
    unavailable = preparation.get("unavailable_dev_users", [])
    if not isinstance(unavailable, list) or len(set(unavailable)) != len(unavailable) or not set(unavailable) <= set(cohorts["dev"]):
        raise ValueError("invalid explicit unavailable dev user declaration")
    dev_path = data_dir / names["dev"]
    # An entirely unavailable fixed dev cohort is legal but scores exactly zero.
    dev = () if not dev_path.stat().st_size and len(unavailable) == len(cohorts["dev"]) else read_queries(dev_path, split="dev", expected_sha256=initial_hashes["dev"])
    available_dev = set(q.user_id for q in dev)
    if set(q.user_id for q in train) != set(cohorts["train"]) or available_dev | set(unavailable) != set(cohorts["dev"]):
        raise ValueError("query files must cover the complete declared train/dev cohorts")
    if available_dev & set(unavailable):
        raise ValueError("unavailable dev users cannot have scorable queries")
    if len(dev) != len(cohorts["dev"]) - len(unavailable):
        raise ValueError("dev must contain exactly one chronological query per user")
    if max(Counter(q.user_id for q in train).values()) > config["data"]["query_cap"]:
        raise ValueError("training exceeds frozen per-user query cap")
    if any(len(q.history) > config["data"]["history_max"] for q in (*train, *dev)):
        raise ValueError("history exceeds frozen limit")
    if candidate_ranking and (any(len(q.candidate_ids) != config["data"]["train_candidates"] for q in train)
                              or any(len(q.candidate_ids) != config["data"]["eval_candidates"] for q in dev)):
        raise ValueError("candidate ranking train/dev group sizes differ from configuration")
    if candidate_ranking and any(q.positive_index is None for q in (*train, *dev)):
        raise ValueError("candidate ranking requires the target in every train/dev group")
    if formal and not candidate_ranking and (any(len(q.candidate_ids) != 8 for q in train) or any(len(q.candidate_ids) != 100 for q in dev)):
        raise ValueError("formal train/dev groups require exactly 8/100 candidates")
    items = _load_catalog(catalog_path, initial_hashes["catalog"])
    for query in (*train, *dev):
        required = {event.item_id for event in query.history} | set(query.candidate_ids) | {query.target_item_id}
        if not required <= items.keys():
            raise ValueError("query catalog is incomplete; metadata cannot silently delete examples")
    files = {name: _file_hash(data_dir / filename) for name, filename in names.items()}
    files["catalog"] = _file_hash(catalog_path)
    if files != initial_hashes:
        raise ValueError("prepared data changed while reading; parsed inputs cannot claim later hashes")
    if formal or candidate_ranking:
        schemas = {"books-candidate-ranking-preparation-v1", "books-candidate-ranking-preparation-v2"} if candidate_ranking else {"books-next-item-preparation-v1"}
        if preparation.get("schema") not in schemas:
            raise ValueError("formal training requires the frozen preparation manifest schema")
        expected_files = {"train.jsonl": files["train"], "dev.jsonl": files["dev"],
                          "cohorts.json": files["cohorts"], catalog_path.name: files["catalog"]}
        if not isinstance(preparation.get("files"), dict) or any(
                preparation["files"].get(name) != digest for name, digest in expected_files.items()):
            raise ValueError("prepared input file hashes differ from the frozen preparation manifest")
        if not isinstance(preparation.get("unavailable_dev_reasons"), dict) or set(
                preparation["unavailable_dev_reasons"]) != set(unavailable):
            raise ValueError("every unavailable dev user requires an explicit preparation reason")
        if (preparation.get("test_labels_accessed") is not False or preparation.get("query_cap") != config["data"]["query_cap"]
                or preparation.get("train_candidates") != (config["data"]["train_candidates"] if candidate_ranking else 8)
                or preparation.get("dev_candidates") != 100):
            raise ValueError("prepared task/candidate/query-cap/sealed-test contract differs from training")
        sources = preparation.get("sources")
        if not isinstance(sources, dict) or not {"ratings_sha256", "catalog_sha256"} <= sources.keys() or any(
                not _hex_digest(value) for value in sources.values()):
            raise ValueError("preparation requires complete frozen source hashes")
        truncation = preparation.get("catalog_truncation")
        tokenizer = truncation.get("tokenizer", {}) if isinstance(truncation, dict) else {}
        if (not isinstance(tokenizer, dict) or not isinstance(tokenizer.get("repo"), str) or not tokenizer["repo"]
                or not _hex_digest(tokenizer.get("revision"), 40) or truncation.get("max_item_tokens") != 512):
            raise ValueError("catalog truncation requires an immutable tokenizer identity and max_item_tokens=512")
        if candidate_ranking:
            from .candidate_data_validation import validate_candidate_preparation
            validate_candidate_preparation(preparation, config, files, train, dev)
        else:
            retrieval = preparation.get("retrieval")
            if (not isinstance(retrieval, dict) or retrieval.get("schema") != "books-next-item-retrieval-v1"
                or retrieval.get("history_only") is not True or retrieval.get("catalog_sha256") != files["catalog"]
                or not all(_hex_digest(retrieval.get(key)) for key in ("inputs_sha256", "catalog_sha256", "results_sha256"))
                or not isinstance(retrieval.get("retriever"), dict) or not retrieval["retriever"]
                or not all(_hex_digest(preparation.get(key)) for key in ("context_manifest_sha256", "retrieval_manifest_sha256"))):
                raise ValueError("prepared retrieval requires the complete frozen history-only provenance contract")
    files["preparation_manifest"] = _file_hash(data_dir / "preparation-manifest.json")
    if files["preparation_manifest"] != initial_manifest_hash:
        raise ValueError("preparation manifest changed while reading")
    cohorts["unavailable_dev_users"] = unavailable
    return train, dev, items, cohorts, files


def bind_teacher_cache(config, cache_path):
    """Bind a completed cache before validation and checkpoint identity hashing.

    This changes a copy of a template configuration. A previously bound identity
    or digest is immutable, so a substituted cache cannot silently resume a run.
    Row coverage and semantic hashes are checked by load_scalar_teacher_cache.
    """
    config = json.loads(json.dumps(config))
    if config.get("loss", {}).get("recipe") != "ce_kd":
        raise ValueError("--teacher-cache applies only to a CE+KD student configuration")
    cache_path = Path(cache_path).expanduser().resolve()
    if cache_path.is_dir():
        cache_path = cache_path / "teacher-cache.json"
    identity_path = cache_path.parent / "teacher_identity.json"
    completion_path = cache_path.with_name(cache_path.name + ".manifest.json")
    if not cache_path.is_file() or not identity_path.is_file() or not completion_path.is_file():
        raise ValueError("teacher cache requires teacher_identity.json and a completed cache manifest")
    teacher = json.loads(identity_path.read_text())
    completion = json.loads(completion_path.read_text())
    expected_schema = ("games" if config_domain(config) == "game" else "books") + "-candidate-teacher-cache-completion-v1"
    digest = _file_hash(cache_path)
    if (not isinstance(teacher, dict) or not _hex_digest(teacher.get("checkpoint_sha256"))
            or completion.get("schema") != expected_schema or completion.get("completed") is not True
            or completion.get("teacher") != teacher
            or completion.get("file") != {"sha256": digest, "bytes": cache_path.stat().st_size}):
        raise ValueError("teacher cache completion identity or file checksum mismatch")
    loss = config["loss"]
    if loss.get("teacher_identity") is not None and loss["teacher_identity"] != teacher:
        raise ValueError("teacher identity differs from the already-bound configuration")
    if loss.get("teacher_cache_sha256") is not None and loss["teacher_cache_sha256"] != digest:
        raise ValueError("teacher cache SHA256 differs from the already-bound configuration")
    loss.update(teacher_cache=str(cache_path), teacher_identity=teacher, teacher_cache_sha256=digest)
    return config


def load_scalar_teacher_cache(path, queries, items, *, expected_teacher=None, domain="book"):
    """Strict new-task scalar logits; old class probabilities are never accepted."""
    path = Path(path)
    expected_cache_hash = _file_hash(path)
    # Completed generated caches store one query per line inside the original
    # JSON object schema. Read them incrementally rather than retaining a second
    # catalog-sized dict of candidate strings and scores alongside the dataset.
    with path.open(encoding="utf-8") as stream:
        first = stream.readline()
    line_format = '"format":"query-lines-v1"' in first
    if line_format:
        suffix = ',"queries":{\n'
        if not first.endswith(suffix):
            raise ValueError("teacher cache query-lines header is malformed")
        cache = json.loads(first[:-len(suffix)] + "}")
        completion = json.loads(path.with_name(path.name + ".manifest.json").read_text()) if path.with_name(path.name + ".manifest.json").is_file() else {}
        if (cache.get("complete") is not True or completion.get("schema") != ("games" if domain == "game" else "books") + "-candidate-teacher-cache-completion-v1"
                or completion.get("completed") is not True
                or completion.get("teacher") != cache.get("teacher")
                or completion.get("queries") != len(queries)
                or completion.get("file", {}).get("sha256") != _file_hash(path)
                or completion.get("file", {}).get("bytes") != path.stat().st_size):
            raise ValueError("teacher cache is incomplete or its completion identity/checksum differs")
        records = None
    else:
        cache = _read_bound_json(path, expected_cache_hash)
    if cache.get("complete") is False or cache.get("completed") is False:
        raise ValueError("teacher cache is incomplete")
    if cache.get("schema") != ("games" if domain == "game" else "books") + "-next-item-scalar-teacher-v1":
        raise ValueError("teacher cache must be new-task scalar scores, not five-class probabilities")
    if not isinstance(cache.get("teacher"), dict) or not cache["teacher"].get("checkpoint_sha256"):
        raise ValueError("teacher cache requires a frozen teacher checkpoint identity")
    if domain == "game" and (cache["teacher"].get("domain") != "game"
            or cache["teacher"].get("source_schema") != GAMES_CANDIDATE_PROTOCOL
            or cache["teacher"].get("train_candidates") != 8
            or cache["teacher"].get("train_policy") != "mixed-history"
            or cache["teacher"].get("prompt_template") != {**domain_template(domain), "sha256": _digest(domain_template(domain))}):
        raise ValueError("Games KD requires the actual same-domain mixed8 teacher and frozen prompt")
    if expected_teacher is not None and cache["teacher"] != expected_teacher:
        raise ValueError("teacher identity differs from configuration")
    if not line_format:
        records = cache.get("queries")
    if not line_format and (not isinstance(records, dict) or set(records) != {q.query_id for q in queries}):
        raise ValueError("teacher cache must exactly cover the prepared training queries")
    result = {}
    def entries():
        if not line_format:
            yield from ((q, records[q.query_id]) for q in queries)
            return
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            header = stream.readline()
            digest.update(header)
            if header.decode("utf-8") != first:
                raise ValueError("teacher cache header changed while reading")
            for query in queries:
                raw = stream.readline()
                digest.update(raw)
                line = raw.decode("utf-8").strip().removesuffix(",")
                value = json.loads("{" + line + "}")
                if list(value) != [query.query_id]:
                    raise ValueError("teacher cache query sequence differs from prepared training queries")
                yield query, value[query.query_id]
            tail = stream.read()
            digest.update(tail)
            if tail.strip() != b"}}":
                raise ValueError("teacher cache must exactly cover the prepared training queries")
        if digest.hexdigest() != expected_cache_hash:
            raise ValueError("parsed teacher cache bytes differ from frozen cache hash")
    for query, row in entries():
        if not isinstance(row, dict) or row.get("input_hash") != query_input_hash(query, items, domain=domain):
            raise ValueError("teacher cache semantic/candidate input hash mismatch")
        if row.get("candidate_ids") != list(query.candidate_ids):
            raise ValueError("teacher candidate order differs from training group")
        scores = row.get("scores")
        if not isinstance(scores, list) or len(scores) != len(query.candidate_ids) or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in scores):
            raise ValueError("teacher scores must be finite scalars for every candidate")
        result[query.query_id] = scores
    if _file_hash(path) != expected_cache_hash:
        raise ValueError("teacher cache changed while reading")
    return result, {"teacher": cache["teacher"], "cache_sha256": expected_cache_hash}


def _autocast(device, dtype):
    return torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16) if dtype == "bfloat16" else nullcontext()


def inference_compute_dtype(config):
    """Preserve the historical training-dtype default unless evaluation opts in."""
    dtype = config["evaluation"].get("compute_dtype", config["training"]["compute_dtype"])
    if not isinstance(dtype, str) or dtype not in {"bfloat16", "float32"}:
        raise ValueError("evaluation compute_dtype must be bfloat16 or float32")
    return dtype


def _prefix_evaluation_metadata(config):
    if config["evaluation"].get("scoring_backend") != "prefix-kv":
        return {}
    return {"scoring_backend": "prefix-kv", "attention_backend": "sdpa", "compute_dtype": "float32",
            "tf32": False, "input_assembly": "batched", "candidate_chunk_size": 8}


def _evaluation_execution_metadata(config):
    """Only explicitly selected evaluator profiles extend legacy metrics."""
    if config["evaluation"].get("scoring_backend") == "tree":
        return {"scoring_backend": "tree", "attention_backend": config["training"]["attention_backend"],
                "compute_dtype": inference_compute_dtype(config), "input_assembly": "legacy", "tf32": False}
    return _prefix_evaluation_metadata(config)


def _evaluation_source_files(config):
    """Bind the optional evaluator without changing the training inventory."""
    if not _prefix_evaluation_metadata(config):
        return {}
    path = Path(__file__).with_name("next_item_teacher_kv.py")
    before = path.stat()
    if path.is_symlink() or not path.is_file() or not 0 < before.st_size <= 16 * 1024**2:
        raise ValueError("evaluation scorer must be a regular thin source file")
    record = {"sha256": _file_hash(path), "bytes": before.st_size}
    after = path.stat()
    if any(getattr(before, key) != getattr(after, key)
           for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")):
        raise ValueError("evaluation scorer source changed while hashing")
    return {path.name: record}


class _TeacherPrefixEvaluation:
    """One shared-weight reader across assessments, with query-local KV only."""
    def __init__(self, model):
        self.model, self.scorer = model, None

    def reset(self):
        # Rebuild after loading the selected checkpoint, including its buffers.
        self.scorer = None

    def __call__(self, queries, items):
        if self.scorer is None:
            from .next_item_teacher_kv import TeacherPrefixKVScorer
            self.scorer = TeacherPrefixKVScorer(self.model, candidate_chunk_size=8, input_assembly="batched")
        output = self.scorer(queries, items)
        expected = {"attention_backend": "sdpa", "compute_dtype": "float32", "tf32": False,
                    "input_assembly": "batched", "candidate_chunk_size": 8}
        statistics = output.get("statistics", {})
        if any(type(statistics.get(name)) is not type(value) or statistics[name] != value
               for name, value in expected.items()):
            raise ValueError("teacher prefix evaluation returned different execution metadata")
        return output


@contextmanager
def inference_context(device, dtype):
    """Use explicit inference precision; restore the caller's CUDA math flags."""
    if not isinstance(dtype, str) or dtype not in {"bfloat16", "float32"}:
        raise ValueError("inference compute_dtype must be bfloat16 or float32")
    device_type = torch.device(device).type
    strict_cuda = device_type == "cuda" and dtype == "float32"
    if strict_cuda:
        matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        cudnn_tf32 = torch.backends.cudnn.allow_tf32
        matmul_precision = torch.get_float32_matmul_precision()
    try:
        if strict_cuda:
            # Normalize the matmul precision before changing TF32 flags. On
            # PyTorch 2.12, disabling only allow_tf32 while the precision is
            # "high"/"medium" leaves conflicting backend settings.
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        context = torch.autocast(device_type=device_type, enabled=False) if dtype == "float32" else _autocast(device, dtype)
        with context:
            yield
    finally:
        if strict_cuda:
            torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
            torch.backends.cudnn.allow_tf32 = cudnn_tf32
            # Restoring only allow_tf32 would turn a prior "medium" setting
            # into "high"; keep the caller's exact matmul precision as well.
            torch.set_float32_matmul_precision(matmul_precision)


class DeviceTeacherScores:
    """Immutable query-to-row map with a single dense score upload at startup.

    Row views use Python integer indexing, so shuffled microbatches create no
    per-query H2D copy. Padding is storage only and never enters a denominator.
    The original cache loader already verifies teacher/input provenance; this
    second boundary verifies exact query IDs, candidate order and finite rows.
    """
    def __init__(self, scores, queries, *, device):
        if not queries or len({q.query_id for q in queries}) != len(queries):
            raise ValueError("teacher staging requires unique nonempty query IDs")
        if set(scores) != {q.query_id for q in queries}:
            raise ValueError("teacher staging must exactly cover the fixed query mapping")
        width = max(len(q.candidate_ids) for q in queries)
        rows, self._mapping = [], {}
        for index, query in enumerate(queries):
            row = scores[query.query_id]
            if (not isinstance(row, (list, tuple)) or len(row) != len(query.candidate_ids)
                    or len(row) < 2 or len(set(query.candidate_ids)) != len(query.candidate_ids)
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) for v in row)):
                raise ValueError("teacher staging scores must match each finite candidate row")
            self._mapping[query.query_id] = (index, tuple(query.candidate_ids))
            rows.append([*row, *([0.0] * (width - len(row)))])
        host = torch.tensor(rows, dtype=torch.float32)
        # Reject otherwise finite Python scalars that overflow the FP32 cache.
        if not torch.isfinite(host).all():
            raise ValueError("teacher staging scores overflow FP32")
        self.rows = host.to(device=device)

    def for_batch(self, queries):
        result = []
        for query in queries:
            try:
                row, candidates = self._mapping[query.query_id]
            except KeyError as error:
                raise ValueError("teacher staging query is outside the fixed mapping") from error
            if tuple(query.candidate_ids) != candidates:
                raise ValueError("teacher staging candidate order differs from the fixed mapping")
            result.append(self.rows[row, :len(candidates)])
        return result


class _PhaseTimer:
    """Wall latency and optional CUDA event timeline, never GPU busy time.

    Events deliberately synchronize at the end only when requested. Their
    elapsed interval includes CPU launch gaps and is not a kernel-only sum.
    """
    def __init__(self, device, *, cuda_events=False):
        self.started = time.perf_counter()
        self.events = None
        if cuda_events and torch.device(device).type == "cuda":
            self.events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            self.stream = torch.cuda.current_stream(device)
            self.events[0].record(self.stream)

    def finish(self):
        result = {}
        if self.events is not None:
            self.events[1].record(self.stream)
            self.events[1].synchronize()
            result["cuda_event_elapsed_seconds"] = self.events[0].elapsed_time(self.events[1]) / 1000
        result["wall_seconds"] = time.perf_counter() - self.started
        return result


def build_optimizer(model, config):
    train = config["training"]
    head = list(model.head.parameters())
    head_ids = {id(parameter) for parameter in head}
    backbone = [(name, parameter) for name, parameter in model.named_parameters() if id(parameter) not in head_ids]
    if not backbone or any(not parameter.requires_grad or parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("protocol requires all parameters trainable and stored FP32")
    decay, no_decay, interface_decay, interface_no_decay = [], [], [], []
    for name, parameter in backbone:
        # Norms, biases, scalar bridge gains and other vectors do not receive
        # decay. Only backbone matrix/tensor weights receive the protocol wd.
        excluded = parameter.ndim < 2 or name.rsplit(".", 1)[-1] == "bias" or "norm" in name.lower()
        interface = "interface_lr" in train and name.split(".", 1)[0] in {
            "resampler", "projection", "bridge", "memory_token", "score_token"}
        target = (interface_no_decay if excluded else interface_decay) if interface else (no_decay if excluded else decay)
        target.append(parameter)
    groups = [{"params": decay, "lr": train["backbone_lr"], "base_lr": train["backbone_lr"],
               "weight_decay": train["weight_decay"], "name": "backbone_decay"},
              {"params": head, "lr": train["head_lr"], "base_lr": train["head_lr"],
               "weight_decay": 0.0, "name": "head"}]
    if no_decay:
        groups.append({"params": no_decay, "lr": train["backbone_lr"], "base_lr": train["backbone_lr"],
                       "weight_decay": 0.0, "name": "backbone_no_decay"})
    for params, name, weight_decay in ((interface_decay, "interface_decay", train["weight_decay"]),
                                      (interface_no_decay, "interface_no_decay", 0.0)):
        if params:
            groups.append({"params": params, "lr": train["interface_lr"], "base_lr": train["interface_lr"],
                           "weight_decay": weight_decay, "name": name})
    options = {}
    if "optimizer_fused" in train:
        if not isinstance(train["optimizer_fused"], bool):
            raise ValueError("optimizer_fused must be boolean when specified")
        options["fused"] = train["optimizer_fused"]
    return torch.optim.AdamW(groups, betas=tuple(train["betas"]), eps=train["eps"],
                             weight_decay=train["weight_decay"], **options)


def lr_factor(step, total_steps, warmup_ratio, min_lr_ratio):
    warmup = min(total_steps, math.ceil(total_steps * warmup_ratio))
    if warmup and step <= warmup:
        return step / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return min_lr_ratio + (1 - min_lr_ratio) * (1 + math.cos(math.pi * min(1, max(0, progress)))) / 2


def beta_for_step(step, total_steps, loss_config):
    if loss_config["recipe"] == "ce":
        return 0.0
    ramp = max(1, math.ceil(total_steps * loss_config["beta_warmup_ratio"]))
    return loss_config["beta_final"] * min(1, step / ramp)


def train_update(model, queries, items, optimizer, *, microbatch_queries, clip,
                 device, compute_dtype, teacher_scores=None, beta=0.0, temperature=1.0, diagnostics=False,
                 defer_scalar_metrics=False):
    """Every query has weight 1/N, including uneven microbatches and epoch tails."""
    if not queries or microbatch_queries < 1:
        raise ValueError("training update needs a nonempty query batch")
    if not isinstance(defer_scalar_metrics, bool):
        raise ValueError("defer_scalar_metrics must be boolean")
    optimizer.zero_grad(set_to_none=True)
    probes = module_probes(model) if diagnostics else {}
    before = {name: parameter.detach().flatten()[:4096].clone() for name, parameter in probes.items()}
    ce, kd, tokens, finite = 0.0, 0.0, 0, None
    counts = {key: 0 for key in ("encoder_tokens", "encoder_unique_items", "reader_packs")}
    for start in range(0, len(queries), microbatch_queries):
        batch = queries[start:start + microbatch_queries]
        teacher = (teacher_scores.for_batch(batch) if isinstance(teacher_scores, DeviceTeacherScores)
                   else None if teacher_scores is None else [teacher_scores[q.query_id] for q in batch])
        with _autocast(device, compute_dtype):
            options = {"defer_scalar_metrics": True} if defer_scalar_metrics else {}
            output = model.loss(batch, items, teacher_scores=teacher, beta=beta, temperature=temperature, **options)
        if defer_scalar_metrics:
            for key in ("ce", "kd", "loss_finite"):
                value = output[key]
                if not torch.is_tensor(value) or value.ndim != 0 or value.device != output["loss"].device:
                    raise ValueError("deferred loss metrics must be scalar device tensors")
            if output["loss_finite"].dtype != torch.bool or output["ce"].requires_grad or output["kd"].requires_grad:
                raise ValueError("deferred CE/KD must be detached and loss_finite must be boolean")
            valid = output["loss_finite"] & torch.isfinite(output["loss"]) & torch.isfinite(output["ce"]) & torch.isfinite(output["kd"])
            if diagnostics:
                valid = valid & torch.stack([torch.isfinite(scores).all() for scores in output["scores"]]).all()
            finite = valid if finite is None else finite & valid
        else:
            if not torch.isfinite(output["loss"]):
                raise FloatingPointError("nonfinite recommendation loss; update was not applied")
            if diagnostics and any(not torch.isfinite(scores).all() for scores in output["scores"]):
                raise FloatingPointError("nonfinite candidate scores")
        weight = len(batch) / len(queries)
        (output["loss"] * weight).backward()
        ce += output["ce"] * weight
        kd += output["kd"] * weight
        statistics = output.get("statistics", {})
        tokens += statistics.get("reader_tokens", 0)
        for key in counts:
            counts[key] += statistics.get(key, 0)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=not defer_scalar_metrics)
    if defer_scalar_metrics:
        # One transfer/host inspection for the entire effective batch. Clipping
        # has not updated any parameter; an invalid microbatch cannot reach Adam.
        valid = finite & torch.isfinite(norm)
        ce, kd, norm_value, valid_value = torch.stack((ce, kd, norm.detach(), valid.to(norm.dtype))).cpu().tolist()
        if not valid_value or not all(math.isfinite(v) for v in (ce, kd, norm_value)):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("nonfinite recommendation loss or gradients; update was not applied")
    else:
        norm_value = norm
    learning = {}
    if diagnostics:
        for name, parameter in probes.items():
            grad = parameter.grad
            learning[name] = {"probe_parameter": parameter.numel(), "gradient_norm": float(grad.norm()) if grad is not None else 0.0,
                              "gradient_present": grad is not None}
    optimizer.step()
    if diagnostics:
        for name, parameter in probes.items():
            learning[name]["sample_max_delta"] = float((parameter.detach().flatten()[:4096] - before[name]).abs().max())
    result = {"ce": ce, "kd": kd, "loss": (1-beta)*ce + beta*kd,
            "queries": len(queries), "candidate_scores": sum(len(q.candidate_ids) for q in queries),
            "reader_tokens": tokens, "grad_norm": float(norm_value), "beta": beta, **counts}
    # This is a sum of within-microbatch unique counts, not global update dedup.
    result["encoder_unique_items_scope"] = "sum_of_microbatch_unique_items"
    if diagnostics:
        result["module_learning"] = learning
        result.update(finite_scores=True, finite_gradients=True,
                      adam_state_initialized=bool(optimizer.state))
    return result


def module_probes(model):
    """One matrix per learned module; cheap gradient/change evidence, not a full copy."""
    groups, names = {}, {}
    for name, parameter in model.named_parameters():
        module = name.split(".", 1)[0]
        if module not in groups or ("embed" in names[module] and "embed" not in name and parameter.ndim >= 2):
            groups[module] = parameter
            names[module] = name
    return groups


def backbone_parameter_counts(model):
    """Report both roles while allowing the raw teacher to have no encoder."""
    return {role: sum(parameter.numel() for parameter in module.parameters()) if module is not None else 0
            for role in ("a", "b") for module in (getattr(model, role),)}


def module_initialization_hashes(model):
    """Hash every initial parameter once, in bounded host chunks."""
    digests = {}
    for name, parameter in model.named_parameters():
        group = name.split(".", 1)[0]
        digest = digests.setdefault(group, hashlib.sha256())
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        flat = parameter.detach().flatten()
        for start in range(0, flat.numel(), 1024 * 1024):
            digest.update(flat[start:start + 1024 * 1024].float().cpu().numpy().tobytes())
    return {name: digest.hexdigest() for name, digest in digests.items()}


@torch.no_grad()
def evaluate(model, queries, items, *, expected_users, device, compute_dtype,
             max_reader_packed_tokens, microbatch_queries=1, unavailable_users=(),
             evaluation_scope=None, prediction_sink=None, prefix_scorer=None,
             progress_sink=None, progress_every_batches=25, progress_every_seconds=30.,
             strict_math=False):
    _positive(microbatch_queries, "evaluation.microbatch_queries", integer=True)
    if not isinstance(strict_math, bool):
        raise ValueError("evaluation strict_math must be a boolean")
    if progress_sink is not None:
        if not callable(progress_sink):
            raise ValueError("evaluation progress_sink must be callable")
        _positive(progress_every_batches, "evaluation.progress_every_batches", integer=True)
        _positive(progress_every_seconds, "evaluation.progress_every_seconds")
    if prefix_scorer is not None and (compute_dtype != "float32" or prefix_scorer.model is not model):
        raise ValueError("teacher prefix evaluation requires its own model and FP32")
    if prefix_scorer is not None and not queries:
        raise ValueError("teacher prefix evaluation requires at least one query")
    was_training = model.training
    modes = {module: module.training for module in model.modules()} if prefix_scorer is not None else None
    model.eval()
    rows, tokens, execution = [], 0, None
    batches = 0
    started = last_progress = time.perf_counter() if progress_sink is not None else None
    last_progress_batch = 0

    def progress(phase, now):
        if progress_sink is not None:
            progress_sink({"phase": phase, "done_queries": len(rows), "total_queries": len(queries),
                           "done_batches": batches, "total_batches": math.ceil(len(queries) / microbatch_queries),
                           "reader_tokens": tokens, "wall_seconds": max(0., now - started)})

    try:
        progress("start", started)
        for start in range(0, len(queries), microbatch_queries):
            batch = queries[start:start + microbatch_queries]
            with (inference_context(device, "float32") if strict_math else nullcontext()), \
                    inference_context(device, compute_dtype):
                output = (model(batch, items, max_reader_packed_tokens=max_reader_packed_tokens)
                          if prefix_scorer is None else prefix_scorer(batch, items))
            if prefix_scorer is not None:
                statistics = output["statistics"]
                actual = {"scoring_backend": "prefix-kv", **{name: statistics[name] for name in (
                    "attention_backend", "compute_dtype", "tf32", "input_assembly", "candidate_chunk_size")}}
                if execution is not None and actual != execution:
                    raise ValueError("teacher prefix execution changed between evaluation batches")
                execution = actual
            if len(output["scores"]) != len(batch):
                raise ValueError("model omitted an evaluation query")
            for query, score in zip(batch, output["scores"]):
                values = score.detach().float().cpu().tolist()
                rows.append((query, values))
                if prediction_sink is not None:
                    order = sorted(range(len(values)), key=lambda i: (-values[i], query.candidate_ids[i]))
                    rank = order.index(query.positive_index) + 1 if query.positive_index is not None else None
                    prediction_sink({"query_id": query.query_id, "user_id": query.user_id,
                                     "target_item_id": query.target_item_id, "candidate_ids": list(query.candidate_ids),
                                     "scores": values, "rank": rank})
            tokens += output.get("statistics", {}).get("reader_tokens", 0)
            batches += 1
            if progress_sink is not None:
                now = time.perf_counter()
                if (batches == 1 or batches - last_progress_batch >= progress_every_batches
                        or now - last_progress >= progress_every_seconds):
                    progress("progress", now)
                    last_progress, last_progress_batch = now, batches
    finally:
        model.train(was_training)
        if modes is not None:
            for module, training in modes.items():
                module.training = training
    metrics = ranking_metrics(rows, expected_user_ids=expected_users,
                              unavailable_user_ids=unavailable_users, k=10)
    metrics["reader_tokens"] = tokens
    metrics["hr@1"] = ranking_metrics(rows, expected_user_ids=expected_users,
                                     unavailable_user_ids=unavailable_users, k=1)["hr@1"]
    if evaluation_scope in {CANDIDATE_PROTOCOL, GAMES_CANDIDATE_PROTOCOL}:
        metrics["candidate_target_coverage"] = metrics.pop("retrieval_recall")
        metrics["evaluation_scope"] = "target-guaranteed-candidate-ranking"
    if execution is not None:
        metrics.update(execution)
    elif strict_math:
        metrics.update(scoring_backend="tree", attention_backend=model.attention_backend,
                       input_assembly=model.input_assembly, compute_dtype=compute_dtype, tf32=False)
    progress("complete", time.perf_counter() if progress_sink is not None else None)
    return metrics


def _capacity_evaluation(model, optimizer, dev, config, assessment):
    """Probe one full configured eval batch with resident Adam and gradients.

    Capacity receipts without an explicit evaluation override keep their
    historical one-query scope. This does not change normal dev assessment.
    """
    microbatch = config["evaluation"].get("microbatch_queries", 1)
    _positive(microbatch, "evaluation.microbatch_queries", integer=True)
    sample = list(dev[:microbatch])
    if len(sample) != microbatch:
        raise ValueError("capacity evaluation lacks the configured number of dev queries")
    if (len({q.query_id for q in sample}) != microbatch
            or len({q.user_id for q in sample}) != microbatch
            or any(len(q.candidate_ids) != 100 or len(set(q.candidate_ids)) != 100 for q in sample)):
        raise ValueError("capacity evaluation requires distinct dev queries/users with 100 unique candidates")
    states = optimizer.state
    adam_buffers = [(parameter, state, state.get("exp_avg"), state.get("exp_avg_sq"))
                    for parameter, state in states.items()]
    if not isinstance(optimizer, torch.optim.AdamW) or not adam_buffers or any(
            not torch.is_tensor(first) or not torch.is_tensor(second)
            or first.numel() == 0 or second.shape != first.shape for _, _, first, second in adam_buffers):
        raise ValueError("capacity evaluation requires initialized resident Adam state")
    gradients = [(parameter, parameter.grad) for parameter in model.parameters() if parameter.grad is not None]
    if not gradients:
        raise ValueError("capacity evaluation must retain the last update's gradient buffers")
    measured = assessment(sample, [q.user_id for q in sample], "capacity-dev")
    # evaluate() checks every raw score for matching length and finiteness.
    if measured.get("queries") != microbatch or not math.isfinite(measured["ndcg@10"]):
        raise ValueError("capacity evaluation must produce finite scores for the whole configured batch")
    if (optimizer.state is not states or len(states) != len(adam_buffers)
            or any(states.get(parameter) is not state or state.get("exp_avg") is not first
                   or state.get("exp_avg_sq") is not second for parameter, state, first, second in adam_buffers)
            or any(parameter.grad is not gradient for parameter, gradient in gradients)):
        raise ValueError("capacity evaluation must retain Adam state and the last update's gradients")
    expected_execution = _evaluation_execution_metadata(config)
    actual_execution = {name: measured.get(name) for name in expected_execution}
    if _digest(actual_execution) != _digest(expected_execution):
        raise ValueError("capacity evaluation lacks matching actual scoring metadata")
    return {"candidates": 100, "queries": len(sample), "microbatch_queries": microbatch,
            "compute_dtype": inference_compute_dtype(config),
            "candidates_per_query": 100, "finite_scores": True,
            "optimizer_state_initialized": True, "optimizer_state_entries": len(adam_buffers),
            "gradients_retained": True, "retained_gradient_buffers": len(gradients),
            "retained_gradient_bytes": sum(gradient.numel() * gradient.element_size() for _, gradient in gradients),
            **actual_execution}


def _atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _log(output, event):
    event = {"time_unix": time.time(), **event}
    with (Path(output) / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    _atomic_json(Path(output) / "status.json", event)


def _log_evaluation_progress(output, event):
    """Observable progress is best effort durability; checkpoints remain durable.

    Avoid a disk barrier for every slow evaluation batch and leave status.json
    as the existing training/selection event. The regular event log still lets
    readers retain the progress records without buffering the complete log.
    """
    event = {"time_unix": time.time(), "event": "evaluation_progress", **event}
    root = Path(output)
    with (root / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
    target = root / "evaluation-progress.json"
    temporary = target.with_name("." + target.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(event, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, target)


def save_checkpoint(output, model, optimizer, state, identity):
    output = Path(output)
    root = output / "checkpoints"
    root.mkdir(exist_ok=True)
    name = f"step-{state['step']:08d}"
    destination, temporary = root / name, root / ("." + name + ".tmp")
    state_hash = _digest(state)
    if destination.exists():
        manifest = _checkpoint_manifest(destination, identity=identity, expected_step=state["step"])
        if manifest.get("state_sha256") != state_hash:
            raise ValueError("checkpoint step is immutable; existing state differs and cannot be overwritten")
        _atomic_json(output / "latest_checkpoint.json", {"path": str(destination), **manifest})
        return destination
    if temporary.exists():
        raise RuntimeError("unfinished checkpoint directory exists; inspect it before recovery")
    temporary.mkdir()
    payload = {"protocol": training_protocol(identity), "identity": identity, "state": state,
               "model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "rng_python": random.getstate(), "rng_torch": torch.get_rng_state(),
               "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    try:
        torch.save(payload, temporary / "training.pt")
        with (temporary / "training.pt").open("rb") as stream:
            os.fsync(stream.fileno())
        manifest = {"protocol": training_protocol(identity), "step": state["step"], "query_cursor": state["cursor"],
                    "identity": identity, "state_sha256": state_hash,
                    "files": {"training.pt": {"sha256": _file_hash(temporary / "training.pt"),
                                                                      "bytes": (temporary / "training.pt").stat().st_size}},
                    "includes": ["model", "optimizer", "cursor", "rng", "best_selection_state"],
                    "remote_backup_verified": False}
        _atomic_json(temporary / "manifest.json", manifest)
        if destination.exists():
            raise RuntimeError("checkpoint destination appeared during save; no checkpoint was overwritten")
        os.rename(temporary, destination)
        _atomic_json(output / "latest_checkpoint.json", {"path": str(destination), **manifest})
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def _checkpoint_manifest(path, *, identity, expected_step=None):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("identity") != identity or manifest.get("protocol") != training_protocol(identity):
        raise ValueError("resume configuration/data/source identity differs from checkpoint")
    if expected_step is not None and manifest.get("step") != expected_step:
        raise ValueError("required best checkpoint step differs from its manifest")
    if not _hex_digest(manifest.get("state_sha256")):
        raise ValueError("checkpoint requires a frozen state signature")
    record = manifest["files"]["training.pt"]
    if _file_hash(path / "training.pt") != record["sha256"] or (path / "training.pt").stat().st_size != record["bytes"]:
        raise ValueError("resume checkpoint checksum/size mismatch")
    return manifest


def _verify_best_checkpoint(path, *, identity, state):
    """Verify payload metadata without copying model/Adam tensors into RAM."""
    manifest = _checkpoint_manifest(path, identity=identity, expected_step=state["best_step"])
    payload = torch.load(Path(path) / "training.pt", map_location="cpu", weights_only=True, mmap=True)
    best = payload.get("state", {})
    if _digest(best) != manifest["state_sha256"]:
        raise ValueError("required best checkpoint state signature mismatch")
    if payload.get("identity") != identity or payload.get("protocol") != training_protocol(identity):
        raise ValueError("required best checkpoint payload identity mismatch")
    if (best.get("step") != state["best_step"] or best.get("cursor") != manifest.get("query_cursor")
            or best.get("best_step") != state["best_step"] or best.get("best_ndcg") != state["best_ndcg"]
            or best.get("best_checkpoint") != state["best_checkpoint"]):
        raise ValueError("required best checkpoint selection metadata differs from resumed state")


def restore_required_best(resume, output, state, *, identity):
    """Bring the inherited best into a moved run before any new update.

    Latest and best can be different checkpoints. A standalone latest is not
    enough for final best reporting: restore the original best too, or fail now.
    Existing destinations are validated and never replaced by this operation.
    """
    pointer, step, score = (state.get(key) for key in ("best_checkpoint", "best_step", "best_ndcg"))
    if pointer is None:
        if step is not None or score is not None:
            raise ValueError("resume best selection state is incomplete")
        return
    if (isinstance(step, bool) or not isinstance(step, int) or not 1 <= step <= state["step"]
            or isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
            or pointer != f"checkpoints/step-{step:08d}"):
        raise ValueError("resume best checkpoint path/step/selection state is invalid")
    destination = Path(output) / pointer
    if destination.exists():
        _verify_best_checkpoint(destination, identity=identity, state=state)
        return
    resume = Path(resume).resolve()
    source = resume.parent.parent / pointer if resume.parent.name == "checkpoints" else None
    if source is None or not source.is_dir():
        raise ValueError(f"resume requires the original best checkpoint {pointer}; restore it together with latest before training")
    _verify_best_checkpoint(source, identity=identity, state=state)
    destination.parent.mkdir(exist_ok=True)
    temporary = destination.with_name(".restore-" + destination.name + ".tmp")
    if temporary.exists():
        raise RuntimeError("unfinished best checkpoint restore exists; inspect it before recovery")
    temporary.mkdir()
    try:
        for name in ("training.pt", "manifest.json"):
            shutil.copyfile(source / name, temporary / name)
            with (temporary / name).open("rb") as stream:
                os.fsync(stream.fileno())
        _verify_best_checkpoint(temporary, identity=identity, state=state)
        if destination.exists():
            raise RuntimeError("best destination appeared during restore; no checkpoint was overwritten")
        # The run's writer lock excludes competing training saves. Renaming a
        # fully verified directory publishes both files together.
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    _atomic_json(Path(output) / "best_checkpoint.json", {"path": str(destination), "step": step,
                                                         "selector_ndcg@10": score,
                                                         "restored_from": str(source)})


def load_checkpoint(path, model, optimizer, *, identity, device):
    path = Path(path)
    manifest = _checkpoint_manifest(path, identity=identity)
    payload = torch.load(path / "training.pt", map_location=device, weights_only=True)
    if payload.get("identity") != identity or payload.get("protocol") != training_protocol(identity):
        raise ValueError("checkpoint payload identity mismatch")
    if payload["state"].get("step") != manifest.get("step") or payload["state"].get("cursor") != manifest.get("query_cursor"):
        raise ValueError("checkpoint state step/cursor differs from its manifest")
    if _digest(payload["state"]) != manifest["state_sha256"]:
        raise ValueError("checkpoint payload state signature differs from its manifest")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    random.setstate(payload["rng_python"])
    torch.set_rng_state(payload["rng_torch"].cpu())
    if payload["rng_cuda"]:
        if not torch.cuda.is_available() or len(payload["rng_cuda"]) != torch.cuda.device_count():
            raise ValueError("resume requires the same CUDA RNG device count")
        torch.cuda.set_rng_state_all([value.cpu() for value in payload["rng_cuda"]])
    return payload["state"]






def _validate_output_identity(output, *, resume, identity):
    """Do not append a resumed experiment to an unrelated or unverified run."""
    output = Path(output)
    if not output.exists() or not any(path.name != "run.lock" for path in output.iterdir()):
        return
    if not resume:
        raise ValueError("fresh training output must be empty; use --resume for an existing run")
    try:
        manifest = json.loads((output / "run_manifest.json").read_text())
    except (OSError, ValueError) as error:
        raise ValueError("resume output requires a readable run manifest with the same experiment identity") from error
    recorded_identity = manifest.get("identity") if isinstance(manifest, dict) else None
    if isinstance(recorded_identity, dict) and "student_template" not in identity:
        recorded_identity = {key: value for key, value in recorded_identity.items() if key != "student_template"}
    if (not isinstance(manifest, dict) or manifest.get("protocol") != training_protocol(identity)
            or recorded_identity != identity):
        raise ValueError("resume output experiment identity differs from configuration/data/source")
    # Identity alone is insufficient if the accompanying experiment record is
    # damaged or was copied without its actual configuration.
    if not isinstance(manifest.get("config"), dict) or _digest(manifest["config"]) != identity["config_sha256"]:
        raise ValueError("resume output manifest configuration differs from its experiment identity")




def _run_training(config, data_dir, output, *, device="cuda", resume=None, model_factory=None,
                  formal=True, max_updates=None, acceptance_only=False, stop_request_file=None,
                  expected_training_identity=None):
    """Run one frozen epoch; factory/max_updates support offline integration tests.

    The public CLI always sets formal=True and offers no reduced-cohort or
    toy-model option. Full-dev reports never choose or replace a checkpoint.
    """
    validate_config(config, formal=formal)
    if formal and torch.device(device).type != "cuda":
        raise ValueError("formal packed training requires a CUDA device")
    if acceptance_only and (not formal or model_factory is not None or max_updates is None or max_updates > 2):
        raise ValueError("capacity acceptance requires the actual formal models and at most two measured updates")
    train, dev, items, cohorts, data_hashes = load_dataset(data_dir, config, formal=formal)
    source_names = ("train_next_item.py", "next_item_metrics.py", "next_item_model.py",
                    "next_item_data.py", "ranking_packing.py", "resampler.py")
    if config.get("schema_version") == CANDIDATE_PROTOCOL:
        source_names += ("prepare_candidate_ranking.py", "candidate_teacher.py", "candidate_data_validation.py")
    domain = config_domain(config)
    if domain == "game":
        source_names += ("games_data.py", "build_games_assets.py", "candidate_teacher.py")
    identity = {"config_sha256": _digest(config), "data_sha256": data_hashes,
                "source_sha256": {name: _file_hash(Path(__file__).parent / name) for name in source_names}}
    evaluation_sources = _evaluation_source_files(config)
    if evaluation_sources:
        identity["evaluation_source_files"] = evaluation_sources
    if domain == "game":
        identity.update(domain=domain, schema_version=GAMES_CANDIDATE_PROTOCOL,
                        prompt_template={**domain_template(domain), "sha256": _digest(domain_template(domain))})
    teacher_scores = None
    if config["loss"]["recipe"] == "ce_kd":
        teacher_scores, teacher_identity = load_scalar_teacher_cache(
            config["loss"]["teacher_cache"], train, items, expected_teacher=config["loss"].get("teacher_identity"), domain=domain)
        expected_hash = config["loss"].get("teacher_cache_sha256")
        if expected_hash is not None and teacher_identity["cache_sha256"] != expected_hash:
            raise ValueError("teacher cache SHA256 differs from the frozen configuration")
        identity["teacher"] = teacher_identity
    if expected_training_identity is not None:
        # These are the existing verified loader results: do not rescan the
        # large inputs/cache. Only model-derived student metadata is deferred;
        # every other declared field, including unknown fields, must match.
        if (not isinstance(expected_training_identity, dict)
                or identity != {key: value for key, value in expected_training_identity.items()
                                if key != "student_template"}):
            raise ValueError("actual training identity differs from expected qualification identity")
    output = Path(output)
    _validate_output_identity(output, resume=resume, identity=identity)
    output.mkdir(parents=True, exist_ok=True)
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    spec, train_config = config["model"], config["training"]
    evaluation_dtype = inference_compute_dtype(config)
    if model_factory is None:
        a_spec = spec.get("a_model", {})
        for local_spec in ([spec["b_model"]] if spec["variant"] == "T" else [a_spec, spec["b_model"]]):
            validate_local_snapshot(local_spec)
        model = GroupRanker.from_pretrained(a_spec.get("local_path", a_spec.get("repo")),
                                            spec["b_model"].get("local_path", spec["b_model"]["repo"]),
                                            a_revision=a_spec.get("revision"), b_revision=spec["b_model"]["revision"],
                                            variant=spec["variant"], device=device, domain=domain,
                                            attention_backend=train_config["attention_backend"],
                                            m=spec.get("m", 8), resampler_layers=spec.get("resampler_layers", 2),
                                            resampler_heads=spec.get("resampler_heads", 8),
                                            a_readout=spec.get("a_readout", "perceiver"),
                                            shared_candidate_header=spec.get("shared_candidate_header", False),
                                            score_readout=spec.get("score_readout", "legacy"),
                                            max_reader_packed_tokens=train_config["max_reader_packed_tokens"],
                                            max_encoder_packed_tokens=train_config["max_encoder_packed_tokens"],
                                            max_logical_tokens=train_config["max_logical_tokens"],
                                            gradient_checkpointing=train_config["gradient_checkpointing"],
                                            reader_gradient_checkpointing=train_config.get("reader_gradient_checkpointing"),
                                            input_assembly=spec.get("input_assembly", "legacy"))
    else:
        model = model_factory(config, device)
    if domain == "game" and getattr(model, "domain", domain) != domain:
        raise ValueError("model factory returned a ranker for another domain")
    if expected_training_identity is not None:
        if spec["variant"] != "T" and hasattr(model, "metadata"):
            identity["student_template"] = model.metadata()["student_template"]
        if identity != expected_training_identity:
            raise ValueError("actual training identity differs from expected qualification identity")
    optimizer = build_optimizer(model, config)
    if spec["variant"] != "T" and hasattr(model, "metadata") and "student_template" not in identity:
        identity["student_template"] = model.metadata()["student_template"]
    _validate_output_identity(output, resume=resume, identity=identity)
    state = {"step": 0, "cursor": 0, "best_step": None, "best_ndcg": None, "best_checkpoint": None}
    if resume:
        state = load_checkpoint(resume, model, optimizer, identity=identity, device=device)
    elif spec["variant"] != "T":
        model.calibrate_bridge(train, items, max_items=256)
    prefix_evaluation = _TeacherPrefixEvaluation(model) if evaluation_sources else None
    order = list(range(len(train)))
    random.Random(config["seed"]).shuffle(order)
    effective = train_config["effective_query_batch"]
    total_steps = math.ceil(len(train) / effective)
    if state["cursor"] != min(state["step"] * effective, len(train)) or not 0 <= state["step"] <= total_steps:
        raise ValueError("resume cursor/step does not match the frozen epoch")
    if resume:
        restore_required_best(resume, output, state, identity=identity)
    if train_config.get("teacher_cache_on_device", False):
        timer = _PhaseTimer(device)
        teacher_scores = DeviceTeacherScores(teacher_scores, train, device=device)
        _log(output, {"event": "teacher_cache_staged", "rows": len(train),
                      "score_dtype": "float32", "storage_shape": list(teacher_scores.rows.shape),
                      "device": str(teacher_scores.rows.device), **timer.finish()})
    if train_config.get("pretokenize_items", False):
        timer = _PhaseTimer(device)
        token_stats = model.prepare_item_tokens(items)
        _log(output, {"event": "item_tokens_prepared", "items": len(items),
                      "statistics": token_stats, **timer.finish()})
    selector_users = tuple(cohorts["checkpoint_selector"])
    monitor_users = fixed_selector(selector_users, min(config["evaluation"]["monitor_users"], len(selector_users)),
                                   namespace="games-next-item-monitor-v1|" if domain == "game" else "books-next-item-monitor-v1|")
    selector_set, monitor_set = set(selector_users), set(monitor_users)
    unavailable_set = set(cohorts["unavailable_dev_users"])
    selector_queries = [query for query in dev if query.user_id in selector_set]
    monitor_queries = [query for query in dev if query.user_id in monitor_set]
    _atomic_json(output / "resolved-config.json", config)
    _atomic_json(output / "run_manifest.json", {"protocol": training_protocol(identity), "identity": identity, "config": config,
                                               "train_queries": len(train), "total_updates": total_steps,
                                               "selector_users": list(selector_users), "monitor_users": list(monitor_users),
                                               "test_labels_accessed": False,
                                               "acceptance_only": acceptance_only,
                                               "stop_request_file": str(stop_request_file) if stop_request_file else None})

    def assessment(queries, users, prediction_name=None, checkpoint_step=None, *, phase=None):
        timer = _PhaseTimer(device, cuda_events=train_config.get("cuda_event_timing", False))
        evaluation_microbatch = config["evaluation"].get("microbatch_queries", train_config["microbatch_queries"])
        prediction_path = output / "predictions" / (prediction_name + ".jsonl") if prediction_name else None
        temporary = prediction_path.with_suffix(".jsonl.tmp") if prediction_path else None
        if prediction_path:
            prediction_path.parent.mkdir(exist_ok=True)
        stream = temporary.open("w") if temporary else None
        def sink(row):
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        def progress_sink(row):
            _log_evaluation_progress(output, {"step": state["step"] if checkpoint_step is None else checkpoint_step,
                "assessment": phase or prediction_name or "evaluation", "compute_dtype": evaluation_dtype,
                "microbatch_queries": evaluation_microbatch, **row})
        try:
            evaluator_options = {"prefix_scorer": prefix_evaluation} if prefix_evaluation is not None else {}
            if config["evaluation"].get("scoring_backend") == "tree":
                evaluator_options["strict_math"] = True
            metrics = evaluate(model, queries, items, expected_users=users, device=device,
                        compute_dtype=evaluation_dtype,
                        max_reader_packed_tokens=config["evaluation"]["max_reader_packed_tokens"],
                        microbatch_queries=evaluation_microbatch,
                        unavailable_users=tuple(user for user in users if user in unavailable_set),
                        evaluation_scope=config.get("schema_version"), prediction_sink=sink if stream else None,
                        progress_sink=progress_sink,
                        **evaluator_options)
            expected_execution = _evaluation_execution_metadata(config)
            execution = {name: metrics.get(name) for name in expected_execution}
            if _digest(execution) != _digest(expected_execution):
                raise ValueError("assessment execution differs from the selected evaluation profile")
            if stream:
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            if stream:
                stream.close()
        if prediction_path:
            os.replace(temporary, prediction_path)
            _atomic_json(prediction_path.with_suffix(".manifest.json"), {
                "schema": "games-next-item-predictions-v1" if domain == "game" else "books-next-item-predictions-v1", "identity_sha256": _digest(identity),
                "step": state["step"] if checkpoint_step is None else checkpoint_step,
                "candidate_seed": config.get("candidate_seed", config["seed"]),
                "compute_dtype": evaluation_dtype, "attention_backend": train_config["attention_backend"],
                "evaluation_microbatch_queries": evaluation_microbatch,
                "expected_users": list(users), "rows": len(queries), "metrics": metrics,
                "file": {"sha256": _file_hash(prediction_path), "bytes": prediction_path.stat().st_size},
                "test_labels_accessed": False, **execution})
        _log(output, {"event": "evaluation_timing", "step": state["step"],
                      "prediction_name": prediction_name, "queries": len(queries),
                      "microbatch_queries": evaluation_microbatch, **timer.finish(),
                      **execution})
        return metrics

    def measured_save():
        timer = _PhaseTimer(device)
        checkpoint = save_checkpoint(output, model, optimizer, state, identity)
        _log(output, {"event": "checkpoint_save", "step": state["step"],
                      "checkpoint": str(checkpoint), **timer.finish()})
        return checkpoint

    last_saved_checkpoint, last_saved_step = None, None

    def stop_if_requested():
        """No signal handlers, partial microbatch checkpoint, or automatic retry.

        The file is deliberately left in place. An explicit resume requires
        removing the request or choosing a new request path first.
        """
        nonlocal last_saved_checkpoint, last_saved_step
        if stop_request_file is None or not Path(stop_request_file).exists():
            return None
        checkpoint = last_saved_checkpoint if last_saved_step == state["step"] else measured_save()
        last_saved_checkpoint, last_saved_step = checkpoint, state["step"]
        report = {"protocol": training_protocol(identity), "identity": identity, "state": state, "completed": False,
                  "status": "interrupted", "interruption_reason": "stop_request",
                  "optimizer_boundary": True, "checkpoint": str(checkpoint),
                  "stop_request_file": str(stop_request_file), "test_labels_accessed": False}
        _atomic_json(output / "report.json", report)
        _log(output, {"event": "interrupted", "status": "interrupted", "step": state["step"],
                      "cursor": state["cursor"], "optimizer_boundary": True,
                      "checkpoint": str(checkpoint), "interruption_reason": "stop_request"})
        return report

    if not resume:
        _atomic_json(output / "initialization.json", {"schema": "games-next-item-initialization-v1" if domain == "game" else "books-next-item-initialization-v1",
            "identity_sha256": _digest(identity), "seed": config["seed"], "module_sha256": module_initialization_hashes(model)})
    stopped = stop_if_requested()
    if stopped is not None:
        return stopped
    if not resume:
        if not acceptance_only:
            _log(output, {"event": "step0_monitor", "step": 0,
                          "metrics": assessment(monitor_queries, monitor_users, "step0-monitor"), "selects_checkpoint": False})

    model.train()
    updates_this_call = 0
    while state["cursor"] < len(train):
        stopped = stop_if_requested()
        if stopped is not None:
            return stopped
        if max_updates is not None and updates_this_call >= max_updates:
            checkpoint = last_saved_checkpoint if last_saved_step == state["step"] else measured_save()
            _log(output, {"event": "interrupted_for_test", **state, "checkpoint": str(checkpoint)})
            result = {"state": state, "completed": False, "checkpoint": str(checkpoint)}
            if acceptance_only:
                evaluation = _capacity_evaluation(model, optimizer, dev, config, assessment)
                result.update(model_parameters=backbone_parameter_counts(model),
                              peak_memory_bytes=torch.cuda.max_memory_allocated(device),
                              evaluation=evaluation,
                              resume_verified=resume is not None)
            return result
        step = state["step"] + 1
        batch = [train[i] for i in order[state["cursor"]:state["cursor"] + effective]]
        factor = lr_factor(step, total_steps, train_config["warmup_ratio"], train_config["min_lr_ratio"])
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * factor
        beta = beta_for_step(step, total_steps, config["loss"])
        timer = _PhaseTimer(device, cuda_events=train_config.get("cuda_event_timing", False))
        update_options = {"defer_scalar_metrics": True} if spec.get("input_assembly", "legacy") == "batched" else {}
        metrics = train_update(model, batch, items, optimizer, microbatch_queries=train_config["microbatch_queries"],
                               clip=train_config["clip"], device=device, compute_dtype=train_config["compute_dtype"],
                               teacher_scores=teacher_scores, beta=beta, temperature=config["loss"]["temperature"],
                               diagnostics=step <= 2, **update_options)
        state.update(step=step, cursor=state["cursor"] + len(batch))
        updates_this_call += 1
        timing = timer.finish()
        _log(output, {"event": "train", **state, **metrics, "seconds": timing["wall_seconds"], **timing,
                      "backbone_lr": optimizer.param_groups[0]["lr"], "head_lr": optimizer.param_groups[1]["lr"]})
        if step % config["evaluation"]["monitor_every"] == 0:
            _log(output, {"event": "monitor", "step": step, "metrics": assessment(monitor_queries, monitor_users, phase="monitor"),
                          "selects_checkpoint": False})
        selection_due = step % config["evaluation"]["selection_every"] == 0 or state["cursor"] == len(train)
        improved = False
        if selection_due:
            scores = assessment(selector_queries, selector_users, phase="selection")
            improved = state["best_ndcg"] is None or scores["ndcg@10"] > state["best_ndcg"]
            if improved:
                state.update(best_step=step, best_ndcg=scores["ndcg@10"],
                             best_checkpoint=f"checkpoints/step-{step:08d}")
            _log(output, {"event": "selection", "step": step, "metrics": scores, "improved": improved,
                          "selects_checkpoint": True})
        if (improved or step == min(config["checkpoint"]["first_step"], total_steps)
                or step % config["checkpoint"]["every"] == 0 or state["cursor"] == len(train)):
            last_saved_checkpoint = measured_save()
            last_saved_step = step
            if improved:
                _atomic_json(output / "best_checkpoint.json", {"path": str(output / state["best_checkpoint"]),
                                                               "step": step, "selector_metrics": scores})
        stopped = stop_if_requested()
        if stopped is not None:
            return stopped
    final_checkpoint = last_saved_checkpoint if last_saved_step == state["step"] else measured_save()
    final_metrics = assessment(dev, cohorts["dev"], "final-dev")
    _log(output, {"event": "full_dev_final", "step": state["step"], "metrics": final_metrics,
                  "selects_checkpoint": False})
    if not state["best_checkpoint"]:
        raise RuntimeError("epoch completed without fixed-selector checkpoint selection")
    best_state = load_checkpoint(output / state["best_checkpoint"], model, optimizer, identity=identity, device=device)
    if prefix_evaluation is not None:
        prefix_evaluation.reset()
    best_metrics = assessment(dev, cohorts["dev"], "best-dev", best_state["step"])
    report = {"protocol": training_protocol(identity), "completed": True, "identity": identity, "state": state,
              "final_checkpoint": str(final_checkpoint), "final_full_dev": final_metrics,
              "best_step": best_state["step"], "best_full_dev": best_metrics,
              "selected_by": "fixed_2000_user_ndcg@10" if formal and domain == "book" else "fixed_user_ndcg@10",
              "full_dev_reselected": False, "test_labels_accessed": False}
    _atomic_json(output / "report.json", report)
    _log(output, {"event": "complete", "step": state["step"], "best_step": best_state["step"],
                  "report": str(output / "report.json")})
    return report


def run_training(config, data_dir, output, **kwargs):
    """One writer per run directory; completed checkpoints are atomic and immutable."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("this training output already has an active writer") from error
        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid()) + "\n")
        stream.flush()
        try:
            return _run_training(config, data_dir, output, **kwargs)
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--teacher-cache", type=Path,
                        help="completed cache directory or teacher-cache.json; binds its teacher identity and SHA256")
    parser.add_argument("--stop-request-file", type=Path,
                        help="save at an optimizer boundary then stop; remove the file before resume")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if args.teacher_cache is not None:
        config = bind_teacher_cache(config, args.teacher_cache)
    report = run_training(config, args.data_dir, args.output, device=args.device, resume=args.resume,
                          stop_request_file=args.stop_request_file)
    print(json.dumps({"report": str(args.output / "report.json"), "best_step": report.get("best_step"),
                      "completed": report.get("completed"),
                      "status": report.get("status", "complete" if report.get("completed") is True else "unknown")}, sort_keys=True))


if __name__ == "__main__":
    main()
