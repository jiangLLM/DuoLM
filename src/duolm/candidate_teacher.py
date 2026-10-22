"""Evaluate existing next-item T weights and cache new candidate-pool logits.

The source checkpoint retains its original identity. Only model weights are
restored; optimizer, RNG and training cursor are never restored into a run.
Generated predictions and resumable cache shards belong outside Git.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import stat

import torch

from .next_item_model import GroupRanker, domain_template, query_input_hash
from . import train_next_item as trainer


CACHE_SCHEMA = "books-next-item-scalar-teacher-v1"
EVALUATION_SCOPE = "target-guaranteed-candidate-ranking"
PREFIX_QUALIFICATION_SMALL_LIMIT = 16 * 1024**2


def _schema(teacher, kind):
    return ("games" if teacher.get("domain") == "game" else "books") + "-candidate-teacher-" + kind + "-v1"


def _record(path):
    path = Path(path)
    return {"sha256": trainer._file_hash(path), "bytes": path.stat().st_size}


def _canonical_model(spec):
    repo, revision = spec["repo"], spec["revision"]
    path = Path(repo)
    if path.parent.name == "snapshots" and path.parent.parent.name.startswith("models--"):
        if path.name != revision:
            raise ValueError("B snapshot revision differs from frozen model revision")
        repo = path.parent.parent.name.removeprefix("models--").replace("--", "/")
    return {"repo": repo, "revision": revision}


def _verify_b(source, target, override):
    source_b, target_b = _canonical_model(source["model"]["b_model"]), _canonical_model(target["model"]["b_model"])
    if source_b != target_b:
        raise ValueError("source teacher and candidate task require the same fixed B repo/revision")
    if override is not None:
        path = Path(override)
        if (not path.is_dir() or path.parent.name != "snapshots"
                or not path.parent.parent.name.startswith("models--")
                or _canonical_model({"repo": str(path), "revision": source_b["revision"]}) != source_b):
            raise ValueError("B override must be a local HF snapshot path matching the frozen repo/revision; "
                             "verify snapshot contents separately before evaluation")
    return source_b


def inspect_checkpoint(checkpoint_dir, checkpoint_config, config, *, formal=True, b_model_path=None):
    """Verify hashes, source config and signed state before constructing a model."""
    checkpoint_dir, checkpoint_config = Path(checkpoint_dir), Path(checkpoint_config)
    source = json.loads(checkpoint_config.read_text())
    trainer.validate_config(source, formal=formal)
    trainer.validate_config(config, formal=formal, require_teacher_cache=False)
    if source.get("schema_version") not in {"books-next-item-v1", trainer.CANDIDATE_PROTOCOL, trainer.GAMES_CANDIDATE_PROTOCOL} or source["model"]["variant"] != "T":
        raise ValueError("candidate teacher requires a next-item T checkpoint, never a star-rating checkpoint")
    if config.get("schema_version") not in {trainer.CANDIDATE_PROTOCOL, trainer.GAMES_CANDIDATE_PROTOCOL}:
        raise ValueError("candidate teacher requires a target-guaranteed candidate ranking configuration")
    domain = trainer.config_domain(config)
    if trainer.config_domain(source) != domain:
        raise ValueError("teacher checkpoint domain differs from the candidate task")
    if domain == "game" and (source["data"].get("train_candidates") != 8
            or source["data"].get("train_policy") != "mixed-history"
            or config["data"].get("train_candidates") != 8 or config["data"].get("train_policy") != "mixed-history"):
        raise ValueError("Games requires an actual mixed8 T checkpoint for mixed8 distillation")
    b_identity = _verify_b(source, config, b_model_path)
    raw_manifest = json.loads((checkpoint_dir / "manifest.json").read_text())
    identity = raw_manifest.get("identity")
    if not isinstance(identity, dict) or identity.get("config_sha256") != trainer._digest(source):
        raise ValueError("checkpoint source configuration hash differs from its identity")
    if domain == "game" and (identity.get("domain") != domain
            or identity.get("schema_version") != trainer.GAMES_CANDIDATE_PROTOCOL
            or identity.get("prompt_template") != {**domain_template(domain), "sha256": trainer._digest(domain_template(domain))}):
        raise ValueError("Games teacher checkpoint must bind its domain, schema and frozen prompt")
    for name in ("data_sha256", "source_sha256"):
        values = identity.get(name)
        if not isinstance(values, dict) or not values or not all(trainer._hex_digest(v) for v in values.values()):
            raise ValueError("checkpoint requires complete source/data hash identities")
    manifest = trainer._checkpoint_manifest(checkpoint_dir, identity=identity)
    # mmap keeps the original checkpoint's Adam storages off the working RAM
    # path. No optimizer object is created or restored by this tool.
    payload = torch.load(checkpoint_dir / "training.pt", map_location="cpu", weights_only=True, mmap=True)
    state = payload.get("state")
    if payload.get("protocol") != trainer.training_protocol(identity) or payload.get("identity") != identity:
        raise ValueError("checkpoint payload protocol/identity differs from its manifest")
    if (not isinstance(state, dict) or trainer._digest(state) != manifest["state_sha256"]
            or state.get("step") != manifest.get("step") or state.get("cursor") != manifest.get("query_cursor")
            or isinstance(state.get("step"), bool) or not isinstance(state.get("step"), int) or state["step"] < 1
            or isinstance(state.get("cursor"), bool) or not isinstance(state.get("cursor"), int) or state["cursor"] < 1):
        raise ValueError("checkpoint payload state signature/step/cursor differs from its manifest")
    weights = payload.get("model")
    if not isinstance(weights, dict) or not weights or not all(isinstance(v, torch.Tensor) for v in weights.values()):
        raise ValueError("checkpoint requires a complete T model state dict")
    teacher = {"model_variant": "T", "checkpoint_sha256": manifest["files"]["training.pt"]["sha256"],
               "checkpoint_manifest_sha256": trainer._file_hash(checkpoint_dir / "manifest.json"),
               "checkpoint_identity_sha256": trainer._digest(identity), "checkpoint_step": state["step"],
               "source_config_sha256": trainer._digest(source),
               "source_config_file_sha256": trainer._file_hash(checkpoint_config),
               "b_model": b_identity, "load_mode": "weights_only"}
    if domain == "game":
        teacher.update(domain=domain, source_schema=source["schema_version"], train_candidates=8,
                       train_policy="mixed-history", prompt_template=identity["prompt_template"])
    return source, weights, teacher


def _build_model(source, config, weights, *, device, b_model_path=None, model_factory=None):
    if model_factory is not None:
        model = model_factory(source, device)
    else:
        spec, runtime = source["model"], config["training"]
        model = GroupRanker.from_pretrained(None, str(b_model_path or spec["b_model"]["repo"]),
                    b_revision=spec["b_model"]["revision"], variant="T", device=device, domain=trainer.config_domain(source),
                    attention_backend=runtime["attention_backend"],
                    max_reader_packed_tokens=runtime["max_reader_packed_tokens"],
                    max_encoder_packed_tokens=runtime["max_encoder_packed_tokens"],
                    max_logical_tokens=runtime["max_logical_tokens"], gradient_checkpointing=False)
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model


def _sources(checkpoint_dir, checkpoint_config, data_dir, data_hashes):
    data_dir = Path(data_dir)
    paths = {"checkpoint": Path(checkpoint_dir) / "training.pt",
             "checkpoint_manifest": Path(checkpoint_dir) / "manifest.json", "checkpoint_config": Path(checkpoint_config)}
    names = {"train": "train.jsonl", "dev": "dev.jsonl", "cohorts": "cohorts.json",
             "catalog": "catalog.json" if (data_dir / "catalog.json").exists() else "catalog.jsonl",
             "preparation_manifest": "preparation-manifest.json", "first_seen": "first-seen.json"}
    paths.update({key: data_dir / names[key] for key in data_hashes})
    paths.update({"code:" + name: Path(__file__).parent / name for name in
                 ("candidate_teacher.py", "train_next_item.py", "next_item_model.py", "next_item_data.py",
                  "next_item_metrics.py", "ranking_packing.py", "resampler.py", "prepare_candidate_ranking.py",
                 "candidate_data_validation.py")})
    if "first_seen" in data_hashes:
        paths.update({"code:" + name: Path(__file__).parent / name for name in ("games_data.py", "build_games_assets.py")})
    hashes = {key: trainer._file_hash(path) for key, path in paths.items()}
    stats = {key: (path.stat().st_size, path.stat().st_mtime_ns) for key, path in paths.items()}
    return paths, hashes, stats


def _verify_sources(paths, hashes, stats, *, full=False):
    for key, path in paths.items():
        current = (path.stat().st_size, path.stat().st_mtime_ns)
        if current != stats[key] or (full and trainer._file_hash(path) != hashes[key]):
            raise ValueError(f"teacher source changed during operation: {key}")


@contextmanager
def _operation(output, *, phase, identity):
    output = Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (output / "operation.lock").open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("teacher output already has an active writer") from error
        try:
            path = output / "operation-manifest.json"
            expected = {"schema": "books-candidate-teacher-operation-v1", "phase": phase,
                        "identity": identity, "identity_sha256": trainer._digest(identity)}
            if path.exists():
                if json.loads(path.read_text()) != expected:
                    raise ValueError("teacher output manifest differs from checkpoint/data/config/source identity")
            else:
                if any(p.name != "operation.lock" for p in output.iterdir()):
                    raise ValueError("teacher output requires a new directory or matching operation manifest")
                trainer._atomic_json(path, expected)
            yield output
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _prepare(checkpoint_dir, checkpoint_config, config, data_dir, *, device, b_model_path, formal):
    source, weights, teacher = inspect_checkpoint(checkpoint_dir, checkpoint_config, config,
                                                  formal=formal, b_model_path=b_model_path)
    train, dev, items, cohorts, data_hashes = trainer.load_dataset(data_dir, config, formal=formal)
    if trainer.config_domain(config) == "game":
        checkpoint_identity = json.loads((Path(checkpoint_dir) / "manifest.json").read_text())["identity"]
        if checkpoint_identity["data_sha256"] != data_hashes:
            raise ValueError("Games mixed8 teacher must have trained on the same frozen candidate groups")
    paths, hashes, stats = _sources(checkpoint_dir, checkpoint_config, data_dir, data_hashes)
    if hashes["checkpoint"] != teacher["checkpoint_sha256"] or hashes["checkpoint_config"] != teacher["source_config_file_sha256"]:
        raise ValueError("teacher checkpoint/config changed while loading the candidate task")
    identity = {"checkpoint_sha256": teacher["checkpoint_sha256"],
                "checkpoint_manifest_sha256": teacher["checkpoint_manifest_sha256"],
                "source_config_sha256": teacher["source_config_sha256"],
                "source_config_file_sha256": teacher["source_config_file_sha256"],
                "config_sha256": trainer._digest(config), "data_sha256": data_hashes,
                "source_sha256": {k.removeprefix("code:"): v for k, v in hashes.items() if k.startswith("code:")},
                "inference": {"device_type": torch.device(device).type, "compute_dtype": trainer.inference_compute_dtype(config),
                              "attention_backend": config["training"]["attention_backend"],
                              "microbatch_queries": config["training"]["microbatch_queries"],
                              "max_reader_packed_tokens": config["training"]["max_reader_packed_tokens"],
                              "max_logical_tokens": config["training"]["max_logical_tokens"]}}
    execution = trainer._evaluation_execution_metadata(config)
    if execution:
        identity["inference"].update(execution)
    return source, weights, teacher, train, dev, items, cohorts, identity, paths, hashes, stats


def _prediction_metrics(path, queries, users, unavailable_users):
    """Recompute scientific metrics from the exact prepared query sequence."""
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for query in queries:
            line = stream.readline()
            if not line:
                raise ValueError("completed teacher predictions omitted an evaluation query")
            row = json.loads(line)
            scores = row.get("scores") if isinstance(row, dict) else None
            if (not isinstance(row, dict) or row.get("query_id") != query.query_id
                    or row.get("user_id") != query.user_id or row.get("target_item_id") != query.target_item_id
                    or row.get("candidate_ids") != list(query.candidate_ids)):
                raise ValueError("completed teacher predictions differ from the prepared query sequence")
            if not isinstance(scores, list) or len(scores) != len(query.candidate_ids) or any(
                    isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in scores):
                raise ValueError("completed teacher predictions require finite scores for every candidate")
            order = sorted(range(len(scores)), key=lambda i: (-scores[i], query.candidate_ids[i]))
            rank = order.index(query.positive_index) + 1 if query.positive_index is not None else None
            if row.get("rank") != rank:
                raise ValueError("completed teacher prediction rank differs from its scores")
            rows.append((query, scores))
        if stream.readline():
            raise ValueError("completed teacher predictions contain extra evaluation queries")
    metrics = trainer.ranking_metrics(rows, expected_user_ids=users, unavailable_user_ids=unavailable_users, k=10)
    metrics["hr@1"] = trainer.ranking_metrics(rows, expected_user_ids=users,
                                           unavailable_user_ids=unavailable_users, k=1)["hr@1"]
    metrics["candidate_target_coverage"] = metrics.pop("retrieval_recall")
    metrics["evaluation_scope"] = EVALUATION_SCOPE
    return metrics


def evaluate_checkpoint(checkpoint_dir, checkpoint_config, config, data_dir, output, *, device,
                        b_model_path=None, subset="selector", formal=True, model_factory=None):
    if subset not in {"selector", "full-dev", "disjoint-report"}:
        raise ValueError("teacher evaluation subset must be selector, full-dev or disjoint-report")
    if config["evaluation"].get("scoring_backend") == "prefix-kv":
        raise ValueError("candidate teacher evaluation uses tree scoring; use duolm.evaluate for a bound prefix-KV checkpoint")
    expected_execution = trainer._evaluation_execution_metadata(config)
    source, weights, teacher, _, dev, items, cohorts, identity, paths, hashes, stats = _prepare(
        checkpoint_dir, checkpoint_config, config, data_dir, device=device, b_model_path=b_model_path, formal=formal)
    selector = set(cohorts["checkpoint_selector"])
    users = tuple(cohorts["checkpoint_selector"] if subset == "selector" else
                  cohorts["dev"] if subset == "full-dev" else [u for u in cohorts["dev"] if u not in selector])
    if not users:
        raise ValueError("teacher evaluation requires a nonempty fixed user subset")
    user_set = set(users)
    queries = [q for q in dev if q.user_id in user_set]
    unavailable_users = tuple(u for u in cohorts["unavailable_dev_users"] if u in user_set)
    identity["subset"] = subset
    identity["evaluation_max_reader_packed_tokens"] = config["evaluation"]["max_reader_packed_tokens"]
    with _operation(output, phase="evaluate", identity=identity) as output:
        report_path = output / "report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text())
            if (report.get("schema") != _schema(teacher, "evaluation")
                    or report.get("completed") is not True or report.get("identity") != identity or report.get("teacher") != teacher
                    or report.get("evaluation_scope") != EVALUATION_SCOPE or report.get("subset") != subset
                    or report.get("test_labels_accessed") is not False or report.get("weights_only") is not True
                    or report.get("predictions_file") != "predictions.jsonl"
                    or report.get("files", {}).get("predictions.jsonl") != _record(output / "predictions.jsonl")):
                raise ValueError("completed teacher evaluation is corrupt or belongs to another identity")
            metrics = report.get("metrics")
            if (not isinstance(metrics, dict) or report.get("metrics_sha256") != trainer._digest(metrics)
                    or isinstance(metrics.get("reader_tokens"), bool)
                    or not isinstance(metrics.get("reader_tokens"), int) or metrics["reader_tokens"] < 0
                    or {k: metrics.get(k) for k in expected_execution} != expected_execution
                    or {k: v for k, v in metrics.items() if k != "reader_tokens" and k not in expected_execution}
                    != _prediction_metrics(output / "predictions.jsonl", queries, users, unavailable_users)):
                raise ValueError("completed teacher evaluation metrics differ from its predictions")
            _verify_sources(paths, hashes, stats, full=True)
            return report
        model = _build_model(source, config, weights, device=device, b_model_path=b_model_path, model_factory=model_factory)
        trainer._atomic_json(output / "teacher_identity.json", teacher)
        temporary = output / ".predictions.jsonl.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            def sink(row):
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            metrics = trainer.evaluate(model, queries, items, expected_users=users, device=device,
                    compute_dtype=trainer.inference_compute_dtype(config),
                    max_reader_packed_tokens=config["evaluation"]["max_reader_packed_tokens"],
                    microbatch_queries=config["training"]["microbatch_queries"],
                    unavailable_users=unavailable_users,
                    evaluation_scope=config["schema_version"], prediction_sink=sink,
                    strict_math=config["evaluation"].get("scoring_backend") == "tree")
            stream.flush()
            os.fsync(stream.fileno())
        _verify_sources(paths, hashes, stats, full=True)
        os.replace(temporary, output / "predictions.jsonl")
        report = {"schema": _schema(teacher, "evaluation"), "completed": True,
                  "evaluation_scope": EVALUATION_SCOPE, "subset": subset, "identity": identity,
                  "teacher": teacher, "metrics": metrics, "metrics_sha256": trainer._digest(metrics),
                  "predictions_file": "predictions.jsonl",
                  "files": {"predictions.jsonl": _record(output / "predictions.jsonl")},
                  "test_labels_accessed": False, "weights_only": True}
        trainer._atomic_json(report_path, report)
        return report


def _validate_rows(rows, queries, items, *, domain="book"):
    if not isinstance(rows, list) or len(rows) != len(queries):
        raise ValueError("teacher shard must contain every query in its declared sequence")
    for row, query in zip(rows, queries):
        scores = row.get("scores") if isinstance(row, dict) else None
        if (not isinstance(row, dict) or row.get("query_id") != query.query_id
                or row.get("input_hash") != query_input_hash(query, items, domain=domain)
                or row.get("candidate_ids") != list(query.candidate_ids)):
            raise ValueError("teacher shard query/input hash/candidate order differs from prepared data")
        if not isinstance(scores, list) or len(scores) != len(query.candidate_ids) or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in scores):
            raise ValueError("teacher shard requires finite scalar scores for every candidate")


def _read_shard(path, queries, items, teacher, start):
    manifest_path = path.with_name(path.name + ".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != _schema(teacher, "shard") or manifest.get("completed") is not True
            or manifest.get("teacher") != teacher or manifest.get("start") != start
            or manifest.get("end") != start + len(queries) or manifest.get("file") != _record(path)):
        raise ValueError("teacher shard checksum/identity/range mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    _validate_rows(rows, queries, items, domain=teacher.get("domain", "book"))
    return rows


def _write_shard(path, rows, teacher, start):
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    trainer._atomic_json(path.with_name(path.name + ".manifest.json"), {
        "schema": _schema(teacher, "shard"), "completed": True, "teacher": teacher,
        "start": start, "end": start + len(rows), "file": _record(path)})


def _merge_cache(output, queries, items, teacher, shard_queries):
    cache = output / "teacher-cache.json"
    temporary = output / ".teacher-cache.json.tmp"
    header = {"schema": "games-next-item-scalar-teacher-v1" if teacher.get("domain") == "game" else CACHE_SCHEMA,
              "teacher": teacher, "complete": True, "format": "query-lines-v1"}
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(header, separators=(",", ":"), allow_nan=False)[:-1] + ',"queries":{\n')
        for start in range(0, len(queries), shard_queries):
            batch = queries[start:start + shard_queries]
            rows = _read_shard(output / "shards" / f"queries-{start:09d}.jsonl", batch, items, teacher, start)
            for offset, row in enumerate(rows):
                key = row.pop("query_id")
                value = json.dumps({key: row}, separators=(",", ":"), allow_nan=False)[1:-1]
                stream.write(value + ("," if start + offset < len(queries) - 1 else "") + "\n")
        stream.write("}}\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, cache)
    completion = {"schema": _schema(teacher, "cache-completion"), "completed": True,
                  "teacher": teacher, "queries": len(queries), "file": _record(cache)}
    return cache, completion


def _qualification_comparison(reference, actual, query):
    """Recompute the fixed score/probability gates, not a saved passed flag."""
    for values in (reference, actual):
        if (not isinstance(values, list) or len(values) != len(query.candidate_ids)
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                       for v in values)):
            raise ValueError("prefix-KV qualification requires finite scores for every candidate")
    tensors = [torch.tensor(values, dtype=torch.float32) for values in (reference, actual)]
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("prefix-KV qualification requires finite FP32 scores")
    score_error = float((tensors[0] - tensors[1]).abs().max())
    probability_error = float((tensors[0].softmax(0) - tensors[1].softmax(0)).abs().max())
    orders = [sorted(range(len(values)), key=lambda i: (-float(values[i]), query.candidate_ids[i]))
              for values in tensors]
    ranks = [order.index(query.positive_index) + 1 for order in orders]
    return {"max_abs_score_error": score_error, "max_abs_probability_error": probability_error,
            "absolute_threshold": 1e-4, "threshold_relaxed": False, "finite_scores": True,
            "reference_target_rank": ranks[0], "actual_target_rank": ranks[1],
            "target_rank_equal": ranks[0] == ranks[1], "full_order_equal": orders[0] == orders[1],
            "passed": score_error <= 1e-4 and probability_error <= 1e-4 and ranks[0] == ranks[1]}


def _small_qualification_file(path, *, allow_empty=False):
    if path.is_symlink():
        raise ValueError("prefix-KV qualification requires bounded regular nonsymlink small files")
    metadata = path.stat()
    if (not stat.S_ISREG(metadata.st_mode)
            or not (0 if allow_empty else 1) <= metadata.st_size <= PREFIX_QUALIFICATION_SMALL_LIMIT):
        raise ValueError("prefix-KV qualification requires bounded regular nonsymlink small files")
    return metadata.st_size, metadata.st_mtime_ns


def _prefix_kv_qualification(directory, *, config, teacher, train, identity, paths, hashes, stats,
                             runtime, kv_source_path=None):
    """Bind actual train-mixed8 CUDA evidence to this producer and checkpoint.

    This is a bounded qualification of selected real groups, not an assertion
    that every training query has been compared or that throughput is accepted.
    Only the small qualification files and KV module are newly hashed.
    """
    from . import next_item_teacher_kv as kv_module

    if (trainer.inference_compute_dtype(config) != "float32"
            or config["model"]["variant"] != "T"
            or config["training"]["attention_backend"] != "flex_attention"
            or config["data"].get("train_candidates") != 8
            or config["data"].get("train_policy") != "mixed-history"):
        raise ValueError("prefix-KV cache requires the qualified FP32 mixed8 T/Flex configuration")
    if (not isinstance(runtime, dict) or set(runtime) != {"torch", "transformers", "cuda", "gpu"}
            or any(not isinstance(value, str) or not value for value in runtime.values())):
        raise ValueError("prefix-KV qualification requires real CUDA/runtime provenance")
    if directory is None:
        raise ValueError("prefix-KV cache requires a train-mixed8 qualification directory")
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("prefix-KV qualification directory must be a nonsymlink directory")
    evidence_paths = {"qualification:" + name: directory / name for name in
                      ("report.json", "completion-manifest.json", "predictions.jsonl")}
    evidence_paths["code:next_item_teacher_kv.py"] = Path(kv_source_path or kv_module.__file__)
    records = {}
    for key, path in evidence_paths.items():
        before = _small_qualification_file(path)
        records[key] = _record(path)
        if (path.stat().st_size, path.stat().st_mtime_ns) != before:
            raise ValueError("prefix-KV qualification source changed during binding")
        paths[key], hashes[key], stats[key] = path, records[key]["sha256"], before
    report = json.loads(evidence_paths["qualification:report.json"].read_text())
    completion = json.loads(evidence_paths["qualification:completion-manifest.json"].read_text())
    protocol = kv_module.PROTOCOL
    if (not isinstance(completion, dict) or completion.get("schema") != protocol + "-qualification-completion"
            or completion.get("completed") is not True
            or completion.get("files") != {name: records["qualification:" + name]
                                           for name in ("report.json", "predictions.jsonl")}):
        raise ValueError("prefix-KV qualification completion checksums differ from its report/predictions")
    reference_execution = {"kind": "current-producer-legacy-tree", "original_trainer_runtime_compared": False,
                           "attention_backend": "flex_attention", "input_assembly": "legacy", "score_readout": "legacy",
                           "shared_candidate_header": False, "max_reader_packed_tokens": config["training"]["max_reader_packed_tokens"],
                           "microbatch_queries": config["training"]["microbatch_queries"],
                           "scoring_call": "cache-microbatch-tree"}
    if (not isinstance(report, dict) or report.get("schema") != protocol + "-qualification"
            or report.get("completed") is not True or report.get("passed") is not True
            or report.get("qualification_cohort") != "train-mixed8" or report.get("candidate_count") != 8
            or report.get("compute_dtype") != "float32" or report.get("tf32") is not False
            or report.get("training_changed") is not False or report.get("test_labels_accessed") is not False
            or report.get("teacher") != teacher or report.get("producer_identity") != identity
            or any(report.get(key) != value for key, value in runtime.items())
            or report.get("predictions_file") != records["qualification:predictions.jsonl"]
            or not isinstance(report.get("source_files"), dict)
            or report.get("source_files", {}).get("code:kv") != records["code:next_item_teacher_kv.py"]
            or report.get("reference_execution") != reference_execution):
        raise ValueError("prefix-KV qualification checkpoint/producer/runtime/source provenance mismatch")
    selection = report.get("query_selection", {})
    ids = selection.get("query_ids") if isinstance(selection, dict) else None
    if (not isinstance(ids, list) or not config["training"]["microbatch_queries"] <= len(ids) <= 16
            or any(not isinstance(value, str) or not value or value != value.strip() for value in ids)
            or len(set(ids)) != len(ids) or type(report.get("queries")) is not int or report["queries"] != len(ids)
            or selection.get("cohort") != "train-mixed8" or selection.get("split") != "train"
            or selection.get("candidate_count") != 8
            or selection.get("policy") not in {"explicit-fixed-training-ids", "fixed-training-prefix"}):
        raise ValueError("prefix-KV qualification requires actual fixed train-mixed8 query selection")
    by_id = {query.query_id: query for query in train}
    if (len(by_id) != len(train) or any(value not in by_id for value in ids)
            or (selection["policy"] == "fixed-training-prefix" and ids != [q.query_id for q in train[:len(ids)]])):
        raise ValueError("prefix-KV qualification query IDs/order differ from current training groups")
    rows = [json.loads(line) for line in evidence_paths["qualification:predictions.jsonl"].read_text().splitlines()]
    if len(rows) != len(ids):
        raise ValueError("prefix-KV qualification predictions omitted or added a query")
    microbatch_queries = config["training"]["microbatch_queries"]
    for index, (row, query_id) in enumerate(zip(rows, ids)):
        query = by_id[query_id]
        batch_start = index // microbatch_queries * microbatch_queries
        batch_ids = ids[batch_start:batch_start + microbatch_queries]
        if (query.split != "train" or len(query.candidate_ids) != 8 or len(set(query.candidate_ids)) != 8
                or type(query.positive_index) is not int or not 0 <= query.positive_index < 8
                or not isinstance(row, dict) or row.get("query_id") != query_id
                or row.get("user_id") != query.user_id or row.get("qualification_cohort") != "train-mixed8"
                or row.get("candidate_ids") != list(query.candidate_ids)
                or type(row.get("positive_index")) is not int or row["positive_index"] != query.positive_index
                or row.get("tree_batch_query_ids") != batch_ids
                or type(row.get("tree_batch_query_count")) is not int or row["tree_batch_query_count"] != len(batch_ids)
                or not isinstance(row.get("chunks"), dict) or set(row["chunks"]) != {"1", "8", "16"}):
            raise ValueError("prefix-KV qualification candidate/query order differs from actual train-mixed8")
        for check in row["chunks"].values():
            if not isinstance(check, dict):
                raise ValueError("prefix-KV qualification requires each chunk's numerical evidence")
            expected = _qualification_comparison(row.get("reference_scores"), check.get("scores"), query)
            if (not expected["passed"] or any(check.get(key) != value for key, value in expected.items())
                    or any(check.get(key) is not value for key, value in expected.items() if isinstance(value, bool))):
                raise ValueError("prefix-KV qualification did not pass fixed 1e-4 score/probability and exact rank gates")
    _verify_sources(paths, hashes, stats)
    return {"protocol": protocol, "files": {key: value for key, value in records.items()},
            "runtime": runtime, "cohort": "train-mixed8", "query_ids": ids,
            "reference_execution": report["reference_execution"]}


def cache_checkpoint(checkpoint_dir, checkpoint_config, config, data_dir, output, *, device,
                     b_model_path=None, shard_queries=512, stop_after_shards=None,
                     formal=True, model_factory=None, scoring_backend="tree", candidate_chunk_size=8,
                     qualification_dir=None):
    if isinstance(shard_queries, bool) or not isinstance(shard_queries, int) or shard_queries < 1:
        raise ValueError("shard_queries must be a positive integer")
    if stop_after_shards is not None and (isinstance(stop_after_shards, bool)
            or not isinstance(stop_after_shards, int) or stop_after_shards < 1):
        raise ValueError("stop_after_shards must be a positive integer")
    if scoring_backend not in {"tree", "prefix-kv"}:
        raise ValueError("teacher scoring_backend must be tree or prefix-kv")
    if type(candidate_chunk_size) is not int or candidate_chunk_size not in {1, 8, 16}:
        raise ValueError("teacher candidate_chunk_size must be 1, 8 or 16")
    if scoring_backend == "tree" and (qualification_dir is not None or candidate_chunk_size != 8):
        raise ValueError("qualification/chunk options are only available for prefix-kv cache")
    source, weights, teacher, train, _, items, _, identity, paths, hashes, stats = _prepare(
        checkpoint_dir, checkpoint_config, config, data_dir, device=device, b_model_path=b_model_path, formal=formal)
    if scoring_backend == "prefix-kv":
        import transformers
        if torch.device(device).type != "cuda" or not torch.cuda.is_available():
            raise ValueError("prefix-KV cache requires real CUDA qualification/runtime")
        runtime = {"torch": torch.__version__, "transformers": transformers.__version__,
                   "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(torch.device(device))}
        qualification = _prefix_kv_qualification(qualification_dir, config=config, teacher=teacher,
                            train=train, identity=identity, paths=paths, hashes=hashes, stats=stats, runtime=runtime)
        identity["source_sha256"]["next_item_teacher_kv.py"] = hashes["code:next_item_teacher_kv.py"]
        identity["scoring"] = {"backend": scoring_backend, "attention_backend": "sdpa", "compute_dtype": "float32",
                               "tf32": False, "candidate_chunk_size": candidate_chunk_size,
                               "qualification": qualification}
    identity["shard_queries"] = shard_queries
    with _operation(output, phase="cache", identity=identity) as output:
        if scoring_backend == "prefix-kv":
            copied = output / "prefix-qualification"
            if copied.is_symlink():
                raise ValueError("cached prefix-KV qualification directory must be nonsymlink")
            copied.mkdir(exist_ok=True)
            for name in ("report.json", "completion-manifest.json", "predictions.jsonl"):
                target = copied / name
                if not target.exists():
                    temporary = copied / ("." + name + ".tmp")
                    if temporary.exists() or temporary.is_symlink():
                        _small_qualification_file(temporary, allow_empty=True)
                    shutil.copyfile(paths["qualification:" + name], temporary)
                    _small_qualification_file(temporary)
                    if _record(temporary) != qualification["files"]["qualification:" + name]:
                        raise ValueError("copied prefix-KV qualification bytes differ from bound evidence")
                    with temporary.open("rb") as stream:
                        os.fsync(stream.fileno())
                    os.replace(temporary, target)
                _small_qualification_file(target)
                if _record(target) != qualification["files"]["qualification:" + name]:
                    raise ValueError("cached prefix-KV qualification bytes differ from bound evidence")
                key = "cached-qualification:" + name
                paths[key], hashes[key] = target, qualification["files"]["qualification:" + name]["sha256"]
                stats[key] = (target.stat().st_size, target.stat().st_mtime_ns)
        teacher_path = output / "teacher_identity.json"
        if teacher_path.exists() and json.loads(teacher_path.read_text()) != teacher:
            raise ValueError("cached teacher identity differs from source checkpoint")
        trainer._atomic_json(teacher_path, teacher)
        cache_path = output / "teacher-cache.json"
        report_path = output / "report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if (report.get("schema") != _schema(teacher, "cache-result")
                    or report.get("completed") is not True or report.get("identity") != identity or report.get("teacher") != teacher
                    or report.get("files", {}).get("teacher-cache.json") != _record(cache_path)):
                raise ValueError("completed teacher cache is corrupt or belongs to another identity")
            completion = json.loads((output / "teacher-cache.json.manifest.json").read_text())
            if completion.get("completed") is not True or completion.get("teacher") != teacher or completion.get("file") != _record(cache_path):
                raise ValueError("completed teacher cache completion record is corrupt")
            if scoring_backend == "prefix-kv":
                _verify_sources(paths, hashes, stats)
            return report
        shards = output / "shards"
        shards.mkdir(exist_ok=True)
        model, scorer, computed, done = None, None, 0, 0
        for start in range(0, len(train), shard_queries):
            _verify_sources(paths, hashes, stats)
            batch = train[start:start + shard_queries]
            path = shards / f"queries-{start:09d}.jsonl"
            if path.is_file() and path.with_name(path.name + ".manifest.json").is_file():
                _read_shard(path, batch, items, teacher, start)
            else:
                if path.with_name(path.name + ".manifest.json").exists():
                    raise ValueError("committed teacher shard is missing its score file")
                if stop_after_shards is not None and computed >= stop_after_shards:
                    trainer._atomic_json(output / "state.json", {"completed": False, "next_query": start,
                                        "queries": len(train), "committed_queries": done, "identity": identity})
                    return {"completed": False, "next_query": start, "teacher": teacher, "identity": identity}
                if model is None:
                    model = _build_model(source, config, weights, device=device, b_model_path=b_model_path, model_factory=model_factory)
                    if scoring_backend == "prefix-kv":
                        from .next_item_teacher_kv import TeacherPrefixKVScorer
                        scorer = TeacherPrefixKVScorer(model, candidate_chunk_size=candidate_chunk_size)
                rows = []
                with torch.no_grad():
                    for offset in range(0, len(batch), config["training"]["microbatch_queries"]):
                        microbatch = batch[offset:offset + config["training"]["microbatch_queries"]]
                        # The explicit BF16 teacher tree profile also forbids
                        # TF32 for FP32 operations outside autocast.
                        strict_tree = config["evaluation"].get("scoring_backend") == "tree"
                        with (trainer.inference_context(device, "float32") if strict_tree else nullcontext()), \
                                trainer.inference_context(device, trainer.inference_compute_dtype(config)):
                            scores = (scorer(microbatch, items) if scorer is not None else
                                      model(microbatch, items, max_reader_packed_tokens=config["training"]["max_reader_packed_tokens"]))["scores"]
                        if len(scores) != len(microbatch):
                            raise ValueError("teacher model omitted a training query")
                        rows.extend({"query_id": q.query_id, "input_hash": query_input_hash(q, items, domain=teacher.get("domain", "book")),
                                     "candidate_ids": list(q.candidate_ids), "scores": s.detach().float().cpu().tolist()}
                                    for q, s in zip(microbatch, scores))
                _validate_rows(rows, batch, items, domain=teacher.get("domain", "book"))
                _verify_sources(paths, hashes, stats)
                _write_shard(path, rows, teacher, start)
                computed += 1
            done += len(batch)
            trainer._atomic_json(output / "state.json", {"completed": False, "next_query": start + len(batch),
                                "queries": len(train), "committed_queries": done, "identity": identity})
        _verify_sources(paths, hashes, stats, full=True)
        cache, completion = _merge_cache(output, train, items, teacher, shard_queries)
        _verify_sources(paths, hashes, stats, full=True)
        trainer._atomic_json(output / "teacher-cache.json.manifest.json", completion)
        report = {"schema": _schema(teacher, "cache-result"), "completed": True,
                  "evaluation_scope": EVALUATION_SCOPE, "teacher": teacher, "identity": identity,
                  "queries": len(train), "cache_file": cache.name, "teacher_identity_file": teacher_path.name,
                  "files": {cache.name: completion["file"]}, "test_labels_accessed": False, "weights_only": True}
        trainer._atomic_json(report_path, report)
        trainer._atomic_json(output / "state.json", {"completed": True, "next_query": len(train),
                            "queries": len(train), "committed_queries": len(train), "identity": identity})
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("evaluate", "cache"):
        command = commands.add_parser(name)
        command.add_argument("--checkpoint-dir", type=Path, required=True)
        command.add_argument("--checkpoint-config", type=Path, required=True)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--data-dir", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", required=True)
        command.add_argument("--b-model-path", type=Path)
        if name == "evaluate":
            command.add_argument("--subset", choices=("selector", "full-dev", "disjoint-report"), default="selector")
        else:
            command.add_argument("--shard-queries", type=int, default=512)
            command.add_argument("--stop-after-shards", type=int)
            command.add_argument("--scoring-backend", choices=("tree", "prefix-kv"), default="tree")
            command.add_argument("--candidate-chunk-size", type=int, choices=(1, 8, 16), default=8)
            command.add_argument("--qualification-dir", type=Path,
                                 help="completed train-mixed8 CUDA qualification for this exact producer/checkpoint")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    kwargs = {"device": args.device, "b_model_path": args.b_model_path}
    if args.command == "evaluate":
        result = evaluate_checkpoint(args.checkpoint_dir, args.checkpoint_config, config, args.data_dir,
                                     args.output, subset=args.subset, **kwargs)
    else:
        result = cache_checkpoint(args.checkpoint_dir, args.checkpoint_config, config, args.data_dir,
                                 args.output, shard_queries=args.shard_queries, stop_after_shards=args.stop_after_shards,
                                 scoring_backend=args.scoring_backend, candidate_chunk_size=args.candidate_chunk_size,
                                 qualification_dir=args.qualification_dir, **kwargs)
    print(json.dumps({"completed": result["completed"], "output": str(args.output),
                      "teacher": result["teacher"], "metrics": result.get("metrics")}, sort_keys=True))


if __name__ == "__main__":
    main()
