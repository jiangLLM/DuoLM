"""Validate independent train and evaluation candidate identities.

The legacy random-only preparation stays valid. New preparations separately
bind preserved mixed training groups and the already evaluated random dev
pool; changing the model seed never silently changes either sampler.
"""
from __future__ import annotations

from collections import Counter

from .prepare_candidate_ranking import (
    DATA_FILES, SOURCE_FILES, PREPARATION_SCHEMA, SPLIT_PREPARATION_SCHEMA,
    PROTOCOL, candidate_policy, manifest_identity, _digest, _hex_digest,
    _validate_source_manifest, _validate_mixed_source_manifest,
)


def configured_candidate_seed(config):
    """Old configs used the model seed; new configs freeze it independently."""
    seed = config.get("candidate_seed", config.get("seed"))
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("candidate_seed must be a nonnegative integer")
    return seed


def configured_train_policy(config):
    return config["data"].get("train_policy", "random-unobserved")


def _source_hashes(hashes, label):
    if (not isinstance(hashes, dict) or set(hashes) != set(SOURCE_FILES)
            or not all(_hex_digest(value) for value in hashes.values())):
        raise ValueError(f"{label} requires the complete five-file snapshot hashes")


def validate_random_evaluation_reference(manifest, hashes, *, seed):
    """Check a frozen original random preparation before reusing dev bytes."""
    _source_hashes(hashes, "evaluation reference")
    if not isinstance(manifest, dict):
        raise ValueError("evaluation reference preparation manifest must be a mapping")
    policy = candidate_policy(train_candidates=manifest.get("train_candidates"), seed=seed)
    if (manifest.get("schema") != PREPARATION_SCHEMA or manifest.get("protocol") != PROTOCOL
            or manifest.get("candidate_policy") != "random-unobserved"
            or manifest.get("target_included") is not True or manifest.get("test_labels_accessed") is not False
            or manifest.get("dev_candidates") != 100 or manifest.get("seed") != seed
            or manifest.get("policy") != policy or manifest.get("policy_sha256") != _digest(policy)
            or manifest.get("identity_sha256") != manifest_identity(manifest)
            or manifest.get("files") != {name: hashes[name] for name in DATA_FILES}
            or "retrieval" in manifest):
        raise ValueError("evaluation reference must be a complete frozen random-100 preparation")
    source_hashes = manifest.get("source_files_sha256")
    _source_hashes(source_hashes, "evaluation reference original source")
    source = manifest.get("source_preparation_manifest")
    _validate_source_manifest(source, source_hashes)
    if manifest.get("source_preparation_manifest_sha256") != source_hashes["preparation-manifest.json"]:
        raise ValueError("evaluation reference original manifest hash mismatch")
    for key in ("sources", "catalog_truncation", "unavailable_dev_users", "unavailable_dev_reasons",
                "chronology_audit", "query_cap", "train_queries", "dev_queries"):
        if manifest.get(key) != source.get(key):
            raise ValueError(f"evaluation reference changed frozen source {key}")
    for name in ("catalog.json", "cohorts.json"):
        if hashes[name] != source_hashes[name]:
            raise ValueError(f"evaluation reference changed frozen source {name}")
    code = manifest.get("code_files_sha256")
    if (not isinstance(code, dict) or not code or not all(_hex_digest(value) for value in code.values())
            or manifest.get("code_sha256") != _digest(code)):
        raise ValueError("evaluation reference preparation code signature mismatch")
    return manifest


def _split_provenance(manifest, *, seed, train_policy):
    source_hashes = manifest.get("source_files_sha256")
    _source_hashes(source_hashes, "training source")
    source = manifest.get("source_preparation_manifest")
    _validate_source_manifest(source, source_hashes)
    if manifest.get("source_preparation_manifest_sha256") != source_hashes["preparation-manifest.json"]:
        raise ValueError("training source manifest hash mismatch")
    if train_policy == "mixed-history":
        _validate_mixed_source_manifest(source, source_hashes, seed=seed)
    reference_hashes = manifest.get("eval_source_files_sha256")
    reference = manifest.get("eval_preparation_manifest")
    validate_random_evaluation_reference(reference, reference_hashes, seed=seed)
    if (manifest.get("eval_preparation_manifest_sha256") != reference_hashes["preparation-manifest.json"]
            or reference.get("source_files_sha256") != source_hashes
            or reference.get("source_preparation_manifest") != source):
        raise ValueError("training and evaluation must originate from exactly the same frozen source")
    for key in ("sources", "catalog_truncation", "unavailable_dev_users", "unavailable_dev_reasons",
                "chronology_audit", "query_cap", "train_queries", "dev_queries"):
        if manifest.get(key) != source.get(key):
            raise ValueError(f"split preparation changed frozen source {key}")
    files = manifest["files"]
    if files["dev.jsonl"] != reference_hashes["dev.jsonl"]:
        raise ValueError("fixed random-100 evaluation bytes differ from the evaluated reference")
    for name in ("catalog.json", "cohorts.json"):
        if files[name] != source_hashes[name] or files[name] != reference_hashes[name]:
            raise ValueError(f"split preparation changed frozen {name}")
    if train_policy == "mixed-history" and files["train.jsonl"] != source_hashes["train.jsonl"]:
        raise ValueError("mixed-history training bytes/order differ from the frozen source")
    expected_train = {"policy": manifest["policy"].get("train_sampling", manifest["policy"]),
                      "source_manifest_sha256": source_hashes["preparation-manifest.json"],
                      "source_train_sha256": source_hashes["train.jsonl"], "seed": seed}
    if train_policy == "mixed-history":
        expected_train.update({"retrieval": source["retrieval"],
                               "context_manifest_sha256": source["context_manifest_sha256"],
                               "retrieval_manifest_sha256": source["retrieval_manifest_sha256"]})
    expected_eval = {"policy": candidate_policy(train_candidates=100, seed=seed),
                     "reference_manifest_sha256": reference_hashes["preparation-manifest.json"],
                     "reference_identity_sha256": reference["identity_sha256"],
                     "reference_dev_sha256": reference_hashes["dev.jsonl"], "seed": seed}
    if (manifest.get("train_sampling_provenance") != expected_train
            or manifest.get("eval_sampling_provenance") != expected_eval):
        raise ValueError("separate training/evaluation sampler provenance mismatch")
    code = manifest.get("code_files_sha256")
    if (not isinstance(code, dict) or not code or not all(_hex_digest(value) for value in code.values())
            or manifest.get("code_sha256") != _digest(code)):
        raise ValueError("split preparation code signature mismatch")


def validate_candidate_preparation(manifest, config, files, train, dev):
    """Check manifest against loader hashes and RecommendationQuery sequences.

    ``files`` accepts the trainer's ``train/dev/catalog/cohorts`` SHA mapping
    or the preparation's explicit four filenames. File hashing and query
    parsing occur in the caller so this validator cannot read test labels.
    """
    data = config["data"]
    if config.get("schema_version") != PROTOCOL:
        raise ValueError("candidate preparation requires the candidate-ranking protocol")
    seed, train_policy = configured_candidate_seed(config), configured_train_policy(config)
    policy = candidate_policy(train_candidates=data.get("train_candidates"), seed=seed,
                              train_policy=train_policy)
    schema = manifest.get("schema")
    if (schema not in {PREPARATION_SCHEMA, SPLIT_PREPARATION_SCHEMA}
            or manifest.get("protocol") != PROTOCOL or manifest.get("target_included") is not True
            or manifest.get("test_labels_accessed") is not False
            or manifest.get("candidate_policy") != "random-unobserved"
            or data.get("candidate_policy") != "random-unobserved"
            or data.get("eval_policy", "random-unobserved") != "random-unobserved"
            or data.get("eval_candidates") != 100 or manifest.get("dev_candidates") != 100
            or manifest.get("train_candidates") != data.get("train_candidates")
            or manifest.get("query_cap") != data.get("query_cap")
            or manifest.get("policy") != policy or manifest.get("policy_sha256") != _digest(policy)
            or manifest.get("identity_sha256") != manifest_identity(manifest)
            or manifest.get("train_queries") != len(train) or manifest.get("dev_queries") != len(dev)
            or "retrieval" in manifest):
        raise ValueError("candidate ranking preparation identity/policy/counts mismatch")
    names = {"train.jsonl": "train", "dev.jsonl": "dev", "catalog.json": "catalog", "cohorts.json": "cohorts"}
    expected_files = {name: files[name] if name in files else files.get(alias) for name, alias in names.items()}
    if (set(manifest.get("files", {})) != set(DATA_FILES) or not all(_hex_digest(v) for v in expected_files.values())
            or manifest["files"] != expected_files):
        raise ValueError("prepared dataset checksum mismatch")
    if schema == PREPARATION_SCHEMA:
        if train_policy != "random-unobserved":
            raise ValueError("legacy random-only preparation cannot be labeled mixed-history")
    else:
        if (manifest.get("train_policy") != train_policy or manifest.get("eval_policy") != "random-unobserved"
                or manifest.get("candidate_seed") != seed or manifest.get("seed") != seed):
            raise ValueError("independent candidate seed/train/evaluation policy mismatch")
        _split_provenance(manifest, seed=seed, train_policy=train_policy)
    query_ids = set()
    for split, queries, count in (("train", train, data["train_candidates"]), ("dev", dev, 100)):
        for query in queries:
            if query.query_id in query_ids:
                raise ValueError("query IDs must be globally unique")
            query_ids.add(query.query_id)
            if (query.split != split or len(query.candidate_ids) != count or query.positive_index is None
                    or len(query.history) > data["history_max"]):
                raise ValueError(f"invalid frozen {split} candidate/history protocol")
    if not train or max(Counter(query.user_id for query in train).values()) > data["query_cap"]:
        raise ValueError("candidate training queries exceed the frozen query cap")
    return manifest
