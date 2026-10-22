"""Export the frozen full-text next-book teacher, without its training split.

Only owned checkpoint tensor data are deserialized (weights_only, CPU). Public
weights are neither loaded nor downloaded: the native config constructs B and
strict checkpoint loading supplies every B/head parameter. Absent training
bytes remain explicitly unverified; archived hashes are provenance, not a fresh
training-data verification.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import torch

from .nextbook_export import (SOURCE_FILES, TOKENIZER_FILES, OPTIONAL_TOKENIZER_FILES,
                             _digest, _drop_modules, _finite, _frozen_modules, _hash,
                             _json, _record, _regular, _require, _write_reader, source_files, config_domain, checkpoint_protocol)

KIND = "mixlm_nextbook_teacher_serving"
PROTOCOL = "mixlm-nextbook-teacher-serving-v1"
CHECKPOINT_PROTOCOL = "books-next-item-training-v1"
TEMPLATE = {
    "version": "books-next-item-teacher-interface-v1", "variant": "T",
    "instruction": "Predict the user's next book interaction from the past history.\n",
    "history_header": "History {index}:\nItem: ",
    "rating": "\nUser rating: {rating}\n", "candidate_header": "Candidate:\nItem: ",
    "score_suffix": "\nNext-book score:", "score_readout": "last_suffix_token",
    "shared_candidate_header": False, "add_special_tokens": False,
    "tokenization": "separate_instruction_history_header_content_rating_candidate_header_content_suffix",
    "append_eos": False,
}


def teacher_template(domain="book"):
    if domain == "book":
        return dict(TEMPLATE)
    _require(domain == "game", "unsupported teacher domain")
    return {**TEMPLATE, "version": "games-next-item-teacher-interface-v1",
            "instruction": "Predict the user's next game-product interaction from the past history.\n",
            "score_suffix": "\nNext-game-product score:"}


def _hex(value, length=64):
    return type(value) is str and len(value) == length and all(c in "0123456789abcdef" for c in value)


def _snapshot_metadata(root, spec):
    """Verify original snapshot inventory and only the files needed for serving."""
    root = Path(root)
    report_path = _regular(root / "snapshot-manifest.json")
    report = _json(report_path)
    signature = _hash(report_path)
    _require(spec.get("snapshot_manifest_sha256") in (None, signature), "snapshot manifest hash differs")
    _require(report.get("repo_id") == spec["repo"] and report.get("revision") == spec["revision"],
             "public reader repository/revision differs")
    records = report.get("files", {})
    _require(isinstance(records, dict) and {"config.json", *TOKENIZER_FILES} <= records.keys(),
             "public snapshot lacks config/tokenizer identity")
    names = {"config.json", *TOKENIZER_FILES} | (set(OPTIONAL_TOKENIZER_FILES) & records.keys())
    checked = {name: _record(root / name) for name in sorted(names)}
    _require(all(value == records[name] for name, value in checked.items()), "public config/tokenizer checksum differs")
    return {"repo": spec["repo"], "revision": spec["revision"], "snapshot_manifest_sha256": signature,
            "files": checked, "unused_unverified_files": sorted(set(records) - names),
            "public_weights_loaded": False}, {str(root / name): value["sha256"] for name, value in checked.items()} | {
                str(report_path): signature}


def _provenance(path, checkpoint, receipt, identity):
    if path is None:
        # Public checkpoints carry their own source/data identities and exact
        # training.pt checksum. No private backup or archive receipt is needed.
        record = _record(checkpoint.parent / "manifest.json")
        return {"record": record, "evidence": {"kind": "local_checkpoint_manifest",
                "checkpoint_sha256": receipt["files"]["training.pt"]["sha256"]}}, {}
    path = _regular(path)
    evidence = _json(path)
    record = _record(path)
    if evidence.get("schema") == "books-next-item-backup-v1":
        metadata = evidence.get("metadata", {})
        _require(evidence.get("kind") == "checkpoint" and metadata.get("identity") == identity
                 and metadata.get("checkpoint_sha256") == receipt["files"]["training.pt"]["sha256"]
                 and metadata.get("step") == receipt["step"]
                 and metadata.get("state_sha256") == receipt["state_sha256"],
                 "archived checkpoint provenance differs")
        records = evidence.get("files", {})
    else:
        _require(evidence.get("all_members_verified") is True and _hex(evidence.get("revision"), 40)
                 and bool(evidence.get("repo_id")) and bool(evidence.get("prefix")),
                 "restore provenance must identify a fully verified immutable archive")
        archive = evidence.get("archive", {})
        _require(_hex(archive.get("sha256")) and type(archive.get("bytes")) is int and archive["bytes"] > 0,
                 "restore archive identity is missing")
        records = evidence.get("restored_files", {})
    for suffix, actual in ((f"{checkpoint.parent.name}/training.pt", _record(checkpoint)),
                           (f"{checkpoint.parent.name}/manifest.json", _record(checkpoint.parent / "manifest.json"))):
        matches = [value for name, value in records.items() if str(name).endswith(suffix)]
        _require(len(matches) == 1 and matches[0] == actual, "archive does not bind original checkpoint bytes: " + suffix)
    return {"record": record, "evidence": evidence}, {str(path): record["sha256"]}


def _inputs(args):
    checkpoint = Path(args.checkpoint)
    if checkpoint.is_dir():
        checkpoint /= "training.pt"
    checkpoint = _regular(checkpoint)
    _require(checkpoint.name == "training.pt", "checkpoint must be original training.pt")
    receipt_path = _regular(checkpoint.parent / "manifest.json")
    receipt = _json(receipt_path)
    identity = receipt.get("identity")
    expected = getattr(args, "expected_checkpoint_sha256", None)
    _require(_hex(expected), "expected_checkpoint_sha256 must pin the authorized teacher")
    _require(receipt.get("protocol") == checkpoint_protocol(identity) and type(receipt.get("step")) is int
             and receipt["step"] > 0, "requires a completed positive-step teacher checkpoint")
    checkpoint_record = _record(checkpoint)
    _require(checkpoint_record == receipt.get("files", {}).get("training.pt")
             and checkpoint_record["sha256"] == expected, "checkpoint checksum/size differs from manifest or pinned teacher")
    _require(isinstance(identity, dict), "checkpoint lacks frozen identity")
    config_path = _regular(getattr(args, "config", None) or checkpoint.parent.parent.parent / "run_manifest.json")
    document = _json(config_path)
    config = document.get("config", document)
    if "config" in document:
        _require(document.get("protocol") == checkpoint_protocol(identity) and document.get("identity") == identity,
                 "run manifest and checkpoint identity differ")
    _require(_digest(config) == identity.get("config_sha256"), "configuration hash differs from checkpoint")
    spec = config.get("model", {})
    _require(spec.get("variant") == "T" and spec.get("score_readout", "legacy") == "legacy"
             and spec.get("shared_candidate_header", False) is False, "requires the frozen full-text teacher template")
    source = Path(args.source) / "src/duolm"
    _require(set(identity.get("source_sha256", {})) == set(source_files(config)), "teacher source inventory differs")
    bound = {str(_regular(source / name)): digest for name, digest in identity["source_sha256"].items()}
    _require(all(_hash(path) == value for path, value in bound.items()), "frozen teacher source hash differs")
    data = Path(args.data)
    catalog = data / "catalog.json"
    if not catalog.exists():
        catalog = data / "catalog.jsonl"
    names = {"catalog": catalog.name, "train": "train.jsonl", "dev": "dev.jsonl",
             "cohorts": "cohorts.json", "preparation_manifest": "preparation-manifest.json"}
    if config_domain(config) == "game":
        names["first_seen"] = "first-seen.json"
    _require(set(identity.get("data_sha256", {})) == set(names), "teacher data identity inventory differs")
    verification = {}
    for key, name in names.items():
        path = data / name
        expected_data = identity["data_sha256"][key]
        _require(_hex(expected_data), "invalid frozen data hash")
        if path.exists():
            _require(_hash(_regular(path)) == expected_data, "original training data checksum differs: " + key)
            bound[str(path)] = expected_data
            verification[key] = {"sha256": expected_data, "status": "verified_local_bytes"}
        else:
            _require(key != "catalog", "canonical teacher catalog is required")
            verification[key] = {"sha256": expected_data, "status": "archived_identity_only_bytes_absent"}
    public, snapshot_bound = _snapshot_metadata(args.b_model, spec["b_model"])
    evidence, evidence_bound = _provenance(getattr(args, "provenance", None), checkpoint, receipt, identity)
    bound.update(snapshot_bound)
    bound.update(evidence_bound)
    bound.update({str(checkpoint): expected, str(receipt_path): _hash(receipt_path), str(config_path): _hash(config_path)})
    return checkpoint, receipt, identity, config, source, catalog, public, verification, evidence, bound


def _check_frozen_template(model):
    # Run the actual frozen assembly method and capture its separate text calls.
    # Old T checkpoints predate CANDIDATE_HEADER/LEGACY_SCORE_SUFFIX constants.
    captured = []
    def capture(text):
        captured.append(text)
        return torch.zeros((1, model.b.config.hidden_size))
    model._native = capture
    query = SimpleNamespace(history=(SimpleNamespace(item_id="h1", rating=1),
                                     SimpleNamespace(item_id="h2", rating=5)), candidate_ids=("c1", "c2"))
    items = {key: "CONTENT_" + key for key in ("h1", "h2", "c1", "c2")}
    try:
        model._query_embeddings(query, items)
    finally:
        del model._native
    template = teacher_template(model.domain)
    expected = [template["instruction"]]
    for index, event in enumerate(query.history, 1):
        expected += [template["history_header"].format(index=index), items[event.item_id],
                     template["rating"].format(rating=event.rating)]
    for item in query.candidate_ids:
        expected += [template["candidate_header"], items[item], template["score_suffix"]]
    _require(captured == expected, "frozen teacher prompt segments differ from the supported template")
    return {**template, "sha256": _digest(template)}


def _load_model(args, payload, config, module):
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    native = AutoConfig.from_pretrained(args.b_model, local_files_only=True, trust_remote_code=False)
    _require(native.model_type == "qwen3" and not getattr(native, "rope_scaling", None),
             "teacher requires native Qwen3 default RoPE")
    # Public Qwen configs declare BF16; from_config honors that dtype unless
    # explicitly overridden. The trained checkpoint is FP32 and may not be
    # rounded merely while constructing the destination for strict loading.
    reader = AutoModel.from_config(native, torch_dtype=torch.float32,
                                   attn_implementation="sdpa", trust_remote_code=False)
    tokens = AutoTokenizer.from_pretrained(args.b_model, local_files_only=True, trust_remote_code=False)
    train = config["training"]
    model = module.GroupRanker(None, reader, None, tokens, variant="T", attention_backend="sdpa", domain=config_domain(config),
                              max_reader_packed_tokens=train["max_reader_packed_tokens"],
                              max_encoder_packed_tokens=train["max_encoder_packed_tokens"],
                              max_logical_tokens=train["max_logical_tokens"], gradient_checkpointing=False)
    weights = payload.get("model")
    _require(isinstance(weights, dict) and set(weights) == set(model.state_dict())
             and all(key == "head.weight" or key.startswith("b.") for key in weights),
             "checkpoint is not an exact B plus scalar-head teacher state")
    _require(all(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                 and value.dtype == torch.float32 and _finite(value) for value in weights.values()),
             "teacher checkpoint requires finite CPU FP32 tensor weights")
    _require(tuple(weights["head.weight"].shape) == (1, native.hidden_size), "teacher scalar head must be [1, width]")
    model.load_state_dict(weights, strict=True)
    _require(all(value.device.type == "cpu" and value.dtype == torch.float32 for value in model.parameters()),
             "teacher parameters must remain CPU FP32 after checkpoint loading")
    template = _check_frozen_template(model)
    metadata = model.metadata()
    metadata.update(teacher_template=template, shared_candidate_header=False, score_readout="last_suffix_token",
                    attention_backend=train["attention_backend"],
                    packing_limits={"logical_context": train["max_logical_tokens"],
                                    "reader_packed_tokens": train["max_reader_packed_tokens"]})
    return model.eval().requires_grad_(False), metadata


def export_nextbook_teacher(args):
    output = Path(args.output).absolute()
    _require(not output.exists() and not output.is_symlink(), "output exists; choose a new directory")
    for root in (args.source, args.data, args.b_model, Path(args.checkpoint).parent):
        _require(not output.resolve().is_relative_to(Path(root).resolve()), "output must be outside immutable inputs")
    _require(importlib.metadata.version("transformers") == "4.57.1", "requires reference Transformers 4.57.1")
    checkpoint, receipt, identity, config, package, catalog, public, verified, provenance, bound = _inputs(args)
    alias, module, trainer = _frozen_modules(package, identity["source_sha256"])
    try:
        trainer.validate_config(config, formal=True)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        _require(payload.get("protocol") == checkpoint_protocol(identity) and payload.get("identity") == identity,
                 "checkpoint payload protocol/identity differs")
        state = payload.get("state", {})
        _require(state.get("step") == receipt["step"]
                 and state.get("cursor") == receipt.get("query_cursor")
                 and _digest(state) == receipt.get("state_sha256"), "checkpoint state/selected step signature differs")
        model, metadata = _load_model(args, payload, config, module)
        # Original T source predates the optional expected-hash argument on
        # trainer._load_catalog. Hash-bound canonical parsing is owned here.
        items = load_catalog(catalog)
        output.mkdir(parents=True, exist_ok=False)
        native, tokenizer_files, placeholder = _write_reader(output, model, args)
        shutil.copyfile(catalog, output / catalog.name)
        _require(all(_hash(_regular(path)) == digest for path, digest in bound.items()), "immutable inputs changed during export")
        records = {path.relative_to(output).as_posix(): _record(path)
                   for path in sorted(output.rglob("*")) if path.is_file()}
        manifest = {
            "schema_version": 1, "protocol": PROTOCOL, "kind": KIND, "variant": "T",
            "checkpoint_protocol": checkpoint_protocol(identity), "checkpoint_sha256": bound[str(checkpoint)],
            "checkpoint_step": state["step"], "checkpoint_state_sha256": receipt["state_sha256"],
            "checkpoint_manifest_sha256": bound[str(checkpoint.parent / "manifest.json")],
            "checkpoint_identity": identity, "config": config, "config_sha256": identity["config_sha256"],
            "source_sha256": identity["source_sha256"], "data_sha256": identity["data_sha256"],
            "data_verification": verified, "archived_provenance": provenance,
            "catalog_file": catalog.name, "catalog_sha256": identity["data_sha256"]["catalog"],
            "catalog_content_sha256": _digest(items), "catalog_items": len(items),
            "native_reader_config": native, "native_tokenizer_sha256": tokenizer_files,
            "model_metadata": metadata, "public_models": {"b": public},
            "scalar_head": {"file": "scalar_head.safetensors", "key": "head.weight", "dtype": "float32",
                            "shape": [1, native["hidden_size"]], **records["scalar_head.safetensors"]},
            "reader": {"directory": "reader", "backbone_dtype": "float32", "backbone_key_mapping": "b.<key> -> model.<key>",
                       "vocabulary_head": {"trained_score_head": False, "used_for_scoring": False, "policy": placeholder}},
            "files": records, "export_source_sha256": _hash(__file__),
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "safetensors")},
            "scope": "Trained full-text teacher B and original FP32 scalar head; serving numerical acceptance is separate",
        }
        manifest["manifest_signature"] = _digest(manifest)
        (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n")
        return verify_nextbook_teacher_export(output)
    finally:
        _drop_modules(alias)


def load_catalog(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        items = {}
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                _require(type(row) is dict and set(row) == {"item_id", "text"} and row["item_id"] not in items,
                         "catalog JSONL requires unique item_id/text rows")
                items[row["item_id"]] = row["text"]
    else:
        items = _json(path)
    _require(type(items) is dict and items and all(type(key) is str and key and type(text) is str and text.strip()
                                                  for key, text in items.items()), "invalid canonical catalog")
    return items


def verify_nextbook_teacher_export(root):
    from safetensors import safe_open
    from safetensors.torch import load_file
    root = Path(root)
    _require(root.is_dir() and not root.is_symlink(), "invalid teacher export root")
    manifest = _json(_regular(root / "manifest.json"))
    _require(manifest.get("manifest_signature") == _digest({key: val for key, val in manifest.items() if key != "manifest_signature"}),
             "teacher manifest signature differs")
    _require(manifest.get("schema_version") == 1 and manifest.get("protocol") == PROTOCOL and manifest.get("kind") == KIND
             and manifest.get("variant") == "T" and manifest.get("checkpoint_protocol") == checkpoint_protocol(manifest.get("checkpoint_identity"))
             and type(manifest.get("checkpoint_step")) is int and manifest["checkpoint_step"] > 0, "unsupported full-text teacher export")
    identity, config = manifest["checkpoint_identity"], manifest["config"]
    _require(_digest(config) == manifest["config_sha256"] == identity["config_sha256"] and config["model"]["variant"] == "T"
             and manifest["source_sha256"] == identity["source_sha256"] and manifest["data_sha256"] == identity["data_sha256"],
             "teacher scientific identity differs")
    metadata = manifest["model_metadata"]
    template = teacher_template(config_domain(config))
    _require(metadata.get("teacher_template") == {**template, "sha256": _digest(template)}
             and metadata.get("variant") == "T" and metadata.get("score_readout") == "last_suffix_token"
             and metadata.get("shared_candidate_header") is False, "teacher template identity differs")
    files, tokenizers = manifest.get("files", {}), manifest.get("native_tokenizer_sha256", {})
    _require(type(tokenizers) is dict and set(TOKENIZER_FILES) <= tokenizers.keys()
             and tokenizers.keys() <= set(TOKENIZER_FILES + OPTIONAL_TOKENIZER_FILES), "teacher tokenizer allowlist differs")
    catalog = manifest.get("catalog_file")
    _require(catalog in {"catalog.json", "catalog.jsonl"}, "unsupported teacher catalog filename")
    allowed = {catalog, "reader/config.json", "reader/model.safetensors", "scalar_head.safetensors"} | {
        "reader/" + name for name in tokenizers}
    _require(type(files) is dict and set(files) == allowed, "teacher export allowed serving files differ")
    actual = set()
    for path in root.rglob("*"):
        _require(not path.is_symlink(), "teacher export contains a symlink")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    _require(actual == allowed | {"manifest.json"}, "teacher export file inventory differs")
    for name, expected in files.items():
        _require(_record(root / name) == expected, "teacher export checksum/size differs: " + name)
    native = manifest["native_reader_config"]
    width = native["hidden_size"]
    _require(native["model_type"] == "qwen3" and type(width) is int and width > 0
             and metadata.get("b_width") == width, "invalid teacher reader dimensions")
    reader_config = _json(root / "reader/config.json")
    # Serialization records the exporting Transformers version, independent of
    # the public snapshot's older producer version. Architectural keys remain
    # exact; only loader metadata and intentionally overridden fields differ.
    ignored_config = {"architectures", "dtype", "torch_dtype", "use_cache", "_name_or_path", "transformers_version"}
    _require(all(reader_config.get(key) == native.get(key) for key in native if key not in ignored_config)
             and reader_config.get("architectures") == ["Qwen3ForCausalLM"]
             and manifest["reader"]["vocabulary_head"]["trained_score_head"] is False
             and manifest["reader"]["vocabulary_head"]["used_for_scoring"] is False, "teacher reader config/head semantics differ")
    with safe_open(str(root / "reader/model.safetensors"), framework="pt", device="cpu") as values:
        keys = set(values.keys())
        _require({"model.embed_tokens.weight", "model.norm.weight"} <= keys
                 and all(key.startswith("model.") or key == "lm_head.weight" for key in keys)
                 and all(values.get_slice(key).get_dtype() == "F32" for key in keys), "teacher B must be native FP32 weights")
        _require(values.get_slice("model.embed_tokens.weight").get_shape() == [native["vocab_size"], width],
                 "teacher reader embedding shape differs")
        if not native["tie_word_embeddings"]:
            _require("lm_head.weight" in keys and values.get_slice("lm_head.weight").get_shape() == [native["vocab_size"], width],
                     "teacher untied reader lacks placeholder vocabulary head")
    head = load_file(str(root / "scalar_head.safetensors"), device="cpu")
    _require(set(head) == {"head.weight"} and head["head.weight"].dtype == torch.float32
             and list(head["head.weight"].shape) == [1, width] and _finite(head["head.weight"]), "invalid teacher scalar head")
    record = manifest["scalar_head"]
    _require(record == {"file": "scalar_head.safetensors", "key": "head.weight", "dtype": "float32",
                        "shape": [1, width], **files["scalar_head.safetensors"]}, "teacher scalar record differs")
    items = load_catalog(root / catalog)
    _require(len(items) == manifest["catalog_items"] and _digest(items) == manifest["catalog_content_sha256"]
             and files[catalog]["sha256"] == manifest["catalog_sha256"] == identity["data_sha256"]["catalog"],
             "teacher catalog differs from checkpoint")
    for name, digest in tokenizers.items():
        _require(files["reader/" + name]["sha256"] == digest == manifest["public_models"]["b"]["files"][name]["sha256"],
                 "teacher tokenizer differs from public snapshot")
    limits = metadata["packing_limits"]
    _require(limits == {"logical_context": config["training"]["max_logical_tokens"],
                        "reader_packed_tokens": config["training"]["max_reader_packed_tokens"]}, "teacher packing limits differ")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "checkpoint", "data", "b-model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--provenance", type=Path, help="Optional archive receipt; local checkpoints do not require it")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    manifest = export_nextbook_teacher(parser.parse_args(argv))
    print(json.dumps({"manifest_signature": manifest["manifest_signature"], "checkpoint_sha256": manifest["checkpoint_sha256"],
                      "checkpoint_step": manifest["checkpoint_step"], "catalog_items": manifest["catalog_items"]}, sort_keys=True))


if __name__ == "__main__":
    main()
