"""No-clobber serving assets for a frozen compact next-book V3 checkpoint.

The HF vocabulary head is a loader placeholder. Only the separately exported
FP32 ``head.weight`` defines the trained scalar score. Vector export records its
precision, but does not certify SGLang/reference numerical equivalence.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
from types import ModuleType, SimpleNamespace
import uuid

import numpy as np
import torch


KIND = "mixlm_nextbook_serving"
PROTOCOL = "books-next-item-training-v1"
SOURCE_FILES = ("train_next_item.py", "next_item_metrics.py", "next_item_model.py",
                "next_item_data.py", "ranking_packing.py", "resampler.py")
CANDIDATE_SOURCE_FILES = ("prepare_candidate_ranking.py", "candidate_teacher.py",
                          "candidate_data_validation.py")
GAMES_SOURCE_FILES = ("games_data.py", "build_games_assets.py", "candidate_teacher.py")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
OPTIONAL_TOKENIZER_FILES = ("vocab.json", "merges.txt", "special_tokens_map.json",
                            "added_tokens.json", "chat_template.jinja")


def _require(condition, message):
    if not condition:
        raise ValueError("Next-book export: " + message)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def _regular(path):
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), "missing/unsafe regular file: " + str(path))
    return path


def _record(path):
    path = _regular(path)
    return {"sha256": _hash(path), "bytes": path.stat().st_size}


def _finite(tensor):
    values = tensor.detach().reshape(-1)
    return all(bool(torch.isfinite(values[start:start + 1048576]).all())
               for start in range(0, values.numel(), 1048576))


def _snapshot(path, spec):
    """Verify a relocated local snapshot against its pinned public identity."""
    path = Path(path)
    _require(path.is_dir() and not path.is_symlink(), "invalid local public model directory")
    manifest = _regular(path / "snapshot-manifest.json")
    digest = _hash(manifest)
    declared = spec.get("snapshot_manifest_sha256")
    _require(declared is None or digest == declared, "public snapshot manifest differs from training config")
    report = _json(manifest)
    _require(report.get("repo_id") == spec["repo"] and report.get("revision") == spec["revision"],
             "public snapshot repository/revision differs")
    actual = {}
    for entry in path.rglob("*"):
        relative = entry.relative_to(path)
        if any(part in {".cache", "__pycache__"} for part in relative.parts) or entry == manifest:
            continue
        _require(not entry.is_symlink(), "public snapshot must contain regular owned files")
        if entry.is_file():
            actual[relative.as_posix()] = _record(entry)
    _require(actual == report.get("files") and any(name.endswith((".safetensors", ".bin")) for name in actual),
             "public snapshot inventory/checksum differs")
    _require(set(TOKENIZER_FILES) <= actual.keys() and "config.json" in actual,
             "public snapshot lacks config/tokenizer files")
    return {"repo": spec["repo"], "revision": spec["revision"],
            "snapshot_manifest_sha256": digest, "files": actual}


def source_files(config):
    """Exact public trainer source inventory for each supported data protocol."""
    schema = config.get("schema_version")
    if schema == "games-candidate-ranking-v1":
        return SOURCE_FILES + GAMES_SOURCE_FILES
    if schema == "books-candidate-ranking-v1":
        return SOURCE_FILES + CANDIDATE_SOURCE_FILES
    return SOURCE_FILES


def checkpoint_protocol(identity):
    return "games-next-item-training-v1" if isinstance(identity, dict) and identity.get("domain") == "game" else PROTOCOL


def config_domain(config):
    return "game" if config.get("schema_version") == "games-candidate-ranking-v1" else "book"


def _inputs(args):
    checkpoint = Path(args.checkpoint)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "training.pt"
    _require(checkpoint.name == "training.pt", "checkpoint must be training.pt or its step directory")
    checkpoint = _regular(checkpoint)
    checkpoint_manifest = _regular(checkpoint.parent / "manifest.json")
    receipt = _json(checkpoint_manifest)
    _require(receipt.get("protocol") == checkpoint_protocol(receipt.get("identity")), "unsupported checkpoint protocol")
    identity = receipt.get("identity")
    _require(isinstance(identity, dict), "checkpoint lacks frozen identity")
    _require(_record(checkpoint) == receipt.get("files", {}).get("training.pt"),
             "checkpoint checksum/size differs from its manifest")
    config_path = getattr(args, "config", None)
    config_path = _regular(config_path or checkpoint.parent.parent.parent / "run_manifest.json")
    document = _json(config_path)
    if "config" in document:
        _require(document.get("protocol") == checkpoint_protocol(identity) and document.get("identity") == identity,
                 "run manifest and checkpoint identity differ")
        config = document["config"]
    else:
        config = document
    _require(_digest(config) == identity.get("config_sha256"), "configuration hash differs from checkpoint")
    spec = config.get("model", {})
    _require(spec.get("variant") == "V3" and spec.get("m") == 1
             and spec.get("a_readout") == "last_token"
             and spec.get("shared_candidate_header") is True and spec.get("score_readout") == "candidate",
             "only compact last-token/shared-header/candidate-readout m1V3 is supported")
    source = Path(args.source)
    package = source / "src" / "duolm"
    _require(package.is_dir(), "source must be the frozen checkout root")
    names = source_files(config)
    sources = identity.get("source_sha256", {})
    _require(set(sources) == set(names), "source identity inventory differs from training protocol")
    bound = {str(_regular(package / name)): digest for name, digest in sources.items()}
    _require(all(_hash(path) == digest for path, digest in bound.items()), "frozen training source hash differs")
    data = Path(args.data)
    catalog = data / "catalog.json"
    if not catalog.exists():
        catalog = data / "catalog.jsonl"
    data_names = {"train": "train.jsonl", "dev": "dev.jsonl", "cohorts": "cohorts.json",
                  "catalog": catalog.name, "preparation_manifest": "preparation-manifest.json"}
    if config_domain(config) == "game":
        data_names["first_seen"] = "first-seen.json"
    _require(set(identity.get("data_sha256", {})) == set(data_names), "data identity inventory differs")
    for key, name in data_names.items():
        bound[str(_regular(data / name))] = identity["data_sha256"][key]
    _require(all(_hash(path) == digest for path, digest in bound.items()), "training data/source checksum differs")
    public = {side: _snapshot(getattr(args, side + "_model"), spec[side + "_model"])
              for side in ("a", "b")}
    bound[str(checkpoint)] = receipt["files"]["training.pt"]["sha256"]
    bound[str(checkpoint_manifest)] = _hash(checkpoint_manifest)
    bound[str(config_path)] = _hash(config_path)
    for side in ("a", "b"):
        model_path = Path(getattr(args, side + "_model"))
        bound[str(model_path / "snapshot-manifest.json")] = public[side]["snapshot_manifest_sha256"]
        bound.update({str(model_path / name): record["sha256"] for name, record in public[side]["files"].items()})
    return checkpoint, receipt, identity, config, package, catalog, public, bound


def _frozen_modules(package, sources):
    """Execute checked source bytes under a private package, ignoring stale pyc."""
    alias = "_duolm_nextbook_export_" + uuid.uuid4().hex
    root = ModuleType(alias)
    root.__path__ = []
    sys.modules[alias] = root
    try:
        for name in ("resampler", "ranking_packing", "next_item_data", "next_item_metrics", "next_item_model", "train_next_item"):
            path = package / (name + ".py")
            content = path.read_bytes()
            _require(hashlib.sha256(content).hexdigest() == sources[name + ".py"], "source changed before frozen import")
            module = ModuleType(alias + "." + name)
            module.__file__, module.__package__ = str(path), alias
            sys.modules[module.__name__] = module
            setattr(root, name, module)
            exec(compile(content, str(path), "exec"), module.__dict__)
        return alias, root.next_item_model, root.train_next_item
    except BaseException:
        _drop_modules(alias)
        raise


def _drop_modules(alias):
    for name in list(sys.modules):
        if name == alias or name.startswith(alias + "."):
            del sys.modules[name]


def _load_model(args, payload, config, identity, model_module):
    spec, train = config["model"], config["training"]
    model = model_module.GroupRanker.from_pretrained(
        args.a_model, args.b_model, variant="V3", device="cpu", attention_backend="sdpa",
        m=1, a_readout="last_token", shared_candidate_header=True, score_readout="candidate",
        domain=config_domain(config),
        resampler_layers=spec.get("resampler_layers", 2), resampler_heads=spec.get("resampler_heads", 8),
        max_reader_packed_tokens=train["max_reader_packed_tokens"],
        max_encoder_packed_tokens=train["max_encoder_packed_tokens"], max_logical_tokens=train["max_logical_tokens"],
        gradient_checkpointing=False)
    weights = payload.get("model")
    _require(isinstance(weights, dict) and weights, "checkpoint lacks model state")
    _require(all(not isinstance(value, torch.Tensor) or not value.is_floating_point()
                 or value.dtype == torch.float32 and _finite(value) for value in weights.values()),
             "model checkpoint requires finite FP32 tensors")
    model.load_state_dict(weights, strict=True)
    _require(all(parameter.device.type == "cpu" and parameter.dtype == torch.float32
                 for parameter in model.parameters()), "model parameters must remain CPU FP32")
    metadata = model.metadata()
    template = identity.get("student_template")
    _require(isinstance(template, dict) and template.get("sha256") ==
             _digest({key: value for key, value in template.items() if key != "sha256"}),
             "frozen student template signature differs")
    actual = metadata["student_template"]
    _require(set(template) == set(actual) and all(value == actual[key] for key, value in template.items()
             if key not in {"sha256", "flex_attention_kernel"}), "model/tokenizers do not reproduce the frozen student template")
    expected_kernel = "forced_prefill" if train["attention_backend"] == "flex_attention" else None
    _require(template["flex_attention_kernel"] == expected_kernel, "frozen template training attention policy differs")
    _require(bool(model.bridge.calibrated) and metadata["bridge"]["calibration"] is not None,
             "checkpoint lacks calibrated RMS bridge state")
    _require(model.a.config.model_type == model.b.config.model_type == "qwen3", "only Qwen3 backbones are supported")
    _require(not getattr(model.b.config, "rope_scaling", None), "reader non-default RoPE is unsupported")
    metadata["student_template"] = template
    metadata["attention_backend"] = train["attention_backend"]
    metadata["packing_limits"] = {key: metadata[key] for key in
                                  ("logical_context", "reader_packed_tokens", "encoder_packed_tokens")}
    metadata["tokenizer_a"] = {"eos_token_id": model.tokenizer_a.eos_token_id,
                                "pad_token_id": model.tokenizer_a.pad_token_id}
    metadata["tokenizer_b"] = {"eos_token_id": model.tokenizer_b.eos_token_id,
                                "pad_token_id": model.tokenizer_b.pad_token_id}
    return model.eval().requires_grad_(False), metadata


def _write_reader(output, model, args):
    from safetensors.torch import save_file
    native = _json(Path(args.b_model) / "config.json")
    width, vocab = native["hidden_size"], native["vocab_size"]
    _require(type(native.get("tie_word_embeddings")) is bool, "native reader must declare embedding tying")
    _require(tuple(model.b.get_input_embeddings().weight.shape) == (vocab, width), "reader embedding/config shape differs")
    reader = output / "reader"
    reader.mkdir()
    config = model.b.config.to_dict()
    config.update(architectures=["Qwen3ForCausalLM"], use_cache=True, torch_dtype="float32")
    (reader / "config.json").write_text(json.dumps(config, sort_keys=True, indent=2) + "\n")
    weights = {"model." + key: value.detach().cpu().contiguous()
               for key, value in model.b.state_dict().items()}
    if native["tie_word_embeddings"]:
        placeholder = "tied_to_trained_input_embedding; omitted_duplicate_safetensor"
    else:
        weights["lm_head.weight"] = torch.zeros((vocab, width), dtype=torch.float32)
        placeholder = "zero_untrained_vocabulary_head"
    save_file(weights, str(reader / "model.safetensors"), metadata={"format": "pt"})
    tokenizers = {}
    for name in (*TOKENIZER_FILES, *OPTIONAL_TOKENIZER_FILES):
        path = Path(args.b_model) / name
        if name in TOKENIZER_FILES or path.exists():
            shutil.copyfile(_regular(path), reader / name)
            tokenizers[name] = _hash(reader / name)
    head = model.head.weight.detach().cpu().contiguous()
    _require(tuple(head.shape) == (1, width) and head.dtype == torch.float32 and _finite(head),
             "scalar head must be finite FP32 [1, width]")
    save_file({"head.weight": head}, str(output / "scalar_head.safetensors"))
    return native, tokenizers, placeholder


def _write_vectors(output, model, items, args):
    device = torch.device(getattr(args, "vector_device", getattr(args, "device", "cpu")))
    compute = getattr(args, "vector_dtype", "float32")
    batch_size = getattr(args, "batch_size", 32)
    _require(device.type in {"cpu", "cuda"} and compute in {"float32", "bfloat16"}, "unsupported vector device/dtype")
    _require(compute == "float32" or device.type == "cuda", "BF16 vector construction requires explicit CUDA")
    _require(type(batch_size) is int and batch_size > 0, "batch_size must be positive")
    # The frozen SDPA backend is a bounded correctness reference. Preserve its
    # exact independent-segment forward while splitting encoder packs before
    # its dense mask bound (training's larger FlexAttention cap is unsuitable).
    model.max_encoder_packed_tokens = min(model.max_encoder_packed_tokens, model.dense_reference_limit)
    ids = sorted(items)
    started = time.perf_counter()
    def progress(completed):
        print(json.dumps({"stage": "vector_export", "completed": completed,
                          "total": len(ids), "elapsed_seconds": time.perf_counter() - started},
                         sort_keys=True, allow_nan=False), flush=True)
    progress(0)
    next_progress = 8192
    for module in (model.a, model.projection, model.bridge, model.head):
        module.to(device)
    vectors = np.lib.format.open_memmap(output / "vectors.npy", mode="w+", dtype=np.float16,
                                       shape=(len(ids), model.b.config.hidden_size))
    autocast = (nullcontext() if compute == "float32" else torch.autocast(device_type="cuda", dtype=torch.bfloat16))
    with (output / "index.jsonl").open("w") as index, torch.inference_mode(), autocast:
        for start in range(0, len(ids), batch_size):
            batch = ids[start:start + batch_size]
            for item in batch:
                tokens = model._ids(model.tokenizer_a, "Item: " + items[item])
                length = len(tokens) + int(model.tokenizer_a.eos_token_id is not None)
                _require(length <= model.a.config.max_position_embeddings, "A item exceeds model context; truncation is forbidden")
            # The frozen implementation owns EOS, packed isolation, actual last
            # state, projection and calibrated RMS. No reimplemented pooling.
            states, references, _ = model._a_blocks([SimpleNamespace(history=(), candidate_ids=tuple(batch))], items)
            rows = torch.cat([states[reference] for reference in references[0][1]])
            _require(tuple(rows.shape) == (len(batch), vectors.shape[1]) and rows.dtype == torch.float32
                     and _finite(rows), "encoder returned invalid post-RMS rows")
            fp16 = rows.detach().to(device="cpu", dtype=torch.float16)
            _require(_finite(fp16), "post-RMS vectors overflowed FP16")
            vectors[start:start + len(batch)] = fp16.numpy()
            for offset, item in enumerate(batch):
                index.write(json.dumps({"item_id": item, "row": start + offset}, sort_keys=True) + "\n")
            completed = start + len(batch)
            if completed >= next_progress and completed < len(ids):
                progress(completed)
                next_progress = (completed // 8192 + 1) * 8192
    vectors.flush()
    del vectors
    progress(len(ids))
    return {"parameter_dtype": "float32", "compute_dtype": compute, "device_type": device.type,
            "autocast": compute != "float32", "tf32": False, "attention_backend": "sdpa",
            "a_readout": "last_token_including_appended_eos", "projection": "checkpoint",
            "bridge": "checkpoint_calibrated_rms", "storage_dtype": "float16",
            "encoder_pack_tokens": model.max_encoder_packed_tokens,
            "numerical_acceptance": "pending_separate_reference_and_sglang_validation"}


def export_nextbook(args):
    """Export a trusted project checkpoint; deserialize only weights-only on CPU.

    Required: source, checkpoint, data, a_model, b_model, output. ``config`` is a
    config JSON or run_manifest JSON (otherwise discovered beside the run).
    Optional vector_device=cpu, vector_dtype=float32, batch_size=32. CUDA BF16
    is opt-in only; it never changes the FP32 reader or scalar-head exports.
    """
    output = Path(args.output).absolute()
    _require(not output.exists() and not output.is_symlink(), "output exists; choose a new directory")
    for root in (args.source, args.data, args.a_model, args.b_model, Path(args.checkpoint).parent):
        _require(not output.resolve().is_relative_to(Path(root).resolve()), "output must be outside immutable inputs")
    _require(importlib.metadata.version("transformers") == "4.57.1", "requires the reference Transformers 4.57.1")
    checkpoint, receipt, identity, config, package, catalog, public, bound = _inputs(args)
    alias, module, trainer = _frozen_modules(package, identity["source_sha256"])
    old_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        trainer.validate_config(config, formal=True)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        _require(payload.get("protocol") == checkpoint_protocol(identity) and payload.get("identity") == identity,
                 "checkpoint payload protocol/identity differs")
        state = payload.get("state", {})
        _require(state.get("step") == receipt.get("step") and state.get("cursor") == receipt.get("query_cursor")
                 and _digest(state) == receipt.get("state_sha256"), "checkpoint payload state/step signature differs")
        model, metadata = _load_model(args, payload, config, identity, module)
        items = trainer._load_catalog(catalog, identity["data_sha256"]["catalog"])
        output.mkdir(parents=True, exist_ok=False)
        native, tokenizer_files, placeholder = _write_reader(output, model, args)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        encoding = _write_vectors(output, model, items, args)
        _require(all(_hash(_regular(path)) == digest for path, digest in bound.items()), "immutable inputs changed during export")
        records = {path.relative_to(output).as_posix(): _record(path)
                   for path in sorted(output.rglob("*")) if path.is_file()}
        width, count = native["hidden_size"], len(items)
        manifest = {"schema_version": 1, "kind": KIND, "variant": "V3", "m": 1,
            "checkpoint_protocol": checkpoint_protocol(identity), "checkpoint_sha256": bound[str(checkpoint)],
            "checkpoint_step": state["step"], "checkpoint_state_sha256": receipt["state_sha256"],
            "checkpoint_manifest_sha256": bound[str(checkpoint.parent / "manifest.json")],
            "checkpoint_identity": identity, "config": config, "config_sha256": identity["config_sha256"],
            "source_sha256": identity["source_sha256"], "data_sha256": identity["data_sha256"],
            "catalog_file": catalog.name, "catalog_sha256": identity["data_sha256"]["catalog"],
            "catalog_content_sha256": _digest(items), "catalog_items": count, "vector_count": count,
            "native_reader_config": native, "model_metadata": metadata, "public_models": public,
            "native_tokenizer_sha256": tokenizer_files,
            "scalar_head": {"file": "scalar_head.safetensors", "key": "head.weight", "dtype": "float32",
                            "shape": [1, width], **records["scalar_head.safetensors"]},
            "vectors": {"file": "vectors.npy", "dtype": "float16", "shape": [count, width], **records["vectors.npy"]},
            "index": {"file": "index.jsonl", **records["index.jsonl"]},
            "reader": {"directory": "reader", "backbone_dtype": "float32", "backbone_key_mapping": "b.<key> -> model.<key>",
                       "vocabulary_head": {"trained_score_head": False, "policy": placeholder, "used_for_scoring": False}},
            "encoding": encoding, "files": records, "export_source_sha256": _hash(__file__),
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "safetensors", "numpy")},
            "scope": "Frozen trained B plus separate scalar head and full-catalog single post-RMS vectors; numerical serving acceptance is separate"}
        manifest["manifest_signature"] = _digest(manifest)
        (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n")
        return verify_nextbook_export(output)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old_tf32
        _drop_modules(alias)


def verify_nextbook_export(root, verify_vectors=True):
    """Hash assets and check scalar/index/vector shapes without loading B weights."""
    from safetensors import safe_open
    from safetensors.torch import load_file
    root = Path(root)
    _require(root.is_dir() and not root.is_symlink() and type(verify_vectors) is bool, "invalid export root/options")
    manifest = _json(_regular(root / "manifest.json"))
    _require(manifest.get("manifest_signature") == _digest({key: value for key, value in manifest.items() if key != "manifest_signature"}),
             "manifest signature differs")
    _require(manifest.get("schema_version") == 1 and manifest.get("kind") == KIND
             and manifest.get("variant") == "V3" and manifest.get("m") == 1
             and manifest.get("checkpoint_protocol") == checkpoint_protocol(manifest.get("checkpoint_identity")), "unsupported next-book export")
    identity = manifest["checkpoint_identity"]
    _require(_digest(manifest["config"]) == manifest["config_sha256"] == identity["config_sha256"]
             and manifest["source_sha256"] == identity["source_sha256"]
             and manifest["data_sha256"] == identity["data_sha256"]
             and manifest["catalog_sha256"] == identity["data_sha256"]["catalog"], "manifest scientific identity differs")
    metadata = manifest["model_metadata"]
    _require(metadata["student_template"] == identity["student_template"]
             and metadata["student_template"]["sha256"] == _digest({key: value for key, value in metadata["student_template"].items() if key != "sha256"}),
             "manifest student template identity differs")
    files = manifest.get("files", {})
    required = {"reader/config.json", "reader/model.safetensors", "scalar_head.safetensors", "vectors.npy", "index.jsonl"}
    tokenizers = manifest.get("native_tokenizer_sha256", {})
    tokenizer_allowlist = set(TOKENIZER_FILES) | set(OPTIONAL_TOKENIZER_FILES)
    _require(type(tokenizers) is dict and set(TOKENIZER_FILES) <= tokenizers.keys()
             and tokenizers.keys() <= tokenizer_allowlist,
             "native tokenizer filenames must match the tokenizer allowlist")
    allowed_files = required | {"reader/" + name for name in tokenizers}
    _require(type(files) is dict and set(files) == allowed_files,
             "export inventory must contain exactly the allowed serving files")
    actual = set()
    for path in root.rglob("*"):
        _require(not path.is_symlink(), "export contains a symlink")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    _require(actual == set(files) | {"manifest.json"}, "export file inventory differs")
    for name, expected in files.items():
        relative = Path(name)
        _require(not relative.is_absolute() and all(part not in {"", ".", ".."} for part in relative.parts), "unsafe export inventory path")
        _require(_record(root / name) == expected, "export checksum/size differs: " + name)
    native = manifest["native_reader_config"]
    count, width = manifest["vector_count"], native["hidden_size"]
    _require(type(count) is int and count > 0 and manifest["catalog_items"] == count
             and type(width) is int and width > 0 and native["model_type"] == "qwen3", "invalid model/vector dimensions")
    for field, filename in (("scalar_head", "scalar_head.safetensors"), ("vectors", "vectors.npy"), ("index", "index.jsonl")):
        _require(manifest[field]["file"] == filename and all(manifest[field][key] == files[filename][key] for key in ("sha256", "bytes")),
                 "asset record/inventory differs: " + field)
    _require(manifest["scalar_head"]["key"] == "head.weight" and manifest["scalar_head"]["shape"] == [1, width]
             and manifest["scalar_head"]["dtype"] == "float32" and manifest["vectors"]["shape"] == [count, width]
             and manifest["vectors"]["dtype"] == "float16", "asset shape/dtype metadata differs")
    with safe_open(str(root / "reader/model.safetensors"), framework="pt", device="cpu") as tensors:
        keys = set(tensors.keys())
        _require("model.embed_tokens.weight" in keys and "model.norm.weight" in keys
                 and all(key.startswith("model.") or key == "lm_head.weight" for key in keys)
                 and all(tensors.get_slice(key).get_dtype() == "F32" for key in keys),
                 "reader safetensors must contain an FP32 backbone")
        _require(tensors.get_slice("model.embed_tokens.weight").get_shape() == [native["vocab_size"], width],
                 "reader vocabulary tensor shape differs")
        if not native["tie_word_embeddings"]:
            _require("lm_head.weight" in keys and tensors.get_slice("lm_head.weight").get_shape()
                     == [native["vocab_size"], width], "untied reader lacks its placeholder vocabulary head")
    head = load_file(str(root / "scalar_head.safetensors"), device="cpu")
    _require(set(head) == {"head.weight"} and tuple(head["head.weight"].shape) == (1, width)
             and head["head.weight"].dtype == torch.float32 and _finite(head["head.weight"]), "invalid scalar head tensor")
    previous, rows = None, 0
    with (root / "index.jsonl").open() as stream:
        for line in stream:
            entry = json.loads(line)
            item = entry.get("item_id")
            _require(set(entry) == {"item_id", "row"} and type(item) is str and item
                     and (previous is None or item > previous) and type(entry["row"]) is int and entry["row"] == rows,
                     "index must contain sorted unique items and contiguous rows")
            previous, rows = item, rows + 1
    _require(rows == count, "index row count differs from catalog")
    reader_config = _json(root / "reader/config.json")
    _require(all(reader_config.get(key) == native.get(key) for key in
                 ("model_type", "vocab_size", "hidden_size", "tie_word_embeddings", "max_position_embeddings"))
             and reader_config.get("architectures") == ["Qwen3ForCausalLM"]
             and manifest["reader"]["vocabulary_head"]["trained_score_head"] is False
             and manifest["reader"]["vocabulary_head"]["used_for_scoring"] is False,
             "reader architecture/placeholder-head semantics differ")
    for name, digest in tokenizers.items():
        _require(files.get("reader/" + name, {}).get("sha256") == digest
                 == manifest["public_models"]["b"]["files"][name]["sha256"], "tokenizer bytes differ from public snapshot")
    if verify_vectors:
        vectors = np.load(root / "vectors.npy", mmap_mode="r", allow_pickle=False)
        _require(vectors.dtype == np.float16 and vectors.shape == (count, width), "vector array dtype/shape differs")
        for start in range(0, count, 8192):
            _require(bool(np.isfinite(vectors[start:start + 8192]).all()), "nonfinite offline vector")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "checkpoint", "data", "a-model", "b-model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--vector-device", default="cpu")
    parser.add_argument("--vector-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    manifest = export_nextbook(args)
    print(json.dumps({"output": str(Path(args.output).absolute()), "manifest_signature": manifest["manifest_signature"],
                      "checkpoint_sha256": manifest["checkpoint_sha256"], "checkpoint_step": manifest["checkpoint_step"],
                      "vector_count": manifest["vector_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
