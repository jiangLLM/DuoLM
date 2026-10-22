"""Portable, weights-only DuoLM inference artifacts; separate from full resume.

Exports original trained tensors without optimizer state or private run config.
Loading is offline and uses bundled public architecture/tokenizer files, never
base-model weight downloads or historical source execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import torch
from safetensors.torch import load_file, save_file

from .assets import MODEL_ASSETS
from .next_item_model import GroupRanker, domain_template

SCHEMA = "duolm-inference-v1"
_ASSETS = {row["name"]: row for row in MODEL_ASSETS}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
DOCUMENT_FILES = {"README.md", "LICENSE", "NOTICE"}


def _require(condition, message):
    if not condition:
        raise ValueError("DuoLM inference checkpoint: " + message)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _hex(value, length=64):
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def _regular(path):
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), f"missing or symlink file: {path.name}")
    return path


def _record(path):
    path = _regular(path)
    return {"sha256": _hash(path), "bytes": path.stat().st_size}


def _json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(_regular(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _canonical_spec(spec):
    repo, revision = spec.get("repo"), spec.get("revision")
    _require(isinstance(repo, str) and _hex(revision, 40), "base model needs a repository and immutable revision")
    # Some historical configs stored the HF snapshot path as repo itself.
    path = Path(repo)
    if path.parent.name == "snapshots" and path.parent.parent.name.startswith("models--"):
        _require(path.name == revision, "snapshot path revision differs")
        repo = path.parent.parent.name.removeprefix("models--").replace("--", "/")
    _require(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) is not None, "base repository must be a public repository ID")
    return {"repo": repo, "revision": revision}


def _architecture(config):
    schema = config.get("schema_version")
    _require(schema in {"books-next-item-v1", "books-candidate-ranking-v1", "games-candidate-ranking-v1"}, "unsupported original data schema")
    spec, training = config["model"], config["training"]
    variant = spec.get("variant")
    _require(variant in {"T", "V3"}, "initial inference format supports full-text T and compact V3")
    domain = "game" if schema == "games-candidate-ranking-v1" else "book"
    _require(spec.get("domain", domain) == domain, "original model domain differs from schema")
    if variant == "V3":
        _require(spec.get("m") == 1 and spec.get("a_readout") == "last_token"
                 and spec.get("shared_candidate_header") is True and spec.get("score_readout") == "candidate",
                 "V3 export requires the main single-vector LastToken interface")
    else:
        _require(spec.get("a_readout", "perceiver") == "perceiver"
                 and spec.get("score_readout", "legacy") == "legacy"
                 and spec.get("shared_candidate_header", False) is False, "unsupported teacher interface")
    result = {"variant": variant, "domain": domain, "m": spec.get("m", 8),
              "a_readout": spec.get("a_readout", "perceiver"),
              "shared_candidate_header": spec.get("shared_candidate_header", False),
              "score_readout": spec.get("score_readout", "legacy"),
              "resampler_layers": spec.get("resampler_layers", 2),
              "resampler_heads": spec.get("resampler_heads", 8),
              "input_assembly": spec.get("input_assembly", "legacy"),
              **{key: training[key] for key in ("max_reader_packed_tokens", "max_encoder_packed_tokens", "max_logical_tokens")}}
    return result


def _validate_architecture(architecture):
    expected = {"variant", "domain", "m", "a_readout", "shared_candidate_header", "score_readout",
                "resampler_layers", "resampler_heads", "input_assembly", "max_reader_packed_tokens",
                "max_encoder_packed_tokens", "max_logical_tokens"}
    _require(isinstance(architecture, dict) and set(architecture) == expected, "architecture field allowlist differs")
    _require(architecture["domain"] in {"book", "game"}, "unsupported domain")
    for key in expected - {"variant", "domain", "a_readout", "shared_candidate_header", "score_readout", "input_assembly"}:
        _require(type(architecture[key]) is int and architecture[key] > 0, f"invalid architecture integer: {key}")
    _require(type(architecture["shared_candidate_header"]) is bool, "shared header must be boolean")
    _require(architecture["input_assembly"] in {"legacy", "batched"}, "invalid input assembly")
    if architecture["variant"] == "T":
        _require(architecture["a_readout"] == "perceiver" and architecture["score_readout"] == "legacy"
                 and not architecture["shared_candidate_header"], "teacher template differs")
    else:
        _require(architecture["variant"] == "V3" and architecture["m"] == 1
                 and architecture["a_readout"] == "last_token" and architecture["score_readout"] == "candidate"
                 and architecture["shared_candidate_header"], "unsupported student template")


def _extra_state(weights, variant):
    non_tensors = {key: value for key, value in weights.items() if not isinstance(value, torch.Tensor)}
    if variant == "T":
        _require(not non_tensors, "teacher must contain tensor-only model state")
        return {}
    _require(set(non_tensors) == {"bridge._extra_state"}, "student has unexpected or missing non-tensor state")
    extra = non_tensors["bridge._extra_state"]
    _require(isinstance(extra, dict) and set(extra) == {"calibration"}, "invalid bridge extra state")
    calibration = extra["calibration"]
    _require(isinstance(calibration, dict) and set(calibration) == {"source", "items", "content_hash"}
             and calibration["source"] == "training_query_content"
             and type(calibration["items"]) is int and 0 < calibration["items"] <= 256
             and _hex(calibration["content_hash"]), "bridge calibration is missing or contains non-allowlisted metadata")
    return json.loads(json.dumps(non_tensors))


def _construct(root, architecture, backend):
    from transformers import AutoModel, AutoTokenizer, Qwen3Config
    _validate_architecture(architecture)
    _require(backend in {"sdpa", "flex_attention"}, "unsupported attention backend")
    modules, tokenizers = {}, {}
    for side in (["b"] if architecture["variant"] == "T" else ["a", "b"]):
        native = _json(root / side / "config.json")
        _require(native.get("model_type") == "qwen3" and not native.get("auto_map"), "only native Qwen3 architecture is supported")
        config = Qwen3Config.from_dict(native)
        # Meta construction avoids allocating or downloading an untrained copy
        # of the two base models. Strict assign installs the trained tensors.
        with torch.device("meta"):
            modules[side] = AutoModel.from_config(config, dtype=torch.float32,
                attn_implementation=backend, trust_remote_code=False)
        # Rotary frequencies are deterministic nonpersistent buffers: they do
        # not appear in state_dict and must be materialized from public config.
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
        modules[side].rotary_emb = Qwen3RotaryEmbedding(config, device="cpu")
        tokenizers[side] = AutoTokenizer.from_pretrained(str(root / side), local_files_only=True, trust_remote_code=False)
    with torch.device("meta"):
        model = GroupRanker(modules.get("a"), modules["b"], tokenizers.get("a"), tokenizers["b"],
            attention_backend=backend, gradient_checkpointing=False, reader_gradient_checkpointing=False,
            **architecture)
    return model


def _load_weights(root, architecture, *, backend="sdpa"):
    model = _construct(root, architecture, backend)
    tensors = load_file(str(root / "model.safetensors"), device="cpu")
    extras = _json(root / "extra-state.json")
    _require(isinstance(extras, dict) and not (set(extras) & set(tensors)), "extra state collides with tensor keys")
    state = {**tensors, **extras}
    _extra_state(state, architecture["variant"])
    expected = model.state_dict()
    _require(set(state) == set(expected), "tensor/extra-state key set differs from architecture")
    for key, value in tensors.items():
        reference = expected[key]
        _require(isinstance(reference, torch.Tensor) and value.shape == reference.shape and value.dtype == reference.dtype,
                 f"tensor shape or dtype differs: {key}")
        if value.is_floating_point():
            _require(value.dtype == torch.float32 and bool(torch.isfinite(value).all()), f"nonfinite or non-FP32 weight: {key}")
    if architecture["variant"] != "T":
        _require(bool(tensors["bridge.calibrated"]) and float(tensors["bridge.rms_reference"]) > 0
                 and bool(torch.isfinite(tensors["bridge.log_gain"].exp())), "invalid calibrated bridge buffers")
    model.load_state_dict(state, strict=True, assign=True)
    _require(not any(value.is_meta for value in (*model.parameters(), *model.buffers())), "model has unfilled meta parameters or buffers")
    return model.eval().requires_grad_(False)


def _template(model):
    if model.variant == "T":
        return {"domain_prompt": domain_template(model.domain), "variant": "T", "score_readout": "last_suffix_token"}
    value = dict(model.metadata()["student_template"])
    # Backend is an explicit runtime choice, not part of semantic tokenization.
    value.pop("sha256")
    value.pop("flex_attention_kernel")
    return value


def _inventory(root):
    files = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        # HF local_dir may contain only downloader bookkeeping under .cache.
        if relative.parts[0] == ".cache":
            continue
        _require(not path.is_symlink(), "artifact contains a symlink")
        if relative.as_posix() == ".gitattributes" and path.is_file():
            continue
        if path.is_file():
            files.add(relative.as_posix())
    return files


def verify_checkpoint(root, *, load_tensors=True):
    """Verify exact artifact inventory and, by default, all model tensors."""
    root = Path(root)
    _require(root.is_dir() and not root.is_symlink(), "artifact directory is missing or a symlink")
    manifest = _json(root / "manifest.json")
    _require(set(manifest) == {"schema", "architecture", "template", "public_models", "recommended_compute_dtype",
                              "files", "manifest_sha256"} and manifest["schema"] == SCHEMA, "unsupported inference manifest")
    _require(manifest["manifest_sha256"] == _digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}), "manifest checksum differs")
    architecture = manifest["architecture"]
    _validate_architecture(architecture)
    _require(manifest["recommended_compute_dtype"] in {"float32", "bfloat16"}, "unsupported recommended compute dtype")
    sides = {"b"} if architecture["variant"] == "T" else {"a", "b"}
    _require(isinstance(manifest["public_models"], dict) and set(manifest["public_models"]) == sides, "base model roles differ")
    for side, spec in manifest["public_models"].items():
        _require(set(spec) == {"repo", "revision"} and _canonical_spec(spec) == spec, "invalid public base model identity")
        asset = _ASSETS["encoder" if side == "a" else "reader"]
        _require(spec == {"repo": asset["repo"], "revision": asset["revision"]}, "unsupported pinned public model")
    required = {"model.safetensors", "extra-state.json", "provenance.json"} | {
        f"{side}/{name}" for side in sides for name in ("config.json", "tokenizer.json", "tokenizer_config.json")}
    allowed = required | DOCUMENT_FILES | {f"{side}/{name}" for side in sides for name in TOKENIZER_FILES}
    records = manifest["files"]
    _require(isinstance(records, dict) and required <= set(records) <= allowed, "artifact file allowlist differs")
    _require(_inventory(root) == set(records) | {"manifest.json"}, "unexpected or missing artifact files")
    for name, record in records.items():
        _require(isinstance(record, dict) and set(record) == {"sha256", "bytes"}
                 and _hex(record["sha256"]) and type(record["bytes"]) is int and record["bytes"] > 0,
                 "invalid artifact file record")
        _require(_record(root / name) == record, "artifact file checksum differs: " + name)
        if "/" in name:
            side, filename = name.split("/")
            asset = _ASSETS["encoder" if side == "a" else "reader"]
            _require(record["sha256"] == asset["files"].get(filename), "pinned public metadata checksum differs: " + name)
    provenance = _json(root / "provenance.json")
    keys = {"checkpoint_sha256", "checkpoint_manifest_sha256", "original_config_sha256", "original_identity_sha256",
            "original_data_identity_sha256", "original_source_sha256", "checkpoint_step", "conversion_source_sha256",
            "historical_behavior_verified", "full_resume_supported"}
    _require(isinstance(provenance, dict) and set(provenance) == keys, "provenance field allowlist differs")
    for key in keys - {"original_source_sha256", "checkpoint_step", "historical_behavior_verified", "full_resume_supported"}:
        _require(_hex(provenance[key]), "invalid provenance hash")
    _require(type(provenance["checkpoint_step"]) is int and provenance["checkpoint_step"] > 0
             and provenance["historical_behavior_verified"] is False and provenance["full_resume_supported"] is False,
             "inference artifact cannot claim full resume or unchecked historic equivalence")
    sources = provenance["original_source_sha256"]
    _require(isinstance(sources, dict) and sources and all(re.fullmatch(r"[A-Za-z0-9_]+\.py", k) and _hex(v)
             for k, v in sources.items()), "source provenance must contain only source basenames and hashes")
    if load_tensors:
        model = _load_weights(root, architecture)
        _require(_template(model) == manifest["template"], "model tokenization/template differs from artifact")
    return manifest


def load_ranker(root, *, device="cpu", attention_backend="sdpa"):
    """Load an ordinary GroupRanker offline; call model(queries, items) to rank."""
    root = Path(root)
    manifest = verify_checkpoint(root, load_tensors=False)
    model = _load_weights(root, manifest["architecture"], backend=attention_backend)
    _require(_template(model) == manifest["template"], "model tokenization/template differs from artifact")
    return model.to(device)


def _copy_assets(path, output, spec, side):
    asset = _ASSETS["encoder" if side == "a" else "reader"]
    expected = {"repo": asset["repo"], "revision": asset["revision"]}
    _require(_canonical_spec(spec) == expected, "original base model is not the supported pinned public revision")
    path = Path(path)
    names = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    names += [name for name in ("vocab.json", "merges.txt") if (path / name).exists()]
    destination = output / side
    destination.mkdir()
    for name in names:
        original = _regular(path / name)
        _require(_hash(original) == asset["files"].get(name), "public model metadata checksum differs: " + side + "/" + name)
        shutil.copyfile(original, destination / name)
        _require(_hash(destination / name) == asset["files"][name], "model metadata changed while copying")
    return expected


def export_checkpoint(checkpoint, config, output, *, b_model, a_model=None,
                      expected_checkpoint_sha256=None, model_card=None, license_file=None, notice_file=None):
    """Convert a trusted owned training checkpoint, never its optimizer/config.

    Original source/data digests remain provenance; they are not replaced with
    public source hashes. Actual historical behavioral equivalence is separate.
    """
    checkpoint, output = Path(checkpoint), Path(output)
    if checkpoint.is_dir():
        checkpoint /= "training.pt"
    checkpoint = _regular(checkpoint)
    receipt_path = checkpoint.parent / "manifest.json"
    receipt = _json(receipt_path)
    identity = receipt.get("identity")
    _require(isinstance(identity, dict), "training checkpoint has no identity")
    original_config = _json(config)
    if "config" in original_config:
        _require(original_config.get("identity") == identity, "run manifest and checkpoint identity differ")
        original_config = original_config["config"]
    _require(_digest(original_config) == identity.get("config_sha256"), "original configuration hash differs")
    record = _record(checkpoint)
    _require(record == receipt.get("files", {}).get("training.pt"), "training checkpoint checksum differs")
    if expected_checkpoint_sha256 is not None:
        _require(_hex(expected_checkpoint_sha256) and record["sha256"] == expected_checkpoint_sha256, "authorized checkpoint SHA256 differs")
    architecture = _architecture(original_config)
    _validate_architecture(architecture)
    protocol = "games-next-item-training-v1" if architecture["domain"] == "game" else "books-next-item-training-v1"
    _require(receipt.get("protocol") == protocol, "checkpoint protocol/domain differs")
    if architecture["domain"] == "game":
        prompt = {**domain_template("game"), "sha256": _digest(domain_template("game"))}
        _require(identity.get("domain") == "game" and identity.get("prompt_template") == prompt,
                 "original Games prompt/domain identity differs")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state = payload.get("state", {})
    _require(payload.get("identity") == identity and payload.get("protocol") == protocol
             and _digest(state) == receipt.get("state_sha256") and state.get("step") == receipt.get("step")
             and state.get("cursor") == receipt.get("query_cursor")
             and type(state.get("step")) is int and state["step"] > 0, "original checkpoint state/identity mismatch")
    weights = payload.get("model")
    _require(isinstance(weights, dict) and weights, "original checkpoint has no model state")
    extras = _extra_state(weights, architecture["variant"])
    _require(not output.exists() and not output.is_symlink(), "output already exists; use a new directory")
    _require(a_model is not None if architecture["variant"] == "V3" else a_model is None,
             "V3 requires A metadata; T must not supply A metadata")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".duolm-export-", dir=output.parent))
    try:
        public = {"b": _copy_assets(b_model, staging, original_config["model"]["b_model"], "b")}
        if architecture["variant"] != "T":
            public["a"] = _copy_assets(a_model, staging, original_config["model"]["a_model"], "a")
        # Save only tensor state. Clone aliases (including tied embeddings) so
        # safetensors has an unambiguous complete original key inventory.
        tensors, storages = {}, set()
        for key, value in weights.items():
            if not isinstance(value, torch.Tensor):
                continue
            value = value.detach().cpu().contiguous()
            storage = value.untyped_storage().data_ptr()
            if storage in storages:
                value = value.clone()
            storages.add(storage)
            tensors[key] = value
        save_file(tensors, str(staging / "model.safetensors"), metadata={"format": "pt"})
        del tensors
        _write_json(staging / "extra-state.json", extras)
        model = _load_weights(staging, architecture)
        if architecture["variant"] != "T":
            original_template = identity.get("student_template")
            _require(isinstance(original_template, dict), "original student template identity is missing")
            original_template = dict(original_template)
            original_signature = original_template.pop("sha256", None)
            _require(original_signature == _digest(original_template), "original student template checksum differs")
            _require(original_template.pop("flex_attention_kernel", None) in {None, "forced_prefill"},
                     "original student attention kernel is unsupported")
            _require(original_template == _template(model), "original student template differs from public architecture")
        # Direct equality includes integer/bool buffers, not just floating weights.
        actual = model.state_dict()
        _require(set(actual) == set(weights), "round-trip state key set changed")
        for key, value in weights.items():
            _require(torch.equal(value, actual[key]) if isinstance(value, torch.Tensor) else value == actual[key],
                     "round-trip state differs: " + key)
        provenance = {"checkpoint_sha256": record["sha256"], "checkpoint_manifest_sha256": _hash(receipt_path),
            "checkpoint_step": state["step"], "original_config_sha256": identity["config_sha256"],
            "original_identity_sha256": _digest(identity), "original_data_identity_sha256": _digest(identity.get("data_sha256", {})),
            "original_source_sha256": identity.get("source_sha256", {}), "conversion_source_sha256": _hash(__file__),
            "historical_behavior_verified": False, "full_resume_supported": False}
        _write_json(staging / "provenance.json", provenance)
        for source, name in ((model_card, "README.md"), (license_file, "LICENSE"), (notice_file, "NOTICE")):
            if source is not None:
                raw = _regular(source).read_bytes()
                raw.decode("utf-8")
                _require(len(raw) > 0, "documentation must not be empty")
                (staging / name).write_bytes(raw)
        manifest = {"schema": SCHEMA, "architecture": architecture, "template": _template(model),
            "public_models": public, "recommended_compute_dtype": original_config.get("evaluation", {}).get("compute_dtype", "float32"),
            "files": {name: _record(staging / name) for name in sorted(_inventory(staging))}}
        manifest["manifest_sha256"] = _digest(manifest)
        _write_json(staging / "manifest.json", manifest)
        verify_checkpoint(staging, load_tensors=False)
        _require(_record(checkpoint) == record and _json(receipt_path) == receipt,
                 "original checkpoint changed during export")
        _require(not output.exists() and not output.is_symlink(), "output appeared during export")
        os.rename(staging, output)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    for name in ("checkpoint", "config", "b-model", "output"):
        export.add_argument("--" + name, type=Path, required=True)
    export.add_argument("--a-model", type=Path)
    export.add_argument("--expected-checkpoint-sha256")
    export.add_argument("--model-card", type=Path)
    export.add_argument("--license-file", type=Path)
    export.add_argument("--notice-file", type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "verify":
        result = verify_checkpoint(args.artifact)
    else:
        values = vars(args).copy()
        values.pop("command")
        result = export_checkpoint(**values)
    print(json.dumps({"schema": result["schema"], "manifest_sha256": result["manifest_sha256"],
                      "variant": result["architecture"]["variant"], "domain": result["architecture"]["domain"]}, sort_keys=True))


if __name__ == "__main__":
    main()
