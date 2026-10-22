"""Evaluate a saved DuoLM checkpoint on fixed development users.

The selector is the hash-defined development subset used during training.
The report subset is its disjoint development complement. This command never
opens test labels or changes checkpoint selection, and restores weights only.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from . import train_next_item as trainer
from .candidate_teacher import _operation, _record
from .next_item_model import GroupRanker, domain_template


def _model(config, device, model_factory=None):
    if model_factory is not None:
        return model_factory(config, device)
    spec, runtime = config["model"], config["training"]
    a, b = spec.get("a_model", {}), spec["b_model"]
    for snapshot in ([b] if spec["variant"] == "T" else [a, b]):
        trainer.validate_local_snapshot(snapshot)
    return GroupRanker.from_pretrained(
        a.get("local_path", a.get("repo")), b.get("local_path", b["repo"]),
        a_revision=a.get("revision"), b_revision=b["revision"],
        variant=spec["variant"], device=device, domain=trainer.config_domain(config),
        attention_backend=runtime["attention_backend"],
        m=spec.get("m", 8), resampler_layers=spec.get("resampler_layers", 2),
        resampler_heads=spec.get("resampler_heads", 8),
        a_readout=spec.get("a_readout", "perceiver"),
        shared_candidate_header=spec.get("shared_candidate_header", False),
        score_readout=spec.get("score_readout", "legacy"),
        max_reader_packed_tokens=config["evaluation"]["max_reader_packed_tokens"],
        max_encoder_packed_tokens=runtime["max_encoder_packed_tokens"],
        max_logical_tokens=runtime["max_logical_tokens"],
        gradient_checkpointing=False, reader_gradient_checkpointing=False,
        input_assembly=spec.get("input_assembly", "legacy"))


def evaluate_checkpoint(checkpoint_dir, config, data_dir, output, *, device="cuda",
                        subset="report", formal=True, model_factory=None):
    if subset not in {"selector", "dev", "report"}:
        raise ValueError("subset must be selector, dev, or report; test evaluation is not supported here")
    trainer.validate_config(config, formal=formal, require_teacher_cache=False)
    checkpoint_dir = Path(checkpoint_dir)
    manifest = json.loads((checkpoint_dir / "manifest.json").read_text())
    identity = manifest.get("identity", {})
    if identity.get("config_sha256") != trainer._digest(config):
        raise ValueError("checkpoint config differs; use the run's resolved-config.json")
    trainer._checkpoint_manifest(checkpoint_dir, identity=identity)
    _, dev, items, cohorts, hashes = trainer.load_dataset(data_dir, config, formal=formal)
    if hashes != identity.get("data_sha256"):
        raise ValueError("evaluation data differ from the checkpoint's frozen dataset")
    sources = identity.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("checkpoint is missing its source identity")
    for name, digest in sources.items():
        if Path(name).name != name or trainer._file_hash(Path(trainer.__file__).parent / name) != digest:
            raise ValueError("checkpoint source differs from this release; use matching source or an explicit weights export")
    if trainer._evaluation_source_files(config) != identity.get("evaluation_source_files", {}):
        raise ValueError("checkpoint evaluation source identity differs from this release")
    payload = torch.load(checkpoint_dir / "training.pt", map_location="cpu", weights_only=True, mmap=True)
    if (payload.get("identity") != identity or payload.get("protocol") != trainer.training_protocol(identity)
            or trainer._digest(payload.get("state")) != manifest["state_sha256"]
            or payload["state"].get("step") != manifest["step"]
            or payload["state"].get("cursor") != manifest["query_cursor"]):
        raise ValueError("checkpoint payload identity or signed state mismatch")
    selector = set(cohorts["checkpoint_selector"])
    users = tuple(cohorts["checkpoint_selector"] if subset == "selector" else
                  cohorts["dev"] if subset == "dev" else [u for u in cohorts["dev"] if u not in selector])
    if not users:
        raise ValueError("evaluation subset has no users")
    wanted = set(users)
    queries = [q for q in dev if q.user_id in wanted]
    unavailable = tuple(u for u in cohorts["unavailable_dev_users"] if u in wanted)
    operation = {"checkpoint_sha256": manifest["files"]["training.pt"]["sha256"],
                 "checkpoint_identity_sha256": trainer._digest(identity), "data_sha256": hashes,
                 "subset": subset, "device_type": torch.device(device).type,
                 "source_sha256": trainer._file_hash(__file__),
                 "compute_dtype": trainer.inference_compute_dtype(config)}
    with _operation(output, phase="student-evaluation", identity=operation) as output:
        # Recompute rather than trusting cached metrics; output files are atomic.
        model = _model(config, device, model_factory)
        model.load_state_dict(payload["model"], strict=True)
        if config["model"]["variant"] != "T" and hasattr(model, "metadata"):
            if identity.get("student_template") != model.metadata()["student_template"]:
                raise ValueError("student checkpoint template differs from the instantiated model")
        if trainer.config_domain(config) == "game":
            expected_prompt = {**domain_template("game"), "sha256": trainer._digest(domain_template("game"))}
            if identity.get("domain") != "game" or identity.get("prompt_template") != expected_prompt:
                raise ValueError("checkpoint Games domain or prompt identity mismatch")
        model.eval()
        prefix = trainer._TeacherPrefixEvaluation(model) if trainer._evaluation_source_files(config) else None
        options = {"strict_math": True} if trainer._evaluation_execution_metadata(config) else {}
        if prefix is not None:
            options["prefix_scorer"] = prefix
        temporary = output / ".predictions.jsonl.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            def sink(row):
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            metrics = trainer.evaluate(
                model, queries, items, expected_users=users, device=device,
                compute_dtype=trainer.inference_compute_dtype(config),
                max_reader_packed_tokens=config["evaluation"]["max_reader_packed_tokens"],
                microbatch_queries=config["evaluation"].get("microbatch_queries", config["training"]["microbatch_queries"]),
                unavailable_users=unavailable, evaluation_scope=config["schema_version"],
                prediction_sink=sink, **options)
            stream.flush()
            os.fsync(stream.fileno())
        if _record(checkpoint_dir / "training.pt") != manifest["files"]["training.pt"]:
            raise ValueError("checkpoint changed during evaluation")
        os.replace(temporary, output / "predictions.jsonl")
        report = {"schema": "duolm-checkpoint-evaluation-v1", "completed": True,
                  "identity": operation, "subset": subset, "users": len(users), "queries": len(queries),
                  "metrics": metrics, "metrics_sha256": trainer._digest(metrics),
                  "checkpoint_step": manifest["step"], "weights_only": True,
                  "selects_checkpoint": False, "test_labels_accessed": False,
                  "files": {"predictions.jsonl": _record(output / "predictions.jsonl")}}
        trainer._atomic_json(output / "report.json", report)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="resolved-config.json from training")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--subset", choices=("selector", "dev", "report"), default="report")
    args = parser.parse_args(argv)
    report = evaluate_checkpoint(args.checkpoint_dir, json.loads(args.config.read_text()),
        args.data_dir, args.output, device=args.device, subset=args.subset)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
