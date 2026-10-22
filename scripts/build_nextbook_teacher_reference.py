"""Independent causal T4000 references: FP16 B/FP32 head plus FP32 diagnostic.

Each candidate is a separate causal sequence with a repeated history prefix.
No MIS input, layout, attention patch, or shared-prefix cache is used. Variable
length sequences use right padding, an explicit mask and actual suffix endpoint.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import uuid

from safetensors.torch import load_file
import torch
from torch.nn import functional as F
from transformers import AutoModel

from duolm.nextbook_benchmark import digest, numerical_check, verify_workload
from duolm.nextbook_teacher_export import verify_nextbook_teacher_export
from duolm.nextbook_teacher_inputs import NextBookTeacherCompiler


def cast_half_preserve_rotary(model):
    rotary = {name: value.detach().clone() for name, value in model.named_buffers()
              if "rotary_emb" in name and value.is_floating_point()}
    model.half()
    for name, value in rotary.items():
        parent, _, attribute = name.rpartition(".")
        setattr(model.get_submodule(parent), attribute, value)
    return model


@torch.inference_mode()
def independent_scores(model, head, prepared, *, token_budget=32768, max_batch_size=32, pad_token_id=0):
    """Score complete causal sequences only; all results retain candidate order."""
    device = next(model.parameters()).device
    if head.dtype != torch.float32 or head.shape != (1, model.config.hidden_size):
        raise ValueError("Independent teacher reference requires original FP32 scalar head")
    if not 0 <= pad_token_id < model.config.vocab_size:
        raise ValueError("Invalid native padding ID")
    sequences = prepared.independent_sequences()
    scores = []
    for indices in prepared.independent_causal_batches(token_budget, max_batch_size):
        lengths = [len(sequences[index]) for index in indices]
        longest = max(lengths)
        ids = torch.full((len(indices), longest), pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        positions = torch.zeros_like(ids)
        for row, index in enumerate(indices):
            length = lengths[row]
            ids[row, :length] = torch.tensor(sequences[index], device=device)
            mask[row, :length] = 1
            positions[row, :length] = torch.arange(length, device=device)
        hidden = model(input_ids=ids, attention_mask=mask, position_ids=positions,
                       use_cache=False, return_dict=True).last_hidden_state
        endpoint = hidden[torch.arange(len(indices), device=device), torch.tensor(lengths, device=device) - 1]
        with torch.autocast(device_type=device.type, enabled=False):
            values = F.linear(endpoint.float(), head).squeeze(-1)
        if not torch.isfinite(values).all():
            raise ValueError("Nonfinite independent teacher scalar score")
        scores.extend(values.cpu().tolist())
    if len(scores) != len(prepared.candidate_ids):
        raise ValueError("Independent teacher reference lost candidate scores")
    return scores


def _verify_workload_source(args, export, workload):
    if workload["dev_sha256"] == export["data_sha256"]["dev"]:
        return {"source": "original_teacher_dev", "dev_sha256": workload["dev_sha256"]}
    # The comparison uses a separately prepared candidate100 dev file. It is
    # bound by the verified V3 export while its text must be the teacher catalog.
    path = getattr(args, "workload_export", None)
    if path is None:
        raise ValueError("Different candidate dev split requires --workload-export provenance")
    from duolm.nextbook_export import verify_nextbook_export
    peer = verify_nextbook_export(path, verify_vectors=False)
    if workload["dev_sha256"] != peer["data_sha256"]["dev"] or peer["catalog_content_sha256"] != export["catalog_content_sha256"]:
        raise ValueError("Comparison workload dev/catalog differs from verified peer export")
    return {"source": "verified_candidate_dev_export", "dev_sha256": workload["dev_sha256"],
            "peer_export_signature": peer["manifest_signature"], "catalog_content_sha256": peer["catalog_content_sha256"]}


def select_requests(workload, candidate_count=None, fp32_groups=None):
    """Choose coverage by workload order only, never by scores or labels."""
    if candidate_count is not None and (type(candidate_count) is not int or candidate_count not in (10, 50, 100)):
        raise ValueError("candidate_count must be 10, 50, 100 or None")
    rows = [row for row in workload["requests"]
            if candidate_count is None or len(row["query"]["candidate_ids"]) == candidate_count]
    if not rows:
        raise ValueError("No workload requests match the candidate count")
    if fp32_groups is not None and (type(fp32_groups) is not int or not 0 <= fp32_groups <= len(rows)):
        raise ValueError("fp32_groups must be between zero and the selected request count")
    return rows, rows if fp32_groups is None else rows[:fp32_groups]


def _write_progress(path, value, *, exclusive=False):
    """Publish a complete JSON snapshot atomically, without resuming old runs."""
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, allow_nan=False, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            # Atomic initial publication must fail if another/old run owns it.
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_reader(export_root, device):
    # Reload original saved FP32 weights for the diagnostic. Converting the
    # serving model back with .float() would retain irreversible FP16 rounding.
    return AutoModel.from_pretrained(Path(export_root) / "reader", torch_dtype=torch.float32,
                                     attn_implementation="sdpa", local_files_only=True,
                                     trust_remote_code=False).to(device).eval()


@torch.inference_mode()
def build(args):
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    export = verify_nextbook_teacher_export(args.export)
    workload = verify_workload(json.loads(Path(args.workload).read_text()))
    provenance = _verify_workload_source(args, export, workload)
    rows, fp32_rows = select_requests(workload, getattr(args, "candidate_count", None), getattr(args, "fp32_groups", None))
    output = Path(args.output)
    progress_path = output.with_name(output.stem + ".progress.json")
    if output.exists() or progress_path.exists():
        raise ValueError("Teacher reference output/progress already exists; old results are not resumed")
    output.parent.mkdir(parents=True, exist_ok=True)
    compiler = NextBookTeacherCompiler(args.export)
    device = torch.device(getattr(args, "device", "cuda"))
    token_budget = getattr(args, "token_budget", 32768)
    batch_size = getattr(args, "batch_size", 32)
    if type(token_budget) is not int or token_budget < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("Positive token budget and batch size are required")
    pad = compiler._tokenizer.pad_token_id
    if pad is None:
        pad = 0
    references, completed_precisions = {}, []
    coverage = {"requested_groups": getattr(args, "fp32_groups", None), "selected_groups": len(rows),
                "diagnostic_groups": len(fp32_rows),
                "selection": "all_selected" if len(fp32_rows) == len(rows) else ("fixed_first_n" if fp32_rows else "skipped")}
    base = {"protocol": "mixlm-nextbook-teacher-reference-v1", "variant": "T",
            "export_signature": export["manifest_signature"], "checkpoint_sha256": export["checkpoint_sha256"],
            # Keep the parent workload identity even when only N100 is selected.
            "workload_sha256": workload["workload_sha256"], "workload_provenance": provenance,
            "selected_candidate_counts": sorted({len(row["query"]["candidate_ids"]) for row in rows}),
            "reference_request_ids": [row["request_id"] for row in rows],
            "fp32_reference_request_ids": [row["request_id"] for row in fp32_rows], "fp32_coverage": coverage,
            "independent_causal_sequences": True, "prefix_cache": False,
            "score_head_dtype": "float32", "serving_reference_reader_dtype": "float16",
            "fp32_scope": "Original FP32 trained T B and FP32 scalar head; only fp32_reference_request_ids are measured",
            "score_readout": "last_suffix_token", "padding": "right_masked_actual_endpoint",
            "batch_size": batch_size, "padded_token_budget": token_budget,
            "precision_order": ["fp16", "fp32"] if fp32_rows else ["fp16"],
            "script_sha256": digest(Path(__file__).read_text())}
    def progress(current_precision, complete=False, exclusive=False):
        value = {**base, "protocol": "mixlm-nextbook-teacher-reference-progress-v1", "complete": complete,
                 "current_precision": current_precision, "completed_precisions": list(completed_precisions),
                 "completed_request_ids": {precision: [row["request_id"] for row in rows
                        if precision in references.get(row["request_id"], {})] for precision in ("fp16", "fp32")},
                 "references": references, "seconds": time.perf_counter() - started}
        value["progress_sha256"] = digest(value)
        _write_progress(progress_path, value, exclusive=exclusive)
    progress("fp16", exclusive=True)
    head = load_file(str(Path(args.export) / "scalar_head.safetensors"))["head.weight"].to(device, torch.float32)
    for precision, phase_rows in (("fp16", rows), ("fp32", fp32_rows)):
        if not phase_rows:
            continue
        model = _load_reader(args.export, device)
        if precision == "fp16":
            cast_half_preserve_rotary(model)
        for index, row in enumerate(phase_rows):
            prepared = compiler.compile(row["query"])
            entry = references.setdefault(row["request_id"], {"input_hash": prepared.input_hash,
                                           "candidate_ids": list(prepared.candidate_ids),
                                           "logical_lengths": list(prepared.logical_lengths)})
            entry[precision] = independent_scores(model, head, prepared, token_budget=token_budget,
                                                  max_batch_size=batch_size, pad_token_id=pad)
            if precision == "fp32":
                entry["fp16_vs_fp32"] = numerical_check(entry["fp16"], entry["fp32"])
            if index + 1 == len(phase_rows):
                completed_precisions.append(precision)
            if (index + 1) % 10 == 0 or index + 1 == len(phase_rows):
                progress(precision)
                print(json.dumps({"stage": "teacher_independent_reference", "precision": precision,
                                  "completed": index + 1, "total": len(phase_rows),
                                  "seconds": time.perf_counter() - started}), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = {**base, "complete": True, "references": references, "seconds": time.perf_counter() - started}
    result["reference_sha256"] = digest(result)
    with output.open("x") as stream:
        json.dump(result, stream, allow_nan=False, sort_keys=True)
    progress(None, complete=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("export", "workload", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workload-export", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--token-budget", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidate-count", type=int, choices=(10, 50, 100),
                        help="Select this candidate count while retaining the full parent workload identity")
    parser.add_argument("--fp32-groups", type=int,
                        help="Diagnose the fixed first N selected groups; default all, zero skips FP32")
    result = build(parser.parse_args())
    print(json.dumps({"complete": True, "reference_sha256": result["reference_sha256"], "seconds": result["seconds"]}))
