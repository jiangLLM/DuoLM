"""Independent causal PyTorch score references using the same offline inputs.

FP16 B / FP32 head is the serving-parity reference. FP32 B / FP32 head is also
reported as a numerical diagnostic; these references do not evaluate accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from safetensors.torch import load_file
import torch
from torch.nn import functional as F
from transformers import AutoModel

from duolm.nextbook_benchmark import digest, numerical_check, verify_workload
from duolm.nextbook_export import verify_nextbook_export
from duolm.nextbook_serving_inputs import NextBookCompiler


def cast_half_preserve_rotary(model):
    rotary = {name: tensor.detach().clone() for name, tensor in model.named_buffers()
              if "rotary_emb" in name and tensor.is_floating_point()}
    model.half()
    for name, tensor in rotary.items():
        parent, _, attribute = name.rpartition(".")
        setattr(model.get_submodule(parent), attribute, tensor)
    return model


@torch.inference_mode()
def build(args):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    export = verify_nextbook_export(args.export)
    workload = verify_workload(json.loads(args.workload.read_text()))
    if workload["dev_sha256"] != export["data_sha256"]["dev"]:
        raise ValueError("Workload is not from the checkpoint's frozen dev split")
    if args.output.exists():
        raise ValueError("Reference output already exists")
    compiler = NextBookCompiler(args.export)
    model = AutoModel.from_pretrained(args.export / "reader", torch_dtype=torch.float32,
                                     attn_implementation="sdpa", local_files_only=True).to("cuda").eval()
    head = load_file(str(args.export / "scalar_head.safetensors"))["head.weight"].to("cuda", torch.float32)
    vectors = torch.tensor(np.load(args.export / "vectors.npy"), device="cuda", dtype=torch.float16)
    vocab = export["native_reader_config"]["vocab_size"]
    def independent_score(prepared):
        # Separate causal sequences duplicate the prefix; no MIS tree or patch
        # is used here. The endpoint is the candidate vector itself.
        sequences = [prepared.query_ids + candidate for candidate in prepared.item_ids]
        values = []
        for start in range(0, len(sequences), 32):
            ids = torch.tensor(sequences[start:start+32], device="cuda", dtype=torch.long)
            embedding = model.get_input_embeddings()(ids.clamp(max=vocab-1))
            table = vectors[(ids-vocab).clamp(0, len(vectors)-1)].to(embedding.dtype)
            embedding = torch.where((ids >= vocab).unsqueeze(-1), table, embedding)
            hidden = model(inputs_embeds=embedding, use_cache=False, return_dict=True).last_hidden_state[:, -1]
            values.extend(F.linear(hidden.float(), head).squeeze(-1).cpu().tolist())
        return values
    references = {}
    start = time.time()
    for row in workload["requests"]:
        prepared = compiler.compile(row["query"])
        references[row["request_id"]] = {"input_hash": prepared.input_hash,
                                          "fp32": independent_score(prepared)}
    # .half() rounds nonpersistent rotary inv_freq buffers as well as weights.
    # Preserve exact FP32 RoPE frequencies, matching SGLang's frequency setup.
    cast_half_preserve_rotary(model)
    for row in workload["requests"]:
        prepared = compiler.compile(row["query"])
        reference = references[row["request_id"]]
        reference["fp16"] = independent_score(prepared)
        reference["fp16_vs_fp32"] = numerical_check(reference["fp16"], reference["fp32"])
    result = {"protocol": "mixlm-nextbook-reference-v1", "export_signature": export["manifest_signature"],
              "workload_sha256": workload["workload_sha256"], "references": references,
              "seconds": time.time()-start, "independent_causal_sequences": True,
              "offline_vectors_dtype": "float16", "score_head_dtype": "float32",
              "fp32_scope": "FP32 B on the same FP16 offline vectors; full A recomputation is separate",
              "script_sha256": digest(Path(__file__).read_text())}
    result["reference_sha256"] = digest(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result, f, allow_nan=False, sort_keys=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--workload", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    result = build(p.parse_args())
    print(json.dumps({"complete": True, "reference_sha256": result["reference_sha256"], "seconds": result["seconds"]}))
