"""Private raw-scalar MIS and causal carriers for the exact-gated next-book reader.

The vocabulary logprob arrays transport raw FP32 endpoint scalars.
Only authenticated private requests can select these carriers. Ordinary
requests retain upstream behavior; a mixed score/ordinary batch fails closed.
The native input embedding and vocabulary head objects are never replaced.
"""
from __future__ import annotations

import copy
import hashlib
import hmac
import importlib.metadata
import json
import math
import os
from pathlib import Path
import secrets
import types

import numpy as np
import torch
from torch.nn import functional as F


PROTOCOL = "mixlm-nextbook-scalar-mis-v1"
ENVIRONMENT = "MIXLM_NEXTBOOK_SCORE_RUNTIME"
FIELD = "mixlm_nextbook_score"
_CONFIG = _RAW_CONFIG = _AUTH_KEY = _EXPORT_ROOT = _INSTALLED = None
_COUNTERS = {"scalar_batches": 0, "scalar_requests": 0, "scalar_positions": 0}


def _require(condition, message):
    if not condition:
        raise ValueError("Next-book scalar score: " + message)


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _public_config(exported):
    native = exported["native_reader_config"]
    _require(native["model_type"] == "qwen3" and exported["variant"] == "V3"
             and exported["m"] == 1, "requires the frozen m=1 V3 Qwen3 reader")
    width, vocab, rows = native["hidden_size"], native["vocab_size"], exported["vector_count"]
    _require(type(width) is int and width > 0 and type(vocab) is int and vocab > 0
             and type(rows) is int and rows > 0, "invalid table dimensions")
    _require(exported["scalar_head"]["key"] == "head.weight"
             and exported["scalar_head"]["dtype"] == "float32"
             and exported["scalar_head"]["shape"] == [1, width]
             and exported["vectors"]["dtype"] == "float16"
             and exported["vectors"]["shape"] == [rows, width], "export tensor contract differs")
    value = {"protocol": PROTOCOL, "manifest_signature": exported["manifest_signature"],
             "scalar_head_sha256": exported["scalar_head"]["sha256"],
             "vectors_sha256": exported["vectors"]["sha256"],
             "index_sha256": exported["index"]["sha256"], "source_sha256": source_hash(),
             "variant": "V3", "m": 1, "native_vocab_size": vocab,
             "output_vocab_size": vocab, "input_vocab_size": vocab + rows,
             "hidden_size": width, "vector_count": rows, "carrier_token_ids": [0],
             "reader_dtype": "float16", "head_dtype": "float32", "score_dtype": "float32",
             "carrier": "private_raw_scalar_v1", "supported_modes": ["mis", "causal"], "prefix_cache": False}
    value["signature"] = _signature(value)
    return value


def _teacher_config(exported):
    """Bind a separately verified full-text teacher without virtual book IDs."""
    native = exported["native_reader_config"]
    width, vocab = native["hidden_size"], native["vocab_size"]
    _require(exported.get("kind") == "mixlm_nextbook_teacher_serving"
             and exported.get("variant") == "T" and native["model_type"] == "qwen3"
             and type(width) is int and width > 0 and type(vocab) is int and vocab > 0,
             "requires a verified full-text Qwen3 teacher")
    _require(exported["scalar_head"]["key"] == "head.weight"
             and exported["scalar_head"]["dtype"] == "float32"
             and exported["scalar_head"]["shape"] == [1, width], "teacher head contract differs")
    value = {"protocol": PROTOCOL, "manifest_signature": exported["manifest_signature"],
             "scalar_head_sha256": exported["scalar_head"]["sha256"],
             "source_sha256": source_hash(), "variant": "T", "m": None,
             "native_vocab_size": vocab, "output_vocab_size": vocab, "input_vocab_size": vocab,
             "hidden_size": width, "vector_count": 0, "carrier_token_ids": [0],
             "reader_dtype": "float16", "head_dtype": "float32", "score_dtype": "float32",
             "carrier": "private_raw_scalar_v1", "supported_modes": ["mis"], "prefix_cache": False}
    value["signature"] = _signature(value)
    return value


def _arithmetic_config(value, mode):
    _require(mode in {"sglang", "reference_fp16"}, "unknown arithmetic mode")
    value = copy.deepcopy(value)
    value.pop("signature", None)
    value.pop("arithmetic_source_sha256", None)
    value["arithmetic_mode"] = mode
    if mode == "reference_fp16":
        from .nextbook_reference_arithmetic import source_hashes
        _require(value["variant"] == "V3" and value["m"] == 1
                 and value["hidden_size"] == 1024, "reference arithmetic requires V3 m=1 width=1024")
        value["arithmetic_source_sha256"] = source_hashes()
    value["signature"] = _signature(value)
    return value


def _verify_assets(root, variant, arithmetic_mode="sglang"):
    if variant == "T":
        from .nextbook_teacher_export import verify_nextbook_teacher_export
        exported = verify_nextbook_teacher_export(root)
        return exported, _arithmetic_config(_teacher_config(exported), arithmetic_mode)
    _require(variant == "V3", "unknown serving variant")
    from .nextbook_export import verify_nextbook_export
    exported = verify_nextbook_export(root, verify_vectors=True)
    return exported, _arithmetic_config(_public_config(exported), arithmetic_mode)


def configure_runtime(export_root, *, variant="V3", arithmetic_mode="sglang"):
    """Verify assets and publish a spawn-inherited, authenticated configuration.

    The returned receipt and all reporting APIs omit the admission key. The
    manifest, exact saved head bytes and this module are bound into the receipt.
    """
    global _CONFIG, _RAW_CONFIG, _AUTH_KEY, _EXPORT_ROOT
    _require(_INSTALLED is None, "an installed worker cannot be reconfigured")
    root = Path(export_root).resolve(strict=True)
    _, value = _verify_assets(root, variant, arithmetic_mode)
    envelope = {"config": value, "export_root": str(root), "admission_key": secrets.token_hex(32)}
    _CONFIG = _RAW_CONFIG = _AUTH_KEY = _EXPORT_ROOT = None
    _COUNTERS.update(dict.fromkeys(_COUNTERS, 0))
    os.environ[ENVIRONMENT] = json.dumps(envelope, sort_keys=True, separators=(",", ":"))
    return copy.deepcopy(runtime_config())


def clear_runtime():
    global _CONFIG, _RAW_CONFIG, _AUTH_KEY, _EXPORT_ROOT
    _require(_INSTALLED is None, "an installed worker cannot be cleared")
    os.environ.pop(ENVIRONMENT, None)
    _CONFIG = _RAW_CONFIG = _AUTH_KEY = _EXPORT_ROOT = None


def runtime_config():
    global _CONFIG, _RAW_CONFIG, _AUTH_KEY, _EXPORT_ROOT
    raw = os.environ.get(ENVIRONMENT)
    if not raw:
        _require(_CONFIG is None, "runtime configuration disappeared after initialization")
        return None
    if _CONFIG is None:
        envelope = json.loads(raw)
        _require(type(envelope) is dict and set(envelope) == {"config", "export_root", "admission_key"},
                 "invalid runtime envelope")
        value = envelope["config"]
        _require(type(value) is dict and value.get("protocol") == PROTOCOL
                 and value.get("source_sha256") == source_hash(), "runtime module/config identity differs")
        unsigned = {key: val for key, val in value.items() if key != "signature"}
        _require(value.get("signature") == _signature(unsigned), "runtime signature differs")
        _require(value.get("carrier_token_ids") == [0] and value.get("prefix_cache") is False
                 and value.get("head_dtype") == "float32", "runtime scalar contract differs")
        _require(_arithmetic_config(value, value.get("arithmetic_mode")) == value,
                 "runtime arithmetic identity differs")
        try:
            key = bytes.fromhex(envelope["admission_key"])
        except (ValueError, TypeError) as exc:
            raise ValueError("Next-book scalar score: invalid admission key") from exc
        _require(len(key) == 32, "invalid admission key")
        _CONFIG, _RAW_CONFIG, _AUTH_KEY, _EXPORT_ROOT = value, raw, key, envelope["export_root"]
    _require(raw == _RAW_CONFIG, "runtime configuration changed after initialization")
    return copy.deepcopy(_CONFIG)


def installed_metadata():
    return copy.deepcopy(_INSTALLED)


def runtime_report():
    value = runtime_config()
    return None if value is None else {**value, "installed": installed_metadata(), "counters": dict(_COUNTERS)}


def _payload(input_ids, delimiters):
    return _signature({"input_ids": list(input_ids),
                       "delimiters": None if delimiters is None else list(delimiters)})


def _new_mark(config, payload, *, mode="mis"):
    _require(mode in config["supported_modes"], "unsupported authenticated score mode")
    data = ":".join((config["signature"], secrets.token_hex(16), payload, mode))
    return data + ":" + hmac.new(_AUTH_KEY, data.encode(), hashlib.sha256).hexdigest()


def _valid_mark(mark, config, payload=None, *, mode=None):
    if not isinstance(mark, str):
        return False
    parts = mark.split(":")
    if (len(parts) != 5 or parts[0] != config["signature"] or len(parts[1]) != 32
            or len(parts[2]) != 64 or (payload is not None and parts[2] != payload)
            or parts[3] not in config["supported_modes"] or (mode is not None and parts[3] != mode)):
        return False
    expected = hmac.new(_AUTH_KEY, ":".join(parts[:4]).encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(parts[4], expected)


def _mark_mode(mark):
    # Call only after validating the signature; this field is authenticated.
    return mark.split(":")[3]


def mark_causal_request(request, *, query, item):
    """Admit exactly one V3 candidate as a normal, unpadded causal sequence."""
    config = runtime_config()
    _require(config is not None and config["variant"] == "V3"
             and config["arithmetic_mode"] == "sglang", "causal scoring requires the V3 sglang runtime")
    _require(type(query) is list and query
             and all(type(token) is int and 0 <= token < config["input_vocab_size"] for token in query)
             and type(item) is list and len(item) == 1 and type(item[0]) is int
             and config["native_vocab_size"] <= item[0] < config["input_vocab_size"],
             "causal scoring requires a query and one bound book input ID")
    ids = query + item
    _require(getattr(request, FIELD, None) is None and request.input_ids == ids
             and getattr(request, "multi_item_delimiter_indices", None) is None
             and request.token_indices_to_pool == [len(ids) - 1],
             "causal input/endpoint layout differs")
    _require(request.sampling_params == {"max_new_tokens": 0} and request.return_logprob is True
             and request.token_ids_logprob == [0] and request.stream is False
             and request.logprob_start_len == 0, "unexpected causal sampling/logprob contract")
    _require(not getattr(request, "top_logprobs_num", 0) and not getattr(request, "return_entropy", False)
             and not getattr(request, "return_hidden_states", False)
             and not getattr(request, "custom_logit_processor", None)
             and getattr(request, "positional_embed_overrides", None) is None
             and getattr(request, "input_embeds", None) is None,
             "unsupported causal auxiliary outputs or embedding overrides")
    setattr(request, FIELD, _new_mark(config, _payload(ids, None), mode="causal"))
    return True


async def score_causal_request(manager, query, item):
    """Private engine callback; every candidate owns a plain GenerateReqInput."""
    from sglang.srt.managers.io_struct import GenerateReqInput
    from sglang.srt.managers.tokenizer_manager_score_mixin import ScoreResult

    ids = list(query) + list(item)
    request = GenerateReqInput(input_ids=ids, sampling_params={"max_new_tokens": 0},
        return_logprob=True, token_ids_logprob=[0], logprob_start_len=0, stream=False,
        token_indices_to_pool=[len(ids) - 1])
    mark_causal_request(request, query=query, item=item)
    responses = manager.generate_request(request, None)
    try:
        result = await responses.__anext__()
        return decode_causal_result(result, request, ScoreResult)
    finally:
        await responses.aclose()


def _validate_causal_layout(request, input_ids):
    _require(getattr(request, "multi_item_delimiter_indices", None) is None
             and getattr(request, "token_indices_to_pool", None) == [len(input_ids) - 1],
             "authenticated causal request changed its endpoint or attention layout")


def mark_score_batch(batch, *, query, items, labels, apply_softmax, is_generation, use_mis, item_first):
    """Admission hook runs after the pinned score_request input validation."""
    config = runtime_config()
    if config is None or type(query) is not list or not query or type(items) is not list or not items:
        return False
    token_items = all(type(item) is list for item in items)
    has_book = token_items and any(type(token) is int and token >= config["native_vocab_size"]
                                   for item in items for token in item)
    teacher = config["variant"] == "T"
    if not teacher and not has_book:
        return False
    _require(apply_softmax is True and is_generation is True and use_mis is True
             and item_first is False and labels == [0], "book scoring requires the fixed scalar MIS contract")
    if teacher:
        _require(all(type(item) is list and len(item) > 0
                     and all(type(token) is int and 0 <= token < config["native_vocab_size"] for token in item)
                     for item in items), "teacher candidates must contain native text token IDs")
    else:
        _require(all(type(item) is list and len(item) == 1
                     and type(item[0]) is int and config["native_vocab_size"] <= item[0] < config["input_vocab_size"]
                     for item in items), "each candidate must contain one bound book input ID")
    if config["arithmetic_mode"] == "reference_fp16":
        _require(len(items) == 100, "reference arithmetic requires exactly 100 candidates")
    _require(all(type(token) is int and 0 <= token < config["input_vocab_size"] for token in query),
             "query ID outside the native/book tables")
    _require(getattr(batch, FIELD, None) is None, "score request was already marked")
    _require(batch.sampling_params == {"max_new_tokens": 0} and batch.return_logprob is True
             and batch.token_ids_logprob == [0] and batch.stream is False
             and batch.logprob_start_len == 0, "unexpected score sampling/logprob contract")
    _require(not getattr(batch, "top_logprobs_num", 0) and not getattr(batch, "return_entropy", False)
             and not getattr(batch, "return_hidden_states", False)
             and not getattr(batch, "custom_logit_processor", None)
             and getattr(batch, "positional_embed_overrides", None) is None,
             "unsupported auxiliary score outputs or embedding overrides")
    _require(type(batch.input_ids) is list and len(batch.input_ids) == 1
             and type(batch.input_ids[0]) is list
             and type(batch.multi_item_delimiter_indices) is list
             and len(batch.multi_item_delimiter_indices) == 1, "need one packed MIS request")
    packed, indices = batch.input_ids[0], batch.multi_item_delimiter_indices[0]
    expected, position = [], len(query)
    for item in items:
        expected.append(position)
        position += len(item) + 1
    expected.append(position)
    _require(indices == expected and len(packed) == position + 1 and packed[:len(query)] == query,
             "packed MIS query/delimiter layout differs")
    delimiter = packed[indices[0]]
    _require(type(delimiter) is int and 0 <= delimiter < config["native_vocab_size"]
             and all(packed[index] == delimiter for index in indices)
             and all(packed[indices[i] + 1:indices[i + 1]] == item for i, item in enumerate(items)),
             "packed MIS candidate order differs")
    setattr(batch, FIELD, _new_mark(config, _payload(packed, indices)))
    return True


def request_marks(requests):
    """Revalidate signed CPU request IDs before they enter ForwardBatch."""
    marks = [getattr(req, FIELD, None) for req in requests]
    if not any(mark is not None for mark in marks):
        return marks
    config = runtime_config()
    _require(config is not None, "marked request without runtime configuration")
    if config["arithmetic_mode"] == "reference_fp16":
        _require(len(requests) == 1 and marks[0] is not None,
                 "reference arithmetic requires one physical marked request")
    for req, mark in zip(requests, marks):
        if mark is not None:
            _require(_valid_mark(mark, config, _payload(req.origin_input_ids, req.multi_item_delimiter_indices)),
                     "request input/model signature differs")
            if _mark_mode(mark) == "causal":
                _validate_causal_layout(req, req.origin_input_ids)
            else:
                _require(req.multi_item_delimiter_indices is not None
                         and getattr(req, "token_indices_to_pool", None) is None,
                         "MIS request changed its readout layout")
            _require(len(req.prefix_indices) == 0 and not req.output_ids,
                     "scalar MIS requires an uncached prefill-only request")
    return marks


def validate_input_request(request, input_ids, native_vocab_size):
    """Keep carrier labels in the native vocabulary and validate book input IDs.

    The pinned tokenizer manager's input-vocabulary helper is unused by its
    request path. This hook applies the extended bound only to authenticated
    scalar calls; it never enlarges the vocabulary used for logprob labels.
    """
    mark = getattr(request, FIELD, None)
    if mark is None:
        return
    config = runtime_config()
    _require(config is not None and native_vocab_size == config["native_vocab_size"],
             "tokenizer/model vocabulary differs")
    _require(_valid_mark(mark, config, _payload(input_ids, request.multi_item_delimiter_indices)),
             "tokenized request input/model signature differs")
    if _mark_mode(mark) == "causal":
        _validate_causal_layout(request, input_ids)
    else:
        _require(request.multi_item_delimiter_indices is not None
                 and getattr(request, "token_indices_to_pool", None) is None,
                 "MIS request changed its readout layout")
    _require(all(type(token) is int and 0 <= token < config["input_vocab_size"] for token in input_ids)
             and request.token_ids_logprob == [0], "tokenized input/carrier vocabulary differs")


def _marked(metadata):
    marks = getattr(metadata, "mixlm_nextbook_scores", None)
    if not marks or not any(mark is not None for mark in marks):
        return []
    config = runtime_config()
    _require(config is not None, "marked request without runtime configuration")
    _require(metadata.is_prefill_only and len(marks) == len(metadata.token_ids_logprobs)
             and len(marks) == len(metadata.top_logprobs_nums), "marked request is not a prefill-only score batch")
    flags = []
    for index, mark in enumerate(marks):
        _require(mark is None or _valid_mark(mark, config), "request/model signature differs")
        flags.append(mark is not None)
        if mark is not None:
            _require(metadata.token_ids_logprobs[index] == [0] and metadata.top_logprobs_nums[index] == 0,
                     "marked carrier token or auxiliary logprobs differ")
    _require(all(flags), "mixed ordinary/scalar batches are unsupported")
    _require(len({_mark_mode(mark) for mark in marks}) == 1, "mixed MIS/causal scalar batches are unsupported")
    return flags


def guard_marked_forward(metadata, delimiters, positions=None):
    flags = _marked(metadata)
    if not flags:
        return
    if _mark_mode(metadata.mixlm_nextbook_scores[0]) == "causal":
        _require(delimiters is None and positions is not None, "causal request changed its attention/readout layout")
    else:
        _require(delimiters is not None and positions is None, "marked request omitted MIS delimiters or changed readout")


def _tensor_identity(tensor):
    return (id(tensor), tensor.data_ptr(), None if tensor.is_inference() else tensor._version,
            tuple(tensor.shape), tensor.dtype, tensor.device)


def scalar_scores(hidden, weights):
    """Apply the trained FP32 head to FP16 B outputs without normalization."""
    _require(isinstance(hidden, torch.Tensor) and isinstance(weights, torch.Tensor)
             and hidden.ndim == weights.ndim == 2 and weights.shape[0] == 1
             and hidden.shape[1] == weights.shape[1] and hidden.dtype == torch.float16
             and weights.dtype == torch.float32 and hidden.device == weights.device,
             "requires FP16 hidden rows and the exact FP32 [1,width] scalar head")
    # Preserve FP32 products and accumulation under surrounding autocast.
    # SGLang's deterministic mode replaces aten::mm/addmm with a Triton kernel
    # whose dot input precision is independent of PyTorch's TF32 setting.
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        return (hidden.float() * weights).sum(dim=-1, keepdim=True, dtype=torch.float32)


def _bound_head(processor, head, config):
    _require(processor.use_fp32_lm_head
             and not getattr(processor, "do_tensor_parallel_all_gather", False)
             and not getattr(processor, "do_tensor_parallel_all_gather_dp_attn", False),
             "scalar runtime requires enable_fp32_lm_head and TP/DP=1")
    weight = getattr(head, "_mixlm_nextbook_scalar_weight", None)
    binding = getattr(head, "_mixlm_nextbook_scalar_binding", None)
    _require(weight is not None and binding is not None and binding[0] == config["signature"]
             and binding[1] == _tensor_identity(weight)
             and tuple(weight.shape) == (1, config["hidden_size"])
             and weight.dtype == torch.float32, "scalar weights/model binding changed or was not installed")
    return weight


def compute_mis_output(processor, input_ids, hidden, head, metadata, delimiters, output_type):
    """Project only marked endpoint rows; never fall back for a marked request."""
    flags = _marked(metadata)
    if not flags:
        return None
    config = runtime_config()
    _require(_mark_mode(metadata.mixlm_nextbook_scores[0]) == "mis", "causal request cannot use the MIS head")
    _require(delimiters is not None and len(delimiters) == len(flags) == len(metadata.extend_seq_lens_cpu),
             "MIS batch lengths differ")
    offsets, counts, offset = [], [], 0
    for length, indices in zip(metadata.extend_seq_lens_cpu, delimiters):
        _require(type(length) is int and length > 0 and isinstance(indices, torch.Tensor)
                 and indices.ndim == 1 and indices.dtype == torch.int64 and indices.device.type == "cpu"
                 and len(indices) >= 2 and int(indices[0]) >= 1
                 and int(indices[-1]) == length - 1
                 and bool(((indices[1:] - indices[:-1] >= 2) if config["variant"] == "T"
                           else (indices[1:] - indices[:-1] == 2)).all()),
                 "invalid uncached MIS delimiter positions for the bound variant")
        offsets.append(indices + offset - 1)
        counts.append(len(indices))
        offset += length
    _require(offset == hidden.shape[0] == input_ids.shape[0], "hidden rows differ from MIS extend lengths")
    positions = torch.cat(offsets).to(hidden.device, non_blocking=True)
    values = scalar_scores(hidden[positions], _bound_head(processor, head, config))
    metadata.extend_logprob_pruned_lens_cpu = counts
    _COUNTERS["scalar_batches"] += 1
    _COUNTERS["scalar_requests"] += len(flags)
    _COUNTERS["scalar_positions"] += sum(counts)
    return output_type(next_token_logits=None,
        input_token_logprobs=torch.zeros(sum(counts), device=input_ids.device),
        input_top_logprobs_val=None, input_top_logprobs_idx=None,
        input_token_ids_logprobs_val=list(values.split(counts)),
        input_token_ids_logprobs_idx=[[[0] for _ in range(count)] for count in counts],
        mm_input_embeds=metadata.mm_input_embeds)


def compute_causal_output(processor, input_ids, hidden, head, metadata, positions, output_type):
    """Read the last candidate token from each separate ordinary causal sequence."""
    flags = _marked(metadata)
    if not flags or _mark_mode(metadata.mixlm_nextbook_scores[0]) != "causal":
        return None
    config = runtime_config()
    lengths = metadata.extend_seq_lens_cpu
    _require(positions is not None and len(positions) == len(lengths) == len(flags),
             "causal batch lengths differ")
    offsets, offset = [], 0
    for length, indices in zip(lengths, positions, strict=True):
        _require(type(length) is int and length > 1 and isinstance(indices, torch.Tensor)
                 and indices.ndim == 1 and indices.dtype == torch.int64 and indices.device.type == "cpu"
                 and indices.tolist() == [length - 1], "causal readout must be the candidate endpoint")
        offset += length
        offsets.append(offset - 1)
    _require(offset == hidden.shape[0] == input_ids.shape[0], "hidden rows differ from causal extend lengths")
    endpoint_ids = torch.tensor(offsets, dtype=torch.int64, device=hidden.device)
    values = scalar_scores(hidden[endpoint_ids], _bound_head(processor, head, config))
    metadata.extend_logprob_pruned_lens_cpu = [1] * len(flags)
    _COUNTERS["scalar_batches"] += 1
    _COUNTERS["scalar_requests"] += len(flags)
    _COUNTERS["scalar_positions"] += len(flags)
    return output_type(next_token_logits=None,
        input_token_logprobs=torch.zeros(len(flags), device=input_ids.device),
        input_top_logprobs_val=None, input_top_logprobs_idx=None,
        input_token_ids_logprobs_val=list(values.split(1)),
        input_token_ids_logprobs_idx=[[[0]] for _ in flags], mm_input_embeds=metadata.mm_input_embeds)


def decode_causal_result(results, request, result_type):
    """Authenticate one scalar, without softmax or vocabulary-logprob conversion."""
    config, mark = runtime_config(), getattr(request, FIELD, None)
    _require(config is not None and _valid_mark(mark, config, _payload(request.input_ids, None), mode="causal"),
             "causal response/request input/model signature differs")
    _validate_causal_layout(request, request.input_ids)
    _require(type(results) is dict and type(results.get("meta_info")) is dict,
             "expected one physical causal result")
    meta = results["meta_info"]
    _require(meta.get(FIELD) == mark and meta.get("cached_tokens") == 0,
             "causal response omitted its signature or used prefix caching")
    rows = meta.get("input_token_ids_logprobs")
    _require(type(rows) is list and len(rows) == 1 and type(rows[0]) is list and len(rows[0]) == 1,
             "causal carrier must contain exactly one candidate scalar")
    row = rows[0][0]
    _require(type(row) in (list, tuple) and len(row) == 3 and type(row[1]) is int and row[1] == 0
             and type(row[0]) in (int, float) and math.isfinite(row[0]),
             "causal carrier must contain a finite raw scalar at token 0")
    return result_type(scores=[[float(row[0])]], prompt_tokens=meta.get("prompt_tokens", 0),
                       cached_tokens=[0], nextbook_score_signature=config["signature"])


def decode_score_result(results, batch, items, result_type):
    """Terminate the private carrier before upstream softmax/exp conversion."""
    config, mark = runtime_config(), getattr(batch, FIELD, None)
    _require(config is not None and _valid_mark(mark, config,
             _payload(batch.input_ids[0], batch.multi_item_delimiter_indices[0]), mode="mis"),
             "response/request input/model signature differs")
    _require(not isinstance(results, list) or len(results) == 1, "expected one physical MIS result")
    result = results[0] if isinstance(results, list) else results
    meta = result["meta_info"]
    _require(meta.get(FIELD) == mark, "response omitted its private carrier signature")
    _require(meta.get("cached_tokens", 0) == 0, "scalar runtime must not use prefix caching")
    rows = meta.get("input_token_ids_logprobs")
    _require(type(rows) is list and len(rows) == len(items) + 1, "carrier delimiter count differs")
    values = []
    for row in rows:
        _require(type(row) is list and len(row) == 1 and len(row[0]) == 3 and row[0][1] == 0
                 and type(row[0][0]) in (int, float) and math.isfinite(row[0][0]),
                 "carrier must contain one finite raw scalar at token 0")
        values.append([float(row[0][0])])
    return result_type(scores=values[1:], prompt_tokens=meta.get("prompt_tokens", 0),
                       cached_tokens=[0], nextbook_score_signature=config["signature"])


def _attach_lookup(embedding, table, native_vocab_size, signature):
    """CPU-testable dual lookup; native weight/head storage stays intact."""
    weight = embedding.weight
    _require(weight.ndim == 2 and weight.dtype == torch.float16 and weight.shape[0] >= native_vocab_size
             and table.ndim == 2 and table.shape[0] > 0 and table.shape[1] == weight.shape[1]
             and table.dtype == torch.float16 and table.device == weight.device
             and not table.requires_grad and not table.is_inference(), "invalid FP16 book input table")
    _require(not hasattr(embedding, "_mixlm_nextbook_table") and "forward" not in embedding.__dict__,
             "input embedding already has a lookup override")
    original_forward = embedding.forward
    identities = (_tensor_identity(weight), _tensor_identity(table))
    input_vocab = native_vocab_size + table.shape[0]
    embedding.register_buffer("_mixlm_nextbook_table", table, persistent=False)

    @torch.no_grad()
    def forward(self, input_ids):
        _require((_tensor_identity(self.weight), _tensor_identity(self._mixlm_nextbook_table)) == identities,
                 "embedding/table changed after installation")
        _require(input_ids.dtype in (torch.int32, torch.int64) and input_ids.device == weight.device,
                 "input IDs require an integer tensor on the model device")
        in_range = ((input_ids >= 0) & (input_ids < input_vocab)).all()
        if input_ids.device.type == "cpu":
            _require(bool(in_range), "input ID outside the native/book tables")
        else:
            torch._assert_async(in_range, "Input ID outside the native/book tables")
        is_book = input_ids >= native_vocab_size
        native_values = original_forward(torch.where(is_book, 0, input_ids))
        book_rows = (input_ids - native_vocab_size).clamp(0, table.shape[0] - 1).long()
        return torch.where(is_book.unsqueeze(-1), F.embedding(book_rows, self._mixlm_nextbook_table), native_values)

    embedding.forward = types.MethodType(forward, embedding)
    embedding._mixlm_nextbook_signature = signature


def _verify_model_source(model):
    import inspect
    from .nextbook_sglang_overlay import verify_overlay

    _require(importlib.metadata.version("sglang") == "0.5.21", "requires SGLang 0.5.21")
    _require(type(model).__name__ == "Qwen3ForCausalLM"
             and type(model).__module__ == "sglang.srt.models.qwen3", "requires the pinned Qwen3 model")
    package = Path(inspect.getfile(type(model))).resolve().parents[2]
    verify_overlay(package.parent)


def install_book_table(model):
    """Qwen3 weight-load hook: verify assets, install FP16 IDs and FP32 head."""
    global _INSTALLED
    config = runtime_config()
    if config is None:
        return None
    _require(_INSTALLED is None, "one immutable table/head per model worker is required")
    from safetensors.torch import load_file

    _verify_model_source(model)
    root = Path(_EXPORT_ROOT)
    exported, verified_config = _verify_assets(root, config["variant"], config["arithmetic_mode"])
    _require(verified_config == config, "worker assets differ from launcher binding")
    native = exported["native_reader_config"]
    embedding, head = model.model.embed_tokens, model.lm_head
    _require(model.config.vocab_size == native["vocab_size"]
             and model.config.hidden_size == native["hidden_size"]
             and model.config.tie_word_embeddings == native["tie_word_embeddings"], "loaded model configuration differs")
    _require(model.pp_group.world_size == 1 and embedding.tp_size == 1 and model.quant_config is None,
             "scalar runtime requires unquantized TP/PP=1")
    _require(embedding.num_embeddings == native["vocab_size"] and embedding.weight.device.type == "cuda"
             and embedding.weight.dtype == torch.float16 and head.weight.dtype == torch.float16
             and tuple(head.weight.shape) == tuple(embedding.weight.shape), "requires native FP16 CUDA embedding/head")
    _require(getattr(embedding, "output_dtype", None) in (None, torch.float16), "embedding output dtype differs")
    if native["tie_word_embeddings"]:
        _require(head is embedding and head.weight is embedding.weight, "native tied head changed")
    original_head, original_weight = head, head.weight
    tensors = load_file(str(root / exported["scalar_head"]["file"]), device="cpu")
    weight = tensors["head.weight"]
    _require(weight.dtype == torch.float32 and tuple(weight.shape) == (1, native["hidden_size"]),
             "saved scalar head was not exact FP32")
    with torch.inference_mode(False):
        scalar_weight = weight.detach().to(device=embedding.weight.device).clone()
    table_bytes = 0
    if config["variant"] == "V3":
        vectors = np.load(root / exported["vectors"]["file"], allow_pickle=False)
        with torch.inference_mode(False):
            table = torch.from_numpy(np.array(vectors, copy=True)).to(device=embedding.weight.device)
        _attach_lookup(embedding, table, native["vocab_size"], config["signature"])
        table_bytes = table.numel() * table.element_size()
    _require(not hasattr(head, "_mixlm_nextbook_scalar_weight"), "scalar head was already bound")
    head.register_buffer("_mixlm_nextbook_scalar_weight", scalar_weight, persistent=False)
    head._mixlm_nextbook_scalar_binding = (config["signature"], _tensor_identity(scalar_weight))
    _require(model.lm_head is original_head and model.lm_head.weight is original_weight,
             "input/scalar hook changed native vocabulary head storage")
    arithmetic_metadata = None
    if config["arithmetic_mode"] == "reference_fp16":
        from .nextbook_reference_arithmetic import install
        arithmetic_metadata = install(model, config)
    _INSTALLED = {**config, "installed": True, "device": str(scalar_weight.device),
                  "reference_arithmetic": arithmetic_metadata,
                  "table_bytes": table_bytes,
                  "scalar_head_bytes": scalar_weight.numel() * scalar_weight.element_size(),
                  "output_head_shape": list(head.weight.shape), "output_head_unchanged": True}
    model._mixlm_nextbook_metadata = installed_metadata()
    return installed_metadata()
