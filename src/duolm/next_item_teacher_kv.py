"""Optional FP32/SDPA next-item T inference with one prefix prefill.

This module does not alter the ranker's forward, gradients, backend or weights.
Callers supply an independent evaluation reader with identical frozen weights
and SDPA already selected. Each physical candidate chunk gets its own full
DynamicCache, cloned from the fixed prefix; no cache survives a scoring call.
"""
from __future__ import annotations

import inspect

import torch
from torch import nn
from torch.nn import functional as F

from .train_next_item import inference_context


PROTOCOL = "next-item-teacher-prefix-kv-fp32-v1"


def _fp32_inference(device_type):
    """Reuse the trainer's verified FP32/TF32 guard, including restoration."""
    return inference_context(device_type, "float32")


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _input_assembly(ranker, value):
    if not isinstance(value, str) or value not in {"legacy", "batched"}:
        raise ValueError("teacher scoring input_assembly must be legacy or batched")
    if value == "batched" and not callable(getattr(ranker, "_batched_native_embeddings", None)):
        raise ValueError("batched teacher scoring requires the ranker's native embedding helper")
    return value


def _backbone(reader):
    if not isinstance(reader, nn.Module) or reader.training:
        raise ValueError("teacher KV inference requires an eval reader")
    # Switching an already compiled or live training backbone is deliberately
    # unsupported. No config/backend mutation occurs anywhere in this module.
    if hasattr(reader, "_orig_mod"):
        raise ValueError("compiled reader wrappers are unsupported; supply an independent SDPA reader")
    backbone = getattr(reader, "model", reader)
    if hasattr(backbone, "_orig_mod"):
        raise ValueError("compiled reader wrappers are unsupported; supply an independent SDPA reader")
    config = getattr(backbone, "config", None)
    if config is None or config.model_type != "qwen3" or not hasattr(backbone, "layers"):
        raise ValueError("teacher KV inference supports a real HF Qwen3 reader only")
    if backbone.training or config._attn_implementation != "sdpa":
        raise ValueError("supply an independent eval SDPA reader; the trainer backend is never switched")
    if any(module.training for module in backbone.modules()):
        raise ValueError("all reader modules must be in evaluation mode")
    if (getattr(config, "attention_dropout", 0) != 0 or getattr(config, "use_sliding_window", False)
            or any(kind != "full_attention" for kind in getattr(config, "layer_types", []))):
        raise ValueError("teacher KV inference requires zero-dropout full causal attention")
    rope = getattr(config, "rope_scaling", None)
    if rope and rope.get("rope_type", rope.get("type", "default")) != "default":
        raise ValueError("non-default RoPE needs separate prefix-sharing validation")
    # Explicit cache_position is required by the 4.57.x Qwen3 API. Reject an
    # unknown API instead of silently relying on an incorrect inferred offset.
    if "cache_position" not in inspect.signature(backbone.forward).parameters:
        raise ValueError("unsupported HF Qwen3 cache API: explicit cache_position is required")
    return backbone


def _head_weight(head, width, device):
    if isinstance(head, nn.Linear):
        if head.training or head.bias is not None:
            raise ValueError("teacher scalar head must be eval and bias-free")
        head = head.weight
    if (not isinstance(head, torch.Tensor) or tuple(head.shape) != (1, width)
            or head.dtype != torch.float32 or head.device != device):
        raise ValueError("teacher scalar head must have shape [1, width], FP32 and the reader device")
    return head


def _cache_factory(config):
    from transformers import DynamicCache
    parameters = inspect.signature(DynamicCache.__init__).parameters
    if "config" not in parameters:
        raise ValueError("unsupported HF DynamicCache API: config-aware full-attention cache required")
    cache = DynamicCache(config=config)
    if not callable(getattr(cache, "update", None)) or not callable(getattr(cache, "get_seq_length", None)):
        raise ValueError("unsupported HF DynamicCache update/length API")
    return cache


def _cache_pairs(cache, config, prefix_length, device):
    """Extract fixed prefix states without keeping mutable cache-layer aliases."""
    if hasattr(cache, "layers"):
        pairs = [(layer.keys, layer.values) for layer in cache.layers]
    elif hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        pairs = list(zip(cache.key_cache, cache.value_cache))
    else:
        raise ValueError("unsupported HF DynamicCache state API")
    if len(pairs) != config.num_hidden_layers:
        raise ValueError("prefill cache must contain every Qwen3 decoder layer")
    expected = (1, config.num_key_value_heads, prefix_length,
                getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    frozen = []
    for key, value in pairs:
        if (not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor)
                or tuple(key.shape) != expected or tuple(value.shape) != expected
                or key.dtype != torch.float32 or value.dtype != torch.float32
                or key.device != device or value.device != device):
            raise ValueError("prefill KV shape/precision/device mismatch")
        frozen.append((key.detach().clone(), value.detach().clone()))
    return tuple(frozen)


def _fork_cache(frozen, config, batch_size):
    """Own every layer and KV tensor; base prefix and other chunks cannot alias."""
    cache = _cache_factory(config)
    for index, (key, value) in enumerate(frozen):
        # Never call batch_repeat_interleave on the original prefix cache:
        # both that helper and forward update change the cache object in place.
        keys = key.expand(batch_size, -1, -1, -1).clone()
        values = value.expand(batch_size, -1, -1, -1).clone()
        cache.update(keys, values, index)
    return cache


def _branch_mask(lengths, prefix_length, branch_length, device):
    """Bool SDPA mask [batch, 1, S, P+S], with offset causality and key padding."""
    queries = prefix_length + torch.arange(branch_length, device=device)
    keys = torch.arange(prefix_length + branch_length, device=device)
    causal = keys[None, :] <= queries[:, None]
    valid = keys[None, :] < (prefix_length + lengths[:, None])
    return causal[None, None] & valid[:, None, None]


@torch.inference_mode()
def score_prefix_kv(reader, head, prefix, branches, *, candidate_chunk_size=8,
                    max_logical_tokens=None):
    """Score one complete T group in original candidate order, using FP32 only.

    prefix is [P,H]; branches is a sequence of [S_i,H] native embeddings from
    GroupRanker._query_embeddings. Each branch already includes the exact
    separately tokenized candidate header, content and final score suffix.
    Scores are raw logits. Group softmax/ranking must consume the whole vector.
    """
    chunk_size = _positive(candidate_chunk_size, "candidate_chunk_size")
    if chunk_size > 16:
        raise ValueError("candidate_chunk_size must be at most 16; no automatic resizing")
    backbone = _backbone(reader)
    config = backbone.config
    width = config.hidden_size
    if not isinstance(prefix, torch.Tensor) or prefix.ndim != 2 or prefix.shape[0] < 1 or prefix.shape[1] != width:
        raise ValueError("prefix must have nonempty shape [P, reader width]")
    if not isinstance(branches, (tuple, list)) or not branches:
        raise ValueError("branches must be a nonempty sequence")
    device = prefix.device
    for value in (prefix, *branches):
        if (not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] < 1
                or value.shape[1] != width or value.dtype != torch.float32 or value.device != device):
            raise ValueError("prefix/branches must be nonempty FP32 embeddings on one reader device")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("teacher KV inference supports CPU or CUDA")
    if any(parameter.device != device or parameter.dtype != torch.float32 for parameter in backbone.parameters()):
        raise ValueError("teacher KV reader parameters must remain original FP32 on the input device")
    weight = _head_weight(head, width, device)
    p = len(prefix)
    lengths = tuple(p + len(branch) for branch in branches)
    limit = config.max_position_embeddings
    if max_logical_tokens is not None:
        limit = min(limit, _positive(max_logical_tokens, "max_logical_tokens"))
    if max(lengths) > limit:
        raise ValueError("teacher path exceeds logical context; no truncation")
    output_scores, padded_tokens, max_mask, max_chunk = [], 0, 0, 0
    with _fp32_inference(device.type):
        finite = torch.stack([torch.isfinite(value).all() for value in (prefix, *branches)])
        if not bool(finite.all()):
            raise ValueError("nonfinite teacher input embeddings")
        prefill_cache = _cache_factory(config)
        positions = torch.arange(p, device=device)
        prefix_mask = torch.ones((1, 1, p, p), dtype=torch.bool, device=device).tril()
        prefill = backbone(inputs_embeds=prefix[None], attention_mask=prefix_mask,
                           position_ids=positions[None], cache_position=positions,
                           past_key_values=prefill_cache, use_cache=True, return_dict=True)
        if prefill.past_key_values is not prefill_cache or prefill_cache.get_seq_length() != p:
            raise ValueError("HF prefill did not return the complete supplied prefix cache")
        frozen = _cache_pairs(prefill_cache, config, p, device)
        cache_bytes = sum(key.numel() * key.element_size() + value.numel() * value.element_size()
                          for key, value in frozen)
        # Only fixed tensor snapshots remain. They are never passed into forward
        # directly, so even use_cache=False mutation cannot affect another chunk.
        del prefill, prefill_cache, prefix_mask
        for start in range(0, len(branches), chunk_size):
            selected = branches[start:start + chunk_size]
            b, s = len(selected), max(len(branch) for branch in selected)
            branch_lengths = torch.tensor([len(branch) for branch in selected], device=device)
            embeddings = torch.zeros((b, s, width), device=device, dtype=torch.float32)
            for row, branch in enumerate(selected):
                embeddings[row, :len(branch)] = branch
            mask = _branch_mask(branch_lengths, p, s, device)
            offset = torch.arange(p, p + s, device=device)
            cache = _fork_cache(frozen, config, b)
            hidden = backbone(inputs_embeds=embeddings, attention_mask=mask,
                              position_ids=offset[None].expand(b, -1), cache_position=offset,
                              past_key_values=cache, use_cache=True, return_dict=True).last_hidden_state
            if tuple(hidden.shape) != (b, s, width) or hidden.dtype != torch.float32:
                raise ValueError("teacher branch states must retain their complete FP32 shape")
            endpoint = hidden[torch.arange(b, device=device), branch_lengths - 1]
            output_scores.append(F.linear(endpoint, weight).squeeze(-1))
            padded_tokens += b * s - sum(len(branch) for branch in selected)
            max_mask = max(max_mask, mask.numel())
            max_chunk = max(max_chunk, b)
            # HF outputs include the mutated cache. Release each chunk before
            # creating the next one; no independent candidate KV is retained.
            del cache, hidden, embeddings, mask
        scores = torch.cat(output_scores)
        if scores.dtype != torch.float32 or not bool(torch.isfinite(scores).all()):
            raise ValueError("nonfinite or non-FP32 teacher scores")
    branch_tokens = sum(len(branch) for branch in branches)
    return {"scores": scores, "logical_lengths": lengths,
            "statistics": {"protocol": PROTOCOL, "prefix_prefills": 1, "candidate_count": len(branches),
                           "candidate_chunk_size": chunk_size,
                           "candidate_chunks": (len(branches) + chunk_size - 1) // chunk_size,
                           "prefix_tokens": p, "branch_tokens": branch_tokens,
                           "reader_tokens": p + branch_tokens + padded_tokens,
                           "reader_padding_tokens": padded_tokens, "prefix_cache_bytes": cache_bytes,
                           "max_chunk_prefix_cache_bytes": max_chunk * cache_bytes,
                           "max_branch_mask_elements": max_mask,
                           "repeated_prefix_reference_tokens": p * len(branches),
                           "encoder_unique_items": 0, "encoder_tokens": 0, "encoder_padding_tokens": 0,
                           "reader_packs": 1 + (len(branches) + chunk_size - 1) // chunk_size,
                           "compute_dtype": "float32", "attention_backend": "sdpa", "tf32": False}}


@torch.inference_mode()
def score_teacher_query(ranker, query, items, *, reader=None, head=None,
                        candidate_chunk_size=8, input_assembly="legacy"):
    """Reuse the exact T compiler and return the single-query forward contract.

    reader may be a standalone HF Qwen3Model or Qwen3ForCausalLM with identical
    frozen parameters. A flex/compiled trainer reader must never be passed here.
    This convenience wrapper obtains input embeddings from ranker.b without
    changing its backend; it never calls the training backbone's forward.
    Batched assembly changes only native embedding lookup, never the source
    ranker's training input_assembly or separately tokenized prompt segments.
    """
    if not isinstance(ranker, nn.Module) or getattr(ranker, "variant", None) != "T" or ranker.training:
        raise ValueError("prefix KV scoring requires an eval T ranker")
    if getattr(ranker, "shared_candidate_header", False) or getattr(ranker, "score_readout", None) != "legacy":
        raise ValueError("prefix KV scoring retains the legacy T candidate/suffix structure")
    input_assembly = _input_assembly(ranker, input_assembly)
    reader = ranker.b if reader is None else reader
    head = ranker.head if head is None else head
    backbone = _backbone(reader)
    if any(parameter.dtype != torch.float32 for parameter in ranker.b.parameters()):
        raise ValueError("T prompt embeddings must come from original FP32 parameters")
    candidates = getattr(query, "candidate_ids", None)
    history = getattr(query, "history", None)
    if (not isinstance(candidates, (list, tuple)) or not candidates or len(set(candidates)) != len(candidates)
            or not isinstance(history, (list, tuple)) or len(history) > 10):
        raise ValueError("teacher query requires unique candidates and at most 10 history entries")
    if any(item not in items or not isinstance(items[item], str) or not items[item].strip()
           for item in [h.item_id for h in history] + list(candidates)):
        raise ValueError("teacher query references missing canonical item content")
    device = next(backbone.parameters()).device
    with _fp32_inference(device.type):
        if input_assembly == "batched":
            # The helper caches integer-ID tensors on the source ranker. Keep
            # them ordinary tensors so a later batched training lookup can
            # safely save those IDs for embedding backward; retain no graph.
            with torch.inference_mode(False), torch.no_grad():
                native = ranker._batched_native_embeddings([query], items)
            prefix, branches = ranker._query_embeddings(query, items, native_embeddings=native)
        else:
            # Original frozen T implementations do not have the new helper or
            # its keyword. The default invokes their exact original API.
            prefix, branches = ranker._query_embeddings(query, items)
        result = score_prefix_kv(reader, head, prefix, branches, candidate_chunk_size=candidate_chunk_size,
                                 max_logical_tokens=ranker.max_logical_tokens)
    if input_assembly == "batched":
        result["statistics"]["input_assembly"] = input_assembly
    return {**result, "scores": [result["scores"]], "logical_lengths": [list(result["logical_lengths"])]}


class TeacherPrefixKVScorer:
    """Independent SDPA structure sharing a synchronous eval T's FP32 storage.

    In-place optimizer updates remain visible between calls. Replacing a source
    parameter/storage requires a new scorer. Training concurrently with a call
    is unsupported and detected through modes and tensor version counters.
    No prefix or candidate KV survives a query.
    """

    def __init__(self, ranker, candidate_chunk_size=8, *, input_assembly="legacy"):
        from copy import deepcopy
        from transformers import Qwen3Model

        self.candidate_chunk_size = _positive(candidate_chunk_size, "candidate_chunk_size")
        if self.candidate_chunk_size > 16:
            raise ValueError("candidate_chunk_size must be at most 16; no automatic resizing")
        self.ranker = ranker
        backbone = self._source_backbone()
        self.input_assembly = _input_assembly(ranker, input_assembly)
        self._source_reader = ranker.b
        self._source_backbone_module = backbone
        self._source_head = ranker.head
        self._source_config = deepcopy(backbone.config.to_dict())
        self._source_backend = backbone.config._attn_implementation
        source_parameters = tuple(backbone.parameters())
        if not source_parameters:
            raise ValueError("teacher source needs real FP32 parameters")
        self.device = source_parameters[0].device
        if self.device.type not in {"cpu", "cuda"} or any(
                parameter.dtype != torch.float32 or parameter.device != self.device
                for parameter in source_parameters):
            raise ValueError("teacher source parameters must be FP32 on one CPU/CUDA device")
        _head_weight(ranker.head, backbone.config.hidden_size, self.device)
        self._source_metadata = self._metadata()
        self._source_buffer_versions = tuple((name, value._version) for name, value in backbone.named_buffers())
        self._versions()  # Reject versionless inference tensors before initialization.

        config = deepcopy(backbone.config)
        config._attn_implementation = "sdpa"
        # CPU construction gives RoPE's nonpersistent buffers real values. A
        # meta skeleton would leave them uninitialized after assign=True.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            self.reader = Qwen3Model(config).float().eval()
            self.head = nn.Linear(config.hidden_size, 1, bias=False, dtype=torch.float32).eval()
        self.reader.load_state_dict(backbone.state_dict(), strict=True, assign=True)
        self.head.load_state_dict(ranker.head.state_dict(), strict=True, assign=True)
        self.reader.requires_grad_(False)
        self.head.requires_grad_(False)
        # Parameters already share the source device. Move only the separately
        # initialized buffers; .to() on the whole model could break that alias.
        source_buffers = dict(backbone.named_buffers())
        for name, buffer in tuple(self.reader.named_buffers()):
            module_name, _, member = name.rpartition(".")
            module = self.reader.get_submodule(module_name)
            buffer = buffer.to(device=self.device)
            with torch.no_grad():
                buffer.copy_(source_buffers[name])
            module._buffers[member] = buffer
        self.reader.rotary_emb.original_inv_freq = self.reader.rotary_emb.inv_freq
        self._owned_modules = (self.reader, self.head)
        self._owned_binding = self._reader_binding()
        _backbone(self.reader)
        self._verify_source()
        source = dict(backbone.named_parameters())
        if any(parameter is source[name] or self._storage(parameter) != self._storage(source[name])
               for name, parameter in self.reader.named_parameters()):
            raise ValueError("independent reader must own Parameters sharing the exact source storage")
        if self.head.weight is ranker.head.weight or self._storage(self.head.weight) != self._storage(ranker.head.weight):
            raise ValueError("independent scalar head must share source storage without its Parameter object")

    def _source_backbone(self):
        ranker = self.ranker
        if (not isinstance(ranker, nn.Module) or getattr(ranker, "variant", None) != "T"
                or ranker.training or hasattr(ranker, "_orig_mod")):
            raise ValueError("prefix KV scorer requires an uncompiled eval T ranker")
        if getattr(ranker, "shared_candidate_header", False) or getattr(ranker, "score_readout", None) != "legacy":
            raise ValueError("prefix KV scorer retains the legacy T candidate/suffix structure")
        reader = ranker.b
        backbone = getattr(reader, "model", reader)
        config = getattr(backbone, "config", None)
        if (not isinstance(reader, nn.Module) or hasattr(reader, "_orig_mod") or hasattr(backbone, "_orig_mod")
                or config is None or config.model_type != "qwen3" or not hasattr(backbone, "layers")):
            raise ValueError("teacher source requires an uncompiled real HF Qwen3 backbone")
        if any(module.training for module in reader.modules()):
            raise ValueError("teacher source must stay in evaluation mode")
        if (getattr(config, "attention_dropout", 0) != 0 or getattr(config, "use_sliding_window", False)
                or any(kind != "full_attention" for kind in getattr(config, "layer_types", []))):
            raise ValueError("teacher source requires zero-dropout full causal attention")
        rope = getattr(config, "rope_scaling", None)
        if rope and rope.get("rope_type", rope.get("type", "default")) != "default":
            raise ValueError("non-default RoPE needs separate prefix-sharing validation")
        return backbone

    @staticmethod
    def _storage(value):
        return (value.untyped_storage().data_ptr(), value.storage_offset(), tuple(value.shape),
                tuple(value.stride()), value.dtype, value.device)

    def _source_tensors(self):
        backbone = getattr(self.ranker.b, "model", self.ranker.b)
        return [("reader:"+name, value) for name, value in backbone.named_parameters()] + [
            ("buffer:"+name, value) for name, value in backbone.named_buffers()] + [
            ("head:"+name, value) for name, value in self.ranker.head.named_parameters()]

    def _metadata(self):
        return tuple((name, id(value), self._storage(value)) for name, value in self._source_tensors())

    def _versions(self):
        try:
            return tuple(value._version for _, value in self._source_tensors())
        except RuntimeError as error:
            raise ValueError("teacher source requires versioned tensors for synchronous inference") from error

    def _reader_binding(self):
        parameters = [("reader:"+name, value) for name, value in self.reader.named_parameters()] + [
            ("head:"+name, value) for name, value in self.head.named_parameters()]
        # Parameter versions may legitimately change with the shared optimizer
        # storage. Independent, nonpersistent buffers must stay fixed instead.
        return (tuple((name, id(value), self._storage(value), value.requires_grad) for name, value in parameters),
                tuple((name, id(value), self._storage(value), value._version)
                      for name, value in self.reader.named_buffers()))

    def _verify_source(self):
        backbone = self._source_backbone()
        if (self.ranker.b is not self._source_reader or backbone is not self._source_backbone_module
                or self.ranker.head is not self._source_head or self._metadata() != self._source_metadata
                or backbone.config.to_dict() != self._source_config
                or backbone.config._attn_implementation != self._source_backend):
            raise ValueError("teacher source parameter/storage/config changed; construct a new scorer")
        if tuple((name, value._version) for name, value in backbone.named_buffers()) != self._source_buffer_versions:
            raise ValueError("teacher source buffer changed; construct a new scorer")
        if (self.reader is not self._owned_modules[0] or self.head is not self._owned_modules[1]
                or self._reader_binding() != self._owned_binding):
            raise ValueError("independent teacher reader/head binding changed; construct a new scorer")
        _head_weight(self.ranker.head, backbone.config.hidden_size, self.device)
        return self._versions()

    def _score_query(self, query, items):
        return score_teacher_query(self.ranker, query, items, reader=self.reader, head=self.head,
                                   candidate_chunk_size=self.candidate_chunk_size, input_assembly=self.input_assembly)

    def score_query(self, query, items):
        versions = self._verify_source()
        result = self._score_query(query, items)
        if self._verify_source() != versions:
            raise ValueError("concurrent teacher updates during prefix KV scoring are unsupported")
        return result

    def __call__(self, queries, items):
        queries = tuple(queries)
        if not queries:
            raise ValueError("teacher scoring requires at least one query")
        versions = self._verify_source()
        scores, lengths, statistics = [], [], None
        resource_fields = {"prefix_cache_bytes", "max_chunk_prefix_cache_bytes", "max_branch_mask_elements"}
        constants = {"protocol", "candidate_chunk_size", "compute_dtype", "attention_backend", "tf32", "input_assembly"}
        for query in queries:
            result = self._score_query(query, items)
            scores.extend(result["scores"])
            lengths.extend(result["logical_lengths"])
            row = result["statistics"]
            if statistics is None:
                statistics = dict(row)
            else:
                for name, value in row.items():
                    if name in constants:
                        if statistics[name] != value:
                            raise ValueError("teacher batch changed its scoring protocol")
                    elif name in resource_fields:
                        statistics[name] = max(statistics[name], value)
                    else:
                        statistics[name] += value
        if self._verify_source() != versions:
            raise ValueError("concurrent teacher updates during prefix KV scoring are unsupported")
        # Queries are scored sequentially: work/token counts add, while live KV
        # and mask resource bounds are the largest single-query values.
        statistics["query_count"] = len(queries)
        return {"scores": scores, "logical_lengths": lengths, "statistics": statistics}
