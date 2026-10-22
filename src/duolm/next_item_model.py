"""Differentiable shared-history ranking, without raw or alignment branches.

Each query is a causal tree: one history prefix and isolated candidate suffixes.
Independent A documents and complete query trees are packed without padding.
The SDPA mask is a small-input correctness reference; CUDA uses sparse FlexAttention.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from .ranking_packing import PackedTreeLayout, QueryLayout, packed_segments
from .resampler import CalibratedRMSBridge, PerceiverResampler


TEMPLATE_VERSION = "books-next-item-shared-tree-v1"
INSTRUCTION = "Predict the user's next book interaction from the past history.\n"
CANDIDATE_HEADER = "Candidate:\nItem: "
LEGACY_SCORE_SUFFIX = "\nNext-book score:"
V1_UNKNOWN_FEEDBACK_SEGMENTS = ("\nUser rating: ", "?", "\n")


def domain_template(domain="book"):
    """The Books default remains byte-for-byte identical to its frozen prompt."""
    if domain == "book":
        return {"version": TEMPLATE_VERSION, "instruction": INSTRUCTION,
                "score_suffix": LEGACY_SCORE_SUFFIX}
    if domain == "game":
        return {"version": "games-next-item-shared-tree-v1",
                "instruction": "Predict the user's next game-product interaction from the past history.\n",
                "score_suffix": "\nNext-game-product score:"}
    raise ValueError("ranking domain must be book or game")


def _last_valid_token(hidden, attention_mask):
    """Read the actual last token, including left or right padded references."""
    if hidden.ndim != 3 or tuple(attention_mask.shape) != tuple(hidden.shape[:2]):
        raise ValueError("readout expects [items, tokens, width] and a matching mask")
    mask = attention_mask.to(device=hidden.device, dtype=torch.bool)
    if not mask.any(-1).all():
        raise ValueError("every encoder item must contain a valid token")
    positions = torch.arange(hidden.shape[1], device=hidden.device).expand_as(mask)
    last = positions.masked_fill(~mask, -1).amax(-1)
    return hidden[torch.arange(len(hidden), device=hidden.device), last][:, None]


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def query_input_hash(query, items, *, domain="book"):
    """Candidate order and visible feedback are part of KD/checkpoint identity."""
    template = domain_template(domain)
    content = {"template": template["version"], "query_id": query.query_id,
                    "history": [(h.item_id, h.rating, h.timestamp, items[h.item_id]) for h in query.history],
                    "candidates": [(i, items[i]) for i in query.candidate_ids]}
    if domain == "game":
        content.update(domain=domain, prompt_template=template)
    return _digest(content)


def group_ranking_loss(scores, positive_indices, *, teacher_scores=None, beta=0.0, temperature=1.0,
                       defer_scalar_metrics=False):
    if isinstance(beta, bool) or not math.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("beta must be finite and in [0,1]")
    if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not scores or len(scores) != len(positive_indices):
        raise ValueError("one complete score group and positive index are required per query")
    if beta and (teacher_scores is None or len(teacher_scores) != len(scores)):
        raise ValueError("KD needs matching teacher score groups")
    if defer_scalar_metrics:
        return _deferred_group_ranking_loss(scores, positive_indices, teacher_scores=teacher_scores,
                                           beta=beta, temperature=temperature)
    ce, kd = [], []
    for index, (group, positive) in enumerate(zip(scores, positive_indices)):
        if group.ndim != 1 or group.numel() < 2 or not torch.isfinite(group).all():
            raise ValueError("candidate scores must be a finite vector with at least two entries")
        if isinstance(positive, bool) or not isinstance(positive, int) or not 0 <= positive < len(group):
            raise ValueError("training requires a valid positive index in each complete group")
        ce.append(F.cross_entropy(group.float()[None], torch.tensor([positive], device=group.device)))
        if beta:
            teacher = torch.as_tensor(teacher_scores[index], device=group.device).detach().float()
            if teacher.shape != group.shape or not torch.isfinite(teacher).all():
                raise ValueError("teacher candidate scores must have matching finite shape")
            kd.append(F.kl_div(F.log_softmax(group.float() / temperature, -1),
                               F.softmax(teacher / temperature, -1), reduction="sum") * temperature**2)
    mean_ce = torch.stack(ce).mean()
    mean_kd = torch.stack(kd).mean() if kd else mean_ce.new_zeros(())
    return {"loss": (1-beta)*mean_ce + beta*mean_kd, "ce": mean_ce.detach().item(),
            "kd": mean_kd.detach().item()}


def _deferred_group_ranking_loss(scores, positive_indices, *, teacher_scores, beta, temperature):
    """Preserve per-query arithmetic, deferring host reads to the update boundary.

    Even FP32 batched softmax/KL reductions can change nearly cancelled
    gradients, which Adam amplifies. Use the original kernel shapes and mean
    order while preparing labels and teacher rows only once per microbatch.
    """
    device = scores[0].device
    lengths = []
    for group, positive in zip(scores, positive_indices):
        if group.ndim != 1 or group.numel() < 2 or group.device != device:
            raise ValueError("candidate scores must be same-device vectors with at least two entries")
        if isinstance(positive, bool) or not isinstance(positive, int) or not 0 <= positive < len(group):
            raise ValueError("training requires a valid positive index in each complete group")
        lengths.append(len(group))
    targets = torch.tensor(positive_indices, device=device, dtype=torch.long)
    finite = torch.stack([torch.isfinite(group).all() for group in scores]).all()
    teacher_rows = None
    if beta:
        # Assemble on the teacher's existing device, then transfer the complete
        # batch once. Device-resident cache rows therefore never round-trip CPU.
        if isinstance(teacher_scores, torch.Tensor):
            if teacher_scores.ndim != 2 or any(n != teacher_scores.shape[1] for n in lengths):
                raise ValueError("teacher candidate scores must have matching shape")
            teacher = teacher_scores.detach().to(device=device, dtype=torch.float32)
            teacher_rows = teacher.unbind(0)
        elif all(isinstance(row, torch.Tensor) for row in teacher_scores):
            teacher_device = teacher_scores[0].device
            if any(row.ndim != 1 or len(row) != n or row.device != teacher_device
                   for row, n in zip(teacher_scores, lengths)):
                raise ValueError("teacher candidate scores must have matching shape and a shared device")
            teacher = torch.cat([row.detach().float() for row in teacher_scores]).to(device)
            teacher_rows = teacher.split(lengths)
        else:
            if any(isinstance(row, torch.Tensor) or len(row) != n
                   for row, n in zip(teacher_scores, lengths)):
                raise ValueError("teacher candidate scores must have matching shape")
            teacher = torch.tensor([value for row in teacher_scores for value in row],
                                   device=device, dtype=torch.float32)
            teacher_rows = teacher.split(lengths)
        finite = finite & torch.isfinite(teacher).all()
    ce, kd = [], []
    for index, group in enumerate(scores):
        ce.append(F.cross_entropy(group.float()[None], targets[index:index+1]))
        if beta:
            kd.append(F.kl_div(F.log_softmax(group.float() / temperature, -1),
                               F.softmax(teacher_rows[index] / temperature, -1),
                               reduction="sum") * temperature**2)
    mean_ce = torch.stack(ce).mean()
    mean_kd = torch.stack(kd).mean() if kd else mean_ce.new_zeros(())
    loss = (1-beta)*mean_ce + beta*mean_kd
    return {"loss": loss, "ce": mean_ce.detach(), "kd": mean_kd.detach(),
            "loss_finite": (finite & torch.isfinite(loss)).detach()}


def _atomic_packs(lengths, capacity):
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        raise ValueError("pack capacity must be a positive integer")
    packs, current, used = [], [], 0
    for index, length in enumerate(lengths):
        if length > capacity:
            raise ValueError(f"atomic item/query length {length} exceeds pack capacity {capacity}; no truncation")
        if current and used + length > capacity:
            packs.append(current)
            current, used = [], 0
        current.append(index)
        used += length
    if current:
        packs.append(current)
    return packs


class GroupRanker(nn.Module):
    def __init__(self, a, b, tokenizer_a, tokenizer_b, *, variant="T", m=8,
                 resampler_layers=2, resampler_heads=8, attention_backend="sdpa",
                 max_reader_packed_tokens=16384, max_encoder_packed_tokens=8192,
                 max_logical_tokens=8192, dense_reference_limit=4096, gradient_checkpointing=False,
                 a_readout="perceiver", shared_candidate_header=False, score_readout="legacy", domain="book",
                 input_assembly="legacy", reader_gradient_checkpointing=None):
        super().__init__()
        if variant not in {"T", "V1", "V3"}:
            raise ValueError("variant must be T, V1 or V3")
        if attention_backend not in {"sdpa", "flex_attention"}:
            raise ValueError("use sdpa reference or flex_attention production backend")
        if variant != "T" and (a is None or tokenizer_a is None):
            raise ValueError("students require A and its tokenizer")
        if a_readout not in {"perceiver", "last_token", "memory"}:
            raise ValueError("A readout must be perceiver, last_token or memory")
        if score_readout not in {"legacy", "candidate", "score_token"}:
            raise ValueError("score readout must be legacy, candidate or score_token")
        if not isinstance(shared_candidate_header, bool):
            raise ValueError("shared_candidate_header must be boolean")
        if input_assembly not in {"legacy", "batched"}:
            raise ValueError("input assembly must be legacy or batched")
        if reader_gradient_checkpointing is not None and not isinstance(reader_gradient_checkpointing, bool):
            raise ValueError("reader gradient checkpointing must be boolean or None")
        if variant == "T" and (a_readout != "perceiver" or shared_candidate_header or score_readout != "legacy"):
            raise ValueError("teacher must preserve its legacy prompt and score readout")
        if variant != "T" and a_readout != "perceiver" and m != 1:
            raise ValueError("last_token and memory A readouts require m=1")
        if variant != "T" and score_readout == "candidate" and m != 1:
            raise ValueError("candidate score readout requires m=1; multi-vector readout needs an explicit rule")
        for name, value in (("m", m), ("max_reader_packed_tokens", max_reader_packed_tokens),
                            ("max_encoder_packed_tokens", max_encoder_packed_tokens),
                            ("max_logical_tokens", max_logical_tokens), ("dense_reference_limit", dense_reference_limit)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.a, self.b = a, b
        self.domain, self.prompt_template = domain, domain_template(domain)
        self.tokenizer_a, self.tokenizer_b = tokenizer_a, tokenizer_b
        self.variant, self.m, self.attention_backend = variant, m, attention_backend
        self.a_readout, self.shared_candidate_header, self.score_readout = a_readout, shared_candidate_header, score_readout
        self.resampler_layers, self.resampler_heads = resampler_layers, resampler_heads
        self.max_reader_packed_tokens = max_reader_packed_tokens
        self.max_encoder_packed_tokens = max_encoder_packed_tokens
        self.max_logical_tokens, self.dense_reference_limit = max_logical_tokens, dense_reference_limit
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.reader_gradient_checkpointing = reader_gradient_checkpointing
        self.input_assembly = input_assembly
        self._token_cache = OrderedDict()
        # Only integer IDs are retained. Embedding values must be looked up
        # afresh after every optimizer update; neither cache enters state_dict.
        self._native_id_cache = {}
        self._item_token_cache = {}
        self._item_tokenizer_identity = None
        self._v1_unknown_feedback_ids = self._unknown_feedback_ids() if variant == "V1" else None
        for module in (a, b):
            if module is None:
                continue
            config = module.config
            if getattr(config, "attention_dropout", 0) or getattr(config, "use_sliding_window", False):
                raise ValueError("shared-tree equivalence requires zero dropout and full causal attention")
            rope = getattr(config, "rope_scaling", None)
            if rope and rope.get("rope_type", rope.get("type", "default")) != "default":
                raise ValueError("non-default RoPE requires a separate sharing validation")
            if getattr(config, "_attn_implementation", attention_backend) != attention_backend:
                raise ValueError("backbone attention implementation must match the ranking backend")
            config.use_cache = False
            module.requires_grad_(True)
            checkpointing = (gradient_checkpointing if module is a or reader_gradient_checkpointing is None
                             else reader_gradient_checkpointing)
            if checkpointing:
                module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            elif module is b and reader_gradient_checkpointing is False:
                module.gradient_checkpointing_disable()
        width_b = b.config.hidden_size
        self.head = nn.Linear(width_b, 1, bias=False, dtype=torch.float32)
        nn.init.normal_(self.head.weight, std=0.02)
        if variant != "T":
            width_a = a.config.hidden_size
            if a_readout == "perceiver":
                self.resampler = PerceiverResampler(width_a, m, resampler_layers, resampler_heads)
            elif a_readout == "memory":
                eos = tokenizer_a.eos_token_id
                initial = (a.get_input_embeddings().weight[eos].detach().float().clone() if eos is not None
                           else torch.empty(width_a, dtype=torch.float32).normal_(std=0.02))
                self.memory_token = nn.Parameter(initial)
            self.projection = nn.Identity() if width_a == width_b else nn.Linear(width_a, width_b, bias=False)
            self.bridge = CalibratedRMSBridge()
            if score_readout == "score_token":
                # One continuous B-width readout position; never a tokenizer expansion.
                self.score_token = nn.Parameter(self._native(self.prompt_template["score_suffix"])[-1].detach().float().clone())

    @classmethod
    def from_pretrained(cls, a_path, b_path, *, a_revision=None, b_revision=None,
                        variant="T", device="cuda", attention_backend="flex_attention", **kwargs):
        from transformers import AutoModel, AutoTokenizer
        tokenizer_b = AutoTokenizer.from_pretrained(b_path, revision=b_revision)
        b = AutoModel.from_pretrained(b_path, revision=b_revision, torch_dtype=torch.float32,
                                      attn_implementation=attention_backend)
        a, tokenizer_a = None, None
        if variant != "T":
            tokenizer_a = AutoTokenizer.from_pretrained(a_path, revision=a_revision)
            a = AutoModel.from_pretrained(a_path, revision=a_revision, torch_dtype=torch.float32,
                                          attn_implementation=attention_backend)
        return cls(a, b, tokenizer_a, tokenizer_b, variant=variant,
                   attention_backend=attention_backend, **kwargs).to(device)

    def _ids(self, tokenizer, text):
        key = (id(tokenizer), text)
        if key not in self._token_cache:
            ids = tuple(tokenizer.encode(text, add_special_tokens=False))
            if not ids:
                raise ValueError("native prompt segments must tokenize to nonempty input")
            self._token_cache[key] = ids
            if len(self._token_cache) > 32768:
                self._token_cache.popitem(last=False)
        else:
            self._token_cache.move_to_end(key)
        return self._token_cache[key]

    def _native(self, text):
        ids = torch.tensor(self._ids(self.tokenizer_b, text), device=self.head.weight.device)
        return self.b.get_input_embeddings()(ids)

    def prepare_item_tokens(self, items):
        """Pre-tokenize V3 canonical A items without retaining learned values.

        This dedicated CPU cache is not the bounded prompt LRU: a complete
        catalog remains available after preparation. V1's rating-dependent A
        content is deliberately left on its original path instead of silently
        reusing V3 IDs or multiplying the full catalog into six copies.
        """
        if self.variant != "V3":
            raise ValueError("catalog token preparation currently supports V3 only")
        prepared = {}
        eos = self.tokenizer_a.eos_token_id
        for item_id, content in items.items():
            if not isinstance(content, str) or not content:
                raise ValueError(f"missing nonempty canonical content for {item_id}")
            ids = tuple(self.tokenizer_a.encode("Item: " + content, add_special_tokens=False))
            if not ids:
                raise ValueError("native item content must tokenize to nonempty input")
            if eos is not None:
                ids += (eos,)
            prepared[item_id] = (content, ids)
        self._item_token_cache = prepared
        self._item_tokenizer_identity = id(self.tokenizer_a)
        return {"variant": "V3", "items": len(prepared),
                "tokens": sum(len(ids) for _, ids in prepared.values()),
                "includes_eos": eos is not None, "cache_values": "CPU token IDs only"}

    def _a_item_ids(self, item_id, content, rating=None):
        prepared = (self._item_token_cache.get(item_id)
                    if self.variant == "V3" and self._item_tokenizer_identity == id(self.tokenizer_a) else None)
        if prepared is not None and prepared[0] == content:
            return prepared[1]
        prefix = f"User rating: {rating}\n" if self.variant == "V1" and rating is not None else ""
        ids = self._ids(self.tokenizer_a, prefix + "Item: " + content)
        if self.tokenizer_a.eos_token_id is not None:
            ids += (self.tokenizer_a.eos_token_id,)
        return ids

    def _apply(self, fn, recurse=True):
        # nn.Module does not move tensors in plain dictionaries. Rebuild these
        # cheap ID tensors after device/dtype moves instead of retaining stale
        # device references or introducing persistent checkpoint buffers.
        self._native_id_cache.clear()
        return super()._apply(fn, recurse=recurse)

    def _batched_native_embeddings(self, queries, items):
        segments = OrderedDict()

        def add(text):
            segments.setdefault(self._ids(self.tokenizer_b, text), None)

        add(self.prompt_template["instruction"])
        for query in queries:
            for index, event in enumerate(query.history):
                add(f"History {index+1}:\nItem: ")
                if self.variant == "T":
                    add(items[event.item_id])
                if self.variant == "V1":
                    segments.setdefault(self._v1_unknown_feedback_ids, None)
                else:
                    add(f"\nUser rating: {event.rating}\n")
            add(CANDIDATE_HEADER)
            if self.variant == "T":
                for item in query.candidate_ids:
                    add(items[item])
            if self.score_readout == "legacy":
                add(self.prompt_template["score_suffix"])
        device = self.head.weight.device
        for ids in segments:
            if ids not in self._native_id_cache:
                self._native_id_cache[ids] = torch.tensor(ids, device=device, dtype=torch.long)
        values = self.b.get_input_embeddings()(torch.cat([self._native_id_cache[ids] for ids in segments]))
        cursor = 0
        for ids in segments:
            segments[ids] = values[cursor:cursor+len(ids)]
            cursor += len(ids)
        return segments

    def _unknown_feedback_ids(self):
        # Qwen BPE merges " ?\n" into one token, whereas numeric feedback
        # occupies three tokens (space, digit, newline). Keep the same text,
        # but tokenize the fixed unknown slot separately to preserve positions.
        before, unknown, after = (self._ids(self.tokenizer_b, segment)
                                  for segment in V1_UNKNOWN_FEEDBACK_SEGMENTS)
        if len(unknown) != 1:
            raise ValueError("V1 unknown-feedback control requires a single fixed unknown B token")
        for rating in range(1, 6):
            value = self._ids(self.tokenizer_b, str(rating))
            actual = self._ids(self.tokenizer_b, f"\nUser rating: {rating}\n")
            if len(value) != 1 or actual != before + value + after:
                raise ValueError("V1 unknown-feedback control must have the same B token length and positions "
                                 "as every rating from 1 to 5")
        return before + unknown + after

    def _mask(self, layout, dtype, device):
        if self.attention_backend == "flex_attention":
            if device.type != "cuda":
                raise ValueError("production flex_attention requires CUDA; use bounded sdpa for CPU reference")
            return layout.block_attention_mask(device=device)
        if layout.token_count > self.dense_reference_limit:
            raise ValueError("dense reference mask is too large; select sparse flex_attention, not an L-squared fallback")
        return layout.dense_attention_mask(dtype=dtype, device=device)

    def _run(self, backbone, inputs, layout):
        options = {}
        if self.variant != "T" and self.attention_backend == "flex_attention":
            # Compact trees can have fewer than 128 physical tokens. Force the
            # training/prefill kernel: Torch AUTO otherwise selects its decoding
            # template, which fails to produce a valid choice on this runtime.
            options["kernel_options"] = {"FORCE_USE_FLEX_ATTENTION": True}
        return backbone(inputs_embeds=inputs[None],
                        attention_mask=self._mask(layout, inputs.dtype, inputs.device),
                        position_ids=layout.position_ids.to(inputs.device),
                        use_cache=False, return_dict=True, **options).last_hidden_state[0]

    @torch.no_grad()
    def calibrate_bridge(self, queries, items, max_items=256):
        if self.variant == "T":
            return None
        seen, total, count = set(), 0.0, 0
        for query in queries:
            for item in [h.item_id for h in query.history] + list(query.candidate_ids):
                if item in seen:
                    continue
                seen.add(item)
                values = self._native(items[item]).float()
                total += values.square().sum().item()
                count += values.numel()
                if len(seen) >= max_items:
                    break
            if len(seen) >= max_items:
                break
        if not count:
            raise ValueError("bridge calibration needs training content")
        metadata = {"source": "training_query_content", "items": len(seen),
                    "content_hash": _digest([(i, items[i]) for i in sorted(seen)])}
        self.bridge.set_reference(math.sqrt(total/count), metadata)
        return self.bridge.metadata()

    def _a_blocks(self, queries, items):
        registry, references = OrderedDict(), []
        for query in queries:
            history, candidates = [], []
            for event in query.history:
                ids = self._a_item_ids(event.item_id, items[event.item_id], event.rating)
                registry.setdefault(ids, len(registry))
                history.append(registry[ids])
            for item in query.candidate_ids:
                ids = self._a_item_ids(item, items[item])
                registry.setdefault(ids, len(registry))
                candidates.append(registry[ids])
            references.append((history, candidates))
        blocks = list(registry)
        states = [None] * len(blocks)
        device = self.head.weight.device
        lengths = [len(ids) + (self.a_readout == "memory") for ids in blocks]
        for indices in _atomic_packs(lengths, self.max_encoder_packed_tokens):
            selected = [blocks[i] for i in indices]
            layout = packed_segments([lengths[i] for i in indices])
            ids = torch.tensor([token for sequence in selected for token in sequence], device=device)
            values = self.a.get_input_embeddings()(ids)
            if self.a_readout == "memory":
                embeddings, cursor = [], 0
                for sequence in selected:
                    stop = cursor + len(sequence)
                    embeddings.extend((values[cursor:stop], self.memory_token[None].to(values.dtype)))
                    cursor = stop
                values = torch.cat(embeddings)
            hidden = self._run(self.a, values, layout)
            for index, (start, stop) in zip(indices, layout.ranges):
                segment = hidden[start:stop][None]
                if self.a_readout == "perceiver":
                    mask = torch.ones(segment.shape[:2], dtype=torch.bool, device=device)
                    readout = self.resampler(segment, mask, [self.m])[0]
                else:
                    # These exact segment ranges contain no padding. Direct
                    # endpoint indexing avoids a per-item GPU mask/sync.
                    readout = segment[0, -1:]
                if self.input_assembly == "legacy":
                    states[index] = self.bridge(self.projection(readout))
                elif isinstance(self.projection, nn.Identity):
                    states[index] = readout
                else:
                    # Keep learned Linear's original per-item GEMMs. Under
                    # BF16 autocast, one batched weight-gradient reduction can
                    # round differently from the separate GEMMs followed by
                    # FP32 accumulation, affecting near-zero Adam directions.
                    states[index] = self.projection(readout)
        if self.input_assembly == "batched":
            # Readouts remain isolated per item (no padded Perceiver batch or
            # extra hidden-state materialization). Equal-width Identity has
            # one call; FP32 RMS shares one calibrated-flag validation in both
            # cases, while learned projections preserve legacy GEMM shapes.
            projected = torch.cat(states)
            if isinstance(self.projection, nn.Identity):
                projected = self.projection(projected)
            projected = self.bridge(projected)
            states = list(projected.split([len(state) for state in states]))
        return states, references, {"encoder_unique_items": len(blocks),
                                    "encoder_tokens": sum(lengths), "encoder_padding_tokens": 0}

    def _query_embeddings(self, query, items, states=None, references=None, native_embeddings=None):
        # Existing Books compilers also call this helper on a minimal adapter.
        template = getattr(self, "prompt_template", None) or domain_template(getattr(self, "domain", "book"))
        native = (self._native if native_embeddings is None
                  else lambda text: native_embeddings[self._ids(self.tokenizer_b, text)])
        prefix = [native(template["instruction"])]
        unknown_feedback = None
        if self.variant == "V1":
            if native_embeddings is None:
                ids = torch.tensor(self._v1_unknown_feedback_ids, device=self.head.weight.device)
                unknown_feedback = self.b.get_input_embeddings()(ids)
            else:
                unknown_feedback = native_embeddings[self._v1_unknown_feedback_ids]
        for index, event in enumerate(query.history):
            prefix.append(native(f"History {index+1}:\nItem: "))
            if self.variant == "T":
                prefix.append(native(items[event.item_id]))
            else:
                prefix.append(states[references[0][index]].to(self.b.get_input_embeddings().weight.dtype))
            prefix.append(unknown_feedback if self.variant == "V1"
                          else native(f"\nUser rating: {event.rating}\n"))
        # Keep the exact separately tokenized legacy header sequence and logical
        # positions; moving its states into the prefix changes sharing only.
        if self.shared_candidate_header:
            prefix.append(native(CANDIDATE_HEADER))
        branches = []
        for index, item in enumerate(query.candidate_ids):
            branch = [] if self.shared_candidate_header else [native(CANDIDATE_HEADER)]
            if self.variant == "T":
                branch.append(native(items[item]))
            else:
                branch.append(states[references[1][index]].to(self.b.get_input_embeddings().weight.dtype))
            if self.score_readout == "legacy":
                branch.append(native(template["score_suffix"]))
            elif self.score_readout == "score_token":
                branch.append(self.score_token[None].to(self.b.get_input_embeddings().weight.dtype))
            branches.append(torch.cat(branch))
        return torch.cat(prefix), branches

    def forward(self, queries, items, *, max_reader_packed_tokens=None):
        if not queries:
            raise ValueError("empty query batch")
        for query in queries:
            for item in [h.item_id for h in query.history] + list(query.candidate_ids):
                if item not in items or not isinstance(items[item], str) or not items[item]:
                    raise ValueError(f"missing nonempty canonical content for {item}")
        states, references, stats = None, None, {"encoder_unique_items": 0, "encoder_tokens": 0,
                                                "encoder_padding_tokens": 0}
        if self.variant != "T":
            states, references, stats = self._a_blocks(queries, items)
        native_embeddings = (self._batched_native_embeddings(queries, items)
                             if self.input_assembly == "batched" else None)
        trees, layouts = [], []
        logical_lengths = []
        for index, query in enumerate(queries):
            prefix, branches = self._query_embeddings(query, items, states,
                                                       None if references is None else references[index],
                                                       native_embeddings)
            lengths = [len(prefix)+len(branch) for branch in branches]
            if max(lengths) > min(self.max_logical_tokens, self.b.config.max_position_embeddings):
                raise ValueError("query path exceeds logical context; no silent history truncation")
            trees.append(torch.cat([prefix]+branches))
            layouts.append(QueryLayout(len(prefix), tuple(len(b) for b in branches)))
            logical_lengths.append(lengths)
        capacity = self.max_reader_packed_tokens if max_reader_packed_tokens is None else max_reader_packed_tokens
        scores = [None] * len(queries)
        packs = _atomic_packs([len(tree) for tree in trees], capacity)
        for indices in packs:
            layout = PackedTreeLayout.from_queries([layouts[i] for i in indices])
            hidden = self._run(self.b, torch.cat([trees[i] for i in indices]), layout)
            for index, score_indices in zip(indices, layout.score_indices):
                endpoints = hidden[torch.tensor(score_indices, device=hidden.device)]
                with torch.autocast(device_type=hidden.device.type, enabled=False):
                    scores[index] = F.linear(endpoints.float(), self.head.weight.float()).squeeze(-1)
        stats.update(reader_tokens=sum(map(len, trees)), reader_padding_tokens=0,
                     reader_packs=len(packs), prefix_tokens=sum(q.prefix_length for q in layouts),
                     repeated_prefix_reference_tokens=sum(q.prefix_length*len(q.branch_lengths) for q in layouts))
        return {"scores": scores, "statistics": stats, "logical_lengths": logical_lengths}

    def loss(self, queries, items, *, teacher_scores=None, beta=0.0, temperature=1.0,
             defer_scalar_metrics=False):
        if self.variant == "T" and (beta or teacher_scores is not None):
            raise ValueError("teacher trains next-item CE without a teacher/raw/alignment branch")
        output = self.forward(queries, items)
        output.update(group_ranking_loss(output["scores"], [q.positive_index for q in queries],
                                        teacher_scores=teacher_scores, beta=beta, temperature=temperature,
                                        defer_scalar_metrics=defer_scalar_metrics))
        return output

    def metadata(self):
        metadata = {"template": self.prompt_template["version"], "variant": self.variant, "m": self.m,
                "attention_backend": self.attention_backend, "shared_history": True,
                "a_query_dedup": True, "alignment": False, "raw_branch": False,
                "reader_packed_tokens": self.max_reader_packed_tokens,
                "encoder_packed_tokens": self.max_encoder_packed_tokens,
                "logical_context": self.max_logical_tokens,
                "a_width": None if self.a is None else self.a.config.hidden_size,
                "b_width": self.b.config.hidden_size,
                "bridge": None if self.variant == "T" else self.bridge.metadata()}
        if self.variant != "T":
            template = {"version": "books-next-item-student-interface-v2" if self.domain == "book" else "games-next-item-student-interface-v1",
                        "instruction": self.prompt_template["instruction"],
                        "variant": self.variant,
                        "feedback_channel": "a_rating_b_unknown" if self.variant == "V1" else "b_explicit_rating",
                        "candidate_header": CANDIDATE_HEADER, "shared_candidate_header": self.shared_candidate_header,
                        "score_readout": self.score_readout,
                        "score_suffix": self.prompt_template["score_suffix"] if self.score_readout == "legacy" else None,
                        "a_readout": self.a_readout, "m": self.m,
                        "a_eos": self.tokenizer_a.eos_token_id,
                        "a_memory_positions": int(self.a_readout == "memory"),
                        "b_score_positions": int(self.score_readout == "score_token"),
                        "flex_attention_kernel": "forced_prefill" if self.attention_backend == "flex_attention" else None,
                        "resampler_layers": self.resampler_layers if self.a_readout == "perceiver" else None,
                        "resampler_heads": self.resampler_heads if self.a_readout == "perceiver" else None}
            if self.variant == "V1":
                template["b_unknown_feedback"] = {
                    "text": "".join(V1_UNKNOWN_FEEDBACK_SEGMENTS),
                    "tokenization": "fixed_segments_v1",
                    "segments": list(V1_UNKNOWN_FEEDBACK_SEGMENTS),
                    "token_ids": list(self._v1_unknown_feedback_ids),
                }
            metadata.update(a_readout=self.a_readout, shared_candidate_header=self.shared_candidate_header,
                            score_readout=self.score_readout,
                            student_template={**template, "sha256": _digest(template)})
        if self.domain == "game":
            metadata.update(domain=self.domain, prompt_template={**self.prompt_template, "sha256": _digest(self.prompt_template)})
        if self.input_assembly != "legacy" or self.reader_gradient_checkpointing is not None:
            metadata.update(input_assembly=self.input_assembly,
                            reader_gradient_checkpointing=(self.gradient_checkpointing
                                if self.reader_gradient_checkpointing is None else self.reader_gradient_checkpointing))
        return metadata
