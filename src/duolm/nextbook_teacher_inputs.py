"""Exact native-text inputs for a frozen full-text scalar teacher.

The history prefix and every candidate header, content and score suffix are
separately tokenized, as in the frozen trainer. Shared batches partition only
candidates; they preserve all text and original candidate order. Logical
positions exclude MIS carrier delimiters and reset at the shared prefix for
every branch. The scalar readout is the final suffix token.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from types import MappingProxyType

from .nextbook_export import _digest, _hash, _require
from .nextbook_teacher_export import TEMPLATE, load_catalog, verify_nextbook_teacher_export

MAX_HISTORY = 10
MAX_CANDIDATES = 100


def _positive(value, name):
    _require(type(value) is int and value > 0, name + " must be a positive integer")
    return value


@dataclass(frozen=True)
class TeacherScoreBatch:
    query_id: str
    candidate_ids: tuple[str, ...]
    candidate_indices: tuple[int, ...]
    query_ids: tuple[int, ...]
    item_ids: tuple[tuple[int, ...], ...]
    input_hash: str
    full_input_hash: str
    batch_hash: str
    logical_lengths: tuple[int, ...]
    packed_length: int

    def score_kwargs(self):
        return {"query": list(self.query_ids), "items": [list(row) for row in self.item_ids],
                "label_token_ids": [0], "apply_softmax": True, "item_first": False}


@dataclass(frozen=True)
class NextBookTeacherRequest:
    query_id: str
    candidate_ids: tuple[str, ...]
    query_ids: tuple[int, ...]
    item_ids: tuple[tuple[int, ...], ...]
    input_hash: str
    logical_lengths: tuple[int, ...]
    packed_length: int
    packed_limit: int

    def independent_sequences(self):
        """Native causal sequences; no carrier tokens or tree layout code."""
        return tuple(self.query_ids + branch for branch in self.item_ids)

    def independent_causal_batches(self, padded_token_budget, max_batch_size=32):
        """Return contiguous candidate indices bounded by actual padded cost."""
        _positive(padded_token_budget, "padded_token_budget")
        _positive(max_batch_size, "max_batch_size")
        batches, current, longest = [], [], 0
        for index, length in enumerate(self.logical_lengths):
            _require(length <= padded_token_budget, "One teacher sequence exceeds independent token budget; no truncation")
            if current and (max(longest, length) * (len(current) + 1) > padded_token_budget or len(current) == max_batch_size):
                batches.append(tuple(current))
                current, longest = [], 0
            current.append(index)
            longest = max(longest, length)
        if current:
            batches.append(tuple(current))
        return tuple(batches)

    def _slice(self, indices):
        branches = tuple(self.item_ids[index] for index in indices)
        signature = _digest({"input_hash": self.input_hash, "candidate_indices": indices})
        return TeacherScoreBatch(
            self.query_id, tuple(self.candidate_ids[index] for index in indices), tuple(indices),
            self.query_ids, branches, signature, self.input_hash, signature,
            tuple(self.logical_lengths[index] for index in indices),
            len(self.query_ids) + 1 + sum(len(row) + 1 for row in branches))

    def independent_batches(self, packed_token_budget=None):
        """One complete single-candidate carrier request for the SG baseline.

        This uses the same backend/scalar carrier as shared serving, while
        duplicating the entire history prefix for each candidate. It is separate
        from independent_causal_batches used by the PyTorch causal reference.
        """
        budget = self.packed_limit if packed_token_budget is None else _positive(packed_token_budget, "packed_token_budget")
        result = tuple(self._slice((index,)) for index in range(len(self.candidate_ids)))
        _require(all(batch.packed_length <= budget for batch in result),
                 "One teacher candidate path exceeds independent physical token budget; no truncation")
        return result

    def shared_batches(self, packed_token_budget=None):
        """Partition full candidate group explicitly, counting MIS delimiters.

        Each returned slice retains full_input_hash and original candidate
        indices. The caller reassembles raw scores and applies group operations
        once over all candidates, never separately within a physical slice.
        """
        budget = self.packed_limit if packed_token_budget is None else _positive(packed_token_budget, "packed_token_budget")
        prefix = len(self.query_ids) + 1
        groups, current, used = [], [], prefix
        for index, branch in enumerate(self.item_ids):
            cost = len(branch) + 1
            _require(prefix + cost <= budget, "One teacher candidate path exceeds MIS physical token budget; no truncation")
            if current and used + cost > budget:
                groups.append(tuple(current))
                current, used = [], prefix
            current.append(index)
            used += cost
        if current:
            groups.append(tuple(current))
        return tuple(self._slice(indices) for indices in groups)

    def score_kwargs(self):
        _require(self.packed_length <= self.packed_limit,
                 "Full teacher request exceeds physical budget; call shared_batches explicitly")
        return self.shared_batches()[0].score_kwargs()


class NextBookTeacherCompiler:
    def __init__(self, export_root, tokenizer=None, *, token_cache_capacity=32768, pretokenize_catalog=False):
        _require(type(pretokenize_catalog) is bool, "pretokenize_catalog must be a boolean")
        self.export_root = Path(export_root)
        manifest = verify_nextbook_teacher_export(self.export_root)
        self.manifest_signature = manifest["manifest_signature"]
        self.vocab_size = _positive(manifest["native_reader_config"]["vocab_size"], "vocab_size")
        metadata = manifest["model_metadata"]
        self.logical_limit = min(_positive(metadata["packing_limits"]["logical_context"], "logical context"),
                                 _positive(manifest["native_reader_config"]["max_position_embeddings"], "native context"))
        self.packed_limit = _positive(metadata["packing_limits"]["reader_packed_tokens"], "reader packed tokens")
        catalog_path = self.export_root / manifest["catalog_file"]
        self._items = MappingProxyType(load_catalog(catalog_path))
        _require(_hash(catalog_path) == manifest["catalog_sha256"], "teacher catalog changed during compilation startup")
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.export_root / "reader", local_files_only=True, trust_remote_code=False)
        self._tokenizer = tokenizer
        self._token_cache_capacity = _positive(token_cache_capacity, "token_cache_capacity")
        self._token_cache = OrderedDict()
        self._token_cache_hits = self._token_cache_misses = 0
        self._pretokenize_catalog = pretokenize_catalog
        self._startup_tokenization = None
        template = manifest["model_metadata"]["teacher_template"]
        self._instruction = self._native(template["instruction"])
        self._history_headers = tuple(self._native(template["history_header"].format(index=index))
                                      for index in range(1, MAX_HISTORY + 1))
        self._ratings = {rating: self._native(template["rating"].format(rating=rating)) for rating in range(1, 6)}
        self._candidate_header = self._native(template["candidate_header"])
        self._score_suffix = self._native(template["score_suffix"])
        if pretokenize_catalog:
            self._pretokenize_items()

    def _checked_tokens(self, tokens):
        result = tuple(tokens)
        _require(result and all(type(token) is int and 0 <= token < self.vocab_size for token in result),
                 "Teacher prompt segment is empty or outside native vocabulary")
        return result

    def _native(self, text):
        return self._checked_tokens(self._tokenizer.encode(text, add_special_tokens=False))

    def _pretokenize_items(self):
        """Encode each content segment alone, before any timed query or GPU work."""
        batch_encode = getattr(self._tokenizer, "batch_encode_plus", None)
        self._startup_tokenization = "batch_encode_plus" if callable(batch_encode) else "encode"
        cache, items = {}, iter(self._items.items())
        while batch := list(islice(items, 256)):
            if callable(batch_encode):
                encoded = batch_encode([text for _, text in batch], add_special_tokens=False,
                                       padding=False, truncation=False,
                                       return_attention_mask=False, return_token_type_ids=False)["input_ids"]
                _require(len(encoded) == len(batch), "Teacher catalog tokenizer changed batch size")
                values = [self._checked_tokens(tokens) for tokens in encoded]
            else:
                # Small injected tokenizers may implement encode only.
                values = [self._native(text) for _, text in batch]
            cache.update((item, tokens) for (item, _), tokens in zip(batch, values))
        self._token_cache = MappingProxyType(cache)
        self._token_cache_capacity = len(cache)

    def _content(self, item):
        _require(type(item) is str and item in self._items, "Unknown teacher book ID: " + repr(item))
        if item in self._token_cache:
            self._token_cache_hits += 1
            if not self._pretokenize_catalog:
                self._token_cache.move_to_end(item)
            return self._token_cache[item]
        self._token_cache_misses += 1
        value = self._native(self._items[item])
        self._token_cache[item] = value
        if len(self._token_cache) > self._token_cache_capacity:
            self._token_cache.popitem(last=False)
        return value

    def token_cache_report(self):
        """Report token-cache state; this is never a model prefix/KV cache."""
        return {"scope": "immutable_catalog_native_token_ids", "capacity": self._token_cache_capacity,
                "entries": len(self._token_cache), "hits": self._token_cache_hits, "misses": self._token_cache_misses,
                "policy": "pretokenized_catalog" if self._pretokenize_catalog else "bounded_lru",
                "pretokenize_catalog": self._pretokenize_catalog,
                "eviction": "disabled" if self._pretokenize_catalog else "lru",
                "startup_tokenization": self._startup_tokenization}

    def compile(self, query):
        _require(isinstance(query, Mapping) and set(query) == {"query_id", "history", "candidate_ids"},
                 "Query must contain exactly query_id, history and candidate_ids")
        _require(type(query["query_id"]) is str and query["query_id"].strip(), "query_id must be a nonempty string")
        history, candidates = query["history"], query["candidate_ids"]
        _require(isinstance(history, list) and len(history) <= MAX_HISTORY, "History must contain at most 10 entries")
        _require(isinstance(candidates, list) and 1 <= len(candidates) <= MAX_CANDIDATES,
                 "Candidates must contain 1 to 100 unique book IDs")
        prefix, canonical_history = list(self._instruction), []
        for index, event in enumerate(history):
            _require(isinstance(event, Mapping) and set(event) == {"item_id", "rating"},
                     "History entries must contain exactly item_id and rating")
            rating = event["rating"]
            _require(type(rating) is int and 1 <= rating <= 5, "History rating must be an integer from 1 to 5")
            prefix.extend(self._history_headers[index])
            prefix.extend(self._content(event["item_id"]))
            prefix.extend(self._ratings[rating])
            canonical_history.append({"item_id": event["item_id"], "rating": rating})
        branches = tuple(self._candidate_header + self._content(item) + self._score_suffix for item in candidates)
        _require(len(set(candidates)) == len(candidates), "Candidates must contain unique book IDs")
        logical = tuple(len(prefix) + len(branch) for branch in branches)
        _require(max(logical) <= self.logical_limit, "Teacher path exceeds logical position limit; no truncation")
        canonical = {"query_id": query["query_id"], "history": canonical_history, "candidate_ids": list(candidates)}
        signature = _digest({"manifest_signature": self.manifest_signature, "query": canonical,
                             "query_ids": prefix, "item_ids": branches})
        return NextBookTeacherRequest(query["query_id"], tuple(candidates), tuple(prefix), branches, signature,
                                      logical, len(prefix) + 1 + sum(len(row) + 1 for row in branches), self.packed_limit)
