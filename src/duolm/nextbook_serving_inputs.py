"""Compile compact V3 ranking inputs from IDs and one verified offline export.

Only the native B prompt fragments are tokenized, once at startup. Book content
is represented by the export's persisted one-vector IDs; requests never encode
book text, load model weights, or reuse a cached whole query.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType


INSTRUCTION = "Predict the user's next book interaction from the past history.\n"
CANDIDATE_HEADER = "Candidate:\nItem: "
MAX_HISTORY = 10
MAX_CANDIDATES = 100


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _positive_int(value, name):
    _require(type(value) is int and value > 0, name + " must be a positive integer")
    return value


def _template_guard(manifest):
    metadata = manifest.get("model_metadata", {})
    template = metadata.get("student_template", {})
    domain = metadata.get("domain", "book")
    _require(domain in {"book", "game"}, "Unknown compact model domain")
    from .next_item_model import domain_template
    instruction = domain_template(domain)["instruction"]
    version = "books-next-item-student-interface-v2" if domain == "book" else "games-next-item-student-interface-v1"
    expected = {"version": version, "instruction": instruction,
                "variant": "V3", "m": 1, "feedback_channel": "b_explicit_rating",
                "a_readout": "last_token", "shared_candidate_header": True,
                "candidate_header": CANDIDATE_HEADER, "score_readout": "candidate", "score_suffix": None,
                "a_memory_positions": 0, "b_score_positions": 0,
                "resampler_layers": None, "resampler_heads": None}
    _require(isinstance(template, dict) and all(template.get(key) == value for key, value in expected.items()),
             "Only the frozen m1 V3 last_token/shared-header/candidate interface is supported")
    _require(type(template.get("m")) is int and template.get("shared_candidate_header") is True,
             "Invalid compact student template dimensions")
    _require(template.get("sha256") == _hash({key: value for key, value in template.items() if key != "sha256"}),
             "Student template identity signature differs")
    _require(manifest.get("variant") == metadata.get("variant") == "V3"
             and manifest.get("m") == metadata.get("m") == 1
             and type(manifest.get("m")) is int and type(metadata.get("m")) is int,
             "Export variant/m differs from the compact student template")
    for key in ("a_readout", "shared_candidate_header", "score_readout"):
        _require(metadata.get(key) == expected[key], "Model metadata differs from student template: " + key)
    return metadata


@dataclass(frozen=True)
class NextBookRequest:
    """One shared native prefix with isolated one-book candidate branches.

    Logical lengths exclude MIS carrier delimiters. ``packed_length`` includes
    one query delimiter and one delimiter after every candidate. Candidate
    scores must come from the book endpoint immediately before each delimiter.
    """

    query_id: str
    candidate_ids: tuple[str, ...]
    query_ids: list[int]
    item_ids: list[list[int]]
    input_hash: str
    logical_lengths: list[int]
    packed_length: int
    attention_mode: str = "mis"

    def score_kwargs(self):
        """Exact Engine.score carrier; the runtime supplies the scalar head."""
        _require(self.attention_mode == "mis", "Causal requests must use the independent scalar endpoint")
        return {"query": list(self.query_ids), "items": [list(row) for row in self.item_ids],
                "label_token_ids": [0], "apply_softmax": True, "item_first": False}

    def independent_batches(self, *, packed_token_budget):
        """One ordinary causal sequence per candidate, without MIS delimiters."""
        _require(type(packed_token_budget) is int and packed_token_budget > 0, "Invalid token budget")
        batches = []
        for candidate, item, length in zip(self.candidate_ids, self.item_ids, self.logical_lengths, strict=True):
            _require(length == len(self.query_ids) + len(item) and length <= packed_token_budget,
                     "Independent candidate exceeds physical input limit")
            batches.append(NextBookRequest(self.query_id, (candidate,), list(self.query_ids), [list(item)],
                                          self.input_hash, [length], length, "causal"))
        return batches


class NextBookCompiler:
    """Verify an immutable export once, then compile canonical single queries.

    The optional tokenizer is an explicit injection for offline equivalence
    tests. Normal operation loads only the export's native reader tokenizer,
    with local files and no remote code. The export verifier owns checksum,
    catalog, vector, reader and checkpoint provenance validation.
    """

    def __init__(self, export_root, tokenizer=None):
        from .nextbook_export import verify_nextbook_export

        self.export_root = Path(export_root)
        manifest = copy.deepcopy(verify_nextbook_export(self.export_root, verify_vectors=True))
        metadata = _template_guard(manifest)
        signature = manifest.get("manifest_signature")
        _require(isinstance(signature, str) and len(signature) == 64
                 and all(character in "0123456789abcdef" for character in signature),
                 "Export manifest identity signature is missing")
        self.manifest_signature = signature
        self.vocab_size = _positive_int(manifest["native_reader_config"]["vocab_size"], "vocab_size")
        count = _positive_int(manifest["vector_count"], "vector_count")
        shape = manifest.get("vectors", {}).get("shape")
        _require(isinstance(shape, list) and len(shape) == 2 and shape[0] == count
                 and type(shape[0]) is int and manifest["vectors"].get("dtype") == "float16",
                 "Book vector count/storage differs from the verified export")
        width = _positive_int(shape[1], "vector width")
        _require(width == manifest["native_reader_config"]["hidden_size"] == metadata.get("b_width"),
                 "Book vector width differs from the native reader")
        limits = metadata.get("packing_limits", {})
        self.logical_limit = min(_positive_int(limits.get("logical_context"), "logical context"),
                                 _positive_int(manifest["native_reader_config"].get("max_position_embeddings"),
                                               "native position limit"))
        self.packed_limit = _positive_int(limits.get("reader_packed_tokens"), "reader packed tokens")
        _require(limits["logical_context"] == metadata.get("logical_context")
                 and self.packed_limit == metadata.get("reader_packed_tokens"),
                 "Packing limits differ from frozen GroupRanker metadata")
        _require(manifest["index"].get("file") == "index.jsonl", "Unsupported book index filename")
        index_path = self.export_root / "index.jsonl"
        book_ids, rows = {}, set()
        digest = hashlib.sha256()
        with index_path.open("rb") as stream:
            for line in stream:
                digest.update(line)
                entry = json.loads(line)
                _require(isinstance(entry, dict) and set(entry) == {"item_id", "row"}
                         and type(entry["item_id"]) is str and entry["item_id"].strip()
                         and type(entry["row"]) is int and 0 <= entry["row"] < count,
                         "Invalid canonical book index row")
                item, row = entry["item_id"], entry["row"]
                _require(item not in book_ids and row not in rows, "Duplicate book ID or vector row")
                book_ids[item] = self.vocab_size + row
                rows.add(row)
        _require(len(book_ids) == count and digest.hexdigest() == manifest["index"]["sha256"],
                 "Book index changed after export verification")
        self._book_ids = MappingProxyType(book_ids)
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.export_root / "reader", local_files_only=True,
                                                      trust_remote_code=False)
        self._tokenizer = tokenizer
        # Preserve GroupRanker._query_embeddings' separate text segments. A
        # concatenated string can have different boundary tokenization.
        self._instruction = self._native(metadata["student_template"]["instruction"])
        self._history_headers = tuple(self._native(f"History {index}:\nItem: ")
                                      for index in range(1, MAX_HISTORY + 1))
        self._ratings = {rating: self._native(f"\nUser rating: {rating}\n") for rating in range(1, 6)}
        self._candidate_header = self._native(CANDIDATE_HEADER)

    def _native(self, text):
        ids = tuple(self._tokenizer.encode(text, add_special_tokens=False))
        _require(ids and all(type(token) is int and 0 <= token < self.vocab_size for token in ids),
                 "Native prompt segment is empty or outside the reader vocabulary")
        return ids

    def _book_id(self, item):
        _require(type(item) is str and item in self._book_ids,
                 "Unknown book ID: " + repr(item))
        return self._book_ids[item]

    def compile(self, query):
        """Compile query_id/history/candidate_ids without truncating any input."""
        _require(isinstance(query, Mapping) and set(query) == {"query_id", "history", "candidate_ids"},
                 "Query must contain exactly query_id, history and candidate_ids")
        _require(type(query["query_id"]) is str and query["query_id"].strip(), "query_id must be a nonempty string")
        history, candidates = query["history"], query["candidate_ids"]
        _require(isinstance(history, list) and len(history) <= MAX_HISTORY, "History must contain at most 10 entries")
        _require(isinstance(candidates, list) and 1 <= len(candidates) <= MAX_CANDIDATES,
                 "Candidates must contain 1 to 100 unique book IDs")
        query_ids, canonical_history = list(self._instruction), []
        for index, event in enumerate(history):
            _require(isinstance(event, Mapping) and set(event) == {"item_id", "rating"},
                     "History entries must contain exactly item_id and rating")
            rating = event["rating"]
            _require(type(rating) is int and 1 <= rating <= 5, "History rating must be an integer from 1 to 5")
            token = self._book_id(event["item_id"])
            query_ids.extend(self._history_headers[index])
            query_ids.append(token)
            query_ids.extend(self._ratings[rating])
            canonical_history.append({"item_id": event["item_id"], "rating": rating})
        query_ids.extend(self._candidate_header)
        item_ids = [[self._book_id(item)] for item in candidates]
        _require(len(set(candidates)) == len(candidates), "Candidates must contain unique book IDs")
        logical_lengths = [len(query_ids) + 1] * len(candidates)
        packed_length = len(query_ids) + len(candidates) + len(candidates) + 1
        _require(logical_lengths[0] <= self.logical_limit, "Query path exceeds logical position limit; no truncation")
        _require(packed_length <= self.packed_limit, "Query tree exceeds reader packed token limit; no truncation")
        canonical = {"query_id": query["query_id"], "history": canonical_history, "candidate_ids": list(candidates)}
        input_hash = _hash({"manifest_signature": self.manifest_signature, "query": canonical,
                            "query_ids": query_ids, "item_ids": item_ids})
        return NextBookRequest(query["query_id"], tuple(candidates), query_ids, item_ids,
                               input_hash, logical_lengths, packed_length)
