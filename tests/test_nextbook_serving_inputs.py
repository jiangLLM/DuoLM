"""ID compiler protocol tests, with a separate frozen GroupRanker oracle."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from duolm.nextbook_serving_inputs import NextBookCompiler


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class SegmentTokenizer:
    """Boundary-sensitive tokenizer: concatenated text cannot mimic segments."""

    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("Native segments must not gain special tokens")
        self.calls.append(text)
        return [2 + sum(map(ord, text)) % 200, 210 + len(text) % 100]


class NextBookCompilerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        entries = [{"item_id": f"book{i:03d}", "row": i} for i in range(101)]
        index = "".join(json.dumps(row) + "\n" for row in entries).encode()
        (self.root / "index.jsonl").write_bytes(index)
        template = {"version": "books-next-item-student-interface-v2",
                    "instruction": "Predict the user's next book interaction from the past history.\n",
                    "variant": "V3", "m": 1, "feedback_channel": "b_explicit_rating",
                    "a_readout": "last_token", "candidate_header": "Candidate:\nItem: ",
                    "shared_candidate_header": True, "score_readout": "candidate", "score_suffix": None,
                    "a_eos": 1, "a_memory_positions": 0, "b_score_positions": 0,
                    "flex_attention_kernel": "forced_prefill", "resampler_layers": None, "resampler_heads": None}
        template["sha256"] = digest(template)
        self.manifest = {"variant": "V3", "m": 1, "vector_count": 101,
                         "manifest_signature": "a" * 64,
                         "native_reader_config": {"vocab_size": 512, "hidden_size": 4, "max_position_embeddings": 1024},
                         "vectors": {"file": "vectors.npy", "dtype": "float16", "shape": [101, 4]},
                         "index": {"file": "index.jsonl", "sha256": hashlib.sha256(index).hexdigest()},
                         "model_metadata": {"variant": "V3", "m": 1, "b_width": 4, "a_readout": "last_token",
                                            "shared_candidate_header": True, "score_readout": "candidate",
                                            "logical_context": 2048, "reader_packed_tokens": 4096,
                                            "packing_limits": {"logical_context": 2048, "reader_packed_tokens": 4096},
                                            "student_template": template}}
        # Delegate byte/vector integrity to the export verifier, independently
        # of prompt compilation. The compiler must invoke it with vectors on.
        self.verifier = Mock(side_effect=lambda root, verify_vectors=True: copy.deepcopy(self.manifest))
        module = ModuleType("duolm.nextbook_export")
        module.verify_nextbook_export = self.verifier
        self.modules = patch.dict("sys.modules", {module.__name__: module})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.tokenizer = SegmentTokenizer()

    def compiler(self):
        return NextBookCompiler(self.root, tokenizer=self.tokenizer)

    def query(self, count=3, history=None):
        return {"query_id": "q0", "history": history if history is not None else [
                {"item_id": "book099", "rating": 1}, {"item_id": "book100", "rating": 5}],
                "candidate_ids": [f"book{i:03d}" for i in range(count)]}

    def limits(self, logical=2048, packed=4096, native=1024):
        metadata = self.manifest["model_metadata"]
        metadata.update(logical_context=logical, reader_packed_tokens=packed)
        metadata["packing_limits"].update(logical_context=logical, reader_packed_tokens=packed)
        self.manifest["native_reader_config"]["max_position_embeddings"] = native

    def test_startup_verifies_vectors_and_compiles_10_50_100_candidates_without_tokenizing_books(self):
        compiler = self.compiler()
        self.verifier.assert_called_once_with(self.root, verify_vectors=True)
        calls = list(self.tokenizer.calls)
        self.assertEqual(len(calls), 17)
        self.assertFalse(any("book" in call for call in calls if not call.startswith("Predict")))
        for count in (10, 50, 100):
            request = compiler.compile(self.query(count))
            self.assertEqual(request.item_ids, [[512 + i] for i in range(count)])
            self.assertEqual(request.logical_lengths, [len(request.query_ids) + 1] * count)
            self.assertEqual(request.packed_length, len(request.query_ids) + count + count + 1)
            self.assertEqual(self.tokenizer.calls, calls)

    def test_separate_segments_match_the_unchanged_frozen_query_embeddings(self):
        import torch
        from duolm.next_item_model import GroupRanker

        compiler = self.compiler()
        query = self.query()
        request = compiler.compile(query)
        tokenizer = SegmentTokenizer()
        oracle = SimpleNamespace(variant="V3", shared_candidate_header=True, score_readout="candidate",
                                 b=SimpleNamespace(get_input_embeddings=lambda: SimpleNamespace(
                                     weight=torch.zeros(1, 1))))
        oracle._native = lambda text: torch.tensor(tokenizer.encode(text, add_special_tokens=False),
                                                   dtype=torch.float32)[:, None]
        histories = [SimpleNamespace(item_id=row["item_id"], rating=row["rating"]) for row in query["history"]]
        frozen_query = SimpleNamespace(history=histories, candidate_ids=query["candidate_ids"])
        ids = [611, 612, 512, 513, 514]
        states = [torch.tensor([[token]], dtype=torch.float32) for token in ids]
        prefix, branches = GroupRanker._query_embeddings(oracle, frozen_query, {}, states,
                                                         ([0, 1], [2, 3, 4]))
        self.assertEqual(request.query_ids, prefix[:, 0].long().tolist())
        self.assertEqual(request.item_ids, [branch[:, 0].long().tolist() for branch in branches])
        expected_segments = ["Predict the user's next book interaction from the past history.\n",
                             "History 1:\nItem: ", "\nUser rating: 1\n",
                             "History 2:\nItem: ", "\nUser rating: 5\n", "Candidate:\nItem: "]
        self.assertEqual(tokenizer.calls, expected_segments)
        self.assertNotEqual(request.query_ids, tokenizer.encode("".join(expected_segments)))

    def test_carrier_is_exact_and_returned_payload_cannot_mutate_a_compiled_query(self):
        request = self.compiler().compile(self.query(1, history=[]))
        kwargs = request.score_kwargs()
        self.assertEqual(set(kwargs), {"query", "items", "label_token_ids", "apply_softmax", "item_first"})
        self.assertEqual(kwargs["label_token_ids"], [0])
        self.assertTrue(kwargs["apply_softmax"])
        self.assertFalse(kwargs["item_first"])
        self.assertEqual(kwargs["items"], [[512]])
        kwargs["query"].clear()
        kwargs["items"][0].clear()
        self.assertTrue(request.query_ids)
        self.assertEqual(request.item_ids, [[512]])

    def test_independent_paths_are_exact_logical_sequences_without_delimiters(self):
        request = self.compiler().compile(self.query(3))
        length = request.logical_lengths[0]
        paths = request.independent_batches(packed_token_budget=length)
        self.assertEqual([path.candidate_ids[0] for path in paths], list(request.candidate_ids))
        self.assertEqual([path.item_ids[0] for path in paths], request.item_ids)
        for path in paths:
            self.assertEqual(path.query_ids, request.query_ids)
            self.assertEqual(path.packed_length, len(path.query_ids + path.item_ids[0]))
            self.assertEqual(path.logical_lengths, [length])
            self.assertEqual(path.input_hash, request.input_hash)
            self.assertEqual(path.attention_mode, "causal")
            with self.assertRaisesRegex(ValueError, "independent scalar"):
                path.score_kwargs()
        with self.assertRaisesRegex(ValueError, "physical input"):
            request.independent_batches(packed_token_budget=length - 1)
        paths[0].query_ids.clear()
        self.assertTrue(request.query_ids)
        self.assertTrue(paths[1].query_ids)

    def test_ratings_history_and_candidate_order_affect_only_the_expected_inputs_and_hash(self):
        compiler = self.compiler()
        query = self.query()
        original = compiler.compile(query)
        changed = copy.deepcopy(query)
        changed["history"][0]["rating"] = 2
        rating = compiler.compile(changed)
        self.assertNotEqual(rating.query_ids, original.query_ids)
        self.assertNotEqual(rating.input_hash, original.input_hash)
        self.assertEqual(rating.item_ids, original.item_ids)
        changed = copy.deepcopy(query)
        changed["history"].reverse()
        reordered_history = compiler.compile(changed)
        self.assertNotEqual(reordered_history.query_ids, original.query_ids)
        self.assertNotEqual(reordered_history.input_hash, original.input_hash)
        changed = copy.deepcopy(query)
        changed["candidate_ids"].reverse()
        reordered = compiler.compile(changed)
        self.assertEqual(reordered.query_ids, original.query_ids)
        self.assertEqual(reordered.item_ids, list(reversed(original.item_ids)))
        self.assertNotEqual(reordered.input_hash, original.input_hash)
        self.assertEqual(reordered.candidate_ids, tuple(changed["candidate_ids"]))
        self.assertEqual(compiler.compile(query).input_hash, original.input_hash)
        self.assertIsNot(compiler.compile(query), original)

    def test_hash_binds_query_id_and_export_signature(self):
        query = self.query()
        original = self.compiler().compile(query)
        query["query_id"] = "another query"
        self.assertNotEqual(original.input_hash, self.compiler().compile(query).input_hash)
        self.manifest["manifest_signature"] = "b" * 64
        query["query_id"] = "q0"
        other_export = self.compiler().compile(query)
        self.assertEqual(original.query_ids, other_export.query_ids)
        self.assertNotEqual(original.input_hash, other_export.input_hash)

    def test_unknown_ids_duplicate_candidates_and_invalid_ratings_fail(self):
        compiler = self.compiler()
        for rating in (True, False, 0, 6, 1.0, "1", None):
            query = self.query()
            query["history"][0]["rating"] = rating
            with self.subTest(rating=rating), self.assertRaisesRegex(ValueError, "rating"):
                compiler.compile(query)
        for unknown in ("missing", "", 1, None):
            query = self.query()
            query["candidate_ids"][0] = unknown
            with self.subTest(item=unknown), self.assertRaisesRegex(ValueError, "Unknown book ID"):
                compiler.compile(query)
        query = self.query()
        query["history"][0]["item_id"] = "missing"
        with self.assertRaisesRegex(ValueError, "Unknown book ID"):
            compiler.compile(query)
        query = self.query()
        query["candidate_ids"][1] = query["candidate_ids"][0]
        with self.assertRaisesRegex(ValueError, "unique"):
            compiler.compile(query)

    def test_history_and_candidate_bounds_reject_without_truncation(self):
        compiler = self.compiler()
        history = [{"item_id": "book099", "rating": i % 5 + 1} for i in range(10)]
        self.assertEqual(len(compiler.compile(self.query(100, history)).candidate_ids), 100)
        with self.assertRaisesRegex(ValueError, "10 entries"):
            compiler.compile(self.query(1, history + [history[0]]))
        for count in (0, 101):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "1 to 100"):
                compiler.compile(self.query(count))

    def test_exact_logical_position_and_physical_packed_budget_boundaries(self):
        query = self.query(3)
        reference = self.compiler().compile(query)
        path, packed = reference.logical_lengths[0], reference.packed_length
        self.limits(logical=path, packed=packed, native=path)
        self.assertEqual(self.compiler().compile(query).packed_length, packed)
        self.limits(logical=path - 1, packed=packed)
        with self.assertRaisesRegex(ValueError, "logical position"):
            self.compiler().compile(query)
        self.limits(logical=path, packed=packed, native=path - 1)
        with self.assertRaisesRegex(ValueError, "logical position"):
            self.compiler().compile(query)
        self.limits(logical=path, packed=packed - 1, native=path)
        with self.assertRaisesRegex(ValueError, "packed token"):
            self.compiler().compile(query)

    def test_manifest_interface_and_template_signature_guards(self):
        original = copy.deepcopy(self.manifest)
        for key, value in (("a_readout", "perceiver"), ("m", 2), ("variant", "V1"),
                           ("shared_candidate_header", False), ("score_readout", "legacy"),
                           ("score_suffix", "\nNext-book score:"), ("instruction", "new instruction")):
            self.manifest = copy.deepcopy(original)
            template = self.manifest["model_metadata"]["student_template"]
            template[key] = value
            template["sha256"] = digest({k: v for k, v in template.items() if k != "sha256"})
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, "Only the frozen"):
                self.compiler()
        self.manifest = copy.deepcopy(original)
        self.manifest["model_metadata"]["student_template"]["sha256"] = "bad"
        with self.assertRaisesRegex(ValueError, "template identity"):
            self.compiler()
        self.manifest = copy.deepcopy(original)
        self.manifest["variant"] = "V1"
        with self.assertRaisesRegex(ValueError, "variant/m"):
            self.compiler()

    def test_index_hash_and_duplicate_row_guard(self):
        path = self.root / "index.jsonl"
        text = path.read_text()
        path.write_text(text.replace('"book000"', '"changed"'))
        with self.assertRaisesRegex(ValueError, "index changed"):
            self.compiler()
        path.write_text(text.replace('"row": 1}', '"row": 0}'))
        with self.assertRaisesRegex(ValueError, "Duplicate book ID or vector row"):
            self.compiler()

    def test_export_verification_failure_propagates_before_tokenizer_use(self):
        self.verifier.side_effect = ValueError("Vector binding differs")
        with self.assertRaisesRegex(ValueError, "Vector binding"):
            self.compiler()
        self.assertEqual(self.tokenizer.calls, [])


if __name__ == "__main__":
    unittest.main()
