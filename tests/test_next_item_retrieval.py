"""Offline producer/restore tests; injected models never exposed by the CLI."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from duolm.build_next_item_assets import (
    ENCODER_REPO, ENCODER_REVISION, verify_public_snapshot,
)
from duolm.next_item_data import user_split
from duolm.next_item_retrieval import (
    export_index, last_valid_unit_vectors, load_index, produce_retrieval,
    read_retrieval_inputs, retrieve_topk,
)
from duolm.prepare_next_item import file_hash


class Tokenizer:
    def encode(self, text, **kwargs):
        return [1 + ord(character) % 31 for character in text]

    def __call__(self, texts, **kwargs):
        if kwargs != {"padding": True, "truncation": False, "add_special_tokens": False, "return_tensors": "pt"}:
            raise AssertionError("batch tokenization must not silently truncate")
        values = [self.encode(text) for text in texts]
        width = max(map(len, values))
        return {"input_ids": torch.tensor([row + [0] * (width - len(row)) for row in values]),
                "attention_mask": torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in values])}


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, input_ids, attention_mask):
        self.calls += 1
        value = input_ids.float()
        return SimpleNamespace(last_hidden_state=torch.stack((torch.ones_like(value), value,
                                                              value.square() + 1, value.remainder(7) + 1), dim=-1))


def user(split):
    for index in range(10_000):
        value = f"retrieval-user-{index}"
        if user_split(value) == split:
            return value
    raise AssertionError


def input_row(items, ordinal=0, split="train", rating=1):
    history = [{"item_id": "book-000", "rating": rating, "timestamp": 1}]
    seen = ["book-000"]
    content = {"history": [("book-000", rating, 1, items["book-000"])], "seen_item_ids": seen}
    return {"query_id": f"q-{ordinal}", "user_id": user(split), "split": split,
            "history": history, "seen_item_ids": seen,
            "history_hash": hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


class PoolingAndRankingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_last_valid_pooling_supports_both_padding_sides(self):
        hidden = torch.tensor([[[99., 99.], [3., 4.], [99., 99.]],
                               [[99., 99.], [99., 99.], [5., 12.]]])
        mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
        result = last_valid_unit_vectors(hidden, mask)
        torch.testing.assert_close(result, torch.tensor([[.6, .8], [5/13, 12/13]]))
        with self.assertRaisesRegex(ValueError, "empty"):
            last_valid_unit_vectors(hidden[:1], torch.zeros((1, 3), dtype=torch.long))

    def test_chunked_scores_exact_reference_with_ties_and_all_past_exclusion(self):
        ids = [f"book-{i:03d}" for i in range(12)]
        vectors = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [0., 1.], [1., 0.], [0., 1.],
                                [-1., 0.], [1., 0.], [0., -1.], [1., 0.], [0., 1.], [-1., 0.]])
        history, seen = [[ids[0], ids[3]], [ids[1]]], [[ids[0], ids[3], ids[2]], [ids[1], ids[4]]]
        expected = []
        for past, excluded in zip(history, seen):
            query = torch.nn.functional.normalize(vectors[[ids.index(item) for item in past]].mean(dim=0), dim=0)
            scores = (vectors @ query).tolist()
            expected.append(sorted((item for item in ids if item not in excluded),
                                   key=lambda item: (-scores[ids.index(item)], item))[:5])
        for chunk_size in (1, 3, 5, 12):
            self.assertEqual(retrieve_topk(ids, vectors, history, seen, k=5, catalog_chunk_size=chunk_size), expected)

    def test_reusable_catalog_mapping_avoids_full_iteration_per_batch(self):
        class NoIterationList(list):
            def __iter__(self):
                raise AssertionError("catalog index must be reused")
        ids = [f"book-{i:03d}" for i in range(12)]
        mapping = {item: index for index, item in enumerate(ids)}
        result = retrieve_topk(NoIterationList(ids), torch.ones((12, 2)), [[ids[0]]], [[ids[0]]],
                               k=3, catalog_chunk_size=4, item_index=mapping)
        self.assertEqual(result, [ids[1:4]])

    def test_history_reduction_has_no_atomic_scatter_and_counts_repeated_books(self):
        ids = [f"book-{i:03d}" for i in range(6)]
        vectors = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.], [.6, .8], [-1., 0.]])
        history = [[ids[0], ids[1], ids[0]], [ids[1]]]
        seen = [[ids[0], ids[1]], [ids[1]]]
        expected = []
        for past, excluded in zip(history, seen):
            query = torch.nn.functional.normalize(vectors[[ids.index(item) for item in past]].sum(dim=0), dim=0)
            scores = (vectors @ query).tolist()
            expected.append(sorted((item for item in ids if item not in excluded),
                                   key=lambda item: (-scores[ids.index(item)], item))[:2])
        with patch.object(torch.Tensor, "index_add_", side_effect=AssertionError("atomic scatter is not replay-safe")):
            self.assertEqual(retrieve_topk(ids, vectors, history, seen, k=2, catalog_chunk_size=3), expected)

    def test_candidate_count_and_zero_mean_fail_explicitly(self):
        ids, vectors = ["a", "b"], torch.tensor([[1., 0.], [-1., 0.]])
        with self.assertRaisesRegex(ValueError, "fixed number"):
            retrieve_topk(ids, vectors, [["a"]], [["a"]], k=2)
        with self.assertRaisesRegex(ValueError, "zero norm"):
            retrieve_topk(ids + ["c", "d"], torch.cat((vectors, vectors)), [["a", "b"]], [["a", "b"]], k=1)


class IndexAndProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.items = {f"book-{i:03d}": f"Title {i}" for i in range(240)}
        self.catalog = self.root / "catalog.json"
        self.catalog.write_text(json.dumps(self.items))
        self.encoder = Encoder()
        self.index = self.root / "index"

    def export(self, **kwargs):
        return export_index(self.catalog, self.index, tokenizer=Tokenizer(), model=self.encoder,
                             device="cpu", batch_size=17, shard_items=50, compute_dtype="float32", **kwargs)

    def test_resumable_fp16_shards_sorted_identity_and_no_reencoding(self):
        partial = self.export(max_new_shards=1)
        self.assertFalse(partial["complete"])
        first = partial["shards"][0]
        original = (self.index / first["file"]).read_bytes()
        complete = self.export()
        self.assertTrue(complete["complete"])
        self.assertEqual((self.index / first["file"]).read_bytes(), original)
        calls = self.encoder.calls
        self.assertEqual(self.export(), complete)
        self.assertEqual(self.encoder.calls, calls)
        ids, vectors, report = load_index(self.index, self.catalog, device="cpu")
        self.assertEqual(ids, sorted(self.items))
        self.assertEqual(report, complete)
        torch.testing.assert_close(torch.linalg.vector_norm(vectors, dim=1), torch.ones(240), atol=1e-6, rtol=1e-6)
        self.assertEqual(np.load(self.index / first["file"], allow_pickle=False).dtype, np.float16)
        with self.assertRaisesRegex(ValueError, "differs"):
            export_index(self.catalog, self.index, tokenizer=Tokenizer(), model=self.encoder, device="cpu",
                          batch_size=18, shard_items=50, compute_dtype="float32")

    def test_corrupt_completed_shard_refused_and_oversize_never_truncated(self):
        self.export()
        manifest = json.loads((self.index / "index-manifest.json").read_text())
        first = self.index / manifest["shards"][0]["file"]
        with first.open("ab") as stream:
            stream.write(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.export()
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_index(self.index, self.catalog, device="cpu")
        other = self.root / "too-long.json"
        other.write_text(json.dumps({"a": "x" * 513}))
        with self.assertRaisesRegex(ValueError, "no implicit truncation"):
            export_index(other, self.root / "long-index", tokenizer=Tokenizer(), model=self.encoder,
                          device="cpu", compute_dtype="float32")

    def test_source_schema_and_content_hash_reject_target_test_or_stale_history(self):
        inputs = self.root / "inputs.jsonl"
        row = input_row(self.items)
        inputs.write_text(json.dumps(row) + "\n")
        self.assertEqual(list(read_retrieval_inputs(inputs, self.items)), [row])
        for changed in ({**row, "target_item_id": "book-002"}, {**row, "split": "test", "user_id": user("test")},
                        {**row, "history_hash": "wrong"}):
            inputs.write_text(json.dumps(changed) + "\n")
            with self.assertRaises(ValueError):
                list(read_retrieval_inputs(inputs, self.items))

    def test_real_producer_resume_contract_and_rating_independence(self):
        self.export()
        rows = [input_row(self.items, 0, "train", 1), input_row(self.items, 1, "train", 5),
                input_row(self.items, 2, "dev", 3)]
        inputs = self.root / "inputs.jsonl"
        inputs.write_text("\n".join(map(json.dumps, rows)) + "\n")
        output = self.root / "retrieval"
        partial = produce_retrieval(self.catalog, inputs, self.index, output, device="cpu", batch_size=2,
                                     catalog_chunk_size=70, max_new_batches=1)
        self.assertFalse(partial["complete"])
        first_bytes = (output / partial["shards"][0]["file"]).read_bytes()
        report = produce_retrieval(self.catalog, inputs, self.index, output, device="cpu", batch_size=2,
                                    catalog_chunk_size=70)
        result = [json.loads(line) for line in (output / "retrieval.jsonl").read_text().splitlines()]
        self.assertEqual([len(row["retrieved_ids"]) for row in result], [200, 200, 100])
        self.assertEqual(result[0]["retrieved_ids"], result[1]["retrieved_ids"])
        self.assertEqual((output / partial["shards"][0]["file"]).read_bytes(), first_bytes)
        self.assertEqual(report["inputs_sha256"], file_hash(inputs))
        self.assertEqual(report["catalog_sha256"], file_hash(self.catalog))
        self.assertEqual(report["results_sha256"], file_hash(output / "retrieval.jsonl"))
        self.assertTrue(report["history_only"])
        self.assertFalse(report["test_labels_accessed"])
        self.assertEqual(produce_retrieval(self.catalog, inputs, self.index, output, device="cpu", batch_size=2,
                                          catalog_chunk_size=70), report)
        inputs.write_text(inputs.read_text() + json.dumps(input_row(self.items, 3)) + "\n")
        with self.assertRaisesRegex(ValueError, "resume differs"):
            produce_retrieval(self.catalog, inputs, self.index, output, device="cpu", batch_size=2,
                               catalog_chunk_size=70)

    def test_snapshot_manifest_requires_correct_public_revision_files_and_safe_paths(self):
        snapshot = self.root / "snapshot"
        snapshot.mkdir()
        (snapshot / "config.json").write_text("{}")
        (snapshot / "weights.safetensors").write_bytes(b"fixture-only-no-model-loaded")
        report = {"repo_id": ENCODER_REPO, "revision": ENCODER_REVISION,
                  "files": {name: {"bytes": (snapshot / name).stat().st_size, "sha256": file_hash(snapshot / name)}
                            for name in ("config.json", "weights.safetensors")}}
        manifest = snapshot / "snapshot-manifest.json"
        manifest.write_text(json.dumps(report))
        self.assertEqual(verify_public_snapshot(snapshot, repo=ENCODER_REPO, revision=ENCODER_REVISION), file_hash(manifest))
        report["revision"] = "0" * 40
        manifest.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "repo/revision"):
            verify_public_snapshot(snapshot, repo=ENCODER_REPO, revision=ENCODER_REVISION)
        report["revision"] = ENCODER_REVISION
        manifest.write_text(json.dumps(report))
        (snapshot / "config.json").write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, "checksum/size/path"):
            verify_public_snapshot(snapshot, repo=ENCODER_REPO, revision=ENCODER_REVISION)


if __name__ == "__main__":
    unittest.main()
