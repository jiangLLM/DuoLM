"""Tiny preparation smoke with real builders and a synthetic CPU encoder.

No downloads, pretrained model weights, real user histories, or paper scores.
"""
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from duolm.build_next_item_assets import build_catalog, build_cohorts
from duolm.candidate_data_validation import validate_candidate_preparation
from duolm.inventory import build_inventory
from duolm.next_item_data import read_queries, user_split
from duolm.next_item_retrieval import export_index, produce_retrieval
from duolm.prepare_candidate_ranking import prepare_candidate_ranking
from duolm.prepare_next_item import assemble_groups, build_context_stage, file_hash
from duolm.build_games_assets import materialize
from duolm.games_data import load_dataset
from tests.test_games_assets import build_fixture


class TinyTokenizer:
    def encode(self, text, **kwargs):
        return [ord(character) % 31 + 1 for character in text]

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            return {"offset_mapping": [(i, i + 1) for i in range(len(texts))]}
        rows = [self.encode(text) for text in texts]
        width = max(map(len, rows))
        return {"input_ids": torch.tensor([row + [0] * (width - len(row)) for row in rows]),
                "attention_mask": torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in rows])}


class TinyEncoder(torch.nn.Module):
    def forward(self, input_ids, attention_mask):
        value = input_ids.float()
        return SimpleNamespace(last_hidden_state=torch.stack(
            (torch.ones_like(value), value, value.remainder(7) + 1), dim=-1))


def synthetic_users(split, count):
    values = []
    for index in range(10000):
        user = f"synthetic-public-smoke-{index}"
        if user_split(user) == split:
            values.append(user)
        if len(values) == count:
            return values
    raise AssertionError("synthetic split search failed")


class PublicDataPipelineSmoke(unittest.TestCase):
    def test_games_stage_through_real_retrieval_and_materialization(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = build_fixture(root, unavailable_dev=True)
            contexts = fixture["contexts"]
            export_index(contexts / "catalog.json", root / "actual-index", tokenizer=TinyTokenizer(),
                         model=TinyEncoder(), device="cpu", compute_dtype="float32", batch_size=16,
                         shard_items=60, model_identity={"fixture": "tiny-synthetic-encoder"}, domain="game")
            produce_retrieval(contexts / "catalog.json", contexts / "retrieval-inputs.jsonl",
                              root / "actual-index", root / "actual-retrieval", device="cpu", domain="game",
                              first_seen=contexts / "first-seen.json", batch_size=2, catalog_chunk_size=47)
            report = materialize(contexts, root / "actual-retrieval/retrieval.jsonl",
                                 root / "actual-retrieval/retrieval-manifest.json", root / "actual-prepared")
            train, dev, _, cohorts, _ = load_dataset(root / "actual-prepared")
            self.assertTrue(train)
            self.assertEqual(dev, ())
            self.assertEqual(len(cohorts["dev"]), 1)
            self.assertEqual(len(cohorts["unavailable_dev_users"]), 1)
            self.assertTrue(report["test_labels_sealed"])
            self.assertTrue(all("future" not in query.candidate_ids for query in train))

    def test_books_inventory_to_frozen_mixed8_and_random100(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_user = synthetic_users("train", 1)[0]
            dev_users = synthetic_users("dev", 2)
            test_user = synthetic_users("test", 1)[0]
            item_ids = [f"book-{i:03d}" for i in range(240)]
            ratings, metadata = root / "ratings.csv", root / "metadata.jsonl"
            with ratings.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["user_id", "parent_asin", "rating", "timestamp"])
                writer.writerows([train_user, item, i % 5 + 1, 1000 + i]
                                 for i, item in enumerate(item_ids[:14]))
                writer.writerows([[dev_users[0], item_ids[0], 2, 100],
                                  [dev_users[0], item_ids[1], 5, 200],
                                  [dev_users[1], item_ids[0], 3, 100],
                                  [dev_users[1], "missing-metadata", 4, 200]])
                # Test identities establish public membership. Their labels
                # cannot be parsed by the Books preparation path.
                writer.writerows([test_user, item, "unreadable-rating", "unreadable-time"] for item in item_ids)
            metadata.write_text("".join(json.dumps({"parent_asin": item, "title": item,
                                                      "author": {"name": "Synthetic author"},
                                                      "description": "Synthetic item text"}) + "\n"
                                        for item in item_ids))
            build_inventory(ratings, root / "inventory")
            tokenizer = TinyTokenizer()
            build_catalog(metadata, root / "inventory/catalog-item-ids.json", root / "assets",
                          inventory_manifest=root / "inventory/manifest.json",
                          b_tokenizer=tokenizer, a_tokenizer=tokenizer)
            build_cohorts(ratings, root / "assets/catalog.json", root / "assets",
                          train_users=1, dev_users=2, test_users=1, selector_users=2)
            cohorts = json.loads((root / "assets/cohorts.json").read_text())
            contexts = build_context_stage(ratings, root / "assets/catalog.json", cohorts, root / "contexts",
                                           tokenizer=tokenizer, tokenizer_identity={"fixture": "character-tokenizer"})
            self.assertEqual(contexts["unavailable_dev_users"], [dev_users[1]])
            inputs = [json.loads(line) for line in (root / "contexts/retrieval-inputs.jsonl").read_text().splitlines()]
            self.assertTrue(all("target_item_id" not in row for row in inputs))
            export_index(root / "contexts/catalog.json", root / "index", tokenizer=tokenizer,
                         model=TinyEncoder(), device="cpu", compute_dtype="float32", batch_size=24,
                         shard_items=80, model_identity={"fixture": "tiny-synthetic-encoder"})
            produce_retrieval(root / "contexts/catalog.json", root / "contexts/retrieval-inputs.jsonl",
                              root / "index", root / "retrieval", device="cpu", batch_size=2,
                              catalog_chunk_size=47)
            assemble_groups(root / "contexts", root / "retrieval/retrieval.jsonl",
                            root / "retrieval/retrieval-manifest.json", root / "mixed8")
            prepare_candidate_ranking(root / "mixed8", root / "random100", train_candidates=100)
            report = prepare_candidate_ranking(root / "mixed8", root / "candidates", train_policy="mixed-history",
                                               eval_source_dir=root / "random100")
            train = read_queries(root / "candidates/train.jsonl", split="train")
            dev = read_queries(root / "candidates/dev.jsonl", split="dev")
            config = {"schema_version": "books-candidate-ranking-v1", "seed": 17,
                      "data": {"train_candidates": 8, "eval_candidates": 100, "history_max": 10,
                               "query_cap": 5, "candidate_policy": "random-unobserved",
                               "train_policy": "mixed-history", "eval_policy": "random-unobserved",
                               "candidate_seed": 17}}
            files = {name: file_hash(root / "candidates" / name) for name in report["files"]}
            validate_candidate_preparation(report, config, files, train, dev)
            self.assertEqual(len(train), 5)
            self.assertEqual(len(dev), 1)
            self.assertEqual(len(cohorts["dev"]), 2)
            self.assertEqual(report["unavailable_dev_users"], [dev_users[1]])
            self.assertEqual((root / "mixed8/train.jsonl").read_bytes(),
                             (root / "candidates/train.jsonl").read_bytes())
            self.assertEqual((root / "random100/dev.jsonl").read_bytes(),
                             (root / "candidates/dev.jsonl").read_bytes())
            self.assertEqual(report["test_labels_accessed"], False)


if __name__ == "__main__":
    unittest.main()
