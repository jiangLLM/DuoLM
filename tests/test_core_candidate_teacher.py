from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from duolm import candidate_teacher as teacher
from duolm import train_next_item as trainer
from duolm.next_item_data import write_queries
from duolm.next_item_model import group_ranking_loss
from tests.test_core_train_next_item import candidate_dataset, config


class PoolRanker(nn.Module):
    """Deterministic scalar scorer with checkpoint-compatible backbone/head."""
    calls = []

    def __init__(self):
        super().__init__()
        self.backbone = nn.Parameter(torch.tensor(.3))
        self.head = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.head.weight.fill_(.4)

    def forward(self, queries, items, **kwargs):
        self.calls.extend(q.query_id for q in queries)
        scores = [torch.tensor([1.0 if item == "positive" else 0.0 for item in q.candidate_ids])
                  * self.backbone * self.head.weight[0, 0] for q in queries]
        return {"scores": scores, "statistics": {"reader_tokens": len(queries) * 7}}

    def loss(self, queries, items, **kwargs):
        result = self(queries, items)
        result.update(group_ranking_loss(result["scores"], [q.positive_index for q in queries], **kwargs))
        return result


class CandidateTeacherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "candidate100"
        self.cfg = candidate_dataset(self.data)
        self.source_cfg = config()
        self.source_config_path = self.root / "source-config.json"
        self.checkpoint = self._checkpoint()
        PoolRanker.calls = []

    def _checkpoint(self, *, source=None, step=1):
        source = self.source_cfg if source is None else source
        self.source_config_path.write_text(json.dumps(source))
        run = self.root / f"teacher-{step}"
        run.mkdir(exist_ok=True)
        model = PoolRanker()
        optimizer = trainer.build_optimizer(model, source)
        identity = {"config_sha256": trainer._digest(source),
                    "data_sha256": {"train": "1" * 64},
                    "source_sha256": {"next_item_model.py": "2" * 64}}
        state = {"step": step, "cursor": step * 3, "best_step": step, "best_ndcg": .5,
                 "best_checkpoint": f"checkpoints/step-{step:08d}"}
        return trainer.save_checkpoint(run, model, optimizer, state, identity)

    def _evaluate(self, output=None, **kwargs):
        return teacher.evaluate_checkpoint(self.checkpoint, self.source_config_path, self.cfg, self.data,
                    output or self.root / "evaluation", device="cpu", formal=False,
                    model_factory=lambda cfg, device: PoolRanker(), **kwargs)

    def _cache(self, output=None, **kwargs):
        return teacher.cache_checkpoint(self.checkpoint, self.source_config_path, self.cfg, self.data,
                    output or self.root / "cache", device="cpu", formal=False,
                    model_factory=lambda cfg, device: PoolRanker(), shard_queries=2, **kwargs)

    def _rewrite_payload(self, callback):
        path = self.checkpoint / "training.pt"
        payload = torch.load(path, weights_only=True)
        callback(payload)
        torch.save(payload, path)
        manifest_path = self.checkpoint / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["training.pt"] = teacher._record(path)
        manifest_path.write_text(json.dumps(manifest))

    def test_old_next_item_teacher_weights_evaluate_new_100_pool_without_resume(self):
        with patch.object(trainer, "load_checkpoint", side_effect=AssertionError("no optimizer resume")):
            report = self._evaluate()
        self.assertTrue(report["completed"])
        self.assertTrue(report["weights_only"])
        self.assertEqual(report["teacher"]["source_config_sha256"], trainer._digest(self.source_cfg))
        self.assertNotEqual(report["teacher"]["source_config_sha256"], trainer._digest(self.cfg))
        self.assertEqual(report["evaluation_scope"], "target-guaranteed-candidate-ranking")
        for key in ("hr@1", "hr@10", "ndcg@10", "candidate_target_coverage"):
            self.assertEqual(report["metrics"][key], 1)
        self.assertNotIn("retrieval_recall", report["metrics"])
        rows = [json.loads(line) for line in (self.root / "evaluation" / report["predictions_file"]).read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(len(row["scores"]) == len(row["candidate_ids"]) == 100 for row in rows))
        self.assertEqual(self._evaluate(), report)
        self.assertEqual(len(PoolRanker.calls), 2)
        independent = self._evaluate(self.root / "report-only", subset="disjoint-report")
        self.assertEqual(independent["metrics"]["users"], 1)

    def test_explicit_bf16_tree_cache_uses_strict_math_and_records_profile(self):
        self.cfg["evaluation"].update(compute_dtype="bfloat16", scoring_backend="tree", input_assembly="legacy")
        with patch.object(trainer, "inference_context", wraps=trainer.inference_context) as contexts:
            report = self._cache()
        inference = report["identity"]["inference"]
        self.assertEqual(inference["compute_dtype"], "bfloat16")
        self.assertEqual(inference["scoring_backend"], "tree")
        self.assertIs(inference["tf32"], False)
        calls = [call.args[1] for call in contexts.call_args_list]
        self.assertEqual(calls, ["float32", "bfloat16"] * (len(calls) // 2))
        self.assertGreater(len(calls), 0)

    def test_new_candidate_teacher_checkpoint_is_also_supported(self):
        self.checkpoint = self._checkpoint(source=self.cfg, step=2)
        self.assertEqual(self._evaluate()["teacher"]["checkpoint_step"], 2)

    def test_checkpoint_source_config_and_fixed_b_are_verified(self):
        self.source_config_path.write_text(json.dumps({**self.source_cfg, "seed": 18}))
        with self.assertRaisesRegex(ValueError, "configuration hash"):
            self._evaluate()
        self.source_config_path.write_text(json.dumps(self.source_cfg))
        self.cfg["model"]["b_model"]["revision"] = "1" * 40
        with self.assertRaisesRegex(ValueError, "fixed B"):
            self._evaluate()

    def test_snapshot_path_override_normalizes_old_absolute_repo_and_public_repo(self):
        snapshot = self.root / "models--fixture--B" / "snapshots" / ("0" * 40)
        snapshot.mkdir(parents=True)
        source = config()
        source["model"]["b_model"]["repo"] = str(snapshot)
        self.cfg["model"]["b_model"]["repo"] = "fixture/B"
        self.checkpoint = self._checkpoint(source=source, step=2)
        report = self._evaluate(b_model_path=snapshot)
        self.assertEqual(report["teacher"]["b_model"], {"repo": "fixture/B", "revision": "0" * 40})
        with self.assertRaisesRegex(ValueError, "local HF snapshot path"):
            self._evaluate(self.root / "bad-override", b_model_path=self.root)

    def test_completed_evaluation_recomputes_metrics_from_predictions(self):
        report = self._evaluate()
        path = self.root / "evaluation" / "report.json"
        for update_receipt in (False, True):
            with self.subTest(update_receipt=update_receipt):
                changed = json.loads(json.dumps(report))
                changed["metrics"]["ndcg@10"] = 9.9
                if update_receipt:
                    changed["metrics_sha256"] = trainer._digest(changed["metrics"])
                path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "metrics differ from its predictions"):
                    self._evaluate()
        path.write_text(json.dumps(report))
        predictions = self.root / "evaluation" / "predictions.jsonl"
        rows = [json.loads(line) for line in predictions.read_text().splitlines()]
        rows[0]["rank"] = 100
        predictions.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        report["files"]["predictions.jsonl"] = teacher._record(predictions)
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "rank differs from its scores"):
            self._evaluate()

    def test_actual_qwen_old_teacher_weights_score_new_100_pool(self):
        from transformers import Qwen3Config, Qwen3Model
        from duolm.next_item_model import GroupRanker
        torch.set_num_threads(1)
        class Tokenizer:
            eos_token_id = 1
            def encode(self, text, add_special_tokens=False):
                return [2 + sum(text.encode()) % 62]
        def factory(cfg, device):
            definition = Qwen3Config(vocab_size=64, hidden_size=8, intermediate_size=16,
                                    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                                    head_dim=4, max_position_embeddings=2048, attention_dropout=0,
                                    use_cache=False)
            definition._attn_implementation = "sdpa"
            return GroupRanker(None, Qwen3Model(definition), None, Tokenizer(), variant="T",
                               max_reader_packed_tokens=1000, max_encoder_packed_tokens=1000,
                               max_logical_tokens=1000).to(device)
        model = factory(self.source_cfg, "cpu")
        optimizer = trainer.build_optimizer(model, self.source_cfg)
        manifest = json.loads((self.checkpoint / "manifest.json").read_text())
        (self.root / "qwen-teacher").mkdir()
        self.checkpoint = trainer.save_checkpoint(self.root / "qwen-teacher", model, optimizer,
                {"step": 4000, "cursor": 128000}, manifest["identity"])
        self.cfg["training"]["microbatch_queries"] = 1
        report = teacher.evaluate_checkpoint(self.checkpoint, self.source_config_path, self.cfg, self.data,
                self.root / "qwen-evaluation", device="cpu", formal=False, model_factory=factory)
        _, dev, items, cohorts, _ = trainer.load_dataset(self.data, self.cfg, formal=False)
        selector = cohorts["checkpoint_selector"]
        expected = trainer.evaluate(model, [q for q in dev if q.user_id in selector], items,
                expected_users=selector, device="cpu", compute_dtype="float32", max_reader_packed_tokens=1000,
                evaluation_scope=trainer.CANDIDATE_PROTOCOL)
        self.assertEqual(report["metrics"], expected)
        self.assertEqual(report["teacher"]["checkpoint_step"], 4000)
        rows = [json.loads(line) for line in (self.root / "qwen-evaluation" / "predictions.jsonl").read_text().splitlines()]
        self.assertTrue(all(len(row["scores"]) == 100 for row in rows))

    def test_missing_or_wrong_shape_scalar_head_is_rejected(self):
        for corruption in ("missing", "shape"):
            with self.subTest(corruption=corruption):
                self.checkpoint = self._checkpoint(step=2 if corruption == "missing" else 3)
                def change(payload):
                    if corruption == "missing":
                        del payload["model"]["head.weight"]
                    else:
                        payload["model"]["head.weight"] = torch.zeros(1, 2)
                self._rewrite_payload(change)
                with self.assertRaisesRegex(RuntimeError, "head.weight"):
                    self._evaluate(self.root / ("bad-head-" + corruption))

    def test_star_rating_checkpoint_is_rejected(self):
        manifest_path = self.checkpoint / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["protocol"] = "books-five-star-training-v1"
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "identity"):
            self._evaluate()
        source = {**self.source_cfg, "schema_version": "books-five-star-v1"}
        self.source_config_path.write_text(json.dumps(source))
        with self.assertRaisesRegex(ValueError, "star-rating"):
            self._evaluate()

    def test_checkpoint_weight_bytes_and_payload_state_or_identity_corruption_rejected(self):
        for corruption in ("weights", "state", "identity"):
            with self.subTest(corruption=corruption):
                self.checkpoint = self._checkpoint(step={"weights": 2, "state": 3, "identity": 4}[corruption])
                if corruption == "weights":
                    with (self.checkpoint / "training.pt").open("ab") as stream:
                        stream.write(b"corrupt")
                else:
                    def change(payload):
                        payload["state"]["cursor"] += 1 if corruption == "state" else 0
                        if corruption == "identity":
                            payload["identity"]["source_sha256"]["next_item_model.py"] = "3" * 64
                    self._rewrite_payload(change)
                with self.assertRaisesRegex(ValueError, "checksum|signature|identity"):
                    self._evaluate(self.root / ("bad-" + corruption))

    def test_cache_resume_reuses_committed_shards_and_finishes_original_scalar_schema(self):
        first = self._cache(stop_after_shards=1)
        self.assertFalse(first["completed"])
        self.assertEqual(first["next_query"], 2)
        self.assertFalse((self.root / "cache" / "teacher-cache.json").exists())
        with self.assertRaises(FileNotFoundError):
            trainer.load_scalar_teacher_cache(self.root / "cache" / "teacher-cache.json", [], {})
        second = self._cache(stop_after_shards=1)
        self.assertEqual(second["next_query"], 4)
        final = self._cache(stop_after_shards=1)
        self.assertTrue(final["completed"])
        self.assertEqual(len(PoolRanker.calls), 5)
        queries, _, items, _, _ = trainer.load_dataset(self.data, self.cfg, formal=False)
        path = self.root / "cache" / "teacher-cache.json"
        parsed = json.loads(path.read_text())
        self.assertEqual(parsed["schema"], teacher.CACHE_SCHEMA)
        self.assertEqual(list(parsed["queries"]), [q.query_id for q in queries])
        loaded, _ = trainer.load_scalar_teacher_cache(path, queries, items, expected_teacher=final["teacher"])
        self.assertEqual(len(loaded), 5)
        self.assertTrue(all(len(value) == 100 for value in loaded.values()))
        self.assertEqual(self._cache(), final)
        self.assertEqual(len(PoolRanker.calls), 5)
        with self.assertRaisesRegex(ValueError, "teacher identity"):
            trainer.load_scalar_teacher_cache(path, queries, items, expected_teacher={"checkpoint_sha256": "0" * 64})

    def test_partial_completion_and_changed_candidate_order_rejected(self):
        self._cache()
        queries, _, items, _, _ = trainer.load_dataset(self.data, self.cfg, formal=False)
        path = self.root / "cache" / "teacher-cache.json"
        completion_path = path.with_name(path.name + ".manifest.json")
        original = completion_path.read_text()
        completion = json.loads(original)
        completion["completed"] = False
        completion_path.write_text(json.dumps(completion))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            trainer.load_scalar_teacher_cache(path, queries, items)
        completion_path.write_text(original)
        changed = [replace(queries[0], candidate_ids=tuple(reversed(queries[0].candidate_ids)),
                           positive_index=99 - queries[0].positive_index), *queries[1:]]
        with self.assertRaisesRegex(ValueError, "semantic|candidate"):
            trainer.load_scalar_teacher_cache(path, changed, items)
        with self.assertRaisesRegex(ValueError, "sequence"):
            trainer.load_scalar_teacher_cache(path, tuple(reversed(queries)), items)

    def test_shard_identity_and_candidate_sequence_tampering_rejected_on_resume(self):
        self._cache(stop_after_shards=1)
        path = self.root / "cache" / "shards" / "queries-000000000.jsonl"
        manifest_path = path.with_name(path.name + ".manifest.json")
        manifest = json.loads(manifest_path.read_text())
        manifest["teacher"]["checkpoint_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "shard checksum/identity"):
            self._cache()
        manifest["teacher"] = json.loads((self.root / "cache" / "teacher_identity.json").read_text())
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["candidate_ids"].reverse()
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        manifest["file"] = teacher._record(path)
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "candidate order"):
            self._cache()

    def test_output_cannot_be_reused_for_another_checkpoint_or_unknown_contents(self):
        self._evaluate()
        self.checkpoint = self._checkpoint(step=2)
        with self.assertRaisesRegex(ValueError, "output manifest"):
            self._evaluate()
        unknown = self.root / "unknown"
        unknown.mkdir()
        (unknown / "keep.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "new directory"):
            self._evaluate(unknown)
        self.assertEqual((unknown / "keep.json").read_text(), "{}")

    def test_source_mutation_during_merge_cannot_publish_complete_cache(self):
        queries, _, items, _, _ = trainer.load_dataset(self.data, self.cfg, formal=False)
        original_merge = teacher._merge_cache
        def mutate(*args, **kwargs):
            result = original_merge(*args, **kwargs)
            with (self.data / "train.jsonl").open("a") as stream:
                stream.write("\n")
            return result
        with patch.object(teacher, "_merge_cache", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "source changed"):
                self._cache()
        self.assertFalse((self.root / "cache" / "report.json").exists())
        with self.assertRaisesRegex(ValueError, "incomplete"):
            trainer.load_scalar_teacher_cache(self.root / "cache" / "teacher-cache.json", queries, items)


class CandidateTrainingTests(unittest.TestCase):
    def test_one_epoch_candidate100_training_and_hr1_scope(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg = candidate_dataset(root / "data", unavailable=True)
            trainer.validate_config(cfg, formal=False)
            report = trainer.run_training(cfg, root / "data", root / "run", device="cpu", formal=False,
                                          model_factory=lambda config, device: PoolRanker())
            self.assertTrue(report["completed"])
            self.assertEqual(report["protocol"], trainer.PROTOCOL)
            metrics = report["best_full_dev"]
            self.assertEqual(metrics["hr@1"], 2 / 3)
            self.assertEqual(metrics["candidate_target_coverage"], 2 / 3)
            self.assertNotIn("retrieval_recall", metrics)
            identity = report["identity"]
            self.assertIn("prepare_candidate_ranking.py", identity["source_sha256"])
            self.assertIn("candidate_teacher.py", identity["source_sha256"])

    def test_candidate_config_counts_target_requirement_and_manifest_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg = candidate_dataset(root / "data")
            train, dev, _, _, _ = trainer.load_dataset(root / "data", cfg, formal=False)
            wrong = json.loads(json.dumps(cfg))
            wrong["data"]["train_candidates"] = 8
            with self.assertRaisesRegex(ValueError, "group sizes"):
                trainer.load_dataset(root / "data", wrong, formal=False)
            wrong["data"]["train_candidates"] = 99
            with self.assertRaisesRegex(ValueError, "train_candidates"):
                trainer.validate_config(wrong, formal=False)
            manifest_path = root / "data" / "preparation-manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["identity_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "identity/policy/counts"):
                trainer.load_dataset(root / "data", cfg, formal=False)
            # A missing target must fail even before the manifest/hash checks.
            missing = replace(dev[0], candidate_ids=tuple(i for i in dev[0].candidate_ids if i != "positive") + ("negative-119",),
                              positive_index=None)
            write_queries(root / "data" / "dev.jsonl", [missing, *dev[1:]])
            with self.assertRaisesRegex(ValueError, "target in every"):
                trainer.load_dataset(root / "data", cfg, formal=False)


if __name__ == "__main__":
    unittest.main()
