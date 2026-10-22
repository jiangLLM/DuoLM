"""Offline CPU proof of the public teacher -> cache -> student -> evaluation flow."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3Model

from duolm import candidate_teacher, evaluate, train_next_item as trainer
from duolm.next_item_model import GroupRanker
from tests.test_core_train_next_item import candidate_dataset, config


class Tokenizer:
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [2 + sum(text.encode()) % 62]


def tiny_model(config, device):
    def backbone():
        definition = Qwen3Config(vocab_size=64, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=4, max_position_embeddings=2048, attention_dropout=0, use_cache=False)
        definition._attn_implementation = "sdpa"
        return Qwen3Model(definition)
    spec = config["model"]
    return GroupRanker(None if spec["variant"] == "T" else backbone(), backbone(),
        None if spec["variant"] == "T" else Tokenizer(), Tokenizer(),
        variant=spec["variant"], m=1, a_readout=spec.get("a_readout", "perceiver"),
        shared_candidate_header=spec.get("shared_candidate_header", False),
        score_readout=spec.get("score_readout", "legacy"), resampler_heads=2,
        max_reader_packed_tokens=1000, max_encoder_packed_tokens=1000,
        max_logical_tokens=1000, domain=trainer.config_domain(config)).to(device)


class PublicPipelineTests(unittest.TestCase):
    def test_teacher_cache_student_evaluation_and_exact_resume_without_private_services(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_data = root / "candidates"
            student_config = candidate_dataset(candidate_data, train_candidates=8)
            source_data = root / "candidates-source"
            teacher_config = config()
            teacher_run = root / "teacher"
            teacher_report = trainer.run_training(teacher_config, source_data, teacher_run,
                device="cpu", formal=False, model_factory=tiny_model)
            teacher_checkpoint = teacher_run / teacher_report["state"]["best_checkpoint"]
            cache_report = candidate_teacher.cache_checkpoint(teacher_checkpoint,
                teacher_run / "resolved-config.json", student_config, candidate_data, root / "cache",
                device="cpu", formal=False, model_factory=tiny_model, shard_queries=2)
            self.assertTrue(cache_report["completed"])
            student_config["model"].update(variant="V3", a_model={"repo": "fixture", "revision": "0" * 40},
                m=1, a_readout="last_token", shared_candidate_header=True, score_readout="candidate")
            student_config["loss"]["recipe"] = "ce_kd"
            bound = trainer.bind_teacher_cache(student_config, root / "cache")
            whole = trainer.run_training(bound, candidate_data, root / "whole", device="cpu",
                formal=False, model_factory=tiny_model)
            partial = trainer.run_training(bound, candidate_data, root / "resumed", device="cpu",
                formal=False, model_factory=tiny_model, max_updates=1)
            self.assertFalse(partial["completed"])
            resumed = trainer.run_training(bound, candidate_data, root / "resumed", device="cpu",
                formal=False, model_factory=tiny_model, resume=partial["checkpoint"])
            self.assertEqual(whole["state"], resumed["state"])
            self.assertEqual(whole["best_full_dev"], resumed["best_full_dev"])
            a = torch.load(Path(whole["final_checkpoint"]) / "training.pt", weights_only=True)
            b = torch.load(Path(resumed["final_checkpoint"]) / "training.pt", weights_only=True)
            def same(left, right):
                if isinstance(left, torch.Tensor):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                elif isinstance(left, dict):
                    self.assertEqual(left.keys(), right.keys())
                    for key in left:
                        same(left[key], right[key])
                elif isinstance(left, (list, tuple)):
                    self.assertEqual(len(left), len(right))
                    for x, y in zip(left, right):
                        same(x, y)
                else:
                    self.assertEqual(left, right)
            same(a["model"], b["model"])
            same(a["optimizer"], b["optimizer"])
            self.assertEqual(a["identity"]["teacher"]["cache_sha256"], bound["loss"]["teacher_cache_sha256"])
            checkpoint = root / "resumed" / resumed["state"]["best_checkpoint"]
            selected_before = (root / "resumed" / "best_checkpoint.json").read_bytes()
            reports = {}
            for subset in ("selector", "dev", "report"):
                report = evaluate.evaluate_checkpoint(checkpoint, bound, candidate_data, root / subset,
                    subset=subset, device="cpu", formal=False, model_factory=tiny_model)
                self.assertFalse(report["test_labels_accessed"])
                self.assertFalse(report["selects_checkpoint"])
                self.assertTrue(report["weights_only"])
                reports[subset] = report
            self.assertEqual(reports["selector"]["users"] + reports["report"]["users"], reports["dev"]["users"])
            self.assertEqual((root / "resumed" / "best_checkpoint.json").read_bytes(), selected_before)
            # The deliberately invalid sealed test file from the fixture remains unread.
            self.assertEqual((candidate_data / "test.jsonl").exists(), False)
            changed = deepcopy(bound)
            changed["loss"]["teacher_cache_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA256"):
                trainer.bind_teacher_cache(changed, root / "cache")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                trainer.run_training(changed, candidate_data, root / "wrong-cache", device="cpu",
                    formal=False, model_factory=tiny_model)
            with self.assertRaisesRegex(ValueError, "test evaluation"):
                evaluate.evaluate_checkpoint(checkpoint, bound, candidate_data, root / "test", subset="test")

    def test_training_cli_exposes_cache_binding_and_no_delivery_service_flags(self):
        with self.assertRaises(SystemExit) as error, patch("sys.stdout"):
            trainer.main(["--help"])
        self.assertEqual(error.exception.code, 0)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "config.json"
            path.write_text(json.dumps(config()))
            with patch.object(trainer, "run_training", return_value={"completed": True}) as run, patch("sys.stdout"):
                trainer.main(["--config", str(path), "--data-dir", str(root), "--output", str(root / "out")])
            self.assertEqual(set(run.call_args.kwargs), {"device", "resume", "stop_request_file"})

    def test_local_stop_saves_at_boundary_and_requires_explicit_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = candidate_dataset(root / "data", train_candidates=8)
            request = root / "stop"
            request.touch()
            report = trainer.run_training(cfg, root / "data", root / "run", device="cpu", formal=False,
                model_factory=tiny_model, stop_request_file=request)
            self.assertEqual(report["status"], "interrupted")
            self.assertEqual(report["state"]["step"], 0)
            self.assertTrue((Path(report["checkpoint"]) / "training.pt").is_file())
            request.unlink()
            completed = trainer.run_training(cfg, root / "data", root / "run", device="cpu", formal=False,
                model_factory=tiny_model, stop_request_file=request, resume=report["checkpoint"])
            self.assertTrue(completed["completed"])


if __name__ == "__main__":
    unittest.main()
