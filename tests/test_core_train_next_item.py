from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from duolm.next_item_data import HistoryEntry, RecommendationQuery, user_split, write_queries
from duolm.next_item_model import group_ranking_loss, query_input_hash
from duolm import train_next_item as trainer


def users(split, count):
    found, index = [], 0
    while len(found) < count:
        value = f"trainer-fixture-{index}"
        if user_split(value) == split:
            found.append(value)
        index += 1
    return found


def config():
    return {"schema_version": "books-next-item-v1",
            "model": {"variant": "T", "b_model": {"repo": "fixture", "revision": "0" * 40}},
            "data": {"train_users": 2, "dev_users": 3, "test_users": 2, "selection_users": 2,
                     "query_cap": 5, "history_max": 10},
            "training": {"epochs": 1, "effective_query_batch": 3, "microbatch_queries": 2,
                         "backbone_lr": .01, "head_lr": .02, "betas": [.9, .95], "eps": 1e-8,
                         "weight_decay": .01, "warmup_ratio": 0, "min_lr_ratio": .1, "clip": 10,
                         "compute_dtype": "float32", "attention_backend": "sdpa", "gradient_checkpointing": False,
                         "max_reader_packed_tokens": 1000, "max_encoder_packed_tokens": 1000,
                         "max_logical_tokens": 1000},
            "loss": {"recipe": "ce", "beta_final": .5, "beta_warmup_ratio": .1, "temperature": 1,
                     "alignment_weight": 0, "raw_ce_weight": 0},
            "evaluation": {"selection_every": 1, "full_epoch_end": True, "monitor_every": 1,
                           "monitor_users": 1, "max_reader_packed_tokens": 1000},
            "checkpoint": {"first_step": 1, "every": 1}, "seed": 17}


def make_query(user, split, i, positive=0):
    return RecommendationQuery(f"{user}-{i}", user, split, (HistoryEntry("history", 2, 1),),
                               ("positive", "negative"), positive, "positive" if positive is not None else "absent",
                               2 + i, ("history",))


def dataset(path, *, unavailable=False):
    path.mkdir()
    train_users, dev_users, test_users = users("train", 2), users("dev", 3), users("test", 2)
    train = [make_query(train_users[i % 2], "train", i) for i in range(5)]
    dev = [make_query(user, "dev", 0) for user in dev_users]
    if unavailable:
        dev = dev[:-1]
    write_queries(path / "train.jsonl", train)
    write_queries(path / "dev.jsonl", dev)
    (path / "catalog.json").write_text(json.dumps({"history": "old book", "positive": "next book", "negative": "other book"}))
    (path / "cohorts.json").write_text(json.dumps({"train": train_users, "dev": dev_users, "test": test_users,
                                                  "checkpoint_selector": trainer.fixed_selector(dev_users, 2)}))
    (path / "preparation-manifest.json").write_text(json.dumps({"unavailable_dev_users": dev_users[-1:] if unavailable else []}))
    # Deliberately invalid sealed file. Training must never parse or hash it.
    (path / "test.jsonl").write_text("do not open sealed test labels")
    return train, dev


def candidate_dataset(path, *, train_candidates=100, unavailable=False):
    """Use the real converter with frozen source provenance and tiny cohorts."""
    from duolm.prepare_candidate_ranking import prepare_candidate_ranking
    source = path.with_name(path.name + "-source")
    train, dev = dataset(source, unavailable=unavailable)
    train = [replace(q, candidate_ids=("positive", *(f"negative-{i}" for i in range(7)))) for q in train]
    dev = [replace(q, candidate_ids=("positive", *(f"negative-{i}" for i in range(99)))) for q in dev]
    write_queries(source / "train.jsonl", train)
    write_queries(source / "dev.jsonl", dev)
    (source / "catalog.json").write_text(json.dumps({"history": "old book", "positive": "next book",
                                                   **{f"negative-{i}": f"other book {i}" for i in range(120)}}))
    manifest = {"schema": "books-next-item-preparation-v1", "unavailable_dev_users": users("dev", 3)[-1:] if unavailable else [],
                "unavailable_dev_reasons": {users("dev", 3)[-1]: {"missing_target_metadata": 1}} if unavailable else {},
                "files": {name: trainer._file_hash(source / name)
                          for name in ("train.jsonl", "dev.jsonl", "cohorts.json", "catalog.json")},
                "test_labels_accessed": False, "query_cap": 5, "train_candidates": 8, "dev_candidates": 100,
                "train_queries": len(train), "dev_queries": len(dev),
                "sources": {"ratings_sha256": "0" * 64, "catalog_sha256": "1" * 64},
                "catalog_truncation": {"tokenizer": {"repo": "fixture", "revision": "0" * 40}, "max_item_tokens": 512},
                "chronology_audit": {"train": {"legal_queries": len(train)}, "dev": {"legal_queries": len(dev)}},
                "context_manifest_sha256": "2" * 64, "retrieval_manifest_sha256": "3" * 64}
    manifest["retrieval"] = {"schema": "books-next-item-retrieval-v1", "history_only": True,
                             "catalog_sha256": manifest["files"]["catalog.json"], "inputs_sha256": "4" * 64,
                             "results_sha256": "5" * 64, "retriever": {"unit_fixture": True}}
    (source / "preparation-manifest.json").write_text(json.dumps(manifest))
    prepare_candidate_ranking(source, path, train_candidates=train_candidates)
    cfg = config()
    cfg["schema_version"] = trainer.CANDIDATE_PROTOCOL
    cfg["data"].update(train_candidates=train_candidates, eval_candidates=100, candidate_policy="random-unobserved")
    return cfg


class TinyRanker(nn.Module):
    """An injected integration-test model, never an available CLI recipe."""
    def __init__(self):
        super().__init__()
        self.backbone = nn.Parameter(torch.tensor(.3))
        self.head = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.head.weight.fill_(.4)

    def forward(self, queries, items, **kwargs):
        scores = []
        for query in queries:
            scale = int(query.query_id.rsplit("-", 1)[-1]) + 1
            scores.append(torch.stack((self.backbone * self.head.weight[0, 0] * scale,
                                       self.backbone * 0)))
        return {"scores": scores, "statistics": {"reader_tokens": len(queries) * 7}}

    def loss(self, queries, items, **kwargs):
        output = self(queries, items)
        output.update(group_ranking_loss(output["scores"], [q.positive_index for q in queries], **kwargs))
        return output


class TrainUpdateTests(unittest.TestCase):
    def test_single_vector_readout_parameters_use_interface_lr_and_learn(self):
        from tests.test_core_compact_next_item_model import fixture, student
        torch.set_num_threads(1)
        query, items = fixture()
        for readout in ("memory", "perceiver"):
            with self.subTest(readout=readout):
                model = student(a_readout=readout, resampler_layers=1, resampler_heads=8).train()
                model.calibrate_bridge([query], items)
                cfg = config()
                cfg["training"].update(backbone_lr=1e-5, interface_lr=1e-4, head_lr=1e-4)
                optimizer = trainer.build_optimizer(model, cfg)
                interface = {name: p for name, p in model.named_parameters()
                             if name.split(".", 1)[0] in {"memory_token", "resampler", "projection", "bridge"}}
                groups = {id(p): group for group in optimizer.param_groups for p in group["params"]}
                self.assertEqual(len(groups), len(list(model.parameters())))
                for name, parameter in model.named_parameters():
                    group = groups[id(parameter)]
                    self.assertEqual(group["lr"], 1e-4 if name in interface or name.startswith("head.") else 1e-5)
                    if name in interface:
                        self.assertTrue(group["name"].startswith("interface"), name)
                        excluded = parameter.ndim < 2 or name.endswith(".bias") or "norm" in name.lower()
                        self.assertEqual(group["weight_decay"], 0 if excluded else .01, name)
                names = (["memory_token"] if readout == "memory" else
                         ["resampler.queries", "resampler.layers.0.attention.in_proj_weight",
                          "resampler.layers.0.ff.0.weight", "resampler.layers.0.ff.2.weight"])
                parameters = dict(model.named_parameters())
                before = {name: parameters[name].detach().clone() for name in names}
                result = trainer.train_update(model, [query], items, optimizer, microbatch_queries=1,
                    clip=1, device="cpu", compute_dtype="float32", diagnostics=True,
                    teacher_scores={query.query_id: [.7, -.2, .1]}, beta=.5, temperature=1)
                module = "memory_token" if readout == "memory" else "resampler"
                for role in ("a", "b", "head", "bridge", module):
                    self.assertGreater(result["module_learning"][role]["gradient_norm"], 0, role)
                    self.assertGreater(result["module_learning"][role]["sample_max_delta"], 0, role)
                for name in names:
                    parameter = parameters[name]
                    self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
                    self.assertGreater((parameter.detach()-before[name]).abs().max().item(), 0, name)
                    self.assertEqual(parameter.dtype, torch.float32)
                    self.assertEqual(optimizer.state[parameter]["exp_avg"].dtype, torch.float32)
                    self.assertTrue(torch.isfinite(optimizer.state[parameter]["exp_avg_sq"]).all(), name)

    def test_interface_lr_is_separate_from_both_backbones_and_head(self):
        class Student(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(3, 3)
                self.b = nn.Linear(3, 3)
                self.projection = nn.Linear(3, 3, bias=False)
                self.bridge = nn.LayerNorm(3)
                self.score_token = nn.Parameter(torch.ones(3))
                self.head = nn.Linear(3, 1, bias=False)
        model, cfg = Student(), config()
        cfg["training"]["interface_lr"] = .05
        optimizer = trainer.build_optimizer(model, cfg)
        self.assertEqual(optimizer.param_groups[1]["name"], "head")
        expected = {id(p) for p in (*model.projection.parameters(), *model.bridge.parameters(), model.score_token)}
        actual = set()
        for group in optimizer.param_groups:
            if group["name"].startswith("interface"):
                self.assertEqual(group["lr"], .05)
                actual.update(id(p) for p in group["params"])
        self.assertEqual(actual, expected)

    def test_first_update_records_real_gradient_and_weight_change(self):
        model, cfg = TinyRanker(), config()
        result = trainer.train_update(model, [make_query(users("train", 1)[0], "train", 0)], {},
            trainer.build_optimizer(model, cfg), microbatch_queries=1, clip=10,
            device="cpu", compute_dtype="float32", diagnostics=True)
        self.assertTrue(result["finite_gradients"])
        for row in result["module_learning"].values():
            self.assertGreater(row["gradient_norm"], 0)
            self.assertGreater(row["sample_max_delta"], 0)

    def test_uneven_microbatches_and_tail_equal_complete_batch_gradient(self):
        qs = [SimpleNamespace(query_id=f"q-{i}", positive_index=i % 2, candidate_ids=("a", "b")) for i in range(5)]
        for count in (5, 2):
            whole, split = TinyRanker(), TinyRanker()
            first = torch.optim.SGD(whole.parameters(), lr=.1)
            second = torch.optim.SGD(split.parameters(), lr=.1)
            trainer.train_update(whole, qs[:count], {}, first, microbatch_queries=count, clip=100,
                                 device="cpu", compute_dtype="float32")
            trainer.train_update(split, qs[:count], {}, second, microbatch_queries=2, clip=100,
                                 device="cpu", compute_dtype="float32")
            for a, b in zip(whole.parameters(), split.parameters()):
                torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)

    def test_config_rejects_raw_alignment_old_recipe_and_formal_tiny_caps(self):
        for key, value in (("raw_ce_weight", 1), ("alignment_weight", .1), ("recipe", "full")):
            cfg = config()
            cfg["loss"][key] = value
            with self.assertRaises(ValueError):
                trainer.validate_config(cfg, formal=False)
        with self.assertRaises(ValueError):
            trainer.validate_config(config(), formal=True)
        cfg = config()
        cfg["model"]["b_model"]["revision"] = "main"
        cfg["training"]["attention_backend"] = "flex_attention"
        with self.assertRaisesRegex(ValueError, "immutable"):
            trainer.validate_config(cfg, formal=True)

    def test_model_parameters_must_all_update_and_remain_fp32(self):
        model = TinyRanker()
        model.backbone.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "all parameters"):
            trainer.build_optimizer(model, config())

    def test_optimizer_decay_groups_keep_norm_bias_scalar_and_head_exempt(self):
        class Parameters(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Linear(3, 3)
                self.norm = nn.LayerNorm(3)
                self.gain = nn.Parameter(torch.tensor(0.0))
                self.head = nn.Linear(3, 1, bias=False)
        model = Parameters()
        optimizer = trainer.build_optimizer(model, config())
        decay, head, excluded = optimizer.param_groups
        self.assertEqual(decay["name"], "backbone_decay")
        self.assertEqual(decay["weight_decay"], config()["training"]["weight_decay"])
        self.assertEqual({id(p) for p in decay["params"]}, {id(model.backbone.weight)})
        self.assertEqual({id(p) for p in head["params"]}, {id(model.head.weight)})
        self.assertEqual(head["weight_decay"], 0)
        self.assertEqual({id(p) for p in excluded["params"]},
                         {id(model.backbone.bias), id(model.norm.weight), id(model.norm.bias), id(model.gain)})
        self.assertEqual(excluded["weight_decay"], 0)
        for group in (decay, excluded):
            self.assertEqual(group["lr"], config()["training"]["backbone_lr"])
            self.assertEqual(group["base_lr"], group["lr"])
        self.assertEqual(head["lr"], config()["training"]["head_lr"])
        self.assertEqual(head["base_lr"], head["lr"])

    def test_formal_schema_required_but_python_fixture_can_omit_it(self):
        cfg = config()
        cfg.pop("schema_version")
        trainer.validate_config(cfg, formal=False)
        with self.assertRaisesRegex(ValueError, "schema_version"):
            trainer.validate_config(cfg, formal=True)

    def test_schedules_are_configurable(self):
        cfg = config()["loss"]
        cfg.update(recipe="ce_kd", beta_final=.9, beta_warmup_ratio=.5, temperature=4)
        self.assertAlmostEqual(trainer.beta_for_step(1, 10, cfg), .18)
        self.assertAlmostEqual(trainer.beta_for_step(10, 10, cfg), .9)
        self.assertAlmostEqual(trainer.lr_factor(10, 10, .1, .1), .1)


class ResumeAndDataTests(unittest.TestCase):
    def test_single_vector_readout_resume_matches_uninterrupted_ce_kd_training(self):
        from duolm.next_item_model import GroupRanker
        from tests.test_core_compact_next_item_model import Tokenizer, backbone
        torch.set_num_threads(1)
        for readout in ("memory", "perceiver"):
            with self.subTest(readout=readout), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                queries, _ = dataset(root / "data")
                items = json.loads((root / "data/catalog.json").read_text())
                teacher = {"checkpoint_sha256": "0" * 64}
                cache = {"schema": "books-next-item-scalar-teacher-v1", "teacher": teacher,
                         "queries": {q.query_id: {"input_hash": query_input_hash(q, items),
                                     "candidate_ids": list(q.candidate_ids), "scores": [.7, -.2]}
                                     for q in queries}}
                (root / "teacher.json").write_text(json.dumps(cache))
                cfg = config()
                cfg["model"].update(variant="V3", a_model={"repo": "fixture", "revision": "0" * 40},
                                    a_readout=readout, m=1, resampler_layers=1, resampler_heads=8,
                                    score_readout="candidate", shared_candidate_header=True)
                cfg["loss"].update(recipe="ce_kd", teacher_cache=str(root / "teacher.json"),
                                   teacher_identity=teacher)
                cfg["training"].update(backbone_lr=1e-5, head_lr=1e-4, interface_lr=1e-4,
                                       gradient_checkpointing=True, microbatch_queries=1)
                def factory(options, device):
                    spec = options["model"]
                    return GroupRanker(backbone(), backbone(), Tokenizer(), Tokenizer(), variant="V3",
                        m=1, a_readout=spec["a_readout"], resampler_layers=1, resampler_heads=8,
                        score_readout="candidate", shared_candidate_header=True, gradient_checkpointing=True,
                        max_encoder_packed_tokens=1000, max_reader_packed_tokens=1000).to(device)
                full = trainer.run_training(cfg, root / "data", root / "full", device="cpu", formal=False,
                                            model_factory=factory)
                partial = trainer.run_training(cfg, root / "data", root / "resume", device="cpu", formal=False,
                                               model_factory=factory, max_updates=1)
                resumed = trainer.run_training(cfg, root / "data", root / "resume", device="cpu", formal=False,
                                               model_factory=factory, resume=partial["checkpoint"])
                self.assertTrue(resumed["completed"])
                self.assertEqual(resumed["state"], full["state"])
                self.assertEqual(resumed["best_full_dev"], full["best_full_dev"])
                first = torch.load(Path(full["final_checkpoint"]) / "training.pt", weights_only=True)
                second = torch.load(Path(resumed["final_checkpoint"]) / "training.pt", weights_only=True)
                for name, value in first["model"].items():
                    if torch.is_tensor(value):
                        torch.testing.assert_close(value, second["model"][name], atol=0, rtol=0, msg=name)
                    else:
                        self.assertEqual(value, second["model"][name], name)
                self.assertEqual(first["optimizer"]["param_groups"], second["optimizer"]["param_groups"])
                self.assertEqual(first["optimizer"]["state"].keys(), second["optimizer"]["state"].keys())
                for parameter, state in first["optimizer"]["state"].items():
                    for name, value in state.items():
                        torch.testing.assert_close(value, second["optimizer"]["state"][parameter][name], atol=0, rtol=0)
                self.assertEqual(first["rng_python"], second["rng_python"])
                torch.testing.assert_close(first["rng_torch"], second["rng_torch"], atol=0, rtol=0)
                for name in ("best-dev", "final-dev"):
                    self.assertEqual((root / f"full/predictions/{name}.jsonl").read_bytes(),
                                     (root / f"resume/predictions/{name}.jsonl").read_bytes())

    def test_step0_and_full_dev_predictions_are_saved_with_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset(root / "data")
            result = trainer.run_training(config(), root / "data", root / "run", device="cpu", formal=False,
                                          model_factory=lambda *_: TinyRanker())
            for name, step, count in (("step0-monitor", 0, 1), ("final-dev", result["state"]["step"], 3),
                                      ("best-dev", result["best_step"], 3)):
                path = root / "run/predictions" / (name + ".jsonl")
                manifest = json.loads(path.with_suffix(".manifest.json").read_text())
                self.assertEqual(manifest["step"], step)
                self.assertEqual(manifest["rows"], count)
                self.assertEqual(manifest["file"]["sha256"], trainer._file_hash(path))
                self.assertEqual(manifest["identity_sha256"], trainer._digest(result["identity"]))
            self.assertTrue((root / "run/initialization.json").is_file())

    def test_actual_qwen_shared_tree_student_updates_and_restores_calibration(self):
        from transformers import Qwen3Config, Qwen3Model
        from duolm.next_item_model import GroupRanker
        torch.set_num_threads(1)
        class Tokenizer:
            eos_token_id = 1
            def encode(self, text, add_special_tokens=False):
                return [2 + ord(c) % 62 for c in text]
        def factory(cfg, device):
            def backbone():
                definition = Qwen3Config(vocab_size=64, hidden_size=8, intermediate_size=16,
                                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                                         head_dim=4, max_position_embeddings=2048, attention_dropout=0,
                                         use_cache=False)
                definition._attn_implementation = "sdpa"
                return Qwen3Model(definition)
            return GroupRanker(backbone(), backbone(), Tokenizer(), Tokenizer(), variant="V3",
                               m=2, resampler_layers=1, resampler_heads=2,
                               gradient_checkpointing=True, max_reader_packed_tokens=1000,
                               max_encoder_packed_tokens=1000, max_logical_tokens=1000)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset(root / "data")
            cfg = config()
            cfg["model"].update(variant="V3", a_model={"repo": "fixture", "revision": "0" * 40})
            interrupted = trainer.run_training(cfg, root / "data", root / "run", device="cpu", formal=False,
                                               model_factory=factory, max_updates=1)
            payload = torch.load(Path(interrupted["checkpoint"]) / "training.pt", weights_only=True)
            self.assertTrue(payload["model"]["bridge.calibrated"].item())
            self.assertEqual(payload["model"]["bridge._extra_state"]["calibration"]["source"], "training_query_content")
            complete = trainer.run_training(cfg, root / "data", root / "run", device="cpu", formal=False,
                                            model_factory=factory, resume=interrupted["checkpoint"])
            self.assertTrue(complete["completed"])
            self.assertEqual(complete["best_full_dev"]["users"], 3)

    def test_resume_matches_uninterrupted_model_optimizer_and_cursor(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset(root / "data")
            factory = lambda cfg, device: TinyRanker()
            full = trainer.run_training(config(), root / "data", root / "full", device="cpu",
                                        formal=False, model_factory=factory)
            partial = trainer.run_training(config(), root / "data", root / "resume", device="cpu", formal=False,
                                           model_factory=factory, max_updates=1)
            continued = trainer.run_training(config(), root / "data", root / "resume", device="cpu", formal=False,
                                             model_factory=factory, resume=partial["checkpoint"])
            self.assertTrue(continued["completed"])
            self.assertEqual(full["state"]["cursor"], 5)
            self.assertEqual(continued["state"], full["state"])
            first = torch.load(Path(full["final_checkpoint"]) / "training.pt", weights_only=True)
            second = torch.load(Path(continued["final_checkpoint"]) / "training.pt", weights_only=True)
            for name in first["model"]:
                torch.testing.assert_close(first["model"][name], second["model"][name], rtol=0, atol=0)
            for key in first["optimizer"]["state"]:
                for name in first["optimizer"]["state"][key]:
                    torch.testing.assert_close(first["optimizer"]["state"][key][name],
                                               second["optimizer"]["state"][key][name], rtol=0, atol=0)

    def test_resume_rejects_another_experiment_output_without_changing_its_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset(root / "data")
            first = trainer.run_training(config(), root / "data", root / "first", device="cpu", formal=False,
                                         model_factory=lambda cfg, device: TinyRanker(), max_updates=1)
            other_config = config()
            other_config["training"]["head_lr"] *= 2
            trainer.run_training(other_config, root / "data", root / "other", device="cpu", formal=False,
                                 model_factory=lambda cfg, device: TinyRanker(), max_updates=1)
            def artifacts():
                return {str(path.relative_to(root / "other")): path.read_bytes()
                        for path in (root / "other").rglob("*") if path.is_file() and path.name != "run.lock"}
            original = artifacts()
            with patch.object(trainer, "train_update") as update, patch.object(trainer, "GroupRanker") as model:
                with self.assertRaisesRegex(ValueError, "resume output experiment identity"):
                    trainer.run_training(config(), root / "data", root / "other", device="cpu", formal=False,
                                         resume=first["checkpoint"])
                model.from_pretrained.assert_not_called()
                update.assert_not_called()
            self.assertEqual(artifacts(), original)

    def test_resume_rejects_nonempty_output_with_missing_or_damaged_run_manifest(self):
        for corruption in ("missing", "invalid_json", "missing_identity", "wrong_protocol", "wrong_config"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                dataset(root / "data")
                partial = trainer.run_training(config(), root / "data", root / "old", device="cpu", formal=False,
                                               model_factory=lambda cfg, device: TinyRanker(), max_updates=1)
                shutil.copytree(root / "old", root / "restored")
                path = root / "restored" / "run_manifest.json"
                manifest = json.loads(path.read_text())
                if corruption == "missing":
                    path.unlink()
                elif corruption == "invalid_json":
                    path.write_text("unfinished {")
                else:
                    if corruption == "missing_identity":
                        manifest.pop("identity")
                    elif corruption == "wrong_protocol":
                        manifest["protocol"] = "different-protocol"
                    else:
                        manifest["config"]["seed"] += 1
                    path.write_text(json.dumps(manifest))
                original = {str(item.relative_to(root / "restored")): item.read_bytes()
                            for item in (root / "restored").rglob("*") if item.is_file() and item.name != "run.lock"}
                with patch.object(trainer, "train_update") as update, patch.object(trainer, "GroupRanker") as model:
                    with self.assertRaisesRegex(ValueError, "resume output.*identity"):
                        trainer.run_training(config(), root / "data", root / "restored", device="cpu", formal=False,
                                             resume=partial["checkpoint"])
                    model.from_pretrained.assert_not_called()
                    update.assert_not_called()
                self.assertEqual({str(item.relative_to(root / "restored")): item.read_bytes()
                                  for item in (root / "restored").rglob("*")
                                  if item.is_file() and item.name != "run.lock"}, original)

    def test_resume_allows_restored_same_identity_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset(root / "data")
            partial = trainer.run_training(config(), root / "data", root / "old", device="cpu", formal=False,
                                           model_factory=lambda cfg, device: TinyRanker(), max_updates=1)
            shutil.copytree(root / "old", root / "restored")
            result = trainer.run_training(config(), root / "data", root / "restored", device="cpu", formal=False,
                                          model_factory=lambda cfg, device: TinyRanker(), resume=partial["checkpoint"])
            self.assertTrue(result["completed"])
            self.assertEqual(result["state"]["cursor"], 5)
            self.assertEqual(result["state"]["step"], 2)

    def test_full_dev_is_reporting_only_even_when_it_prefers_another_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset(root / "data")
            calls = []
            def fake_evaluate(model, queries, items, *, expected_users, **kwargs):
                users_count = len(expected_users)
                calls.append(users_count)
                selector_calls = calls.count(2)
                score = 1 if users_count == 2 and selector_calls == 1 else 0
                if users_count == 3:
                    score = 100 if calls.count(3) == 1 else -100
                return {"ndcg@10": score}
            with patch.object(trainer, "evaluate", side_effect=fake_evaluate):
                report = trainer.run_training(config(), root / "data", root / "run", device="cpu", formal=False,
                                              model_factory=lambda cfg, device: TinyRanker())
            self.assertEqual(report["best_step"], 1)
            self.assertFalse(report["full_dev_reselected"])
            self.assertEqual(calls.count(3), 2)

    def test_resume_rejects_changed_data_and_checkpoint_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset(root / "data")
            run = trainer.run_training(config(), root / "data", root / "run", device="cpu", formal=False,
                                       model_factory=lambda cfg, device: TinyRanker(), max_updates=1)
            changed = config()
            changed["training"]["head_lr"] *= 2
            with self.assertRaisesRegex(ValueError, "identity"):
                trainer.run_training(changed, root / "data", root / "run", device="cpu", formal=False,
                                     model_factory=lambda cfg, device: TinyRanker(), resume=run["checkpoint"])
            catalog_path = root / "data" / "catalog.json"
            original_catalog = catalog_path.read_text()
            catalog_path.write_text(original_catalog + "\n")
            with self.assertRaisesRegex(ValueError, "identity"):
                trainer.run_training(config(), root / "data", root / "run", device="cpu", formal=False,
                                     model_factory=lambda cfg, device: TinyRanker(), resume=run["checkpoint"])
            catalog_path.write_text(original_catalog)
            (Path(run["checkpoint"]) / "training.pt").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                trainer.run_training(config(), root / "data", root / "run", device="cpu", formal=False,
                                     model_factory=lambda cfg, device: TinyRanker(), resume=run["checkpoint"])

    def test_unavailable_dev_is_preserved_and_test_file_is_not_opened(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset(root / "data", unavailable=True)
            opened = []
            original = Path.open
            def tracked(path, *args, **kwargs):
                opened.append(path.name)
                return original(path, *args, **kwargs)
            with patch.object(Path, "open", tracked):
                report = trainer.run_training(config(), root / "data", root / "run", device="cpu", formal=False,
                                              model_factory=lambda cfg, device: TinyRanker())
            self.assertNotIn("test.jsonl", opened)
            self.assertEqual(report["best_full_dev"]["users"], 3)
            self.assertEqual(report["best_full_dev"]["unavailable_users"], 1)

    def test_unavailable_dev_cannot_hide_undeclared_omissions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset(root / "data", unavailable=True)
            (root / "data" / "preparation-manifest.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "complete"):
                trainer.load_dataset(root / "data", config(), formal=False)

    def test_formal_preparation_manifest_is_checked_against_actual_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            train, dev = dataset(root / "data")
            train = [replace(q, candidate_ids=("positive", *(f"negative-{i}" for i in range(7)))) for q in train]
            dev = [replace(q, candidate_ids=("positive", *(f"negative-{i}" for i in range(99)))) for q in dev]
            write_queries(root / "data" / "train.jsonl", train)
            write_queries(root / "data" / "dev.jsonl", dev)
            (root / "data" / "catalog.json").write_text(json.dumps({"history": "old", "positive": "next",
                                                                   **{f"negative-{i}": f"other {i}" for i in range(99)}}))
            manifest = {"schema": "books-next-item-preparation-v1", "unavailable_dev_users": [],
                        "unavailable_dev_reasons": {},
                        "files": {name: trainer._file_hash(root / "data" / name)
                                  for name in ("train.jsonl", "dev.jsonl", "cohorts.json", "catalog.json")}}
            manifest.update(test_labels_accessed=False, query_cap=5, train_candidates=8, dev_candidates=100,
                            sources={"ratings_sha256": "0" * 64, "catalog_sha256": "1" * 64},
                            catalog_truncation={"tokenizer": {"repo": "fixture", "revision": "0" * 40}, "max_item_tokens": 512},
                            context_manifest_sha256="2" * 64, retrieval_manifest_sha256="3" * 64,
                            retrieval={"schema": "books-next-item-retrieval-v1", "history_only": True,
                                       "catalog_sha256": manifest["files"]["catalog.json"], "inputs_sha256": "4" * 64,
                                       "results_sha256": "5" * 64, "retriever": {"unit_fixture": True}})
            path = root / "data" / "preparation-manifest.json"
            path.write_text(json.dumps(manifest))
            trainer.load_dataset(root / "data", config(), formal=True)
            manifest["files"]["train.jsonl"] = "1" * 64
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "file hashes"):
                trainer.load_dataset(root / "data", config(), formal=True)

    def test_dev_retrieval_miss_still_requires_target_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, dev = dataset(root / "data")
            dev[0] = replace(dev[0], positive_index=None, target_item_id="missing-target")
            write_queries(root / "data" / "dev.jsonl", dev)
            with self.assertRaisesRegex(ValueError, "catalog is incomplete"):
                trainer.load_dataset(root / "data", config(), formal=False)

    def test_step_checkpoint_is_immutable_and_epoch_serializes_each_step_once(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset(root / "data")
            original = torch.save
            with patch.object(torch, "save", wraps=original) as saves:
                report = trainer.run_training(config(), root / "data", root / "run", device="cpu", formal=False,
                                              model_factory=lambda cfg, device: TinyRanker())
            self.assertEqual(saves.call_count, 2)
            model = TinyRanker()
            optimizer = trainer.build_optimizer(model, config())
            payload = torch.load(Path(report["final_checkpoint"]) / "training.pt", weights_only=True)
            with patch.object(torch, "save", side_effect=AssertionError("must not reserialize immutable step")):
                trainer.save_checkpoint(root / "run", model, optimizer, payload["state"], payload["identity"])
            changed_state = {**payload["state"], "best_ndcg": .123456}
            with self.assertRaisesRegex(ValueError, "immutable"):
                trainer.save_checkpoint(root / "run", model, optimizer, changed_state, payload["identity"])

    def test_new_output_resume_restores_original_best_even_without_improvement(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset(root / "data")
            cfg = config()
            cfg["training"]["effective_query_batch"] = 2
            selector_calls = 0
            def evaluation(model, queries, items, *, expected_users, **kwargs):
                nonlocal selector_calls
                if len(expected_users) == 2:
                    selector_calls += 1
                    return {"ndcg@10": .8 if selector_calls == 1 else .1}
                return {"ndcg@10": .2}
            with patch.object(trainer, "evaluate", side_effect=evaluation):
                partial = trainer.run_training(cfg, root / "data", root / "old", device="cpu", formal=False,
                                               model_factory=lambda cfg, device: TinyRanker(), max_updates=2)
                self.assertEqual(partial["state"]["best_step"], 1)
                result = trainer.run_training(cfg, root / "data", root / "new", device="cpu", formal=False,
                                              model_factory=lambda cfg, device: TinyRanker(), resume=partial["checkpoint"])
            self.assertTrue(result["completed"])
            self.assertEqual(result["best_step"], 1)
            old_best = root / "old" / partial["state"]["best_checkpoint"]
            new_best = root / "new" / partial["state"]["best_checkpoint"]
            self.assertEqual(trainer._file_hash(old_best / "training.pt"), trainer._file_hash(new_best / "training.pt"))

    def test_missing_or_corrupt_inherited_best_fails_before_any_update(self):
        for corruption in ("missing", "hash"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                dataset(root / "data")
                cfg = config()
                cfg["training"]["effective_query_batch"] = 2
                calls = 0
                def evaluation(model, queries, items, *, expected_users, **kwargs):
                    nonlocal calls
                    if len(expected_users) == 2:
                        calls += 1
                        return {"ndcg@10": .8 if calls == 1 else .1}
                    return {"ndcg@10": .2}
                with patch.object(trainer, "evaluate", side_effect=evaluation):
                    partial = trainer.run_training(cfg, root / "data", root / "old", device="cpu", formal=False,
                                                   model_factory=lambda cfg, device: TinyRanker(), max_updates=2)
                best = root / "old" / partial["state"]["best_checkpoint"]
                if corruption == "missing":
                    shutil.rmtree(best)
                    pattern = "restore it together with latest"
                else:
                    (best / "training.pt").write_bytes(b"corrupt best weights")
                    pattern = "checksum"
                with patch.object(trainer, "train_update") as update:
                    with self.assertRaisesRegex(ValueError, pattern):
                        trainer.run_training(cfg, root / "data", root / "new", device="cpu", formal=False,
                                             model_factory=lambda cfg, device: TinyRanker(), resume=partial["checkpoint"])
                    update.assert_not_called()
                self.assertFalse((root / "new" / partial["state"]["best_checkpoint"]).exists())

    def test_scalar_cache_identity_feedback_and_candidate_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            queries, _ = dataset(root / "data")
            items = json.loads((root / "data" / "catalog.json").read_text())
            cache = {"schema": "books-next-item-scalar-teacher-v1", "teacher": {"checkpoint_sha256": "0" * 64},
                     "queries": {q.query_id: {"input_hash": query_input_hash(q, items),
                                                "candidate_ids": list(q.candidate_ids), "scores": [.2, -.3]}
                                 for q in queries}}
            path = root / "teacher.json"
            path.write_text(json.dumps(cache))
            loaded, identity = trainer.load_scalar_teacher_cache(path, queries, items)
            self.assertEqual(loaded[queries[0].query_id], [.2, -.3])
            changed = [replace(queries[0], history=(HistoryEntry("history", 5, 1),)), *queries[1:]]
            with self.assertRaisesRegex(ValueError, "semantic"):
                trainer.load_scalar_teacher_cache(path, changed, items)
            with self.assertRaisesRegex(ValueError, "identity"):
                trainer.load_scalar_teacher_cache(path, queries, items, expected_teacher={"checkpoint_sha256": "1" * 64})
            cache["schema"] = "old-five-class-probabilities"
            path.write_text(json.dumps(cache))
            with self.assertRaisesRegex(ValueError, "five-class"):
                trainer.load_scalar_teacher_cache(path, queries, items)

    def test_transient_training_label_swap_cannot_claim_original_input_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg = candidate_dataset(root / "data")
            path = root / "data" / "train.jsonl"
            original_bytes = path.read_bytes()
            original_reader = trainer.read_queries
            original_queries = original_reader(path, split="train")
            def swap_only_during_parse(filename, *, split, expected_sha256=None):
                if split != "train":
                    return original_reader(filename, split=split, expected_sha256=expected_sha256)
                query = original_queries[0]
                other = (query.positive_index + 1) % len(query.candidate_ids)
                changed = replace(query, target_item_id=query.candidate_ids[other], positive_index=other)
                # Initial/final file hashes both see original bytes. Only a hash
                # of the consumed bytes catches this different CE target.
                write_queries(path, (changed, *original_queries[1:]))
                try:
                    return original_reader(filename, split=split, expected_sha256=expected_sha256)
                finally:
                    path.write_bytes(original_bytes)
            with patch.object(trainer, "read_queries", side_effect=swap_only_during_parse):
                with self.assertRaisesRegex(ValueError, "parsed query bytes differ from frozen input hash"):
                    trainer.load_dataset(root / "data", cfg, formal=False)
            self.assertEqual(path.read_bytes(), original_bytes)


if __name__ == "__main__":
    unittest.main()
