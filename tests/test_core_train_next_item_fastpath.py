"""Books and Games training fastpath invariants; CPU checks make no GPU capacity claim."""
import copy
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from duolm import train_next_item as trainer
from duolm.next_item_model import group_ranking_loss
from tests.test_core_train_next_item import TinyRanker, config, dataset, make_query, users


class DeferredTinyRanker(TinyRanker):
    def loss(self, queries, items, *, teacher_scores=None, beta=0, temperature=1,
             defer_scalar_metrics=False):
        output = self(queries, items)
        output.update(group_ranking_loss(output["scores"], [q.positive_index for q in queries],
            teacher_scores=teacher_scores, beta=beta, temperature=temperature,
            defer_scalar_metrics=defer_scalar_metrics))
        return output


class TrainerFastpathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_optional_engine_fields_are_strict_and_absent_stays_legacy(self):
        cfg = config()
        trainer.validate_config(cfg, formal=False)
        for section, key in (("model", "input_assembly"),
                *(("training", name) for name in ("reader_gradient_checkpointing", "optimizer_fused",
                                                  "teacher_cache_on_device", "cuda_event_timing", "pretokenize_items"))):
            for invalid in (None, 1, "yes", []):
                with self.subTest(key=key, invalid=invalid):
                    bad = copy.deepcopy(cfg)
                    bad[section][key] = invalid
                    with self.assertRaisesRegex(ValueError, key):
                        trainer.validate_config(bad, formal=False)
        bad = copy.deepcopy(cfg)
        bad["training"]["teacher_cache_on_device"] = True
        with self.assertRaisesRegex(ValueError, "CE\\+KD"):
            trainer.validate_config(bad, formal=False)
        bad = copy.deepcopy(cfg)
        bad["training"]["pretokenize_items"] = True
        with self.assertRaisesRegex(ValueError, "V3"):
            trainer.validate_config(bad, formal=False)
        # No explicit fused kwarg is inserted into the old optimizer invocation.
        optimizer = trainer.build_optimizer(TinyRanker(), cfg)
        self.assertIsNone(optimizer.defaults["fused"])
        for value in (False, True):
            opts = copy.deepcopy(cfg)
            opts["training"]["optimizer_fused"] = value
            self.assertIs(trainer.build_optimizer(TinyRanker(), opts).defaults["fused"], value)

    def test_default_loss_caller_does_not_require_new_keyword(self):
        class LegacyOnly(TinyRanker):
            def loss(self, queries, items, *, teacher_scores=None, beta=0, temperature=1):
                return super().loss(queries, items, teacher_scores=teacher_scores, beta=beta,
                                    temperature=temperature)
        query = make_query(users("train", 1)[0], "train", 0)
        model = LegacyOnly()
        trainer.train_update(model, [query], {}, trainer.build_optimizer(model, config()),
                             microbatch_queries=1, clip=10, device="cpu", compute_dtype="float32")

    def test_pretrained_constructor_receives_only_optional_engine_overrides(self):
        for assembly, gc in (("legacy", None), ("batched", False), ("batched", True)):
            with self.subTest(assembly=assembly, gc=gc), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                dataset(root / "data")
                cfg = config()
                if assembly != "legacy":
                    cfg["model"]["input_assembly"] = assembly
                if gc is not None:
                    cfg["training"]["reader_gradient_checkpointing"] = gc
                with patch.object(trainer.GroupRanker, "from_pretrained", return_value=DeferredTinyRanker()) as load:
                    trainer.run_training(cfg, root / "data", root / "run", device="cpu", formal=False)
                self.assertEqual(load.call_args.kwargs["input_assembly"], assembly)
                self.assertIs(load.call_args.kwargs["reader_gradient_checkpointing"], gc)
                self.assertEqual(load.call_args.kwargs["gradient_checkpointing"], cfg["training"]["gradient_checkpointing"])

    def test_cli_accepts_old_reports_without_claiming_completed_and_preserves_interrupted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "config.json"
            source.write_text("{}")
            for report, completed, status in (({"best_step": 7}, None, "unknown"),
                    ({"completed": False, "status": "interrupted"}, False, "interrupted"),
                    ({"completed": True, "best_step": 7}, True, "complete")):
                output = io.StringIO()
                with patch.object(trainer, "run_training", return_value=report), redirect_stdout(output):
                    trainer.main(["--config", str(source), "--data-dir", str(root / "data"), "--output", str(root / "run")])
                row = json.loads(output.getvalue())
                self.assertIs(row["completed"], completed)
                self.assertEqual(row["status"], status)
                self.assertEqual(row["best_step"], report.get("best_step"))

    def test_evaluation_microbatch_field_requires_actual_positive_int(self):
        for value in (True, False, 0, -1, 1.0, 4.0, float("nan"), float("inf"), "4", None, []):
            cfg = config()
            cfg["evaluation"]["microbatch_queries"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "evaluation.microbatch_queries"):
                trainer.validate_config(cfg, formal=False)
        cfg = config()
        cfg["evaluation"]["microbatch_queries"] = 4
        trainer.validate_config(cfg, formal=False)
        # Evaluation packing is independent of the effective training batch.
        self.assertGreater(cfg["evaluation"]["microbatch_queries"], cfg["training"]["effective_query_batch"])
        with self.assertRaisesRegex(ValueError, "evaluation.microbatch_queries"):
            trainer.evaluate(TinyRanker(), [], {}, expected_users=["missing"], device="cpu",
                             compute_dtype="float32", max_reader_packed_tokens=1000,
                             microbatch_queries=True, unavailable_users=["missing"])

    def capacity_fixture(self):
        class CapacityTiny(TinyRanker):
            def __init__(self):
                super().__init__()
                self.evaluation_batches = []
                self.invalid_scores = False

            @property
            def a(self):
                return self.head

            @property
            def b(self):
                return self.head

            def forward(self, queries, items, **kwargs):
                output = super().forward(queries, items, **kwargs)
                output["scores"] = [torch.cat((score, score.new_zeros(len(query.candidate_ids) - 2)))
                                    for query, score in zip(queries, output["scores"])]
                if not self.training:
                    self.evaluation_batches.append([query.query_id for query in queries])
                    if self.invalid_scores:
                        output["scores"][0][0] = float("nan")
                return output

        dev = [replace(make_query(user, "dev", i), candidate_ids=tuple(f"candidate-{j}" for j in range(100)),
                       target_item_id="candidate-0") for i, user in enumerate(users("dev", 5))]
        model = CapacityTiny().train()
        cfg = config()
        optimizer = trainer.build_optimizer(model, cfg)
        trainer.train_update(model, [make_query(users("train", 1)[0], "train", 0)], {}, optimizer,
                             microbatch_queries=1, clip=10, device="cpu", compute_dtype="float32")
        return model, optimizer, dev, cfg

    def capacity_assessment(self, model, cfg):
        def assess(queries, expected_users, name):
            self.assertEqual(name, "capacity-dev")
            return trainer.evaluate(model, queries, {}, expected_users=expected_users, device="cpu",
                compute_dtype="float32", max_reader_packed_tokens=1000,
                microbatch_queries=cfg["evaluation"].get("microbatch_queries", cfg["training"]["microbatch_queries"]))
        return assess

    def test_capacity_evaluation_executes_one_actual_batch4_with_resident_adam_and_gradients(self):
        model, optimizer, dev, cfg = self.capacity_fixture()
        cfg["evaluation"]["microbatch_queries"] = 4
        before_parameters = copy.deepcopy(model.state_dict())
        before_gradients = {parameter: parameter.grad for parameter in model.parameters()}
        before_adam = {parameter: {key: value.clone() for key, value in state.items()}
                       for parameter, state in optimizer.state.items()}
        result = trainer._capacity_evaluation(model, optimizer, dev, cfg, self.capacity_assessment(model, cfg))
        self.assertEqual(model.evaluation_batches, [[query.query_id for query in dev[:4]]])
        self.assertEqual(result["queries"], 4)
        self.assertEqual(result["microbatch_queries"], 4)
        self.assertEqual(result["compute_dtype"], "float32")
        self.assertEqual(result["candidates_per_query"], 100)
        self.assertIs(result["finite_scores"], True)
        self.assertIs(result["optimizer_state_initialized"], True)
        self.assertEqual(result["optimizer_state_entries"], len(optimizer.state))
        self.assertIs(result["gradients_retained"], True)
        self.assertEqual(result["retained_gradient_buffers"], len(before_gradients))
        self.assertEqual(result["retained_gradient_bytes"], sum(g.numel() * g.element_size() for g in before_gradients.values()))
        self.assertTrue(model.training)
        for name, parameter in model.state_dict().items():
            torch.testing.assert_close(parameter, before_parameters[name], atol=0, rtol=0)
        for parameter, state in optimizer.state.items():
            self.assertIs(parameter.grad, before_gradients[parameter])
            for key, value in state.items():
                torch.testing.assert_close(value, before_adam[parameter][key], atol=0, rtol=0)

    def test_capacity_evaluation_without_override_keeps_first_single_query(self):
        model, optimizer, dev, cfg = self.capacity_fixture()
        cfg["training"]["microbatch_queries"] = 8
        result = trainer._capacity_evaluation(model, optimizer, dev, cfg, self.capacity_assessment(model, cfg))
        self.assertEqual(model.evaluation_batches, [[dev[0].query_id]])
        self.assertEqual(result["queries"], 1)
        self.assertEqual(result["microbatch_queries"], 1)
        self.assertEqual(result["candidates"], 100)

    def test_capacity_evaluation_fails_closed_for_missing_or_invalid_dev_rows(self):
        model, optimizer, dev, cfg = self.capacity_fixture()
        cfg["evaluation"]["microbatch_queries"] = 4
        invalid = [[], dev[:3], [dev[0], dev[0], *dev[2:4]],
                   [replace(dev[0], candidate_ids=dev[0].candidate_ids[:8]), *dev[1:]],
                   [SimpleNamespace(query_id=dev[0].query_id, user_id=dev[0].user_id,
                                    candidate_ids=("repeat",) * 100), *dev[1:]],
                   [dev[0], replace(dev[1], user_id=dev[0].user_id), *dev[2:]]]
        for sample in invalid:
            with self.subTest(rows=len(sample)), patch.object(trainer, "evaluate") as evaluation:
                with self.assertRaisesRegex(ValueError, "capacity evaluation"):
                    trainer._capacity_evaluation(model, optimizer, sample, cfg, self.capacity_assessment(model, cfg))
                evaluation.assert_not_called()

    def test_capacity_evaluation_rejects_missing_adam_gradients_and_nonfinite_scores(self):
        for reason in ("adam", "gradients", "scores", "discard_gradients", "discard_adam"):
            model, optimizer, dev, cfg = self.capacity_fixture()
            cfg["evaluation"]["microbatch_queries"] = 4
            assessment = self.capacity_assessment(model, cfg)
            if reason == "adam":
                optimizer.state.clear()
            elif reason == "gradients":
                optimizer.zero_grad(set_to_none=True)
            elif reason == "scores":
                model.invalid_scores = True
            else:
                original = assessment
                def assessment(queries, users, name, original=original, reason=reason):
                    measured = original(queries, users, name)
                    if reason == "discard_gradients":
                        optimizer.zero_grad(set_to_none=True)
                    else:
                        optimizer.state.clear()
                    return measured
            with self.subTest(reason=reason), self.assertRaises(ValueError):
                trainer._capacity_evaluation(model, optimizer, dev, cfg, assessment)

    def test_acceptance_branch_returns_actual_batch4_evidence_and_preserves_single_query_control(self):
        for evaluation_micro in (None, 4):
            model, _, dev, cfg = self.capacity_fixture()
            if evaluation_micro is not None:
                cfg["evaluation"]["microbatch_queries"] = evaluation_micro
            with self.subTest(evaluation_micro=evaluation_micro), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                dataset(root / "data")
                train, _, items, cohorts, hashes = trainer.load_dataset(root / "data", cfg, formal=False)
                cohorts["dev"] = [query.user_id for query in dev]
                # Replace only hardware/loading guards. Real CPU Adam updates,
                # assessment, serialization and acceptance branch still run.
                with patch.object(trainer, "validate_config", side_effect=lambda c, **_: None), \
                     patch.object(trainer, "load_dataset", return_value=(train, dev, items, cohorts, hashes)), \
                     patch.object(trainer.GroupRanker, "from_pretrained", return_value=model), \
                     patch.object(torch.cuda, "is_available", return_value=False), \
                     patch.object(torch.cuda, "max_memory_allocated", return_value=1234):
                    result = trainer.run_training(cfg, root / "data", root / "run", device="cuda:0", formal=True,
                                                  max_updates=1, acceptance_only=True)
                actual_micro = evaluation_micro or 1
                self.assertEqual(result["evaluation"]["queries"], actual_micro)
                self.assertEqual(result["evaluation"]["microbatch_queries"], actual_micro)
                self.assertIs(result["evaluation"]["gradients_retained"], True)
                self.assertEqual(result["peak_memory_bytes"], 1234)
                rows = [json.loads(line) for line in (root / "run/predictions/capacity-dev.jsonl").read_text().splitlines()]
                self.assertEqual([row["query_id"] for row in rows], [query.query_id for query in dev[:actual_micro]])
                self.assertTrue(all(len(row["scores"]) == 100 for row in rows))
                self.assertEqual(model.evaluation_batches[-1], [query.query_id for query in dev[:actual_micro]])

    def test_real_qwen_eval_batch4_tail2_preserves_query_predictions_and_cohort_denominator(self):
        from tests.test_core_compact_next_item_model import fixture, student
        from duolm.next_item_data import HistoryEntry
        query, items = fixture()
        cohort = users("dev", 7)
        queries = []
        for index, user in enumerate(cohort[:6]):
            candidates = query.candidate_ids[index % 3:] + query.candidate_ids[:index % 3]
            positive = index % 3 if index != 5 else None
            queries.append(replace(query, query_id=f"eval-{index}", user_id=user, split="dev",
                candidate_ids=candidates, positive_index=positive,
                target_item_id=candidates[positive] if positive is not None else "retrieval-miss",
                history=(HistoryEntry("h0", 1 + index % 5, 1), HistoryEntry("h1", 5 - index % 5, 2))))
        for readout in ("memory", "perceiver"):
            with self.subTest(readout=readout):
                model = student(a_readout=readout, resampler_layers=1, resampler_heads=8,
                                input_assembly="batched", max_reader_packed_tokens=8192).train()
                model.calibrate_bridge(queries, items)
                sequential, grouped = [], []
                common = dict(expected_users=cohort, device="cpu", compute_dtype="float32",
                              max_reader_packed_tokens=8192, unavailable_users=[cohort[-1]],
                              evaluation_scope=trainer.CANDIDATE_PROTOCOL)
                first = trainer.evaluate(model, queries, items, microbatch_queries=1,
                                         prediction_sink=sequential.append, **common)
                with patch.object(model, "forward", wraps=model.forward) as forward:
                    second = trainer.evaluate(model, queries, items, microbatch_queries=4,
                                              prediction_sink=grouped.append, **common)
                self.assertEqual([len(call.args[0]) for call in forward.call_args_list], [4, 2])
                self.assertTrue(model.training)
                self.assertEqual(first, second)
                self.assertEqual(second["users"], 7)
                self.assertEqual(second["queries"], 6)
                self.assertEqual(second["unavailable_users"], 1)
                self.assertAlmostEqual(second["candidate_target_coverage"], 5 / 7)
                for expected, actual, prepared in zip(sequential, grouped, queries):
                    self.assertEqual(actual["query_id"], prepared.query_id)
                    self.assertEqual(actual["candidate_ids"], list(prepared.candidate_ids))
                    for key in expected.keys() - {"scores"}:
                        self.assertEqual(actual[key], expected[key])
                    torch.testing.assert_close(actual["scores"], expected["scores"], atol=3e-6, rtol=3e-4)
                # Only a declared missing user can contribute a zero row.
                with self.assertRaisesRegex(ValueError, "cohort is incomplete"):
                    trainer.evaluate(model, queries, items, expected_users=cohort, device="cpu",
                        compute_dtype="float32", max_reader_packed_tokens=8192, microbatch_queries=4)
                self.assertTrue(model.training)

    def test_eval_batch_ties_and_missing_users_have_identical_fixed_user_metrics(self):
        class Ties(TinyRanker):
            def forward(self, queries, items, **kwargs):
                return {"scores": [torch.zeros(len(q.candidate_ids)) for q in queries],
                        "statistics": {"reader_tokens": 3 * len(queries)}}
        cohort = users("dev", 7)
        queries = [replace(make_query(user, "dev", i), candidate_ids=("z", "a"),
                           positive_index=i % 2, target_item_id=("z", "a")[i % 2])
                   for i, user in enumerate(cohort[:6])]
        rows = []
        common = dict(expected_users=cohort, unavailable_users=[cohort[-1]], device="cpu",
                      compute_dtype="float32", max_reader_packed_tokens=1000)
        first = trainer.evaluate(Ties(), queries, {}, **common)
        second = trainer.evaluate(Ties(), queries, {}, microbatch_queries=4, prediction_sink=rows.append, **common)
        self.assertEqual(first, second)
        self.assertEqual([row["rank"] for row in rows], [2, 1, 2, 1, 2, 1])
        self.assertAlmostEqual(second["hr@1"], 3 / 7)

    def test_assessment_explicit_eval_batch_is_independent_and_missing_field_preserves_legacy(self):
        class Record(TinyRanker):
            def __init__(self):
                super().__init__()
                self.eval_batch_sizes = []
            def forward(self, queries, items, **kwargs):
                if not self.training:
                    self.eval_batch_sizes.append(len(queries))
                return super().forward(queries, items, **kwargs)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset(root / "data", unavailable=True)
            reports, models = [], []
            for label, evaluation_micro in (("legacy", None), ("single", 1), ("batch4", 4)):
                cfg = config()
                if evaluation_micro is not None:
                    cfg["evaluation"]["microbatch_queries"] = evaluation_micro
                model = Record()
                reports.append(trainer.run_training(cfg, root / "data", root / label, device="cpu", formal=False,
                                                    model_factory=lambda *_, model=model: model))
                models.append(model)
            # Old training.microbatch=2 continues to define assessment default.
            self.assertIn(2, models[0].eval_batch_sizes)
            self.assertTrue(all(size == 1 for size in models[1].eval_batch_sizes))
            self.assertTrue(all(size <= 4 for size in models[2].eval_batch_sizes))
            for report in reports[1:]:
                self.assertEqual(report["state"], reports[0]["state"])
                self.assertEqual(report["best_step"], 1)  # Tied selector scores retain first best.
                self.assertEqual(report["selected_by"], reports[0]["selected_by"])
                self.assertEqual(report["best_full_dev"], reports[0]["best_full_dev"])
                self.assertEqual(report["final_full_dev"], reports[0]["final_full_dev"])
            for name in ("step0-monitor", "final-dev", "best-dev"):
                expected = (root / "single/predictions" / (name + ".jsonl")).read_text()
                self.assertEqual((root / "batch4/predictions" / (name + ".jsonl")).read_text(), expected)
                manifest = json.loads((root / "batch4/predictions" / (name + ".manifest.json")).read_text())
                self.assertEqual(manifest["evaluation_microbatch_queries"], 4)
                self.assertEqual(manifest["metrics"]["users"], 1 if name == "step0-monitor" else 3)
                if name != "step0-monitor":
                    self.assertEqual(manifest["metrics"]["unavailable_users"], 1)
            class LegacySingle(Record):
                def forward(self, queries, items, **kwargs):
                    if not self.training:
                        assert len(queries) == 1, "legacy single-query evaluator must stay compatible"
                    return super().forward(queries, items, **kwargs)
            legacy_cfg = config()
            legacy_cfg["training"]["microbatch_queries"] = 1
            trainer.run_training(legacy_cfg, root / "data", root / "legacy-single", device="cpu", formal=False,
                                 model_factory=lambda *_: LegacySingle())

    def test_eval_microbatch_replay_is_identity_bound_and_same_setting_resumes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset(root / "data")
            cfg = config()
            cfg["evaluation"]["microbatch_queries"] = 4
            first = trainer.run_training(cfg, root / "data", root / "run", device="cpu", formal=False,
                model_factory=lambda *_: TinyRanker(), max_updates=1)
            changed = copy.deepcopy(cfg)
            changed["evaluation"]["microbatch_queries"] = 1
            with patch.object(trainer, "train_update") as update:
                with self.assertRaisesRegex(ValueError, "identity differs"):
                    trainer.run_training(changed, root / "data", root / "run", device="cpu", formal=False,
                        model_factory=lambda *_: TinyRanker(), resume=first["checkpoint"])
                update.assert_not_called()
            full = trainer.run_training(cfg, root / "data", root / "full", device="cpu", formal=False,
                                        model_factory=lambda *_: TinyRanker())
            resumed = trainer.run_training(cfg, root / "data", root / "run", device="cpu", formal=False,
                model_factory=lambda *_: TinyRanker(), resume=first["checkpoint"])
            self.assertEqual(resumed["state"], full["state"])
            self.assertEqual(resumed["best_full_dev"], full["best_full_dev"])
            self.assertEqual(resumed["final_full_dev"], full["final_full_dev"])

    def test_staging_fixed_mapping_shuffle_padding_and_no_per_query_copy(self):
        qs = [SimpleNamespace(query_id="short", candidate_ids=("a", "b")),
              SimpleNamespace(query_id="long", candidate_ids=("c", "b", "a"))]
        staged = trainer.DeviceTeacherScores({"long": [3, 4, 5], "short": [1, 2]}, qs, device="cpu")
        self.assertEqual(staged.rows.shape, (2, 3))
        self.assertEqual(staged.rows.dtype, torch.float32)
        with patch.object(torch.Tensor, "to", side_effect=AssertionError("batch selection must not transfer scores")):
            selected = staged.for_batch([qs[1], qs[0]])
        self.assertEqual([row.tolist() for row in selected], [[3, 4, 5], [1, 2]])
        self.assertEqual(selected[0].untyped_storage().data_ptr(), staged.rows.untyped_storage().data_ptr())
        for bad_scores in ({"short": [1, 2]}, {"short": [1, 2], "long": [3, 4, 5], "extra": [1, 2]},
                           {"short": [1, 2], "long": [3, 4]}, {"short": [1, float("nan")], "long": [3, 4, 5]},
                           {"short": [1, 1e99], "long": [3, 4, 5]}):
            with self.assertRaises(ValueError):
                trainer.DeviceTeacherScores(bad_scores, qs, device="cpu")
        with self.assertRaisesRegex(ValueError, "unique"):
            trainer.DeviceTeacherScores({"short": [1, 2]}, [qs[0], qs[0]], device="cpu")
        for query in (SimpleNamespace(query_id="missing", candidate_ids=("a", "b")),
                      SimpleNamespace(query_id="short", candidate_ids=("b", "a"))):
            with self.assertRaisesRegex(ValueError, "mapping"):
                staged.for_batch([query])

    def test_deferred_uneven_microbatches_tail_and_one_metrics_readback(self):
        qs = [SimpleNamespace(query_id=f"q-{i}", positive_index=i % 2, candidate_ids=("a", "b")) for i in range(32)]
        cache = {q.query_id: [i * .02, -.3] for i, q in enumerate(qs)}
        staged = trainer.DeviceTeacherScores(cache, qs, device="cpu")
        for count, micro in ((32, 8), (5, 2), (2, 8)):
            with self.subTest(count=count, micro=micro):
                whole, split = DeferredTinyRanker(), DeferredTinyRanker()
                opt_a, opt_b = trainer.build_optimizer(whole, config()), trainer.build_optimizer(split, config())
                expected = trainer.train_update(whole, qs[:count], {}, opt_a,
                    microbatch_queries=count, clip=100, device="cpu", compute_dtype="float32",
                    teacher_scores=cache, beta=.5, temperature=4)
                original_cpu, calls = torch.Tensor.cpu, []
                def cpu(tensor, *args, **kwargs):
                    calls.append(tuple(tensor.shape))
                    return original_cpu(tensor, *args, **kwargs)
                with patch.object(torch.Tensor, "cpu", cpu):
                    actual = trainer.train_update(split, qs[:count], {}, opt_b,
                        microbatch_queries=micro, clip=100, device="cpu", compute_dtype="float32",
                        teacher_scores=staged, beta=.5, temperature=4, defer_scalar_metrics=True)
                self.assertEqual(calls, [(4,)])
                for key in ("ce", "kd", "loss", "grad_norm"):
                    self.assertAlmostEqual(expected[key], actual[key], places=6)
                self.assertEqual(actual["queries"], count)
                for first, second in zip(whole.parameters(), split.parameters()):
                    torch.testing.assert_close(first, second, atol=1e-7, rtol=1e-6)
                    torch.testing.assert_close(opt_a.state[first]["exp_avg"], opt_b.state[second]["exp_avg"],
                                               atol=1e-7, rtol=1e-6)

    def test_nonfinite_loss_flag_or_gradients_never_apply_adam(self):
        qs = [SimpleNamespace(query_id=f"q-{i}", positive_index=0, candidate_ids=("a", "b")) for i in range(3)]
        for reason in ("loss", "flag", "gradient"):
            class Bad(DeferredTinyRanker):
                def loss(self, queries, items, **kwargs):
                    output = super().loss(queries, items, **kwargs)
                    if queries[0].query_id == "q-2":
                        if reason == "loss":
                            output["loss"] = output["loss"] * float("nan")
                        elif reason == "flag":
                            output["loss_finite"] = torch.tensor(False)
                    return output
            model = Bad()
            if reason == "gradient":
                model.backbone.register_hook(lambda grad: grad * float("inf"))
            optimizer = trainer.build_optimizer(model, config())
            before = copy.deepcopy(model.state_dict())
            with patch.object(optimizer, "step", wraps=optimizer.step) as step:
                with self.assertRaisesRegex(FloatingPointError, "not applied"):
                    trainer.train_update(model, qs, {}, optimizer, microbatch_queries=2, clip=10,
                                         device="cpu", compute_dtype="float32", defer_scalar_metrics=True)
                step.assert_not_called()
            self.assertFalse(optimizer.state)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, before[name], atol=0, rtol=0)
            self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_token_statistics_are_microbatch_sums_without_tensor_readback(self):
        class Counts(DeferredTinyRanker):
            def forward(self, queries, items, **kwargs):
                output = super().forward(queries, items, **kwargs)
                output["statistics"].update(encoder_tokens=12, encoder_unique_items=2, reader_packs=1)
                return output
        model = Counts()
        queries = [SimpleNamespace(query_id=f"q-{i}", positive_index=0, candidate_ids=("a", "b")) for i in range(5)]
        result = trainer.train_update(model, queries, {}, trainer.build_optimizer(model, config()),
            microbatch_queries=2, clip=10, device="cpu", compute_dtype="float32", defer_scalar_metrics=True)
        self.assertEqual(result["encoder_tokens"], 36)
        self.assertEqual(result["encoder_unique_items"], 6)
        self.assertEqual(result["reader_packs"], 3)
        self.assertEqual(result["encoder_unique_items_scope"], "sum_of_microbatch_unique_items")
        timer = trainer._PhaseTimer("cpu", cuda_events=True)
        measured = timer.finish()
        self.assertGreaterEqual(measured["wall_seconds"], 0)
        self.assertNotIn("cuda_event_elapsed_seconds", measured)

    def test_real_tiny_qwen_ce_kd_adam_updates_match_legacy_for_all_m1_readouts(self):
        from tests.test_core_compact_next_item_model import fixture, student
        query, items = fixture()
        queries = [replace(query, query_id=f"query-{i}", positive_index=i % 3,
                           target_item_id=query.candidate_ids[i % 3]) for i in range(5)]
        cache = {q.query_id: [.7 + i * .1, -.2, .1] for i, q in enumerate(queries)}
        staged = trainer.DeviceTeacherScores(cache, queries, device="cpu")
        cfg = config()
        cfg["training"].update(backbone_lr=1e-5, head_lr=1e-4, interface_lr=1e-4)
        for readout in ("last_token", "memory", "perceiver"):
            with self.subTest(readout=readout):
                legacy = student(a_readout=readout, resampler_layers=1, resampler_heads=8,
                                 max_encoder_packed_tokens=1024, gradient_checkpointing=True).train()
                legacy.calibrate_bridge(queries, items)
                fast = copy.deepcopy(legacy)
                fast.input_assembly = "batched"
                fast.reader_gradient_checkpointing = False
                fast.b.gradient_checkpointing_disable()
                fast.max_encoder_packed_tokens = 32768
                fast.prepare_item_tokens(items)
                opt_a, opt_b = trainer.build_optimizer(legacy, cfg), trainer.build_optimizer(fast, cfg)
                for selected in (queries, queries[-2:]):
                    a = trainer.train_update(legacy, selected, items, opt_a, microbatch_queries=1,
                        clip=1, device="cpu", compute_dtype="float32", teacher_scores=cache, beta=.5, temperature=4)
                    b = trainer.train_update(fast, selected, items, opt_b, microbatch_queries=3,
                        clip=1, device="cpu", compute_dtype="float32", teacher_scores=staged,
                        beta=.5, temperature=4, defer_scalar_metrics=True)
                    for key in ("ce", "kd", "loss", "grad_norm"):
                        torch.testing.assert_close(a[key], b[key], atol=3e-6, rtol=3e-4)
                    for (name, left), (_, right) in zip(legacy.named_parameters(), fast.named_parameters()):
                        torch.testing.assert_close(left.grad, right.grad, atol=3e-6, rtol=3e-4, msg=name)
                        if name == "resampler.layers.0.attention.in_proj_bias":
                            # K bias shifts every key's logit by one common
                            # constant, so its true derivative is zero. FP32
                            # accumulation noise (~1e-9) is amplified by Adam's
                            # frozen eps=1e-8. Its raw value is a free coordinate,
                            # so compare behavior under a K-bias shift below,
                            # while keeping learned Q/V bias comparisons strict.
                            width = left.numel() // 3
                            for bias in (left, right):
                                self.assertLess(bias.grad[width:2*width].abs().max().item(), 3e-9)
                            torch.testing.assert_close(torch.cat((left[:width], left[2*width:])),
                                                       torch.cat((right[:width], right[2*width:])), atol=2e-7, rtol=1e-5)
                        else:
                            torch.testing.assert_close(left, right, atol=2e-7, rtol=1e-5, msg=name)
                        for key in ("exp_avg", "exp_avg_sq", "step"):
                            torch.testing.assert_close(opt_a.state[left][key], opt_b.state[right][key],
                                                       atol=3e-7, rtol=3e-4, msg=name + ":" + key)
                    with torch.no_grad():
                        first, second = legacy(selected, items), fast(selected, items)
                    for left, right in zip(first["scores"], second["scores"]):
                        torch.testing.assert_close(left, right, atol=3e-6, rtol=3e-4)
                    if readout == "perceiver":
                        # A sizable nonuniform shift exercises K-bias invariance,
                        # rather than only this platform's tiny Adam drift.
                        # Restore it before the next real optimizer update.
                        with torch.no_grad():
                            for model, reference in ((legacy, first), (fast, second)):
                                bias = model.resampler.layers[0].attention.in_proj_bias
                                width = bias.numel() // 3
                                key_bias = bias[width:2*width]
                                original = key_bias.clone()
                                try:
                                    key_bias.add_(torch.linspace(-.25, .25, width, dtype=bias.dtype, device=bias.device))
                                    shifted = model(selected, items)
                                finally:
                                    key_bias.copy_(original)
                                for expected, actual in zip(reference["scores"], shifted["scores"]):
                                    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-4)

    def test_real_fast_training_stages_and_pretokenizes_once_per_start_then_resumes(self):
        from duolm.next_item_model import query_input_hash
        from tests.test_core_compact_next_item_model import student
        with tempfile.TemporaryDirectory() as folder:
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
            cfg["model"].update(variant="V3", a_model={"repo": "fixture", "revision": "0"*40},
                input_assembly="batched", a_readout="memory", m=1, resampler_layers=1,
                resampler_heads=8, score_readout="candidate", shared_candidate_header=True)
            cfg["training"].update(reader_gradient_checkpointing=False, optimizer_fused=False,
                teacher_cache_on_device=True, pretokenize_items=True, max_encoder_packed_tokens=32768)
            cfg["loss"].update(recipe="ce_kd", teacher_cache=str(root / "teacher.json"), teacher_identity=teacher)
            models = []
            def factory(options, device):
                model = student(a_readout="memory", input_assembly="batched", reader_gradient_checkpointing=False,
                    gradient_checkpointing=True, max_encoder_packed_tokens=32768).to(device)
                model.prepare_item_tokens = unittest.mock.Mock(wraps=model.prepare_item_tokens)
                models.append(model)
                return model
            full = trainer.run_training(cfg, root / "data", root / "full", device="cpu", formal=False,
                                        model_factory=factory)
            partial = trainer.run_training(cfg, root / "data", root / "resume", device="cpu", formal=False,
                                           model_factory=factory, max_updates=1)
            resumed = trainer.run_training(cfg, root / "data", root / "resume", device="cpu", formal=False,
                                           model_factory=factory, resume=partial["checkpoint"])
            self.assertEqual(resumed["state"], full["state"])
            for model in models:
                model.prepare_item_tokens.assert_called_once_with(items)
            for path in (root / "full", root / "resume"):
                events = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]
                staged = [row for row in events if row["event"] == "teacher_cache_staged"]
                prepared = [row for row in events if row["event"] == "item_tokens_prepared"]
                self.assertEqual(len(staged), 1 if path.name == "full" else 2)
                self.assertEqual(len(prepared), len(staged))
                self.assertTrue(all(row["storage_shape"] == [5, 2] for row in staged))
                self.assertTrue(all(row["statistics"]["cache_values"] == "CPU token IDs only" for row in prepared))
            first = torch.load(Path(full["final_checkpoint"]) / "training.pt", weights_only=True)
            second = torch.load(Path(resumed["final_checkpoint"]) / "training.pt", weights_only=True)
            for name, value in first["model"].items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(value, second["model"][name], atol=0, rtol=0)
                else:
                    self.assertEqual(value, second["model"][name])
            self.assertEqual(first["optimizer"]["param_groups"], second["optimizer"]["param_groups"])
            self.assertEqual(first["optimizer"]["state"].keys(), second["optimizer"]["state"].keys())
            for parameter, state in first["optimizer"]["state"].items():
                for key, value in state.items():
                    torch.testing.assert_close(value, second["optimizer"]["state"][parameter][key], atol=0, rtol=0)
            torch.testing.assert_close(first["rng_torch"], second["rng_torch"], atol=0, rtol=0)
            self.assertEqual(first["rng_python"], second["rng_python"])








if __name__ == "__main__":
    unittest.main()
