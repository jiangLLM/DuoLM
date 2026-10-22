"""Offline input-assembly proofs with real, tiny Qwen3 backbones.

CPU is the default. From this repository, a CUDA-equipped Python environment can
run the additional FP32/BF16 gate with:

    MIXLM_FASTPATH_CUDA=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \\
        python -B -m unittest tests.test_core_next_item_fastpath -v

The CUDA gate uses the same SDPA MATH backend on both paths, with FP32 master
weights. It does not replace the separate production FlexAttention BF16 gate or
the required real 0.6B/H100 training and throughput benchmark.
"""

import copy
from dataclasses import replace
import json
import os
import unittest
from unittest.mock import patch

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import Qwen3Config, Qwen3Model

from duolm.next_item_data import HistoryEntry, RecommendationQuery, user_split
from duolm.next_item_model import (
    GroupRanker, V1_UNKNOWN_FEEDBACK_SEGMENTS, group_ranking_loss,
)


class TinyTokenizer:
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [2 + ord(character) % 250 for character in text]


class FeedbackTokenizer(TinyTokenizer):
    """Retain the numeric/unknown feedback merge asymmetry of Qwen's BPE."""

    before = [10, 11, 12, 13, 14]
    after = [10]

    def encode(self, text, add_special_tokens=False):
        if text == V1_UNKNOWN_FEEDBACK_SEGMENTS[0]:
            return self.before.copy()
        if text == "?":
            return [20]
        if text == "\n":
            return self.after.copy()
        if len(text) == 1 and text in "12345":
            return [14 + int(text)]
        if text == "\nUser rating: ?\n":
            return self.before[:-1] + [21]
        for rating in range(1, 6):
            if text == f"\nUser rating: {rating}\n":
                return self.before + [14 + rating] + self.after
        return super().encode(text, add_special_tokens=add_special_tokens)


def tiny_qwen(width):
    config = Qwen3Config(
        vocab_size=256, hidden_size=width, intermediate_size=width * 2,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=width // 2, max_position_embeddings=2048,
        attention_dropout=0.0, use_cache=False,
    )
    config._attn_implementation = "sdpa"
    return Qwen3Model(config)


def batch_fixture():
    user = next(f"fastpath-{i}" for i in range(100) if user_split(f"fastpath-{i}") == "train")

    def query(name, history, candidates):
        events = tuple(HistoryEntry(item, rating, index + 1)
                       for index, (item, rating) in enumerate(history))
        return RecommendationQuery(
            name, user, "train", events, tuple(candidates), candidates.index("c0"),
            "c0", len(events) + 1, tuple(dict.fromkeys(event.item_id for event in events)),
        )

    queries = [
        query("three-candidates", (("h0", 1), ("h1", 5)), ("c0", "c1", "c2")),
        query("reordered-two", (("h0", 5), ("h1", 2)), ("c2", "c0")),
        query("repeated-book-four", (("h0", 3), ("h0", 1), ("h2", 4)),
              ("c3", "c1", "c0", "c2")),
    ]
    # Two identities intentionally have identical canonical content. A dedup is
    # by complete token sequence, including the rating channel for V1.
    items = {"h0": "history alpha", "h1": "longer history beta omega",
             "h2": "third history gamma", "c0": "candidate target theta",
             "c1": "different candidate kappa", "c2": "short zeta",
             "c3": "different candidate kappa"}
    return queries, items


def changed_batch(queries):
    changed = []
    orders = (("c2", "c0"), ("c1", "c0", "c2"), ("c2", "c0", "c1", "c3"))
    for index, (query, order) in enumerate(zip(queries, orders)):
        history = tuple(replace(event, rating=(event.rating + index) % 5 + 1)
                        for event in query.history)
        changed.append(replace(query, query_id=f"changed-{index}", history=history,
                               candidate_ids=order, positive_index=order.index("c0")))
    return changed


def ranker(variant="V3", readout="last_token", *, backbone_width_a=16, backbone_width_b=24, **kwargs):
    torch.manual_seed(703)
    options = dict(
        variant=variant, m=1, a_readout=readout, resampler_layers=2,
        resampler_heads=2, shared_candidate_header=True, score_readout="candidate",
        max_encoder_packed_tokens=2048, max_reader_packed_tokens=2048,
    )
    options.update(kwargs)
    # Different A/B widths make the learned projection part of every proof.
    return GroupRanker(tiny_qwen(backbone_width_a), tiny_qwen(backbone_width_b),
                       TinyTokenizer(), FeedbackTokenizer(), **options)


class NoHostScalarReads(TorchDispatchMode):
    """Catch Tensor.item()/Python bool conversions inside the deferred loss."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten._local_scalar_dense.default:
            raise AssertionError("deferred loss read a tensor scalar on the host")
        return func(*args, **(kwargs or {}))


def assert_groupwise_deferred_arithmetic(test, device):
    """Real loss kernels: ragged and effective32 cohorts, including staging."""
    for lengths in ((2, 4, 3), (8,) * 32):
        positives = [index % count for index, count in enumerate(lengths)]
        rows = [torch.linspace(-.7, .9, count, device=device) + index * .01
                for index, count in enumerate(lengths)]
        teacher_rows = [torch.linspace(.4, -.5, count, device=device) + index * .02
                        for index, count in enumerate(lengths)]
        kinds = ("lists", "device_rows", "dense") if len(set(lengths)) == 1 else ("lists", "device_rows")
        for beta in (0.0, .35, 1.0):
            for kind in kinds:
                with test.subTest(device=device, cohort=len(rows), beta=beta, teacher_kind=kind):
                    expected_scores = [row.detach().clone().requires_grad_() for row in rows]
                    actual_scores = [row.detach().clone().requires_grad_() for row in rows]
                    teachers = ([row.tolist() for row in teacher_rows] if kind == "lists" else
                                torch.stack(teacher_rows).requires_grad_() if kind == "dense" else
                                [row.detach().clone().requires_grad_() for row in teacher_rows])
                    expected = group_ranking_loss(expected_scores, positives, teacher_scores=teachers,
                                                  beta=beta, temperature=1.7)
                    original_ce, original_kl = torch.nn.functional.cross_entropy, torch.nn.functional.kl_div
                    with patch("duolm.next_item_model.F.cross_entropy", wraps=original_ce) as ce, \
                         patch("duolm.next_item_model.F.kl_div", wraps=original_kl) as kl, \
                         patch("duolm.next_item_model.torch.tensor", wraps=torch.tensor) as constructors:
                        with NoHostScalarReads():
                            actual = group_ranking_loss(actual_scores, positives, teacher_scores=teachers,
                                                        beta=beta, temperature=1.7, defer_scalar_metrics=True)
                            actual["loss"].backward()
                        test.assertEqual([call.args[0].shape for call in ce.call_args_list],
                                         [torch.Size((1, count)) for count in lengths])
                        targets = [call.args[1] for call in ce.call_args_list]
                        test.assertEqual(len({row.untyped_storage().data_ptr() for row in targets}), 1)
                        test.assertEqual([row.storage_offset() for row in targets], list(range(len(rows))))
                        test.assertEqual(constructors.call_count, 1 + int(bool(beta) and kind == "lists"))
                        test.assertEqual(constructors.call_args_list[0].kwargs["dtype"], torch.long)
                        test.assertEqual([call.args[0].shape for call in kl.call_args_list],
                                         [torch.Size((count,)) for count in lengths] if beta else [])
                        test.assertTrue(all(call.kwargs["reduction"] == "sum" for call in kl.call_args_list))
                    # Exact equality is appropriate here: preserve kernel shape,
                    # per-query temperature scaling and stack-then-mean order.
                    torch.testing.assert_close(expected["loss"], actual["loss"], atol=0, rtol=0)
                    test.assertTrue(actual["loss_finite"].item())
                    for metric in ("ce", "kd"):
                        test.assertFalse(actual[metric].requires_grad)
                        torch.testing.assert_close(actual[metric], actual[metric].new_tensor(expected[metric]),
                                                   atol=0, rtol=0)
                    expected["loss"].backward()
                    for left, right in zip(expected_scores, actual_scores):
                        torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)
                    if kind == "device_rows":
                        test.assertTrue(all(row.grad is None for row in teachers))
                    elif kind == "dense":
                        test.assertIsNone(teachers.grad)


class NextItemFastpathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assert_outputs_equal(self, expected, actual):
        self.assertEqual(expected["statistics"], actual["statistics"])
        self.assertEqual(expected["logical_lengths"], actual["logical_lengths"])
        self.assertEqual(len(expected["scores"]), len(actual["scores"]))
        for index, (left, right) in enumerate(zip(expected["scores"], actual["scores"])):
            torch.testing.assert_close(left, right, atol=2e-6, rtol=1e-5, msg=f"scores group {index}")
        torch.testing.assert_close(expected["loss"], actual["loss"], atol=1e-6, rtol=1e-6)
        for metric in ("ce", "kd"):
            self.assertIsInstance(expected[metric], float)
            self.assertIsInstance(actual[metric], torch.Tensor)
            self.assertEqual(actual[metric].shape, torch.Size([]))
            self.assertFalse(actual[metric].requires_grad)
            torch.testing.assert_close(torch.tensor(expected[metric]), actual[metric], atol=1e-6, rtol=1e-6)
        self.assertIsInstance(actual["loss_finite"], torch.Tensor)
        self.assertEqual(actual["loss_finite"].shape, torch.Size([]))
        self.assertEqual(actual["loss_finite"].dtype, torch.bool)
        self.assertEqual(actual["loss_finite"].device, actual["loss"].device)
        self.assertTrue(actual["loss_finite"].item())

    def assert_gradients_equal(self, expected, actual):
        left, right = dict(expected.named_parameters()), dict(actual.named_parameters())
        self.assertEqual(left.keys(), right.keys())
        for name in left:
            self.assertIsNotNone(left[name].grad, name)
            self.assertIsNotNone(right[name].grad, name)
            self.assertTrue(torch.isfinite(left[name].grad).all(), name)
            self.assertTrue(torch.isfinite(right[name].grad).all(), name)
            torch.testing.assert_close(left[name].grad, right[name].grad,
                                       atol=3e-6, rtol=3e-4, msg=f"gradient {name}")
        for name in ("a.embed_tokens.weight", "b.embed_tokens.weight", "projection.weight",
                     "head.weight", "bridge.log_gain"):
            if name in left:
                self.assertGreater(left[name].grad.abs().sum().item(), 0, name)

    def assert_parameters_equal(self, expected, actual, *, atol=3e-6, rtol=3e-4):
        for (name, left), (other, right) in zip(expected.named_parameters(), actual.named_parameters()):
            self.assertEqual(name, other)
            torch.testing.assert_close(left, right, atol=atol, rtol=rtol, msg=f"Adam update {name}")

    def assert_adam_states_equal(self, expected_model, actual_model, expected_optimizer, actual_optimizer):
        for (name, left), (_, right) in zip(expected_model.named_parameters(), actual_model.named_parameters()):
            expected = expected_optimizer.state[left]
            actual = actual_optimizer.state[right]
            self.assertEqual(expected.keys(), actual.keys(), name)
            for key in expected:
                # Moments inherit FP32 gradient reduction roundoff. The first
                # moment needs a small floor at cancellation near zero.
                tolerance = 1e-7 if key == "exp_avg" else 1e-9
                torch.testing.assert_close(expected[key], actual[key], atol=tolerance, rtol=3e-4,
                                           msg=f"Adam {key} for {name}")

    def assert_default_adamw_parameters_equal(self, expected, actual):
        for (name, left), (_, right) in zip(expected.named_parameters(), actual.named_parameters()):
            if name.startswith("resampler.") and name.endswith("attention.in_proj_bias"):
                width = left.numel() // 3
                key = slice(width, 2 * width)
                # A constant bias on every key adds one constant to each query's
                # attention logits, which softmax cancels. Its exact gradient is
                # zero. FP32 residual gradients (~3e-9 here) are amplified by
                # AdamW eps=1e-8: the measured two-step difference is <=2.45e-6.
                for parameter in (left, right):
                    self.assertLess(parameter.grad[key].abs().max().item(), 1e-7, name)
                torch.testing.assert_close(left[key], right[key], atol=3e-6, rtol=0,
                                           msg=f"AdamW null key-bias direction {name}")
                left = torch.cat((left[:width], left[2 * width:]))
                right = torch.cat((right[:width], right[2 * width:]))
            # All other parameters, including Q/V bias slices, measured <=1.5e-8.
            torch.testing.assert_close(left, right, atol=5e-8, rtol=1e-5,
                                       msg=f"default AdamW update {name}")

    def training_pair(self, variant, readout, **kwargs):
        queries, items = batch_fixture()
        legacy = ranker(variant, readout, input_assembly="legacy", **kwargs).train()
        legacy.calibrate_bridge(queries, items)
        batched = ranker(variant, readout, input_assembly="batched", **kwargs).train()
        batched.load_state_dict(legacy.state_dict(), strict=True)
        self.assertEqual(legacy.state_dict().keys(), batched.state_dict().keys())
        self.assertEqual(legacy.metadata()["student_template"], batched.metadata()["student_template"])
        return legacy, batched, queries, items

    def test_v1_v3_all_readouts_match_scores_loss_gradients_and_two_adam_steps(self):
        for variant in ("V1", "V3"):
            for readout in ("last_token", "memory", "perceiver"):
                with self.subTest(variant=variant, readout=readout):
                    legacy, batched, queries, items = self.training_pair(variant, readout)
                    # A normal nonzero epsilon prevents nearly zero roundoff
                    # gradients from dominating the comparison of Adam steps.
                    optimizers = [torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-6)
                                  for model in (legacy, batched)]
                    for batch in (queries, changed_batch(queries)):
                        for optimizer in optimizers:
                            optimizer.zero_grad(set_to_none=True)
                        teachers = [torch.linspace(-.4, .8, len(query.candidate_ids)) for query in batch]
                        first = legacy.loss(batch, items, teacher_scores=teachers, beta=.35, temperature=1.7)
                        second = batched.loss(batch, items, teacher_scores=teachers, beta=.35,
                                              temperature=1.7, defer_scalar_metrics=True)
                        self.assert_outputs_equal(first, second)
                        first["loss"].backward()
                        second["loss"].backward()
                        self.assert_gradients_equal(legacy, batched)
                        for optimizer in optimizers:
                            optimizer.step()
                        self.assert_parameters_equal(legacy, batched)
                        self.assert_adam_states_equal(legacy, batched, *optimizers)
                    # State/cache warmed twice: checkpoint topology still matches.
                    self.assertEqual(legacy.state_dict().keys(), batched.state_dict().keys())

    def test_default_adamw_epsilon_preserves_two_updates_at_training_learning_rate(self):
        for variant in ("V1", "V3"):
            for readout in ("last_token", "memory", "perceiver"):
                with self.subTest(variant=variant, readout=readout):
                    legacy, batched, queries, items = self.training_pair(variant, readout)
                    # Keep AdamW's actual default eps=1e-8. Near-zero gradients,
                    # including invariant attention bias directions, can amplify
                    # FP32 reduction-order roundoff into a small update difference.
                    optimizers = [torch.optim.AdamW(model.parameters(), lr=1e-5)
                                  for model in (legacy, batched)]
                    self.assertTrue(all(optimizer.defaults["eps"] == 1e-8 for optimizer in optimizers))
                    for batch in (queries, changed_batch(queries)):
                        for optimizer in optimizers:
                            optimizer.zero_grad(set_to_none=True)
                        teachers = [torch.linspace(-.4, .8, len(query.candidate_ids)) for query in batch]
                        expected = legacy.loss(batch, items, teacher_scores=teachers, beta=.35, temperature=1.7)
                        actual = batched.loss(batch, items, teacher_scores=teachers, beta=.35,
                                              temperature=1.7, defer_scalar_metrics=True)
                        self.assert_outputs_equal(expected, actual)
                        expected["loss"].backward()
                        actual["loss"].backward()
                        self.assert_gradients_equal(legacy, batched)
                        for optimizer in optimizers:
                            optimizer.step()
                        self.assert_default_adamw_parameters_equal(legacy, batched)
                        self.assert_adam_states_equal(legacy, batched, *optimizers)
                        with torch.no_grad():
                            expected_scores = legacy(batch, items)["scores"]
                            actual_scores = batched(batch, items)["scores"]
                        for left, right in zip(expected_scores, actual_scores):
                            torch.testing.assert_close(left, right, atol=1e-7, rtol=1e-5)

    def test_single_native_lookup_and_bridge_preserve_per_book_learned_projection(self):
        for variant in ("V1", "V3"):
            for readout in ("last_token", "memory", "perceiver"):
                with self.subTest(variant=variant, readout=readout):
                    _, model, queries, items = self.training_pair(variant, readout)
                    # Warm runtime ID caches, then split both encoder documents
                    # and complete reader trees across multiple backbone calls.
                    probe = model(queries, items)
                    reader_capacity = max(lengths[0] + len(lengths) - 1
                                          for lengths in probe["logical_lengths"])
                    model.max_encoder_packed_tokens = 80
                    calls = {"native": [], "projection": [], "bridge": [], "encoder": []}
                    handles = [
                        model.b.get_input_embeddings().register_forward_pre_hook(
                            lambda module, inputs: calls["native"].append(inputs[0].detach().clone())),
                        model.projection.register_forward_pre_hook(
                            lambda module, inputs: calls["projection"].append(inputs[0].shape)),
                        model.bridge.register_forward_pre_hook(
                            lambda module, inputs: calls["bridge"].append(inputs[0].shape)),
                        model.a.register_forward_hook(
                            lambda module, inputs, output: calls["encoder"].append(output.last_hidden_state.shape)),
                    ]
                    try:
                        output = model(queries, items, max_reader_packed_tokens=reader_capacity)
                    finally:
                        for handle in handles:
                            handle.remove()
                    self.assertEqual(len(calls["native"]), 1)
                    self.assertEqual(calls["native"][0].dtype, torch.long)
                    self.assertEqual(len(calls["projection"]), output["statistics"]["encoder_unique_items"])
                    self.assertTrue(all(shape == (1, 16) for shape in calls["projection"]))
                    self.assertEqual(len(calls["bridge"]), 1)
                    self.assertGreater(len(calls["encoder"]), 1)
                    self.assertGreater(output["statistics"]["reader_packs"], 1)
                    self.assertEqual(output["statistics"]["encoder_unique_items"], 9 if variant == "V1" else 6)
                    self.assertEqual([len(group) for group in output["scores"]], [3, 2, 4])

    def test_equal_width_identity_keeps_one_projection_call_and_matches_default_adamw(self):
        for variant in ("V1", "V3"):
            for readout in ("last_token", "memory", "perceiver"):
                with self.subTest(variant=variant, readout=readout):
                    legacy, batched, queries, items = self.training_pair(
                        variant, readout, backbone_width_a=24, backbone_width_b=24)
                    self.assertIsInstance(batched.projection, torch.nn.Identity)
                    optimizers = [torch.optim.AdamW(model.parameters(), lr=1e-5) for model in (legacy, batched)]
                    for batch in (queries, changed_batch(queries)):
                        for optimizer in optimizers:
                            optimizer.zero_grad(set_to_none=True)
                        calls = []
                        handle = batched.projection.register_forward_pre_hook(
                            lambda module, inputs: calls.append(inputs[0].shape))
                        teachers = [torch.linspace(-.4, .8, len(query.candidate_ids)) for query in batch]
                        try:
                            expected = legacy.loss(batch, items, teacher_scores=teachers, beta=.35, temperature=1.7)
                            actual = batched.loss(batch, items, teacher_scores=teachers, beta=.35,
                                                  temperature=1.7, defer_scalar_metrics=True)
                        finally:
                            handle.remove()
                        self.assertEqual(len(calls), 1)
                        self.assert_outputs_equal(expected, actual)
                        expected["loss"].backward()
                        actual["loss"].backward()
                        self.assert_gradients_equal(legacy, batched)
                        for optimizer in optimizers:
                            optimizer.step()
                        self.assert_default_adamw_parameters_equal(legacy, batched)
                        self.assert_adam_states_equal(legacy, batched, *optimizers)
                        for a, b in zip(legacy(batch, items)["scores"], batched(batch, items)["scores"]):
                            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-5)

    def test_cache_uses_updated_trainable_native_embeddings_and_is_not_checkpoint_state(self):
        queries, items = batch_fixture()
        for variant in ("V1", "V3"):
            with self.subTest(variant=variant):
                legacy, batched, _, _ = self.training_pair(variant, "memory")
                initial_keys = tuple(batched.state_dict())
                initial_embeddings = batched.b.get_input_embeddings().weight.detach().clone()
                initial_scores = [score.detach().clone() for score in batched(queries, items)["scores"]]
                optimizers = [torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-6)
                              for model in (legacy, batched)]
                for model, optimizer in zip((legacy, batched), optimizers):
                    optimizer.zero_grad(set_to_none=True)
                    model.loss(queries, items, defer_scalar_metrics=True)["loss"].backward()
                    self.assertGreater(model.b.get_input_embeddings().weight.grad.abs().sum().item(), 0)
                    optimizer.step()
                self.assertFalse(torch.equal(initial_embeddings, batched.b.get_input_embeddings().weight))
                warmed = batched(queries, items)
                reference = legacy(queries, items)
                for before, actual, expected in zip(initial_scores, warmed["scores"], reference["scores"]):
                    self.assertFalse(torch.equal(before, actual))
                    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
                self.assertEqual(initial_keys, tuple(batched.state_dict()))
                # A fresh object gets no runtime caches from the checkpoint.
                restored = ranker(variant, "memory", input_assembly="batched").train()
                restored.load_state_dict(batched.state_dict(), strict=True)
                for actual, expected in zip(restored(queries, items)["scores"], warmed["scores"]):
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                # Moving parameters invalidates no IDs; cached floating embeddings
                # would retain FP32 values or a stale graph across this transition.
                batched.double()
                restored.double()
                for actual, expected in zip(batched(queries, items)["scores"], restored(queries, items)["scores"]):
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_unshared_headers_score_tokens_and_multilatent_suffix_keep_equivalence(self):
        for variant in ("V1", "V3"):
            for readout, options in (("memory", {"score_readout": "score_token"}),
                                     ("perceiver", {"m": 2, "score_readout": "legacy"})):
                with self.subTest(variant=variant, readout=readout):
                    legacy, batched, queries, items = self.training_pair(
                        variant, readout, shared_candidate_header=False, **options)
                    first = legacy.loss(queries, items)
                    second = batched.loss(queries, items, defer_scalar_metrics=True)
                    self.assert_outputs_equal(first, second)
                    first["loss"].backward()
                    second["loss"].backward()
                    self.assert_gradients_equal(legacy, batched)
                    if readout == "memory":
                        self.assertGreater(batched.score_token.grad.abs().sum().item(), 0)

    def test_reader_checkpointing_defaults_inherit_a_and_explicit_flags_are_independent(self):
        for encoder in (False, True):
            for reader in (None, False, True):
                with self.subTest(encoder=encoder, reader=reader):
                    model = ranker(input_assembly="batched", gradient_checkpointing=encoder,
                                   reader_gradient_checkpointing=reader)
                    self.assertEqual(model.a.is_gradient_checkpointing, encoder)
                    self.assertEqual(model.b.is_gradient_checkpointing, encoder if reader is None else reader)
        # The original constructor remains compatible with its combined switch.
        combined = ranker(gradient_checkpointing=True)
        self.assertTrue(combined.a.is_gradient_checkpointing)
        self.assertTrue(combined.b.is_gradient_checkpointing)

    def test_independent_checkpointing_keeps_forward_and_every_backward_gradient(self):
        queries, items = batch_fixture()
        for variant in ("V1", "V3"):
            reference = ranker(variant, "perceiver", input_assembly="batched").train()
            reference.calibrate_bridge(queries, items)
            expected = reference.loss(queries, items)
            expected["loss"].backward()
            for encoder, reader in ((True, False), (False, True), (True, True)):
                with self.subTest(variant=variant, encoder=encoder, reader=reader):
                    checkpointed = ranker(variant, "perceiver", input_assembly="batched",
                                          gradient_checkpointing=encoder,
                                          reader_gradient_checkpointing=reader).train()
                    checkpointed.load_state_dict(reference.state_dict(), strict=True)
                    actual = checkpointed.loss(queries, items, defer_scalar_metrics=True)
                    self.assert_outputs_equal(expected, actual)
                    actual["loss"].backward()
                    self.assert_gradients_equal(reference, checkpointed)

    def test_deferred_finite_flag_checks_scores_even_if_cross_entropy_is_finite(self):
        output = group_ranking_loss([torch.tensor([float("-inf"), .2])], [1],
                                    defer_scalar_metrics=True)
        self.assertTrue(torch.isfinite(output["loss"]).item())
        self.assertFalse(output["loss_finite"].item())
        for invalid in (torch.tensor([float("inf"), .2]), torch.tensor([float("nan"), .2])):
            output = group_ranking_loss([invalid], [1], defer_scalar_metrics=True)
            self.assertFalse(output["loss_finite"].item())
        for invalid in (float("inf"), float("nan")):
            output = group_ranking_loss([torch.tensor([.1, .2])], [1], beta=.3,
                                        teacher_scores=[torch.tensor([invalid, .2])],
                                        defer_scalar_metrics=True)
            self.assertFalse(output["loss_finite"].item())
        with self.assertRaises(ValueError):
            group_ranking_loss([torch.tensor([float("-inf"), .2])], [1])

    def test_deferred_ragged_loss_matches_beta_endpoints_without_host_scalar_reads(self):
        with self.assertRaisesRegex(AssertionError, "read a tensor scalar"):
            with NoHostScalarReads():
                torch.tensor(1.0).item()
        assert_groupwise_deferred_arithmetic(self, "cpu")

    def test_prepared_v3_item_ids_preserve_outputs_gradients_and_content_changes(self):
        queries, items = batch_fixture()
        for readout in ("last_token", "memory", "perceiver"):
            with self.subTest(readout=readout):
                ordinary = ranker("V3", readout, input_assembly="batched").train()
                ordinary.calibrate_bridge(queries, items)
                prepared = ranker("V3", readout, input_assembly="batched").train()
                prepared.load_state_dict(ordinary.state_dict(), strict=True)
                keys = tuple(prepared.state_dict())
                prepared.prepare_item_tokens(items)
                self.assertEqual(keys, tuple(prepared.state_dict()))
                first = ordinary.loss(queries, items)
                second = prepared.loss(queries, items, defer_scalar_metrics=True)
                self.assert_outputs_equal(first, second)
                first["loss"].backward()
                second["loss"].backward()
                self.assert_gradients_equal(ordinary, prepared)
                changed_items = {**items, "h0": "new history content delta",
                                 "c1": "new candidate content rho"}
                with torch.no_grad():
                    actual = prepared(queries, changed_items)
                    expected = ordinary(queries, changed_items)
                for left, right in zip(actual["scores"], expected["scores"]):
                    torch.testing.assert_close(left, right, atol=0, rtol=0)
                self.assertEqual(keys, tuple(prepared.state_dict()))
        with self.assertRaises(ValueError):
            ranker("V1", input_assembly="batched").prepare_item_tokens(items)

    def test_prepared_catalog_survives_more_entries_than_the_prompt_token_lru(self):
        queries, items = batch_fixture()
        model = ranker("V3", input_assembly="batched")
        model.calibrate_bridge(queries, items)
        original_keys = tuple(model.state_dict())
        # Query books come first, so a 32768-entry prompt LRU alone would evict
        # every one before the complete catalog finished being prepared.
        catalog = {**items, **{f"catalog-{index}": f"book {index}" for index in range(32769)}}
        model.prepare_item_tokens(catalog)
        with patch.object(model.tokenizer_a, "encode", side_effect=AssertionError("prepared item retokenized")):
            with torch.no_grad():
                output = model(queries, catalog)
        self.assertEqual([len(group) for group in output["scores"]], [3, 2, 4])
        self.assertEqual(original_keys, tuple(model.state_dict()))


@unittest.skipUnless(os.environ.get("MIXLM_FASTPATH_DIAGNOSTICS") in {"cpu", "cuda"},
                     "opt-in diagnostic: MIXLM_FASTPATH_DIAGNOSTICS=cpu or cuda")
class NextItemFastpathRoundingDiagnostics(unittest.TestCase):
    """Print attribution data; this diagnostic makes no accuracy-gate claim.

    Run the class alone, optionally selecting MIXLM_FASTPATH_DIAG_VARIANT=V1
    and MIXLM_FASTPATH_DIAG_READOUT=perceiver. It preserves AdamW lr=1e-5,
    eps=1e-8 and separates the old batched Linear, bridge, loss and native
    lookup changes by replaying the same initial weights and query order.
    """

    @classmethod
    def setUpClass(cls):
        cls.device = os.environ["MIXLM_FASTPATH_DIAGNOSTICS"]
        if cls.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA rounding diagnostic explicitly requires CUDA")
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, threads)
        if cls.device == "cuda":
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("CUDA rounding diagnostic requires BF16 hardware")
            previous = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            cls.addClassCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", previous)

    @staticmethod
    def snapshot(model, batch, items, bf16, deferred, teachers, optimizer):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(model.head.weight.device.type, dtype=torch.bfloat16, enabled=bf16):
            output = model.loss(batch, items, teacher_scores=teachers, beta=.35, temperature=1.7,
                                defer_scalar_metrics=deferred)
        output["loss"].backward()
        gradients, before, predictions = {}, {}, {}
        for name, parameter in model.named_parameters():
            gradients[name] = parameter.grad.detach().clone()
            before[name] = parameter.detach().clone()
            state = optimizer.state[parameter]
            step = int(state["step"].item()) + 1 if state else 1
            previous_m = state["exp_avg"].double() if state else torch.zeros_like(parameter, dtype=torch.float64)
            previous_v = state["exp_avg_sq"].double() if state else torch.zeros_like(parameter, dtype=torch.float64)
            gradient = parameter.grad.double()
            beta1, beta2 = optimizer.defaults["betas"]
            m = beta1 * previous_m + (1 - beta1) * gradient
            v = beta2 * previous_v + (1 - beta2) * gradient.square()
            lr, eps, decay = (optimizer.defaults[key] for key in ("lr", "eps", "weight_decay"))
            predictions[name] = (-lr * decay * parameter.detach().double()
                                 - lr * (m / (1 - beta1**step))
                                 / ((v / (1 - beta2**step)).sqrt() + eps))
        optimizer.step()
        after = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        with torch.no_grad(), torch.autocast(model.head.weight.device.type, dtype=torch.bfloat16, enabled=bf16):
            post = torch.cat(model(batch, items)["scores"]).detach()
        return {"grads": gradients, "before": before, "after": after,
                "deltas": {name: after[name] - before[name] for name in after},
                "predictions": predictions, "scores": torch.cat(output["scores"]).detach(),
                "post_scores": post, "loss": output["loss"].detach()}

    @staticmethod
    def parameter_comparison(name, left, right, section=None):
        def value(kind, snapshot):
            tensor = snapshot[kind][name]
            return tensor if section is None else tensor[section]

        difference = value("after", left) - value("after", right)
        flat = difference.abs().reshape(-1)
        index = int(flat.argmax().item())
        shape = difference.shape
        coordinates, remainder = [], index
        for width in reversed(shape):
            coordinates.append(remainder % width)
            remainder //= width
        coordinates.reverse()
        slice_coordinates = coordinates.copy()
        if section is not None:
            coordinates[0] += section.start
        delta_gap = value("deltas", left) - value("deltas", right)
        gradient_gap = value("grads", left) - value("grads", right)
        predicted_gap = value("predictions", left) - value("predictions", right)
        at = lambda tensor: tensor.reshape(-1)[index].item()
        rms = lambda tensor: tensor.double().square().mean().sqrt().item()
        return {"name": name, "slice": "all" if section is None else "K-bias",
                "index": coordinates, "index_in_slice": slice_coordinates,
                "parameter_max_diff": flat.max().item(),
                "update_max_diff": delta_gap.abs().max().item(), "update_rms_diff": rms(delta_gap),
                "gradient_max_diff": gradient_gap.abs().max().item(),
                "gradient_max_abs_reference": value("grads", left).abs().max().item(),
                "gradient_max_abs_case": value("grads", right).abs().max().item(),
                "gradient_at_index_reference": at(value("grads", left)),
                "gradient_at_index_case": at(value("grads", right)),
                "delta_at_index_reference": at(value("deltas", left)),
                "delta_at_index_case": at(value("deltas", right)),
                "predicted_delta_diff_at_index": at(predicted_gap),
                "predicted_delta_rms_diff": rms(predicted_gap)}

    def test_rounding_attribution(self):
        variant = os.environ.get("MIXLM_FASTPATH_DIAG_VARIANT", "V3")
        readout = os.environ.get("MIXLM_FASTPATH_DIAG_READOUT", "memory")
        self.assertIn(variant, ("V1", "V3"))
        self.assertIn(readout, ("last_token", "memory", "perceiver"))
        queries, items = batch_fixture()
        precisions = (False, True) if self.device == "cuda" else (False,)
        with sdpa_kernel(SDPBackend.MATH):
            for bf16 in precisions:
                for width_a in (16, 24):
                    base = ranker(variant, readout, input_assembly="legacy", backbone_width_a=width_a)
                    base.calibrate_bridge(queries, items)
                    initial = copy.deepcopy(base.state_dict())
                    reference = None
                    for mode in ("legacy", "old_batched_projection", "per_book_projection",
                                 "groupwise_loss", "groupwise_loss_per_book_bridge", "legacy_native_control"):
                        model = ranker(variant, readout, input_assembly="legacy" if mode == "legacy" else "batched",
                                       backbone_width_a=width_a).to(self.device).train()
                        model.load_state_dict(initial, strict=True)
                        if mode == "old_batched_projection" and not isinstance(model.projection, torch.nn.Identity):
                            # Replay the failed v1 shape change without adding a
                            # production option or changing any weight/state key.
                            project, bridge = model.projection.forward, model.bridge.forward
                            model.projection.forward = lambda states: states
                            model.bridge.forward = lambda states: bridge(project(states))
                        if mode in {"groupwise_loss_per_book_bridge", "legacy_native_control"}:
                            bridge = model.bridge.forward
                            model.bridge.forward = lambda states: torch.cat([bridge(row) for row in states.split(1)])
                        if mode == "legacy_native_control":
                            model._batched_native_embeddings = lambda batch, catalog: None
                        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
                        self.assertEqual(optimizer.defaults["eps"], 1e-8)
                        snapshots = []
                        for step, batch in enumerate((queries, changed_batch(queries)), 1):
                            teachers = [torch.linspace(-.4, .8, len(query.candidate_ids), device=self.device)
                                        for query in batch]
                            deferred = mode in {"old_batched_projection", "per_book_projection"}
                            actual = self.snapshot(model, batch, items, bf16, deferred, teachers, optimizer)
                            snapshots.append(actual)
                            if mode == "legacy":
                                continue
                            expected = reference[step - 1]
                            details = [self.parameter_comparison(name, expected, actual) for name in actual["after"]]
                            selected = sorted(details, key=lambda entry: entry["parameter_max_diff"], reverse=True)[:5]
                            names = {entry["name"] for entry in selected}
                            for name in ("projection.weight", "bridge.log_gain", "a.layers.1.mlp.up_proj.weight"):
                                if name in actual["after"] and name not in names:
                                    selected.append(self.parameter_comparison(name, expected, actual))
                            for name, parameter in model.named_parameters():
                                if name.startswith("resampler.") and name.endswith("attention.in_proj_bias"):
                                    width = parameter.numel() // 3
                                    selected.append(self.parameter_comparison(name, expected, actual, slice(width, 2 * width)))
                            print(json.dumps({"diagnostic_only": True, "accuracy_gate_asserted": False,
                                              "device": self.device, "torch": torch.__version__,
                                              "backend": "sdpa_math", "master_dtype": "float32",
                                              "projection": type(model.projection).__name__,
                                              "threads": torch.get_num_threads(),
                                              "tf32_matmul": torch.backends.cuda.matmul.allow_tf32 if self.device == "cuda" else False,
                                              "precision": "bf16" if bf16 else "fp32", "variant": variant,
                                              "readout": readout, "width_a": width_a, "width_b": 24,
                                              "mode": mode, "step": step, "lr": 1e-5, "eps": 1e-8,
                                              "loss_abs_diff": abs((expected["loss"] - actual["loss"]).item()),
                                              "score_max_diff": (expected["scores"] - actual["scores"]).abs().max().item(),
                                              "post_update_score_max_diff": (expected["post_scores"] - actual["post_scores"]).abs().max().item(),
                                              "parameters": selected}, sort_keys=True), flush=True)
                        if mode == "legacy":
                            reference = snapshots


@unittest.skipUnless(os.environ.get("MIXLM_FASTPATH_CUDA") == "1",
                     "opt-in CUDA gate: set MIXLM_FASTPATH_CUDA=1")
class NextItemFastpathCudaTests(unittest.TestCase):
    """Explicit CUDA correctness gate; enabling it without CUDA is an error.

    CUDA tolerances are proposed until this opt-in gate runs on the target GPU.
    BF16 checks every actual CE/KD gradient tensor with a 6% RMS criterion and
    global gradient RMS floor, in addition to the 6% full-parameter score VJP.
    AdamW update differences use 6% update RMS plus .003 * learning rate; first
    and second moments use 6% and 12% RMS bounds with global moment RMS floors.
    Only the mathematically null Perceiver K-bias slice has a separate physical
    AdamW update bound, coupled to its gradient-null and post-update function
    checks. These numerical gates do not assert bitwise training equivalence.
    """

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("MIXLM_FASTPATH_CUDA=1 explicitly requires a CUDA runtime; gate was not run")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("MIXLM_FASTPATH_CUDA=1 requires BF16-capable CUDA hardware")
        cls.previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        cls.previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cls.addClassCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", cls.previous_tf32)
        cls.addClassCleanup(setattr, torch.backends.cudnn, "allow_tf32", cls.previous_cudnn_tf32)

    def assert_cuda_outputs_equal(self, expected, actual, bf16):
        self.assertEqual(expected["statistics"], actual["statistics"])
        self.assertEqual(expected["logical_lengths"], actual["logical_lengths"])
        for left, right in zip(expected["scores"], actual["scores"]):
            torch.testing.assert_close(left, right, atol=.003 if bf16 else 2e-6,
                                       rtol=.03 if bf16 else 1e-5)
        torch.testing.assert_close(expected["loss"], actual["loss"],
                                   atol=.003 if bf16 else 2e-6, rtol=.03 if bf16 else 1e-5)
        for metric in ("ce", "kd"):
            self.assertEqual(actual[metric].shape, torch.Size([]))
            self.assertFalse(actual[metric].requires_grad)
            torch.testing.assert_close(actual[metric], actual[metric].new_tensor(expected[metric]),
                                       atol=.003 if bf16 else 2e-6, rtol=.03 if bf16 else 1e-5)
        self.assertEqual(actual["loss_finite"].device.type, "cuda")
        self.assertTrue(actual["loss_finite"].item())

    def assert_all_gradients_finite(self, model):
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.dtype, torch.float32, name)
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
        parameters = dict(model.named_parameters())
        for name in ("a.embed_tokens.weight", "b.embed_tokens.weight", "projection.weight",
                     "head.weight", "bridge.log_gain"):
            if name in parameters:
                self.assertGreater(parameters[name].grad.abs().sum().item(), 0, name)

    @staticmethod
    def rms(tensor):
        return tensor.double().square().mean().sqrt().item()

    @staticmethod
    def global_rms(tensors):
        count = sum(tensor.numel() for tensor in tensors)
        return (sum(tensor.double().square().sum() for tensor in tensors) / count).sqrt().item()

    def assert_scale_aware_pairs(self, pairs, relative):
        global_scale = max(self.global_rms([a for _, a, _ in pairs]),
                           self.global_rms([b for _, _, b in pairs]))
        for name, a, b in pairs:
            error = self.rms(a - b)
            bound = relative * max(self.rms(a), self.rms(b), global_scale)
            self.assertLessEqual(error, bound + 1e-15,
                                 f"{name}: RMS difference {error} exceeds proposed bound {bound}")

    def assert_cuda_optimizer_equal(self, legacy, batched, optimizers, before, step, bf16):
        parameters = list(zip(legacy.named_parameters(), batched.named_parameters()))
        gradient_scale = max(self.global_rms([a.grad for (name, a), _ in parameters]),
                             self.global_rms([b.grad for _, (name, b) in parameters]))
        for (name, a), (other, b) in parameters:
            self.assertEqual(name, other)
            delta_a, delta_b = a.detach() - before[0][name], b.detach() - before[1][name]
            if name.startswith("resampler.") and name.endswith("attention.in_proj_bias"):
                width = a.numel() // 3
                key = slice(width, 2 * width)
                for parameter in (a, b):
                    if bf16:
                        bound = .003 * max(self.rms(parameter.grad), gradient_scale)
                        self.assertLessEqual(self.rms(parameter.grad[key]), bound + 1e-15, name)
                    else:
                        self.assertLess(parameter.grad[key].abs().max().item(), 1e-7, name)
                # Each AdamW step is bounded by learning rate in this null slice.
                key_bound = 2 * step * 1e-5 + 1e-9 if bf16 else 3e-6
                torch.testing.assert_close(a[key], b[key], atol=key_bound, rtol=0, msg=name)
                delta_a = torch.cat((delta_a[:width], delta_a[2 * width:]))
                delta_b = torch.cat((delta_b[:width], delta_b[2 * width:]))
                a = torch.cat((a[:width], a[2 * width:]))
                b = torch.cat((b[:width], b[2 * width:]))
            if bf16:
                # Compare updates themselves, so FP32 master-weight magnitudes
                # cannot hide a wrong optimizer update behind a loose relative
                # tolerance on the parameter's pre-existing value.
                bound = .06 * max(self.rms(delta_a), self.rms(delta_b)) + .003 * 1e-5
                self.assertLessEqual(self.rms(delta_a - delta_b), bound, name)
            else:
                torch.testing.assert_close(a, b, atol=2e-7, rtol=1e-5, msg=name)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            pairs = [(name, optimizers[0].state[a][key], optimizers[1].state[b][key])
                     for (name, a), (_, b) in parameters]
            for name, a, b in pairs:
                self.assertTrue(torch.isfinite(a).all().item(), name)
                self.assertTrue(torch.isfinite(b).all().item(), name)
            if key == "step":
                for name, a, b in pairs:
                    torch.testing.assert_close(a, b, atol=0, rtol=0, msg=name)
            elif bf16:
                self.assert_scale_aware_pairs(pairs, .06 if key == "exp_avg" else .12)
            else:
                for name, a, b in pairs:
                    torch.testing.assert_close(a, b, atol=1e-7 if key == "exp_avg" else 1e-9,
                                               rtol=3e-4, msg=f"{key} {name}")

    def assert_bf16_full_parameter_vjp(self, legacy, batched, expected, actual):
        # Weighted scores prevent near-zero CE cancellations from making a
        # relative BF16 criterion meaningless. Every trainable parameter enters
        # this VJP; actual CE+KD gradients are also checked before AdamW below.
        def scalar(output):
            return sum((index + 1) * (group * torch.linspace(-.7, .9, len(group),
                                                           device=group.device)).sum()
                       for index, group in enumerate(output["scores"]))

        left = torch.autograd.grad(scalar(expected), tuple(legacy.parameters()), retain_graph=True)
        right = torch.autograd.grad(scalar(actual), tuple(batched.parameters()), retain_graph=True)
        self.assertEqual(len(left), len(right))
        for (name, _), a, b in zip(legacy.named_parameters(), left, right):
            self.assertTrue(torch.isfinite(a).all().item(), name)
            self.assertTrue(torch.isfinite(b).all().item(), name)
        difference = sum((a.double() - b.double()).square().sum() for a, b in zip(left, right)).sqrt()
        reference = sum(a.double().square().sum() for a in left).sqrt().clamp_min(1e-12)
        self.assertLessEqual((difference / reference).item(), .06)

    def test_cuda_fp32_bf16_scores_all_gradients_adamw_and_cache_with_independent_gc(self):
        queries, items = batch_fixture()
        # MATH avoids compiling a production FlexAttention graph in this small
        # correctness gate. Both paths use the identical backend and precision.
        with sdpa_kernel(SDPBackend.MATH):
            assert_groupwise_deferred_arithmetic(self, "cuda")
            for bf16 in (False, True):
                for variant in ("V1", "V3"):
                    for readout in ("last_token", "memory", "perceiver"):
                        for reader_gc in (None, False):
                            for width_a in (16, 24):
                                with self.subTest(precision="bf16" if bf16 else "fp32", variant=variant,
                                                  readout=readout, reader_gc=reader_gc, width_a=width_a):
                                    self.cuda_training_case(variant, readout, reader_gc, bf16, queries, items, width_a)

    def cuda_training_case(self, variant, readout, reader_gc, bf16, queries, items, width_a):
        options = dict(gradient_checkpointing=True, reader_gradient_checkpointing=reader_gc,
                       backbone_width_a=width_a, backbone_width_b=24)
        legacy = ranker(variant, readout, input_assembly="legacy", **options).train()
        legacy.calibrate_bridge(queries, items)
        batched = ranker(variant, readout, input_assembly="batched", **options).train()
        batched.load_state_dict(legacy.state_dict(), strict=True)
        legacy.cuda()
        batched.cuda()
        checkpoint_keys = tuple(batched.state_dict())
        for model in (legacy, batched):
            self.assertTrue(model.a.is_gradient_checkpointing)
            self.assertEqual(model.b.is_gradient_checkpointing, reader_gc is None)
            self.assertTrue(all(parameter.dtype == torch.float32 for parameter in model.parameters()))
        optimizers = [torch.optim.AdamW(model.parameters(), lr=1e-5) for model in (legacy, batched)]
        self.assertTrue(all(optimizer.defaults["eps"] == 1e-8 for optimizer in optimizers))
        lookups = []
        handle = batched.b.get_input_embeddings().register_forward_hook(
            lambda module, args, output: lookups.append((args[0].detach().clone(), output.detach().clone())))
        try:
            for step, batch in enumerate((queries, changed_batch(queries)), 1):
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                teachers = [torch.linspace(-.4, .8, len(query.candidate_ids), device="cuda") for query in batch]
                lookups.clear()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                    expected = legacy.loss(batch, items, teacher_scores=teachers, beta=.35, temperature=1.7)
                    actual = batched.loss(batch, items, teacher_scores=teachers, beta=.35, temperature=1.7,
                                          defer_scalar_metrics=True)
                self.assertEqual(len(lookups), 1)
                self.assertEqual(lookups[0][0].dtype, torch.long)
                self.assert_cuda_outputs_equal(expected, actual, bf16)
                if bf16:
                    self.assert_bf16_full_parameter_vjp(legacy, batched, expected, actual)
                expected["loss"].backward()
                actual["loss"].backward()
                self.assert_all_gradients_finite(legacy)
                self.assert_all_gradients_finite(batched)
                if bf16:
                    self.assert_scale_aware_pairs(
                        [(name, a.grad, b.grad) for (name, a), (_, b)
                         in zip(legacy.named_parameters(), batched.named_parameters())], .06)
                else:
                    for (name, a), (_, b) in zip(legacy.named_parameters(), batched.named_parameters()):
                        torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=3e-4, msg=name)
                before = [{name: parameter.detach().clone() for name, parameter in model.named_parameters()}
                          for model in (legacy, batched)]
                for optimizer in optimizers:
                    optimizer.step()
                self.assertFalse(torch.equal(before[1]["b.embed_tokens.weight"],
                                             batched.b.get_input_embeddings().weight))
                # Optimizer equivalence is checked jointly with the resulting
                # function. Null Perceiver K-bias gradients have the FP32 AdamW
                # caveat documented by the CPU test, so bitwise updates are not
                # required for that invariant direction on CUDA either.
                lookups.clear()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                    reference = legacy.loss(batch, items, teacher_scores=teachers, beta=.35, temperature=1.7)
                    warmed = batched.loss(batch, items, teacher_scores=teachers, beta=.35, temperature=1.7,
                                          defer_scalar_metrics=True)
                self.assert_cuda_outputs_equal(reference, warmed, bf16)
                self.assertEqual(len(lookups), 1)
                ids, values = lookups[0]
                torch.testing.assert_close(values, torch.nn.functional.embedding(
                    ids, batched.b.get_input_embeddings().weight), atol=0, rtol=0)
                self.assertEqual(checkpoint_keys, tuple(batched.state_dict()))
                # A fresh model receives weights/calibration and no runtime IDs
                # or embedding graph cache from this warmed model.
                fresh = ranker(variant, readout, input_assembly="batched", **options).cuda().train()
                fresh.load_state_dict(batched.state_dict(), strict=True)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                    clean = fresh(batch, items)
                for a, b, c in zip(reference["scores"], warmed["scores"], clean["scores"]):
                    torch.testing.assert_close(a, b, atol=.003 if bf16 else 2e-6,
                                               rtol=.03 if bf16 else 1e-5)
                    torch.testing.assert_close(b, c, atol=.003 if bf16 else 2e-6,
                                               rtol=.03 if bf16 else 1e-5)
                self.assert_cuda_optimizer_equal(legacy, batched, optimizers, before, step, bf16)
        finally:
            handle.remove()


if __name__ == "__main__":
    unittest.main()
