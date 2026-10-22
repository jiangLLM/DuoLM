"""Compact student interfaces against real, tiny Qwen3 causal backbones."""

import copy
from dataclasses import replace
import io
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3Model

from duolm.next_item_data import HistoryEntry, RecommendationQuery, user_split
from duolm.next_item_model import (
    CANDIDATE_HEADER, INSTRUCTION, LEGACY_SCORE_SUFFIX, TEMPLATE_VERSION,
    GroupRanker, _last_valid_token, query_input_hash,
)
from duolm.resampler import PerceiverResampler
from duolm.ranking_packing import packed_segments


class Tokenizer:
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [2 + ord(character) % 250 for character in text]


def backbone(width=16):
    config = Qwen3Config(vocab_size=256, hidden_size=width, intermediate_size=2 * width,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=width // 2, max_position_embeddings=2048,
                         attention_dropout=0, use_cache=False)
    config._attn_implementation = "sdpa"
    return Qwen3Model(config)


def fixture():
    user = next(f"compact-{i}" for i in range(100) if user_split(f"compact-{i}") == "train")
    history = (HistoryEntry("h0", 1, 1), HistoryEntry("h1", 5, 2))
    query = RecommendationQuery("compact-query", user, "train", history,
                                ("c0", "c1", "c2"), 0, "c0", 3, ("h0", "h1"))
    content = {"h0": "short history", "h1": "a history book with more text",
               "c0": "first book", "c1": "a quite different longer book",
               "c2": "third"}
    return query, content


def student(variant="V3", **kwargs):
    torch.manual_seed(901)
    options = dict(m=1, a_readout="last_token", shared_candidate_header=True,
                   score_readout="candidate", max_encoder_packed_tokens=1024,
                   max_reader_packed_tokens=2048)
    options.update(kwargs)
    return GroupRanker(backbone(), backbone(), Tokenizer(), Tokenizer(), variant=variant, **options)


class CompactRankerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_gradients_equal(self, left, right):
        a, b = dict(left.named_parameters()), dict(right.named_parameters())
        self.assertEqual(set(a), set(b))
        for name in a:
            self.assertIsNotNone(a[name].grad, name)
            self.assertIsNotNone(b[name].grad, name)
            torch.testing.assert_close(a[name].grad, b[name].grad, atol=3e-6, rtol=3e-4, msg=name)

    def test_teacher_default_keys_template_and_outputs_keep_legacy_checkpoint_compatibility(self):
        torch.manual_seed(10)
        legacy = GroupRanker(None, backbone(), None, Tokenizer())
        torch.manual_seed(10)
        explicit = GroupRanker(None, backbone(), None, Tokenizer(), a_readout="perceiver",
                               shared_candidate_header=False, score_readout="legacy")
        explicit.load_state_dict(legacy.state_dict(), strict=True)
        self.assertEqual(legacy.metadata(), explicit.metadata())
        self.assertEqual(legacy.metadata()["template"], TEMPLATE_VERSION)
        self.assertNotIn("student_template", legacy.metadata())
        self.assertFalse(any("score_token" in key or "memory_token" in key for key in legacy.state_dict()))
        q, content = fixture()
        prefix, branches = legacy._query_embeddings(q, content)
        expected = [legacy._native(INSTRUCTION)]
        for index, event in enumerate(q.history):
            expected.extend((legacy._native(f"History {index+1}:\nItem: "),
                             legacy._native(content[event.item_id]),
                             legacy._native(f"\nUser rating: {event.rating}\n")))
        torch.testing.assert_close(prefix, torch.cat(expected), atol=0, rtol=0)
        for item, branch in zip(q.candidate_ids, branches):
            expected = torch.cat((legacy._native(CANDIDATE_HEADER), legacy._native(content[item]),
                                  legacy._native(LEGACY_SCORE_SUFFIX)))
            torch.testing.assert_close(branch, expected, atol=0, rtol=0)
        torch.testing.assert_close(legacy([q], content)["scores"][0],
                                   explicit([q], content)["scores"][0], atol=0, rtol=0)

    def test_exact_candidate_header_sharing_preserves_outputs_and_all_gradients(self):
        q, content = fixture()
        for variant in ("V1", "V3"):
            repeated = student(variant, score_readout="legacy", shared_candidate_header=False).train()
            repeated.calibrate_bridge([q], content)
            shared = copy.deepcopy(repeated)
            shared.shared_candidate_header = True
            first, second = repeated.loss([q], content), shared.loss([q], content)
            torch.testing.assert_close(first["scores"][0], second["scores"][0], atol=3e-6, rtol=1e-5)
            first["loss"].backward()
            second["loss"].backward()
            self.assert_gradients_equal(repeated, shared)
            saved = first["statistics"]["reader_tokens"] - second["statistics"]["reader_tokens"]
            self.assertEqual(saved, (len(q.candidate_ids)-1) * len(shared._ids(Tokenizer(), CANDIDATE_HEADER)))
            self.assertEqual(first["logical_lengths"], second["logical_lengths"])

    def test_no_suffix_single_candidate_position_is_isolated_and_user_conditioned(self):
        q, content = fixture()
        for readout in ("last_token", "memory", "perceiver"):
            with self.subTest(readout=readout):
                model = student(a_readout=readout, resampler_layers=1, resampler_heads=8)
                model.calibrate_bridge([q], content)
                states, references, _ = model._a_blocks([q], content)
                prefix, branches = model._query_embeddings(q, content, states, references[0])
                self.assertTrue(all(branch.shape == (1, model.b.config.hidden_size) for branch in branches))
                output = model([q], content)["scores"][0]
                changed = model([q], {**content, "c1": "totally changed candidate"})["scores"][0]
                torch.testing.assert_close(output[[0, 2]], changed[[0, 2]], atol=2e-6, rtol=1e-5)
                self.assertFalse(torch.equal(output[1], changed[1]))
                permuted = replace(q, candidate_ids=q.candidate_ids[::-1], positive_index=2)
                torch.testing.assert_close(output, model([permuted], content)["scores"][0].flip(0), atol=2e-6, rtol=1e-5)
                changed_history = model([q], {**content, "h0": "an entirely different user's previous reading"})["scores"][0]
                self.assertFalse(torch.equal(output, changed_history))
                (output * output.new_tensor([1., -.3, .2])).sum().backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                required = ["a.embed_tokens.weight", "b.embed_tokens.weight", "head.weight", "bridge.log_gain"]
                if readout == "memory":
                    required.append("memory_token")
                elif readout == "perceiver":
                    required.extend(("resampler.queries", "resampler.layers.0.attention.in_proj_weight",
                                     "resampler.layers.0.ff.0.weight", "resampler.layers.0.ff.2.weight"))
                parameters = dict(model.named_parameters())
                for name in required:
                    self.assertGreater(parameters[name].grad.abs().sum().item(), 0, name)
                self.assertEqual(len(prefix) + 1, model([q], content)["logical_lengths"][0][0])

    def test_score_token_is_one_trainable_position_and_roundtrips_checkpoint(self):
        model, (q, content) = student(a_readout="memory", score_readout="score_token"), fixture()
        model.calibrate_bridge([q], content)
        states, references, _ = model._a_blocks([q], content)
        _, branches = model._query_embeddings(q, content, states, references[0])
        self.assertTrue(all(len(branch) == 2 for branch in branches))
        self.assertEqual(model.memory_token.shape, (model.a.config.hidden_size,))
        self.assertEqual(model.score_token.shape, (model.b.config.hidden_size,))
        model.loss([q], content)["loss"].backward()
        for parameter in (model.memory_token, model.score_token):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        with torch.no_grad():
            model.memory_token.add_(.001 * model.memory_token.grad)
            model.score_token.add_(.001 * model.score_token.grad)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = student(a_readout="memory", score_readout="score_token")
        restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
        self.assertEqual(model.metadata(), restored.metadata())
        torch.testing.assert_close(model([q], content)["scores"][0], restored([q], content)["scores"][0], atol=0, rtol=0)

    def test_single_vector_readouts_match_independent_documents_and_pack_boundaries(self):
        q, content = fixture()
        for readout in ("last_token", "memory", "perceiver"):
            model = student(a_readout=readout, resampler_layers=1, resampler_heads=8)
            model.calibrate_bridge([q], content)
            actual, refs, stats = model._a_blocks([q], content)
            for item, reference in zip([h.item_id for h in q.history] + list(q.candidate_ids), refs[0][0]+refs[0][1]):
                ids = model._ids(model.tokenizer_a, "Item: " + content[item]) + (Tokenizer.eos_token_id,)
                native = model.a.get_input_embeddings()(torch.tensor(ids))
                if readout == "memory":
                    native = torch.cat((native, model.memory_token[None]))
                hidden = model.a(inputs_embeds=native[None], position_ids=torch.arange(len(native))[None],
                                 attention_mask=torch.ones((1, len(native)), dtype=torch.long),
                                 use_cache=False).last_hidden_state
                summary = (model.resampler(hidden, torch.ones(hidden.shape[:2], dtype=torch.bool), [1])[0]
                           if readout == "perceiver" else hidden[0, -1:])
                expected = model.bridge(model.projection(summary))
                torch.testing.assert_close(actual[reference], expected, atol=3e-6, rtol=1e-5)
            self.assertEqual(stats["encoder_unique_items"], 5)
            self.assertEqual(stats["encoder_padding_tokens"], 0)
            second = copy.deepcopy(model)
            second.max_encoder_packed_tokens = 80
            split, _, split_stats = second._a_blocks([q], content)
            for a, b in zip(actual, split):
                torch.testing.assert_close(a, b, atol=3e-6, rtol=1e-5)
            self.assertEqual(stats, split_stats)

    def test_last_valid_readout_handles_left_right_padding_without_pooling(self):
        hidden = torch.arange(2 * 5 * 3).reshape(2, 5, 3).float()
        mask = torch.tensor([[False, False, True, True, True], [True, True, False, False, False]])
        torch.testing.assert_close(_last_valid_token(hidden, mask), torch.stack((hidden[0, 4], hidden[1, 1]))[:, None])
        for invalid in (torch.zeros_like(mask), torch.ones((2, 4), dtype=torch.bool)):
            with self.assertRaises(ValueError):
                _last_valid_token(hidden, invalid)

    def test_perceiver_one_and_two_layers_ignore_padding(self):
        torch.manual_seed(49)
        hidden = torch.randn((2, 5, 16))
        mask = torch.tensor([[True, True, False, False, False], [False, True, True, True, False]])
        for layers in (1, 2):
            readout = PerceiverResampler(16, 2, layers, 2)
            first = readout(hidden, mask, [1, 2])
            changed = hidden.clone()
            changed[~mask] = torch.randn_like(changed[~mask]) * 100
            torch.testing.assert_close(first, readout(changed, mask, [1, 2]), atol=1e-6, rtol=1e-5)
            self.assertTrue(torch.equal(first[0, 1], torch.zeros_like(first[0, 1])))
            first.sum().backward()
            self.assertIsNotNone(readout.queries.grad)

    def test_v1_feedback_is_only_in_a_while_v3_feedback_remains_explicit_in_b(self):
        q, content = fixture()
        changed = replace(q, history=(replace(q.history[0], rating=5), q.history[1]))
        for variant in ("V1", "V3"):
            model = student(variant)
            template = model.metadata()["student_template"]
            self.assertEqual(template["variant"], variant)
            self.assertEqual(template["feedback_channel"], "a_rating_b_unknown" if variant == "V1" else "b_explicit_rating")
            model.calibrate_bridge([q], content)
            states, refs, _ = model._a_blocks([q, changed], content)
            same_block = refs[0][0][0] == refs[1][0][0]
            self.assertEqual(same_block, variant == "V3")
            # Substitute the same A states to isolate the B feedback channel.
            first, _ = model._query_embeddings(q, content, states, refs[0])
            second, _ = model._query_embeddings(changed, content, states, refs[0])
            self.assertEqual(torch.equal(first, second), variant == "V1")

    def test_interface_metadata_separates_student_templates_from_unchanged_teacher_hash(self):
        q, content = fixture()
        teacher_hash = query_input_hash(q, content)
        signatures = set()
        for options in ({}, {"score_readout": "score_token"}, {"a_readout": "memory"},
                        {"a_readout": "perceiver", "resampler_layers": 1, "resampler_heads": 2}):
            model = student(**options)
            signatures.add(model.metadata()["student_template"]["sha256"])
            self.assertEqual(query_input_hash(q, content), teacher_hash)
        self.assertEqual(len(signatures), 4)
        self.assertNotEqual(student("V1").metadata()["student_template"]["sha256"],
                            student("V3").metadata()["student_template"]["sha256"])

    def test_invalid_or_ambiguous_multivector_interfaces_are_rejected(self):
        for options in ({"a_readout": "unknown"}, {"score_readout": "unknown"},
                        {"a_readout": "last_token", "m": 2}, {"a_readout": "memory", "m": 8},
                        {"a_readout": "perceiver", "score_readout": "candidate", "m": 2},
                        {"shared_candidate_header": "yes"}):
            with self.assertRaises(ValueError):
                student(**options)
        for options in ({"shared_candidate_header": True}, {"score_readout": "candidate"}, {"a_readout": "memory"}):
            with self.assertRaisesRegex(ValueError, "teacher must preserve"):
                GroupRanker(None, backbone(), None, Tokenizer(), **options)

    def test_only_student_flex_attention_forwards_forced_prefill_kernel_options(self):
        layout = packed_segments([2])
        for variant in ("T", "V1", "V3"):
            model = (GroupRanker(None, backbone(), None, Tokenizer()) if variant == "T"
                     else student(variant))
            for backend in ("sdpa", "flex_attention"):
                model.attention_backend = backend
                for module in ((model.b,) if variant == "T" else (model.a, model.b)):
                    inputs = torch.randn((2, module.config.hidden_size))
                    output = SimpleNamespace(last_hidden_state=inputs[None])
                    with patch.object(module, "forward", return_value=output) as forward, \
                            patch.object(model, "_mask", return_value=None):
                        torch.testing.assert_close(model._run(module, inputs, layout), inputs)
                    expected = {"FORCE_USE_FLEX_ATTENTION": True} if variant != "T" and backend == "flex_attention" else None
                    self.assertEqual(forward.call_args.kwargs.get("kernel_options"), expected)
                if variant != "T":
                    expected = "forced_prefill" if backend == "flex_attention" else None
                    self.assertEqual(model.metadata()["student_template"]["flex_attention_kernel"], expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for compact BF16 FlexAttention acceptance")
    def test_cuda_compact_no_suffix_students_match_sdpa(self):
        child_variant = os.environ.get("MIXLM_COMPACT_CUDA_VARIANT")
        if child_variant is None:
            for variant, readout in (("V1", "last_token"), ("V3", "last_token"),
                                     ("V3", "memory"), ("V3", "perceiver")):
                env = dict(os.environ, MIXLM_COMPACT_CUDA_VARIANT=variant,
                           MIXLM_COMPACT_CUDA_READOUT=readout)
                env["PYTHONPATH"] = os.pathsep.join(filter(None, (
                    str(Path(__file__).resolve().parents[1] / "src"), env.get("PYTHONPATH"))))
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                         "CompactRankerTests.test_cuda_compact_no_suffix_students_match_sdpa"],
                                        env=env, capture_output=True, text=True, timeout=600)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(f"MIXLM_COMPACT_CUDA_COMPLETED={variant}", result.stdout)
                self.assertIn(f"MIXLM_COMPACT_CUDA_READOUT_COMPLETED={variant}/{readout}", result.stdout)
            return
        self.assertIn(child_variant, ("V1", "V3"))
        child_readout = os.environ.get("MIXLM_COMPACT_CUDA_READOUT", "last_token")
        self.assertIn(child_readout, ("last_token", "memory", "perceiver"))
        q, content = fixture()
        queries = [q, replace(q, query_id="second", candidate_ids=q.candidate_ids[::-1], positive_index=2)]
        torch.manual_seed(901)
        # Match the production Qwen3 head width (128) in the CUDA gate. The
        # smaller CPU fixture's head_dim=32 failed autotuning in the pinned
        # H100 runtime; that failure cannot validate the production shape.
        reference = GroupRanker(backbone(256), backbone(256), Tokenizer(), Tokenizer(), variant=child_variant,
                                m=1, a_readout=child_readout, resampler_layers=1, resampler_heads=8,
                                shared_candidate_header=True,
                                score_readout="candidate", gradient_checkpointing=True,
                                max_encoder_packed_tokens=2048, max_reader_packed_tokens=2048).cuda().train()
        reference.calibrate_bridge(queries, content)
        sparse = copy.deepcopy(reference)
        sparse.attention_backend = "flex_attention"
        for module in (sparse.a, sparse.b):
            module.set_attn_implementation("flex_attention")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual, expected = sparse.loss(queries, content), reference.loss(queries, content)
        for a, b in zip(actual["scores"], expected["scores"]):
            torch.testing.assert_close(a, b, atol=.003, rtol=.03)
        torch.testing.assert_close(actual["loss"], expected["loss"], atol=.003, rtol=.03)
        actual["loss"].backward()
        expected["loss"].backward()
        for model in (sparse, reference):
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            model.zero_grad(set_to_none=True)
        # Compare the Jacobian with a non-cancelling VJP; tiny random students'
        # candidate differences make relative CE-gradient roundoff ill-conditioned.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual, expected = sparse(queries, content), reference(queries, content)
        for output in (actual, expected):
            torch.stack([(scores * scores.new_tensor([1., -.3, .2])).sum() * (i+1)
                         for i, scores in enumerate(output["scores"])]).mean().backward()
        differences, magnitudes = [], []
        for (name, a), (other, b) in zip(sparse.named_parameters(), reference.named_parameters()):
            self.assertEqual(name, other)
            self.assertIsNotNone(a.grad, name)
            self.assertIsNotNone(b.grad, name)
            differences.append((a.grad.double()-b.grad.double()).square().sum())
            magnitudes.append(b.grad.double().square().sum())
        self.assertGreater(torch.stack(magnitudes).sum().item(), 0)
        relative = (torch.stack(differences).sum() / torch.stack(magnitudes).sum()).sqrt().item()
        self.assertLess(relative, .06)
        self.assertEqual(actual["statistics"]["reader_padding_tokens"], 0)
        print(f"MIXLM_COMPACT_CUDA_COMPLETED={child_variant}", flush=True)
        print(f"MIXLM_COMPACT_CUDA_READOUT_COMPLETED={child_variant}/{child_readout}", flush=True)


if __name__ == "__main__":
    if os.environ.get("MIXLM_COMPACT_CUDA_VARIANT") is not None and not torch.cuda.is_available():
        raise SystemExit("compact CUDA child requires a visible CUDA device; skipping cannot pass this gate")
    unittest.main()
