"""Real randomly initialized Qwen3 proofs; no downloaded weights or training runs."""

import copy
from dataclasses import replace
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
import unittest

import torch
from transformers import Qwen3Config, Qwen3Model

from duolm.next_item_data import HistoryEntry, RecommendationQuery, user_split
from duolm.next_item_model import (
    CANDIDATE_HEADER, INSTRUCTION, V1_UNKNOWN_FEEDBACK_SEGMENTS,
    GroupRanker, group_ranking_loss, query_input_hash,
)


class Tokenizer:
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [2 + ord(character) % 250 for character in text]


class FeedbackBPETokenizer(Tokenizer):
    """Qwen's actual feedback merge boundaries, with IDs remapped to a tiny vocab.

    Real Qwen encodes numeric feedback as [198,1474,10728,25,220,digit,198]
    but unknown feedback as [198,1474,10728,25,17607]: " ?\n" is one token.
    """
    unknown_id = 20
    before = [10, 11, 12, 13, 14]
    after = [10]

    def encode(self, text, add_special_tokens=False):
        if text == V1_UNKNOWN_FEEDBACK_SEGMENTS[0]:
            return self.before.copy()
        if text == "?":
            return [self.unknown_id]
        if text == "\n":
            return self.after.copy()
        if text in "12345" and len(text) == 1:
            return [14 + int(text)]
        if text == "\nUser rating: ?\n":
            return self.before[:-1] + [21]
        for rating in range(1, 6):
            if text == f"\nUser rating: {rating}\n":
                return self.before + [14 + rating] + self.after
        return super().encode(text, add_special_tokens=add_special_tokens)


def feedback_model(variant="V1", tokenizer_b=None, tokenizer_a=None):
    return GroupRanker(backbone(), backbone(), tokenizer_a or Tokenizer(),
                       tokenizer_b or FeedbackBPETokenizer(), variant=variant,
                       m=1, a_readout="last_token", shared_candidate_header=True,
                       score_readout="candidate", max_reader_packed_tokens=4096,
                       max_encoder_packed_tokens=4096)


def backbone(width=16):
    cfg = Qwen3Config(vocab_size=256, hidden_size=width, intermediate_size=width*2,
                     num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                     head_dim=width//2, max_position_embeddings=8192, attention_dropout=0.0,
                     use_cache=False)
    cfg._attn_implementation = "sdpa"
    return Qwen3Model(cfg)


def user():
    return next(f"ranking-model-{i}" for i in range(100) if user_split(f"ranking-model-{i}") == "train")


def query(name="q", history=2, candidates=3):
    events = tuple(HistoryEntry(f"h{i}", i % 5 + 1, i+1) for i in range(history))
    ids = tuple(f"c{i}" for i in range(candidates))
    return RecommendationQuery(name, user(), "train", events, ids, 0, ids[0], history+1,
                               tuple(h.item_id for h in events))


def items(history=10, candidates=8):
    return {**{f"h{i}": "history " + chr(65+i) for i in range(history)},
            **{f"c{i}": "candidate " + chr(65+i) for i in range(candidates)}}


def varied_cuda_inputs():
    # Distinct token IDs and lengths avoid the almost identical candidates in
    # the FP32 fixture. No candidate target is included in these input texts.
    rng = random.Random(3107)
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    content = {item: "".join(rng.choice(alphabet) for _ in range(length))
               for item, length in zip(("h0", "h1", "h2", "c0", "c1", "c2"),
                                       (43, 87, 127, 41, 113, 211))}
    qs = [query("first", history=3), query("second", history=3)]
    qs[1] = replace(qs[1], candidate_ids=("c2", "c0", "c1"), positive_index=1,
                    history=(replace(qs[1].history[0], rating=5), *qs[1].history[1:]))
    return qs, content


def model(variant="T", unequal=False, checkpointing=False):
    torch.manual_seed(123)
    return GroupRanker(None if variant == "T" else backbone(16), backbone(32 if unequal else 16),
                       None if variant == "T" else Tokenizer(), Tokenizer(), variant=variant,
                       m=2, resampler_layers=1, resampler_heads=2, gradient_checkpointing=checkpointing,
                       max_reader_packed_tokens=4096, max_encoder_packed_tokens=4096)


class GroupRankerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_gradients_match(self, actual, expected):
        for (name, left), (other, right) in zip(actual.named_parameters(), expected.named_parameters()):
            self.assertEqual(name, other)
            self.assertIsNotNone(left.grad, name)
            self.assertIsNotNone(right.grad, name)
            torch.testing.assert_close(left.grad, right.grad, rtol=3e-4, atol=3e-6, msg=name)

    def test_teacher_shared_history_matches_repeated_prefix_logits_loss_and_all_gradients(self):
        shared = model(checkpointing=True).train()
        repeated = copy.deepcopy(shared)
        qs, content = [query(), query("other")], items()
        packed = shared.loss(qs, content)
        groups = []
        for q in qs:
            prefix, branches = repeated._query_embeddings(q, content)
            scores = []
            for branch in branches:
                sequence = torch.cat([prefix, branch])[None]
                hidden = repeated.b(inputs_embeds=sequence, position_ids=torch.arange(sequence.shape[1])[None],
                                    attention_mask=torch.ones(sequence.shape[:2], dtype=torch.long),
                                    use_cache=False).last_hidden_state[0, -1]
                scores.append(repeated.head(hidden.float()).squeeze())
            groups.append(torch.stack(scores))
        reference = group_ranking_loss(groups, [q.positive_index for q in qs])
        for scores, other in zip(packed["scores"], groups):
            torch.testing.assert_close(scores, other, rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(packed["loss"], reference["loss"], rtol=1e-6, atol=1e-6)
        packed["loss"].backward()
        reference["loss"].backward()
        self.assert_gradients_match(shared, repeated)
        stats = packed["statistics"]
        self.assertEqual(stats["repeated_prefix_reference_tokens"], stats["prefix_tokens"] * 3)
        self.assertEqual(stats["reader_padding_tokens"], 0)

    def test_student_shared_a_matches_repeated_history_with_projection_and_all_gradients(self):
        for variant in ("V1", "V3"):
            shared = model(variant, unequal=True).train()
            q, content = query(), items()
            shared.calibrate_bridge([q], content)
            repeated = copy.deepcopy(shared)
            packed = shared.loss([q], content)
            scores = []
            for candidate in q.candidate_ids:
                single = SimpleNamespace(**{**q.to_dict(), "history": q.history,
                                            "candidate_ids": (candidate,)})
                scores.append(repeated([single], content)["scores"][0][0])
            group = torch.stack(scores)
            reference = group_ranking_loss([group], [q.positive_index])
            torch.testing.assert_close(packed["scores"][0], group, rtol=1e-5, atol=3e-6)
            packed["loss"].backward()
            reference["loss"].backward()
            self.assert_gradients_match(shared, repeated)
            self.assertEqual(packed["statistics"]["encoder_unique_items"], 5)
            self.assertEqual(packed["statistics"]["encoder_padding_tokens"], 0)

    def test_ten_history_and_eight_candidates_encode_eighteen_a_items_not_eighty_eight(self):
        student = model("V3")
        q, content = query(history=10, candidates=8), items()
        student.calibrate_bridge([q], content)
        result = student([q], content)
        self.assertEqual(result["statistics"]["encoder_unique_items"], 18)
        self.assertEqual(result["statistics"]["reader_padding_tokens"], 0)
        self.assertFalse(student.metadata()["raw_branch"])
        self.assertFalse(student.metadata()["alignment"])

    def test_v1_unknown_feedback_survives_qwen_bpe_merge_with_one_fixed_slot(self):
        tokenizer = FeedbackBPETokenizer()
        self.assertEqual(len(tokenizer.encode("\nUser rating: ?\n")), 5)
        student = feedback_model(tokenizer_b=tokenizer)
        ids = student._v1_unknown_feedback_ids
        self.assertEqual(ids, tuple(tokenizer.before + [tokenizer.unknown_id] + tokenizer.after))
        for rating in range(1, 6):
            actual = tokenizer.encode(f"\nUser rating: {rating}\n")
            self.assertEqual(len(ids), len(actual))
            self.assertEqual(ids[:-2], tuple(actual[:-2]))
            self.assertEqual(ids[-1], actual[-1])
            self.assertNotEqual(ids[-2], actual[-2])

    def test_v1_complete_b_prefix_is_rating_independent_and_position_matched_to_v3(self):
        v1, v3 = feedback_model(), feedback_model("V3")
        v3.b.load_state_dict(v1.b.state_dict())
        q, content = query(history=10), items()
        states = [torch.randn((1, 16)) for _ in range(13)]
        refs = (list(range(10)), list(range(10, 13)))
        initial, initial_branches = v1._query_embeddings(q, content, states, refs)
        expected_lengths = []
        for rating in range(1, 6):
            changed = replace(q, history=tuple(replace(h, rating=rating) for h in q.history))
            prefix, branches = v1._query_embeddings(changed, content, states, refs)
            torch.testing.assert_close(prefix, initial, rtol=0, atol=0)
            for actual, expected in zip(branches, initial_branches):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            explicit, _ = v3._query_embeddings(changed, content, states, refs)
            self.assertEqual(len(prefix), len(explicit))
            # Full construction proves unchanged native V3 feedback and every
            # history/candidate position, not only isolated feedback lengths.
            parts, cursor = [v3._native(INSTRUCTION)], len(v3._native(INSTRUCTION))
            for index, event in enumerate(changed.history):
                header = v3._native(f"History {index+1}:\nItem: ")
                feedback = v3._native(f"\nUser rating: {rating}\n")
                parts.extend((header, states[index], feedback))
                slot = cursor + len(header) + 1 + len(FeedbackBPETokenizer.before)
                torch.testing.assert_close(prefix[cursor:slot], explicit[cursor:slot], rtol=0, atol=0)
                torch.testing.assert_close(prefix[slot+1:slot+2], explicit[slot+1:slot+2], rtol=0, atol=0)
                cursor += len(header) + 1 + len(feedback)
            parts.append(v3._native(CANDIDATE_HEADER))
            torch.testing.assert_close(explicit, torch.cat(parts), rtol=0, atol=0)
            expected_lengths.append(len(explicit))
        self.assertEqual(len(set(expected_lengths)), 1)

    def test_v1_a_keeps_all_five_rating_values_while_candidates_are_unrated(self):
        class RecordingTokenizer(Tokenizer):
            def __init__(self):
                self.inputs = []

            def encode(self, text, add_special_tokens=False):
                self.inputs.append(text)
                return super().encode(text, add_special_tokens=add_special_tokens)

        tokenizer_a = RecordingTokenizer()
        student = feedback_model(tokenizer_a=tokenizer_a)
        q, content = query(history=1), items()
        rated = [replace(q, history=(replace(q.history[0], rating=r),)) for r in range(1, 6)]
        student.calibrate_bridge(rated, content)
        _, refs, stats = student._a_blocks(rated, content)
        self.assertEqual(len({r[0][0] for r in refs}), 5)
        self.assertEqual(stats["encoder_unique_items"], 5 + len(q.candidate_ids))
        for rating in range(1, 6):
            self.assertIn(f"User rating: {rating}\nItem: {content['h0']}", tokenizer_a.inputs)
        for candidate in q.candidate_ids:
            self.assertIn("Item: " + content[candidate], tokenizer_a.inputs)
        self.assertFalse(any("User rating:" in text and "candidate" in text for text in tokenizer_a.inputs))

    def test_v1_feedback_validation_checks_every_rating_and_rejects_position_mismatch(self):
        class BadFifthRating(FeedbackBPETokenizer):
            def encode(self, text, add_special_tokens=False):
                result = super().encode(text, add_special_tokens=add_special_tokens)
                if text == "\nUser rating: 5\n":
                    result[0] = 22  # Same length is insufficient: prefix shifted.
                return result

        class MultiTokenUnknown(FeedbackBPETokenizer):
            def encode(self, text, add_special_tokens=False):
                return [20, 22] if text == "?" else super().encode(text, add_special_tokens=add_special_tokens)

        with self.assertRaisesRegex(ValueError, "same B token length and positions"):
            feedback_model(tokenizer_b=BadFifthRating())
        with self.assertRaisesRegex(ValueError, "single fixed unknown B token"):
            feedback_model(tokenizer_b=MultiTokenUnknown())

    def test_only_v1_template_signature_binds_unknown_tokenization_and_ids(self):
        class OtherUnknownToken(FeedbackBPETokenizer):
            unknown_id = 23

        first, other = feedback_model(), feedback_model(tokenizer_b=OtherUnknownToken())
        template = first.metadata()["student_template"]
        control = template["b_unknown_feedback"]
        self.assertEqual(control["text"], "\nUser rating: ?\n")
        self.assertEqual(control["segments"], list(V1_UNKNOWN_FEEDBACK_SEGMENTS))
        self.assertEqual(control["token_ids"], list(first._v1_unknown_feedback_ids))
        self.assertNotEqual(template["sha256"], other.metadata()["student_template"]["sha256"])
        v3 = feedback_model("V3")
        other_v3 = feedback_model("V3", tokenizer_b=OtherUnknownToken())
        self.assertNotIn("b_unknown_feedback", v3.metadata()["student_template"])
        self.assertEqual(v3.metadata()["student_template"], other_v3.metadata()["student_template"])

    def test_candidate_and_query_isolation(self):
        teacher = model().eval()
        q, content = query(), items()
        first = teacher([q], content)["scores"][0].detach()
        changed = dict(content, c1="completely different candidate data")
        other = teacher([q, query("second")], changed)["scores"][0].detach()
        torch.testing.assert_close(first[[0, 2]], other[[0, 2]], rtol=1e-5, atol=2e-6)
        changed_q = replace(q, candidate_ids=tuple(reversed(q.candidate_ids)), positive_index=2)
        reordered = teacher([changed_q], content)["scores"][0]
        torch.testing.assert_close(first, reordered.flip(0), rtol=1e-5, atol=2e-6)

    def test_hash_binds_feedback_content_and_candidate_order_without_target_rating(self):
        q, content = query(), items()
        before = query_input_hash(q, content)
        changed = replace(q, history=(replace(q.history[0], rating=5), *q.history[1:]))
        self.assertNotEqual(before, query_input_hash(changed, content))
        self.assertNotEqual(before, query_input_hash(q, dict(content, c0="new content")))
        self.assertNotEqual(before, query_input_hash(replace(q, candidate_ids=q.candidate_ids[::-1], positive_index=2), content))

    def test_no_detached_or_cross_update_encoder_cache(self):
        student, q, content = model("V3"), query(), items()
        student.calibrate_bridge([q], content)
        initial = student([q], content)["scores"][0].detach().clone()
        with torch.no_grad():
            student.a.get_input_embeddings().weight.add_(torch.randn_like(student.a.get_input_embeddings().weight)*.01)
        later = student.loss([q], content)
        self.assertFalse(torch.equal(initial, later["scores"][0]))
        later["loss"].backward()
        self.assertIsNotNone(student.a.get_input_embeddings().weight.grad)

    def test_limits_do_not_silently_recompute_or_truncate_history(self):
        teacher = model()
        teacher.max_reader_packed_tokens = 1
        with self.assertRaisesRegex(ValueError, "atomic item/query"):
            teacher([query()], items())
        teacher = model()
        teacher.dense_reference_limit = 1
        with self.assertRaisesRegex(ValueError, "dense reference mask"):
            teacher([query()], items())
        with self.assertRaisesRegex(ValueError, "teacher trains"):
            teacher.loss([query()], items(), teacher_scores=[torch.zeros(3)], beta=.5)

    def test_kd_uses_complete_group_raw_scores_temperature_and_detached_teacher(self):
        scores = [torch.tensor([.2, 2., -1.], requires_grad=True)]
        teacher = [torch.tensor([2., -1., .5], requires_grad=True)]
        out = group_ranking_loss(scores, [0], teacher_scores=teacher, beta=.9, temperature=4.)
        out["loss"].backward()
        self.assertIsNone(teacher[0].grad)
        self.assertTrue(torch.isfinite(scores[0].grad).all())
        for bad in (-1., 2., True, float("nan")):
            with self.assertRaises(ValueError):
                group_ranking_loss(scores, [0], beta=bad)
        with self.assertRaises(ValueError):
            group_ranking_loss(scores, [None])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for BF16 FlexAttention training acceptance")
    def test_cuda_bf16_checkpointed_teacher_and_students_match_sdpa(self):
        # HF shares one compiled FlexAttention callable. Compile incompatible
        # A/B shapes in separate processes, so an unsuccessful compilation also
        # cannot poison the CPU tests that follow this acceptance gate.
        child_variant = os.environ.get("MIXLM_NEXT_ITEM_CUDA_GATE_VARIANT")
        if child_variant is not None:
            self.assertIn(child_variant, ("T", "V1", "V3"))
            self._assert_cuda_bf16_variant(child_variant)
            print(f"MIXLM_CUDA_GATE_COMPLETED={child_variant}", flush=True)
            return
        for variant in ("T", "V1", "V3"):
            with self.subTest(variant=variant):
                env = dict(os.environ, MIXLM_NEXT_ITEM_CUDA_GATE_VARIANT=variant)
                source = str(Path(__file__).resolve().parents[1] / "src")
                env["PYTHONPATH"] = os.pathsep.join(filter(None, (source, env.get("PYTHONPATH"))))
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()),
                     "GroupRankerTests.test_cuda_bf16_checkpointed_teacher_and_students_match_sdpa"],
                    env=env, capture_output=True, text=True, timeout=600,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(f"MIXLM_CUDA_GATE_COMPLETED={variant}", result.stdout,
                              "CUDA child did not execute its acceptance gate:\n" + result.stdout + result.stderr)

    def _assert_cuda_finite_gradients(self, actual, expected):
        differences, magnitudes = [], []
        for (name, got), (other_name, want) in zip(actual.named_parameters(), expected.named_parameters()):
            self.assertEqual(name, other_name)
            for parameter in (got, want):
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            differences.append((got.grad.double() - want.grad.double()).square().sum())
            magnitudes.append(want.grad.double().square().sum())
        magnitude = torch.stack(magnitudes).sum()
        self.assertGreater(magnitude.item(), 0)
        return (torch.stack(differences).sum() / magnitude).sqrt().item()

    def _assert_cuda_bf16_variant(self, variant):
        # Exercises the production module/checkpointing path with BF16
        # activations and FP32 trainable parameters; no model is downloaded.
        torch.manual_seed(97)
        reference = GroupRanker(None if variant == "T" else backbone(64), backbone(64),
                                None if variant == "T" else Tokenizer(), Tokenizer(),
                                variant=variant, m=2, resampler_layers=1, resampler_heads=2,
                                gradient_checkpointing=True,
                                max_reader_packed_tokens=4096, max_encoder_packed_tokens=4096).cuda().train()
        qs, content = varied_cuda_inputs()
        reference.calibrate_bridge(qs, content)
        sparse = copy.deepcopy(reference)
        sparse.attention_backend = "flex_attention"
        for module in (sparse.a, sparse.b):
            if module is not None:
                module.set_attn_implementation("flex_attention")
        # Cloning/switching must leave the SDPA reference independent.
        for actual_module, expected_module in ((sparse.a, reference.a), (sparse.b, reference.b)):
            if actual_module is not None:
                self.assertIsNot(actual_module.config, expected_module.config)
                self.assertEqual(expected_module.config._attn_implementation, "sdpa")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = reference.loss(qs, content)
            actual = sparse.loss(qs, content)
        for got, want in zip(actual["scores"], expected["scores"]):
            torch.testing.assert_close(got, want, rtol=.03, atol=.003)
        torch.testing.assert_close(actual["loss"], expected["loss"], rtol=.03, atol=.003)
        actual["loss"].backward()
        expected["loss"].backward()
        ce_relative_error = self._assert_cuda_finite_gradients(sparse, reference)
        if variant == "T":
            self.assertLess(ce_relative_error, .06)
        self.assertEqual(actual["statistics"]["reader_padding_tokens"], 0)

        # Group CE removes the common-logit component. A random student's
        # similar compressed candidates nearly cancel its parameter gradients;
        # dividing backend roundoff by that tiny residual is ill-conditioned.
        # FP32 CE already has exact all-parameter proofs above. Here a VJP with
        # nonzero-sum readout weights measures the BF16 network Jacobian without
        # that cancellation, while preserving the original .06 acceptance bound.
        sparse.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = reference(qs, content)
            actual = sparse(qs, content)
        def readout(result):
            terms = []
            for index, scores in enumerate(result["scores"]):
                weights = scores.new_tensor([1., -.3, .2], dtype=torch.float32)
                terms.append((scores.float() * weights).sum() * (index + 1) / scores.numel())
            return torch.stack(terms).mean()
        readout(actual).backward()
        readout(expected).backward()
        self.assertLess(self._assert_cuda_finite_gradients(sparse, reference), .06)


if __name__ == "__main__":
    if os.environ.get("MIXLM_NEXT_ITEM_CUDA_GATE_VARIANT") is not None and not torch.cuda.is_available():
        raise SystemExit("CUDA acceptance child requires a visible CUDA device; skipping cannot pass this gate")
    unittest.main()
