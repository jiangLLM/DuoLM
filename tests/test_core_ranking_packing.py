"""No-download attention and autograd equivalence tests for packed training."""

import copy
import unittest
from unittest.mock import patch

import torch

from duolm.ranking_packing import PackedTreeLayout, QueryLayout, packed_segments


class LayoutTests(unittest.TestCase):
    def test_positions_scores_and_ranges(self):
        layout = PackedTreeLayout.from_queries([QueryLayout(3, (2, 4)), QueryLayout(2, (1, 2))])
        self.assertEqual(layout.token_count, 14)
        self.assertEqual(layout.prefix_ranges, ((0, 3), (9, 11)))
        self.assertEqual(layout.branch_ranges, (((3, 5), (5, 9)), ((11, 12), (12, 14))))
        self.assertEqual(layout.score_indices, ((4, 8), (11, 13)))
        self.assertEqual(layout.query_ranges, ((0, 9), (9, 14)))
        self.assertEqual(layout.position_ids.tolist(), [[0, 1, 2, 3, 4, 3, 4, 5, 6, 0, 1, 2, 2, 3]])
        self.assertEqual(layout.token_branch_ids.tolist(), [-1, -1, -1, 0, 0, 1, 1, 1, 1, -1, -1, 0, 1, 1])

    def test_every_causal_edge_is_query_and_branch_isolated(self):
        layout = PackedTreeLayout.from_queries([QueryLayout(3, (2, 4)), QueryLayout(2, (1, 2))])
        mask = layout.dense_attention_mask(torch.float64)[0, 0]
        for q in range(layout.token_count):
            for k in range(layout.token_count):
                expected = False
                for prefix, branches in zip(layout.prefix_ranges, layout.branch_ranges):
                    if prefix[0] <= q < prefix[1]:
                        expected = prefix[0] <= k <= q
                    for start, end in branches:
                        if start <= q < end:
                            expected = prefix[0] <= k < prefix[1] or start <= k <= q
                self.assertEqual(layout.causal_allowed(q, k), expected, (q, k))
                self.assertEqual(mask[q, k].item() == 0.0, expected, (q, k))
        self.assertFalse(layout.causal_allowed(-1, 0))
        self.assertFalse(layout.causal_allowed(0, layout.token_count))

    def test_segment_layout_has_no_padding_or_cross_segment_attention(self):
        layout = packed_segments([4, 1, 3])
        self.assertEqual(layout.token_count, 8)
        self.assertEqual(layout.ranges, ((0, 4), (4, 5), (5, 8)))
        self.assertEqual(layout.offsets, (0, 4, 5, 8))
        self.assertEqual(layout.position_ids.tolist(), [[0, 1, 2, 3, 0, 0, 1, 2]])
        self.assertTrue(layout.causal_allowed(7, 5))
        self.assertFalse(layout.causal_allowed(7, 4))
        self.assertFalse(layout.causal_allowed(4, 5))

    def test_lengths_and_mask_dtype_are_validated(self):
        for value in [0, -1, True, False, 1.2, "2", None]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    QueryLayout(value, (2,))
                with self.assertRaises(ValueError):
                    QueryLayout(2, (value,))
                with self.assertRaises(ValueError):
                    packed_segments([value])
        with self.assertRaises(ValueError):
            PackedTreeLayout.from_queries([])
        with self.assertRaises(TypeError):
            PackedTreeLayout.from_queries(["not a layout"])
        with self.assertRaises(ValueError):
            packed_segments([2]).dense_attention_mask(torch.int32)

    def test_sparse_block_rows_and_exact_tail_mask_without_dense_builder(self):
        layout = PackedTreeLayout.from_queries([QueryLayout(5, (3, 6)), QueryLayout(3, (2, 5))])
        with patch("torch.nn.attention.flex_attention.create_block_mask", side_effect=AssertionError("dense builder")):
            mask = layout.block_attention_mask(block_size=4)
        for q in range(28):
            for k in range(28):
                allowed = bool(mask.mask_mod(None, None, torch.tensor(q), torch.tensor(k)))
                self.assertEqual(allowed, layout.causal_allowed(q, k), (q, k))
                if allowed:
                    row = q // 4
                    blocks = mask.kv_indices[0, 0, row, :mask.kv_num_blocks[0, 0, row]].tolist()
                    self.assertIn(k // 4, blocks)

    def test_block_builder_storage_scales_with_blocks_not_tokens(self):
        layout = PackedTreeLayout.from_queries([QueryLayout(4096, (128,) * 8), QueryLayout(4096, (128,) * 8)])
        with patch.object(PackedTreeLayout, "dense_attention_mask", side_effect=AssertionError("dense fallback")):
            mask = layout.block_attention_mask()
        blocks = (layout.token_count + 127) // 128
        self.assertEqual(mask.kv_indices.numel(), blocks * blocks)
        self.assertLess(mask.kv_indices.numel() * 1000, layout.token_count * layout.token_count)


class ActualQwen3EquivalenceTests(unittest.TestCase):
    @staticmethod
    def model():
        from transformers import Qwen3Config, Qwen3Model

        torch.manual_seed(771)
        config = Qwen3Config(
            vocab_size=97, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=128, attention_dropout=0.0,
            use_cache=False,
        )
        config._attn_implementation = "sdpa"
        return Qwen3Model(config).double().train()

    def assert_same_gradients(self, packed, separate):
        for (name, actual), (other_name, expected) in zip(packed.named_parameters(), separate.named_parameters()):
            self.assertEqual(name, other_name)
            self.assertIsNotNone(actual.grad, name)
            self.assertIsNotNone(expected.grad, name)
            # Qwen3 RMSNorm performs its reduction in FP32 even with double
            # model weights, so repeated-prefix gradient sums may round once
            # per branch. The objective is normalized to keep this bound small.
            torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-6, atol=1e-7, msg=name)

    def test_shared_teacher_prefix_matches_independent_outputs_and_all_parameter_gradients(self):
        packed, separate = self.model(), None
        separate = copy.deepcopy(packed)
        cases = [([2, 3, 4, 5], [[6, 7], [8, 9, 10]]), ([11, 12], [[13], [14, 15]])]
        layout = PackedTreeLayout.from_queries(QueryLayout(len(prefix), tuple(map(len, branches))) for prefix, branches in cases)
        tokens = [token for prefix, branches in cases for token in prefix + sum(branches, [])]
        hidden = packed(
            input_ids=torch.tensor([tokens]), position_ids=layout.position_ids,
            attention_mask=layout.dense_attention_mask(torch.float64), use_cache=False,
        ).last_hidden_state
        # Use unequal branch objectives to verify summation of every branch's
        # contribution to the one shared prefix's parameter gradients.
        direction = torch.linspace(-0.8, 1.1, 32, dtype=torch.float64) / 32
        packed_loss = torch.zeros((), dtype=torch.float64)
        separate_loss = torch.zeros((), dtype=torch.float64)
        for query_index, (prefix, branches) in enumerate(cases):
            for branch_index, branch in enumerate(branches):
                reference = separate(input_ids=torch.tensor([prefix + branch]), use_cache=False).last_hidden_state
                start, end = layout.branch_ranges[query_index][branch_index]
                actual = hidden[:, start:end]
                torch.testing.assert_close(actual, reference[:, len(prefix):], rtol=2e-7, atol=2e-8)
                factor = 1 + query_index * 0.7 + branch_index * 0.3
                packed_loss = packed_loss + factor * (actual[:, -1] @ direction).square().sum()
                separate_loss = separate_loss + factor * (reference[:, -1] @ direction).square().sum()
        packed_loss.backward()
        separate_loss.backward()
        self.assert_same_gradients(packed, separate)

    def test_a_packed_independent_texts_match_separate_states_and_gradients(self):
        packed = self.model()
        separate = copy.deepcopy(packed)
        cases = [[2, 3, 4], [5], [6, 7, 8, 9, 10]]
        layout = packed_segments(map(len, cases))
        hidden = packed(
            input_ids=torch.tensor([sum(cases, [])]), position_ids=layout.position_ids,
            attention_mask=layout.dense_attention_mask(torch.float64), use_cache=False,
        ).last_hidden_state
        packed_loss = torch.zeros((), dtype=torch.float64)
        separate_loss = torch.zeros((), dtype=torch.float64)
        direction = torch.linspace(-1.0, 1.0, 32, dtype=torch.float64) / 32
        for index, (tokens, (start, end)) in enumerate(zip(cases, layout.ranges)):
            reference = separate(input_ids=torch.tensor([tokens]), use_cache=False).last_hidden_state
            actual = hidden[:, start:end]
            torch.testing.assert_close(actual, reference, rtol=2e-7, atol=2e-8)
            packed_loss = packed_loss + (index + 1) * (actual @ direction).square().sum()
            separate_loss = separate_loss + (index + 1) * (reference @ direction).square().sum()
        packed_loss.backward()
        separate_loss.backward()
        self.assert_same_gradients(packed, separate)

    def test_mutating_sibling_candidate_or_other_query_cannot_change_score(self):
        model = self.model().eval()
        layout = PackedTreeLayout.from_queries([QueryLayout(3, (2, 2)), QueryLayout(2, (2,))])
        original = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
        changed = original.clone()
        changed[:, 5:] = torch.tensor([[81, 82, 83, 84, 85, 86]])
        kwargs = dict(position_ids=layout.position_ids, attention_mask=layout.dense_attention_mask(torch.float64), use_cache=False)
        with torch.no_grad():
            before = model(input_ids=original, **kwargs).last_hidden_state[:, layout.score_indices[0][0]]
            after = model(input_ids=changed, **kwargs).last_hidden_state[:, layout.score_indices[0][0]]
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for compiled FlexAttention backward")
    def test_cuda_flex_mask_matches_sdpa_forward_and_parameter_gradients(self):
        # This is the production-backend acceptance test, separate from the
        # CPU semantic proof. No weights are downloaded. Multi-block lengths
        # exercise query boundaries, sibling branches, and partial tail tiles.
        from transformers import Qwen3Config, Qwen3Model

        torch.manual_seed(442)
        config = Qwen3Config(
            vocab_size=97, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=32, max_position_embeddings=512, attention_dropout=0.0,
            use_cache=False,
        )
        config._attn_implementation = "sdpa"
        reference = Qwen3Model(config).cuda().train()
        sparse = copy.deepcopy(reference)
        sparse.set_attn_implementation("flex_attention")
        layout = PackedTreeLayout.from_queries([QueryLayout(67, (69, 35)), QueryLayout(53, (41, 26))])
        tokens = torch.randint(2, 97, (1, layout.token_count), device="cuda")
        kwargs = dict(input_ids=tokens, position_ids=layout.position_ids.cuda(), use_cache=False)
        expected = reference(attention_mask=layout.dense_attention_mask(device="cuda"), **kwargs).last_hidden_state
        actual = sparse(attention_mask=layout.block_attention_mask(device="cuda"), **kwargs).last_hidden_state
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-5)
        indexes = torch.tensor(sum(layout.score_indices, ()), device="cuda")
        direction = torch.linspace(-1, 1, 64, device="cuda") / 64
        (expected[0, indexes] @ direction).square().sum().backward()
        (actual[0, indexes] @ direction).square().sum().backward()
        for (name, got), (_, want) in zip(sparse.named_parameters(), reference.named_parameters()):
            torch.testing.assert_close(got.grad, want.grad, rtol=3e-4, atol=2e-5, msg=name)


if __name__ == "__main__":
    unittest.main()
