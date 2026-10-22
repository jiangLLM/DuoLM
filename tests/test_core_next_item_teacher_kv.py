"""Real HF Qwen3 FP32 CPU regressions for optional T prefix-KV inference."""
from copy import deepcopy
import importlib.util
from types import SimpleNamespace
import unittest

import torch
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
if HAS_TRANSFORMERS:
    from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3Model

from duolm.next_item_model import GroupRanker
from duolm.next_item_teacher_kv import (
    _branch_mask, score_prefix_kv, score_teacher_query,
)

ATOL = 2e-6
RTOL = 2e-6


def fixture(count=19, *, causal_lm=False):
    torch.manual_seed(941)
    config = Qwen3Config(vocab_size=96, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=8, max_position_embeddings=1024, attention_dropout=0.0,
                        use_cache=True, rope_scaling=None, use_sliding_window=False)
    config._attn_implementation = "sdpa"
    reader = (Qwen3ForCausalLM(config) if causal_lm else Qwen3Model(config)).float().eval()
    backbone = reader.model if causal_lm else reader
    prefix = backbone.get_input_embeddings()(torch.tensor([4, 5, 6, 7, 8]))
    branches = [backbone.get_input_embeddings()(torch.tensor([10 + i % 40] * (i % 7 + 1)))
                for i in range(count)]
    head = torch.randn(1, config.hidden_size) * .07
    return reader, head, prefix, branches


@torch.inference_mode()
def full_scores(reader, head, prefix, branches):
    backbone = getattr(reader, "model", reader)
    values = []
    for branch in branches:
        sequence = torch.cat((prefix, branch))[None]
        positions = torch.arange(sequence.shape[1])[None]
        hidden = backbone(inputs_embeds=sequence, position_ids=positions, use_cache=False,
                          return_dict=True).last_hidden_state[0, -1]
        values.append(torch.nn.functional.linear(hidden, head).reshape(()))
    return torch.stack(values)


def pairs(cache):
    return [(layer.keys, layer.values) for layer in cache.layers]


@unittest.skipUnless(HAS_TRANSFORMERS, "real HF Qwen3 tests require transformers")
class KVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assert_scores(self, actual, expected):
        torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertFalse(actual.requires_grad)

    def test_fp32_real_qwen3_matches_full_sequences_across_chunks(self):
        reader, head, prefix, branches = fixture()
        expected = full_scores(reader, head, prefix, branches)
        before = {name: parameter.detach().clone() for name, parameter in reader.named_parameters()}
        for chunk_size in (1, 3, 8, 16):
            with self.subTest(chunk_size=chunk_size):
                result = score_prefix_kv(reader, head, prefix, branches, candidate_chunk_size=chunk_size)
                self.assert_scores(result["scores"], expected)
                self.assertEqual(result["statistics"]["prefix_prefills"], 1)
                self.assertEqual(result["statistics"]["candidate_chunks"], (len(branches) + chunk_size - 1) // chunk_size)
                self.assertEqual(result["logical_lengths"], tuple(len(prefix) + len(branch) for branch in branches))
        self.assertEqual(reader.config._attn_implementation, "sdpa")
        for name, parameter in reader.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]), name)
            self.assertTrue(parameter.requires_grad)
            self.assertIsNone(parameter.grad)

    def test_causal_lm_wrapper_bypasses_vocabulary_logits(self):
        reader, head, prefix, branches = fixture(5, causal_lm=True)
        calls = []
        hook = reader.lm_head.register_forward_pre_hook(lambda *args: calls.append(True))
        try:
            actual = score_prefix_kv(reader, head, prefix, branches, candidate_chunk_size=3)
            expected = full_scores(reader, head, prefix, branches)
        finally:
            hook.remove()
        self.assert_scores(actual["scores"], expected)
        self.assertFalse(calls)

    def test_prefill_once_fresh_cache_each_chunk_and_offsets(self):
        reader, head, prefix, branches = fixture(10)
        calls, snapshots = [], []
        def before(module, args, kwargs):
            cache = kwargs["past_key_values"]
            calls.append({"cache": cache, "length": cache.get_seq_length(),
                          "mask": kwargs["attention_mask"].clone(), "positions": kwargs["position_ids"].clone(),
                          "cache_position": kwargs["cache_position"].clone(),
                          "initial_pairs": [(k.clone(), v.clone()) for k, v in pairs(cache) if k is not None]})
        def after(module, args, kwargs, output):
            if len(calls) == 1:
                snapshots.extend((k.clone(), v.clone()) for k, v in pairs(output.past_key_values))
        pre_hook = reader.register_forward_pre_hook(before, with_kwargs=True)
        post_hook = reader.register_forward_hook(after, with_kwargs=True)
        try:
            result = score_prefix_kv(reader, head, prefix, branches, candidate_chunk_size=3)
        finally:
            pre_hook.remove()
            post_hook.remove()
        self.assertEqual(len(calls), 5)
        self.assertEqual(calls[0]["length"], 0)
        self.assertEqual(len({id(row["cache"]) for row in calls}), len(calls))
        self.assertEqual(calls[0]["cache"].get_seq_length(), len(prefix))
        for pair, frozen in zip(pairs(calls[0]["cache"]), snapshots):
            self.assertTrue(torch.equal(pair[0], frozen[0]))
            self.assertTrue(torch.equal(pair[1], frozen[1]))
        for row in calls[1:]:
            self.assertEqual(row["length"], len(prefix))
            self.assertEqual(row["cache_position"][0].item(), len(prefix))
            self.assertTrue(torch.equal(row["positions"], row["cache_position"][None].expand_as(row["positions"])))
            self.assertEqual(row["mask"].shape[-1], len(prefix) + row["positions"].shape[-1])
            for (keys, values), (base_key, base_value) in zip(row["initial_pairs"], snapshots):
                self.assertTrue(torch.equal(keys, base_key.expand_as(keys)))
                self.assertTrue(torch.equal(values, base_value.expand_as(values)))
                self.assertNotEqual(keys.data_ptr(), base_key.data_ptr())
        self.assert_scores(result["scores"], full_scores(reader, head, prefix, branches))

    def test_variable_lengths_padding_endpoint_and_mask(self):
        reader, head, prefix, branches = fixture(8)
        output = score_prefix_kv(reader, head, prefix, branches, candidate_chunk_size=8)
        self.assert_scores(output["scores"], full_scores(reader, head, prefix, branches))
        expected_padding = 8 * max(map(len, branches)) - sum(map(len, branches))
        self.assertEqual(output["statistics"]["reader_padding_tokens"], expected_padding)
        self.assertEqual(output["statistics"]["reader_tokens"], len(prefix) + sum(map(len, branches)) + expected_padding)
        mask = _branch_mask(torch.tensor([1, 3]), 4, 3, torch.device("cpu"))
        self.assertEqual(tuple(mask.shape), (2, 1, 3, 7))
        self.assertTrue(mask[0, 0, 0, :5].all())
        self.assertFalse(mask[0, 0, :, 5:].any())
        self.assertFalse(mask[1, 0, 0, 5:].any())
        self.assertTrue(mask[1, 0, 2].all())

    def test_repeat_calls_and_candidate_isolation(self):
        reader, head, prefix, branches = fixture(12)
        first = score_prefix_kv(reader, head, prefix, branches, candidate_chunk_size=8)
        repeated = score_prefix_kv(reader, head, prefix, branches, candidate_chunk_size=8)
        self.assertTrue(torch.equal(first["scores"], repeated["scores"]))
        changed = list(branches)
        changed[1] = reader.get_input_embeddings()(torch.tensor([71, 72, 73, 74, 75, 76, 77, 78]))
        other = score_prefix_kv(reader, head, prefix, changed, candidate_chunk_size=8)
        indices = torch.tensor([i for i in range(len(branches)) if i != 1])
        self.assert_scores(first["scores"][indices], other["scores"][indices])
        self.assertGreater(abs(first["scores"][1].item() - other["scores"][1].item()), 1e-4)

    def test_strict_fp32_nested_autocast_and_global_precision_restore(self):
        reader, head, prefix, branches = fixture(4)
        expected = full_scores(reader, head, prefix, branches)
        previous = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                    torch.backends.cudnn.allow_tf32)
        seen = []
        def check(module, args, kwargs):
            seen.append((torch.is_grad_enabled(), torch.is_autocast_enabled("cpu"),
                         torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                         torch.get_float32_matmul_precision()))
        hook = reader.register_forward_pre_hook(check, with_kwargs=True)
        try:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            caller = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                      torch.backends.cudnn.allow_tf32)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                result = score_prefix_kv(reader, head, prefix, branches)
                self.assertTrue(torch.is_autocast_enabled("cpu"))
            self.assertEqual((torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                              torch.backends.cudnn.allow_tf32), caller)
            # CUDA math flags do not apply to CPU inference. The shared guard
            # disables CPU autocast and leaves the caller's CUDA settings alone.
            self.assertTrue(all(values == (False, False, caller[1], caller[2], caller[0])
                                for values in seen), seen)
            self.assert_scores(result["scores"], expected)
            # An exception inside the precision context must restore its state.
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                score_prefix_kv(reader, head, prefix * float("nan"), branches)
            self.assertEqual((torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                              torch.backends.cudnn.allow_tf32), caller)
        finally:
            hook.remove()
            torch.set_float32_matmul_precision(previous[0])
            torch.backends.cuda.matmul.allow_tf32 = previous[1]
            torch.backends.cudnn.allow_tf32 = previous[2]

    def test_exact_group_ranker_template_including_games(self):
        class Tokenizer:
            def encode(self, value, **kwargs):
                return [ord(character) % 90 + 4 for character in value]
        for domain in ("book", "game"):
            with self.subTest(domain=domain):
                reader, head, _, _ = fixture(3)
                ranker = GroupRanker(None, reader, None, Tokenizer(), variant="T", domain=domain,
                                     attention_backend="sdpa", max_logical_tokens=1024).eval()
                ranker.head.weight.data.copy_(head)
                query = SimpleNamespace(history=(SimpleNamespace(item_id="h", rating=1),), candidate_ids=("a", "b", "c"))
                items = {"h": "Historic product", "a": "A", "b": "B longer text", "c": "C"}
                with torch.inference_mode():
                    prefix, branches = ranker._query_embeddings(query, items)
                    expected = full_scores(reader, head, prefix, branches)
                independent_reader = deepcopy(reader).eval()
                result = score_teacher_query(ranker, query, items, reader=independent_reader, candidate_chunk_size=2)
                self.assert_scores(result["scores"][0], expected)
                self.assertEqual(result["logical_lengths"], [[len(prefix) + len(branch) for branch in branches]])
                self.assertEqual(reader.config._attn_implementation, "sdpa")
                self.assertFalse(reader.config.use_cache)  # ranker's declaration is retained.
                # History-deletion diagnostics retain the exact empty-history
                # instruction prefix, without fabricating a history entry.
                empty = SimpleNamespace(history=(), candidate_ids=query.candidate_ids)
                with torch.inference_mode():
                    empty_prefix, empty_branches = ranker._query_embeddings(empty, items)
                    empty_expected = full_scores(reader, head, empty_prefix, empty_branches)
                empty_result = score_teacher_query(ranker, empty, items, reader=independent_reader)
                self.assert_scores(empty_result["scores"][0], empty_expected)
                ranker.variant = "V3"
                with self.assertRaisesRegex(ValueError, "eval T"):
                    score_teacher_query(ranker, query, items, reader=independent_reader)

    def test_malformed_inputs_fail_before_prefill(self):
        reader, head, prefix, branches = fixture(2)
        cases = [dict(prefix=prefix[:0]), dict(prefix=prefix.half()), dict(branches=[]),
                 dict(branches=[branches[0][:0]]), dict(branches=[torch.randn(2, 31)]),
                 dict(head=head.half()), dict(head=torch.zeros(2, 32)), dict(candidate_chunk_size=True),
                 dict(candidate_chunk_size=0), dict(candidate_chunk_size=17), dict(max_logical_tokens=1)]
        calls = []
        hook = reader.register_forward_pre_hook(lambda *args: calls.append(True))
        try:
            for change in cases:
                with self.subTest(change=tuple(change)), self.assertRaises(ValueError):
                    score_prefix_kv(**{**dict(reader=reader, head=head, prefix=prefix, branches=branches), **change})
            self.assertFalse(calls)
        finally:
            hook.remove()
        reader.train()
        with self.assertRaisesRegex(ValueError, "eval"):
            score_prefix_kv(reader, head, prefix, branches)
        reader.eval()
        reader.config._attn_implementation = "flex_attention"
        with self.assertRaisesRegex(ValueError, "independent eval SDPA"):
            score_prefix_kv(reader, head, prefix, branches)
        self.assertEqual(reader.config._attn_implementation, "flex_attention")
        reader.config._attn_implementation = "sdpa"
        reader._orig_mod = SimpleNamespace()
        with self.assertRaisesRegex(ValueError, "compiled"):
            score_prefix_kv(reader, head, prefix, branches)
        del reader._orig_mod
        reader.half()
        with self.assertRaisesRegex(ValueError, "original FP32"):
            score_prefix_kv(reader, head, prefix, branches)


if __name__ == "__main__":
    unittest.main()
