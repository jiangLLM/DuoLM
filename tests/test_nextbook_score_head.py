"""Raw scalar math, input binding and private IPC without SGLang or a GPU."""
from __future__ import annotations

import asyncio
import copy
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from duolm import nextbook_export, nextbook_score_head as score


def exported():
    return {"native_reader_config": {"model_type": "qwen3", "vocab_size": 31, "hidden_size": 4},
            "variant": "V3", "m": 1, "vector_count": 8, "manifest_signature": "a" * 64,
            "scalar_head": {"key": "head.weight", "dtype": "float32", "shape": [1, 4], "sha256": "b" * 64},
            "vectors": {"dtype": "float16", "shape": [8, 4], "sha256": "c" * 64},
            "index": {"sha256": "d" * 64}}


def batch(query=None, items=None):
    query, items = query or [1, 31, 2], items or [[32], [33]]
    packed, indices = list(query), []
    for item in items:
        indices.append(len(packed))
        packed += [0, *item]
    indices.append(len(packed))
    packed.append(0)
    return NS(sampling_params={"max_new_tokens": 0}, return_logprob=True, token_ids_logprob=[0],
              stream=False, input_ids=[packed], multi_item_delimiter_indices=[indices],
              logprob_start_len=0, mixlm_nextbook_score=None, positional_embed_overrides=None)


def admission(value, **kwargs):
    arguments = dict(query=[1, 31, 2], items=[[32], [33]], labels=[0],
                     apply_softmax=True, is_generation=True, use_mis=True, item_first=False)
    arguments.update(kwargs)
    return score.mark_score_batch(value, **arguments)


def metadata(marks, lengths=None):
    return NS(mixlm_nextbook_scores=marks, is_prefill_only=True, token_ids_logprobs=[[0] for _ in marks],
              top_logprobs_nums=[0] * len(marks), extend_seq_lens_cpu=lengths or [8] * len(marks),
              mm_input_embeds=None)


def head(config, weight=None):
    weight = weight if weight is not None else torch.tensor([[.1234567, -.9876543, 1.0007, -.004567]], dtype=torch.float32)
    return NS(_mixlm_nextbook_scalar_weight=weight,
              _mixlm_nextbook_scalar_binding=(config["signature"], score._tensor_identity(weight)))


def carrier(values, mark):
    return {"meta_info": {score.FIELD: mark, "prompt_tokens": 8, "cached_tokens": 0,
                         "input_token_ids_logprobs": [[(float(value), 0, None)] for value in values]}}


def fp64_scalar_reference(hidden, weights):
    # FP16 x FP32 products fit exactly in Python float's 53-bit significand;
    # fsum independently controls cancellation without invoking tensor matmul.
    rows, weight = hidden.tolist(), weights[0].tolist()
    values = [math.fsum(float(x) * float(w) for x, w in zip(row, weight)) for row in rows]
    return torch.tensor(values, dtype=torch.float64).reshape(-1, 1)


def fp32_rounding_bound(hidden, weights):
    # Conservative bound for width products and width-1 additions, without
    # assuming a particular CPU/CUDA reduction tree or using serving tolerance.
    operations = 2 * hidden.shape[1]
    unit_roundoff = 2 ** -24
    gamma = operations * unit_roundoff / (1 - operations * unit_roundoff)
    products = hidden.double() * weights.double()
    return gamma * products.abs().sum(dim=-1, keepdim=True)


class TruncatedMatrixMultiply(TorchDispatchMode):
    """Simulate a global aten override truncating a FP32 head's mantissa."""
    def __init__(self):
        super().__init__()
        self.calls = []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if function in (torch.ops.aten.mm.default, torch.ops.aten.addmm.default):
            self.calls.append(function)
            operands = list(args)
            right_index = 1 if function == torch.ops.aten.mm.default else 2
            right = operands[right_index].contiguous()
            # Keep FP32 exponent/range while discarding 13 fraction bits.
            operands[right_index] = right.view(torch.int32).bitwise_and(-8192).view(torch.float32)
            return function(*operands, **kwargs)
        return function(*args, **kwargs)


class TestScalarRuntime(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ)
        self.env.start()
        self.addCleanup(self.env.stop)
        for name in ("_CONFIG", "_RAW_CONFIG", "_AUTH_KEY", "_EXPORT_ROOT", "_INSTALLED"):
            setattr(score, name, None)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.export = exported()
        self.verify = patch.object(nextbook_export, "verify_nextbook_export", return_value=self.export)
        self.verify.start()
        self.addCleanup(self.verify.stop)
        self.addCleanup(score.clear_runtime)
        self.config = score.configure_runtime(self.tmp.name)

    def test_public_metadata_binds_assets_module_and_never_exposes_key(self):
        self.assertEqual(self.config["scalar_head_sha256"], self.export["scalar_head"]["sha256"])
        self.assertEqual(self.config["source_sha256"], score.source_hash())
        self.assertEqual(self.config["input_vocab_size"], 39)
        report = json.dumps(score.runtime_report())
        envelope = json.loads(os.environ[score.ENVIRONMENT])
        self.assertNotIn("admission_key", report)
        self.assertNotIn(envelope["admission_key"], report)
        self.config["carrier_token_ids"][0] = 19
        self.assertEqual(score.runtime_config()["carrier_token_ids"], [0])
        old = score.runtime_config()["signature"]
        self.export["scalar_head"]["sha256"] = "e" * 64
        self.assertNotEqual(score.configure_runtime(self.tmp.name)["signature"], old)

    def test_spawn_envelope_restores_key_and_module_binding(self):
        value = batch()
        admission(value)
        for name in ("_CONFIG", "_RAW_CONFIG", "_AUTH_KEY", "_EXPORT_ROOT"):
            setattr(score, name, None)
        self.assertEqual(score._marked(metadata([value.mixlm_nextbook_score])), [True])
        score._CONFIG = None
        with patch.object(score, "source_hash", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "module/config identity"):
                score.runtime_config()

    def test_reconfiguration_rotates_admission_key_without_changing_public_identity(self):
        value = batch()
        admission(value)
        self.assertEqual(score.configure_runtime(self.tmp.name), self.config)
        with self.assertRaisesRegex(ValueError, "signature differs"):
            score._marked(metadata([value.mixlm_nextbook_score]))
        with self.assertRaisesRegex(ValueError, "signature differs"):
            score._marked(metadata([self.config["signature"]]))

    def test_only_one_vector_book_branches_and_fixed_carrier_are_admitted(self):
        normal = batch(items=[[3], [4]])
        self.assertFalse(admission(normal, items=[[3], [4]]))
        for kwargs in ({"apply_softmax": False}, {"labels": [1]}, {"use_mis": False},
                       {"is_generation": False}, {"item_first": True}, {"items": [[32, 33], [34]]},
                       {"items": [[32], [3]]}, {"items": [[39], [33]]}, {"query": [1, -1, 2]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                admission(batch(), **kwargs)
        for field, invalid in (("sampling_params", {"max_new_tokens": 1}), ("top_logprobs_num", 1),
                               ("return_entropy", True), ("positional_embed_overrides", [[1]])):
            value = batch()
            setattr(value, field, invalid)
            with self.subTest(field=field), self.assertRaises(ValueError):
                admission(value)
        value = batch()
        value.input_ids[0][4], value.input_ids[0][6] = value.input_ids[0][6], value.input_ids[0][4]
        with self.assertRaisesRegex(ValueError, "candidate order"):
            admission(value)

    def test_signed_ids_order_survives_tokenization_and_forward_batch(self):
        value = batch()
        admission(value)
        request = NS(**{**vars(value), "multi_item_delimiter_indices": value.multi_item_delimiter_indices[0],
                        "origin_input_ids": value.input_ids[0], "prefix_indices": [], "output_ids": []})
        score.validate_input_request(request, request.origin_input_ids, 31)
        self.assertEqual(score.request_marks([request]), [value.mixlm_nextbook_score])
        changed = copy.deepcopy(request)
        changed.origin_input_ids[4], changed.origin_input_ids[6] = 33, 32
        with self.assertRaisesRegex(ValueError, "input/model signature"):
            score.request_marks([changed])
        with self.assertRaisesRegex(ValueError, "input/model signature"):
            score.validate_input_request(changed, changed.origin_input_ids, 31)
        request.prefix_indices = [0]
        with self.assertRaisesRegex(ValueError, "uncached"):
            score.request_marks([request])

    def test_raw_scalar_uses_saved_fp32_head_not_a_rounded_fp16_head(self):
        hidden = torch.tensor([[.1, .2, -.3, 4], [2, -1, .5, -3]], dtype=torch.float16)
        weights = head(self.config)._mixlm_nextbook_scalar_weight
        actual = score.scalar_scores(hidden, weights)
        expected = fp64_scalar_reference(hidden, weights)
        self.assertTrue(bool(((actual.double() - expected).abs() <= fp32_rounding_bound(hidden, weights)).all()))
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            cast_context = score.scalar_scores(hidden, weights)
        self.assertEqual(cast_context.dtype, torch.float32)
        torch.testing.assert_close(cast_context, actual, rtol=0, atol=0)
        self.assertFalse(torch.equal(actual, hidden.float() @ weights.half().float().T))
        self.assertTrue(bool((actual < 0).any()))
        for bad_hidden, bad_weight in ((hidden.float(), weights), (hidden, weights.half()),
                                       (hidden, weights.expand(2, -1))):
            with self.assertRaises(ValueError):
                score.scalar_scores(bad_hidden, bad_weight)

    def test_scalar_head_avoids_global_aten_matrix_precision_override(self):
        hidden = torch.tensor([[1, 1, 1, 1], [-1, -1, -1, -1]], dtype=torch.float16)
        weights = torch.tensor([[1.0003, -1, .5, -.25]], dtype=torch.float32)
        expected = fp64_scalar_reference(hidden, weights)
        override = TruncatedMatrixMultiply()
        with override:
            legacy = torch.nn.functional.linear(hidden.float(), weights)
            self.assertEqual(len(override.calls), 1)  # The simulation really intercepted aten::mm.
            override.calls.clear()
            actual = score.scalar_scores(hidden, weights)
            self.assertEqual(override.calls, [])
        bound = fp32_rounding_bound(hidden, weights)
        self.assertTrue(bool(((actual.double() - expected).abs() <= bound).all()))
        self.assertTrue(bool(((legacy.double() - expected).abs() > bound).all()))

    def test_scalar_fp64_oracle_covers_cancellation_and_fp32_accumulation(self):
        hidden = torch.tensor([[60000, 60000, -60000, -60000],
                               [-60000, -60000, 60000, 60000]], dtype=torch.float16)
        weights = torch.tensor([[1.0003, 1.0001, 1.0002, 1.00005]], dtype=torch.float32)
        actual = score.scalar_scores(hidden, weights)
        expected = fp64_scalar_reference(hidden, weights)
        bound = fp32_rounding_bound(hidden, weights)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(actual).all()))
        self.assertTrue(bool(((actual.double() - expected).abs() <= bound).all()))
        rounded_head = fp64_scalar_reference(hidden, weights.half().float())
        self.assertTrue(bool(((rounded_head - expected).abs() > bound).all()))

    def test_scalar_rows_batches_noncontiguous_operands_and_weights_are_preserved(self):
        generator = torch.Generator().manual_seed(17)
        storage = torch.randn(606, 2048, generator=generator).half()
        hidden = storage[::2, ::2]
        weight_storage = torch.randn(1, 2048, generator=generator)
        weights = weight_storage[:, ::2]
        self.assertFalse(hidden.is_contiguous())
        self.assertFalse(weights.is_contiguous())
        hidden_before, weights_before = hidden.clone(), weights.clone()
        hidden_version, weights_version = hidden._version, weights._version
        for rows in (0, 1, 101, 303):
            with self.subTest(rows=rows), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                actual = score.scalar_scores(hidden[:rows], weights)
                self.assertEqual(tuple(actual.shape), (rows, 1))
                self.assertEqual(actual.dtype, torch.float32)
                expected = fp64_scalar_reference(hidden[:rows], weights)
                self.assertTrue(bool(((actual.double() - expected).abs() <=
                                     fp32_rounding_bound(hidden[:rows], weights)).all()))
        combined = score.scalar_scores(hidden, weights)
        separate = torch.cat([score.scalar_scores(part, weights) for part in hidden.split(101)])
        torch.testing.assert_close(combined, separate, rtol=0, atol=0)
        torch.testing.assert_close(hidden, hidden_before, rtol=0, atol=0)
        torch.testing.assert_close(weights, weights_before, rtol=0, atol=0)
        self.assertEqual((hidden._version, weights._version), (hidden_version, weights_version))

    def test_endpoint_order_and_request_isolation_without_full_vocabulary_projection(self):
        one, two = batch(), batch(query=[1], items=[[34], [35]])
        admission(one)
        admission(two, query=[1], items=[[34], [35]])
        marks = [one.mixlm_nextbook_score, two.mixlm_nextbook_score]
        meta = metadata(marks, [8, 6])
        delimiters = [torch.tensor([3, 5, 7]), torch.tensor([1, 3, 5])]
        hidden = torch.arange(56).reshape(14, 4).half()
        bound = head(self.config)
        processor = NS(use_fp32_lm_head=True)
        output = score.compute_mis_output(processor, torch.arange(14), hidden, bound, meta, delimiters, NS)
        endpoints = hidden[[2, 4, 6, 8, 10, 12]]
        expected = fp64_scalar_reference(endpoints, bound._mixlm_nextbook_scalar_weight)
        actual = torch.cat(output.input_token_ids_logprobs_val)
        self.assertTrue(bool(((actual.double() - expected).abs() <=
                             fp32_rounding_bound(endpoints, bound._mixlm_nextbook_scalar_weight)).all()))
        self.assertEqual(meta.extend_logprob_pruned_lens_cpu, [3, 3])
        self.assertEqual(output.input_token_ids_logprobs_idx, [[[0]] * 3, [[0]] * 3])
        for index, request in enumerate((one, two)):
            decoded = score.decode_score_result(carrier(output.input_token_ids_logprobs_val[index].flatten(),
                request.mixlm_nextbook_score), request, [[32], [33]], NS)
            self.assertEqual(decoded.scores, actual[index * 3 + 1:index * 3 + 3].tolist())
            self.assertEqual(decoded.nextbook_score_signature, self.config["signature"])
        mutated = hidden.clone()
        mutated[8:] += 99
        again = score.compute_mis_output(processor, torch.arange(14), mutated, bound, meta, delimiters, NS)
        torch.testing.assert_close(again.input_token_ids_logprobs_val[0], output.input_token_ids_logprobs_val[0])
        bound._mixlm_nextbook_scalar_weight.add_(1)
        with self.assertRaisesRegex(ValueError, "binding changed"):
            score.compute_mis_output(processor, torch.arange(14), hidden, bound, meta, delimiters, NS)

    def test_unmarked_calls_decline_and_mixed_or_missing_mis_fail_closed(self):
        self.assertIsNone(score.compute_mis_output(None, None, None, None, metadata([None]), None, NS))
        score.guard_marked_forward(metadata([None]), None)
        value = batch()
        admission(value)
        with self.assertRaisesRegex(ValueError, "mixed ordinary/scalar"):
            score.compute_mis_output(None, None, None, None, metadata([value.mixlm_nextbook_score, None]), None, NS)
        with self.assertRaisesRegex(ValueError, "omitted MIS"):
            score.guard_marked_forward(metadata([value.mixlm_nextbook_score]), None)
        wrong = metadata([value.mixlm_nextbook_score])
        wrong.is_prefill_only = False
        with self.assertRaisesRegex(ValueError, "prefill-only"):
            score.guard_marked_forward(wrong, [])

    def test_response_signature_rows_and_cache_proof_are_mandatory(self):
        value = batch()
        admission(value)
        good = carrier([-7.5, 2.5, -1.25], value.mixlm_nextbook_score)
        self.assertEqual(score.decode_score_result(good, value, [[32], [33]], NS).scores, [[2.5], [-1.25]])
        other = batch()
        admission(other)
        for change in (lambda obj: obj["meta_info"].pop(score.FIELD),
                       lambda obj: obj["meta_info"].update({score.FIELD: other.mixlm_nextbook_score}),
                       lambda obj: obj["meta_info"].update(cached_tokens=1),
                       lambda obj: obj["meta_info"]["input_token_ids_logprobs"].pop(),
                       lambda obj: obj["meta_info"]["input_token_ids_logprobs"][1].__setitem__(0, (float("nan"), 0, None)),
                       lambda obj: obj["meta_info"]["input_token_ids_logprobs"][1].__setitem__(0, (1, 31, None))):
            bad = copy.deepcopy(good)
            change(bad)
            with self.assertRaises(ValueError):
                score.decode_score_result(bad, value, [[32], [33]], NS)

    def test_per_request_marks_are_unique_under_async_admission(self):
        async def run():
            async def one(index):
                value = batch()
                admission(value)
                await asyncio.sleep(0)
                return value.mixlm_nextbook_score
            return await asyncio.gather(*(one(index) for index in range(24)))
        marks = asyncio.run(run())
        self.assertEqual(len(set(marks)), 24)
        self.assertTrue(all(score._valid_mark(mark, score.runtime_config()) for mark in marks))

    def test_runtime_envelope_mutation_is_rejected(self):
        envelope = json.loads(os.environ[score.ENVIRONMENT])
        envelope["config"]["scalar_head_sha256"] = "f" * 64
        os.environ[score.ENVIRONMENT] = json.dumps(envelope)
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            score.runtime_config()
        score._CONFIG = None
        with self.assertRaisesRegex(ValueError, "signature differs"):
            score.runtime_config()

    def test_book_lookup_retains_native_weight_and_rejects_table_mutation(self):
        embedding = torch.nn.Embedding(31, 4, dtype=torch.float16)
        original = embedding.weight
        table = torch.arange(32).reshape(8, 4).half()
        score._attach_lookup(embedding, table, 31, self.config["signature"])
        ids = torch.tensor([1, 31, 2, 38], dtype=torch.long)
        torch.testing.assert_close(embedding(ids), torch.stack([original[1], table[0], original[2], table[7]]))
        self.assertIs(embedding.weight, original)
        for invalid in (torch.tensor([39]), torch.tensor([-1]), torch.tensor([32.])):
            with self.assertRaises(ValueError):
                embedding(invalid)
        table.add_(1)
        with self.assertRaisesRegex(ValueError, "table changed"):
            embedding(ids)


if __name__ == "__main__":
    unittest.main()
