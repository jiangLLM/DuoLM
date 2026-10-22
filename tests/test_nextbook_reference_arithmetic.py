"""CPU arithmetic and direct-hook contracts; GPU serving parity is separate."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from duolm import nextbook_reference_arithmetic as repair
from duolm import nextbook_score_head as scalar


def runtime():
    value = repair._Runtime.__new__(repair._Runtime)
    value.owner = (os.getpid(), threading.get_ident())
    value.current = None
    value.signature = "test-signature"
    value.device = torch.device("cpu")
    value.is_capture = lambda: False
    value.library = None
    value.registry = NS(is_batch_invariant_mode_enabled=lambda: True, _batch_invariant_LIB=object())
    return value


def request():
    ids = torch.zeros(203, dtype=torch.int64)
    positions = torch.arange(203)
    batch = NS(mixlm_nextbook_scores=["mark"], batch_size=1,
               forward_mode=NS(is_extend=lambda: True), extend_prefix_lens_cpu=[0],
               extend_seq_lens_cpu=[203], multi_item_delimiter_indices=[torch.arange(2, 203, 2)])
    return ids, positions, batch


class MathTests(unittest.TestCase):
    def setUp(self):
        self.rng = torch.Generator().manual_seed(301)

    def half(self, *shape):
        return torch.randn(shape, generator=self.rng).half()

    def test_rounding_matches_independent_torch_formulas(self):
        x, residual, weight = self.half(7, 1024), self.half(7, 1024), self.half(1024)
        for r in (None, residual):
            actual = repair.rmsnorm(x, weight, 1e-6, r)
            combined = x if r is None else x + r
            normal = combined.float() * torch.rsqrt(combined.float().square().mean(-1, keepdim=True) + 1e-6)
            normed = weight * normal.to(x.dtype)
            expected = normed if r is None else (normed, combined)
            if r is None:
                self.assertTrue(torch.equal(actual, expected))
            else:
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))
        packed = self.half(7, 6144)
        self.assertTrue(torch.equal(repair.silu_and_mul(packed), torch.nn.functional.silu(packed[:, :3072]) * packed[:, 3072:]))
        positions = torch.tensor([0, 1, 17, 8191, 17, 32767, 2])
        q, k = self.half(7, 2048), self.half(7, 1024)
        frequencies = 1.0 / (1000000.0 ** (torch.arange(0, 128, 2).float() / 128))
        angles = positions.float()[:, None] * frequencies[None, :]
        phase = torch.cat((angles, angles), dim=-1)
        cosine, sine = phase.cos().half(), phase.sin().half()
        def rotary(value, heads):
            value = value.reshape(7, heads, 128)
            rotated = torch.cat((-value[..., 64:], value[..., :64]), dim=-1)
            return (value * cosine[:, None] + rotated * sine[:, None]).reshape(7, heads * 128)
        for actual, expected in zip(repair.rope(positions, q, k, repair.inv_freq_cpu()), (rotary(q, 16), rotary(k, 8))):
            self.assertTrue(torch.equal(actual, expected))
        w = self.half(33, 1024)
        self.assertTrue(torch.equal(repair.linear(x, w), torch.nn.functional.linear(x, w)))

    def test_attention_gather_mask_and_scatter_match_independent_sequences(self):
        prefix = 3
        q, k, v = self.half(204, 2048), self.half(204, 1024), self.half(204, 1024)
        actual = repair.independent_inputs(q, k, v, prefix)
        for result, packed, heads in zip(actual, (q, k, v), (16, 8, 8)):
            for index in range(100):
                expected = torch.cat((packed[:prefix], packed[prefix + 1 + 2 * index:prefix + 2 + 2 * index]))
                self.assertTrue(torch.equal(result[index], expected.reshape(prefix + 1, heads, 128).transpose(0, 1)))
        mask = repair.explicit_mask(168, "cpu")
        self.assertTrue(torch.equal(mask, torch.ones(100, 1, 169, 169, dtype=torch.bool).tril()))
        self.assertEqual(mask.stride(), (28561, 28561, 169, 1))
        _, key, _ = actual
        self.assertTrue(torch.equal(repair.repeat_kv(key), key.repeat_interleave(2, dim=1)))
        independent = self.half(100, prefix + 1, 2048)
        independent[:, :prefix] = independent[0, :prefix].clone()
        expected = q.clone()
        expected[:prefix] = independent[0, :prefix]
        for index in range(100):
            expected[prefix + 1 + 2 * index] = independent[index, -1]
        self.assertTrue(torch.equal(repair.combine(q, independent, prefix), expected))

    def test_out_overloads_bypass_default_cpu_registry(self):
        code = '''
import torch
from duolm import nextbook_reference_arithmetic as r
x=torch.arange(24,dtype=torch.float16).view(3,8)/13
w=torch.arange(40,dtype=torch.float16).view(5,8)/17
s=x.float().pow(2)
mm=r.linear(x,w); mean=r.mean_lastdim(s)
lib=torch.library.Library("aten","IMPL","CPU")
lib.impl("mm", lambda a,b: torch.zeros((a.shape[0],b.shape[1]),dtype=a.dtype))
lib.impl("mean.dim", lambda x,dim,keepdim=False,dtype=None: torch.full((x.shape[0],1),-123,dtype=x.dtype))
assert not torch.equal(torch.mm(x,w.t()),mm)
assert not torch.equal(s.mean(-1,keepdim=True),mean)
assert torch.equal(r.linear(x,w),mm)
assert torch.equal(r.mean_lastdim(s),mean)
'''
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


class HookTests(unittest.TestCase):
    def setUp(self):
        self.rt = runtime()

    def test_unmarked_model_delegation_and_marked_exception_cleanup(self):
        ids, positions, batch = request()
        calls = []
        def original(*args, **kwargs):
            calls.append((args, kwargs, self.rt.current))
            if self.rt.current is not None:
                raise RuntimeError("reader failed")
            return "ordinary"
        self.rt.model = NS(forward=original)
        self.rt._patch_model()
        batch.mixlm_nextbook_scores = [None]
        self.assertEqual(self.rt.model.forward(ids, positions, batch), "ordinary")
        self.assertIsNone(calls[0][2])
        batch.mixlm_nextbook_scores = ["mark"]
        with patch.object(scalar, "runtime_config", return_value={"signature": self.rt.signature}), \
             patch.object(scalar, "_valid_mark", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "reader failed"):
                self.rt.model.forward(ids, positions, batch)
        self.assertIsNone(self.rt.current)
        self.assertIsNotNone(calls[-1][2])

    def test_authentication_layout_and_registry_guards(self):
        ids, positions, batch = request()
        with patch.object(scalar, "runtime_config", return_value={"signature": self.rt.signature}), \
             patch.object(scalar, "_valid_mark", return_value=False):
            with self.assertRaisesRegex(ValueError, "Unauthenticated"):
                self.rt._begin(ids, positions, batch, None, False, None)
        with patch.object(scalar, "runtime_config", return_value={"signature": self.rt.signature}), \
             patch.object(scalar, "_valid_mark", return_value=True):
            batch.extend_prefix_lens_cpu = [1]
            with self.assertRaisesRegex(ValueError, "uncached"):
                self.rt._begin(ids, positions, batch, None, False, None)
            batch.extend_prefix_lens_cpu = [0]
            batch.multi_item_delimiter_indices = [torch.arange(2, 201, 2)]
            with self.assertRaisesRegex(ValueError, "100 single-book"):
                self.rt._begin(ids, positions, batch, None, False, None)
            batch.multi_item_delimiter_indices = [torch.arange(2, 203, 2)]
            self.assertTrue(self.rt._begin(ids, positions, batch, None, False, None))
            with self.assertRaisesRegex(ValueError, "Interleaved"):
                self.rt._begin(ids, positions, batch, None, False, None)
            self.rt.current = None
            self.rt.registry._batch_invariant_LIB = object()
            with self.assertRaisesRegex(ValueError, "registry changed"):
                self.rt._begin(ids, positions, batch, None, False, None)
        self.assertIsNone(self.rt.current)

    def test_shared_linear_method_only_changes_owned_layers(self):
        layer = NS(weight=torch.ones(2, 3, dtype=torch.float16))
        foreign = NS()
        method = NS(apply=lambda layer, x, bias=None: "original")
        self.rt.linears = {id(layer): layer}
        self.rt._patch_linear(method)
        x = torch.ones(1, 3, dtype=torch.float16)
        self.assertEqual(method.apply(layer, x), "original")
        self.rt.current = {}
        self.assertEqual(method.apply(foreign, x), "original")
        self.assertTrue(torch.equal(method.apply(layer, x), torch.full((1, 2), 3, dtype=torch.float16)))
        with self.assertRaisesRegex(ValueError, "linear arguments"):
            method.apply(layer, x, torch.zeros(2))

    def test_attention_retains_original_call_and_restores_tf32_on_failure(self):
        batch = object()
        calls = []
        module = NS(forward=lambda *a, **kw: calls.append((a, kw)) or "original-output")
        self.rt._patch_attention(0, module)
        self.rt.current = {"batch": batch, "prefix": 2, "attention_calls": 0}
        before = (torch.backends.cudnn.allow_tf32, torch.backends.cudnn.enabled,
                  torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
        def replacement(q, k, v, prefix, original):
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            self.assertEqual(original, "original-output")
            raise RuntimeError("unsupported kernel")
        with patch.object(repair, "attention", replacement):
            with self.assertRaisesRegex(RuntimeError, "unsupported kernel"):
                module.forward(None, None, None, batch)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], {"save_kv_cache": True})
        self.assertEqual(before, (torch.backends.cudnn.allow_tf32, torch.backends.cudnn.enabled,
                                 torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic))
        self.assertEqual(self.rt.current["attention_calls"], 0)

    def test_shared_rope_preserves_api_and_requires_owned_positions(self):
        module = NS(forward=lambda *a: "original")
        self.rt.inv_freq = repair.inv_freq_cpu()
        self.rt._patch_rope(module)
        positions = torch.arange(2)
        q, k = torch.zeros(2, 2048, dtype=torch.float16), torch.zeros(2, 1024, dtype=torch.float16)
        self.assertEqual(module.forward(positions, q, k), "original")
        self.rt.current = {"positions": positions}
        for _ in range(28):
            self.assertTrue(torch.equal(module.forward(positions, q, k)[0], q))
        with self.assertRaisesRegex(ValueError, "position/cache"):
            module.forward(positions.clone(), q, k)
        with self.assertRaisesRegex(ValueError, "position/cache"):
            module.forward(positions, q, k, offsets=positions)

    def test_norm_actual_residual_api(self):
        norm = NS(forward=lambda *a: "original", weight=torch.ones(128, dtype=torch.float16), variance_epsilon=1e-6)
        self.rt._patch_norm("model.layers.0.self_attn.q_norm", norm)
        x = torch.ones(2, 128, dtype=torch.float16)
        self.assertEqual(norm.forward(x), "original")
        self.rt.current = {}
        self.assertTrue(torch.equal(norm.forward(x, quant_linear=NS(quant_config=None)), repair.rmsnorm(x, norm.weight, 1e-6)))
        with self.assertRaisesRegex(ValueError, "residual/quantization"):
            norm.forward(x, residual=x)


if __name__ == "__main__":
    unittest.main()
