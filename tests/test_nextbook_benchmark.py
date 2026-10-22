import asyncio
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from duolm.nextbook_benchmark import build_workload, numerical_check, verify_workload
from duolm.nextbook_server import NextBookServer


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WorkloadTests(unittest.TestCase):
    def test_nested_candidates_are_label_independent(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "dev.jsonl"
            row = {"query_id": "q", "user_id": "u", "history": [
                {"item_id": str(i), "rating": 1} for i in range(10)],
                "candidate_ids": ["c" + str(i) for i in range(100)], "positive_index": 12}
            p.write_text(json.dumps(row) + "\n")
            a = build_workload(p, users=1)
            row["positive_index"] = 93
            p.write_text(json.dumps(row) + "\n")
            b = build_workload(p, users=1)
            self.assertEqual(a["requests"], b["requests"])
            verify_workload(a)
            lists = [set(r["query"]["candidate_ids"]) for r in a["requests"]]
            self.assertTrue(lists[0] <= lists[1] <= lists[2])

    def test_wrong_inventory_and_mutation_fail(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "test.jsonl"
            path.write_text("")
            with self.assertRaises(ValueError):
                build_workload(path)
        with self.assertRaises(ValueError):
            verify_workload({"protocol": "wrong"})

    def test_scalar_gate_catches_rank_and_probability_errors(self):
        self.assertTrue(numerical_check([0, 2, -1], [0, 2, -1])["accepted"])
        self.assertFalse(numerical_check([2, 0, -1], [0, 2, -1])["accepted"])
        with self.assertRaises(ValueError):
            numerical_check([float("nan")], [0])

    def test_real_rotary_precision_is_preserved(self):
        import torch
        from transformers import Qwen3Config, Qwen3Model
        helper = script("build_nextbook_reference")
        model = Qwen3Model(Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=16))
        original = model.rotary_emb.inv_freq.clone()
        helper.cast_half_preserve_rotary(model)
        self.assertEqual(model.embed_tokens.weight.dtype, torch.float16)
        self.assertEqual(model.rotary_emb.inv_freq.dtype, torch.float32)
        torch.testing.assert_close(model.rotary_emb.inv_freq, original, rtol=0, atol=0)


class ServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_flush_requires_success_receipt(self):
        class Manager:
            async def flush_cache(self):
                return SimpleNamespace(success=False)
        server = NextBookServer(None, SimpleNamespace(tokenizer_manager=Manager()), {})
        try:
            with self.assertRaises(ValueError):
                await server.dispatch({"id": "f", "kind": "flush"})
            server.active = 1
            with self.assertRaises(ValueError):
                await server.dispatch({"id": "s", "kind": "stop"})
            self.assertFalse(server.stop.is_set())
        finally:
            server.pool.shutdown(wait=True)

    async def test_scalar_shape_binding_and_stable_sort(self):
        class Compiler:
            def compile(self, query):
                return SimpleNamespace(item_ids=[[100], [101]], packed_length=20,
                                       input_hash="input", score_kwargs=lambda: {})
        class Engine:
            tokenizer_manager = SimpleNamespace(max_req_input_len=100)
            async def async_score(self, **kwargs):
                return SimpleNamespace(scores=[[-2.0], [-2.0]], nextbook_score_signature="sig")
        server = NextBookServer(Compiler(), Engine(), {"export_signature": "exp", "runtime_signature": "sig"})
        request = {"id": "x", "kind": "score", "query": {}, "manifest_signature": "exp"}
        try:
            result = await server.dispatch(request)
            self.assertEqual(result["scores"], [-2, -2])
            self.assertEqual(result["ranking"], [0, 1])
            self.assertEqual(server.active, 0)
            with self.assertRaises(ValueError):
                await server.dispatch({**request, "manifest_signature": "other"})
            self.assertEqual(server.active, 0)
        finally:
            server.pool.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
