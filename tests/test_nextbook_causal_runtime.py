"""Independent V3 scoring: ordinary token sequences and authenticated endpoints."""
import asyncio
import ast
import copy
from dataclasses import dataclass
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from duolm import nextbook_score_head as score
from duolm import nextbook_bcg_runtime, nextbook_server, nextbook_serving_inputs, nextbook_sglang_overlay
from duolm.nextbook_server import NextBookServer
from duolm.nextbook_serving_inputs import NextBookRequest
from tests import test_nextbook_score_head as fixtures
from tests.test_nextbook_score_head import admission, batch, carrier, head, metadata
from tests.test_nextbook_sglang_overlay import SG21


def causal(query=None, item=None):
    query, item = query or [1, 31, 2], item or [32]
    ids = query + item
    request = NS(input_ids=ids, sampling_params={"max_new_tokens": 0}, return_logprob=True,
                 token_ids_logprob=[0], logprob_start_len=0, stream=False,
                 multi_item_delimiter_indices=None, token_indices_to_pool=[len(ids) - 1])
    score.mark_causal_request(request, query=query, item=item)
    return request


class CausalRuntimeTests(unittest.TestCase):
    setUp = fixtures.TestScalarRuntime.setUp

    def test_signed_plain_sequence_and_endpoint_survive_worker_transfer(self):
        request = causal()
        self.assertEqual(request.input_ids, [1, 31, 2, 32])
        self.assertEqual(request.token_indices_to_pool, [3])
        self.assertIsNone(request.multi_item_delimiter_indices)
        score.validate_input_request(request, request.input_ids, 31)
        worker = NS(**vars(request), origin_input_ids=request.input_ids, prefix_indices=[], output_ids=[])
        self.assertEqual(score.request_marks([worker]), [request.mixlm_nextbook_score])
        for field, value in (("origin_input_ids", [1, 31, 2, 33]), ("token_indices_to_pool", [2]),
                             ("multi_item_delimiter_indices", [2, 3]), ("prefix_indices", [0]),
                             ("output_ids", [0])):
            broken = copy.deepcopy(worker)
            setattr(broken, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                score.request_marks([broken])
        modified_mode = request.mixlm_nextbook_score.replace(":causal:", ":mis:")
        self.assertFalse(score._valid_mark(modified_mode, self.config))

    def test_variable_lengths_only_read_candidate_endpoints_and_preserve_rows(self):
        requests = [causal(), causal([1], [33]), causal([2, 3], [34])]
        lengths = [4, 2, 3]
        positions = [torch.tensor([length - 1]) for length in lengths]
        meta = metadata([r.mixlm_nextbook_score for r in requests], lengths)
        ids = torch.tensor([token for request in requests for token in request.input_ids])
        hidden = torch.arange(36).reshape(9, 4).half()
        weights = head(self.config)
        score.guard_marked_forward(meta, None, positions)
        output = score.compute_causal_output(NS(use_fp32_lm_head=True), ids, hidden,
                                             weights, meta, positions, NS)
        expected = score.scalar_scores(hidden[[3, 5, 8]], weights._mixlm_nextbook_scalar_weight)
        torch.testing.assert_close(torch.cat(output.input_token_ids_logprobs_val), expected, rtol=0, atol=0)
        self.assertEqual(meta.extend_logprob_pruned_lens_cpu, [1, 1, 1])
        self.assertEqual(output.input_token_ids_logprobs_idx, [[[0]], [[0]], [[0]]])
        for request, value in zip(requests, expected.flatten()):
            result = score.decode_causal_result(carrier([value], request.mixlm_nextbook_score), request, NS)
            self.assertEqual(result.scores, [[float(value)]])
            self.assertEqual(result.nextbook_score_signature, self.config["signature"])
        hidden[:3] += 100
        again = score.compute_causal_output(NS(use_fp32_lm_head=True), ids, hidden, weights, meta, positions, NS)
        torch.testing.assert_close(torch.cat(again.input_token_ids_logprobs_val), expected, rtol=0, atol=0)
        positions[0] = torch.tensor([2])
        with self.assertRaisesRegex(ValueError, "candidate endpoint"):
            score.compute_causal_output(NS(use_fp32_lm_head=True), ids, hidden, weights, meta, positions, NS)

    def test_wrong_carrier_shapes_and_attention_modes_fail_closed(self):
        request, mis = causal(), batch()
        admission(mis)
        for marks in ([request.mixlm_nextbook_score, None],
                      [request.mixlm_nextbook_score, mis.mixlm_nextbook_score]):
            with self.assertRaises(ValueError):
                score.guard_marked_forward(metadata(marks), None, [torch.tensor([3])])
        for delimiters, positions in (([torch.tensor([1, 3])], [torch.tensor([3])]), (None, None)):
            with self.assertRaises(ValueError):
                score.guard_marked_forward(metadata([request.mixlm_nextbook_score]), delimiters, positions)
        self.assertIsNone(score.compute_causal_output(None, None, None, None, metadata([None]), None, NS))
        for values in ([], [1, 2], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                score.decode_causal_result(carrier(values, request.mixlm_nextbook_score), request, NS)
        good = carrier([1], request.mixlm_nextbook_score)
        for replacement in ({"cached_tokens": 1}, {"cached_tokens": None},
                            {score.FIELD: mis.mixlm_nextbook_score},
                            {"input_token_ids_logprobs": [[(1, 7, None)]]}):
            invalid = copy.deepcopy(good)
            invalid["meta_info"].update(replacement)
            with self.assertRaises(ValueError):
                score.decode_causal_result(invalid, request, NS)

    def test_private_callback_closes_generator_on_valid_and_invalid_results(self):
        seen, closed = [], []
        async def generate(request, unused):
            seen.append(request)
            try:
                values = [2.25] if len(seen) == 1 else [2.25, 1]
                yield carrier(values, request.mixlm_nextbook_score)
            finally:
                closed.append(request)
        modules = {"sglang.srt.managers.io_struct": NS(GenerateReqInput=NS),
                   "sglang.srt.managers.tokenizer_manager_score_mixin": NS(ScoreResult=NS)}
        async def run():
            first = await score.score_causal_request(NS(generate_request=generate), [1, 31, 2], [32])
            self.assertEqual(first.scores, [[2.25]])
            with self.assertRaises(ValueError):
                await score.score_causal_request(NS(generate_request=generate), [1], [33])
        with patch.dict("sys.modules", modules):
            asyncio.run(run())
        self.assertEqual(closed, seen)
        self.assertEqual([r.input_ids for r in seen], [[1, 31, 2, 32], [1, 33]])
        self.assertTrue(all(getattr(r, "multi_item_delimiter_indices", None) is None for r in seen))

    def test_server_launch_disables_mis_radix_graph_and_keeps_head_precision(self):
        options, identities = [], []
        class Engine:
            def __init__(self, **kwargs):
                options.append(kwargs)
            def shutdown(self):
                pass
        class Service:
            def __init__(self, compiler, engine, identity, **kwargs):
                identities.append(identity)
            async def serve(self, *args):
                pass
        args = NS(export=self.tmp.name, overlay="unused", variant="V3", score_mode="independent",
                  prefill_graph="disabled", arithmetic_mode="sglang", kv_tokens=32768,
                  chunk_concurrency=8, socket="unused", ready="unused")
        with patch.object(nextbook_serving_inputs, "NextBookCompiler"), \
                patch.object(nextbook_sglang_overlay, "verify_overlay", return_value={}), \
                patch.object(nextbook_sglang_overlay, "activate_overlay"), \
                patch.object(nextbook_server, "NextBookServer", Service), \
                patch.dict("sys.modules", {"sglang": NS(Engine=Engine)}):
            nextbook_server.run(args)
            args.prefill_graph = "breakable"
            with self.assertRaisesRegex(ValueError, "graphs disabled"):
                nextbook_server.run(args)
        self.assertEqual(len(options), 1)
        self.assertFalse(options[0]["enable_mis"])
        self.assertTrue(options[0]["disable_radix_cache"])
        self.assertTrue(options[0]["disable_cuda_graph"])
        self.assertTrue(options[0]["enable_fp32_lm_head"])
        self.assertEqual(options[0]["dtype"], "float16")
        self.assertEqual(options[0]["chunked_prefill_size"], -1)
        self.assertEqual(identities[0]["attention_mode"], "causal")

    def test_breakable_graph_admission_preserves_shared_only_contract(self):
        bcg = nextbook_bcg_runtime
        cfg = NS(cuda_graph_config=NS(prefill=NS(backend="breakable", bs=[4, 8])))
        args = NS(_cuda_graph_config_locked={("prefill", "backend")})
        bcg.configure_graph_admission("breakable", [4, 8])
        self.assertTrue(bcg.allow_scalar_mis_bcg(args, cfg))
        self.assertFalse(bcg.allow_scalar_mis_bcg(NS(), cfg))
        cfg.cuda_graph_config.prefill.bs = [8]
        self.assertFalse(bcg.allow_scalar_mis_bcg(args, cfg))
        shared = batch()
        admission(shared)
        self.assertTrue(bcg.authenticated_bcg_batch(metadata([shared.mixlm_nextbook_score])))
        self.assertFalse(bcg.authenticated_bcg_batch(metadata([causal().mixlm_nextbook_score])))
        bcg.configure_graph_admission("disabled", [4, 8])
        self.assertFalse(bcg.allow_scalar_mis_bcg(args, cfg))
        self.assertEqual(bcg.prefill_graph_options("disabled", [4, 8]), {})
        self.assertEqual(bcg.prefill_graph_options("breakable", [4, 8])["cuda_graph_config"]["prefill"],
                         {"backend": "breakable", "bs": [4, 8]})


class CausalServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_group_order_and_multiple_groups_are_isolated(self):
        def compile_query(query):
            return NextBookRequest(query["id"], ("a", "b", "c"), [1, 2], [[31], [32], [33]],
                                   query["id"], [3, 3, 3], 9)
        service = NextBookServer(NS(compile=compile_query), NS(tokenizer_manager=NS(max_req_input_len=5)),
            {"variant": "V3", "score_mode": "independent", "export_signature": "export", "runtime_signature": "runtime"},
            chunk_concurrency=2)
        self.addCleanup(service.pool.shutdown, wait=True)
        calls = []
        async def score_one(manager, query, item):
            calls.append((query, item))
            await asyncio.sleep(.01 if item == [31] else 0)
            return NS(scores=[[float(item[0])]], nextbook_score_signature="runtime")
        with patch.object(score, "score_causal_request", side_effect=score_one):
            results = await asyncio.gather(*(service.dispatch({"id": key, "kind": "score", "query": {"id": key},
                "manifest_signature": "export"}) for key in ("first", "second")))
        self.assertEqual(len(calls), 6)
        for key, result in zip(("first", "second"), results):
            self.assertEqual(result["scores"], [31, 32, 33])
            self.assertEqual(result["ranking"], [2, 1, 0])
            self.assertEqual(result["input_hash"], key)
            self.assertEqual(result["physical_requests"], 3)
            self.assertEqual(result["physical_input_tokens"], 9)
        self.assertEqual(service.active, 0)

    async def test_failure_and_cancellation_wait_for_all_owned_candidates(self):
        batches = NextBookRequest("q", ("a", "b"), [1], [[31], [32]], "h", [2, 2], 6).independent_batches(
            packed_token_budget=3)
        service = NextBookServer(None, NS(tokenizer_manager=NS()), {"runtime_signature": "runtime"})
        self.addCleanup(service.pool.shutdown, wait=True)
        finished = []
        async def score_one(manager, query, item):
            await asyncio.sleep(.01 if item == [32] else 0)
            finished.append(item[0])
            if item == [31]:
                raise ValueError("candidate failed")
            return NS(scores=[[32.0]], nextbook_score_signature="runtime")
        with patch.object(score, "score_causal_request", side_effect=score_one):
            with self.assertRaisesRegex(ValueError, "candidate failed"):
                await service.score_batches(batches)
            self.assertEqual(finished, [31, 32])
            finished.clear()
            pending = asyncio.create_task(service.score_batches(batches))
            await asyncio.sleep(.001)
            pending.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await pending
            self.assertEqual(finished, [31, 32])


@unittest.skipUnless(SG21.is_dir(), "SGLang 0.5.21 source package is not installed")
class PinnedCausalSourceTests(unittest.TestCase):
    def test_native_flashinfer_does_not_build_mis_metadata_when_disabled(self):
        text = (SG21 / "srt/layers/attention/flashinfer_backend.py").read_text()
        tree = ast.parse(text)
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                      and node.name == "_process_multi_item_scoring")
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                                 method], type_ignores=[])
        namespace = {"MultiItemScoringParams": lambda: "disabled"}
        exec(compile(ast.fix_missing_locations(module), "flashinfer_backend.py", "exec"), namespace)
        # No field on the batch is inspected when MIS is off, including pooled positions.
        self.assertEqual(namespace[method.name](NS(enable_mis=False), object()), "disabled")
        self.assertNotIn("token_indices_to_pool", text)

    def test_pinned_logprob_pipeline_preserves_the_single_candidate_scalar(self):
        path = SG21 / "srt/managers/scheduler_components/logprob_result_processor.py"
        tree = ast.parse(path.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[])
        namespace = {"dataclass": dataclass, "torch": torch, "MIS_DELIMITER_TOKEN_ID": 0,
                     "get_exec": lambda: NS(features=NS(enable_mis=False))}
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
        processor = namespace[cls.name](model_config=NS(vocab_size=31))
        req = NS(is_prefill_only=True, token_indices_to_pool=[3], multi_item_delimiter_indices=None,
                 logprob=NS(token_ids_logprob=[0]),
                 temp_input_token_ids_logprobs_val=[torch.tensor([[-2.75]])],
                 temp_input_token_ids_logprobs_idx=[[[0]]])
        self.assertEqual(processor.calculate_num_input_logprobs(req, 4, 0), 1)
        processor._process_input_token_ids_logprobs(req)
        self.assertEqual(req.logprob.input_token_ids_logprobs_val, [[-2.75]])
        self.assertEqual(req.logprob.input_token_ids_logprobs_idx, [[0]])
        self.assertIsNone(req.temp_input_token_ids_logprobs_val)


if __name__ == "__main__":
    unittest.main()
