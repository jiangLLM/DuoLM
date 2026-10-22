"""Teacher carrier boundaries and lossless multi-request score assembly."""
import asyncio
import copy
import os
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from duolm import nextbook_score_head as score
from duolm.nextbook_server import NextBookServer
from tests.test_nextbook_score_head import exported, batch, admission, head, metadata


class TeacherRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ)
        self.env.start()
        self.addCleanup(self.env.stop)
        for name in ('_CONFIG', '_RAW_CONFIG', '_AUTH_KEY', '_EXPORT_ROOT', '_INSTALLED'):
            setattr(score, name, None)
        self.addCleanup(score.clear_runtime)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.export = exported()
        self.export.update(kind='mixlm_nextbook_teacher_serving', variant='T')
        from duolm import nextbook_teacher_export
        self.verify = patch.object(nextbook_teacher_export, 'verify_nextbook_teacher_export', return_value=self.export)
        self.verify.start()
        self.addCleanup(self.verify.stop)
        self.config = score.configure_runtime(self.temp.name, variant='T')

    def test_native_vocabulary_has_no_book_extension(self):
        self.assertEqual(self.config['input_vocab_size'], 31)
        self.assertEqual(self.config['vector_count'], 0)
        self.assertNotIn('vectors_sha256', self.config)
        with self.assertRaises(ValueError):
            score._public_config(self.export)
        with self.assertRaises(ValueError):
            score.configure_runtime(self.temp.name, variant='unknown')

    def test_native_multiple_token_branches_signed_and_tamper_rejected(self):
        query, items = [1, 2], [[3, 4, 5], [6, 7]]
        value = batch(query, items)
        self.assertTrue(admission(value, query=query, items=items))
        request = NS(**{**vars(value), 'multi_item_delimiter_indices': value.multi_item_delimiter_indices[0]})
        score.validate_input_request(request, value.input_ids[0], 31)
        changed = copy.deepcopy(request)
        changed.input_ids[0][4] = 8
        with self.assertRaises(ValueError):
            score.validate_input_request(changed, changed.input_ids[0], 31)
        for bad in ([[31]], [[]], [[True]], [[-1]]):
            with self.assertRaises(ValueError):
                admission(batch(query, bad), query=query, items=bad)
        with self.assertRaises(ValueError):
            admission(batch(query, items), query=query, items=items, labels=[1])

    def test_variable_branch_endpoints_and_fp32_head(self):
        query, items = [1, 2], [[3, 4, 5], [6, 7]]
        value = batch(query, items)
        admission(value, query=query, items=items)
        ids = torch.tensor(value.input_ids[0])
        hidden = torch.arange(len(ids)*4).reshape(len(ids), 4).half() / 100
        weights = head(self.config)
        result = score.compute_mis_output(
            NS(use_fp32_lm_head=True), ids, hidden, weights,
            metadata([value.mixlm_nextbook_score], [len(ids)]),
            [torch.tensor(value.multi_item_delimiter_indices[0])], NS)
        # Read BEFORE each delimiter: history endpoint and each suffix endpoint.
        expected = score.scalar_scores(hidden[[1, 5, 8]], weights._mixlm_nextbook_scalar_weight)
        torch.testing.assert_close(result.input_token_ids_logprobs_val[0], expected, rtol=0, atol=0)
        bad = torch.tensor([2, 6, 6, 9])
        with self.assertRaises(ValueError):
            score.compute_mis_output(NS(use_fp32_lm_head=True), ids, hidden, weights,
                metadata([value.mixlm_nextbook_score], [len(ids)]), [bad], NS)


class TeacherChunkTests(unittest.IsolatedAsyncioTestCase):
    async def test_out_of_order_completion_preserves_all_candidates(self):
        def chunk(values):
            return NS(item_ids=tuple((x,) for x in values), packed_length=10,
                      score_kwargs=lambda: {'items': values})
        prepared = NS(item_ids=((1,), (2,), (3,)), input_hash='full',
                      shared_batches=lambda **kw: (chunk([1, 2]), chunk([3])))
        class Engine:
            tokenizer_manager = NS(max_req_input_len=16)
            async def async_score(self, items):
                if items[0] == 1:
                    await asyncio.sleep(.01)
                return NS(scores=[[float(x)] for x in items], nextbook_score_signature='runtime')
        service = NextBookServer(NS(compile=lambda q: prepared), Engine(),
            {'variant': 'T', 'score_mode': 'shared', 'export_signature': 'export', 'runtime_signature': 'runtime'})
        try:
            result = await service.dispatch({'id': 'r', 'kind': 'score', 'query': {}, 'manifest_signature': 'export'})
            self.assertEqual(result['scores'], [1, 2, 3])
            self.assertEqual(result['ranking'], [2, 1, 0])
            self.assertEqual(result['input_hash'], 'full')
            self.assertEqual(result['physical_requests'], 2)
            self.assertEqual(result['physical_input_tokens'], 20)
            prepared.shared_batches = lambda **kw: (chunk([1]),)
            with self.assertRaises(ValueError):
                service.compile_batches({})
        finally:
            service.pool.shutdown(wait=True)

    async def test_failure_drains_all_owned_chunks(self):
        complete = []
        class Engine:
            tokenizer_manager = NS(max_req_input_len=100)
            async def async_score(self, value):
                if value == 1:
                    raise ValueError('one failure')
                await asyncio.sleep(.01)
                complete.append(value)
                return NS(scores=[[value]], nextbook_score_signature='runtime')
        service = NextBookServer(None, Engine(), {'runtime_signature': 'runtime'}, chunk_concurrency=2)
        batches = [NS(item_ids=[[i]], score_kwargs=lambda i=i: {'value': i}) for i in (1, 2, 3)]
        try:
            with self.assertRaises(ValueError):
                await service.score_batches(batches)
            self.assertEqual(sorted(complete), [2, 3])
        finally:
            service.pool.shutdown(wait=True)


if __name__ == '__main__':
    unittest.main()
