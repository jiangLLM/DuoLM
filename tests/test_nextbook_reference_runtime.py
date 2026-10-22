"""Opt-in mode identity, admission and installed-worker readiness contracts."""
import copy
import os
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from duolm import nextbook_export, nextbook_score_head as score
from duolm.nextbook_server import verify_reference_worker
from tests.test_nextbook_score_head import exported, batch, admission


class ReferenceRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        for name in ("_CONFIG", "_RAW_CONFIG", "_AUTH_KEY", "_EXPORT_ROOT", "_INSTALLED"):
            setattr(score, name, None)
        self.addCleanup(score.clear_runtime)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.export = exported()
        self.export["native_reader_config"]["hidden_size"] = 1024
        self.export["scalar_head"]["shape"] = [1, 1024]
        self.export["vectors"]["shape"] = [8, 1024]
        verify = patch.object(nextbook_export, "verify_nextbook_export", return_value=self.export)
        verify.start()
        self.addCleanup(verify.stop)

    def configure(self, mode="reference_fp16"):
        return score.configure_runtime(self.temp.name, arithmetic_mode=mode)

    def test_mode_and_helper_source_are_bound_and_spawn_checked(self):
        default = self.configure("sglang")
        corrected = self.configure()
        self.assertNotEqual(default["signature"], corrected["signature"])
        self.assertEqual(default["arithmetic_mode"], "sglang")
        self.assertNotIn("arithmetic_source_sha256", default)
        self.assertTrue(corrected["arithmetic_source_sha256"])
        from duolm import nextbook_reference_arithmetic as arithmetic
        for name in ("_CONFIG", "_RAW_CONFIG", "_AUTH_KEY", "_EXPORT_ROOT"):
            setattr(score, name, None)
        with patch.object(arithmetic, "source_hashes", return_value={"changed.py": "bad"}):
            with self.assertRaisesRegex(ValueError, "arithmetic identity"):
                score.runtime_config()
        self.assertEqual(score.runtime_config(), corrected)

    def test_reference_rejects_other_models_or_modes(self):
        with self.assertRaisesRegex(ValueError, "unknown arithmetic"):
            self.configure("unrecognized")
        config = score._public_config(self.export)
        for field, value in (("variant", "T"), ("m", 8), ("hidden_size", 4096)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                score._arithmetic_config({**config, field: value}, "reference_fp16")

    def test_n100_and_single_physical_request_are_enforced_before_forward(self):
        self.configure()
        with self.assertRaisesRegex(ValueError, "exactly 100"):
            admission(batch())
        items = [[32]] * 100
        value = batch(items=items)
        self.assertTrue(admission(value, items=items))
        request = NS(**{**vars(value), "multi_item_delimiter_indices": value.multi_item_delimiter_indices[0],
                        "origin_input_ids": value.input_ids[0], "prefix_indices": [], "output_ids": []})
        self.assertEqual(score.request_marks([request]), [value.mixlm_nextbook_score])
        for requests in ([request, request], [request, NS(mixlm_nextbook_score=None)]):
            with self.assertRaisesRegex(ValueError, "one physical"):
                score.request_marks(requests)
        request.prefix_indices = [0]
        with self.assertRaisesRegex(ValueError, "uncached"):
            score.request_marks([request])

    def test_worker_readiness_requires_installed_hashes_and_effective_batch_limit(self):
        config = self.configure()
        arithmetic = {"installed": True, "mode": "reference_fp16",
                      "source_sha256": config["arithmetic_source_sha256"]}
        state = {"effective_max_running_requests_per_dp": 1,
                 "mixlm_nextbook": {**config, "installed": {"reference_arithmetic": arithmetic}}}
        identity = {"arithmetic_mode": "reference_fp16"}
        verify_reference_worker(state, identity)
        for limit in (None, 2, 32):
            with self.assertRaises(ValueError):
                verify_reference_worker({**state, "effective_max_running_requests_per_dp": limit}, identity)
        bad = copy.deepcopy(state)
        bad["mixlm_nextbook"]["installed"]["reference_arithmetic"]["source_sha256"] = {}
        with self.assertRaises(ValueError):
            verify_reference_worker(bad, identity)
        verify_reference_worker({}, {"arithmetic_mode": "sglang"})


if __name__ == "__main__":
    unittest.main()
