"""Source receipts and real pinned score hooks; CPU only, no SGLang import."""
from __future__ import annotations

import asyncio
import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from duolm import nextbook_export, nextbook_score_head as score, nextbook_sglang_overlay as overlay


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fixtures():
    """Small compilable excerpts exercise every patch anchor without vendoring."""
    before = lambda name, index: overlay.REPLACEMENTS[name][index][0]
    qwen = "srt/models/qwen3.py"
    io = "srt/managers/io_struct.py"
    tm = "srt/managers/tokenizer_manager.py"
    score_file = "srt/managers/tokenizer_manager_score_mixin.py"
    req = "srt/managers/schedule_batch.py"
    scheduler = "srt/managers/scheduler.py"
    forward = "srt/model_executor/forward_batch_info.py"
    logits = "srt/layers/logits_processor.py"
    attention = "srt/arg_groups/attention_hook.py"
    graph = "srt/model_executor/runner/prefill_cuda_graph_runner.py"
    files = {
        qwen: "class Model:\n    def load_weights(self, weights):\n        pass\n" + before(qwen, 0) + "        return None\n",
        io: ("from __future__ import annotations\nclass Generate:\n" + before(io, 0)
             + "    def get(self):\n        return Obj(\n" + before(io, 1) + "                None),\n        )\n"
             + "class Tokenized:\n" + before(io, 2)),
        tm: ("class TM:\n    def validate(self, obj, input_ids):\n" + before(tm, 0) + "        pass\n"
             + "    def create(self, obj):\n        if True:\n            return Obj(\n" + before(tm, 1) + "            )\n"
             + "    def report(self, state):\n        if True:\n" + before(tm, 2) + "        return meta_info\n"),
        score_file: ("from __future__ import annotations\nfrom dataclasses import dataclass\n"
                    + "@dataclass\nclass ScoreResult:\n    scores: list\n    prompt_tokens: int = 0\n"
                    + before(score_file, 0)
                    + "class TM:\n    async def score_request(self, batch_request, request, query, items, label_token_ids, apply_softmax, is_generation, use_multi_item_scoring, item_first):\n"
                    + before(score_file, 1) + "        return 'upstream'\n"),
        req: ("from __future__ import annotations\nclass Req:\n    def __init__(self,\n" + before(req, 0)
              + "    ):\n" + before(req, 1)),
        scheduler: ("class Scheduler:\n    def state(self):\n        ret = {}\n" + before(scheduler, 1) + "        return ret\n"
                    + "    def dispatch(self, recv_req):\n        if True:\n            return Req(\n" + before(scheduler, 0) + "            )\n"),
        forward: ("from __future__ import annotations\nclass Forward:\n" + before(forward, 0)
                  + "    def from_batch(self, batch):\n        return Forward(\n" + before(forward, 1) + "        )\n"),
        logits: ("from __future__ import annotations\nclass Metadata:\n" + before(logits, 0)
                 + "    def from_forward(self, forward_batch):\n        return Metadata(\n" + before(logits, 1) + "        )\n"
                 + "class Processor:\n    def forward(self, input_ids, hidden_states, lm_head, logits_metadata, multi_item_delimiter_indices, token_indices_to_pool=None):\n"
                 + before(logits, 2) + before(logits, 3) + "        return 'upstream'\n"),
        attention: "def resolve(server_args, cfg):\n" + before(attention, 0),
        graph: ("class Runner:\n    def capture(self):\n" + before(graph, 0)
                + "    def replay(self):\n" + before(graph, 1) + before(graph, 2)
                + before(graph, 3) + "            pass\n"),
    }
    for name in overlay.EXPECTED_SOURCE_SHA256:
        files.setdefault(name, "# unchanged source dependency\nVALUE = 1\n")
    return {name: text.encode() for name, text in files.items()}


class TestOverlay(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "installed/sglang"
        self.source.mkdir(parents=True)
        self.sources = fixtures()
        gates = {name: digest(data) for name, data in self.sources.items()}
        gate = patch.dict(overlay.EXPECTED_SOURCE_SHA256, gates, clear=True)
        gate.start()
        self.addCleanup(gate.stop)
        for name, data in self.sources.items():
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (self.source / "__init__.py").write_text("# full test package\n")
        (self.source / "unchanged.py").write_text("VALUE = 7\n")
        self.output = self.root / "overlay"

    def test_complete_receipt_exact_originals_and_no_prefix_cache_hooks(self):
        original = overlay._inventory(self.source)
        receipt = overlay.install_overlay(self.source, self.output)
        self.assertEqual(overlay._inventory(self.source), original)
        self.assertEqual(receipt["source_files_sha256"], original)
        self.assertEqual(receipt["runtime_source_sha256"], score.source_hash())
        self.assertIs(receipt["prefix_cache"], False)
        self.assertEqual(receipt["supported_sglang_version"], "0.5.21")
        self.assertEqual(set(receipt["patched_files_sha256"]), set(overlay.REPLACEMENTS))
        self.assertNotIn("srt/mem_cache/radix_cache.py", overlay.REPLACEMENTS)
        self.assertNotIn("srt/mem_cache/registry.py", overlay.REPLACEMENTS)
        self.assertEqual((self.output / "sglang/unchanged.py").read_text(), "VALUE = 7\n")
        self.assertEqual(overlay.verify_overlay(self.output), receipt)
        for name in overlay.REPLACEMENTS:
            compile((self.output / "sglang" / name).read_text(), name, "exec")
        contents = (self.output / overlay.PATCH_FILE).read_text()
        self.assertIn("nextbook_score_head", contents)
        self.assertNotIn("sglang_score_head", contents)
        self.assertNotIn("sglang_prefix_contract", contents)

    def test_source_hash_and_unique_anchors_gate_before_writes(self):
        name = "srt/models/qwen3.py"
        (self.source / name).write_bytes(self.sources[name] + b"\n# drift\n")
        with self.assertRaisesRegex(ValueError, "source SHA"):
            overlay.install_overlay(self.source, self.output)
        self.assertFalse(self.output.exists())
        for data in (b"pass\n", self.sources[name] + self.sources[name]):
            with patch.dict(overlay.EXPECTED_SOURCE_SHA256, {name: digest(data)}):
                with self.assertRaisesRegex(ValueError, "one SGLang patch anchor"):
                    overlay.patch_source(name, data)

    def test_no_clobber_partial_package_and_symlinks_are_rejected(self):
        overlay.install_overlay(self.source, self.output)
        with self.assertRaisesRegex(ValueError, "already exists"):
            overlay.install_overlay(self.source, self.output)
        with self.assertRaisesRegex(ValueError, "outside"):
            overlay.install_overlay(self.source, self.source / "nested")
        (self.source / "__init__.py").unlink()
        with self.assertRaisesRegex(ValueError, "entire installed"):
            overlay.install_overlay(self.source, self.root / "partial")
        (self.source / "__init__.py").write_text("pass\n")
        (self.source / "link.py").symlink_to(self.source / "unchanged.py")
        with self.assertRaisesRegex(ValueError, "Symlinks"):
            overlay.install_overlay(self.source, self.root / "symlink")

    def test_file_runtime_patch_and_rewritten_manifest_mutations_fail(self):
        overlay.install_overlay(self.source, self.output)
        with patch.object(score, "source_hash", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "module identity"):
                overlay.verify_overlay(self.output)
        with patch.object(overlay, "support_source_hashes", return_value={"changed.py": "bad"}):
            with self.assertRaisesRegex(ValueError, "module identity"):
                overlay.verify_overlay(self.output)
        target = self.output / "sglang/srt/models/qwen3.py"
        target.write_text(target.read_text() + "\n# drift\n")
        with self.assertRaisesRegex(ValueError, "source files changed"):
            overlay.verify_overlay(self.output)
        # Even rewriting the manifest hash cannot bypass a fixed original gate.
        manifest = json.loads((self.output / overlay.MANIFEST).read_text())
        manifest["overlay_files_sha256"]["srt/models/qwen3.py"] = digest(target.read_bytes())
        manifest["patched_files_sha256"]["srt/models/qwen3.py"] = digest(target.read_bytes())
        (self.output / overlay.MANIFEST).write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Original source gate"):
            overlay.verify_overlay(self.output)

    def test_copy_race_cannot_produce_completion_receipt(self):
        original_copy = overlay.shutil.copytree
        def race(*args, **kwargs):
            result = original_copy(*args, **kwargs)
            (self.source / "unchanged.py").write_text("VALUE = 9\n")
            return result
        with patch.object(overlay.shutil, "copytree", side_effect=race):
            with self.assertRaisesRegex(ValueError, "changed during"):
                overlay.install_overlay(self.source, self.output)
        self.assertFalse((self.output / overlay.MANIFEST).exists())

    def test_activation_is_spawn_inherited_and_refuses_late_import(self):
        overlay.install_overlay(self.source, self.output)
        with patch.object(overlay.sys, "path", ["keep"]), patch.dict(os.environ, {"PYTHONPATH": "other"}):
            overlay.activate_overlay(self.output)
            self.assertEqual(overlay.sys.path[0], str(self.output.resolve()))
            self.assertEqual(os.environ["PYTHONPATH"], str(self.output.resolve()) + os.pathsep + "other")
        with patch.dict(overlay.sys.modules, {"sglang.srt": NS()}):
            with self.assertRaisesRegex(RuntimeError, "before importing"):
                overlay.activate_overlay(self.output)


# Optional integration hooks use a publicly installed SGLang source package.
# The normal CPU suite needs neither SGLang nor private experiment archives.
_spec = importlib.util.find_spec("sglang")
SG21 = (Path(_spec.origin).parent if _spec is not None and _spec.origin
        else Path(__file__).parent / "_sglang_not_installed")

@unittest.skipUnless(SG21.is_dir(), "SGLang 0.5.21 source package is not installed")
class TestPinnedSourceHooks(unittest.TestCase):
    def sources(self):
        result = {}
        for name, expected in overlay.EXPECTED_SOURCE_SHA256.items():
            path = SG21 / name
            data = path.read_bytes()
            self.assertEqual(digest(data), expected, name)
            result[name] = data
        return result

    def test_actual_source_files_patch_and_compile(self):
        for name, data in self.sources().items():
            compile(overlay.patch_source(name, data), name, "exec")

    def test_actual_score_request_decodes_raw_scalars_and_declines_native_calls(self):
        name = "srt/managers/tokenizer_manager_score_mixin.py"
        tree = ast.parse(overlay.patch_source(name, self.sources()[name]))
        result_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ScoreResult")
        manager = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TokenizerManagerScoreMixin")
        method = next(node for node in manager.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "score_request")
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                                 result_class, method], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {"dataclass": dataclass, "math": math, "GenerateReqInput": NS,
                     "get_exec": lambda: NS(features=NS(enable_mis=True))}
        exec(compile(module, name, "exec"), namespace)
        exported = {"native_reader_config": {"model_type": "qwen3", "vocab_size": 31, "hidden_size": 4},
                    "variant": "V3", "m": 1, "vector_count": 4, "manifest_signature": "a" * 64,
                    "scalar_head": {"key": "head.weight", "dtype": "float32", "shape": [1, 4], "sha256": "b" * 64},
                    "vectors": {"dtype": "float16", "shape": [4, 4], "sha256": "c" * 64}, "index": {"sha256": "d" * 64}}
        seen = []
        async def generate(value, request):
            seen.append(value)
            yield {"meta_info": {score.FIELD: getattr(value, score.FIELD, None), "cached_tokens": 0,
                    "input_token_ids_logprobs": [[(number, 0, None)] for number in [-9.5, 4.25, -2.75]]}}
        def build(query, items, *args):
            return None, [[*query, 0, *items[0], 0, *items[1], 0]], None, [3, 5, 7]
        manager = NS(is_generation=True, tokenizer=None, _build_token_id_inputs=build,
                     generate_request=generate, _process_multi_item_scoring_results=lambda *args: "upstream")
        async def run():
            method = namespace["score_request"]
            scalar = await method(manager, query=[1, 31, 2], items=[[32], [33]], label_token_ids=[0], apply_softmax=True)
            ordinary = await method(manager, query=[1, 2, 3], items=[[4], [5]], label_token_ids=[0], apply_softmax=True)
            return scalar, ordinary
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ), patch.object(nextbook_export, "verify_nextbook_export", return_value=exported):
            score.clear_runtime()
            config = score.configure_runtime(tmp)
            try:
                scalar, ordinary = asyncio.run(run())
            finally:
                score.clear_runtime()
        self.assertEqual(scalar.scores, [[4.25], [-2.75]])
        self.assertEqual(scalar.nextbook_score_signature, config["signature"])
        self.assertEqual(ordinary, "upstream")
        self.assertIsNone(getattr(seen[1], score.FIELD, None))


if __name__ == "__main__":
    unittest.main()
