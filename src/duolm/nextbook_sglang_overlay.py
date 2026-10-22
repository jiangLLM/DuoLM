"""Build a minimal, exact-source SGLang 0.5.21 next-book scalar overlay.

This overlay adds book-input lookup at Qwen3 weight loading and an authenticated
raw-scalar carriers for MIS and ordinary causal sequences. Reference arithmetic is an explicit runtime option;
prefix caching stays disabled. Activate before importing SGLang or workers.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

from . import nextbook_score_head as runtime
from .nextbook_reference_arithmetic import source_hashes as arithmetic_source_hashes


PROTOCOL = runtime.PROTOCOL
UPSTREAM_VERSION = '0.5.21'
UPSTREAM_COMMIT = None
MANIFEST = "mixlm_nextbook_overlay.json"
PATCH_FILE = "sglang-nextbook.patch"
RUNTIME_MODULE = "duolm.nextbook_score_head"

# Recorded from the previously validated installation, not a nearby release.
EXPECTED_SOURCE_SHA256 = {'srt/arg_groups/attention_hook.py': 'f8dbe17da1c928f356ac33d31b2c3427718b1ec52098cdcfadf40acf7292d10b',
 'srt/layers/attention/flashinfer_backend.py': '5c8baba0d14eeca4c9d4bef8674318be1697b0160f105224282cd4aa27575810',
 'srt/layers/logits_processor.py': 'cfa7d5aaf8b37078a956c2fefd61ca0f069b6a4159f97e0e618c0f0ddf0b4ea9',
 'srt/layers/vocab_parallel_embedding.py': 'a50b43809cba9ba91157dc02bf2bf60cc0761c1ad8aa5828bbac133da614c8a1',
 'srt/managers/io_struct.py': 'c1a1863e431b84836ece23a3a5550c562e34148c771811064a81c810ecd68554',
 'srt/managers/schedule_batch.py': 'b2f78c781de88c36ce27cc5d44b343420338b4da299e885c575665d6fab843ca',
 'srt/managers/scheduler.py': 'b61df5b750adc748a99ceb381b8494c199fb3f704d793da5a0217bd6bbb606c4',
 'srt/managers/scheduler_components/logprob_result_processor.py': 'abc8c77e0784a66b3cf9f78726816cadf44708e03a352fa337db897ec71a34c5',
 'srt/managers/tokenizer_manager.py': '60992c06c787a3630faa5e27b3d3f1a4530e68870a53272372ed469031a25d28',
 'srt/managers/tokenizer_manager_score_mixin.py': '37610e02eb53d5775a35f5eabd44011b3f2efe58b4a37d3cfb93561e7c7f0717',
 'srt/model_executor/forward_batch_info.py': 'ef4426b3ece1b515eeb5223d394de6a8d154d912bcf62e7ed4bae20e17064797',
 'srt/models/qwen2.py': '6b9b99ea024e7f014349264055dc1968fa55723cdd705a37f5aa7085be9ef5b7',
 'srt/models/qwen3.py': '21e42f97190a30d6c27cb7ae5438eb94ae6c3be1ee97f7a9f3b6008b38648510',
 'srt/model_executor/runner/prefill_cuda_graph_runner.py': '3f9c35d036ed8c4469c262f6cc00084987deafc9787721d5d3475e93dcafcd6f'}


REPLACEMENTS = {'srt/models/qwen3.py': [('    def get_embed_and_head(self):\n',
                          '        from duolm.nextbook_score_head import install_book_table\n'
                          '        install_book_table(self)\n'
                          '\n'
                          '    def get_embed_and_head(self):\n')],
 'srt/managers/io_struct.py': [('    # Cache namespace used to isolate otherwise-identical prefixes.\n'
                                '    cache_salt: Optional[Union[List[str], str]] = None\n',
                                '    # Authenticated private next-book scalar carrier.\n'
                                '    mixlm_nextbook_score: Optional[str] = None\n'
                                '\n'
                                '    # Cache namespace used to isolate otherwise-identical prefixes.\n'
                                '    cache_salt: Optional[Union[List[str], str]] = None\n'),
                               ('            received_time=self.received_time,\n'
                                '            multi_item_delimiter_indices=(\n',
                                '            received_time=self.received_time,\n'
                                '            mixlm_nextbook_score=self.mixlm_nextbook_score,\n'
                                '            multi_item_delimiter_indices=(\n'),
                               ('    # Cache namespace used to isolate otherwise-identical prefixes.\n'
                                '    cache_salt: Optional[str] = None\n',
                                '    mixlm_nextbook_score: Optional[str] = None\n'
                                '\n'
                                '    # Cache namespace used to isolate otherwise-identical prefixes.\n'
                                '    cache_salt: Optional[str] = None\n')],
 'srt/managers/tokenizer_manager.py': [('        # Validate input length\n',
                                        '        from duolm.nextbook_score_head import '
                                        'validate_input_request\n'
                                        '        validate_input_request(obj, input_ids, '
                                        'self.model_config.vocab_size)\n'
                                        '        # Validate input length\n'),
                                       ('                '
                                        'multi_item_delimiter_indices=obj.multi_item_delimiter_indices,\n'
                                        '                token_indices_to_pool=obj.token_indices_to_pool,\n'
                                        '                encoder_urls=obj.encoder_urls,\n',
                                        '                '
                                        'multi_item_delimiter_indices=obj.multi_item_delimiter_indices,\n'
                                        '                mixlm_nextbook_score=obj.mixlm_nextbook_score,\n'
                                        '                token_indices_to_pool=obj.token_indices_to_pool,\n'
                                        '                encoder_urls=obj.encoder_urls,\n'),
                                       ('            meta_info["input_token_ids_logprobs"] = '
                                        'state.input_token_ids_logprobs\n',
                                        '            meta_info["input_token_ids_logprobs"] = '
                                        'state.input_token_ids_logprobs\n'
                                        '            if getattr(state.obj, "mixlm_nextbook_score", None) '
                                        'is not None:\n'
                                        '                meta_info["mixlm_nextbook_score"] = '
                                        'state.obj.mixlm_nextbook_score\n')],
 'srt/managers/tokenizer_manager_score_mixin.py': [('    pooled_hidden_states: '
                                                    'Optional[PooledHiddenStateOutput] = None\n',
                                                    '    pooled_hidden_states: '
                                                    'Optional[PooledHiddenStateOutput] = None\n'
                                                    '    cached_tokens: Optional[List[int]] = None\n'
                                                    '    nextbook_score_signature: Optional[str] = None\n'),
                                                   ('        results = await '
                                                    'self.generate_request(batch_request, '
                                                    'request).__anext__()\n',
                                                    '        from duolm.nextbook_score_head '
                                                    'import mark_score_batch, decode_score_result\n'
                                                    '        mark_score_batch(batch_request, query=query, '
                                                    'items=items, labels=(request_labels if is_generation '
                                                    'else label_token_ids),\n'
                                                    '                         apply_softmax=apply_softmax, '
                                                    'is_generation=is_generation,\n'
                                                    '                         '
                                                    'use_mis=use_multi_item_scoring, '
                                                    'item_first=item_first)\n'
                                                    '        results = await '
                                                    'self.generate_request(batch_request, '
                                                    'request).__anext__()\n'
                                                    '        if getattr(batch_request, '
                                                    "'mixlm_nextbook_score', None) is not None:\n"
                                                    '            return decode_score_result(results, '
                                                    'batch_request, items, ScoreResult)\n')],
 'srt/managers/schedule_batch.py': [('        multi_item_delimiter_indices: Optional[List[int]] = None,\n',
                                     '        multi_item_delimiter_indices: Optional[List[int]] = None,\n'
                                     '        mixlm_nextbook_score: Optional[str] = None,\n'),
                                    ('        self.multi_item_delimiter_indices = '
                                     'multi_item_delimiter_indices\n',
                                     '        self.multi_item_delimiter_indices = '
                                     'multi_item_delimiter_indices\n'
                                     '        self.mixlm_nextbook_score = mixlm_nextbook_score\n')],
 'srt/managers/scheduler.py': [('                '
                                'multi_item_delimiter_indices=recv_req.multi_item_delimiter_indices,\n',
                                '                '
                                'multi_item_delimiter_indices=recv_req.multi_item_delimiter_indices,\n'
                                '                mixlm_nextbook_score=recv_req.mixlm_nextbook_score,\n'),
                               ('        ret["effective_max_running_requests_per_dp"] = '
                                'self.max_running_requests\n',
                                '        ret["effective_max_running_requests_per_dp"] = '
                                'self.max_running_requests\n'
                                '        from duolm.nextbook_score_head import '
                                'installed_metadata, runtime_report\n'
                                '        ret["mixlm_nextbook"] = runtime_report()\n'
                                '        from duolm.nextbook_upstream_provenance_sg0521 import '
                                'worker_provenance\n'
                                '        ret["mixlm_sglang_provenance"] = worker_provenance()\n'
                                '        from duolm.nextbook_bcg_runtime import graph_report\n'
                                '        ret["mixlm_nextbook_prefill_graph"] = graph_report()\n')],
 'srt/model_executor/forward_batch_info.py': [('    token_ids_logprobs: Optional[List[List[int]]] = None\n',
                                               '    token_ids_logprobs: Optional[List[List[int]]] = None\n'
                                               '    mixlm_nextbook_scores: Optional[List[Optional[str]]] = '
                                               'None\n'),
                                              ('            token_ids_logprobs=batch.token_ids_logprobs,\n',
                                               '            token_ids_logprobs=batch.token_ids_logprobs,\n'
                                               '            '
                                               "mixlm_nextbook_scores=__import__('duolm.nextbook_score_head', "
                                               "fromlist=['request_marks']).request_marks(batch.reqs),\n")],
 'srt/layers/logits_processor.py': [('    token_ids_logprobs: Optional[List[List[int]]] = None\n',
                                     '    token_ids_logprobs: Optional[List[List[int]]] = None\n'
                                     '    mixlm_nextbook_scores: Optional[List[Optional[str]]] = None\n'),
                                    ('            token_ids_logprobs=forward_batch.token_ids_logprobs,\n',
                                     '            token_ids_logprobs=forward_batch.token_ids_logprobs,\n'
                                     '            '
                                     'mixlm_nextbook_scores=forward_batch.mixlm_nextbook_scores,\n'),
                                    ('        # Setwise scoring (CausalLM): read label-token logprobs AT '
                                     'each anchor\n',
                                     '        from duolm.nextbook_score_head import '
                                     'guard_marked_forward, compute_causal_output, compute_mis_output\n'
                                     '        guard_marked_forward(logits_metadata, '
                                     'multi_item_delimiter_indices, token_indices_to_pool)\n'
                                     '        scalar_output = compute_causal_output(self, input_ids, '
                                     'hidden_states, lm_head,\n'
                                     '            logits_metadata, token_indices_to_pool, '
                                     'LogitsProcessorOutput)\n'
                                     '        if scalar_output is not None:\n'
                                     '            return scalar_output\n'
                                     '        # Setwise scoring (CausalLM): read label-token logprobs AT '
                                     'each anchor\n'),
                                    ('        if multi_item_delimiter_indices is not None and '
                                     'logits_metadata.is_prefill_only:\n'
                                     '            return self.compute_logprobs_for_multi_item_scoring(\n'
                                     '                input_ids,\n'
                                     '                hidden_states,\n'
                                     '                lm_head,\n'
                                     '                logits_metadata,\n'
                                     '                multi_item_delimiter_indices,\n'
                                     '            )\n',
                                     '        if multi_item_delimiter_indices is not None and '
                                     'logits_metadata.is_prefill_only:\n'
                                     '            scalar_output = compute_mis_output(self, input_ids, '
                                     'hidden_states, lm_head,\n'
                                     '                logits_metadata, multi_item_delimiter_indices, '
                                     'LogitsProcessorOutput)\n'
                                     '            if scalar_output is not None:\n'
                                     '                return scalar_output\n'
                                     '            return self.compute_logprobs_for_multi_item_scoring(\n'
                                     '                input_ids, hidden_states, lm_head, logits_metadata, '
                                     'multi_item_delimiter_indices)\n')],
 'srt/arg_groups/attention_hook.py': [('    declare_resolution(\n'
                                       '        server_args,\n'
                                       '        "_handle_multi_item_scoring",\n'
                                       '        cuda_graph_config=with_phase(\n'
                                       '            cfg.cuda_graph_config, Phase.PREFILL, '
                                       'backend=Backend.DISABLED\n'
                                       '        ),\n'
                                       '    )\n',
                                       '    from duolm.nextbook_bcg_runtime import '
                                       'allow_scalar_mis_bcg\n'
                                       '    if not allow_scalar_mis_bcg(server_args, cfg):\n'
                                       '        declare_resolution(\n'
                                       '            server_args,\n'
                                       '            "_handle_multi_item_scoring",\n'
                                       '            cuda_graph_config=with_phase(\n'
                                       '                cfg.cuda_graph_config, Phase.PREFILL, '
                                       'backend=Backend.DISABLED\n'
                                       '            ),\n'
                                       '        )\n')],
 'srt/model_executor/runner/prefill_cuda_graph_runner.py': [('        self.capture()\n',
                                                             '        self.capture()\n'
                                                             '        if isinstance(self.backend, '
                                                             'BreakableCudaGraphBackend):\n'
                                                             '            from '
                                                             'duolm.nextbook_bcg_runtime import '
                                                             'record_capture\n'
                                                             '            record_capture(self.backend, '
                                                             'self.capture_num_tokens)\n'),
                                                            ('        full_path = self._is_full_backend\n'
                                                             '\n'
                                                             '        def replay_layer_forward(*args, '
                                                             '**layer_kwargs):\n',
                                                             '        full_path = self._is_full_backend\n'
                                                             '        from '
                                                             'duolm.nextbook_bcg_runtime import '
                                                             'authenticated_bcg_batch, record_replay\n'
                                                             '        scalar_bcg = not full_path and '
                                                             'authenticated_bcg_batch(forward_batch)\n'
                                                             '        if scalar_bcg:\n'
                                                             '            # Native MIS metadata rewrites '
                                                             'the raw positions after registry fill.\n'
                                                             '            '
                                                             'static_forward_batch.positions[:raw_num_tokens].copy_(\n'
                                                             '                '
                                                             'forward_batch.positions[:raw_num_tokens])\n'
                                                             '\n'
                                                             '        def replay_layer_forward(*args, '
                                                             '**layer_kwargs):\n'),
                                                            ('            hs = '
                                                             'self.backend.replay(shape_key, '
                                                             'static_forward_batch, **kwargs)\n'
                                                             '            return _slice_output_rows(hs, '
                                                             'raw_num_tokens) if full_path else hs\n',
                                                             '            hs = '
                                                             'self.backend.replay(shape_key, '
                                                             'static_forward_batch, **kwargs)\n'
                                                             '            if scalar_bcg:\n'
                                                             '                record_replay(forward_batch, '
                                                             'raw_num_tokens, static_num_tokens,\n'
                                                             '                    '
                                                             'debug_eager=bool(getattr(self.backend, '
                                                             "'_debug_eager', False)))\n"
                                                             '            return _slice_output_rows(hs, '
                                                             'raw_num_tokens) if (full_path or scalar_bcg) '
                                                             'else hs\n'),
                                                            ('        if full_path:\n'
                                                             '            # Run the eager tail against the '
                                                             'raw user-facing batch so it uses\n',
                                                             '        if full_path or scalar_bcg:\n'
                                                             '            # Run the eager tail against the '
                                                             'raw user-facing batch so it uses\n')]}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def support_source_hashes():
    return {name: _sha(Path(__file__).with_name(name).read_bytes()) for name in
            ("nextbook_bcg_runtime.py", "nextbook_upstream_provenance_sg0521.py")}


def patch_source(name, data):
    if EXPECTED_SOURCE_SHA256.get(name) != _sha(data):
        raise ValueError("Unsupported SGLang source SHA: " + name)
    text = data.decode("utf-8")
    for index, (before, after) in enumerate(REPLACEMENTS.get(name, ())):
        if text.count(before) != 1:
            raise ValueError(f"Expected one SGLang patch anchor: {name}:{index}")
        text = text.replace(before, after, 1)
    compile(text, name, "exec")
    return text.encode("utf-8")


def _inventory(directory):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Overlay inventory requires a regular directory")
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlinks are not supported in an overlay: " + str(path))
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = _sha(path.read_bytes())
        elif not path.is_dir():
            raise ValueError("Nonregular overlay input: " + str(path))
    return result


def installed_package():
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        raise ValueError("SGLang is not installed; provide source_package")
    return Path(spec.origin).resolve().parent


def install_overlay(source_package, output):
    source = Path(source_package)
    if source.is_symlink():
        raise ValueError("Source package cannot be a symlink")
    source = source.resolve(strict=True)
    output = Path(output).absolute()
    if not source.is_dir() or source.name != "sglang":
        raise ValueError("source_package must identify the installed sglang directory")
    if output.exists() or output.is_symlink():
        raise ValueError("Overlay output already exists; choose a new directory")
    if output.resolve().is_relative_to(source):
        raise ValueError("Overlay output must be outside the installed package")
    original = _inventory(source)
    if "__init__.py" not in original:
        raise ValueError("Source must be the entire installed SGLang package")
    transformed, differences = {}, []
    for name, expected in EXPECTED_SOURCE_SHA256.items():
        if original.get(name) != expected:
            raise ValueError("Unsupported/missing SGLang source SHA: " + name)
        before = (source / name).read_bytes()
        after = patch_source(name, before)
        transformed[name] = after
        if before != after:
            differences.extend(difflib.unified_diff(before.decode().splitlines(keepends=True),
                after.decode().splitlines(keepends=True), fromfile="a/sglang/" + name, tofile="b/sglang/" + name))
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source, output / "sglang", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    if _inventory(source) != original or _inventory(output / "sglang") != original:
        raise ValueError("SGLang source changed during overlay copy")
    for name, data in transformed.items():
        (output / "sglang" / name).write_bytes(data)
    patch = "".join(differences).encode()
    (output / PATCH_FILE).write_bytes(patch)
    manifest = {"schema_version": 1, "protocol": PROTOCOL, "supported_sglang_version": UPSTREAM_VERSION,
                "reference_upstream_commit": UPSTREAM_COMMIT, "source_package": str(source),
                "source_files_sha256": original, "gated_source_files_sha256": dict(EXPECTED_SOURCE_SHA256),
                "overlay_files_sha256": _inventory(output / "sglang"),
                "patched_files_sha256": {name: _sha(transformed[name]) for name in REPLACEMENTS},
                "patch_sha256": _sha(patch), "builder_sha256": _sha(Path(__file__).read_bytes()),
                "runtime_module": RUNTIME_MODULE, "runtime_source_sha256": runtime.source_hash(),
                "arithmetic_source_sha256": arithmetic_source_hashes(),
                "support_source_sha256": support_source_hashes(),
                "prefix_cache": False, "carrier": "private_raw_scalar_v1",
                "limitations": ["Source installation is not GPU validation", "Reference arithmetic is opt-in",
                                "TP/PP=1, uncached prefill-only MIS or causal scalar requests"]}
    (output / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return verify_overlay(output)


def verify_overlay(output):
    output = Path(output)
    if output.is_symlink():
        raise ValueError("Overlay cannot be a symlink")
    output = output.resolve(strict=True)
    path = output / MANIFEST
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing overlay completion manifest")
    manifest = json.loads(path.read_text())
    if (manifest.get("schema_version") != 1 or manifest.get("protocol") != PROTOCOL
            or manifest.get("builder_sha256") != _sha(Path(__file__).read_bytes())
            or manifest.get("runtime_module") != RUNTIME_MODULE
            or manifest.get("runtime_source_sha256") != runtime.source_hash()
            or manifest.get("arithmetic_source_sha256") != arithmetic_source_hashes()
            or manifest.get("support_source_sha256") != support_source_hashes()
            or manifest.get("gated_source_files_sha256") != EXPECTED_SOURCE_SHA256
            or manifest.get("prefix_cache") is not False):
        raise ValueError("Overlay manifest/module identity mismatch")
    actual = _inventory(output / "sglang")
    if actual != manifest.get("overlay_files_sha256"):
        raise ValueError("Overlay source files changed")
    if set(manifest.get("patched_files_sha256", {})) != set(REPLACEMENTS):
        raise ValueError("Patched file receipt differs")
    for name, expected in EXPECTED_SOURCE_SHA256.items():
        text = (output / "sglang" / name).read_text()
        for before, after in reversed(REPLACEMENTS.get(name, ())):
            if text.count(after) != 1:
                raise ValueError("Patched anchor identity mismatch: " + name)
            text = text.replace(after, before, 1)
        if _sha(text.encode()) != expected or manifest.get("source_files_sha256", {}).get(name) != expected:
            raise ValueError("Original source gate differs: " + name)
        if name in REPLACEMENTS and actual[name] != manifest["patched_files_sha256"][name]:
            raise ValueError("Patched file identity mismatch: " + name)
    patch = output / PATCH_FILE
    if patch.is_symlink() or not patch.is_file() or _sha(patch.read_bytes()) != manifest.get("patch_sha256"):
        raise ValueError("Overlay patch receipt changed")
    return manifest


def activate_overlay(output):
    manifest = verify_overlay(output)
    if any(name == "sglang" or name.startswith("sglang.") for name in sys.modules):
        raise RuntimeError("Activate overlay before importing SGLang")
    prefix = str(Path(output).resolve())
    sys.path[:] = [prefix, *(item for item in sys.path if item != prefix)]
    existing = [item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item and item != prefix]
    os.environ["PYTHONPATH"] = os.pathsep.join([prefix, *existing])
    importlib.invalidate_caches()
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("install")
    build.add_argument("--source-package", type=Path)
    build.add_argument("--output", required=True, type=Path)
    check = sub.add_parser("verify")
    check.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = (install_overlay(args.source_package or installed_package(), args.output)
                if args.command == "install" else verify_overlay(args.output))
    print(json.dumps({"output": str(args.output), "protocol": PROTOCOL,
                      "patch_sha256": manifest["patch_sha256"], "verified": True}, sort_keys=True))


if __name__ == "__main__":
    main()
