"""Resident private scalar MIS service with timed, one-worker ID compilation."""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import socket
import time

from .nextbook_benchmark import digest


def verify_reference_worker(state, identity):
    if identity.get("arithmetic_mode", "sglang") != "reference_fp16":
        return
    runtime = state.get("mixlm_nextbook") or {}
    arithmetic = (runtime.get("installed") or {}).get("reference_arithmetic") or {}
    if (runtime.get("arithmetic_mode") != "reference_fp16"
            or arithmetic.get("installed") is not True
            or arithmetic.get("mode") != "reference_fp16"
            or not runtime.get("arithmetic_source_sha256")
            or arithmetic.get("source_sha256") != runtime.get("arithmetic_source_sha256")
            or state.get("effective_max_running_requests_per_dp") != 1):
        raise ValueError("Worker did not install bound single-request reference arithmetic")


class NextBookServer:
    def __init__(self, compiler, engine, identity, *, chunk_concurrency=8):
        if type(chunk_concurrency) is not int or chunk_concurrency < 1:
            raise ValueError("Positive chunk concurrency required")
        self.compiler, self.engine, self.identity = compiler, engine, identity
        self.chunk_concurrency = chunk_concurrency
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nextbook-compiler")
        self.stop = asyncio.Event()
        self.active = 0

    def compiler_cache_report(self):
        report = getattr(self.compiler, "token_cache_report", None)
        return report() if callable(report) else None

    def compile_batches(self, query):
        prepared = self.compiler.compile(query)
        limit = self.engine.tokenizer_manager.max_req_input_len
        if self.identity.get("variant", "V3") == "T":
            batches = (prepared.independent_batches(packed_token_budget=limit - 1) if self.identity["score_mode"] == "independent"
                       else prepared.shared_batches(packed_token_budget=limit - 1))
        elif self.identity.get("score_mode") == "independent":
            batches = prepared.independent_batches(packed_token_budget=limit - 1)
        else:
            batches = [prepared]
        if not batches or any(batch.packed_length >= limit for batch in batches):
            raise ValueError("Request exceeds the runtime's physical input limit")
        if [tuple(item) for batch in batches for item in batch.item_ids] != [tuple(item) for item in prepared.item_ids]:
            raise ValueError("Physical chunks changed candidate order or content")
        return prepared, batches

    async def score_batches(self, batches):
        semaphore = asyncio.Semaphore(self.chunk_concurrency)
        async def score_one(batch):
            async with semaphore:
                if getattr(batch, "attention_mode", "mis") == "causal":
                    from .nextbook_score_head import score_causal_request
                    result = await score_causal_request(self.engine.tokenizer_manager,
                                                        batch.query_ids, batch.item_ids[0])
                else:
                    result = await self.engine.async_score(**batch.score_kwargs())
            if (getattr(result, "nextbook_score_signature", None) != self.identity["runtime_signature"]
                    or len(result.scores) != len(batch.item_ids)
                    or any(not isinstance(row, list) or len(row) != 1 or type(row[0]) not in (int, float)
                           or not math.isfinite(row[0]) for row in result.scores)):
                raise ValueError("Wrong scalar result identity or shape")
            return [row[0] for row in result.scores]
        # Wait for all owned requests even on failure; no orphan chunks may
        # continue after the service advertises an idle/flushable state.
        pending = asyncio.gather(*(score_one(batch) for batch in batches), return_exceptions=True)
        try:
            outcomes = await asyncio.shield(pending)
        except asyncio.CancelledError:
            await asyncio.shield(pending)
            raise
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
        return [score for outcome in outcomes for score in outcome]

    async def dispatch(self, request):
        if not isinstance(request, dict) or not isinstance(request.get("id"), str):
            raise ValueError("Missing private request ID")
        kind = request.get("kind")
        if kind == "info":
            states = await self.engine.tokenizer_manager.get_internal_state()
            if not isinstance(states, list) or len(states) != 1:
                raise ValueError("Need one installed model worker")
            worker_state = {key: states[0].get(key) for key in (
                "mixlm_nextbook", "mixlm_nextbook_prefill_graph",
                "effective_max_running_requests_per_dp")}
            return {"id": request["id"], "identity": self.identity,
                    "active": self.active, "worker_state": worker_state,
                    "token_cache": self.compiler_cache_report()}
        if kind in {"flush", "stop"}:
            if self.active:
                raise ValueError("Cannot flush/stop while scoring requests remain active")
            if kind == "flush":
                result = await self.engine.tokenizer_manager.flush_cache()
                if getattr(result, "success", None) is not True:
                    raise ValueError("Engine refused cache flush")
            else:
                self.stop.set()
            return {"id": request["id"], "ok": True}
        if kind != "score" or set(request) != {"id", "kind", "query", "manifest_signature"}:
            raise ValueError("Unsupported scalar request")
        if request["manifest_signature"] != self.identity["export_signature"]:
            raise ValueError("Request model identity differs")
        self.active += 1
        try:
            start = time.perf_counter()
            prepared, batches = await asyncio.get_running_loop().run_in_executor(self.pool, self.compile_batches, request["query"])
            compiled = time.perf_counter()
            scores = await self.score_batches(batches)
            inference = time.perf_counter()
            # Private carrier terminates in the overlay; no vocabulary logprobs
            # or rating softmax is accepted by this scalar-only API.
            ranking = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
            finished = time.perf_counter()
            return {"id": request["id"], "scores": scores, "ranking": ranking,
                    "input_hash": prepared.input_hash, "manifest_signature": self.identity["export_signature"],
                    "physical_requests": len(batches),
                    "physical_input_tokens": sum(batch.packed_length for batch in batches),
                    "timings": {"compile_and_queue_seconds": compiled-start,
                                "inference_seconds": inference-compiled,
                                "sort_seconds": finished-inference}}
        finally:
            self.active -= 1

    async def client(self, reader, writer):
        try:
            while line := await reader.readline():
                request = json.loads(line)
                try:
                    result = await self.dispatch(request)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    result = {"id": request.get("id"), "error_type": type(e).__name__}
                writer.write(json.dumps(result, allow_nan=False, separators=(",", ":")).encode() + b"\n")
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def serve(self, socket_path, ready):
        socket_path, ready = Path(socket_path), Path(ready)
        if socket_path.exists() or socket_path.is_symlink() or ready.exists():
            raise ValueError("Server output already exists")
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        ready.parent.mkdir(parents=True, exist_ok=True)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server = None
        inode = None
        try:
            listener.bind(str(socket_path))
            inode = socket_path.stat().st_ino
            os.chmod(socket_path, 0o600)
            listener.listen(128)
            listener.setblocking(False)
            server = await asyncio.start_unix_server(self.client, sock=listener, limit=1 << 20)
            states = await self.engine.tokenizer_manager.get_internal_state()
            if not isinstance(states, list) or len(states) != 1:
                raise ValueError("Need one installed model worker")
            runtime = states[0].get("mixlm_nextbook")
            installed = runtime.get("installed", {}) if isinstance(runtime, dict) else {}
            if (runtime.get("signature") != self.identity["runtime_signature"]
                    or installed.get("signature") != self.identity["runtime_signature"]
                    or installed.get("installed") is not True
                    or installed.get("output_head_unchanged") is not True
                    or installed.get("scalar_head_bytes") != runtime.get("hidden_size", 0) * 4
                    or installed.get("table_bytes") != runtime.get("vector_count", 0) * runtime.get("hidden_size", 0) * 2):
                raise ValueError("Worker did not install the bound scalar head/book table")
            verify_reference_worker(states[0], self.identity)
            with ready.open("x") as f:
                json.dump({"ready": True, "pid": os.getpid(), "socket": str(socket_path),
                           "identity": self.identity, "installed_runtime": runtime,
                           "effective_max_running_requests_per_dp":
                               states[0].get("effective_max_running_requests_per_dp"),
                           "mixlm_sglang_provenance": states[0].get("mixlm_sglang_provenance"),
                           "mixlm_nextbook_prefill_graph":
                               states[0].get("mixlm_nextbook_prefill_graph")}, f)
            await self.stop.wait()
        finally:
            if server:
                server.close()
                await server.wait_closed()
            listener.close()
            self.pool.shutdown(wait=True)
            if inode is not None and socket_path.exists() and socket_path.stat().st_ino == inode:
                socket_path.unlink()


def run(args):
    pretokenize_catalog = getattr(args, "pretokenize_catalog", False)
    if pretokenize_catalog and args.variant != "T":
        raise ValueError("--pretokenize-catalog requires the full-text T variant")
    from .nextbook_export import verify_nextbook_export
    from .nextbook_serving_inputs import NextBookCompiler
    from .nextbook_score_head import configure_runtime
    from .nextbook_sglang_overlay import activate_overlay, verify_overlay
    import torch
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    if args.variant == "T":
        from .nextbook_teacher_export import verify_nextbook_teacher_export
        from .nextbook_teacher_inputs import NextBookTeacherCompiler
        exported = verify_nextbook_teacher_export(args.export)
        compiler = NextBookTeacherCompiler(args.export, pretokenize_catalog=pretokenize_catalog)
    else:
        exported = verify_nextbook_export(args.export)
        compiler = NextBookCompiler(args.export)
    overlay = verify_overlay(args.overlay)
    activate_overlay(args.overlay)
    arithmetic_mode = getattr(args, "arithmetic_mode", "sglang")
    causal = args.variant == "V3" and args.score_mode == "independent"
    prefill_graph = getattr(args, "prefill_graph", "disabled")
    if causal and (arithmetic_mode != "sglang" or prefill_graph != "disabled"):
        raise ValueError("Independent V3 causal scoring requires sglang arithmetic with graphs disabled")
    runtime = configure_runtime(args.export, variant=args.variant, arithmetic_mode=arithmetic_mode)
    options = {"model_path": str(Path(args.export) / "reader"), "dtype": "float16",
               "attention_backend": "flashinfer", "enable_fp32_lm_head": True,
               "enable_mis": not causal, "disable_radix_cache": True, "chunked_prefill_size": -1,
               "disable_cuda_graph": True, "mem_fraction_static": 0.6,
               "context_length": 8192, "random_seed": 17, "log_level": "warning",
               "enable_deterministic_inference": True, "page_size": 1, "tp_size": 1,
               "max_total_tokens": args.kv_tokens}
    from .nextbook_bcg_runtime import parse_token_buckets, prefill_graph_options, configure_graph_admission
    token_buckets = parse_token_buckets(getattr(args, "prefill_graph_token_buckets",
        '369,738,1476,2952,5904,11808,17712,23616'))
    options.update(prefill_graph_options(prefill_graph, token_buckets, arithmetic_mode))
    configure_graph_admission(prefill_graph, token_buckets)
    if arithmetic_mode == "reference_fp16":
        # Numerical acceptance covers one uncached physical MIS request. Client
        # concurrency may queue work but must not change its batch geometry.
        options["max_running_requests"] = 1
    import sglang
    engine = sglang.Engine(**options)
    identity = {"export_signature": exported["manifest_signature"], "runtime_signature": runtime["signature"],
                "variant": args.variant, "score_mode": args.score_mode, "chunk_concurrency": args.chunk_concurrency,
                "attention_mode": "causal" if causal else "mis",
                "arithmetic_mode": arithmetic_mode, "prefill_graph": prefill_graph,
                "prefill_graph_token_buckets": token_buckets,
                "token_cache": compiler.token_cache_report() if args.variant == "T" else None,
                "options": options, "overlay_signature": digest(overlay), "protocol": "mixlm-nextbook-service-v1"}
    try:
        asyncio.run(NextBookServer(compiler, engine, identity, chunk_concurrency=args.chunk_concurrency).serve(args.socket, args.ready))
    finally:
        engine.shutdown()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--overlay", type=Path, required=True)
    p.add_argument("--socket", type=Path, required=True)
    p.add_argument("--ready", type=Path, required=True)
    p.add_argument("--kv-tokens", type=int, default=32768)
    p.add_argument("--variant", choices=["V3", "T"], default="V3")
    p.add_argument("--pretokenize-catalog", action="store_true",
                   help="T only: encode all item content at startup in an immutable CPU token cache")
    p.add_argument("--score-mode", choices=["shared", "independent"], default="shared")
    p.add_argument("--chunk-concurrency", type=int, default=8)
    p.add_argument("--arithmetic-mode", choices=["sglang", "reference_fp16"], default="sglang")
    p.add_argument("--prefill-graph", choices=["disabled", "breakable"], default="disabled")
    p.add_argument("--prefill-graph-token-buckets", default='369,738,1476,2952,5904,11808,17712,23616')
    run(p.parse_args())


if __name__ == "__main__":
    main()
