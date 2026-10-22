"""Common logical-request timing for raw and compact next-book services.

No model is selected here. All inputs are label-free dev workloads. Numerical
comparisons are exported as diagnostics, outside the timed request callback.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import time

from .nextbook_benchmark import digest, numerical_check, verify_workload
from .serving_capacity import CapacityConfig, open_loop_replay, percentile


async def control(socket, kind, timeout=30):
    reader, writer = await asyncio.open_unix_connection(str(socket), limit=1 << 22)
    try:
        key = "benchmark-control-" + kind
        writer.write(json.dumps({"id": key, "kind": kind}).encode() + b"\n")
        await writer.drain()
        data = json.loads(await asyncio.wait_for(reader.readline(), timeout))
        if data.get("id") != key or "error_type" in data:
            raise ValueError("Service control failed")
        if kind in {"flush", "stop"} and data.get("ok") is not True:
            raise ValueError("Service control refused")
        return data
    finally:
        writer.close()
        await writer.wait_closed()


class Transport:
    """One connection per logical client, never per candidate."""
    def __init__(self, socket, identity, timeout=120):
        self.socket, self.identity, self.timeout = socket, identity, timeout
        self.peers = []

    async def open(self, count):
        try:
            for _ in range(count):
                self.peers.append(await asyncio.open_unix_connection(str(self.socket), limit=1 << 22))
        except BaseException:
            await self.close()
            raise

    async def close(self):
        for _, writer in self.peers:
            writer.close()
        await asyncio.gather(*(w.wait_closed() for _, w in self.peers), return_exceptions=True)
        self.peers = []

    async def send(self, worker, row, request_id):
        reader, writer = self.peers[worker]
        writer.write(json.dumps({"id": request_id, "kind": "score", "query": row["query"],
            "manifest_signature": self.identity["export_signature"]},
            separators=(",", ":"), allow_nan=False).encode() + b"\n")
        await writer.drain()
        data = json.loads(await asyncio.wait_for(reader.readline(), self.timeout))
        scores = data.get("scores")
        if (data.get("id") != request_id or "error_type" in data
                or data.get("manifest_signature") != self.identity["export_signature"]
                or not isinstance(scores, list) or len(scores) != len(row["episodes"])
                or any(type(x) not in (float, int) or not math.isfinite(x) for x in scores)
                or data.get("ranking") != sorted(range(len(scores)), key=lambda i: (-scores[i], i))
                or not isinstance(data.get("input_hash"), str)):
            raise ValueError("Service changed request identity, scalar outputs or candidate ordering")
        timings = data.get("timings", {})
        if not isinstance(timings, dict) or any(type(v) not in (float, int) or not math.isfinite(v) or v < 0
                                                for v in timings.values()):
            raise ValueError("Invalid service timings")
        return {"scores": scores, "ranking": data["ranking"], "input_hash": data["input_hash"],
                "timings": timings, "physical_requests": data.get("physical_requests"),
                "physical_input_tokens": data.get("physical_input_tokens")}


class ClosedReplayError(RuntimeError):
    """A failed replay retains its settled counts without claiming completion."""
    def __init__(self, summary):
        super().__init__("Closed replay failed; preserve raw results and inspect service idle state")
        self.summary = summary


def closed_summary(records, *, cycles, unique_queries, concurrency, elapsed, failure=None,
                   unsubmitted_in_last_pass=0):
    successes = [r for r in records if r["status"] == "ok"]
    latency = [r["latency_seconds"] * 1000 for r in successes]
    candidates = sum(r["candidates"] for r in successes)
    stages = sorted({k for r in successes for k in r["timings"]})
    return {"mode": "closed", "cycles": cycles, "unique_queries": unique_queries,
            "submitted_requests": len(records), "terminal_requests": len(records),
            "completed_requests": len(successes), "failed_requests": len(records) - len(successes),
            "submitted_candidates": sum(r["candidates"] for r in records),
            "candidate_outputs": sum(r["returned_score_count"] for r in records), "candidates": candidates,
            "unsubmitted_in_last_pass": unsubmitted_in_last_pass,
            "elapsed_including_drain_seconds": elapsed,
            "requests_per_second": len(successes) / elapsed, "candidates_per_second": candidates / elapsed,
            "p50_ms": percentile(latency, .5), "p95_ms": percentile(latency, .95),
            "p99_ms": percentile(latency, .99), "concurrency_logical_groups": concurrency,
            "measurement_complete": failure is None, "trace_passes_complete": failure is None,
            "capacity_accepted": False, "failure_cause": failure,
            "latency_scope": "Successful requests only; failed requests and missing candidate outputs are counted separately",
            "stages_mean_ms": {k: statistics.mean(r["timings"][k] * 1000 for r in successes if k in r["timings"])
                               for k in stages},
            "scope": "Whole-trace closed-loop replay; not an open-loop latency-SLA capacity claim"}


async def closed_replay(rows, send, *, concurrency, minimum_seconds, timeout_seconds,
                        seed=17, sink=None):
    """Complete whole trace passes so fast/slow services see the same input mix.

Each full pass includes each row exactly once. Multiplicity can differ between
services, and is explicitly reported. This is not an open-loop SLA measurement.
"""
    if not rows or concurrency < 1 or minimum_seconds <= 0 or timeout_seconds <= 0:
        raise ValueError("Positive replay settings and a nonempty trace required")
    start = time.perf_counter()
    cycles, records = 0, []
    while cycles == 0 or time.perf_counter() - start < minimum_seconds:
        order = list(range(len(rows)))
        random.Random(seed + cycles).shuffle(order)
        next_index = 0
        failure = None
        failure_record = None

        async def worker(worker_id):
            nonlocal next_index, failure, failure_record
            while next_index < len(order) and failure is None:
                local = next_index
                next_index += 1
                row = rows[order[local]]
                begin = time.perf_counter()
                record = {"sequence": cycles * len(rows) + local, "cycle": cycles,
                          "workload_request_id": row["request_id"], "candidates": len(row["episodes"]),
                          "returned_score_count": 0}
                try:
                    response = await asyncio.wait_for(send(worker_id, row, f"closed-{cycles}-{local}"), timeout_seconds)
                    record.update(response)
                    record["status"] = "ok"
                    record["returned_score_count"] = len(response["scores"])
                except Exception as error:
                    if failure is None:
                        failure = error
                        failure_record = {"error_type": type(error).__name__, "message": str(error),
                                          "sequence": record["sequence"], "workload_request_id": row["request_id"]}
                    record.update(status="error", error_type=type(error).__name__, error_message=str(error))
                record["latency_seconds"] = time.perf_counter() - begin
                if sink is not None:
                    sink(record)
                records.append({k: v for k, v in record.items() if k not in {"scores", "ranking"}})

        await asyncio.gather(*(worker(i) for i in range(concurrency)))
        if failure is not None:
            summary = closed_summary(records, cycles=cycles, unique_queries=len(rows), concurrency=concurrency,
                elapsed=time.perf_counter() - start, failure=failure_record,
                unsubmitted_in_last_pass=len(order) - next_index)
            raise ClosedReplayError(summary) from failure
        cycles += 1
    return closed_summary(records, cycles=cycles, unique_queries=len(rows), concurrency=concurrency,
                          elapsed=time.perf_counter() - start)


async def gpu_sampler(gpu, path, stop):
    fields = "uuid,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm"
    samples, errors = [], 0
    with path.open("x") as stream:
        while not stop.is_set():
            try:
                p = await asyncio.to_thread(subprocess.run, ["nvidia-smi", "-i", gpu,
                    "--query-gpu=" + fields, "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10)
                values = [x.strip() for x in p.stdout.strip().split(",")]
                if p.returncode or len(values) != 6:
                    raise ValueError("GPU sample unavailable")
                sample = {"time_unix": time.time(), "uuid": values[0],
                          **dict(zip(fields.split(",")[1:], map(float, values[1:])))}
                samples.append(sample)
            except Exception as error:
                errors += 1
                sample = {"time_unix": time.time(), "error_type": type(error).__name__}
            stream.write(json.dumps(sample) + "\n")
            stream.flush()
            try:
                await asyncio.wait_for(stop.wait(), 1)
            except TimeoutError:
                pass
    return {"samples": len(samples), "errors": errors,
            "mean_utilization_percent": statistics.mean(x["utilization.gpu"] for x in samples) if samples else None,
            "peak_memory_mib": max((x["memory.used"] for x in samples), default=None),
            "gpu_uuids": sorted({x["uuid"] for x in samples})}


def numerical_report(raw_path, references):
    """Compare first occurrence of each query; diagnostic, outside timing."""
    seen, checks = set(), []
    for line in raw_path.read_text().splitlines():
        row = json.loads(line)
        key = row["workload_request_id"]
        if key in seen or row.get("status") != "ok" or key not in references:
            continue
        seen.add(key)
        expected = references[key]
        if row["input_hash"] != expected["input_hash"]:
            raise ValueError("Reference input identity differs")
        checks.append({"request_id": key, **numerical_check(row["scores"], expected["fp16"])})
    return {"scope": "First observed response for each covered reference query; original numerical gate is diagnostic",
            "reference_queries": len(references), "compared_queries": len(checks), "checks": checks}


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False, indent=2)
        stream.write("\n")


async def benchmark(args):
    workload = verify_workload(json.loads(args.workload.read_text()))
    rows = [r for r in workload["requests"] if len(r["episodes"]) == args.candidates]
    if not rows or len({r["request_id"] for r in rows}) != len(rows):
        raise ValueError("Unique selected workload queries required")
    args.output.mkdir(parents=True, exist_ok=False)
    before = await control(args.socket, "info")
    if before.get("active", 0):
        raise ValueError("Service has active requests from another run")
    identity = before["identity"]
    spec = {"schema": "nextbook-comparison-replay-v1", "service": before,
            "workload_sha256": workload["workload_sha256"], "selected_trace_sha256": digest(rows),
            "unique_queries": len(rows), "candidates": args.candidates,
            "mode": args.mode, "concurrency": args.concurrency, "seconds": args.seconds,
            "warmup_requests": args.warmup, "flush_after_warmup": args.flush_after_warmup,
            "arrival_rate": args.arrival_rate, "seed": args.seed,
            "result_cache_allowed": False, "labels_used": False,
            "timing_boundary": "Client serialization, Unix IPC, CPU assembly, queue, GPU, sorting, reply and structural validation; excludes startup/offline build/WAN/retrieval",
            "start_unix": time.time()}
    write_json(args.output / "spec.json", spec)
    transport = Transport(args.socket, identity, args.timeout)
    await transport.open(args.concurrency)
    stop = asyncio.Event()
    sampler = None
    replay_error = None
    try:
        await control(args.socket, "flush")
        for i in range(args.warmup):
            await transport.send(0, rows[i % len(rows)], f"warmup-{i}")
        if args.flush_after_warmup:
            await control(args.socket, "flush")
        sampler = asyncio.create_task(gpu_sampler(args.gpu, args.output / "gpu.jsonl", stop)) if args.gpu else None
        with (args.output / "requests.jsonl").open("x") as stream:
            def sink(record):
                stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
            if args.mode == "closed":
                summary = await closed_replay(rows, transport.send, concurrency=args.concurrency,
                    minimum_seconds=args.seconds, timeout_seconds=args.timeout, seed=args.seed, sink=sink)
            else:
                if args.arrival_rate is None:
                    raise ValueError("Open-loop requires --arrival-rate")
                config = CapacityConfig(arrival_rate=args.arrival_rate, concurrency=args.concurrency,
                    duration_seconds=args.seconds, min_requests=args.min_requests,
                    max_queue=args.max_queue, timeout_seconds=args.timeout, sla_ms=args.sla_ms)
                result = await open_loop_replay(rows, transport.send, config, sink=sink)
                summary = result["summary"]
                summary["mode"] = "open"
                summary["config"] = asdict(config)
                summary["measurement_complete"] = result["complete"]
                summary["abort"] = result["abort"]
        after = await control(args.socket, "info")
        if after.get("active", 0):
            raise ValueError("Service did not drain")
        if after["identity"] != identity:
            raise ValueError("Service identity changed during replay")
        summary["service_before"] = before
        summary["service_after"] = after
        stop.set()
        summary["gpu"] = await sampler if sampler else None
        sampler = None
        write_json(args.output / "summary.json", summary)
        if args.reference:
            reference = json.loads(args.reference.read_text())
            if reference["export_signature"] != identity["export_signature"]:
                raise ValueError("Reference model differs")
            write_json(args.output / "numerical.json", numerical_report(args.output / "requests.jsonl", reference["references"]))
        return summary
    except BaseException as error:
        replay_error = error
        failure = {"error_type": type(error).__name__, "message": str(error), "time_unix": time.time(),
                   "instruction": "Inspect the existing owned service and active requests before any retry"}
        if isinstance(error, ClosedReplayError):
            summary = {**error.summary, "service_before": before, "service_after": None,
                       "raw_records_file": "requests.jsonl", "backend_idle_verified": False}
            failure["cause"] = summary["failure_cause"]
            # Read-only and bounded: never flush a backend after a failed replay.
            try:
                after = await asyncio.wait_for(control(args.socket, "info"), min(args.timeout, 5))
                summary["service_after"] = after
                summary["backend_idle_verified"] = (after.get("identity") == identity
                    and type(after.get("active")) is int and after["active"] == 0)
            except Exception as diagnostic_error:
                summary["service_diagnostic_error"] = {"error_type": type(diagnostic_error).__name__,
                                                       "message": str(diagnostic_error)}
            stop.set()
            if sampler:
                try:
                    summary["gpu"] = await sampler
                except Exception as diagnostic_error:
                    summary["gpu_diagnostic_error"] = {"error_type": type(diagnostic_error).__name__,
                                                       "message": str(diagnostic_error)}
                finally:
                    sampler = None
            try:
                write_json(args.output / "summary.json", summary)
            except Exception as report_error:
                error.add_note("Could not write failed replay summary: " + str(report_error))
        try:
            write_json(args.output / "failure.json", failure)
        except Exception as report_error:
            error.add_note("Could not write failure report: " + str(report_error))
        raise
    finally:
        stop.set()
        cleanup = [transport.close()]
        if sampler:
            cleanup.append(sampler)
        for result in await asyncio.gather(*cleanup, return_exceptions=True):
            if isinstance(result, BaseException):
                if replay_error is None:
                    raise result
                replay_error.add_note("Replay cleanup failed: " + str(result))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--socket", type=Path, required=True)
    p.add_argument("--workload", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--mode", choices=("closed", "open"), default="closed")
    p.add_argument("--candidates", type=int, default=100)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--flush-after-warmup", action="store_true")
    p.add_argument("--arrival-rate", type=float)
    p.add_argument("--min-requests", type=int, default=10_000)
    p.add_argument("--max-queue", type=int, default=1024)
    p.add_argument("--sla-ms", type=float, default=500)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--gpu")
    args = p.parse_args(argv)
    if args.concurrency < 1 or args.seconds <= 0 or args.timeout <= 0 or args.warmup < 0:
        p.error("Replay settings must be positive (warmup may be zero)")
    result = asyncio.run(benchmark(args))
    print(json.dumps({k: result.get(k) for k in ("mode", "candidates_per_second", "p50_ms", "p99_ms")}))


if __name__ == "__main__":
    main()
