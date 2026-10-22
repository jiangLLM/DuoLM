import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from duolm.nextbook_comparison_benchmark import (
    ClosedReplayError, Transport, benchmark, closed_replay, numerical_report,
)


def rows(n=5):
    return [{"request_id": str(i), "query": {"query_id": str(i)}, "episodes": ["a", "b"]}
            for i in range(n)]


def test_closed_replay_completes_whole_balanced_trace():
    seen = []
    async def send(worker, row, identifier):
        await asyncio.sleep(.0001)
        seen.append(row["request_id"])
        return {"scores": [2., 1.], "ranking": [0, 1], "timings": {"inference_seconds": .001}}
    output = []
    result = asyncio.run(closed_replay(rows(), send, concurrency=3,
        minimum_seconds=.006, timeout_seconds=2, sink=output.append))
    assert result["completed_requests"] == 5 * result["cycles"]
    assert all(seen.count(str(i)) == result["cycles"] for i in range(5))
    assert len(output) == result["completed_requests"]
    assert result["candidates"] == result["completed_requests"] * 2
    assert result["submitted_requests"] == result["completed_requests"]
    assert result["failed_requests"] == 0
    assert result["candidate_outputs"] == result["candidates"]
    assert result["measurement_complete"] is True
    assert result["capacity_accepted"] is False
    assert result["stages_mean_ms"]["inference_seconds"] == 1.


def test_replay_failure_does_not_claim_complete_trace():
    async def send(worker, row, identifier):
        if identifier == "closed-0-1":
            raise ValueError("wrong output")
        await asyncio.sleep(.005)
        return {"scores": [2., 1.], "ranking": [0, 1], "timings": {"inference_seconds": .001}}
    output = []
    with unittest.TestCase().assertRaises(ClosedReplayError) as caught:
        asyncio.run(closed_replay(rows(), send, concurrency=2, minimum_seconds=.01,
                                timeout_seconds=1, sink=output.append))
    result = caught.exception.summary
    assert isinstance(caught.exception.__cause__, ValueError)
    assert result["submitted_requests"] == result["terminal_requests"] == 2
    assert result["completed_requests"] == result["failed_requests"] == 1
    assert result["submitted_candidates"] == 4
    assert result["candidate_outputs"] == result["candidates"] == 2
    assert result["unsubmitted_in_last_pass"] == 3
    assert result["cycles"] == 0
    assert result["measurement_complete"] is result["trace_passes_complete"] is False
    assert result["capacity_accepted"] is False
    assert result["failure_cause"]["message"] == "wrong output"
    assert result["requests_per_second"] == 1 / result["elapsed_including_drain_seconds"]
    assert result["candidates_per_second"] == 2 / result["elapsed_including_drain_seconds"]
    assert len(output) == 2 and {r["status"] for r in output} == {"ok", "error"}
    assert next(r for r in output if r["status"] == "ok")["scores"] == [2., 1.]


def assert_transport_rejects_changed_scalar_contract(tmp_path, field, value):
    async def run():
        path = tmp_path / "test.sock"
        async def handler(reader, writer):
            request = json.loads(await reader.readline())
            response = {"id": request["id"], "scores": [2., 1.], "ranking": [0, 1],
                        "manifest_signature": "owned", "input_hash": "input", "timings": {}}
            response[field] = value
            writer.write(json.dumps(response).encode() + b"\n")
            await writer.drain()
            writer.close()
            await writer.wait_closed()
        server = await asyncio.start_unix_server(handler, str(path))
        transport = Transport(path, {"export_signature": "owned"})
        try:
            await transport.open(1)
            with unittest.TestCase().assertRaises(ValueError):
                await transport.send(0, rows()[0], "correct")
        finally:
            await transport.close()
            server.close()
            await server.wait_closed()
    asyncio.run(run())


def test_numerical_diagnostic_keeps_coverage_and_first_response(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in [
        {"workload_request_id": "a", "status": "ok", "input_hash": "same", "scores": [1., 2.]},
        {"workload_request_id": "a", "status": "ok", "input_hash": "same", "scores": [9., 4.]},
        {"workload_request_id": "b", "status": "ok", "input_hash": "b", "scores": [0., 1.]}]))
    result = numerical_report(path, {"a": {"input_hash": "same", "fp16": [1., 2.]},
                                    "unmeasured": {"input_hash": "x", "fp16": [0., 0.]}})
    assert result["compared_queries"] == 1
    assert result["reference_queries"] == 2
    assert result["checks"][0]["max_score_difference"] == 0


def test_numerical_diagnostic_rejects_changed_input(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text(json.dumps({"workload_request_id": "a", "status": "ok",
                               "input_hash": "wrong", "scores": [1., 2.]}))
    with unittest.TestCase().assertRaisesRegex(ValueError, "input identity"):
        numerical_report(path, {"a": {"input_hash": "correct", "fp16": [1., 2.]}})


class ComparisonTests(unittest.TestCase):
    def test_balanced_trace(self):
        test_closed_replay_completes_whole_balanced_trace()

    def test_failure(self):
        test_replay_failure_does_not_claim_complete_trace()

    def test_benchmark_persists_partial_summary_and_read_only_service_diagnostics(self):
        for diagnostic in ({"identity": {"export_signature": "owned"}, "active": 1},
                           RuntimeError("service unavailable")):
            with self.subTest(diagnostic=diagnostic), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                workload = {"requests": rows(), "workload_sha256": "trace"}
                workload_path = root / "workload.json"
                workload_path.write_text(json.dumps(workload))
                args = SimpleNamespace(workload=workload_path, candidates=2, output=root / "result",
                    socket=root / "unused.sock", timeout=1, concurrency=1, seconds=.01,
                    mode="closed", warmup=0, flush_after_warmup=False, arrival_rate=None,
                    seed=17, gpu=None, reference=None)
                before = {"identity": {"export_signature": "owned"}, "active": 0}
                responses = [before, {"ok": True}, diagnostic]
                transport = SimpleNamespace(open=AsyncMock(), close=AsyncMock(),
                    send=AsyncMock(side_effect=[{"scores": [2., 1.], "ranking": [0, 1], "timings": {}},
                                                ValueError("wrong output")]))
                with patch("duolm.nextbook_comparison_benchmark.verify_workload", return_value=workload), \
                     patch("duolm.nextbook_comparison_benchmark.Transport", return_value=transport), \
                     patch("duolm.nextbook_comparison_benchmark.control", new=AsyncMock(side_effect=responses)) as probe:
                    with self.assertRaises(ClosedReplayError) as caught:
                        asyncio.run(benchmark(args))
                self.assertIsInstance(caught.exception.__cause__, ValueError)
                self.assertEqual([call.args[1] for call in probe.await_args_list], ["info", "flush", "info"])
                transport.close.assert_awaited_once()
                summary = json.loads((args.output / "summary.json").read_text())
                failure = json.loads((args.output / "failure.json").read_text())
                raw = [json.loads(line) for line in (args.output / summary["raw_records_file"]).read_text().splitlines()]
                self.assertEqual(summary["submitted_requests"], 2)
                self.assertEqual(summary["completed_requests"], 1)
                self.assertEqual(summary["failed_requests"], 1)
                self.assertEqual(summary["candidate_outputs"], 2)
                self.assertFalse(summary["measurement_complete"])
                self.assertFalse(summary["trace_passes_complete"])
                self.assertFalse(summary["capacity_accepted"])
                self.assertFalse(summary["backend_idle_verified"])
                self.assertEqual(failure["cause"]["message"], "wrong output")
                self.assertEqual([r["status"] for r in raw], ["ok", "error"])
                if isinstance(diagnostic, Exception):
                    self.assertEqual(summary["service_diagnostic_error"]["message"], "service unavailable")
                else:
                    self.assertEqual(summary["service_after"]["active"], 1)

    def test_scalar_contract(self):
        for field, value in [("id", "wrong"), ("scores", [1.]),
                             ("scores", [float("nan"), 1.]), ("ranking", [1, 0]),
                             ("manifest_signature", "other"), ("timings", {"x": -1.})]:
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as directory:
                assert_transport_rejects_changed_scalar_contract(Path(directory), field, value)

    def test_diagnostic_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            test_numerical_diagnostic_keeps_coverage_and_first_response(Path(directory))

    def test_diagnostic_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            test_numerical_diagnostic_rejects_changed_input(Path(directory))
