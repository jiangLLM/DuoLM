"""Open-loop overload accounting cannot hide failed or queued arrivals."""
import asyncio
import unittest

from duolm.serving_capacity import CapacityConfig, open_loop_replay, summarize_capacity


REQUESTS = [{'request_id': 'private-user-identity', 'episodes': [{}] * 5}]


class CapacityAccounting(unittest.TestCase):
    def test_defaults_cover_both_minimum_duration_and_request_count(self):
        self.assertEqual(CapacityConfig(10).requests, 10_000)
        self.assertEqual(CapacityConfig(10).offered_seconds, 1000)
        self.assertEqual(CapacityConfig(100).requests, 12_000)
        for config in (CapacityConfig(float('nan')), CapacityConfig(-1),
                       CapacityConfig(1, max_queue=0), CapacityConfig(1, concurrency=True)):
            with self.assertRaises(ValueError):
                config.validate()

    def test_timeout_counts_as_failure_and_violation_not_missing_latency(self):
        config = CapacityConfig(1, duration_seconds=120, min_requests=10_000)
        rows = [{'arrival_index': i, 'status': 'ok', 'candidates': 100,
                 'latency_seconds': .02, 'schedule_lag_seconds': 0} for i in range(10_000)]
        samples = [(i * 100, 1) for i in range(100)]
        accepted = summarize_capacity(rows, config, elapsed_seconds=10_000, backlog_samples=samples)
        self.assertTrue(accepted['capacity_accepted'])
        rows[0]['status'] = 'timeout'
        failed = summarize_capacity(rows, config, elapsed_seconds=10_001, backlog_samples=samples)
        self.assertFalse(failed['capacity_accepted'])
        self.assertEqual(failed['failures'], 1)
        self.assertEqual(failed['sla_violation_rate_including_failures'], .0001)
        self.assertAlmostEqual(failed['requests_per_second'], 9999 / 10001)
        self.assertEqual(failed['p99_ms'], 20)
        with self.assertRaises(ValueError):
            summarize_capacity(rows[:-1], config, elapsed_seconds=1, backlog_samples=samples)

    def test_growing_backlog_fails_even_when_success_p99_is_good(self):
        config = CapacityConfig(100, duration_seconds=120, min_requests=10_000)
        rows = [{'arrival_index': i, 'status': 'ok', 'candidates': 5,
                 'latency_seconds': .01, 'schedule_lag_seconds': 0} for i in range(config.requests)]
        summary = summarize_capacity(rows, config, elapsed_seconds=120,
                                     backlog_samples=[(i, i) for i in range(120)])
        self.assertFalse(summary['no_sustained_backlog_growth'])
        self.assertFalse(summary['capacity_accepted'])


class CapacityReplay(unittest.IsolatedAsyncioTestCase):
    async def test_slow_service_rejects_explicitly_and_keeps_planned_wait(self):
        async def send(worker, row, request_id):
            await asyncio.sleep(.02)
            return {'scores': [[.2] * 5] * 5, 'input_hashes': ['secret']}
        raw = []
        config = CapacityConfig(500, concurrency=1, duration_seconds=.025, min_requests=12,
                                max_queue=1, timeout_seconds=1)
        result = await open_loop_replay(REQUESTS, send, config, sink=raw.append)
        summary = result['summary']
        self.assertEqual(len(raw), config.requests)
        self.assertGreater(summary['status_counts'].get('rejected_queue_full', 0), 0)
        self.assertGreater(summary['queue_wait_p99_ms'], 1)
        self.assertFalse(summary['capacity_accepted'])
        self.assertNotIn('private-user-identity', str(result))
        self.assertNotIn('secret', str(result))
        self.assertTrue(any('scores' in row for row in raw))
        self.assertEqual(sum(summary['status_counts'].values()), config.requests)

    async def test_deadline_includes_admission_wait_and_all_arrivals_settle(self):
        async def send(*args):
            await asyncio.sleep(.03)
            return {}
        raw = []
        config = CapacityConfig(500, concurrency=1, duration_seconds=.01, min_requests=5,
                                max_queue=10, timeout_seconds=.005)
        result = await open_loop_replay(REQUESTS, send, config, sink=raw.append)
        self.assertEqual(result['summary']['status_counts'], {'aborted': config.requests - 1, 'timeout': 1})
        self.assertEqual(result['summary']['failure_rate'], 1)
        self.assertEqual(result['summary']['sla_violation_rate_including_failures'], 1)
        self.assertIsNone(result['summary']['p99_ms'])
        self.assertFalse(result['complete'])
        self.assertTrue(result['backend_idle_or_restart_required'])

    async def test_timeout_stops_dispatch_while_cancelled_backend_work_remains_alive(self):
        backend_release = asyncio.Event()
        backend_tasks, dispatches, raw = [], [], []

        async def send(worker, row, request_id):
            # Mirrors socket cancellation: the GPU task outlives its awaiting client.
            task = asyncio.create_task(backend_release.wait())
            backend_tasks.append(task)
            dispatches.append(request_id)
            await asyncio.shield(task)
            return {}

        config = CapacityConfig(1000, concurrency=2, duration_seconds=1,
                                min_requests=1000, max_queue=100, timeout_seconds=.02)
        try:
            result = await asyncio.wait_for(open_loop_replay(REQUESTS, send, config, sink=raw.append), .5)
            self.assertEqual(len(dispatches), 2)
            self.assertTrue(all(not task.done() for task in backend_tasks))
            self.assertEqual(result['summary']['status_counts'], {'aborted': 998, 'timeout': 2})
            self.assertEqual(result['summary']['dispatched_requests'], 2)
            self.assertEqual(len({row['arrival_index'] for row in raw}), config.requests)
            self.assertTrue(any(row['status'] == 'aborted' and row['schedule_lag_seconds'] is None for row in raw))
            self.assertTrue(any(row['status'] == 'aborted' and row['schedule_lag_seconds'] is not None for row in raw))
            self.assertTrue(result['backend_idle_or_restart_required'])
            self.assertFalse(result['backend_idle_verified'])
            self.assertFalse(result['summary']['capacity_accepted'])
        finally:
            backend_release.set()
            await asyncio.gather(*backend_tasks)

    async def test_first_error_aborts_future_arrivals_without_waiting_for_next_arrival(self):
        dispatches, raw = [], []
        async def send(worker, row, request_id):
            dispatches.append(request_id)
            raise ValueError('stop now')
        config = CapacityConfig(.1, concurrency=8, duration_seconds=120, min_requests=20)
        result = await asyncio.wait_for(open_loop_replay(REQUESTS, send, config, sink=raw.append), .2)
        self.assertEqual(dispatches, ['capacity-0'])
        self.assertEqual(result['summary']['status_counts'], {'aborted': 19, 'error': 1})
        self.assertEqual(result['abort']['reason'], 'error')
        self.assertFalse(result['summary']['adequate_single_replay_duration_and_count'])

    async def test_errors_are_private_and_short_smoke_is_not_capacity(self):
        async def send(*args):
            raise ValueError('private path and candidate')
        config = CapacityConfig(100, duration_seconds=.02, min_requests=2)
        raw = []
        result = await open_loop_replay(REQUESTS, send, config, sink=raw.append)
        self.assertEqual(result['summary']['status_counts'], {'aborted': 1, 'error': 1})
        self.assertIn('private path', next(row['error'] for row in raw if row['status'] == 'error'))
        self.assertNotIn('private path', str(result))
        self.assertFalse(result['summary']['adequate_single_replay_duration_and_count'])

    async def test_failed_private_sink_does_not_deadlock_the_drain(self):
        async def send(*args):
            return {}
        def sink(record):
            raise OSError('disk full')
        config = CapacityConfig(1000, duration_seconds=.002, min_requests=2)
        with self.assertRaisesRegex(OSError, 'disk full'):
            await asyncio.wait_for(open_loop_replay(REQUESTS, send, config, sink=sink), .2)


if __name__ == '__main__':
    unittest.main()
