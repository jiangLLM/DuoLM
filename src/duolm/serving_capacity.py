"""Open-loop capacity accounting, independent of any inference runtime.

Every planned arrival receives one terminal record, including client admission
rejections and deadline failures. Queueing starts at the planned arrival, not at
socket dispatch. A bounded queue prevents an overloaded benchmark consuming
unbounded memory. Raw scores belong in the caller's private sink, never summaries.
"""
from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import asdict, dataclass
import math
import statistics
import time


@dataclass(frozen=True)
class CapacityConfig:
    arrival_rate: float
    concurrency: int = 8
    duration_seconds: float = 120.0
    min_requests: int = 10_000
    max_queue: int = 1024
    timeout_seconds: float = 120.0
    sla_ms: float = 500.0
    backlog_growth_allowance: float = 1.0

    def validate(self):
        for name in ('arrival_rate', 'duration_seconds', 'timeout_seconds', 'sla_ms'):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('Invalid positive setting: ' + name)
        for name in ('concurrency', 'min_requests', 'max_queue'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError('Invalid positive integer: ' + name)
        if not math.isfinite(self.backlog_growth_allowance) or self.backlog_growth_allowance < 0:
            raise ValueError('Invalid backlog allowance')
        if self.requests > 2_000_000:
            raise ValueError('Replay exceeds two million scheduled arrivals')

    @property
    def requests(self):
        return max(self.min_requests, math.ceil(self.arrival_rate * self.duration_seconds))

    @property
    def offered_seconds(self):
        return self.requests / self.arrival_rate


def percentile(values, quantile):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def summarize_capacity(records, config, *, elapsed_seconds, backlog_samples):
    """Summarize settled arrivals; no identity, raw probability or error text."""
    config.validate()
    if len(records) != config.requests or {r['arrival_index'] for r in records} != set(range(config.requests)):
        raise ValueError('Every planned arrival must have exactly one terminal record')
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise ValueError('Invalid measured duration')
    successes = [r for r in records if r['status'] == 'ok']
    latencies = [r['latency_seconds'] * 1000 for r in successes]
    counts = Counter(r['status'] for r in records)
    failed = len(records) - len(successes)
    violations = failed + sum(latency > config.sla_ms for latency in latencies)
    buckets = [[] for _ in range(4)]
    for offset, outstanding in backlog_samples:
        bucket = min(3, max(0, int(offset / config.offered_seconds * 4)))
        buckets[bucket].append(outstanding)
    means = [statistics.mean(values) if values else None for values in buckets]
    growth = None if means[1] is None or means[3] is None else means[3] - means[1]
    no_growth = growth is not None and growth <= config.backlog_growth_allowance
    p99 = percentile(latencies, .99)
    aborted = counts.get('aborted', 0)
    enough = (not aborted and config.offered_seconds >= 120
              and elapsed_seconds >= 120 and len(records) >= 10_000)
    passed = (failed == 0 and p99 is not None and p99 <= config.sla_ms and no_growth)
    stage_keys = sorted({key for r in successes for key in r.get('timings', {})})
    return {
        'planned_requests': len(records), 'status_counts': dict(sorted(counts.items())),
        'successes': len(successes), 'failures': failed, 'aborted_requests': aborted,
        'observed_requests': len(records) - sum(r.get('schedule_lag_seconds') is None for r in records),
        'dispatched_requests': sum(bool(r.get('dispatched')) for r in records),
        'planned_arrival_window_seconds': config.offered_seconds,
        'elapsed_including_drain_seconds': elapsed_seconds,
        'drain_seconds': max(0.0, elapsed_seconds - config.offered_seconds),
        'requests_per_second': len(successes) / elapsed_seconds,
        'candidates_per_second': sum(r['candidates'] for r in successes) / elapsed_seconds,
        'failure_rate': failed / len(records),
        'sla_violation_rate_including_failures': violations / len(records),
        'p50_ms': percentile(latencies, .5), 'p95_ms': percentile(latencies, .95), 'p99_ms': p99,
        'latency_scope': 'Successful requests only, from planned arrival through postprocessing; failures separately counted as SLA violations',
        'driver_schedule_lag_p99_ms': percentile([r['schedule_lag_seconds'] * 1000 for r in records
                                                if r.get('schedule_lag_seconds') is not None], .99),
        'queue_wait_p99_ms': percentile([r['admission_wait_seconds'] * 1000 for r in records
                                        if r.get('admission_wait_seconds') is not None], .99),
        'stages_mean_ms': {key: statistics.mean(r['timings'][key] * 1000 for r in successes
                                               if key in r.get('timings', {})) for key in stage_keys},
        'outstanding_quarter_means': means,
        'outstanding_last_minus_second_quarter': growth,
        'backlog_growth_allowance_requests': config.backlog_growth_allowance,
        'no_sustained_backlog_growth': no_growth,
        'all_arrivals_settled': True, 'performance_gate_passed': passed,
        'adequate_single_replay_duration_and_count': enough,
        'capacity_accepted': passed and enough,
        'scope': 'Single open-loop replay; three independent accepted replays and numerical validation are required for a capacity claim',
    }


async def open_loop_replay(requests, send, config, *, sink=None):
    """Call ``await send(worker_id, row, unique_id)`` on fixed-rate arrivals.

    The sink receives private completed records synchronously. Its I/O may delay
    the load generator; schedule lag and planned-arrival latency include that
    interference. The first timeout/error prevents further dispatch immediately.
    Queued and future arrivals are marked aborted. Already dispatched requests
    settle without replacement. Cancellation does not prove GPU cancellation;
    callers must verify backend idle or restart it before any subsequent replay.
    """
    config.validate()
    if not requests:
        raise ValueError('Empty workload')
    queue = asyncio.Queue(maxsize=config.max_queue)
    records, samples = [], []
    stopped = asyncio.Event()
    abort = None
    settled = 0
    admitted = 0
    start = time.perf_counter()
    start_unix = time.time()

    def stop_dispatch(reason, index):
        nonlocal abort
        if not stopped.is_set():
            abort = {'reason': reason, 'arrival_index': index,
                     'offset_seconds': time.perf_counter() - start}
            stopped.set()

    def settle(record):
        nonlocal settled
        if sink is not None:
            sink(record)
        # Large per-candidate arrays go only to the private sink.
        records.append({k: v for k, v in record.items() if k in {
            'arrival_index', 'status', 'candidates', 'latency_seconds',
            'schedule_lag_seconds', 'admission_wait_seconds', 'timings', 'dispatched'}})
        settled += 1

    def aborted_record(index, planned, row, observed=None):
        return {'arrival_index': index, 'workload_request_id': row['request_id'],
                'candidates': len(row['episodes']), 'status': 'aborted', 'dispatched': False,
                'planned_offset_seconds': planned - start,
                'schedule_lag_seconds': None if observed is None else observed - planned,
                'admission_wait_seconds': None, 'latency_seconds': None,
                'abort_reason': 'Replay stopped before dispatch'}

    async def worker(worker_id):
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                index, planned, observed, row = item
                if stopped.is_set():
                    settle(aborted_record(index, planned, row, observed))
                    continue
                dispatch = time.perf_counter()
                record = {'arrival_index': index, 'workload_request_id': row['request_id'],
                          'candidates': len(row['episodes']),
                          'planned_offset_seconds': planned - start,
                          'schedule_lag_seconds': observed - planned,
                          'admission_wait_seconds': dispatch - observed, 'dispatched': False}
                remaining = planned + config.timeout_seconds - dispatch
                sending = None
                try:
                    if remaining <= 0:
                        raise TimeoutError('Deadline expired before dispatch')
                    async def guarded_send():
                        if stopped.is_set():
                            return None
                        record['dispatched'] = True
                        try:
                            value = await send(worker_id, row, f'capacity-{index}')
                            if time.perf_counter() > planned + config.timeout_seconds:
                                raise TimeoutError('Deadline expired before callback returned')
                            return value
                        except Exception as error:
                            stop_dispatch('timeout' if isinstance(error, TimeoutError) else 'error', index)
                            raise
                    sending = asyncio.create_task(guarded_send())
                    done, _ = await asyncio.wait({sending}, timeout=remaining)
                    if not done:
                        # Lock dispatch before cancellation/connection cleanup can yield.
                        stop_dispatch('timeout', index)
                        raise TimeoutError('Dispatch deadline expired')
                    result = sending.result()
                    if not record['dispatched']:
                        settle(aborted_record(index, planned, row, observed))
                        continue
                    finished = time.perf_counter()
                    # A blocking callback can return after wait_for's nominal deadline.
                    if finished > planned + config.timeout_seconds:
                        raise TimeoutError('Deadline expired before callback returned')
                    record.update(result)
                    record['status'] = 'ok'
                except TimeoutError:
                    stop_dispatch('timeout', index)
                    record['status'] = 'timeout'
                except Exception as error:
                    stop_dispatch('error', index)
                    record.update(getattr(error, 'capacity_details', {}))
                    record['status'] = 'error'
                    record['error'] = type(error).__name__ + ': ' + str(error)
                finally:
                    if sending is not None and not sending.done():
                        sending.cancel()
                        await asyncio.gather(sending, return_exceptions=True)
                record['latency_seconds'] = time.perf_counter() - planned
                settle(record)
            finally:
                queue.task_done()

    tasks = [asyncio.create_task(worker(i)) for i in range(config.concurrency)]
    try:
        for index in range(config.requests):
            for task in tasks:
                if task.done():
                    task.result()  # A failed private sink must not deadlock queue.join().
            planned = start + index / config.arrival_rate
            delay = planned - time.perf_counter()
            if delay > 0 and not stopped.is_set():
                try:
                    await asyncio.wait_for(stopped.wait(), delay)
                except TimeoutError:
                    pass
            if stopped.is_set():
                for future in range(index, config.requests):
                    settle(aborted_record(future, start + future / config.arrival_rate,
                                          requests[future % len(requests)]))
                break
            observed = time.perf_counter()
            row = requests[index % len(requests)]
            try:
                queue.put_nowait((index, planned, observed, row))
                admitted += 1
            except asyncio.QueueFull:
                settle({'arrival_index': index, 'workload_request_id': row['request_id'],
                        'status': 'rejected_queue_full', 'candidates': len(row['episodes']),
                        'planned_offset_seconds': planned - start,
                        'schedule_lag_seconds': observed - planned,
                        'latency_seconds': observed - planned})
            # Number scheduled minus number terminal includes active and queued work.
            samples.append((index / config.arrival_rate, index + 1 - settled))
        delay = start + config.offered_seconds - time.perf_counter()
        if delay > 0 and not stopped.is_set():
            try:
                await asyncio.wait_for(stopped.wait(), delay)
            except TimeoutError:
                pass
        drain = asyncio.create_task(queue.join())
        try:
            completed, _ = await asyncio.wait([drain, *tasks], return_when=asyncio.FIRST_COMPLETED)
            for task in completed:
                task.result()
        finally:
            if not drain.done():
                drain.cancel()
                await asyncio.gather(drain, return_exceptions=True)
        for _ in tasks:
            await queue.put(None)
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.perf_counter() - start
    return {'config': asdict(config), 'measurement_start_unix': start_unix,
            'measurement_end_unix': time.time(), 'admitted': admitted,
            'complete': not stopped.is_set(), 'abort': abort,
            'backend_idle_or_restart_required': stopped.is_set(),
            'backend_idle_verified': False if stopped.is_set() else None,
            'summary': summarize_capacity(records, config, elapsed_seconds=elapsed,
                                          backlog_samples=samples)}
