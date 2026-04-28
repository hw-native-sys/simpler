# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Device-aware subprocess scheduler for parallel L3 test dispatch.

Given a list of jobs, each declaring how many devices it needs, this module
runs them as isolated subprocesses in parallel up to the device pool's
capacity. Used by the pytest test dispatcher (conftest.py) and the standalone
runner (scene_test.run_module) to parallelize Level-3 test cases.

Concurrency bound: the caller's ``device_ids`` list size. No separate
``--max-parallel`` knob — shrink ``-d`` to throttle.

Static safety: any job where ``device_count > len(device_ids)`` fails the
whole batch up front (otherwise it would deadlock on the wait queue).

Fail-fast: when ``fail_fast=True`` and any job fails, cancel the pending
queue, SIGTERM running children, and return promptly.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Job:
    """A single scheduled subprocess.

    ``build_cmd`` receives the list of device ids allocated to this job and
    must return the argv. We build the command lazily so the caller can
    substitute the allocated ``--device <range>`` at launch time.
    """

    label: str  # Human label for logs
    device_count: int  # Devices required
    build_cmd: Callable[[list[int]], list[str]]  # Given allocated ids → argv
    cwd: str | None = None
    env: dict | None = None


@dataclass
class JobResult:
    label: str
    returncode: int
    device_ids: list[int]
    output: str = ""  # Captured combined stdout+stderr
    duration_s: float = 0.0


@dataclass
class _RunningJob:
    """Per-subprocess book-keeping held while the job is in flight."""

    job: Job
    device_ids: list[int]
    start_time: float
    output_lines: list[str]
    pump_thread: threading.Thread


@dataclass
class _RunState:
    free_devices: list[int]
    running: dict[subprocess.Popen, _RunningJob] = field(default_factory=dict)
    results: list[JobResult] = field(default_factory=list)
    failed: bool = False
    cancelled: bool = False


def _device_range_str(ids: list[int]) -> str:
    """Format a device-id list as a CLI-friendly range or comma list.

    [0,1,2,3] -> "0-3"   (contiguous ascending)
    [4]       -> "4"
    [0,2,5]   -> "0,2,5" (non-contiguous)
    """
    if not ids:
        return ""
    if len(ids) == 1:
        return str(ids[0])
    s = sorted(ids)
    contiguous = all(s[i + 1] - s[i] == 1 for i in range(len(s) - 1))
    if contiguous:
        return f"{s[0]}-{s[-1]}"
    return ",".join(str(i) for i in s)


def _acquire_devices(state: _RunState, count: int) -> list[int] | None:
    """Try to grab ``count`` device ids from the free pool; None if short."""
    if len(state.free_devices) < count:
        return None
    allocated = state.free_devices[:count]
    state.free_devices = state.free_devices[count:]
    return allocated


def _release_devices(state: _RunState, ids: list[int]) -> None:
    state.free_devices.extend(ids)


def _terminate_all(state: _RunState, timeout_s: float = 5.0) -> None:
    """SIGTERM all running children, wait briefly, then SIGKILL stragglers."""
    for p in list(state.running):
        if p.poll() is None:
            try:
                p.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
    for p in list(state.running):
        try:
            p.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                p.kill()
                p.wait(timeout=1.0)
            except Exception:  # noqa: BLE001
                pass


def run_jobs(
    jobs: list[Job],
    device_ids: list[int],
    *,
    max_parallel: int | None = None,
    fail_fast: bool = False,
    poll_interval_s: float = 0.1,
    on_job_done: Callable[[JobResult], None] | None = None,
) -> list[JobResult]:
    """Run jobs concurrently up to the device-pool capacity.

    Scheduling is FIFO: jobs are popped in order; if the head can't be placed
    (not enough free devices right now, or ``max_parallel`` already reached),
    we wait for a running job to finish rather than skip ahead. Simple and
    predictable.

    Args:
        jobs: jobs to dispatch, in preferred order.
        device_ids: full pool of available device ids. ``len(device_ids)`` is
            the device-side hard cap — a single job is still capped by its
            own ``device_count``, and the sum of running jobs' ``device_count``
            can never exceed the pool.
        max_parallel: max number of subprocesses in flight simultaneously
            (``-j`` semantics, same as ``make -j``). ``None`` means "no
            extra cap" and the only bound is the device pool. Use this to
            throttle CPU contention on sim where one subprocess can already
            fork many internal workers.
        fail_fast: on first non-zero return code, cancel remaining queue and
            terminate running children.
        poll_interval_s: how often to check for finished children.
        on_job_done: called (thread-safe w.r.t. this function — which is
            single-threaded) whenever a job completes, useful for streaming
            pytest reports or printing PASS/FAIL lines.

    Returns:
        List of JobResult in completion order. Jobs cancelled under
        ``fail_fast`` do not appear.
    """
    # Static check: no job can ever be placed if its device_count exceeds the
    # total pool. Fail the batch before dispatching anything.
    for j in jobs:
        if j.device_count > len(device_ids):
            raise ValueError(
                f"job {j.label!r} needs {j.device_count} devices but pool has "
                f"{len(device_ids)}; widen --device range or shrink the case's device_count"
            )

    state = _RunState(free_devices=list(device_ids))
    queue = list(jobs)

    def _pump_stdout(p: subprocess.Popen, sink: list[str]) -> None:
        """Drain ``p.stdout`` line-by-line into ``sink``.

        Without a continuous reader the child would block once the OS pipe
        buffer fills up (~64 KB on Linux / macOS) — typical for a verbose
        pytest run. Runs on its own daemon thread per job so the pump
        cannot contend with the scheduler loop's ``poll`` cadence.
        """
        assert p.stdout is not None
        for line in iter(p.stdout.readline, ""):
            sink.append(line)
        p.stdout.close()

    def _try_launch_head() -> bool:
        """Launch queue[0] if it fits; return True if launched or queue empty/blocked."""
        if not queue:
            return False
        # Respect the in-flight subprocess cap if one is set.
        if max_parallel is not None and len(state.running) >= max_parallel:
            return False
        head = queue[0]
        allocated = _acquire_devices(state, head.device_count)
        if allocated is None:
            return False
        queue.pop(0)
        cmd = head.build_cmd(allocated)
        try:
            # Capture both streams into a single pipe so the buffer we replay
            # preserves natural interleaving. Without ``text=True`` readline()
            # would return bytes and we'd have to decode ourselves.
            p = subprocess.Popen(
                cmd,
                cwd=head.cwd,
                env=head.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # line-buffered; pump sees each line as it's written
            )
        except Exception:
            _release_devices(state, allocated)
            raise
        output_lines: list[str] = []
        pump = threading.Thread(target=_pump_stdout, args=(p, output_lines), daemon=True)
        pump.start()
        state.running[p] = _RunningJob(
            job=head,
            device_ids=allocated,
            start_time=time.monotonic(),
            output_lines=output_lines,
            pump_thread=pump,
        )
        return True

    def _reap_one() -> JobResult | None:
        """Poll for one finished child; return its JobResult or None."""
        for p in list(state.running):
            rc = p.poll()
            if rc is None:
                continue
            rj = state.running.pop(p)
            _release_devices(state, rj.device_ids)
            # Wait for the pump to drain remaining buffer (EOF after the child
            # exits). ``join`` is essentially instant at this point.
            rj.pump_thread.join(timeout=2.0)
            duration = time.monotonic() - rj.start_time
            res = JobResult(
                label=rj.job.label,
                returncode=rc,
                device_ids=rj.device_ids,
                output="".join(rj.output_lines),
                duration_s=duration,
            )
            state.results.append(res)
            if rc != 0:
                state.failed = True
            return res
        return None

    try:
        # Fill until blocked
        while queue and _try_launch_head():
            pass

        while state.running or queue:
            # If cancelled, stop pulling from queue
            if state.cancelled:
                queue.clear()

            reaped = _reap_one()
            if reaped is not None:
                if on_job_done is not None:
                    on_job_done(reaped)
                if fail_fast and state.failed and not state.cancelled:
                    state.cancelled = True
                    queue.clear()
                    continue
                # Fill freed slot
                while queue and _try_launch_head():
                    pass
                continue

            # Nothing to reap; wait briefly
            try:
                # Block on the first running child up to poll_interval_s
                first = next(iter(state.running), None)
                if first is None:
                    break
                first.wait(timeout=poll_interval_s)
            except subprocess.TimeoutExpired:
                pass
    finally:
        if state.cancelled and state.running:
            _terminate_all(state)
            # Sweep final results for the terminated ones
            for p in list(state.running):
                rc = p.poll()
                if rc is None:
                    rc = -signal.SIGTERM
                rj = state.running.pop(p)
                _release_devices(state, rj.device_ids)
                rj.pump_thread.join(timeout=2.0)
                duration = time.monotonic() - rj.start_time
                state.results.append(
                    JobResult(
                        label=rj.job.label,
                        returncode=rc,
                        device_ids=rj.device_ids,
                        output="".join(rj.output_lines),
                        duration_s=duration,
                    )
                )

    return state.results


# ---------------------------------------------------------------------------
# Lock used by callers that share a scheduler across threads (reserved for
# future use; the current single-threaded scheduler above does not need it).
# ---------------------------------------------------------------------------

_scheduler_lock = threading.Lock()


def default_max_parallel(platform: str, device_ids: list[int]) -> int:
    """Compute the ``-j auto`` default for this invocation.

    - Hardware (platform does not end in ``"sim"``): bound is the device count.
      Host CPU is mostly idle waiting on the NPU, so no further cap is useful.
    - Sim: bound is ``min(nproc, len(device_ids))``. Each sim subprocess
      already multiplexes many internal threads onto CPUs; running one
      subprocess per CPU is the practical ceiling.
    """
    n_dev = len(device_ids)
    if not platform.endswith("sim"):
        return max(n_dev, 1)
    try:
        cpu = os.cpu_count() or 1
    except Exception:  # noqa: BLE001
        cpu = 1
    return max(1, min(n_dev, cpu))


def device_range_to_list(spec: str) -> list[int]:
    """Parse a --device spec into a sorted deduplicated list of ints.

    Supports comma-separated mixed forms including ranges:

      ``"0"``        → ``[0]``
      ``"0-3"``      → ``[0, 1, 2, 3]``
      ``"0,2,5"``    → ``[0, 2, 5]``
      ``"0,2-4,7"``  → ``[0, 2, 3, 4, 7]``

    Whitespace inside comma-separated parts is trimmed. Empty input returns
    an empty list so callers can uniformly handle the unset case.
    Kept here (rather than in conftest.py) so standalone can share the same
    parsing.
    """
    if not spec:
        return []
    ids: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            ids.update(range(int(start), int(end) + 1))
        else:
            ids.add(int(part))
    return sorted(ids)


def format_device_range(ids: list[int]) -> str:
    """Inverse of device_range_to_list — public wrapper around _device_range_str."""
    return _device_range_str(ids)
