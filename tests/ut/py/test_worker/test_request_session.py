# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import threading
import time
from typing import cast

from simpler.request_session import RequestEmitter, RequestSession, RequestStream
from simpler.worker import Worker


class _StreamSession:
    def cancel(self, _request_id: int) -> bool:
        return False


class _ConcurrentWorker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.both_started = threading.Event()
        self.release = threading.Event()

    def run(self, task_orch, args=None, config=None) -> None:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if self.active == 2:
                self.both_started.set()
        try:
            task_orch(None, args, config)
            assert self.release.wait(1.0)
        finally:
            with self._lock:
                self.active -= 1

    def _release_request_session(self, _session) -> None:
        pass


class _ConcurrentOrchestrator:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.clear_count = 0
        self.drain_count = 0

    def _clear_error(self) -> None:
        with self._lock:
            self.clear_count += 1

    def _scope_begin(self) -> None:
        pass

    def _scope_end(self) -> None:
        pass

    def _drain(self) -> None:
        with self._lock:
            self.drain_count += 1


def test_emit_waits_until_consumer_receives_token() -> None:
    stream = RequestStream(7, cast(RequestSession, _StreamSession()))
    emitter = RequestEmitter(stream)
    producer_done = threading.Event()

    def produce() -> None:
        emitter.emit("token-1")
        producer_done.set()

    thread = threading.Thread(target=produce)
    thread.start()
    time.sleep(0.02)
    assert not producer_done.is_set()
    assert stream.next(timeout=1.0) == "token-1"
    assert producer_done.wait(1.0)
    stream._finish(None)
    thread.join()


def test_session_runs_two_submitted_requests_concurrently() -> None:
    worker = _ConcurrentWorker()
    session = RequestSession(worker, lambda *_args: None, max_active_runs=2)
    session._start()
    try:
        stream_a = session.submit(object(), request_id=1)
        stream_b = session.submit(object())
        assert stream_b.request_id == 2
        assert worker.both_started.wait(1.0)
        assert worker.max_active == 2
        worker.release.set()
        stream_a.wait(timeout=1.0)
        stream_b.wait(timeout=1.0)
    finally:
        worker.release.set()
        session.close()


def test_worker_groups_overlapping_run_calls_into_one_drain() -> None:
    worker = Worker(level=3, device_ids=[])
    orch = _ConcurrentOrchestrator()
    worker._initialized = True
    worker._worker = object()
    worker._orch = orch
    worker._start_hierarchical = lambda: None
    worker._release_active_remote_slot_refs = lambda: None
    worker._flush_pending_remote_frees = lambda: None
    worker._cleanup_l3_l2_regions = lambda: None
    worker._execute_pending_domain_releases = lambda: None
    started = 0
    started_lock = threading.Lock()
    both_started = threading.Event()
    release = threading.Event()
    errors = []

    def task_orch(_orch, _args, _config) -> None:
        nonlocal started
        with started_lock:
            started += 1
            if started == 2:
                both_started.set()
        assert release.wait(1.0)

    def run() -> None:
        try:
            worker.run(task_orch)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=run), threading.Thread(target=run)]
    for thread in threads:
        thread.start()
    assert both_started.wait(1.0)
    release.set()
    for thread in threads:
        thread.join()

    assert not errors
    assert orch.clear_count == 1
    assert orch.drain_count == 1
