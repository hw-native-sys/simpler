#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import threading
from types import SimpleNamespace

from _task_interface import RunTiming
from simpler.task_interface import CallConfig, ChipCallable, ChipStorageTaskArgs
from simpler.worker import RegisterHandle, RunHandle, Worker


def _chip_callable() -> ChipCallable:
    return ChipCallable.build(signature=[], func_name="test", binary=b"\x00", children=[])


class _FakeChipWorker:
    def __init__(self):
        self.calls = []
        self.unregisters = []
        self.run_count = 0
        self.entered = threading.Event()
        self.release_first = threading.Event()

    def _run_slot(self, slot_id, args, config):
        index = self.run_count
        self.run_count += 1
        self.calls.append(("start", index, slot_id, args, config))
        if index == 0:
            self.entered.set()
            self.release_first.wait(timeout=2.0)
        self.calls.append(("end", index, slot_id))
        return RunTiming(index + 1, 0)

    def _unregister_slot(self, slot_id):
        self.unregisters.append(slot_id)


def _make_l2_worker():
    worker = Worker(level=2, platform="a2a3sim", runtime="tensormap_and_ringbuffer")
    handle = worker.register(_chip_callable())
    fake = _FakeChipWorker()
    worker._chip_worker = fake
    worker._initialized = True
    worker._start_l2_run_lane()
    return worker, handle, fake


def test_l2_run_async_executes_fifo_on_one_run_lane():
    worker, handle, fake = _make_l2_worker()
    try:
        first = worker.run_async(handle, ChipStorageTaskArgs(), CallConfig())
        second = worker.run_async(handle, ChipStorageTaskArgs(), CallConfig())

        assert isinstance(first, RunHandle)
        assert fake.entered.wait(timeout=2.0)
        assert not second.completed
        fake.release_first.set()

        assert first.wait().host_wall_ns == 1
        assert second.wait().host_wall_ns == 2
        assert [entry[:2] for entry in fake.calls] == [
            ("start", 0),
            ("end", 0),
            ("start", 1),
            ("end", 1),
        ]
    finally:
        worker._stop_l2_run_lane()


def test_l2_sync_run_waits_behind_prior_async_run():
    worker, handle, fake = _make_l2_worker()
    sync_timing = []
    try:
        first = worker.run_async(handle, ChipStorageTaskArgs(), CallConfig())
        assert fake.entered.wait(timeout=2.0)

        sync_thread = threading.Thread(
            target=lambda: sync_timing.append(worker.run(handle, ChipStorageTaskArgs(), CallConfig()))
        )
        sync_thread.start()
        assert not sync_timing

        fake.release_first.set()
        assert first.wait().host_wall_ns == 1
        sync_thread.join(timeout=2.0)
        assert not sync_thread.is_alive()
        assert sync_timing[0].host_wall_ns == 2
        assert [entry[:2] for entry in fake.calls] == [
            ("start", 0),
            ("end", 0),
            ("start", 1),
            ("end", 1),
        ]
    finally:
        worker._stop_l2_run_lane()


def test_l2_unregister_async_tombstones_and_defers_free_until_run_finishes():
    worker, handle, fake = _make_l2_worker()
    try:
        first = worker.run_async(handle, ChipStorageTaskArgs(), CallConfig())
        assert fake.entered.wait(timeout=2.0)

        unreg = worker.unregister_async(handle)
        assert not unreg.completed
        try:
            worker.run_async(handle, ChipStorageTaskArgs(), CallConfig())
        except KeyError:
            pass
        else:
            raise AssertionError("tombstoned handle should reject new runs")

        fake.release_first.set()
        assert first.wait().host_wall_ns == 1
        unreg.wait()
        assert fake.unregisters == [0]
    finally:
        worker._stop_l2_run_lane()


def test_async_register_unregister_reject_non_chip_targets():
    worker = Worker(level=3, device_ids=[0])
    try:
        worker.register_async(lambda: None)
    except TypeError:
        pass
    else:
        raise AssertionError("register_async should reject non-ChipCallable targets")

    try:
        worker.unregister_async(object())
    except TypeError:
        pass
    else:
        raise AssertionError("unregister_async should reject non-ChipCallable handles")


def test_l3_run_async_runs_dag_on_worker_queue():
    worker = Worker(level=3, device_ids=[0])
    worker._initialized = True
    entered = threading.Event()
    release_first = threading.Event()
    calls = []
    run_count = 0

    def fake_run_dag(orch_fn, args, config):
        nonlocal run_count
        index = run_count
        run_count += 1
        calls.append(("start", index, orch_fn, args, config))
        if index == 0:
            entered.set()
            release_first.wait(timeout=2.0)
        calls.append(("end", index))
        return RunTiming(index + 10, 0)

    worker._run_dag_sync_impl = fake_run_dag
    try:
        first = worker.run_async(lambda orch, args, cfg: None, "first", CallConfig())
        assert isinstance(first, RunHandle)
        assert entered.wait(timeout=2.0)

        sync_result = []
        sync_thread = threading.Thread(
            target=lambda: sync_result.append(worker.run(lambda orch, args, cfg: None, "second", CallConfig()))
        )
        sync_thread.start()
        assert not sync_result

        release_first.set()
        assert first.wait().host_wall_ns == 10
        sync_thread.join(timeout=2.0)
        assert not sync_thread.is_alive()
        assert sync_result[0].host_wall_ns == 11
        assert [entry[:2] for entry in calls] == [
            ("start", 0),
            ("end", 0),
            ("start", 1),
            ("end", 1),
        ]
    finally:
        worker._stop_dag_run_lane()


def test_l3_run_async_does_not_accept_worker_keyword():
    worker = Worker(level=3, device_ids=[0])
    worker._initialized = True
    try:
        try:
            worker.run_async(lambda orch, args, cfg: None, worker=0)  # type: ignore[call-arg]
        except TypeError:
            pass
        else:
            raise AssertionError("L3 public run_async must not expose direct chip worker selection")
    finally:
        worker._stop_dag_run_lane()


def test_l3_register_async_returns_handle_after_remote_wait():
    worker = Worker(level=3, device_ids=[0])

    class FakeWorker:
        def __init__(self):
            self.broadcasts = []
            self.waits = []

        def broadcast_register_async_all(self, blob_ptr, blob_size, digest):
            self.broadcasts.append((blob_ptr, blob_size, digest))
            return [SimpleNamespace(worker_type="NEXT_LEVEL", worker_id=0, ok=True, remote_handle=23, error_message="")]

        def control_wait_register(self, worker_id, handle_id):
            self.waits.append((worker_id, handle_id))

    fake = FakeWorker()
    worker._initialized = True
    worker._hierarchical_started = True
    worker._hierarchical_start_state = "started"
    worker._worker = fake

    pending = worker.register_async(_chip_callable())
    assert isinstance(pending, RegisterHandle)
    assert not pending.completed

    handle = pending.wait()
    assert pending.completed
    assert fake.broadcasts[0][2] == handle.digest
    assert fake.waits == [(0, 23)]
