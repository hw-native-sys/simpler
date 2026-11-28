# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock, Thread

import pytest

from simpler_setup import compile_pool


def test_default_compile_workers_reserves_two_cpus_and_caps_at_eight(monkeypatch):
    monkeypatch.setattr(compile_pool, "_available_cpu_count", lambda: 4)
    assert compile_pool.default_compile_workers() == 2

    monkeypatch.setattr(compile_pool, "_available_cpu_count", lambda: 1)
    assert compile_pool.default_compile_workers() == 1

    monkeypatch.setattr(compile_pool, "_available_cpu_count", lambda: 3)
    assert compile_pool.default_compile_workers() == 1

    monkeypatch.setattr(compile_pool, "_available_cpu_count", lambda: 320)
    assert compile_pool.default_compile_workers() == 8


def test_available_cpu_count_uses_process_affinity(monkeypatch):
    monkeypatch.setattr(compile_pool.os, "sched_getaffinity", lambda _pid: set(range(4)), raising=False)
    monkeypatch.setattr(compile_pool.os, "cpu_count", lambda: 320)

    assert compile_pool._available_cpu_count() == 4


def test_available_cpu_count_falls_back_when_affinity_is_unavailable(monkeypatch):
    monkeypatch.delattr(compile_pool.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(compile_pool.os, "cpu_count", lambda: 4)
    assert compile_pool._available_cpu_count() == 4

    monkeypatch.setattr(compile_pool.os, "cpu_count", lambda: None)
    assert compile_pool._available_cpu_count() == 1


def test_compile_slots_bound_nested_thread_pools():
    first_two_started = Barrier(3, timeout=2)
    release = Event()
    state_lock = Lock()
    active = 0
    peak = 0

    def compile_one(index):
        nonlocal active, peak
        with compile_pool.compile_slot():
            with state_lock:
                active += 1
                peak = max(peak, active)
            if index < 2:
                first_two_started.wait()
            release.wait(timeout=2)
            with state_lock:
                active -= 1

    with compile_pool.compile_worker_budget(2):
        with ThreadPoolExecutor(max_workers=3) as outer:
            futures = [outer.submit(compile_one, index) for index in range(3)]
            first_two_started.wait()
            assert peak == 2
            release.set()
            for future in futures:
                future.result()

    assert peak == 2


def test_compile_worker_budget_rejects_nested_overrides():
    with compile_pool.compile_worker_budget(2), pytest.raises(RuntimeError, match="only one process-wide"):
        with compile_pool.compile_worker_budget(1):
            pass


def test_compile_worker_budget_rejects_concurrent_overrides():
    entered = Event()
    release = Event()

    def hold_override():
        with compile_pool.compile_worker_budget(2):
            entered.set()
            assert release.wait(timeout=2)

    worker = Thread(target=hold_override)
    worker.start()
    assert entered.wait(timeout=2)
    with pytest.raises(RuntimeError, match="only one process-wide"):
        with compile_pool.compile_worker_budget(1):
            pass
    release.set()
    worker.join(timeout=2)
    assert not worker.is_alive()


def test_budget_override_counts_already_held_slots():
    attempting = Event()
    admitted = Event()

    def compile_one():
        attempting.set()
        with compile_pool.compile_slot():
            admitted.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with compile_pool.compile_slot(), compile_pool.compile_worker_budget(1):
            future = pool.submit(compile_one)
            assert attempting.wait(timeout=2)
            assert not admitted.wait(timeout=0.2)
        future.result(timeout=2)
        assert admitted.is_set()
