#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Private delegated-region fault injection for reply loss, timeout, child death, and fatal teardown."""

from __future__ import annotations

import os
import signal

import pytest
from simpler.worker import Worker

from ._helpers import (
    _RUNTIME,
    FAULT_PLATFORMS,
    _install_release_probe,
    _stream_config,
    build_chip_callable,
    close_owned_workers,
    create_live_region,
    make_l3_forward,
)

pytestmark = [
    pytest.mark.runtime_product("program"),
    pytest.mark.execution_mode("eager"),
]


def _l3(platform: str, device_id: int) -> Worker:
    worker = Worker(
        level=3,
        device_ids=[int(device_id)],
        num_sub_workers=0,
        platform=platform,
        runtime=_RUNTIME,
    )
    try:
        worker.register(build_chip_callable(platform))
        worker.init()
        return worker
    except BaseException as primary:
        close_owned_workers(primary, worker)
        raise


def _l4(platform: str, device_id: int) -> Worker:
    l3 = Worker(
        level=3,
        device_ids=[int(device_id)],
        num_sub_workers=0,
        platform=platform,
        runtime=_RUNTIME,
    )
    worker = None
    try:
        chip = build_chip_callable(platform)
        chip_handle = l3.register(chip)
        worker = Worker(level=4, num_sub_workers=0, platform=platform, runtime=_RUNTIME)
        worker.add_worker(l3)
        worker.register(chip)
        worker.register(make_l3_forward(chip_handle))
        worker.init()
        return worker
    except BaseException as primary:
        close_owned_workers(primary, worker, l3)
        raise


def _create(orch_handle, provider_path: str):
    return create_live_region(orch_handle, provider_path)


def _exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    return chain


def _run_expect_fatal(worker: Worker, orch) -> None:
    with pytest.raises(BaseException) as excinfo:  # noqa: PT011
        worker.run(orch, args=None, config=_stream_config())
    fatal = worker._delegated_session_fatal
    assert fatal is not None
    assert any(item is fatal for item in _exception_chain(excinfo.value))


def _assert_fatal_no_release(worker: Worker, releases: list[dict[str, object]]) -> None:
    assert worker._delegated_session_fatal is not None
    assert worker._region_instance_registry._delegated_admission_closed is True
    live = [instance for instance in worker._region_instance_registry._instances.values() if instance._ever_live]
    assert live == []
    assert releases == []


@pytest.mark.platforms(FAULT_PLATFORMS)
@pytest.mark.device_count(1)
@pytest.mark.runtime(_RUNTIME)
def test_malformed_reply_does_not_release_and_latches_fatal(st_platform, st_device_ids):
    worker = _l3(st_platform, int(st_device_ids[0]))
    releases = _install_release_probe(worker)
    original = worker._exchange_delegated_region

    def _smash(staged) -> None:
        original(staged)
        staged[:4] = b"XXXX"

    worker._exchange_delegated_region = _smash
    try:

        def orch(orch_handle, _args, cfg):
            _create(orch_handle, "L3/L2[0]")

        _run_expect_fatal(worker, orch)
        _assert_fatal_no_release(worker, releases)
    finally:
        worker.close()
    assert worker._region_instance_registry._instances == {}


@pytest.mark.platforms(FAULT_PLATFORMS)
@pytest.mark.device_count(1)
@pytest.mark.runtime(_RUNTIME)
def test_reply_loss_after_control_done_does_not_release(st_platform, st_device_ids):
    worker = _l3(st_platform, int(st_device_ids[0]))
    releases = _install_release_probe(worker)
    original = worker._exchange_delegated_region

    def _lose(staged) -> None:
        original(staged)
        raise RuntimeError("delegated region reply lost after CONTROL_DONE")

    worker._exchange_delegated_region = _lose
    try:

        def orch(orch_handle, _args, cfg):
            _create(orch_handle, "L3/L2[0]")

        _run_expect_fatal(worker, orch)
        _assert_fatal_no_release(worker, releases)
    finally:
        worker.close()
    assert worker._region_instance_registry._instances == {}


@pytest.mark.platforms(FAULT_PLATFORMS)
@pytest.mark.device_count(1)
@pytest.mark.runtime(_RUNTIME)
def test_timeout_before_side_effect_does_not_release(st_platform, st_device_ids):
    worker = _l3(st_platform, int(st_device_ids[0]))
    releases = _install_release_probe(worker)

    def _timeout(_staged) -> None:
        raise RuntimeError("delegated_region timed out waiting for CONTROL_DONE")

    worker._exchange_delegated_region = _timeout
    try:

        def orch(orch_handle, _args, cfg):
            _create(orch_handle, "L3/L2[0]")

        _run_expect_fatal(worker, orch)
        _assert_fatal_no_release(worker, releases)
    finally:
        worker.close()


@pytest.mark.platforms(FAULT_PLATFORMS)
@pytest.mark.device_count(1)
@pytest.mark.runtime(_RUNTIME)
def test_chip_death_latches_fatal_and_close_tears_down_tree(st_platform, st_device_ids):
    worker = _l3(st_platform, int(st_device_ids[0]))
    releases = _install_release_probe(worker)
    pid = int(worker._chip_pids[0])
    os.kill(pid, signal.SIGKILL)
    try:

        def orch(orch_handle, _args, cfg):
            _create(orch_handle, "L3/L2[0]")

        _run_expect_fatal(worker, orch)
        _assert_fatal_no_release(worker, releases)
    finally:
        worker.close()


@pytest.mark.platforms(FAULT_PLATFORMS)
@pytest.mark.device_count(1)
@pytest.mark.runtime(_RUNTIME)
def test_l4_child_death_latches_fatal_without_uncertainty_release(st_platform, st_device_ids):
    worker = _l4(st_platform, int(st_device_ids[0]))
    releases = _install_release_probe(worker)
    pid = int(worker._next_level_pids[0])
    os.kill(pid, signal.SIGKILL)
    try:

        def orch(orch_handle, _args, cfg):
            _create(orch_handle, "L4/L3[0]/L2[0]")

        _run_expect_fatal(worker, orch)
        _assert_fatal_no_release(worker, releases)
    finally:
        worker.close()
