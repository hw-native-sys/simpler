# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""A managed chip child publishes its arenas at init and gives them back at close.

The one thing only a device can establish about this route: that a forked chip
child whose four regions have an owner completes the managed teardown its
parent has no view into. ``tensormap_and_ringbuffer`` is the runtime that sizes
its three arenas from ring config alone, so a ``prewarm_config`` commits them
during ``init`` — at the context epoch, holding no run reference. Close must
then return them through ``release_unreferenced`` and the terminal sweep, and a
failure there is a non-zero child exit the parent's reap reports.

The Python surface this leaves out — that no route names the retired keyword
and that its spelling survives — is in ``test_default_workspace_management.py``.
Neither file reaches the init entry's own refusals.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import traceback

import pytest

_RUNTIME = "tensormap_and_ringbuffer"


def _prewarmed_l3_entry(platform: str, runtime: str, device_id: int, result_queue) -> None:
    """Keep ACL init/reset out of pytest's process, as the other onboard cases do."""
    from simpler.task_interface import CallConfig
    from simpler.worker import Worker

    result: dict[str, object] = {}
    worker = None
    try:
        worker = Worker(level=3, device_ids=[device_id], num_sub_workers=0, platform=platform, runtime=runtime)
        config = CallConfig()
        config.runtime_env.ring_task_window = 64
        worker.init(prewarm_config=config)
        result["ready"] = True
    except BaseException:  # noqa: BLE001 -- marshal child failures back to pytest
        result["error"] = traceback.format_exc()
    finally:
        if worker is not None:
            try:
                worker.close()
                result["closed"] = True
            except BaseException:  # noqa: BLE001
                result["close_error"] = traceback.format_exc()
    result_queue.put(result)


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3", "a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(_RUNTIME)
def test_onboard_a_managed_chip_child_closes_after_prewarming_its_arenas(st_platform, st_device_ids):
    device_id = int(st_device_ids[0])
    from simpler_setup.runtime_builder import RuntimeBuilder

    try:
        RuntimeBuilder(platform=st_platform).get_binaries(_RUNTIME, build=bool(os.environ.get("PTO_UT_BUILD")))
    except FileNotFoundError as exc:
        pytest.skip(f"{st_platform}/{_RUNTIME} runtime binaries unavailable: {exc}")

    ctx = mp.get_context("fork")
    result_queue = ctx.Queue()
    process = ctx.Process(target=_prewarmed_l3_entry, args=(st_platform, _RUNTIME, device_id, result_queue))
    process.start()
    process.join(timeout=300)
    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
        pytest.fail("managed chip child did not finish")
    assert process.exitcode == 0, f"managed chip child exited with {process.exitcode}"
    result = result_queue.get(timeout=5)

    assert "error" not in result, result.get("error")
    assert result["ready"] is True
    # The managed terminal sequence ran to completion inside each child: a
    # release it could not prove would have failed the teardown and reached
    # here as a non-zero exit the reap reports.
    assert "close_error" not in result, result.get("close_error")
    assert result["closed"] is True


def _default_install_entry(platform: str, runtime: str, device_id: int, result_queue) -> None:
    """Report what a context that asked for nothing ended up with."""
    from simpler.worker import Worker

    result: dict[str, object] = {}
    worker = None
    try:
        # No budget, no keyword, nothing staged: the plain construction.
        worker = Worker(level=2, device_id=device_id, platform=platform, runtime=runtime)
        worker.init()
        assert worker._chip_worker is not None
        status, report = worker._chip_worker._impl.workspace_report()
        result["status"] = status
        result["budget_enforced"] = None if report is None else int(report["budget_enforced"])
        result["limit_bytes"] = None if report is None else int(report["limit_bytes"])
    except BaseException:  # noqa: BLE001 -- marshal child failures back to pytest
        result["error"] = traceback.format_exc()
    finally:
        if worker is not None:
            try:
                worker.close()
                result["closed"] = True
            except BaseException:  # noqa: BLE001
                result["close_error"] = traceback.format_exc()
    result_queue.put(result)


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3", "a5"])
@pytest.mark.device_count(1)
# One `runtime` mark per parameter, carrying that parameter's own value. The
# dispatcher reads this mark to build the job, and the child it launches passes
# `--runtime` back through the isolation filter — so a single function-level
# mark would deselect the other parameter in its own child and pass having run
# nothing.
@pytest.mark.parametrize(
    "runtime",
    [
        pytest.param("host_build_graph", marks=pytest.mark.runtime("host_build_graph")),
        pytest.param("tensormap_and_ringbuffer", marks=pytest.mark.runtime("tensormap_and_ringbuffer")),
    ],
)
def test_onboard_a_context_that_asked_for_nothing_is_managed(st_platform, st_device_ids, runtime):
    """The default, on a real device: a program context nobody configured comes
    up with its four regions owned and no byte limit.

    ``available`` is the answer only a managed context gives — an unmanaged one
    reports ``disabled`` — so this is the install itself, observed through the
    accessor rather than inferred from a route.
    """
    device_id = int(st_device_ids[0])
    from simpler_setup.runtime_builder import RuntimeBuilder

    try:
        RuntimeBuilder(platform=st_platform).get_binaries(runtime, build=bool(os.environ.get("PTO_UT_BUILD")))
    except FileNotFoundError as exc:
        pytest.skip(f"{st_platform}/{runtime} runtime binaries unavailable: {exc}")

    ctx = mp.get_context("fork")
    result_queue = ctx.Queue()
    process = ctx.Process(target=_default_install_entry, args=(st_platform, runtime, device_id, result_queue))
    process.start()
    process.join(timeout=300)
    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
        pytest.fail("default-install child did not finish")
    assert process.exitcode == 0, f"default-install child exited with {process.exitcode}"
    result = result_queue.get(timeout=5)

    assert "error" not in result, result.get("error")
    assert result["status"] == "available"
    # Ownership without a limit: the budget stayed optional and unasked.
    assert result["budget_enforced"] == 0
    assert result["limit_bytes"] == 0
    assert "close_error" not in result, result.get("close_error")
    assert result["closed"] is True
