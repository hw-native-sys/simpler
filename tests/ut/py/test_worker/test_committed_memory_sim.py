# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Sim-backend test for ``Worker.committed_device_memory()`` (level 2).

Exercises the full Python -> ChipWorker -> dlsym'd
``committed_device_memory_ctx`` -> SimDeviceRunnerBase -> MemoryAllocator stack
on the sim backend (no real hardware). Asserts the reported bytes rise after a
device malloc and fall after the matching free.
"""

from __future__ import annotations

import pytest


def _sim_binaries():
    """Resolve pre-built a2a3sim runtime binaries, or skip if unavailable."""
    from simpler_setup.runtime_builder import RuntimeBuilder

    try:
        bins = RuntimeBuilder(platform="a2a3sim").get_binaries("tensormap_and_ringbuffer")
    except FileNotFoundError as e:
        pytest.skip(f"a2a3sim runtime binaries unavailable: {e}")
    return bins


def _make_l2_worker():
    from simpler.worker import Worker

    _sim_binaries()  # trigger the skip on absence
    return Worker(
        level=2,
        device_id=0,
        platform="a2a3sim",
        runtime="tensormap_and_ringbuffer",
    )


class TestCommittedDeviceMemory:
    def test_simulator_refuses_vmm_storage_before_any_acl_access(self):
        worker = _make_l2_worker()
        try:
            worker.init()
            with pytest.raises(ValueError, match="onboard ChipWorker"):
                worker.malloc(256, exportable=True)
            ordinary = worker.malloc(256)
            worker.free(ordinary)
        finally:
            worker.close()

    def test_rises_on_malloc_and_falls_on_free(self):
        worker = _make_l2_worker()
        worker.init()
        try:
            baseline = worker.committed_device_memory()
            assert isinstance(baseline, int)
            assert baseline >= 0  # arenas lazily committed on sim (>=0); device path verified in serving (43.44 GB)

            size = 1 << 20  # 1 MiB
            handle = worker.malloc(size)
            assert handle.base != 0
            after_alloc = worker.committed_device_memory()
            # The allocator tracks the requested size, so the total must have
            # grown by at least the malloc'd bytes on top of the init baseline.
            assert after_alloc >= baseline + size, (baseline, after_alloc, size)

            worker.free(handle)
            after_free = worker.committed_device_memory()
            assert after_free <= after_alloc - size, (after_alloc, after_free, size)
        finally:
            worker.close()

    def test_l3_raises_not_implemented(self):
        from simpler.worker import Worker

        # Level-3 aggregation is deferred: the facade must refuse rather than
        # silently return a wrong (parent-side, HBM-less) number. Uses the
        # pure parent worker (num_sub_workers=0) — no sim binaries needed.
        worker = Worker(level=3, num_sub_workers=0)
        worker.init()
        try:
            with pytest.raises(NotImplementedError, match="requires at least one forked chip worker"):
                worker.committed_device_memory()
        finally:
            worker.close()
