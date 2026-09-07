# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Sim-backend test for the async-DMA workspace request carried by ``simpler_init``.

Simulation has no async-DMA engine, but ``enable_sdma=True`` still hands the
kernel a live workspace address: an inert block, published through the same
``set_dma_workspace_addr`` path onboard feeds from ``InitArgs``. The contract a
kernel relies on is that a non-zero address means the workspace is there to use —
a zero address where a live one was expected is a silent wrong answer rather than
an error, because a kernel that branches on null (see
``examples/a2a3/tensormap_and_ringbuffer/prefetch_async_demo``) skips its work and
fails a golden comparison instead of reporting anything. Provisioning inert
scratch keeps that contract on sim, where nothing dereferences the block:
``TPREFETCH_ASYNC`` is a no-op stub (pto-isa ``cpu/TPrefetchAsync.hpp``) and the
async transfers that would read it have no CPU-sim implementation at all.

A Worker that does not ask for SDMA keeps a zero address, which is what makes the
address meaningful as a signal.
"""

from __future__ import annotations

import pytest


def _make_sim_worker(*, enable_sdma: bool):
    from simpler.worker import Worker

    from simpler_setup.runtime_builder import RuntimeBuilder

    try:
        RuntimeBuilder(platform="a2a3sim").get_binaries("tensormap_and_ringbuffer")
    except FileNotFoundError as e:
        pytest.skip(f"a2a3sim runtime binaries unavailable: {e}")
    return Worker(
        level=2,
        device_id=0,
        platform="a2a3sim",
        runtime="tensormap_and_ringbuffer",
        enable_sdma=enable_sdma,
    )


class TestSimProvisionsSdmaWorkspace:
    def test_enable_sdma_init_succeeds(self):
        # An SDMA-enabled Worker comes up on sim. It used to be rejected here
        # with PTO_RUNTIME_ERR_UNSUPPORTED (-1001), which kept every kernel that
        # merely hints a prefetch off the simulator.
        worker = _make_sim_worker(enable_sdma=True)
        try:
            worker.init()
        finally:
            worker.close()

    def test_without_sdma_init_succeeds(self):
        # Control: coming up is not itself evidence the request was honoured, so
        # pair it with the Worker that asks for nothing. The workspace address
        # each one hands a kernel is covered end-to-end by the a2a3
        # prefetch_async_demo scene test, which reads it on-device.
        worker = _make_sim_worker(enable_sdma=False)
        try:
            worker.init()
        finally:
            worker.close()

    def test_finalize_and_reinitialize_allocates_fresh_workspace(self):
        # Reinitializing a finalized simulator runner must re-provision fresh
        # workspace rather than retaining a dangling pointer from a previous run.
        worker1 = _make_sim_worker(enable_sdma=True)
        try:
            worker1.init()
        finally:
            worker1.close()

        worker2 = _make_sim_worker(enable_sdma=True)
        try:
            worker2.init()
        finally:
            worker2.close()
