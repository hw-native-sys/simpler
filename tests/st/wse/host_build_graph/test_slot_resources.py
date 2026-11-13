# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Two-slot carrier retention with explicit fixed-SRAM input and output views."""

import time

import pytest
import torch
from simpler.task_interface import CallConfig
from simpler.worker import Worker

from tests.st.wse.host_build_graph.fixed_sram_layout import release_view
from tests.st.wse.host_build_graph.vector_example.test_vector_example import build_vector, check_vector, vector_args

pytestmark = [pytest.mark.platforms(["wsesim"]), pytest.mark.runtime("host_build_graph"), pytest.mark.device_count(1)]


def test_two_slots_keep_distinct_carriers_and_fixed_outputs(st_platform):
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    pending = []
    try:
        worker.init()
        chip = worker._chip_worker
        assert chip.pipeline_depth == chip.runtime_slot_count == 2
        handle = worker.register(build_vector(st_platform))
        state = worker._resolve_handle(handle)
        addresses = []
        windows_all = []
        for index, slot in enumerate((0, 1, 0, 1)):
            args, windows, values, expected = vector_args(worker, index)
            windows_all.extend(windows)
            chip_args = worker._materialize_l2_args(args)
            run = chip._prepare_native_run_with_pipeline_lease(
                state.slot_id, chip_args, config=CallConfig(), slot_id=slot, generation=index + 1
            )
            pending.append((run, args, chip_args, windows, values, expected))
            storage = chip.runtime_buffer_addrs[slot]
            assert storage and storage not in addresses
            addresses.append(storage)
            chip._launch_native_run(run)
            deadline = time.monotonic() + 30
            while not chip._poll_native_run(run):
                assert time.monotonic() < deadline, "native execution did not retire"
                time.sleep(0)
            # Explicit device outputs do not overwrite the caller's Host mirror at poll.
            for _, _, _, _, retained, _ in pending:
                assert bool(torch.all(retained[2] < 0))
        while pending:
            run, _, _, windows, values, expected = pending[-1]
            chip._wait_native_run(run)
            chip._finalize_native_run(run)
            pending.pop()
            check_vector(worker, windows, values, expected)
        for window in windows_all:
            release_view(worker, window)
        worker.unregister(handle)
    finally:
        for run, *_ in reversed(pending):
            worker._chip_worker._wait_native_run(run)
            worker._chip_worker._finalize_native_run(run)
        worker.close()
