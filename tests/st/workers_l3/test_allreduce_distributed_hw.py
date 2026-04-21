# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Hardware ST for examples/workers/l3/allreduce_distributed.

Exercises the full L1a..L6 distributed stack end-to-end on 2 Ascend devices:

  * L6: ``Worker(level=3, chip_bootstrap_configs=[...])`` eager bootstrap
  * L5: ``ChipWorker.bootstrap_context`` per-chip bring-up
  * L2: ``ChipBootstrapChannel`` SUCCESS publishing
  * L1a: HCCL ``comm_init`` / ``comm_alloc_windows``
  * Kernel: cross-rank MTE2 via ``CommRemotePtr(ctx, input, pe)``

Uses the ``st_device_ids`` fixture (via ``device_count(2)``) so the test
plays nicely with the device pool, matching ``test_worker_distributed_hw.py``.
"""

from __future__ import annotations

import os
import sys

import pytest

EXAMPLE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", "workers", "l3", "allreduce_distributed")
)


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(2)
def test_allreduce_distributed(st_device_ids):
    assert len(st_device_ids) >= 2, "device_count(2) fixture must yield >= 2 ids"

    # Import the example's main() by path (not a package). Matches the
    # "each example is self-contained, tests point at it" pattern used by
    # tests/st/a2a3/.../test_l3_*.py.
    sys.path.insert(0, EXAMPLE_DIR)
    try:
        import main as example_main  # noqa: PLC0415
    finally:
        sys.path.pop(0)

    argv = ["main.py", "-d", f"{int(st_device_ids[0])}-{int(st_device_ids[1])}"]
    saved = sys.argv
    sys.argv = argv
    try:
        rc = example_main.main()
    finally:
        sys.argv = saved

    assert rc == 0, f"allreduce_distributed main() returned {rc}"
