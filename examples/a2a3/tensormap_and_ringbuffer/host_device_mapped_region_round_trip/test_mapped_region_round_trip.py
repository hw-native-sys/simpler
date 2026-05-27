# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pytest entrypoint for the HostDeviceMappedRegion round-trip example.

This example demonstrates host CPU to device NPU communication through a
``HostDeviceMappedRegion``. The host opens one mapped region, reuses it for 10
iterations, writes a sequence-dependent input pattern, notifies signal slot 0,
submits an AIV task with the returned device pointers, waits for signal slot 1,
then reads and checks the output bytes.

Run directly:

    python examples/a2a3/tensormap_and_ringbuffer/host_device_mapped_region_round_trip/main.py -p a2a3sim -d 0
    python examples/a2a3/tensormap_and_ringbuffer/host_device_mapped_region_round_trip/main.py -p a2a3 -d 0

Use ``--build`` to rebuild the runtime from source. Use ``--iters N`` to adjust
the number of reused-region iterations; support gating should keep the default
10 iterations.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(REPO_ROOT), str(REPO_ROOT / "python")]
    venv_lib = REPO_ROOT / ".venv" / "lib"
    if venv_lib.exists():
        paths.extend(str(p) for p in sorted(venv_lib.glob("python*/site-packages")))
    existing = env.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


@pytest.mark.platforms(["a2a3sim", "a2a3"])
def test_host_device_mapped_region_round_trip(request):
    platform = request.config.getoption("--platform", default=None) or "a2a3sim"
    device = request.config.getoption("--device", default=None)
    device_id = int(str(device).split(",")[0].split("-")[0]) if device is not None else 0
    result = subprocess.run(
        [sys.executable, str(HERE / "main.py"), "-p", platform, "-d", str(device_id)],
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
        env=_subprocess_env(),
    )
    assert result.returncode == 0, result.stdout + result.stderr
