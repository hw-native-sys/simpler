# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CI gates for host-device mapped-region protocol examples."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "a2a3" / "tensormap_and_ringbuffer"
SPSC_EXAMPLE = EXAMPLE_ROOT / "host_device_spsc_queue_protocol" / "main.py"
SHM_EXAMPLE = EXAMPLE_ROOT / "host_device_shm_buffer_protocol" / "main.py"


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


def _run_example(args: list[str], timeout: int = 300) -> None:
    result = subprocess.run(
        [sys.executable, *args],
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
        env=_subprocess_env(),
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.requires_hardware("a2a3")
@pytest.mark.platforms(["a2a3"])
def test_a2a3_onboard_spsc_queue_protocol_smoke(st_device_ids):
    _run_example(
        [
            str(SPSC_EXAMPLE),
            "-p",
            "a2a3",
            "-d",
            str(int(st_device_ids[0])),
            "--epochs",
            "2",
            "--messages-per-epoch",
            "2",
        ]
    )


@pytest.mark.requires_hardware("a2a3")
@pytest.mark.platforms(["a2a3"])
def test_a2a3_onboard_shm_buffer_protocol_smoke(st_device_ids):
    _run_example(
        [
            str(SHM_EXAMPLE),
            "-p",
            "a2a3",
            "-d",
            str(int(st_device_ids[0])),
            "--epochs",
            "2",
            "--payload-bytes",
            "4096",
        ]
    )
