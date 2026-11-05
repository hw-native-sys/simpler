# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fresh-process controls for forked chip startup after direct L2 teardown.

Run under task-submit with a bounded task lifetime. The scheduler owns timeout
and descendant reclamation, including chip children in separate process groups.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.manual
@pytest.mark.platforms(["a2a3", "a5"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(2)
@pytest.mark.parametrize("prefix", [0, 1], ids=["fresh-l3", "l2-close-then-l3"])
def test_l2_then_l3(st_platform, st_device_ids, prefix, tmp_path):
    if not os.environ.get("TASKQUEUE_TASK_ID"):
        pytest.fail("This hardware lifecycle probe requires a bounded task-submit allocation")
    logs = tmp_path / "ascend"
    logs.mkdir()
    command = [
        sys.executable,
        "-X",
        "faulthandler",
        "-u",
        str(Path(__file__).with_name("probe_l2_then_l3.py")),
        "--platform",
        st_platform,
        "--devices",
        ",".join(str(device) for device in st_device_ids[:2]),
        "--prefix",
        str(prefix),
    ]
    with (tmp_path / "probe.log").open("w") as output:
        result = subprocess.run(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
            env={
                **os.environ,
                "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
                "PYTHONNOUSERSITE": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "ASCEND_PROCESS_LOG_PATH": str(logs),
            },
        )
    evidence = (tmp_path / "probe.log").read_text()
    print(evidence, end="", flush=True)
    assert result.returncode == 0, evidence
    assert "PROBE_PASS" in evidence, evidence
