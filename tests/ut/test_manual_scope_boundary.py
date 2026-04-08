import os
import subprocess
import sys
from pathlib import Path

import pytest

from hardware_test_utils import get_test_device_id


PROJECT_ROOT = Path(__file__).parent.parent.parent
RUN_EXAMPLE = PROJECT_ROOT / "examples" / "scripts" / "run_example.py"
KERNELS_DIR = (
    PROJECT_ROOT / "tests" / "st" / "a2a3" / "tensormap_and_ringbuffer" / "manual_scope_outer_multiwrite" / "kernels"
)
GOLDEN = PROJECT_ROOT / "tests" / "st" / "a2a3" / "tensormap_and_ringbuffer" / "manual_scope_outer_multiwrite" / "golden.py"
PTO_ISA_COMMIT = "6622890"


@pytest.mark.requires_hardware
@pytest.mark.skipif(not os.getenv("ASCEND_HOME_PATH"), reason="ASCEND_HOME_PATH not set; Ascend toolkit required")
def test_manual_scope_outer_multiwrite_boundary():
    device_id = get_test_device_id()
    command = (
        f"source {os.environ['ASCEND_HOME_PATH']}/bin/setenv.bash >/dev/null 2>&1 && "
        f"{sys.executable} {RUN_EXAMPLE} --build --silent "
        f"-k {KERNELS_DIR} -g {GOLDEN} -p a2a3 -d {device_id} "
        f"--clone-protocol https -c {PTO_ISA_COMMIT}"
    )
    result = subprocess.run(
        ["bash", "-lc", command],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
