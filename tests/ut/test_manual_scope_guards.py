import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hardware_test_utils import get_test_device_id


PROJECT_ROOT = Path(__file__).parent.parent.parent
RUN_EXAMPLE = PROJECT_ROOT / "examples" / "scripts" / "run_example.py"
KERNELS_DIR = PROJECT_ROOT / "tests" / "st" / "a2a3" / "tensormap_and_ringbuffer" / "manual_scope_guard_negative" / "kernels"
GOLDEN = PROJECT_ROOT / "tests" / "st" / "a2a3" / "tensormap_and_ringbuffer" / "manual_scope_guard_negative" / "golden.py"
PTO_ISA_COMMIT = "d96c8784"


@pytest.mark.requires_hardware
@pytest.mark.skipif(not os.getenv("ASCEND_HOME_PATH"), reason="ASCEND_HOME_PATH not set; Ascend toolkit required")
@pytest.mark.parametrize(
    ("case_name", "expected_message"),
    [
        (
            "NestedManualScope",
            "nested scope inside PTO2_SCOPE(PTO2ScopeMode::MANUAL) is not supported in v1",
        ),
        (
            "NestedAutoInsideManual",
            "nested scope inside PTO2_SCOPE(PTO2ScopeMode::MANUAL) is not supported in v1",
        ),
        (
            "ManualGetTensorData",
            "blocking tensor data access is not supported inside PTO2_SCOPE(PTO2ScopeMode::MANUAL); exit the manual scope first",
        ),
        (
            "ManualSetTensorData",
            "blocking tensor data access is not supported inside PTO2_SCOPE(PTO2ScopeMode::MANUAL); exit the manual scope first",
        ),
        (
            "ManualSelfDependency",
            "add_dependency does not allow self-dependency",
        ),
    ],
)
def test_manual_scope_guard_failures(case_name, expected_message):
    device_id = get_test_device_id()
    log_dir = Path.home() / "ascend" / "log" / "debug" / f"device-{device_id}"
    if os.getenv("ASCEND_WORK_PATH"):
        work_log_dir = Path(os.environ["ASCEND_WORK_PATH"]).expanduser() / "log" / "debug" / f"device-{device_id}"
        if work_log_dir.exists():
            log_dir = work_log_dir
    before_logs = set(log_dir.glob("*.log")) if log_dir.exists() else set()
    command = (
        f"source {os.environ['ASCEND_HOME_PATH']}/bin/setenv.bash >/dev/null 2>&1 && "
        f"{sys.executable} {RUN_EXAMPLE} --build --silent "
        f"-k {KERNELS_DIR} -g {GOLDEN} -p a2a3 -d {device_id} "
        f"--case {case_name} --clone-protocol https -c {PTO_ISA_COMMIT}"
    )
    result = subprocess.run(
        ["bash", "-lc", command],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    combined_output = result.stdout + result.stderr

    new_log = None
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        current_logs = set(log_dir.glob("*.log")) if log_dir.exists() else set()
        created = current_logs - before_logs
        if created:
            new_log = max(created, key=lambda path: path.stat().st_mtime)
            break
        time.sleep(0.5)

    if new_log is None:
        logs = list(log_dir.glob("*.log")) if log_dir.exists() else []
        assert logs, "expected a device log for the failed manual-scope case"
        new_log = max(logs, key=lambda path: path.stat().st_mtime)

    log_text = new_log.read_text(encoding="utf-8", errors="ignore")
    assert expected_message in combined_output or expected_message in log_text
