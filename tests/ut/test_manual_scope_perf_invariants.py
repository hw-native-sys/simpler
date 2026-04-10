import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hardware_test_utils import get_test_device_id


PROJECT_ROOT = Path(__file__).parent.parent.parent
RUN_EXAMPLE = PROJECT_ROOT / "examples" / "scripts" / "run_example.py"
KERNELS_DIR = (
    PROJECT_ROOT / "tests" / "st" / "a2a3" / "tensormap_and_ringbuffer" / "manual_scope_outer_multiwrite" / "kernels"
)
GOLDEN = PROJECT_ROOT / "tests" / "st" / "a2a3" / "tensormap_and_ringbuffer" / "manual_scope_outer_multiwrite" / "golden.py"
PTO_ISA_COMMIT = "d96c8784"


def _device_log_dir(device_id: str) -> Path:
    log_dir = Path.home() / "ascend" / "log" / "debug" / f"device-{device_id}"
    if os.getenv("ASCEND_WORK_PATH"):
        work_log_dir = Path(os.environ["ASCEND_WORK_PATH"]).expanduser() / "log" / "debug" / f"device-{device_id}"
        if work_log_dir.exists():
            return work_log_dir
    return log_dir


def _run_manual_scope_outer_multiwrite(extra_env: dict[str, str]) -> tuple[subprocess.CompletedProcess[str], str]:
    device_id = get_test_device_id()
    log_dir = _device_log_dir(device_id)
    before_logs = set(log_dir.glob("*.log")) if log_dir.exists() else set()

    env_prefix = " ".join(f"{key}={value}" for key, value in extra_env.items())
    command = (
        f"source {os.environ['ASCEND_HOME_PATH']}/bin/setenv.bash >/dev/null 2>&1 && "
        f"{env_prefix} "
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
        if logs:
            new_log = max(logs, key=lambda path: path.stat().st_mtime)

    log_text = ""
    if new_log is not None:
        log_text = new_log.read_text(encoding="utf-8", errors="ignore")

    return result, result.stdout + result.stderr + log_text


@pytest.mark.requires_hardware
@pytest.mark.skipif(not os.getenv("ASCEND_HOME_PATH"), reason="ASCEND_HOME_PATH not set; Ascend toolkit required")
def test_manual_scope_tail_consumer_path_keeps_fast_publish():
    result, combined_text = _run_manual_scope_outer_multiwrite(
        {
            "PTO2_DEBUG_DUMP_MANUAL_SCOPE": "1",
            "PTO2_EXPECT_MANUAL_SCOPE_REPAIR": "0",
        }
    )

    assert result.returncode == 0, combined_text
    assert "manual_scope_repair_needed=0" in combined_text


@pytest.mark.requires_hardware
@pytest.mark.skipif(not os.getenv("ASCEND_HOME_PATH"), reason="ASCEND_HOME_PATH not set; Ascend toolkit required")
def test_manual_scope_retroactive_edge_enables_repair_fallback():
    result, combined_text = _run_manual_scope_outer_multiwrite(
        {
            "PTO2_DEBUG_DUMP_MANUAL_SCOPE": "1",
            "PTO2_DEBUG_FORCE_RETROACTIVE_MANUAL_EDGE": "1",
            "PTO2_EXPECT_MANUAL_SCOPE_REPAIR": "1",
        }
    )

    assert result.returncode == 0, combined_text
    assert "manual_scope_repair_needed=1" in combined_text
