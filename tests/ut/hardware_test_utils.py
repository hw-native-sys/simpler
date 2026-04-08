import os
import re
import subprocess


def get_test_device_id(default: str = "0") -> str:
    """Pick a hardware test device.

    Respect PTO_TEST_DEVICE_ID when explicitly provided. Otherwise prefer the
    lowest-ID NPU that reports no running processes in `npu-smi info`, which is
    more stable than blindly defaulting to device 0 on shared machines.
    """

    configured = os.environ.get("PTO_TEST_DEVICE_ID")
    if configured:
        return configured

    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return default

    if result.returncode != 0:
        return default

    free_devices = sorted({int(match) for match in re.findall(r"No running processes found in NPU (\d+)", result.stdout)})
    if free_devices:
        return str(free_devices[0])
    return default
