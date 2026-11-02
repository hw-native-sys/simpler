# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Empty Worker lifecycle probe; run inside an allocated hardware task."""

import argparse
import json
import os
from pathlib import Path


def snapshot(stage):
    task_dir = Path("/proc/self/task")
    print(
        json.dumps(
            {
                "stage": stage,
                "pid": os.getpid(),
                "threads": len(list(task_dir.iterdir())) if task_dir.exists() else None,
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("a2a3", "a5"), required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--prefix", type=int, choices=(0, 1, 2), required=True)
    options = parser.parse_args()
    devices = [int(item) for item in options.devices.split(",")]
    from simpler.worker import Worker  # noqa: PLC0415 -- keep --help independent of the native installation

    def cycle(label, **config):
        worker = Worker(platform=options.platform, runtime="tensormap_and_ringbuffer", **config)
        try:
            snapshot(label + " INIT_BEGIN")
            worker.init()
            snapshot(label + " INIT_OK")
        finally:
            snapshot(label + " CLOSE_BEGIN")
            worker.close()
            snapshot(label + " CLOSE_OK")

    for index in range(options.prefix):
        cycle(f"L2_{index + 1}", level=2, device_id=devices[0])
    cycle("L3", level=3, device_ids=devices, num_sub_workers=0, startup_timeout_s=120, heap_ring_size=0)
    print("PROBE_PASS", flush=True)


if __name__ == "__main__":
    main()
