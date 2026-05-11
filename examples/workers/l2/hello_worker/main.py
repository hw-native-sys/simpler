#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L2 Worker lifecycle demo — no kernels, no data, just init + close.

This is the smallest correct program that drives the Worker API. It proves:

  1. Your venv can import ``simpler.Worker`` (i.e. the nanobind extension is built).
  2. Pre-built runtime binaries exist under ``build/lib/<platform>/tensormap_and_ringbuffer/``
     so that ``RuntimeBuilder`` can find them on ``Worker.init()``.
  3. ``ChipWorker.init(device_id)`` + ACL init on the chosen platform works end-to-end.

If this example runs cleanly, moving on to ``vector_add/`` (which adds a real
kernel, TaskArgs, and a golden check) is safe.

Run:
    python examples/workers/l2/hello_worker/main.py -p a2a3sim -d 0
"""

import argparse
import sys

from simpler.worker import Worker


def parse_args() -> argparse.Namespace:
    """Parse the uniform CLI used by every example under ``examples/workers/``."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-p",
        "--platform",
        required=True,
        choices=["a2a3sim", "a2a3", "a5sim", "a5"],
        help="Target platform. 'sim' variants run on the CPU simulator and need no NPU.",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=int,
        default=0,
        help="Device id to bind this L2 worker to. Sim platforms accept any integer.",
    )
    return parser.parse_args()


def run(platform: str, device_id: int) -> int:
    """Core logic — callable from both CLI and pytest."""

    # Worker(level=2, ...) wraps a single C++ ChipWorker. Construction does NOT
    # load any binaries or touch the device — it just stashes config. The heavy
    # work happens in init().
    worker = Worker(
        level=2,
        platform=platform,
        runtime="tensormap_and_ringbuffer",
        device_id=device_id,
    )

    # init() resolves ``build/lib/<platform>/tensormap_and_ringbuffer/*`` via
    # RuntimeBuilder, dlopens host_runtime.so, loads aicpu.so + aicore.o, and
    # calls aclrtSetDevice(device_id). If any of those fails this raises.
    print(f"[hello_worker] init on {platform} device={device_id} ...")
    worker.init()

    try:
        # Nothing useful to do here — this example only proves the lifecycle.
        # A real L2 program would now malloc device buffers, build a
        # ChipCallable from compiled kernels, and call worker.run(...).
        #
        # We at least exercise the memory-control path: one malloc/free round
        # trip confirms the host<->device mailbox works.
        ptr = worker.malloc(4096)
        assert ptr != 0, "malloc returned NULL — device memory is exhausted or not mapped"
        worker.free(ptr)
        print(f"[hello_worker] malloc/free round-trip OK (ptr=0x{ptr:x})")
    finally:
        # close() releases the ACL device, unloads the runtime libraries, and
        # frees all per-worker C++ state. ALWAYS call it in a finally — a
        # leaked ChipWorker on a self-hosted runner poisons the next CI job
        # (see https://github.com/hw-native-sys/simpler/issues/604).
        worker.close()
        print("[hello_worker] close OK — lifecycle complete.")

    return 0


def main() -> int:
    args = parse_args()
    return run(args.platform, args.device)


if __name__ == "__main__":
    sys.exit(main())
