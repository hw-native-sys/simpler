#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L2 Worker memory-control demo — malloc / copy_to / copy_from / free, no kernels.

This example exercises the four host<->device memory primitives on the
``Worker`` API in isolation, without ever calling ``worker.run()``:

  * ``worker.malloc(nbytes)``   — allocate device memory, returns uint64 ptr
  * ``worker.copy_to(dst, src, n)`` — H2D byte copy
  * ``worker.copy_from(dst, src, n)`` — D2H byte copy
  * ``worker.free(ptr)``        — release device memory

Why a standalone example for these? On real hardware (a2a3 / a5 onboard) the
CANN device context is per-thread, so ``rtMalloc`` only succeeds on a thread
that previously executed ``rtSetDevice``. Until you call ``worker.run(...)``
the only thing that has bound the device on the calling Python thread is
``Worker.init() -> ChipWorker::set_device(...)``. If that path is broken,
this example fails at the first ``worker.malloc`` with CANN error 107002.
``vector_add`` happens to mask that bug because its first malloc lands on
the same thread that ``run()`` later attaches; this example doesn't ``run``
at all, so it's a focused regression check for the standalone alloc path.

Run:
    python examples/workers/l2/worker_malloc/main.py -p a2a3sim -d 0
    python examples/workers/l2/worker_malloc/main.py -p a2a3   -d 0
"""

import argparse
import ctypes
import sys

from simpler.worker import Worker

# Mix of sizes — a small one to exercise the small-bin path of the allocator,
# a page-aligned one, and one that is intentionally not a multiple of any
# common alignment to make sure copy_to/copy_from honor the byte count.
SIZES = [4096, 65536, 12345]


def parse_args() -> argparse.Namespace:
    """Same CLI shape as every example under ``examples/workers/``."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-p",
        "--platform",
        required=True,
        choices=["a2a3sim", "a2a3", "a5sim", "a5"],
        help="Target platform.",
    )
    parser.add_argument("-d", "--device", type=int, default=0, help="Device id to bind this L2 worker to.")
    return parser.parse_args()


def _make_pattern(nbytes: int, seed: int) -> bytes:
    """Deterministic byte pattern of length nbytes — round-trip will compare byte-perfect."""
    # Cheap PRNG-ish pattern; the exact bytes don't matter, we just need the
    # H2D-write and D2H-read to compare equal byte-for-byte.
    return bytes(((seed + i) * 131 + i * i) & 0xFF for i in range(nbytes))


def _round_trip(worker: Worker, nbytes: int, seed: int) -> None:
    """malloc -> H2D -> D2H -> compare -> free, with byte-exact assertions."""
    src = _make_pattern(nbytes, seed)
    dst = bytearray(nbytes)  # zero-filled; D2H must overwrite every byte

    src_buf = (ctypes.c_uint8 * nbytes).from_buffer_copy(src)
    dst_buf = (ctypes.c_uint8 * nbytes).from_buffer(dst)

    dev_ptr = worker.malloc(nbytes)
    assert dev_ptr != 0, f"malloc({nbytes}) returned NULL"
    try:
        worker.copy_to(dev_ptr, ctypes.addressof(src_buf), nbytes)
        worker.copy_from(ctypes.addressof(dst_buf), dev_ptr, nbytes)
    finally:
        worker.free(dev_ptr)

    if bytes(dst) != src:
        # Find first diverging byte to make the failure actionable.
        first_diff = next(i for i, (a, b) in enumerate(zip(src, dst)) if a != b)
        raise AssertionError(
            f"round-trip mismatch at byte {first_diff}/{nbytes}: "
            f"sent 0x{src[first_diff]:02x}, got 0x{dst[first_diff]:02x}"
        )
    print(f"[worker_malloc]   {nbytes:>6} bytes round-trip OK (ptr=0x{dev_ptr:x})")


def run(platform: str, device_id: int) -> int:
    """Core logic — callable from both CLI and pytest."""
    worker = Worker(
        level=2,
        platform=platform,
        runtime="tensormap_and_ringbuffer",
        device_id=device_id,
    )

    print(f"[worker_malloc] init on {platform} device={device_id} ...")
    worker.init()
    try:
        # 1. Each size, on its own — proves the basic primitive works.
        print("[worker_malloc] single-buffer round trips:")
        for i, n in enumerate(SIZES):
            _round_trip(worker, n, seed=i)

        # 2. Hold several allocations live concurrently — exercises the
        # allocator's free-list bookkeeping (no buffer aliasing).
        print("[worker_malloc] concurrent live allocations:")
        ptrs = [worker.malloc(n) for n in SIZES]
        assert all(p != 0 for p in ptrs), "concurrent malloc returned NULL"
        assert len(set(ptrs)) == len(ptrs), f"allocator handed out duplicate pointers: {ptrs}"
        for p in ptrs:
            worker.free(p)
        print(f"[worker_malloc]   {len(ptrs)} concurrent buffers, all distinct, freed cleanly")

        # 3. Repeat the cycle — confirms free actually returns the slab so
        # we don't OOM after a handful of iterations.
        print("[worker_malloc] alloc/free churn:")
        for _ in range(8):
            p = worker.malloc(SIZES[0])
            worker.free(p)
        print(f"[worker_malloc]   8x alloc/free of {SIZES[0]} bytes OK")
    finally:
        worker.close()
        print("[worker_malloc] close OK.")

    return 0


def main() -> int:
    args = parse_args()
    return run(args.platform, args.device)


if __name__ == "__main__":
    sys.exit(main())
