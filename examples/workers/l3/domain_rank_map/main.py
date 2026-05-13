#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Minimal L3 multi-communication-domain rank-map demo.

Three chip workers declare two overlapping domains:

  even = workers [0, 2]
  tail = workers [1, 2]

The example does not submit kernels.  It bootstraps the worker, prints the
domain-local rank view seen by each chip child, and verifies that:

  * non-member domains are absent and raise KeyError through ctx.domains;
  * worker_indices order defines the dense domain_rank;
  * overlapping domains have separate buffer pointers.

Run:
    python examples/workers/l3/domain_rank_map/main.py -p a2a3sim -d 0-2
    python examples/workers/l3/domain_rank_map/main.py -p a2a3    -d 0-2
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from simpler.task_interface import ChipBufferSpec, CommDomain, CommDomainPlan  # noqa: E402
from simpler.worker import Worker  # noqa: E402

DOMAINS = {
    "even": [0, 2],
    "tail": [1, 2],
}


def parse_device_range(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = (int(x) for x in spec.split("-"))
        ids = list(range(lo, hi + 1))
    else:
        ids = [int(spec)]
    if len(ids) != 3:
        raise ValueError(f"domain_rank_map needs exactly 3 devices, got {ids}")
    return ids


def _make_comm_plan() -> CommDomainPlan:
    return CommDomainPlan(
        domains=[
            CommDomain(
                name=name,
                worker_indices=worker_indices,
                window_size=4096,
                buffers=[
                    ChipBufferSpec(
                        name="scratch",
                        dtype="float32",
                        count=1,
                        nbytes=4,
                    ),
                ],
            )
            for name, worker_indices in DOMAINS.items()
        ]
    )


def _check_missing(ctx_domains: dict, name: str) -> bool:
    try:
        _ = ctx_domains[name]
    except KeyError:
        return True
    return False


def run(platform: str, device_ids: list[int]) -> int:
    print(f"[domain_rank_map] platform={platform} devices={device_ids}")

    worker = Worker(
        level=3,
        platform=platform,
        runtime="tensormap_and_ringbuffer",
        device_ids=device_ids,
        num_sub_workers=0,
        comm_plan=_make_comm_plan(),
    )

    try:
        print("[domain_rank_map] init worker...")
        worker.init()
        contexts = worker.chip_contexts

        expected = {
            0: {"even": 0},
            1: {"tail": 0},
            2: {"even": 1, "tail": 1},
        }
        ok = len(contexts) == 3

        for worker_idx, ctx in enumerate(contexts):
            actual_names = sorted(ctx.domains)
            print(f"[domain_rank_map] worker {worker_idx}: domains={actual_names}")

            expected_names = sorted(expected[worker_idx])
            if actual_names != expected_names:
                print(f"  expected domains {expected_names}, got {actual_names}")
                ok = False

            for name, expected_rank in expected[worker_idx].items():
                domain = ctx.domains[name]
                print(
                    f"  {name}: domain_rank={domain.domain_rank} "
                    f"domain_size={domain.domain_size} "
                    f"scratch=0x{domain.buffer_ptrs['scratch']:x}"
                )
                if domain.domain_rank != expected_rank or domain.domain_size != 2:
                    ok = False
                if domain.device_ctx == 0 or domain.buffer_ptrs["scratch"] == 0:
                    ok = False

        if not _check_missing(contexts[0].domains, "tail"):
            print("[domain_rank_map] worker 0 unexpectedly has tail domain")
            ok = False
        if not _check_missing(contexts[1].domains, "even"):
            print("[domain_rank_map] worker 1 unexpectedly has even domain")
            ok = False

        worker2 = contexts[2]
        if worker2.domains["even"].buffer_ptrs["scratch"] == worker2.domains["tail"].buffer_ptrs["scratch"]:
            print("[domain_rank_map] worker 2 domains share a scratch pointer")
            ok = False

        if not ok:
            print("[domain_rank_map] checks FAILED")
            return 1
        print("[domain_rank_map] rank-map checks PASSED")
        return 0
    finally:
        worker.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-p", "--platform", required=True, choices=["a2a3sim", "a2a3"])
    parser.add_argument("-d", "--device", default="0-2", help="Device range, e.g. '0-2'. Three chips required.")
    cli = parser.parse_args()
    return run(cli.platform, parse_device_range(cli.device))


if __name__ == "__main__":
    sys.exit(main())
