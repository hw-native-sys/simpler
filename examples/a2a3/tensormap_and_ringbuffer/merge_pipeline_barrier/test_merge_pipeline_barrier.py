#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Merged 3-stage pipeline as ONE SPMD task with an intra-task cross-core barrier.

Three stages (x+1 -> *2 -> +1 = 2x+3) run as a single block_num=8 AIV task; the
inter-stage ordering is an intra-task cross-core barrier instead of separate
scheduled tasks. Stage A does 8 tiles on block 0 only, stage B 2 tiles each on
blocks 0..3, stage C 1 tile each on all 8; every block hits every barrier (idle
blocks still arrive), so each barrier waits for block_num.

The barrier has two compile-time variants (MERGE_BARRIER_COUNTER in
merge_pipeline.cpp): a per-slot st_dev poll (O(n) non-cacheable reads) and a
single-counter st_atomic+dcci arrival + one-ld_dev poll (O(1) read). The kernel
records each block's per-barrier gap with get_sys_cnt; the hook below prints
them (ticks at 50 MHz -> us) and skips golden-checking the diagnostic output.
"""

import sys

import torch
from simpler.task_interface import ArgDirection as D

import simpler_setup.scene_test  # noqa: F401
from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test

TILES = 8
M = 128
SIZE = TILES * M  # 1024
NBLK = 8

# Print the kernel's per-block timing (get_sys_cnt deltas, ticks->us at 50 MHz)
# and skip golden-checking the diagnostic 'timing' output. Each block's slots are
# strided by 32 int32 (one cacheline) because scalar st_dev only commits the
# first 8 int32 of a 128B line.
_st = sys.modules["simpler_setup.scene_test"]
_orig_cmp = _st._compare_outputs


def _cmp_with_gaps(test_args, golden_args, output_names, rtol, atol):
    if "timing" in output_names:
        tm = test_args.timing.reshape(NBLK, 32)[:, :6].to(torch.float64) / 50.0
        labels = ["segA", "bar1", "segB", "bar2", "segC", "total"]
        print("[GAP] per-block us (50 MHz counter):")
        print("  blk " + " ".join(f"{lbl:>7}" for lbl in labels))
        for b in range(NBLK):
            print(f"  {b:>3} " + " ".join(f"{tm[b, k]:7.2f}" for k in range(6)))
        print(
            f"[GAP] barrier1: max={tm[:, 1].max():.2f} min={tm[:, 1].min():.2f} us | "
            f"barrier2: max={tm[:, 3].max():.2f} min={tm[:, 3].min():.2f} us"
        )
        output_names = [n for n in output_names if n != "timing"]
    return _orig_cmp(test_args, golden_args, output_names, rtol, atol)


_st._compare_outputs = _cmp_with_gaps


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestMergePipelineBarrier(SceneTestCase):
    """out = 2x + 3 across 3 stages, ordered by an intra-task cross-core barrier."""

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/merge_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aiv/merge_pipeline.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT, D.OUT, D.OUT, D.OUT],
            },
        ],
    }

    CASES = [
        {
            # Onboard only: the busy-wait barrier needs all block_num logical
            # blocks co-resident, so config block_dim (24) > block_num (8, set
            # per-task via launch_spec.set_block_num in merge_orch.cpp).
            "name": "merge",
            "platforms": ["a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {},
        },
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            Tensor("x", torch.arange(SIZE, dtype=torch.float32) / 7.0),
            Tensor("sync", torch.zeros(NBLK, dtype=torch.int32)),  # barrier slots / counter (zero-init)
            Tensor("out", torch.zeros(SIZE, dtype=torch.float32)),
            Tensor("timing", torch.zeros(NBLK * 32, dtype=torch.int32)),  # per-block gaps (32-int32 stride)
        )

    def compute_golden(self, args, params):
        args.out[:] = 2.0 * args.x + 3.0


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
