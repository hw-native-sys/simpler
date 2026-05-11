#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""dep_gen capture sim test.

Re-runs the ``vector_example`` orchestration with ``--enable-dep-gen`` and
asserts that ``<output_prefix>/submit_trace.bin`` was produced with the
expected size for the 5 ``submit_task`` calls the orchestration issues. This
validates the device_runner wiring for the dep_gen sub-feature on a2a3sim
(host collector, AICPU writer, kernel_args.dep_gen_data_base hand-off).

Pytest entry: rely on the standard ``--enable-dep-gen`` CLI option from
``conftest.py``. Standalone entry: also requires ``--enable-dep-gen`` so the
trace path exists when post-run verification runs.

Compute correctness is delegated to the upstream ``vector_example`` test —
this case re-uses the same orchestration to keep coverage focused on the
capture pipeline.
"""

import struct
import sys

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test
from simpler_setup.scene_test import _outputs_dir, _sanitize_for_filename

KERNELS_BASE = "../../../../../examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels"

# Matches struct DepGenRecord in src/a2a3/platform/include/common/dep_gen.h:
#   8 (task_id) + 4 (flags) + 2 (tensor_count) + 2 (explicit_dep_count)
#   + 16*8 (explicit_deps) + 16 (arg_types) + 16 (_pad0) + 16*128 (tensors)
#   = 2240 bytes, aligned(64).
_DEP_GEN_RECORD_SIZE = 2240
# example_orchestration.cpp issues 5 submit_task calls (t0..t4).
_EXPECTED_SUBMIT_COUNT = 5


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestDepGenCapture(SceneTestCase):
    """Vector example, run with dep_gen enabled, then verify submit_trace.bin."""

    CALLABLE = {
        "orchestration": {
            "source": f"{KERNELS_BASE}/orchestration/example_orchestration.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": f"{KERNELS_BASE}/aiv/kernel_add.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "source": f"{KERNELS_BASE}/aiv/kernel_add_scalar.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 2,
                "source": f"{KERNELS_BASE}/aiv/kernel_mul.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "default",
            "platforms": ["a2a3sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 3},
            "params": {},
        },
    ]

    def generate_args(self, params):
        SIZE = 128 * 128
        return TaskArgsBuilder(
            Tensor("a", torch.full((SIZE,), 2.0, dtype=torch.float32)),
            Tensor("b", torch.full((SIZE,), 3.0, dtype=torch.float32)),
            Tensor("f", torch.zeros(SIZE, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        args.f[:] = (args.a + args.b + 1) * (args.a + args.b + 2) + (args.a + args.b)

    def _post_validate(self, case):
        """Hook invoked after the case ran when --enable-dep-gen is in effect.

        Locates the per-case output_prefix directory and asserts the
        ``submit_trace.bin`` file has size == 5 * sizeof(DepGenRecord).
        """
        case_name = case["name"]
        safe_label = _sanitize_for_filename(f"TestDepGenCapture_{case_name}")
        outputs = _outputs_dir()
        matches = sorted(outputs.glob(f"{safe_label}_*"), key=lambda p: p.stat().st_mtime)
        assert matches, f"no output prefix under {outputs} matching {safe_label}_*"
        trace = matches[-1] / "submit_trace.bin"
        assert trace.exists(), f"submit_trace.bin not produced under {matches[-1]}"

        size = trace.stat().st_size
        expected = _EXPECTED_SUBMIT_COUNT * _DEP_GEN_RECORD_SIZE
        assert size == expected, (
            f"submit_trace.bin size {size} != expected {expected} "
            f"({_EXPECTED_SUBMIT_COUNT} records * {_DEP_GEN_RECORD_SIZE} B per record)"
        )

        # Spot-check first record's tensor_count to confirm bytes are real:
        # t0 (kernel_add) has 3 args (a, b, c=output) -> tensor_count == 3.
        with trace.open("rb") as f:
            header = f.read(16)
        _task_id, _flags, tensor_count, _expl = struct.unpack("=QIHH", header)
        assert tensor_count == 3, f"first record tensor_count={tensor_count}, expected 3"


def _post_run_verify():
    """Iterate all cases and run _post_validate. Used by standalone main."""
    inst = TestDepGenCapture()
    for case in TestDepGenCapture.CASES:
        inst._post_validate(case)


if __name__ == "__main__":
    # The standalone entry exits with 0/1 from inside SceneTestCase.run_module,
    # so the verification has to happen *before* the exit, or we need to catch
    # the SystemExit and only do verification on success. Easier: catch.
    enable_dep_gen = "--enable-dep-gen" in sys.argv
    try:
        SceneTestCase.run_module(__name__)
    except SystemExit as e:
        if e.code not in (0, None):
            raise
        if enable_dep_gen:
            _post_run_verify()
            print("[dep_gen_capture] post-run verification PASSED")
        sys.exit(0)
