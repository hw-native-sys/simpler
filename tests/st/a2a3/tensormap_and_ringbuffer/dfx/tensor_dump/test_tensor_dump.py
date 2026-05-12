#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""tensor_dump profiling smoke — capture pipeline produces a usable
``tensor_dump/`` directory.

Re-uses ``vector_example`` (5 submit_task calls). With ``--dump-tensor`` the
AICPU writer captures every task's tensor I/O into a manifest + raw-byte
payload pair under ``<output_prefix>/tensor_dump/``. Smoke asserts:
manifest exists + parses, the ``bin_file`` field it names exists, and at
least one entry was captured.
"""

import json

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test
from simpler_setup.scene_test import _outputs_dir, _sanitize_for_filename

KERNELS_BASE = "../../../../../../examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels"


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestTensorDump(SceneTestCase):
    """Vector example with --dump-tensor, then assert tensor_dump/."""

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
            "platforms": ["a2a3sim", "a2a3"],
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

    def test_run(self, st_platform, st_worker, request):
        super().test_run(st_platform, st_worker, request)
        if not request.config.getoption("--dump-tensor", default=False):
            return
        for case in self.CASES:
            if st_platform in case["platforms"]:
                self._validate_dump_artifact(case)

    def _validate_dump_artifact(self, case):
        safe_label = _sanitize_for_filename(f"TestTensorDump_{case['name']}")
        matches = sorted(_outputs_dir().glob(f"{safe_label}_*"), key=lambda p: p.stat().st_mtime)
        if not matches:
            return
        dump_dir = matches[-1] / "tensor_dump"
        assert dump_dir.is_dir(), f"tensor_dump/ missing under {matches[-1]} — dump capture failed?"
        manifest = dump_dir / "tensor_dump.json"
        assert manifest.exists(), f"tensor_dump.json missing under {dump_dir} — collector finalize failed?"
        with manifest.open() as f:
            data = json.load(f)
        bin_name = data.get("bin_file")
        assert bin_name, f"manifest missing bin_file field: {data}"
        bin_path = dump_dir / bin_name
        assert bin_path.exists(), f"manifest names bin_file={bin_name!r} but {bin_path} not found"
        # Tensors list is keyed by run / task / arg — exact shape varies with
        # the collector's chosen schema across runtimes, but we know
        # vector_example reads/writes ≥1 tensor and the manifest can't be empty
        # if anything was captured. Robust to schema add/remove of new fields.
        assert bin_path.stat().st_size > 0, "tensor_dump.bin is empty"


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
