#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tensormap-and-ringbuffer vector example: f = (a+b+1)*(a+b+2) + (a+b)."""

import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_root = _here
while _root != _root.parent:
    if (_root / "pyproject.toml").exists():
        break
    _root = _root.parent
for _d in [str(_root / "tests" / "st"), str(_root / "python"), str(_root / "examples" / "scripts")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

import torch  # noqa: E402
from setup import SceneTestCase, scene_test  # noqa: E402
from setup.environment import ensure_python_path  # noqa: E402

ensure_python_path()
from task_interface import ArgDirection as D  # noqa: E402


@scene_test(level=2, platforms=["a2a3sim", "a2a3"], runtime="tensormap_and_ringbuffer")
class TestVectorExample(SceneTestCase):
    """f = (a+b+1)*(a+b+2) + (a+b), where a=2.0, b=3.0 -> f=47.0."""

    ORCHESTRATION = {
        "source": "kernels/orchestration/example_orchestration.cpp",
        "function_name": "aicpu_orchestration_entry",
        "signature": [D.IN, D.IN, D.OUT],
    }
    KERNELS = [
        {"func_id": 0, "source": "kernels/aiv/kernel_add.cpp", "core_type": "aiv", "signature": [D.IN, D.IN, D.OUT]},
        {"func_id": 1, "source": "kernels/aiv/kernel_add_scalar.cpp", "core_type": "aiv", "signature": [D.IN, D.OUT]},
        {"func_id": 2, "source": "kernels/aiv/kernel_mul.cpp", "core_type": "aiv", "signature": [D.IN, D.IN, D.OUT]},
    ]
    RUNTIME_CONFIG = {"aicpu_thread_num": 4, "block_dim": 3}
    __outputs__ = ["f"]

    def generate_inputs(self, params):
        SIZE = 128 * 128
        return [
            ("a", torch.full((SIZE,), 2.0, dtype=torch.float32)),
            ("b", torch.full((SIZE,), 3.0, dtype=torch.float32)),
            ("f", torch.zeros(SIZE, dtype=torch.float32)),
        ]

    def compute_golden(self, tensors, params):
        a = torch.as_tensor(tensors["a"])
        b = torch.as_tensor(tensors["b"])
        tensors["f"][:] = (a + b + 1) * (a + b + 2) + (a + b)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
