# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public direct-L2 handles retain distinct Program outputs across slot reuse."""

import copy
import json
import time
from pathlib import Path

import pytest
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _build_l2_ref_args, compile_chip_callable_spec

_ROOT = Path(__file__).resolve().parents[4]
_EXPECTED = (3, 16, 25, 40, 47, 64)
_WAIT_ORDER = (5, 0, 4, 1, 3, 2)
_SIZE = 128 * 128


def _callable_spec(name):
    return {
        "orchestration": {
            "source": str(_ROOT / "examples/workers/l2/framework_eager/kernels/lifecycle_orchestration.cpp"),
            "function_name": "framework_owner_" + name,
            "signature": [D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": str(_ROOT / "examples/workers/l2/vector_add/kernels/aiv/vector_add_kernel.cpp"),
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            }
        ],
    }


class _RunCompletion(SceneTestCase):
    CALLABLE = _callable_spec("x")
    CASES = [{"name": "completed_unobserved_outputs", "platforms": ["a2a3sim", "a5sim", "a2a3", "a5"]}]

    def generate_args(self, params):
        index = params.get("index", 0)
        return TaskArgsBuilder(
            TensorArg("a", torch.full((_SIZE,), float(10 * index + 2), dtype=torch.float32)),
            TensorArg("b", torch.full((_SIZE,), float(index + 1), dtype=torch.float32)),
            TensorArg("out", torch.full((_SIZE,), -999.0, dtype=torch.float32)),
        )

    def test_run(self, st_platform, st_worker, request):
        if not self._matching_cases(st_platform, request):
            pytest.skip("No matching completed-unobserved output case")
        x = self.build_callable(st_platform)
        y = compile_chip_callable_spec(
            _callable_spec("y"), st_platform, self._st_runtime, (type(self).__name__, st_platform, "y")
        )
        handles = {"x": st_worker.register(x), "y": st_worker.register(y)}
        frames = []
        for index in range(len(_EXPECTED)):
            tensors = self.generate_args({"index": index})
            args, _ = _build_l2_ref_args(tensors, self.CALLABLE["orchestration"]["signature"], st_worker)
            run = st_worker.submit(handles["x" if index % 2 == 0 else "y"], args)
            frames.append((run, tensors, args))
            deadline = time.monotonic() + 30
            while not run.done:
                assert time.monotonic() < deadline, f"Run {index} did not complete before the deadline"
            torch.testing.assert_close(tensors.out, torch.full_like(tensors.out, -999.0))
            assert all(previous.done for previous, _, _ in frames)
        assert len(frames) == 6
        for index in _WAIT_ORDER:
            run, tensors, _ = frames[index]
            run.wait()
            torch.testing.assert_close(tensors.out, torch.full_like(tensors.out, _EXPECTED[index]))
            run.result()
            assert run.done
        for index, (_, tensors, _) in enumerate(frames):
            torch.testing.assert_close(tensors.out, torch.full_like(tensors.out, _EXPECTED[index]))
        for handle in handles.values():
            st_worker.unregister(handle)
        print(
            "PROGRAM_COMPLETION_PASS "
            + json.dumps(
                {
                    "runtime": self._st_runtime,
                    "platform": st_platform,
                    "completed_before_wait": len(frames),
                    "wait_order": _WAIT_ORDER,
                    "observed": [float(tensors.out[0]) for _, tensors, _ in frames],
                },
                sort_keys=True,
            ),
            flush=True,
        )


@scene_test(level=2, runtime="host_build_graph")
class TestRunCompletionHbg(_RunCompletion):
    CALLABLE = copy.deepcopy(_RunCompletion.CALLABLE)


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestRunCompletionTmr(_RunCompletion):
    CALLABLE = copy.deepcopy(_RunCompletion.CALLABLE)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
