#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L3 host-buffer registration (issue #1027).

A host tensor created *after* the chip children are forked (lazily on the
first ``Worker.run()``) is not visible to those children: the orch fn runs in
the parent and ``orch.copy_to`` carries a raw parent VA that is unmapped (or
stale) in the child. ``Worker.register_host_buffer`` maps a named shm into
each child post-fork so a later run can copy through it.

Covers the mechanism end-to-end (B — register a post-fork buffer, run, get the
correct result). The unregistered-tensor error path (C) is a pure host-side
classifier with no kernel/device dependency, unit-tested in
``tests/ut/py/test_worker/test_host_buffer_registration.py``.

a2a3sim: ``register_host_buffer`` is pure host-side (POSIX shm + a control
broadcast to the forked chip children) with no platform branching, so the sim
backend exercises the full mechanism without needing a device. The
vector_example orchestration kernels exist only for a2a3.
"""

import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, TaskArgs, TensorArgType

from simpler_setup import SceneTestCase, make_tensor_arg, scene_test

KERNELS_BASE = "../../../../examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels"

SIZE = 128 * 128


def _golden(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    s = a + b
    return (s + 1) * (s + 2) + s


def _one_task_orch(chip_handle, a, b, out):
    def orch_fn(orch, _args, cfg):
        ta = TaskArgs()
        ta.add_tensor(make_tensor_arg(a), TensorArgType.INPUT)
        ta.add_tensor(make_tensor_arg(b), TensorArgType.INPUT)
        ta.add_tensor(make_tensor_arg(out), TensorArgType.OUTPUT_EXISTING)
        orch.submit_next_level(chip_handle, ta, cfg, worker=0)

    return orch_fn


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestPostForkHostBufferRegistration(SceneTestCase):
    """Post-fork host-buffer registration on a single L3 worker (issue #1027)."""

    CALLABLE = {
        "callables": [
            {
                "name": "vector",
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
            },
        ],
    }

    CASES = [
        {"name": "post_fork_registration", "platforms": ["a2a3sim"]},
    ]

    def _force_fork(self, worker, chip_handle):
        """Run once with pre-fork shared tensors so the chip children get forked."""
        a = torch.full((SIZE,), 2.0, dtype=torch.float32).share_memory_()
        b = torch.full((SIZE,), 3.0, dtype=torch.float32).share_memory_()
        out = torch.zeros(SIZE, dtype=torch.float32).share_memory_()
        worker.run(_one_task_orch(chip_handle, a, b, out), args=None, config=CallConfig())
        assert torch.allclose(out, _golden(a, b), rtol=self.RTOL, atol=self.ATOL)

    def test_run(self, st_worker):
        """Mechanism B: a host tensor created AFTER the fork and registered with
        ``register_host_buffer`` is visible to the chip child and round-trips to the
        correct result.

        Overrides the default ``generate_args``/``compute_golden`` flow: issue #1027
        is about *timing* — the buffer must be created AFTER the chip children are
        forked, which the standard pre-fork arg-generation path cannot express. The
        ``st_worker`` L3 fixture owns the worker lifecycle (build/init/close).
        """
        worker = st_worker
        chip_handle = type(self)._st_chip_handles["vector"]

        self._force_fork(worker, chip_handle)

        # Created AFTER the fork — invisible to the child until register maps it in.
        a = torch.full((SIZE,), 5.0, dtype=torch.float32).share_memory_()
        b = torch.full((SIZE,), 7.0, dtype=torch.float32).share_memory_()
        out = torch.zeros(SIZE, dtype=torch.float32).share_memory_()
        ha = worker.register_host_buffer(a)
        hb = worker.register_host_buffer(b)
        hout = worker.register_host_buffer(out)
        try:
            worker.run(_one_task_orch(chip_handle, a, b, out), args=None, config=CallConfig())
            assert torch.allclose(out, _golden(a, b), rtol=self.RTOL, atol=self.ATOL)
        finally:
            worker.unregister_host_buffer(ha)
            worker.unregister_host_buffer(hb)
            worker.unregister_host_buffer(hout)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
