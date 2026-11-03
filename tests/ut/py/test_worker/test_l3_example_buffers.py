# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exercise standalone example input/output plumbing without loading an SDK."""

import ctypes
import importlib.util
from pathlib import Path

import pytest
import torch
from simpler.buffer import ImportContext, ImportRegistry
from simpler.comm_endpoints import DEVICE_AICPU
from simpler.worker import Worker


@pytest.mark.parametrize("example", ["multi_chip_dispatch", "per_task_runtime_env"])
def test_vector_examples_use_attachable_buffers_and_observe_outputs(monkeypatch, example):
    path = Path(__file__).resolve().parents[4] / "examples/workers/l3" / example / "main.py"
    spec = importlib.util.spec_from_file_location(example, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    registries = []
    workers = []

    class LocalWorker(Worker):
        def __init__(self, **_kwargs):
            super().__init__(level=3, num_sub_workers=1)
            workers.append(self)

        def register(self, _callable):
            return 1

        def run(self, graph, args=None, config=None):
            registry = ImportRegistry(ImportContext(deployment=DEVICE_AICPU, fork_inherited=False))
            registries.append(registry)

            class Dispatch:
                def submit_next_level(self, _handle, task_args, _cfg, *, worker):
                    tensors = [task_args.tensor(i) for i in range(3)]
                    mapped = [registry.materialize(t.buffer) for t in tensors]
                    count = mapped[0].nbytes // 4
                    a, b, out = [(ctypes.c_float * count).from_address(h.base) for h in mapped]
                    for i in range(count):
                        out[i] = a[i] + b[i]

                def submit_sub(self, _handle, task_args):
                    for i in range(task_args.tensor_count()):
                        registry.materialize(task_args.tensor(i).buffer)

            try:
                graph(Dispatch(), args, config)
            finally:
                registry.close()

    monkeypatch.setattr(module, "Worker", LocalWorker)
    monkeypatch.setattr(module, "build_chip_callable", lambda _platform: object())
    if example == "multi_chip_dispatch":
        assert module.run("a5", [0, 1]) == 0
    else:
        assert module.run("a5", 0) == 0
    assert workers and not workers[0]._buffers
    assert registries and all(not r._by_identity for r in registries)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int32])
def test_shared_tensor_preserves_shape_dtype_and_independent_source(dtype):
    from simpler_setup.torch_interop import create_shared_tensor  # noqa: PLC0415

    worker = Worker(level=3, num_sub_workers=1)
    worker.init()
    registry = ImportRegistry(ImportContext(deployment=DEVICE_AICPU, fork_inherited=False))
    source = torch.arange(6, dtype=dtype).reshape(2, 3)
    view = None
    try:
        ref, view = create_shared_tensor(worker, source)
        assert view.shape == source.shape and view.dtype == dtype
        assert torch.equal(source, view)
        mapped = registry.materialize(ref.buffer)
        ctypes.memset(mapped.base, 0, mapped.nbytes)
        assert torch.count_nonzero(view) == 0
        assert torch.equal(source, torch.arange(6, dtype=dtype).reshape(2, 3))
    finally:
        registry.close()
        view = None
        worker.close()


@pytest.mark.parametrize("source", [torch.zeros(0), torch.zeros(2, 3).T, torch.empty(2, device="meta")])
def test_shared_tensor_rejects_inputs_before_allocating(source):
    from simpler_setup.torch_interop import create_shared_tensor  # noqa: PLC0415

    worker = Worker(level=2)
    with pytest.raises(ValueError, match="contiguous, nonempty CPU"):
        create_shared_tensor(worker, source)
    assert not worker._buffers
