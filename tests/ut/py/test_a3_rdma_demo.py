# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The direct SDK demo obeys the real Worker host-argument contract without a device."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from simpler.worker import Worker


def test_disabled_demo_skip_matches_its_declared_coverage_exception(monkeypatch):
    path = (
        Path(__file__).resolve().parents[3]
        / "examples/a2a3/tensormap_and_ringbuffer/rdma_async_completion_demo/test_rdma_async_completion_demo.py"
    )
    spec = importlib.util.spec_from_file_location("rdma_demo_skip_contract", path)
    assert spec is not None and spec.loader is not None
    demo = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, demo)
    spec.loader.exec_module(demo)
    monkeypatch.delenv("SIMPLER_ENABLE_PTO_RDMA_WORKSPACE", raising=False)
    declaration = next(marker for marker in demo.pytestmark if marker.name == "coverage_skip")
    with pytest.raises(pytest.skip.Exception) as skipped:
        demo.test_rdma_async_completion_demo("a2a3", [0, 1])
    assert skipped.value.msg == declaration.kwargs["reason"]


def test_demo_captures_host_requests_without_using_task_args_for_python_objects(monkeypatch):
    path = (
        Path(__file__).resolve().parents[3]
        / "examples/a2a3/tensormap_and_ringbuffer/rdma_async_completion_demo/test_rdma_async_completion_demo.py"
    )
    name = "rdma_demo_argument_contract"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    demo = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, demo)
    spec.loader.exec_module(demo)
    calls = []
    instances = []
    domain_identity = object()

    class HostArgumentWorker:
        level = 3
        _config = {"platform": "a2a3sim"}
        _host_group_mappings = None

        def __init__(self, **kwargs):
            self.registered = []
            self.initialized = self.closed = False
            instances.append(self)

        def register(self, callable):
            self.registered.append(callable)
            return len(self.registered)

        def init(self):
            self.initialized = True

        def run(self, callable, args=None, config=None):
            Worker._prepare_l3_host_args(cast(Worker, self), args)
            callable(None, args, config)

        def close(self):
            self.closed = True

    def submit(_orch, _args, _config, *, request):
        request.identity = domain_identity
        request.contexts = [101, 102]
        calls.append(("submit", request))

    def readback(_orch, _args, _config, *, request):
        request.received = [request.pattern(1), request.pattern(0)]
        calls.append(("readback", request))

    monkeypatch.setattr(demo, "Worker", HostArgumentWorker)
    monkeypatch.setattr(demo, "_build_rdma_callable", lambda operation: operation)
    monkeypatch.setattr(demo, "rdma_host_orchestration", submit)
    monkeypatch.setattr(demo, "_readback", readback)
    demo.run([14, 15], roce_ips=("192.0.2.1", "192.0.2.2"))
    assert len(instances) == 1
    assert instances[0].registered == ["put", "get"]
    assert instances[0].initialized and instances[0].closed
    assert [kind for kind, _request in calls] == ["submit", "readback"] * 4
    for iteration in range(4):
        request = calls[2 * iteration][1]
        assert calls[2 * iteration + 1][1] is request
        assert request.iteration == iteration
        assert request.operation == ("put" if iteration % 2 == 0 else "get")
    assert len({id(request) for kind, request in calls if kind == "submit"}) == 4


def test_demo_declares_storage_covering_every_staged_window_byte(monkeypatch):
    path = (
        Path(__file__).resolve().parents[3]
        / "examples/a2a3/tensormap_and_ringbuffer/rdma_async_completion_demo/test_rdma_async_completion_demo.py"
    )
    name = "rdma_demo_window_contract"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    demo = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, demo)
    spec.loader.exec_module(demo)
    copies = []
    submissions = []

    class WindowOrchestrator:
        def get_or_allocate_domain(self, **kwargs):
            self.options = kwargs
            return SimpleNamespace(ref=object(), device_context=lambda rank, **_kwargs: 101 + rank)

        def copy_to_domain(self, domain, rank, data, *, offset=0):
            declared = sum(buffer.nbytes for buffer in self.options.get("buffers", ()))
            assert offset + len(data) <= declared, "staging exceeds the declared domain buffer slices"
            copies.append((rank, offset, data))

        def submit_next_level(self, handle, args, config, *, worker):
            submissions.append((handle, worker))

    orch = WindowOrchestrator()
    request = demo.Transfer(7, "put", 0, 64, 0, 64, 512)
    demo.rdma_host_orchestration(orch, None, None, request=request)
    assert [(rank, offset, len(data)) for rank, offset, data in copies] == [
        (0, 0, demo.WINDOW_BYTES),
        (0, 64, 64),
        (1, 0, demo.WINDOW_BYTES),
        (1, 64, 64),
    ]
    assert submissions == [(7, 0), (7, 1)]
