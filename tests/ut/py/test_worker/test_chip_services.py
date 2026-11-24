# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Allocation lifetime and owner-process startup for optional chip services."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from multiprocessing import Pipe
from types import SimpleNamespace

import pytest
from simpler.buffer import mint_owner_instance_id, wrap_device_malloc
from simpler.task_interface import DataType
from simpler.worker import Worker, _ChipServiceContext, _Lifecycle

from ._harness import fake_chip_l3, requires_sim_binaries
from .test_child_addr_guard import _l3_ready, _record_malloc


def test_retained_allocation_refuses_free_and_close_without_changing_lifecycle():
    worker, native = _l3_ready()
    handle = _record_malloc(worker, 0, 0x1000)
    first, descriptor = worker.retain_service_buffer(handle)
    second, _ = worker.retain_service_buffer(handle)
    assert descriptor == handle.to_descriptor()
    with pytest.raises(RuntimeError, match="service"):
        worker.free(handle)
    with pytest.raises(RuntimeError, match="service"):
        worker.close()
    assert worker._lifecycle is _Lifecycle.READY
    assert worker._child_alloc.get(handle.identity) is not None
    native.free.assert_not_called()
    worker.release_service_buffer(first)
    with pytest.raises(RuntimeError, match="service"):
        worker.free(handle)
    worker.release_service_buffer(second)
    worker.free(handle)
    native.free.assert_called_once_with(0, 0x1000)
    with pytest.raises(ValueError, match="live"):
        worker.retain_service_buffer(handle)
    with pytest.raises(ValueError, match="token"):
        worker.release_service_buffer(second)


@pytest.mark.parametrize("field,value", [("nbytes", 128), ("body", (0x2000).to_bytes(8, "little"))])
def test_retain_rejects_modified_descriptor(field, value):
    worker, _ = _l3_ready()
    handle = _record_malloc(worker, 0, 0x1000)
    with pytest.raises(ValueError, match="descriptor"):
        worker.retain_service_buffer(replace(handle, **{field: value}))
    assert not worker._service_buffers


def test_retain_rejects_exportable_allocation_before_creating_token(monkeypatch):
    worker, _ = _l3_ready()
    monkeypatch.setattr(worker, "_vmm_request", lambda *_args: (0x1000, 1, 4096, 0, 0))
    handle = worker.alloc_child_tensor(0, (64,), DataType.UINT8, exportable=True)
    with pytest.raises(ValueError, match="exportable"):
        worker.retain_service_buffer(handle)
    assert not worker._service_buffers


def test_retain_and_free_have_one_allocation_lifetime_boundary():
    for _ in range(16):
        worker, native = _l3_ready()
        handle = _record_malloc(worker, 0, 0x1000)
        gate = threading.Barrier(2)

        def race(operation):
            gate.wait(timeout=5)
            try:
                return operation(handle)
            except (RuntimeError, ValueError) as exc:
                return exc

        with ThreadPoolExecutor(max_workers=2) as pool:
            retain = pool.submit(race, worker.retain_service_buffer)
            free = pool.submit(race, worker.free)
            retained, freed = retain.result(5), free.result(5)
        if isinstance(retained, tuple):
            assert isinstance(freed, RuntimeError)
            native.free.assert_not_called()
            worker.release_service_buffer(retained[0])
            worker.free(handle)
        else:
            assert isinstance(retained, ValueError) and freed is None
        assert not worker._service_buffers
        native.free.assert_called_once_with(0, 0x1000)


def test_owner_context_validates_actual_malloc_and_preserves_pins():
    owner = mint_owner_instance_id()
    context = _ChipServiceContext(owner, 0)
    handle = wrap_device_malloc(0x1000, 64, owner, 1)
    context.record(0x1000, 64, handle.identity)
    assert context.pin("registration", handle.to_descriptor()) == (0x1000, 64)
    with pytest.raises(RuntimeError, match="service"):
        context.free(0x1000, lambda ptr: pytest.fail("freed pinned allocation"))
    with pytest.raises(ValueError, match="token"):
        context.pin("registration", handle.to_descriptor())
    for invalid in (
        replace(handle, nbytes=128),
        replace(handle, body=(0x2000).to_bytes(8, "little")),
        wrap_device_malloc(0x1000, 64, mint_owner_instance_id(), 2),
        wrap_device_malloc(0x1000, 64, owner, 2),
        wrap_device_malloc(0x1000, 64, owner, 1, generation=2),
    ):
        with pytest.raises(ValueError):
            context.pin("invalid", invalid.to_descriptor())
    context.unpin("registration")
    freed = []
    context.free(0x1000, freed.append)
    assert freed == [0x1000]
    with pytest.raises(ValueError):
        context.pin("stale", handle.to_descriptor())
    replacement = wrap_device_malloc(0x1000, 64, owner, 3)
    context.record(0x1000, 64, replacement.identity)
    with pytest.raises(ValueError):
        context.pin("stale", handle.to_descriptor())
    assert context.pin("replacement", replacement.to_descriptor()) == (0x1000, 64)


class _Service:
    def __init__(self, context, channel):
        self.context, self.channel = context, channel
        self.thread = threading.Thread(target=self.serve)
        self.thread.start()

    def serve(self):
        self.channel.send((os.getpid(), self.context.worker_id))
        while True:
            op, args = self.channel.recv()
            if op == "stop":
                self.channel.send(None)
                return
            try:
                if op == "pin":
                    token, (address, size, owner, buffer_id) = args
                    args = token, wrap_device_malloc(address, size, owner, buffer_id).to_descriptor()
                self.channel.send((getattr(self.context, op)(*args), None))
            except Exception as exc:
                self.channel.send((None, str(exc)))

    def close(self):
        self.thread.join(2)
        assert not self.thread.is_alive()
        self.channel.close()


@requires_sim_binaries
def test_service_runs_in_real_chip_child_and_releases_before_finalize(monkeypatch):
    parent, child = Pipe()
    try:
        with fake_chip_l3(monkeypatch, chip_service_factories={0: lambda ctx: _Service(ctx, child)}) as worker:
            assert parent.poll(5)
            pid, rank = parent.recv()
            assert pid != os.getpid() and rank == 0
            handle = worker.alloc_child_tensor(0, (64,), DataType.UINT8)
            token, descriptor = worker.retain_service_buffer(handle)
            try:
                wire = (
                    handle.base,
                    handle.nbytes,
                    bytes(descriptor.identity.owner_instance_id),
                    descriptor.identity.buffer_id,
                )
                parent.send(("pin", (token, wire)))
                assert parent.recv() == ((handle.base, 64), None)
                with pytest.raises(RuntimeError, match="service"):
                    worker.close()
            finally:
                parent.send(("unpin", (token,)))
                assert parent.recv() == (None, None)
                worker.release_service_buffer(token)
                worker.free(handle)
                parent.send(("stop", ()))
                assert parent.recv() is None
    finally:
        parent.close()
        child.close()


@requires_sim_binaries
def test_service_factory_failure_is_startup_failure(monkeypatch):
    def fail(_context):
        raise RuntimeError("service startup failed")

    with fake_chip_l3(monkeypatch, init=False, chip_service_factories={0: fail}) as worker:
        with pytest.raises(RuntimeError, match="service startup failed"):
            worker.init()


@requires_sim_binaries
def test_service_teardown_failure_reports_the_service_error(monkeypatch):
    def fail_close():
        raise RuntimeError("unregister failed")

    with (
        pytest.raises(RuntimeError, match="chip service dev=0 teardown: RuntimeError: unregister failed"),
        fake_chip_l3(monkeypatch, chip_service_factories={0: lambda _ctx: SimpleNamespace(close=fail_close)}),
    ):
        pass


def test_service_configuration_is_startup_only_and_chip_local():
    with pytest.raises(ValueError, match="chip_service_factories"):
        Worker(level=4, chip_service_factories={0: lambda ctx: None})
    with pytest.raises(ValueError, match="chip_service_factories"):
        Worker(level=3, device_ids=[0], chip_service_factories={1: lambda ctx: None})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
