# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public kernel API lifecycle scenes with device numerical oracles."""

import ctypes

from simpler.task_interface import ChipStorageTaskArgs, ChipTensor, DataType

from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.kernel_capture_values import (
    _COUNT,
    _check,
    _check_resident,
    _TensorIO,
)
from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.ordinary_launch import build_ordinary
from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.test_kernel_mode_capture import (
    RUNTIME,
    _binaries,
    _close,
    _config,
    _fail,
    _guarded,
    _initialize,
    _prepare_callable,
)

POISON = -999.0


class Session:
    def __init__(self, device, scenario, build):
        chips, self.lib, self.streams, self.caller, self.ctx, self.observer = _initialize(device, scenario, build)
        self.chip = chips[0]
        self.device = device
        self.allocations = []
        self.io = _TensorIO(self.lib, self.allocations)
        self.graphs = []
        self.launches = 0
        self.args = ChipStorageTaskArgs()
        self.prepare()

    def prepare(self):
        self.cid = _prepare_callable(self.observer, self.lib, self.ctx, self.chip, 0)
        self.committed = self.lib.committed_device_memory_ctx(self.ctx)

    def launch(self, source, output, scalar, count=_COUNT):
        self.args.clear()
        for address in (source, output):
            self.args.add_tensor(ChipTensor.make(address.value, (count,), DataType.FLOAT32, child_memory=True))
        self.args.add_scalar(ctypes.c_float(scalar))
        self.observer.capture_observer_invocation_scope(1)
        try:
            _guarded(
                self.observer, self.lib.simpler_kernel_mode_launch, self.ctx, self.cid, self.args.__ptr__(), self.caller
            )
        finally:
            self.observer.capture_observer_invocation_scope(0)
        self.args.clear()
        self.launches += 1

    def record(self, submit):
        graph = ctypes.c_void_p()
        _check(self.lib.aclmdlRICaptureBegin(self.caller, 0), "capture begin")
        submit()
        _check(self.lib.aclmdlRICaptureEnd(self.caller, ctypes.byref(graph)), "capture end")
        assert graph.value
        self.graphs.append(graph)
        return graph

    def replay(self, graph):
        before = self.observer.capture_observer_cpu_launches()
        _check(self.lib.aclmdlRIExecuteAsync(graph, self.caller), "replay")
        assert self.observer.capture_observer_cpu_launches() == before

    def sync(self):
        _check(self.lib.aclrtSynchronizeStreamWithTimeout(self.caller, 10000), "caller sync")

    def destroy(self, graph):
        _check(self.lib.aclmdlRIDestroy(graph), "destroy graph")
        self.graphs.remove(graph)

    def stable(self):
        _check_resident(self.observer, self.launches)
        assert self.lib.committed_device_memory_ctx(self.ctx) == self.committed

    def close(self):
        self.stable()
        _close(self.lib, self.ctx, self.allocations, self.streams, self.device, self.graphs)


def _buffers(session, count):
    addresses = [session.io.allocate() for _ in range(count)]
    for address in addresses:
        session.io.write(address, [POISON] * _COUNT)
    return addresses


def _verify_prefix(session, output, values, count, scalar):
    session.io.verify(output, [v + scalar for v in values[:count]] + [POISON] * (_COUNT - count))


def _geometry(session):
    x, y, z, eager_out = _buffers(session, 4)
    shapes = (128, 257)
    graphs = [
        session.record(lambda n=n, out=out: session.launch(x, out, 3.0, n))
        for n, out in zip(shapes, (y, z), strict=True)
    ]
    for iteration in range(100):
        values = [float(i % 127 + iteration) for i in range(_COUNT)]
        session.io.write(x, values)
        for output in (y, z, eager_out):
            session.io.write(output, [POISON] * _COUNT)
        session.replay(graphs[0])
        session.launch(x, eager_out, -5.0, 65)
        session.replay(graphs[1])
        session.sync()
        for output, count, scalar in ((y, 128, 3.0), (z, 257, 3.0), (eager_out, 65, -5.0)):
            _verify_prefix(session, output, values, count, scalar)
        session.stable()


def _survivor(session):
    x, y, z, w = _buffers(session, 4)
    first = session.record(lambda: session.launch(x, y, 1.0))
    survivor = session.record(lambda: session.launch(x, z, 3.0))
    replacement = None
    for iteration in range(100):
        if iteration == 25:
            session.destroy(first)
        if iteration == 50:
            replacement = session.record(lambda: session.launch(x, w, -7.0))
        values = [float(i % 127 + iteration) for i in range(_COUNT)]
        session.io.write(x, values)
        for output in (y, z, w):
            session.io.write(output, [POISON] * _COUNT)
        if iteration < 25:
            session.replay(first)
        session.replay(survivor)
        if replacement is not None:
            session.replay(replacement)
        session.sync()
        session.io.verify(y, [v + 1.0 for v in values] if iteration < 25 else [POISON] * _COUNT)
        session.io.verify(z, [v + 3.0 for v in values])
        session.io.verify(w, [v - 7.0 for v in values] if replacement is not None else [POISON] * _COUNT)
        session.stable()


def _mixed(session, scenario, build):
    ordinary, binary = build_ordinary(build)
    handle = ctypes.c_void_p()
    _check(ordinary.ordinary_open(str(binary).encode(), ctypes.byref(handle)), "register ordinary kernel")
    x, before, middle, output = _buffers(session, 4)
    count = 128

    def submit():
        _check(ordinary.ordinary_launch(handle, session.caller, x, before, count, 3.0), "ordinary predecessor")
        session.launch(before, middle, 5.0, count)
        _check(ordinary.ordinary_launch(handle, session.caller, middle, output, count, 11.0), "ordinary successor")

    graph = session.record(submit) if scenario == "mixed_stream_graph" else None
    for iteration in range(100):
        values = [float(i % 127 + iteration) for i in range(_COUNT)]
        session.io.write(x, values)
        for address in (before, middle, output):
            session.io.write(address, [POISON] * _COUNT)
        if graph is None:
            submit()
        else:
            session.replay(graph)
        session.sync()
        for address, expected in (
            (before, [-v + 3.0 for v in values[:count]]),
            (middle, [-v + 8.0 for v in values[:count]]),
            (output, [v + 3.0 for v in values[:count]]),
        ):
            session.io.verify(address, expected + [POISON] * (_COUNT - count))
        session.stable()
    if graph is not None:
        session.destroy(graph)
    _check(ordinary.ordinary_close(handle), "unregister ordinary kernel")


def _ring_wrap(session, scenario):
    x, y = _buffers(session, 2)
    graph = session.record(lambda: session.launch(x, y, 1.0)) if scenario == "ring_wrap_graph" else None
    for iteration in range(100):
        values = [float(i % 127 + iteration) for i in range(_COUNT)]
        session.io.write(x, values)
        session.io.write(y, [POISON] * _COUNT)
        if graph is None:
            session.launch(x, y, 1.0)
        else:
            session.replay(graph)
        session.sync()
        session.io.verify(y, [v + 257.0 for v in values])
        session.stable()


def _callable_growth(session):
    x, y, z = _buffers(session, 3)
    graph = session.record(lambda: session.launch(x, y, 3.0))
    ids = [session.cid]
    before_growth = session.committed
    # The resident cache commits 2 MiB blocks; charge includes 64-byte alignment.
    charged = (session.chip.buffer_size() + 63) // 64 * 64
    registrations = max(2, (2 * 1024 * 1024) // charged + 1)
    for _ in range(registrations - 1):
        ids.append(_prepare_callable(session.observer, session.lib, session.ctx, session.chip, 0))
    assert sorted(ids) == list(range(registrations)), ids
    session.committed = session.lib.committed_device_memory_ctx(session.ctx)
    assert session.committed > before_growth, "registration did not cross a device allocation block"
    session.cid = ids[-1]
    for iteration in range(100):
        values = [float(i % 127 + iteration) for i in range(_COUNT)]
        session.io.write(x, values)
        session.replay(graph)
        session.launch(y, z, -5.0)
        session.sync()
        session.io.verify(y, [v + 3.0 for v in values])
        session.io.verify(z, [v - 2.0 for v in values])
        session.stable()


def _early_dispatch(session, scenario):
    x, y = _buffers(session, 2)
    graph = session.record(lambda: session.launch(x, y, 3.0)) if scenario == "early_dispatch_graph" else None
    for iteration in range(100):
        values = [float(i % 127 + iteration) for i in range(_COUNT)]
        session.io.write(x, values)
        session.io.write(y, [POISON] * _COUNT)
        if graph is None:
            session.launch(x, y, 3.0)
        else:
            session.replay(graph)
        session.sync()
        session.io.verify(y, [1.0] + [v + 6.0 for v in values[1:]])
        session.stable()


def _init_context(session, ctx, generation):
    config = _config()
    cpu, core, dispatcher = _binaries("a2a3", RUNTIME)
    return session.lib.simpler_kernel_mode_init(
        ctx,
        session.device,
        cpu,
        len(cpu),
        core,
        len(core),
        dispatcher,
        len(dispatcher),
        ctypes.byref(config),
        generation,
    )


def _context_lifecycle(session):
    x, y = _buffers(session, 2)
    graph = session.record(lambda: session.launch(x, y, 3.0))
    rejected = session.lib.create_device_context()
    assert rejected
    assert _init_context(session, rejected, 72) == -1003
    _check(session.lib.finalize_device(rejected), "close rejected context")
    session.lib.destroy_device_context(rejected)
    for iteration in range(100):
        if iteration == 50:
            session.destroy(graph)
            session.stable()
            _check(session.lib.finalize_device(session.ctx), "close first context")
            assert session.lib.committed_device_memory_ctx(session.ctx) == 0
            session.lib.destroy_device_context(session.ctx)
            session.ctx = session.lib.create_device_context()
            assert session.ctx
            _check(_init_context(session, session.ctx, 73), "initialize replacement context")
            session.observer.capture_observer_begin()
            session.launches = 0
            session.prepare()
            graph = session.record(lambda: session.launch(x, y, 3.0))
        values = [float(i % 127 + iteration) for i in range(_COUNT)]
        session.io.write(x, values)
        session.replay(graph)
        session.launch(y, y, 2.0)
        session.sync()
        session.io.verify(y, [v + 5.0 for v in values])
        session.stable()


def run_lifecycle_case(device, scenario, build):
    try:
        session = Session(device, scenario, build)
        if scenario == "geometry_snapshot":
            _geometry(session)
        elif scenario == "graph_survivor":
            _survivor(session)
        elif scenario == "context_lifecycle":
            _context_lifecycle(session)
        elif scenario in ("ring_wrap_eager", "ring_wrap_graph"):
            _ring_wrap(session, scenario)
        elif scenario == "callable_growth":
            _callable_growth(session)
        elif scenario in ("early_dispatch_eager", "early_dispatch_graph"):
            _early_dispatch(session, scenario)
        else:
            _mixed(session, scenario, build)
        session.close()
    except BaseException:
        _fail()
    print(f"PASS {scenario} rounds=100 forbidden_sync=0", flush=True)
