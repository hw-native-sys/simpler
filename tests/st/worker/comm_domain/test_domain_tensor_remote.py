# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One Buffer view and explicit-context contract across local, TCP and MPI L3s."""

import ctypes
import socket
import struct
import subprocess
import sys
import time

import pytest
from simpler.task_interface import (
    CommBufferSpec,
    CommDomainRef,
    DataType,
    RemoteBufferHandle,
    RemoteTensorRef,
    TaskArgs,
    TensorArgType,
)
from simpler.worker import RemoteCallable, RemoteWorkerSpec, Worker

from examples.workers.l3.multi_chip_dispatch.main import N_COLS, N_ROWS, build_chip_callable
from examples.workers.l4.global_tload_mixed_l3.main import COUNT, _build_tload_callable

_NUMEL = N_ROWS * N_COLS
_NBYTES = _NUMEL * 4
_RUNTIME = "tensormap_and_ringbuffer"
_ACCEPTED = {}


def forward_domain_task(orch, args, cfg, chips=None):
    from simpler.remote_l3_session import get_inner_handle  # noqa: PLC0415

    if args.tensor_count() == 0:
        ref, chip, old_args, target = _ACCEPTED.pop(int(args.scalar(0)))
        with pytest.raises(KeyError):
            orch.get_domain(ref)
        with pytest.raises(ValueError, match="stale|not live"):
            orch.submit_next_level(chip, old_args, cfg, worker=target)
        return
    ref = CommDomainRef(int(args.scalar(4)).to_bytes(8, "little"), int(args.scalar(5)), int(args.scalar(6)))
    rank, context_mode = int(args.scalar(7)), bool(args.scalar(8))
    domain = orch.get_domain(ref)
    member = domain.member(rank)
    assert domain.borrowed and domain.ref == ref and orch.get_domain(ref) is domain
    for other in domain.members:
        assert bool(other.buffers) == (other.node_worker_id == member.node_worker_id)
        if other.node_worker_id != member.node_worker_id:
            with pytest.raises(KeyError):
                domain.local_context(other.domain_rank)
    for operation in (domain.release, domain.__enter__):
        with pytest.raises(RuntimeError, match="borrowed"):
            operation()
    chip = (
        chips[context_mode]
        if chips
        else get_inner_handle(b"".join(int(args.scalar(index)).to_bytes(8, "little") for index in range(4)).hex())
    )
    chip_args = TaskArgs()
    tags = (
        (TensorArgType.INPUT, TensorArgType.OUTPUT_EXISTING)
        if context_mode
        else (TensorArgType.INPUT, TensorArgType.INPUT, TensorArgType.OUTPUT_EXISTING)
    )
    for index, tag in enumerate(tags):
        tensor = args.tensor(index)
        if context_mode or index < 2:
            name = "context_result" if context_mode and index == 1 else "payload"
            assert tensor.buffer.identity == member.buffers[name].identity
            assert tensor.byte_offset == (64 if context_mode or index == 0 else 128)
            assert tensor.shapes == ((COUNT,) if context_mode else (_NUMEL,))
            assert tensor.strides == (1,) and tensor.dtype == DataType.FLOAT32.value
        chip_args.add_tensor(tensor, tag)
    assert domain.buffers[0].count == _NUMEL + 32
    if context_mode:
        context = domain.local_context(rank)
        assert context.domain_rank == rank
        chip_args.add_scalar(context.domain_size)
        chip_args.add_scalar(context.device_ctx)
    else:
        assert orch.copy_from_domain(domain, rank, 4, buffer="payload", offset=64) == struct.pack("<f", rank * 100 + 16)
        with pytest.raises(ValueError, match="wrong worker|stale"):
            orch.submit_next_level(chip, chip_args, cfg, worker=member.local_worker_id ^ 1)
        _ACCEPTED[rank] = ref, chip, chip_args, member.local_worker_id
    orch.submit_next_level(chip, chip_args, cfg, worker=member.local_worker_id)


def exercise_domain_tensors(worker, members, platform, *, local=None, cached=True):
    outputs, saved, cached_args = [], [], []
    callables = (build_chip_callable(platform), _build_tload_callable(platform, _RUNTIME))
    chips = tuple(worker.register(callable_obj) for callable_obj in callables)
    nodes = tuple(dict.fromkeys(node for node, _chip in members))
    remote_nodes = tuple(node for node in nodes if local is None or node != local[0])
    routes = {}
    if local:

        def forward(orch, args, cfg):
            forward_domain_task(orch, args, cfg, local[1])

        routes[local[0]] = worker.register(forward)
    if remote_nodes:
        remote = worker.register(RemoteCallable(__name__ + ":forward_domain_task"), workers=remote_nodes)
        routes.update(dict.fromkeys(remote_nodes, remote))
    request = dict(
        name="tp",
        members=members,
        window_size=_NBYTES + 4096,
        buffers=[
            CommBufferSpec("payload", "float32", _NUMEL + 32, _NBYTES + 128),
            CommBufferSpec("context_result", "float32", COUNT + 16, (COUNT + 16) * 4),
        ],
        retain_after_run=True,
    )

    def route_args(domain, rank, tensors, context_mode=False):
        args = TaskArgs()
        for tensor, tag in tensors:
            args.add_tensor(tensor, tag)
        for offset in range(0, 32, 8):
            args.add_scalar(int.from_bytes(chips[context_mode].digest[offset : offset + 8], "little"))
        for value in (
            int.from_bytes(domain.ref.owner_instance_id, "little"),
            domain.ref.domain_id,
            domain.ref.generation,
            rank,
            int(context_mode),
        ):
            args.add_scalar(value)
        return args

    try:
        worker.init()
        for node, _chip in members:
            outputs.append(
                worker.remote_malloc(worker=node, nbytes=_NBYTES)
                if node in remote_nodes
                else worker.create_buffer(_NBYTES)
            )

        def first(orch, _args, _cfg):
            domain = orch.get_or_allocate_domain(**request)
            assert orch.get_or_allocate_domain(**request) is domain
            saved.append(domain)
            for rank, output in enumerate(outputs):
                values = tuple(float(rank * 100 + index) for index in range(_NUMEL + 32))
                orch.copy_to_domain(domain, rank, struct.pack(f"<{len(values)}f", *values), buffer="payload")
                payload = domain.member(rank).buffers["payload"]
                result = (
                    RemoteTensorRef(output, shape=(_NUMEL,), dtype=DataType.FLOAT32)
                    if isinstance(output, RemoteBufferHandle)
                    else output.tensor([_NUMEL], DataType.FLOAT32)
                )
                tensors = [
                    (payload.tensor([_NUMEL], DataType.FLOAT32, byte_offset=offset), TensorArgType.INPUT)
                    for offset in (64, 128)
                ] + [(result, TensorArgType.OUTPUT_EXISTING)]
                cached_args.append(route_args(domain, rank, tensors))

        worker.run(first)
        domain, allocation_count = saved[0], worker._next_alloc_id
        old_ref = domain.ref
        identities = [args.tensor(0).buffer.identity for args in cached_args]

        def second(orch, _args, cfg):
            if not cached:
                assert orch.get_or_allocate_domain(**request) is domain
            assert worker._next_alloc_id == allocation_count
            for (node, _chip), args in zip(members, cached_args):
                orch.submit_next_level(routes[node], args, cfg, worker=node)
            if not cached:
                domain.release()

        run = worker.submit(second)
        assert run._resources.requires_ordered_cleanup
        if cached:
            assert run._resources.live_global_domains["tp"] is domain
        else:
            assert domain in run._resources.pending_release_global_domains
        domain.release()
        assert domain.released and not domain.freed
        run.wait()
        assert domain.freed and worker.live_domains == {}
        for rank, output in enumerate(outputs):
            host = ctypes.create_string_buffer(_NBYTES)
            if isinstance(output, RemoteBufferHandle):
                worker.remote_copy_from(output, host, _NBYTES)
            raw = host.raw if isinstance(output, RemoteBufferHandle) else output.shm.buf
            assert struct.unpack_from(f"<{_NUMEL}f", raw) == tuple(
                float(rank * 200 + 48 + index * 2) for index in range(_NUMEL)
            )

        def rebuild(orch, _args, cfg):
            replacement = orch.get_or_allocate_domain(**request)
            assert replacement.ref != old_ref
            saved.append(replacement)
            for rank, (node, _chip) in enumerate(members):
                assert replacement.member(rank).buffers["payload"].identity != identities[rank]
                values = tuple(float(rank * 100 + index) for index in range(COUNT + 16))
                orch.copy_to_domain(replacement, rank, struct.pack(f"<{COUNT + 16}f", *values), buffer="payload")
                orch.copy_to_domain(
                    replacement, rank, struct.pack(f"<{COUNT + 16}f", *([-1.0] * (COUNT + 16))), buffer="context_result"
                )
            for rank, (node, _chip) in enumerate(members):
                stale = TaskArgs()
                stale.add_scalar(rank)
                orch.submit_next_level(routes[node], stale, cfg, worker=node)
                tensors = [
                    (replacement.member(rank).buffers[name].tensor([COUNT], DataType.FLOAT32, byte_offset=64), tag)
                    for name, tag in (
                        ("payload", TensorArgType.INPUT),
                        ("context_result", TensorArgType.OUTPUT_EXISTING),
                    )
                ]
                orch.submit_next_level(routes[node], route_args(replacement, rank, tensors, True), cfg, worker=node)

        worker.run(rebuild)

        def finish(orch, _args, _cfg):
            replacement = orch.get_domain("tp")
            assert replacement is saved[1] and not replacement.released
            count = len(members)
            for rank in range(count):
                values = struct.unpack(
                    f"<{COUNT + 16}f",
                    orch.copy_from_domain(replacement, rank, (COUNT + 16) * 4, buffer="context_result"),
                )
                assert values[:16] == (-1.0,) * 16
                assert values[16:] == tuple(
                    float(count * (index + 16) + 100 * count * (count - 1) // 2) for index in range(COUNT)
                )
            replacement.release()

        worker.run(finish)
        assert saved[1].freed and worker.live_domains == {}
    finally:
        try:
            for output in reversed(outputs):
                if isinstance(output, RemoteBufferHandle):
                    worker.remote_free(output)
        finally:
            worker.close()


@pytest.fixture
def tcp_daemon(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with (tmp_path / "daemon.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "simpler.remote_l3_worker", "--host", "127.0.0.1", "--port", str(port)],
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 10
            while True:
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                        break
                except OSError:
                    if time.monotonic() >= deadline or process.poll() is not None:
                        raise RuntimeError("TCP daemon did not start")
                    time.sleep(0.01)
            yield f"127.0.0.1:{port}"
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


@pytest.mark.platforms(["a2a3sim", "a5sim"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(1)
def test_domain_tensor_reaches_remote_chip(tcp_daemon, st_platform, st_device_ids):
    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=30)
    node = worker.add_remote_worker(
        RemoteWorkerSpec(
            endpoint=tcp_daemon, platform=st_platform, device_ids=(int(st_device_ids[0]),), comm_profile="sim"
        )
    )
    exercise_domain_tensors(worker, [(node, 0)], st_platform)


@pytest.mark.platforms(["a2a3sim", "a5sim"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(1)
@pytest.mark.parametrize("cached", [False, True])
def test_domain_tensor_reaches_local_l3(st_platform, st_device_ids, cached):
    worker = Worker(level=4, num_sub_workers=0)
    child = Worker(
        level=3, platform=st_platform, runtime=_RUNTIME, device_ids=st_device_ids, num_sub_workers=0, comm_profile="sim"
    )
    chips = tuple(
        child.register(obj) for obj in (build_chip_callable(st_platform), _build_tload_callable(st_platform, _RUNTIME))
    )
    node = worker.add_worker(child)
    exercise_domain_tensors(worker, [(node, 0)], st_platform, local=(node, chips), cached=cached)
