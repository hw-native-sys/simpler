# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared L4 domain-Tensor acceptance flow for local, TCP and MPI workers."""

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
_ACCEPTED_ARGS = {}
_CONTEXT_ARGS = {}


def _borrow_domain(orch, args):
    ref = CommDomainRef(int(args.scalar(4)).to_bytes(8, "little"), int(args.scalar(5)), int(args.scalar(6)))
    rank = int(args.scalar(7))
    domain = orch.get_domain(ref)
    assert domain.borrowed and domain.ref == ref
    node = domain.member(rank).node_worker_id
    for member in domain.members:
        assert bool(member.buffers) == (member.node_worker_id == node)
        if member.node_worker_id != node:
            with pytest.raises(KeyError):
                domain.local_context(member.domain_rank)
    return domain, rank


def _add_domain_scalars(args, domain, rank, chip):
    for offset in range(0, 32, 8):
        args.add_scalar(int.from_bytes(chip.digest[offset : offset + 8], "little"))
    args.add_scalar(int.from_bytes(domain.ref.owner_instance_id, "little"))
    args.add_scalar(domain.ref.domain_id)
    args.add_scalar(domain.ref.generation)
    args.add_scalar(rank)


def remote_domain_add(orch, args, cfg, chip=None):
    from simpler.remote_l3_session import get_inner_handle  # noqa: PLC0415

    if args.tensor_count() == 0:
        ref, chip, chip_args, target = _ACCEPTED_ARGS.pop(int(args.scalar(0)))
        with pytest.raises(KeyError):
            orch.get_domain(ref)
        with pytest.raises(ValueError, match="stale|not live"):
            orch.submit_next_level(chip, chip_args, cfg, worker=target)
        return
    domain, rank = _borrow_domain(orch, args)
    member = domain.member(rank)
    payload = member.buffers["payload"]
    for tensor in (args.tensor(0), args.tensor(1)):
        assert tensor.buffer.identity == payload.identity
        assert tensor.buffer.nbytes == payload.nbytes
        assert (tensor.shapes, tensor.strides, tensor.dtype) == ((_NUMEL,), (1,), DataType.FLOAT32.value)
    assert domain.buffers[0].count == _NUMEL + 32
    assert (args.tensor(0).byte_offset, args.tensor(1).byte_offset) == (64, 128)
    assert orch.copy_from_domain(domain, rank, 4, buffer="payload", offset=64) == struct.pack("<f", rank * 100 + 16)
    with pytest.raises(RuntimeError, match="borrowed"):
        domain.release()
    with pytest.raises(RuntimeError, match="borrowed"):
        domain.__enter__()
    digest = b"".join(int(args.scalar(i)).to_bytes(8, "little") for i in range(4))
    if chip is None:
        chip = get_inner_handle(digest.hex())
    chip_args = TaskArgs()
    for i, tag in enumerate((TensorArgType.INPUT, TensorArgType.INPUT, TensorArgType.OUTPUT_EXISTING)):
        chip_args.add_tensor(args.tensor(i), tag)
    with pytest.raises(ValueError, match="wrong worker|stale"):
        orch.submit_next_level(chip, chip_args, cfg, worker=member.local_worker_id ^ 1)
    _ACCEPTED_ARGS[rank] = domain.ref, chip, chip_args, member.local_worker_id
    orch.submit_next_level(chip, chip_args, cfg, worker=member.local_worker_id)


def remote_domain_tload(orch, args, cfg, chip=None):
    from simpler.remote_l3_session import get_inner_handle  # noqa: PLC0415

    domain, rank = _borrow_domain(orch, args)
    context = domain.local_context(rank)
    assert context.domain_rank == rank
    member = domain.member(rank)
    if chip is None:
        digest = b"".join(int(args.scalar(i)).to_bytes(8, "little") for i in range(4))
        chip = get_inner_handle(digest.hex())
    chip_args = TaskArgs()
    for index, (name, tag) in enumerate(
        (("payload", TensorArgType.INPUT), ("context_result", TensorArgType.OUTPUT_EXISTING))
    ):
        tensor = args.tensor(index)
        assert tensor.buffer.identity == member.buffers[name].identity
        assert tensor.byte_offset == 64
        chip_args.add_tensor(tensor, tag)
    chip_args.add_scalar(context.domain_size)
    chip_args.add_scalar(context.device_ctx)
    _CONTEXT_ARGS[rank] = chip_args
    orch.submit_next_level(chip, chip_args, cfg, worker=member.local_worker_id)


def exercise_domain_tensors(  # noqa: PLR0915 -- shared setup, cross-run lifetime and peer-context assertions
    worker, members, callable_obj, *, tload_callable, local_chip=None, local_tload_chip=None, cached=True
):
    """Keep setup and assertions identical while topology and routing vary."""
    outputs, saved, cached_args = [], [], []
    nodes = tuple(dict.fromkeys(node for node, _chip in members))
    remote_nodes = [node for node in nodes if local_chip is None or node != local_chip[0]]
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
    try:
        chip = worker.register(callable_obj)
        tload_chip = worker.register(tload_callable)
        routes, context_routes = {}, {}
        if local_chip is not None:
            assert local_tload_chip is not None

            def forward(orch, args, cfg):
                remote_domain_add(orch, args, cfg, local_chip[1])

            def forward_context(orch, args, cfg):
                remote_domain_tload(orch, args, cfg, local_tload_chip)

            routes[local_chip[0]] = worker.register(forward)
            context_routes[local_chip[0]] = worker.register(forward_context)
        if remote_nodes:
            remote = worker.register(RemoteCallable(__name__ + ":remote_domain_add"), workers=remote_nodes)
            routes.update(dict.fromkeys(remote_nodes, remote))
            remote_context = worker.register(RemoteCallable(__name__ + ":remote_domain_tload"), workers=remote_nodes)
            context_routes.update(dict.fromkeys(remote_nodes, remote_context))
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
                data = [float(rank * 100 + index) for index in range(_NUMEL + 32)]
                orch.copy_to_domain(domain, rank, struct.pack(f"<{len(data)}f", *data), buffer="payload")
                payload = domain.member(rank).buffers["payload"]
                args = TaskArgs()
                for offset in (64, 128):
                    args.add_tensor(payload.tensor([_NUMEL], DataType.FLOAT32, byte_offset=offset), TensorArgType.INPUT)
                result = (
                    RemoteTensorRef(output, shape=(_NUMEL,), dtype=DataType.FLOAT32)
                    if isinstance(output, RemoteBufferHandle)
                    else output.tensor([_NUMEL], DataType.FLOAT32)
                )
                args.add_tensor(result, TensorArgType.OUTPUT_EXISTING)
                _add_domain_scalars(args, domain, rank, chip)
                cached_args.append(args)

        worker.run(first)
        domain = saved[0]
        domain_ref = domain.ref
        allocation_count = worker._next_alloc_id
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
        if cached:
            assert run._resources.live_global_domains["tp"] is domain
            domain.release()
        assert domain.released and not domain.freed
        run.wait()
        assert domain.freed and worker.live_domains == {}
        for rank, output in enumerate(outputs):
            if isinstance(output, RemoteBufferHandle):
                host = ctypes.create_string_buffer(_NBYTES)
                worker.remote_copy_from(output, host, _NBYTES)
                data = host.raw
            else:
                data = output.shm.buf
            assert struct.unpack_from(f"<{_NUMEL}f", data) == tuple(
                float(rank * 200 + 48 + index * 2) for index in range(_NUMEL)
            )

        def rebuild(orch, _args, cfg):
            replacement = orch.get_or_allocate_domain(**request)
            assert replacement.ref != domain_ref
            saved.append(replacement)
            for rank, (node, _chip) in enumerate(members):
                assert replacement.member(rank).buffers["payload"].identity != identities[rank]
                values = [float(rank * 100 + index) for index in range(COUNT + 16)]
                orch.copy_to_domain(replacement, rank, struct.pack(f"<{len(values)}f", *values), buffer="payload")
                orch.copy_to_domain(
                    replacement,
                    rank,
                    struct.pack(f"<{COUNT + 16}f", *([-1.0] * (COUNT + 16))),
                    buffer="context_result",
                )
            for rank, (node, _chip) in enumerate(members):
                args = TaskArgs()
                args.add_scalar(rank)
                orch.submit_next_level(routes[node], args, cfg, worker=node)
                args = TaskArgs()
                for name, tag in (("payload", TensorArgType.INPUT), ("context_result", TensorArgType.OUTPUT_EXISTING)):
                    args.add_tensor(
                        replacement.member(rank).buffers[name].tensor([COUNT], DataType.FLOAT32, byte_offset=64),
                        tag,
                    )
                _add_domain_scalars(args, replacement, rank, tload_chip)
                orch.submit_next_level(context_routes[node], args, cfg, worker=node)

        worker.run(rebuild)
        replacement = saved[-1]
        assert not replacement.released

        def read_and_release(orch, _args, _cfg):
            assert orch.get_domain("tp") is replacement
            rank_count = len(members)
            expected = tuple(
                float(rank_count * (index + 16) + 100 * rank_count * (rank_count - 1) // 2) for index in range(COUNT)
            )
            for rank in range(rank_count):
                raw = orch.copy_from_domain(replacement, rank, (COUNT + 16) * 4, buffer="context_result")
                values = struct.unpack(f"<{COUNT + 16}f", raw)
                assert values[:16] == (-1.0,) * 16
                assert values[16:] == expected
            replacement.release()

        worker.run(read_and_release)
        assert replacement.freed
        assert worker.live_domains == {}
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
            endpoint=tcp_daemon,
            platform=st_platform,
            device_ids=(int(st_device_ids[0]),),
            comm_profile="sim",
        )
    )
    exercise_domain_tensors(
        worker,
        [(node, 0)],
        build_chip_callable(st_platform),
        tload_callable=_build_tload_callable(st_platform, _RUNTIME),
    )
