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

_NUMEL = N_ROWS * N_COLS
_NBYTES = _NUMEL * 4
_RUNTIME = "tensormap_and_ringbuffer"
_ACCEPTED_ARGS = {}


def remote_domain_add(orch, args, cfg, chip=None):
    from simpler.remote_l3_session import get_inner_handle  # noqa: PLC0415

    if args.tensor_count() == 0:
        ref, chip, chip_args, target = _ACCEPTED_ARGS.pop(int(args.scalar(0)))
        with pytest.raises(KeyError):
            orch.get_domain(ref)
        with pytest.raises(ValueError, match="stale|not live"):
            orch.submit_next_level(chip, chip_args, cfg, worker=target)
        return
    rank = int(args.scalar(4))
    ref = CommDomainRef.from_tensor(args.tensor(0))
    domain = orch.get_domain(ref)
    assert domain.borrowed and domain.ref == ref
    member = domain.member(rank)
    payload = member.buffers["payload"]
    for tensor in (args.tensor(0), args.tensor(1)):
        assert tensor.buffer == payload.to_descriptor()
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
    _ACCEPTED_ARGS[rank] = ref, chip, chip_args, member.local_worker_id
    orch.submit_next_level(chip, chip_args, cfg, worker=member.local_worker_id)


def exercise_domain_tensors(worker, members, callable_obj, *, local_chip=None, cached=True):
    """Keep setup and assertions identical while topology and routing vary."""
    outputs, saved, cached_args = [], [], []
    nodes = tuple(dict.fromkeys(node for node, _chip in members))
    remote_nodes = [node for node in nodes if local_chip is None or node != local_chip[0]]
    request = dict(
        name="tp",
        members=members,
        window_size=_NBYTES + 4096,
        buffers=[CommBufferSpec("payload", "float32", _NUMEL + 32, _NBYTES + 128)],
        retain_after_run=True,
    )
    try:
        chip = worker.register(callable_obj)
        routes = {}
        if local_chip is not None:

            def forward(orch, args, cfg):
                remote_domain_add(orch, args, cfg, local_chip[1])

            routes[local_chip[0]] = worker.register(forward)
        if remote_nodes:
            remote = worker.register(RemoteCallable(__name__ + ":remote_domain_add"), workers=remote_nodes)
            routes.update(dict.fromkeys(remote_nodes, remote))
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
                for offset in range(0, 32, 8):
                    args.add_scalar(int.from_bytes(chip.digest[offset : offset + 8], "little"))
                args.add_scalar(rank)
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
            for rank, (node, _chip) in enumerate(members):
                assert replacement.member(rank).buffers["payload"].identity != identities[rank]
                args = TaskArgs()
                args.add_scalar(rank)
                orch.submit_next_level(routes[node], args, cfg, worker=node)
            replacement.release()

        worker.run(rebuild)
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
    exercise_domain_tensors(worker, [(node, 0)], build_chip_callable(st_platform))
