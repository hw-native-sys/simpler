# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

import pytest
from simpler.global_comm_domain import (
    GLOBAL_DOMAIN_DESCRIPTOR_BYTES,
    GLOBAL_DOMAIN_PROFILE_IDS,
    GLOBAL_DOMAIN_VERSION,
    GlobalCommInitCommand,
    GlobalDomainBuffer,
    GlobalDomainCommand,
    GlobalDomainDescriptor,
    GlobalDomainMember,
    GlobalDomainPhase,
    decode_comm_init,
    decode_descriptor_table,
    decode_domain_command,
    encode_comm_init,
    encode_descriptor_table,
    encode_domain_command,
    validate_descriptor_table,
)


def _members() -> tuple[GlobalDomainMember, ...]:
    return (
        GlobalDomainMember(0, 0, 3, 0),
        GlobalDomainMember(1, 0, 7, 1),
    )


def _descriptors() -> tuple[GlobalDomainDescriptor, ...]:
    return tuple(
        GlobalDomainDescriptor(
            version=GLOBAL_DOMAIN_VERSION,
            profile_id=GLOBAL_DOMAIN_PROFILE_IDS["sim"],
            domain_rank=rank,
            rank_count=2,
            mapping_size=4096,
            handle=f"/simpler-test-{rank}".encode(),
        )
        for rank in range(2)
    )


def test_global_domain_wire_round_trips_topology_and_descriptor_table():
    init = GlobalCommInitCommand("cluster", "topology", "sim", 0, 2, _members())
    command = GlobalDomainCommand(
        phase=GlobalDomainPhase.IMPORT,
        domain_id=11,
        generation=1,
        name="tp",
        profile="sim",
        window_size=2048,
        members=_members(),
        buffers=(GlobalDomainBuffer("payload", 128),),
        descriptors=_descriptors(),
    )

    assert decode_comm_init(encode_comm_init(init)) == init
    assert decode_domain_command(encode_domain_command(command)) == command
    assert decode_descriptor_table(encode_descriptor_table(_descriptors())) == _descriptors()
    assert GLOBAL_DOMAIN_DESCRIPTOR_BYTES == 288


def test_global_domain_descriptor_table_rejects_different_mapping_sizes():
    descriptors = list(_descriptors())
    descriptors[1] = GlobalDomainDescriptor(
        version=GLOBAL_DOMAIN_VERSION,
        profile_id=GLOBAL_DOMAIN_PROFILE_IDS["sim"],
        domain_rank=1,
        rank_count=2,
        mapping_size=8192,
        handle=b"/simpler-test-1",
    )

    with pytest.raises(ValueError, match="mapping sizes differ"):
        validate_descriptor_table(tuple(descriptors), rank_count=2, profile="sim")


def test_global_domain_release_retries_after_callback_failure():
    from simpler.task_interface import GlobalCommDomainHandle  # noqa: PLC0415

    attempts = 0

    def release_fn(_handle):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient release failure")

    handle = GlobalCommDomainHandle(
        name="retry",
        members=(),
        buffers=(),
        domain_id=17,
        generation=1,
        mapping_size=4096,
        retain_after_run=False,
        _release_fn=release_fn,
    )

    with pytest.raises(RuntimeError, match="transient release failure"):
        handle.release()

    assert not handle.released
    handle.release()
    assert handle.released
    assert attempts == 2


def test_remote_compute_orch_submits_local_l2_add(monkeypatch):
    from simpler.global_comm_smoke import remote_compute_orch  # noqa: PLC0415
    from simpler.remote_l3_session import get_inner_handle  # noqa: PLC0415
    from simpler.task_interface import CallConfig, TaskArgs, TensorArgType  # noqa: PLC0415

    digest = bytes(range(32))
    chip_handle = object()
    monkeypatch.setattr(
        "simpler.remote_l3_session.get_inner_handle",
        lambda hashid: chip_handle if hashid == digest.hex() else get_inner_handle(hashid),
    )

    class FakeContext:
        buffer_ptrs = {"lhs": 0x1000, "rhs": 0x2000, "input": 0x3000}

    class FakeOrchestrator:
        def __init__(self):
            self.submitted = None

        def get_global_domain(self, domain_id):
            assert domain_id == 17
            return {0: FakeContext()}

        def submit_next_level(self, handle, task_args, cfg, *, worker):
            self.submitted = (handle, task_args, cfg, worker)

    args = TaskArgs()
    args.add_scalar(17)
    args.add_scalar(0)
    for offset in range(0, 32, 8):
        args.add_scalar(int.from_bytes(digest[offset : offset + 8], "little"))
    config = CallConfig()
    orch = FakeOrchestrator()

    remote_compute_orch(orch, args, config)  # type: ignore[arg-type]

    assert orch.submitted is not None
    handle, chip_args, submitted_config, worker_id = orch.submitted
    assert handle is chip_handle
    assert submitted_config is config
    assert worker_id == 0
    assert chip_args.tensor_count() == 3
    assert [chip_args.tensor(index).data for index in range(3)] == [0x1000, 0x2000, 0x3000]
    assert [chip_args.tensor(index).child_memory for index in range(3)] == [True, True, True]
    assert [chip_args.tag(index) for index in range(3)] == [
        TensorArgType.INPUT,
        TensorArgType.INPUT,
        TensorArgType.OUTPUT_EXISTING,
    ]


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_tcp_ports(ports: tuple[int, ...], timeout_s: float = 5.0) -> None:
    pending = set(ports)
    deadline = time.monotonic() + timeout_s
    while pending:
        for port in tuple(pending):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"remote L3 daemons did not become ready on ports {sorted(pending)}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=min(0.1, remaining)):
                    pending.remove(port)
            except OSError:
                pass
        if pending:
            time.sleep(0.01)


@pytest.mark.skipif(os.name == "nt", reason="hierarchical workers require fork")
def test_two_remote_daemons_build_and_copy_global_domain_without_mpirun():
    from simpler.task_interface import CommBufferSpec  # noqa: PLC0415
    from simpler.worker import RemoteWorkerSpec, Worker  # noqa: PLC0415

    ports = (_free_tcp_port(), _free_tcp_port())
    daemons = [
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "simpler.remote_l3_worker",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for port in ports
    ]
    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=20)
    captured: dict[str, object] = {}
    try:
        _wait_for_tcp_ports(ports)
        node_ids = tuple(
            worker.add_remote_worker(
                RemoteWorkerSpec(
                    endpoint=f"127.0.0.1:{port}",
                    platform="a2a3sim",
                    device_ids=(0,),
                    comm_profile="sim",
                )
            )
            for port in ports
        )
        worker.init()

        def parent_orch(orch, _args, _cfg):
            domain = orch.allocate_global_domain(
                name="tcp-global",
                members=((node_ids[0], 0), (node_ids[1], 0)),
                window_size=4096,
                buffers=(CommBufferSpec("payload", "uint8", 64, 64),),
                retain_after_run=True,
            )
            orch.copy_to_global_domain(domain, 0, b"node-zero", buffer="payload")
            orch.copy_to_global_domain(domain, 1, b"node-one", buffer="payload")
            captured["ranks"] = tuple(member.global_device_rank for member in domain.members)
            captured["handle"] = domain

        worker.run(parent_orch)
        assert not captured["handle"].freed

        def read_orch(orch, _args, _cfg):
            domain = captured["handle"]
            try:
                captured["rank0"] = orch.copy_from_global_domain(domain, 0, len(b"node-zero"), buffer="payload")
                captured["rank1"] = orch.copy_from_global_domain(domain, 1, len(b"node-one"), buffer="payload")
            finally:
                domain.release()

        worker.run(read_orch)
        assert captured["rank0"] == b"node-zero"
        assert captured["rank1"] == b"node-one"
        assert captured["ranks"] == (0, 1)
        assert captured["handle"].freed
    finally:
        worker.close()
        for daemon in daemons:
            daemon.terminate()
            try:
                daemon.wait(timeout=5)
            except subprocess.TimeoutExpired:
                daemon.kill()
                daemon.wait(timeout=5)
