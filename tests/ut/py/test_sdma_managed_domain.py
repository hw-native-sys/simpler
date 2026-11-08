# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import ctypes
import struct
from contextlib import nullcontext
from multiprocessing.shared_memory import SharedMemory
from types import SimpleNamespace
from typing import Any, cast

import pytest
import simpler.worker as worker_mod
from simpler.comm_domain import DomainConfig
from simpler.orchestrator import Orchestrator, _callback_run
from simpler.worker import Worker


def config(q=None, engines=("sdma",)):
    return DomainConfig.capture((0, 1), 128, (), True, ("a5",), engines, q)


@pytest.mark.parametrize("q", [1, 3, 48])
def test_explicit_q_is_part_of_domain_reuse_identity(q):
    explicit = config(q)
    assert explicit.sdma_sdk_queue_num == q
    assert not explicit.difference(config(q))
    assert "sdma_sdk_queue_num" in explicit.difference(config())
    assert "sdma_sdk_queue_num" in explicit.difference(config(2 if q != 2 else 3))


@pytest.mark.parametrize("q", [True, False, 0, -1, 49, 1.5, "1"])
def test_invalid_queue_count_does_not_become_default_one(q):
    with pytest.raises((TypeError, ValueError)):
        config(q)


@pytest.mark.parametrize("engines", [(), ("urma",), ("sdma", "urma"), ("rdma",)])
def test_managed_contract_is_explicitly_sdma_only(engines):
    with pytest.raises(ValueError, match="exactly"):
        config(1, engines)


def test_managed_contract_refuses_non_a5_before_allocation():
    worker = cast(Worker, SimpleNamespace(level=3, _config={"device_ids": (0, 1), "platform": "a2a3sim"}))
    with pytest.raises(ValueError, match="requires A5"):
        Worker._domain_config(worker, (0, 1), 128, (), True, ("sdma",), 1)


@pytest.mark.parametrize("method", ["allocate_domain", "get_or_allocate_domain"])
@pytest.mark.parametrize("queues", [None, 2], ids=["legacy", "multi-queue"])
def test_public_orchestration_preserves_sdk_queue_configuration(monkeypatch, method, queues):
    worker = Worker(level=3, platform="a5", runtime="host_build_graph", device_ids=[0, 1])
    monkeypatch.setattr(worker, "_building_run_resources", object())
    monkeypatch.setattr(worker, "_control_admission", lambda _api: nullcontext())
    allocated = []

    class AllocationReached(Exception):
        pass

    def allocate(**request):
        allocated.append(request)
        raise AllocationReached

    # Exercise the public API and real configuration validation without touching hardware.
    monkeypatch.setattr(worker, "_allocate_domain", allocate)
    orch = Orchestrator(cast(Any, None), worker)
    with _callback_run(1, worker), pytest.raises(AllocationReached):
        getattr(orch, method)(
            name="queues", members=[0, 1], window_size=128, workspace_engines=("sdma",), sdma_sdk_queue_num=queues
        )
    assert allocated[0]["sdma_sdk_queue_num"] == queues
    assert allocated[0]["workspace_mask"] == 1


def request_bytes(**changes):
    fields = dict(
        version=1,
        header_bytes=48,
        allocation_id=7,
        rank_count=2,
        domain_rank=0,
        window_size=128,
        buffer_count=1,
        workspace_mask=1,
        access_contract=2,
        queue_num=3,
    )
    fields.update(changes)
    return worker_mod._DOMAIN_SDMA_REQ_HEADER.pack(*fields.values()) + struct.pack("<QII", 64, 0, 1)


@pytest.mark.parametrize(
    "change",
    [
        {"version": 0},
        {"header_bytes": 32},
        {"workspace_mask": 0},
        {"workspace_mask": 3},
        {"access_contract": 1},
        {"queue_num": 0},
        {"queue_num": 49},
        {"rank_count": 0},
        {"domain_rank": 2},
        {"buffer_count": 2**32 - 1},
    ],
)
def test_managed_wire_rejects_invalid_contract_or_extent(change):
    with pytest.raises(ValueError):
        worker_mod._decode_sdma_domain_request(request_bytes(**change))


def test_managed_wire_refuses_legacy_short_and_trailing_bytes():
    for payload in (bytes(32), request_bytes()[:-1], request_bytes() + b"x"):
        with pytest.raises(ValueError):
            worker_mod._decode_sdma_domain_request(payload)
    assert worker_mod._decode_sdma_domain_request(request_bytes()) == (7, 2, 0, 128, 1, 1, 3)


def test_actual_control_handler_uses_explicit_entry_and_preserves_commit_flag():
    payload = request_bytes()
    request = SharedMemory(create=True, size=len(payload))
    reply = SharedMemory(create=True, size=worker_mod._DOMAIN_REPLY_HEADER.size + 8)
    try:
        request_buf, reply_buf = request.buf, reply.buf
        assert request_buf is not None and reply_buf is not None
        request_buf[:] = payload
        mailbox = memoryview(bytearray(worker_mod.mailbox_size(worker_mod._DEFAULT_PIPELINE_DEPTH_REQUEST)))
        for offset, name in (
            (worker_mod._OFF_ARGS, request.name),
            (worker_mod._OFF_ARGS + worker_mod._CTRL_SHM_NAME_BYTES, reply.name),
        ):
            mailbox[offset : offset + len(name)] = name.encode()
        called = []

        def managed(handle, allocation, ranks, rank, size, commit, *, queue_num):
            called.append((handle, allocation, ranks, rank, size, queue_num))
            ctypes.c_uint64.from_address(commit).value = 1
            return 0xC7, 0xB000, size

        cw = cast(Any, SimpleNamespace(comm_alloc_domain_windows_sdma_sdk_v1=managed))
        worker_mod._handle_ctrl_alloc_domain(cw, mailbox, base_comm_handle=1, sdma_sdk=True)
        assert called == [(1, 7, [0, 1], 0, 128, 3)]
        assert worker_mod._domain_reply_committed(reply)
        assert worker_mod._DOMAIN_REPLY_HEADER.unpack_from(reply_buf)[1:4] == (0xC7, 0xB000, 128)
    finally:
        for shm in (request, reply):
            shm.close()
            shm.unlink()
