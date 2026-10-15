# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Legacy communicator controls use child-local state, not runtime attributes."""

import ctypes
import struct
from contextlib import ExitStack
from multiprocessing.shared_memory import SharedMemory
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
import simpler.worker as worker_mod
from simpler.buffer import LocalEndpointBufferIdentityAllocator
from simpler.comm_provider import (
    DeviceAllocationTarget,
    ProviderRegionStore,
    RegionAllocationContext,
    RegionEnvironmentKind,
)


@pytest.fixture(autouse=True, params=[1, 16384], ids=["exact-size", "page-rounded"])
def shared_memory_granularity(monkeypatch, request):
    shared_memory = SharedMemory

    def allocate(*, create, size):
        granularity = request.param
        rounded_size = ((size + granularity - 1) // granularity) * granularity
        return shared_memory(create=create, size=rounded_size)

    monkeypatch.setattr(f"{__name__}.SharedMemory", allocate)


def _run_controls(monkeypatch, init_results, commands):
    cw = MagicMock(spec_set=("comm_init", "_impl", "set_exported_device_regions_live"))
    cw.comm_init.side_effect = init_results
    cw._impl = SimpleNamespace(
        comm_alloc_domain_windows=MagicMock(return_value=(0xC7, 0xB000)),
        comm_release_domain_windows=MagicMock(),
    )
    responses = []
    owner = worker_mod.mint_owner_instance_id()
    store = ProviderRegionStore(
        RegionAllocationContext(environment_kind=RegionEnvironmentKind.SIM, target=DeviceAllocationTarget(0)),
        identity_allocator=LocalEndpointBufferIdentityAllocator(owner),
    )
    with ExitStack() as stack:

        def allocate(size):
            shm = SharedMemory(create=True, size=size)
            stack.callback(shm.unlink)
            stack.callback(shm.close)
            return shm

        rootinfo_path = b"/shared/rootinfo\0"
        init_request = allocate(worker_mod._COMM_INIT_HEADER.size + len(rootinfo_path))
        init_buf = init_request.buf
        assert init_buf is not None
        worker_mod._COMM_INIT_HEADER.pack_into(init_buf, 0, 0, 1)
        path_offset = worker_mod._COMM_INIT_HEADER.size
        init_buf[path_offset : path_offset + len(rootinfo_path)] = rootinfo_path
        domain_request = allocate(worker_mod._DOMAIN_REQ_HEADER.size + 4)
        domain_buf = domain_request.buf
        assert domain_buf is not None
        worker_mod._DOMAIN_REQ_HEADER.pack_into(domain_buf, 0, 7, 1, 0, 64, 0)
        struct.pack_into("<I", domain_buf, worker_mod._DOMAIN_REQ_HEADER.size, 0)
        reply = allocate(worker_mod._DOMAIN_REPLY_HEADER.size)
        with memoryview(bytearray(worker_mod.mailbox_size(1))) as mailbox:
            address = ctypes.addressof(ctypes.c_char.from_buffer(mailbox))

            def serve(_buf, _state_addr, *, handle_control, **_kwargs):
                for command in commands:
                    request = init_request if command == worker_mod._CTRL_COMM_INIT else domain_request
                    for offset, name in (
                        (worker_mod._OFF_ARGS, request.name),
                        (worker_mod._OFF_ARGS + worker_mod._CTRL_SHM_NAME_BYTES, reply.name),
                    ):
                        mailbox[offset : offset + worker_mod._CTRL_SHM_NAME_BYTES] = name.encode().ljust(
                            worker_mod._CTRL_SHM_NAME_BYTES, b"\0"
                        )
                    responses.append(handle_control(command))

            monkeypatch.setattr(worker_mod, "_run_mailbox_loop", serve)
            worker_mod._run_chip_main_loop(
                cw,
                mailbox,
                address,
                address + worker_mod._OFF_STATE,
                0,
                {},
                {},
                {},
                owner,
                chip_platform="a2a3sim",
                provider_region_store=store,
            )
        reply_buf = reply.buf
        assert reply_buf is not None
        published = worker_mod._DOMAIN_REPLY_HEADER.unpack_from(reply_buf, 0)
    return cw, responses, published


def test_bootstrap_handle_is_forwarded_without_mutating_runtime(monkeypatch):
    cw, responses, published = _run_controls(
        monkeypatch,
        [0x1234, 0x1234],
        [
            worker_mod._CTRL_COMM_INIT,
            worker_mod._CTRL_COMM_INIT,
            worker_mod._CTRL_ALLOC_DOMAIN,
            worker_mod._CTRL_RELEASE_DOMAIN,
        ],
    )
    assert responses == [(0, "")] * 4
    assert cw.comm_init.call_args_list == [call(0, 1, "/shared/rootinfo")] * 2
    allocated = cw._impl.comm_alloc_domain_windows.call_args.args
    assert allocated[:5] == (0x1234, 7, [0], 0, 64)
    assert allocated[5] != 0
    cw._impl.comm_release_domain_windows.assert_called_once_with(0x1234, 7, 1, 0)
    assert published == (1, 0xC7, 0xB000, 0)

    cold, refused, _ = _run_controls(monkeypatch, [], [worker_mod._CTRL_ALLOC_DOMAIN, worker_mod._CTRL_RELEASE_DOMAIN])
    assert all(code == 1 and "CTRL_COMM_INIT must succeed first" in message for code, message in refused)
    cold.comm_init.assert_not_called()
    cold._impl.comm_alloc_domain_windows.assert_not_called()
    cold._impl.comm_release_domain_windows.assert_not_called()


@pytest.mark.parametrize("failed_init", [0, RuntimeError("init failed")])
def test_failed_bootstrap_rejects_domain_controls_until_retry(monkeypatch, failed_init):
    cw, responses, _ = _run_controls(
        monkeypatch,
        [failed_init, 0x2345],
        [
            worker_mod._CTRL_ALLOC_DOMAIN,
            worker_mod._CTRL_RELEASE_DOMAIN,
            worker_mod._CTRL_COMM_INIT,
            worker_mod._CTRL_ALLOC_DOMAIN,
            worker_mod._CTRL_RELEASE_DOMAIN,
            worker_mod._CTRL_COMM_INIT,
            worker_mod._CTRL_ALLOC_DOMAIN,
            worker_mod._CTRL_RELEASE_DOMAIN,
        ],
    )
    assert all(
        code == 1 and "CTRL_COMM_INIT must succeed first" in message
        for code, message in [responses[i] for i in (0, 1, 3, 4)]
    )
    assert responses[2][0] == 1
    assert ("returned 0 handle" if failed_init == 0 else "init failed") in responses[2][1]
    assert responses[5:] == [(0, "")] * 3
    assert cw._impl.comm_alloc_domain_windows.call_count == 1
    cw._impl.comm_release_domain_windows.assert_called_once_with(0x2345, 7, 1, 0)
    assert cw._impl.comm_alloc_domain_windows.call_args.args[0] == 0x2345


@pytest.mark.parametrize("failed_init", [0, RuntimeError("retry failed")])
def test_failed_reinitialization_preserves_the_established_handle(monkeypatch, failed_init):
    cw, responses, _ = _run_controls(
        monkeypatch,
        [0x3456, failed_init],
        [
            worker_mod._CTRL_COMM_INIT,
            worker_mod._CTRL_COMM_INIT,
            worker_mod._CTRL_ALLOC_DOMAIN,
            worker_mod._CTRL_RELEASE_DOMAIN,
        ],
    )
    assert responses[0] == (0, "")
    assert responses[1][0] == 1
    assert ("returned 0 handle" if failed_init == 0 else "retry failed") in responses[1][1]
    assert responses[2:] == [(0, "")] * 2
    assert cw._impl.comm_alloc_domain_windows.call_args.args[0] == 0x3456
    cw._impl.comm_release_domain_windows.assert_called_once_with(0x3456, 7, 1, 0)
