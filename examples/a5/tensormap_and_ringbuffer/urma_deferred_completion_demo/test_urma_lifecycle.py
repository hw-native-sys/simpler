# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise real HCCL teardown and check its return code before device reset."""

import ctypes as C
import multiprocessing
import os
import time

import pytest

from simpler_setup.runtime_builder import RuntimeBuilder


def _lifecycle_rank(rank, device, host_path, rootinfo):
    acl = C.CDLL("libascendcl.so")
    host = C.CDLL(host_path)
    void = C.c_void_p
    u64 = C.c_uint64
    u32 = C.c_uint32
    host.comm_init.argtypes = [C.c_int, C.c_int, void, C.c_char_p]
    host.comm_init.restype = void
    host.comm_alloc_windows.argtypes = [void, C.c_size_t, C.POINTER(u64)]
    host.comm_alloc_domain_windows.argtypes = [
        void,
        u64,
        C.POINTER(u32),
        C.c_size_t,
        u32,
        C.c_size_t,
        C.POINTER(u64),
        C.POINTER(u64),
    ]
    host.comm_release_domain_windows.argtypes = [void, u64, C.c_size_t, u32]
    host.comm_destroy.argtypes = [void]
    host.comm_derive_context.argtypes = [
        void,
        C.POINTER(u32),
        C.c_size_t,
        u32,
        C.c_size_t,
        C.c_size_t,
        C.POINTER(u64),
    ]
    acl.aclrtCreateStream.argtypes = [C.POINTER(void)]
    acl.aclrtDestroyStream.argtypes = [void]
    assert acl.aclInit(None) == 0
    assert acl.aclrtSetDevice(device) == 0
    stream = void()
    assert acl.aclrtCreateStream(C.byref(stream)) == 0
    members = (u32 * 2)(0, 1)
    for iteration in range(2):
        handle = host.comm_init(rank, 2, stream, f"{rootinfo}-{iteration}".encode())
        assert handle
        base_ctx = u64()
        assert host.comm_alloc_windows(handle, 4 * 1024 * 1024, C.byref(base_ctx)) == 0
        invalid_ctx = u64()
        assert host.comm_derive_context(handle, members, 2, rank, 1, 64, C.byref(invalid_ctx)) != 0
        assert invalid_ctx.value == 0
        for domain_id in (1, 2, 3):
            ctx, window = u64(), u64()
            assert (
                host.comm_alloc_domain_windows(
                    handle,
                    domain_id,
                    members,
                    2,
                    rank,
                    4 * 1024 * 1024,
                    C.byref(ctx),
                    C.byref(window),
                )
                == 0
            )
            assert ctx.value and window.value
            if domain_id != 2:
                assert host.comm_release_domain_windows(handle, domain_id, 2, rank) == 0
                # The ID still names retained HCCL resources and cannot be reused.
                assert (
                    host.comm_alloc_domain_windows(
                        handle,
                        domain_id,
                        members,
                        2,
                        rank,
                        4096,
                        C.byref(ctx),
                        C.byref(window),
                    )
                    != 0
                )
        # Keep domain 2 live, while 1 and 3 are already logically released.
        # This assertion precedes ACL reset: quarantine cannot count as success.
        assert host.comm_destroy(handle) == 0
    assert acl.aclrtDestroyStream(stream) == 0
    assert acl.aclrtResetDevice(device) == 0
    assert acl.aclFinalize() == 0


@pytest.mark.platforms(["a5"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(2)
@pytest.mark.skipif(
    os.environ.get("SIMPLER_ENABLE_PTO_URMA_WORKSPACE", "").upper() not in {"ON", "1", "TRUE", "YES"},
    reason="requires explicit URMA overlay",
)
def test_urma_native_teardown(st_device_ids, st_platform, tmp_path):
    host = RuntimeBuilder(st_platform).get_binaries("tensormap_and_ringbuffer").host_path
    ctx = multiprocessing.get_context("spawn")
    processes = [
        ctx.Process(target=_lifecycle_rank, args=(rank, int(device), str(host), str(tmp_path / "root")))
        for rank, device in enumerate(st_device_ids)
    ]
    assert len(processes) == 2
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + 300
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        assert [p.exitcode for p in processes] == [0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join()
