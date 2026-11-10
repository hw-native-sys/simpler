# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import gc
import subprocess
import sys
import threading
import weakref
from multiprocessing.shared_memory import SharedMemory
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from simpler.buffer import AddressSpace, wrap_vmm_window
from simpler.comm_domain import DomainConfig, HostBufferMapping
from simpler.task_interface import CommDomainHandle
from simpler.worker import Worker


def test_memory_source_is_part_of_reuse_identity():
    config = DomainConfig.capture([0, 1], 4096, [], True, (), ("urma",))
    host = DomainConfig.capture([0, 1], 4096, [], True, (), ("urma",), AddressSpace.HOST)
    assert config.memory_source == AddressSpace.DEVICE
    assert config.difference(host).startswith("memory_source:")
    assert host.difference(host) == ""
    with pytest.raises(ValueError, match="memory_source"):
        DomainConfig.capture([0], 4096, [], True, (), memory_source="host")


@pytest.mark.parametrize(
    "platform,runtime,engines,accepted",
    [
        ("a5", "tensormap_and_ringbuffer", (), True),
        ("a5", "tensormap_and_ringbuffer", ("urma",), True),
        ("a5", "host_build_graph", ("urma",), False),
        ("a2a3", "tensormap_and_ringbuffer", (), True),
        ("a2a3", "tensormap_and_ringbuffer", ("urma",), False),
        ("a2a3sim", "tensormap_and_ringbuffer", (), False),
        ("a5sim", "tensormap_and_ringbuffer", (), False),
        ("a5", "tensormap_and_ringbuffer", ("sdma",), False),
        ("a5", "tensormap_and_ringbuffer", ("rdma",), False),
    ],
)
def test_host_domain_capability_before_backend(platform, runtime, engines, accepted):
    owner = SimpleNamespace(level=3, _config={"device_ids": [0, 1], "platform": platform, "runtime": runtime})

    def call():
        return Worker._domain_config(cast(Any, owner), [0, 1], 4096, [], True, engines, AddressSpace.HOST)

    if accepted:
        assert call().memory_source == AddressSpace.HOST
    else:
        with pytest.raises(ValueError, match="Host-source"):
            call()


def test_cpu_window_exports_prevent_domain_release(native_host_memory):
    owner = native_host_memory()
    mapping = HostBufferMapping(owner.name, owner.size)
    domain = CommDomainHandle(name="host", _release_fn=lambda _: mapping.close(), memory_source=AddressSpace.HOST)
    domain._host_mappings.append(mapping)
    buf = wrap_vmm_window(0x100080, 64, b"hosttest", 1, "L3", owner_worker_id=0)
    buf.shm = mapping
    assert isinstance(buf.shm, SharedMemory)
    assert buf.shm.buf is not None
    assert len(buf.shm.buf) == owner.size
    exported = buf.shm.buf[128:192]
    try:
        exported[:] = b"a" * 64
        assert bytes(owner.buf[128:192]) == b"a" * 64
        assert bytes(owner.buf[:128]) == bytes(128)
        owner.buf[130] = 42
        assert exported[2] == 42
        with pytest.raises(ValueError, match="domain.release"):
            buf.close()
        with pytest.raises(RuntimeError, match="belongs to L2"):
            buf.shm.unlink()
        assert not buf.closed and buf.shm.buf is not None
        with pytest.raises(BufferError, match="export"):
            domain.release()
        assert not domain.released
        assert exported[2] == 42
        exported[3] = 43
        assert owner.buf[131] == 43
    finally:
        exported.release()
        domain.release()
    assert domain.released and buf.shm.buf is None


@pytest.mark.parametrize("kind", ["ndarray", "numpy_slice", "frombuffer", "memoryview_slice"])
def test_cpu_exports_pin_mapping_until_last_derivative_is_dropped(native_host_memory, kind):
    owner = native_host_memory()
    mapping = HostBufferMapping(owner.name, owner.size)
    assert mapping.buf is not None
    if kind == "memoryview_slice":
        view = mapping.buf[4:12]
    elif kind == "frombuffer":
        view = np.frombuffer(mapping.buf, dtype=np.int32)[1:3]
    else:
        view = np.ndarray((8,), dtype=np.int32, buffer=mapping.buf)
        if kind == "numpy_slice":
            view = view[1:3]
    try:
        with pytest.raises(BufferError, match="export"):
            mapping.close()
        # Only touch the view after checking that close really was refused.
        view[0] = 17
        assert view[0] == 17
        assert mapping.buf is not None
    finally:
        del view
        mapping.close()
    assert mapping.buf is None


def test_domain_release_checks_all_rank_exports_before_any_close(native_host_memory):
    owners = [native_host_memory(), native_host_memory()]
    mappings = [HostBufferMapping(owner.name, owner.size) for owner in owners]
    calls = []

    def release(_):
        calls.append("release")
        for mapping in mappings:
            mapping.close()

    domain = CommDomainHandle(name="host", _release_fn=release, memory_source=AddressSpace.HOST)
    domain._host_mappings.extend(mappings)
    view = np.ndarray((8,), dtype=np.int32, buffer=mappings[1].buf)[1:3]
    try:
        with pytest.raises(BufferError, match="export"):
            domain.release()
        assert not domain.released and calls == []
        first_export = mappings[0].buf
        assert first_export is not None
        with first_export as first:
            first[0] = 23
        assert owners[0].buf[0] == 23
    finally:
        del view
        domain.release()
    assert calls == ["release"]


def test_numpy_slice_keeps_borrow_after_original_memoryview_is_released(native_host_memory):
    owner = native_host_memory()
    mapping = HostBufferMapping(owner.name, owner.size)
    assert mapping.buf is not None
    with mapping.buf as exported:
        array = np.ndarray((8,), dtype=np.int32, buffer=exported)
        view = array[1:3]
    del array
    try:
        with pytest.raises(BufferError, match="export"):
            mapping.close()
        view[:] = 29
        assert bytes(owner.buf[4:12]) == np.array([29, 29], dtype=np.int32).tobytes()
    finally:
        del view
        mapping.close()


def test_last_numpy_export_releases_mapping_after_owner_reference_is_dropped(native_host_memory):
    owner = native_host_memory()
    mapping = HostBufferMapping(owner.name, owner.size)
    backing = mapping._mmap
    owner_ref = weakref.ref(mapping)
    view = np.ndarray((8,), dtype=np.int32, buffer=mapping.buf)[1:3]
    del mapping
    gc.collect()
    assert owner_ref() is not None and not backing.closed
    view[0] = 43
    del view
    gc.collect()
    assert owner_ref() is None and backing.closed


def test_domain_release_fences_exports_before_backend_call(native_host_memory):
    owner = native_host_memory()
    mapping = HostBufferMapping(owner.name, owner.size)
    entered, finish = threading.Event(), threading.Event()
    errors = []

    def release(_):
        entered.set()
        assert finish.wait(5)
        mapping.close()

    domain = CommDomainHandle(name="host", _release_fn=release, memory_source=AddressSpace.HOST)
    domain._host_mappings.append(mapping)

    def close_domain():
        try:
            domain.release()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=close_domain)
    thread.start()
    try:
        assert entered.wait(5)
        with pytest.raises(BufferError, match="closing"):
            mapping.buf
    finally:
        finish.set()
        thread.join(5)
    assert not thread.is_alive() and not errors
    assert mapping.buf is None


def test_cpu_import_does_not_unlink_native_owned_name(native_host_memory):
    owner = native_host_memory()
    mapping = HostBufferMapping(owner.name, owner.size)
    mapping.close()
    reopened = HostBufferMapping(owner.name, owner.size)
    reopened.close()
    with pytest.raises(ValueError, match="smaller"):
        HostBufferMapping(owner.name, owner.size + 1)
    reopened = HostBufferMapping(owner.name, owner.size)
    reopened.close()


@pytest.mark.parametrize("close_mapping", [False, True])
def test_borrower_process_exit_keeps_native_backing(native_host_memory, close_mapping):
    owner = native_host_memory()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from simpler.comm_domain import HostBufferMapping; "
            f"shm = HostBufferMapping({owner.name!r}, {owner.size}); "
            "shm.buf[17] = 42; " + ("shm.close()" if close_mapping else ""),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert "resource_tracker" not in result.stderr
    assert owner.buf[17] == 42
    reopened = HostBufferMapping(owner.name, owner.size)
    assert reopened.buf is not None
    assert reopened.buf[17] == 42
    reopened.close()


def test_device_buffer_has_no_cpu_shared_memory():
    buf = wrap_vmm_window(0x100000, 64, b"hosttest", 1, "L3", owner_worker_id=0)
    assert buf.shm is None
