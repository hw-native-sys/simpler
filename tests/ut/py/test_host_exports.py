# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import gc
import threading
import weakref
from contextlib import closing, contextmanager
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pytest
from simpler._host_exports import HostSharedMemory, seal_host_mappings


@pytest.fixture
def storage():
    mapping = HostSharedMemory(create=True, size=64)
    yield mapping
    mapping.close()
    mapping.unlink()


@pytest.mark.parametrize("kind", ["ndarray", "numpy_slice", "frombuffer", "memoryview_slice"])
@pytest.mark.parametrize("action", ["seal", "close", "unlink"])
def test_derivatives_block_release_without_revoking_storage(storage, kind, action):
    if kind == "memoryview_slice":
        view = storage.buf[4:12]
    elif kind == "frombuffer":
        view = np.frombuffer(storage.buf, dtype=np.int32)[1:3]
    else:
        view = np.ndarray((16,), dtype=np.int32, buffer=storage.buf)
        if kind == "numpy_slice":
            view = view[1:3]
    try:
        with pytest.raises(BufferError, match="live CPU exports"):
            getattr(storage, action)()
        assert not storage._closing
        view[0] = 31
        assert view[0] == 31
        with storage.buf as other:
            assert len(other) == storage.size
        with closing(SharedMemory(name=storage.name)) as check:
            assert check.size == storage.size
    finally:
        del view


def test_numpy_slice_pins_export_after_explicit_memoryview_release(storage):
    with storage.buf as exported:
        array = np.ndarray((16,), dtype=np.int32, buffer=exported)
        view = array[2:4]
    del array
    try:
        with pytest.raises(BufferError, match="export"):
            storage.close()
        view[:] = 47
        with storage.buf as check:
            assert bytes(check[8:16]) == np.array([47, 47], dtype=np.int32).tobytes()
    finally:
        del view


def test_last_array_keeps_storage_alive_until_its_release():
    storage = HostSharedMemory(create=True, size=64)
    name = storage.name
    backing = storage._mmap
    owner = weakref.ref(storage)
    array = np.ndarray((16,), dtype=np.int32, buffer=storage.buf)[2:4]
    del storage
    gc.collect()
    try:
        assert owner() is not None and not backing.closed
        array[0] = 59
    finally:
        del array
        gc.collect()
        cleanup = SharedMemory(name=name)
        cleanup.close()
        cleanup.unlink()
    assert owner() is None and backing.closed


def test_export_check_without_sealing_preserves_admission(storage):
    with storage.buf as view:
        with pytest.raises(BufferError, match="export"):
            seal_host_mappings((storage,), seal=False)
        assert not storage._closing
        view[0] = 17
    seal_host_mappings((storage,), seal=False)
    with storage.buf as view:
        assert view[0] == 17
    assert not storage._closing


def test_group_seal_is_all_or_none(storage):
    other = HostSharedMemory(create=True, size=64)
    view = other.buf
    assert view is not None
    try:
        with pytest.raises(BufferError, match="export"):
            seal_host_mappings((storage, other))
        assert not storage._closing and not other._closing
        with storage.buf as writable:
            writable[0] = 17
        view.release()
        seal_host_mappings((storage, other))
        for mapping in (storage, other):
            with pytest.raises(BufferError, match="closing"):
                mapping.buf
    finally:
        view.release()
        other.close()
        other.unlink()


def test_export_and_seal_have_one_admission_boundary(storage):
    entered, finish = threading.Event(), threading.Event()
    errors = []

    def seal_storage():
        try:
            storage.seal()
            entered.set()
            assert finish.wait(5)
            storage.close()
        except BaseException as error:
            errors.append(error)
            entered.set()

    thread = threading.Thread(target=seal_storage)
    thread.start()
    try:
        assert entered.wait(5)
        assert not errors
        with pytest.raises(BufferError, match="closing"):
            storage.buf
    finally:
        finish.set()
        thread.join(5)
    assert not thread.is_alive() and not errors
    assert storage.buf is None


@pytest.mark.parametrize("action", ["close", "unlink"])
def test_owner_guard_refuses_release_before_sealing_exports(storage, action):
    active = True

    @contextmanager
    def guard():
        if active:
            raise RuntimeError("registered device use")
        yield

    storage._release_guard = guard
    try:
        with pytest.raises(RuntimeError, match="registered"):
            getattr(storage, action)()
        assert not storage._closing
        with storage.buf as view:
            view[0] = 53
        storage.seal()
        assert storage._closing
    finally:
        active = False


@pytest.mark.parametrize("action", ["buf", "seal", "close", "unlink"])
def test_foreign_process_is_rejected_before_inherited_lock(storage, action):
    class InheritedLock:
        def __enter__(self):
            pytest.fail("foreign process acquired the inherited export lock")

        def __exit__(self, *_):
            return None

    pid, lock = storage._export_pid, storage._export_lock
    storage._export_pid = -1
    storage._export_lock = InheritedLock()
    try:
        with pytest.raises(ValueError, match="another process"):
            value = getattr(storage, action)
            if action != "buf":
                value()
    finally:
        storage._export_pid, storage._export_lock = pid, lock
