# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared Tensor arguments at the synchronous ChipWorker boundary."""

import ctypes
import threading

import pytest
from simpler.buffer import (
    AccessMode,
    AddressSpace,
    ImportRegistry,
    TensorTransfer,
    create_host_shared_buffer,
    mint_owner_instance_id,
    wrap_device_malloc,
)
from simpler.task_interface import (
    ArgDirection,
    ChipCallable,
    ChipStorageTaskArgs,
    ChipWorker,
    DataType,
    TaskArgs,
    Tensor,
    TensorArgType,
)


@pytest.fixture
def backing():
    buffer = create_host_shared_buffer(32, mint_owner_instance_id(), 1)
    ctypes.c_float.from_address(buffer.base + 4).value = 7.0
    yield buffer
    buffer.close()


@pytest.fixture
def chip():
    class Native:
        initialized = True
        device_id = 0

        def __init__(self):
            self.runs = []
            self.fail_run = False
            self.fail_finalize = False

        def register_callable(self, *_):
            pass

        def run(self, slot, args, config):
            # A native boundary accepts only the materialized POD.
            assert isinstance(args, ChipStorageTaskArgs)
            self.runs.append(args)
            if self.fail_run:
                raise RuntimeError("unproven completion")
            if args.tensor_count():
                tensor = args.tensor(0)
                assert tensor.address_space == AddressSpace.HOST
                assert args.transfer(0) in (TensorTransfer.H2D, TensorTransfer.NONE)
                ctypes.c_float.from_address(tensor.data).value += 5.0

        def finalize(self):
            if self.fail_finalize:
                raise RuntimeError("teardown failed")
            self.initialized = False

    worker = ChipWorker()
    worker._impl = Native()
    handle = worker.register_callable(
        ChipCallable.build(signature=[ArgDirection.INOUT], func_name="test", binary=b"x", children=[])
    )
    yield worker, handle
    worker._impl.fail_finalize = False
    worker.finalize()


def args_for(buffer, *, scalar=True):
    args = TaskArgs()
    args.add_tensor(Tensor(buffer, shapes=(2,), dtype=DataType.FLOAT32, byte_offset=4), TensorArgType.INOUT)
    if scalar:
        args.add_scalar(0x8000000000000001)
    return args


def test_host_view_uses_common_materializer_and_closes_import(chip, backing, monkeypatch):
    worker, handle = chip
    args = args_for(backing)
    imports = []
    original = ImportRegistry.materialize

    def materialize(registry, descriptor):
        result = original(registry, descriptor)
        imports.append(result)
        # Accepted views and scalar bits are independent of the source container.
        args.clear()
        return result

    monkeypatch.setattr(ImportRegistry, "materialize", materialize)
    worker.run(handle, args)
    assert ctypes.c_float.from_address(backing.base + 4).value == 12.0
    native = worker._impl.runs[0]
    assert native.tensor(0).shapes == (2,)
    assert native.scalar(0) == 0x8000000000000001
    assert imports[0].shm._mmap is None


@pytest.mark.parametrize("reason", ["device", "grant", "strided"])
def test_whole_call_rejection_precedes_any_import(chip, backing, monkeypatch, reason):
    worker, handle = chip
    args = args_for(backing, scalar=False)
    if reason == "device":
        buffer = wrap_device_malloc(1, 8, mint_owner_instance_id(), 2)
        args.add_tensor(Tensor(buffer, shapes=(2,), dtype=DataType.FLOAT32))
    elif reason == "grant":
        backing.access = AccessMode.READ
        args.add_tensor(Tensor(backing, shapes=(1,), dtype=DataType.FLOAT32))
        args.set_tag(1, TensorArgType.OUTPUT_EXISTING)
    else:
        args = TaskArgs()
        args.add_tensor(Tensor(backing, shapes=(1,), dtype=DataType.FLOAT32))
        args.add_tensor(Tensor(backing, shapes=(2,), strides=(2,), dtype=DataType.FLOAT32))

    def forbidden(*_):
        raise AssertionError("invalid call reached import")

    monkeypatch.setattr(ImportRegistry, "materialize", forbidden)
    with pytest.raises((ValueError, RuntimeError), match="DEVICE|grant|READ|HOST|transfer|contiguous"):
        worker.run(handle, args)
    assert worker._impl.runs == []


def test_partial_bind_failure_closes_import_and_allows_retry(chip, backing, monkeypatch):
    worker, handle = chip
    args = args_for(backing, scalar=False)
    args.add_tensor(Tensor(backing, shapes=(1,), dtype=DataType.FLOAT32))
    original = ImportRegistry.materialize
    imports = []

    def materialize(registry, descriptor):
        if imports:
            raise RuntimeError("bind failed")
        result = original(registry, descriptor)
        imports.append(result)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(ImportRegistry, "materialize", materialize)
        with pytest.raises(RuntimeError, match="bind failed"):
            worker.run(handle, args)
    assert imports[0].shm._mmap is None
    assert worker._impl.runs == []
    worker.run(handle, args_for(backing))
    assert len(worker._impl.runs) == 1


def test_unproven_run_keeps_import_through_failed_teardown(chip, backing, monkeypatch):
    worker, handle = chip
    original = ImportRegistry.materialize
    imports = []

    def materialize(registry, descriptor):
        result = original(registry, descriptor)
        imports.append(result)
        return result

    monkeypatch.setattr(ImportRegistry, "materialize", materialize)
    worker._impl.fail_run = True
    with pytest.raises(RuntimeError, match="unproven completion"):
        worker.run(handle, args_for(backing))
    assert imports[0].shm._mmap is not None
    worker._impl.fail_run = False
    with pytest.raises(RuntimeError, match="finalize"):
        worker.run(handle, args_for(backing))
    assert len(worker._impl.runs) == 1
    worker._impl.fail_finalize = True
    with pytest.raises(RuntimeError, match="teardown failed"):
        worker.finalize()
    assert imports[0].shm._mmap is not None
    worker._impl.fail_finalize = False
    worker.finalize()
    assert imports[0].shm._mmap is None


def test_scalar_only_call_preserves_exact_bits(chip):
    worker, handle = chip
    args = TaskArgs()
    args.add_scalar(0xFFFFFFFFFFFFFFFF)
    args.add_scalar(0x80000000)
    worker.run(handle, args)
    assert worker._impl.runs[0].tensor_count() == 0
    assert [worker._impl.runs[0].scalar(i) for i in range(2)] == [0xFFFFFFFFFFFFFFFF, 0x80000000]


def test_finalize_waits_until_the_public_run_releases_its_imports(chip, backing, monkeypatch):
    worker, handle = chip
    run_entered = threading.Event()
    finish_run = threading.Event()
    finalize_attempted = threading.Event()
    native_finalized = threading.Event()
    errors = []
    original_run = worker._impl.run
    original_finalize = worker._impl.finalize
    lock = threading.Lock()

    class ObservedLock:
        def __enter__(self):
            if threading.current_thread().name == "finalizer":
                finalize_attempted.set()
            lock.acquire()

        def __exit__(self, *_):
            lock.release()

    def run(*args):
        original_run(*args)
        run_entered.set()
        assert finish_run.wait(5)

    def finalize():
        native_finalized.set()
        original_finalize()

    def checked(fn):
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    monkeypatch.setattr(worker, "_run_lock", ObservedLock())
    monkeypatch.setattr(worker._impl, "run", run)
    monkeypatch.setattr(worker._impl, "finalize", finalize)
    runner = threading.Thread(target=checked, args=(lambda: worker.run(handle, args_for(backing)),))
    finalizer = threading.Thread(target=checked, args=(worker.finalize,), name="finalizer")
    runner.start()
    try:
        assert run_entered.wait(5)
        finalizer.start()
        assert finalize_attempted.wait(5)
        assert not native_finalized.is_set()
        assert worker._argument_imports.require(backing.identity).shm._mmap is not None
    finally:
        finish_run.set()
        runner.join(5)
        if finalizer.ident is not None:
            finalizer.join(5)
    assert not runner.is_alive() and not finalizer.is_alive()
    assert not errors
    assert native_finalized.is_set()
    assert worker._argument_imports is None


def test_invalid_config_does_not_retain_imports(chip, backing):
    worker, handle = chip
    with pytest.raises(AttributeError):
        worker.run(handle, args_for(backing), no_such_config_field=1)
    assert worker._argument_imports is None
    assert worker._impl.runs == []
    worker.run(handle, args_for(backing))
    assert len(worker._impl.runs) == 1


def test_host_none_request_survives_public_binding(chip, backing):
    worker, handle = chip
    args = TaskArgs()
    args.add_tensor(Tensor(backing, shapes=(2,), dtype=DataType.FLOAT32), transfer=TensorTransfer.NONE)
    worker.run(handle, args)
    assert worker._impl.runs[0].transfer(0) == TensorTransfer.NONE


@pytest.mark.parametrize("direction", [ArgDirection.OUT, ArgDirection.INOUT])
def test_host_control_grant_checked_against_callable_before_import(chip, backing, monkeypatch, direction):
    worker, _ = chip
    handle = worker.register_callable(
        ChipCallable.build(signature=[direction], func_name="write", binary=b"y", children=[])
    )
    backing.access = AccessMode.READ
    args = TaskArgs()
    args.add_tensor(
        Tensor(backing, shapes=(2,), dtype=DataType.FLOAT32), TensorArgType.INPUT, transfer=TensorTransfer.NONE
    )

    def forbidden(*_):
        raise AssertionError("readonly host control reached import")

    monkeypatch.setattr(ImportRegistry, "materialize_args", forbidden)
    with pytest.raises(ValueError, match="HOST/NONE.*grant"):
        worker.run(handle, args)
    assert worker._impl.runs == []
    assert worker._argument_imports is None
