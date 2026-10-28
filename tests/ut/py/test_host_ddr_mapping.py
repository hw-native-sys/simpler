# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host policy, source provenance and lifetime checks without an NPU."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from _task_interface import _decode_buffer_descriptor_wire, _encode_buffer_descriptor_wire
from simpler.buffer import AccessMode, AddressSpace, re_export
from simpler.task_interface import (
    ArgDirection,
    DataType,
    TaskArgs,
    TensorArgType,
    TensorTransfer,
    _validate_host_control_access,
)
from simpler.worker import RunHandle, Worker, _Lifecycle


@pytest.fixture
def mapped_worker():
    worker = Worker(level=2, platform="a5", runtime="tensormap_and_ringbuffer")
    worker._lifecycle = _Lifecycle.READY
    worker._chip_worker = SimpleNamespace(
        _validate_execution_context=lambda: None,
        _host_ddr_mapping_supported=True,
        _map_host_ddr=Mock(return_value=0x40000000),
        _unmap_host_ddr=Mock(),
    )
    yield worker
    worker._chip_runs.clear()
    worker._chip_run_touched_identities.clear()
    if worker._host_ddr_source is not None:
        for run_id in tuple(worker._host_ddr_source._uses):
            worker._host_ddr_source.release(run_id)
    if worker._chip_worker is not None:
        worker._chip_worker._unmap_host_ddr.side_effect = None
    worker._release_all_buffers()
    worker._chip_worker = None
    worker.close()


def input_args(buffer):
    args = TaskArgs()
    args.add_tensor(buffer.tensor((4,), DataType.INT32, byte_offset=16))
    return args


def test_host_policy_survives_views_and_wire_without_staging(mapped_worker):
    buffer = mapped_worker.create_buffer(64, device_readable=True)
    assert buffer.address_space == AddressSpace.HOST
    assert buffer.access == AccessMode.READ
    assert buffer.device_readable
    with pytest.raises(AttributeError):
        buffer.device_readable = False
    args = input_args(buffer)
    assert args.tensor(0).buffer.device_readable
    assert args.transfer(0) == TensorTransfer.NONE
    descriptor = buffer.to_descriptor()
    assert _decode_buffer_descriptor_wire(_encode_buffer_descriptor_wire(descriptor)) == descriptor
    chip_args = mapped_worker._materialize_l2_args(args)
    assert chip_args.tensor(0).data == 0x40000010
    assert chip_args.tensor(0).child_memory
    mapped_worker._materialize_l2_args(args)
    mapped_worker._chip_worker._map_host_ddr.assert_called_once_with(buffer.base, 64)
    assert mapped_worker._chip_import_registry._by_identity == {}


@pytest.mark.parametrize("wire_roundtrip", [False, True])
def test_re_export_rejects_readable_table_without_downgrading_policy(mapped_worker, wire_roundtrip):
    table = mapped_worker.create_buffer(64, device_readable=True)
    descriptor = table.to_descriptor()
    if wire_roundtrip:
        descriptor = _decode_buffer_descriptor_wire(_encode_buffer_descriptor_wire(descriptor))
    with pytest.raises(ValueError, match="device_readable.*re-export"):
        re_export(descriptor)
    assert table.device_readable
    args = input_args(table)
    assert args.transfer(0) == TensorTransfer.NONE
    assert mapped_worker._materialize_l2_args(args).tensor(0).data == 0x40000010
    other = Worker(level=2)
    try:
        with pytest.raises(ValueError, match="owning L2"):
            other._materialize_l2_args(args)
    finally:
        other.close()


@pytest.mark.parametrize(
    "tag", [TensorArgType.OUTPUT, TensorArgType.OUTPUT_EXISTING, TensorArgType.INOUT, TensorArgType.NO_DEP]
)
def test_readable_table_rejects_other_tags(mapped_worker, tag):
    buffer = mapped_worker.create_buffer(64, device_readable=True)
    with pytest.raises(ValueError):
        TaskArgs().add_tensor(buffer.tensor((4,), DataType.INT32), tag)


@pytest.mark.parametrize("direction", [None, ArgDirection.OUT, ArgDirection.INOUT, ArgDirection.SCALAR])
def test_requires_explicit_in_direction(mapped_worker, direction):
    buffer = mapped_worker.create_buffer(64, device_readable=True)
    target = SimpleNamespace(sig_count=1, sig=lambda _: direction) if direction is not None else None
    with pytest.raises(ValueError, match="explicit IN"):
        _validate_host_control_access(target, input_args(buffer))
    _validate_host_control_access(SimpleNamespace(sig_count=1, sig=lambda _: ArgDirection.IN), input_args(buffer))


def test_explicit_h2d_cannot_override_buffer_policy(mapped_worker):
    buffer = mapped_worker.create_buffer(64, device_readable=True)
    with pytest.raises(ValueError, match="no staging"):
        TaskArgs().add_tensor(buffer.tensor((4,), DataType.INT32), transfer=TensorTransfer.H2D)
    with pytest.raises(ValueError):
        buffer.tensor((20,), DataType.INT32)


def test_direct_close_and_worker_release_refuse_reserved_use(mapped_worker):
    buffer = mapped_worker.create_buffer(64, device_readable=True)
    source = mapped_worker._host_ddr_source
    source.acquire(1, input_args(buffer))
    try:
        for release in (
            buffer.close,
            buffer.shm.close,
            buffer.shm.unlink,
            lambda: mapped_worker.release_buffer(buffer),
        ):
            with pytest.raises(RuntimeError, match="in-flight"):
                release()
        assert not buffer.closed
        mapped_worker._chip_worker._unmap_host_ddr.assert_not_called()
    finally:
        source.release(1)
    mapped_worker.release_buffer(buffer)
    mapped_worker._chip_worker._unmap_host_ddr.assert_called_once()
    assert buffer.closed


def test_descriptor_mutation_and_wrong_context_rejected(mapped_worker):
    buffer = mapped_worker.create_buffer(64, device_readable=True)
    original = input_args(buffer)
    buffer.nbytes = 128
    with pytest.raises(ValueError, match="descriptor"):
        buffer.to_descriptor()
    buffer.nbytes = 64
    other = Worker(level=2)
    try:
        with pytest.raises(ValueError, match="owning L2"):
            other._materialize_l2_args(original)
    finally:
        other.close()


def test_mapping_failure_rolls_back_backing(mapped_worker):
    mapped_worker._chip_worker._map_host_ddr.side_effect = RuntimeError("mapping refused")
    with pytest.raises(RuntimeError, match="mapping refused"):
        mapped_worker.create_buffer(64, device_readable=True)
    assert not mapped_worker._buffers
    mapped_worker._chip_worker._unmap_host_ddr.assert_called_once()


def test_failed_mapping_and_failed_rollback_keep_backing_owned(mapped_worker):
    chip = mapped_worker._chip_worker
    chip._map_host_ddr.side_effect = RuntimeError("mapping failed after registration")
    chip._unmap_host_ddr.side_effect = RuntimeError("rollback unregister failed")
    with pytest.raises(RuntimeError, match="rollback unregister failed"):
        mapped_worker.create_buffer(64, device_readable=True)
    assert len(mapped_worker._buffers) == 1
    buffer = next(iter(mapped_worker._buffers.values()))
    assert not buffer.closed
    assert buffer.shm is not None
    chip._unmap_host_ddr.side_effect = None
    mapped_worker.release_buffer(buffer)


def test_close_drains_unawaited_use_before_unmapping(mapped_worker):
    chip = mapped_worker._chip_worker
    order = []
    chip.workspace_report = lambda: ("disabled", {})
    chip._close_chip_run_lane = lambda: order.append("drain")
    chip.finalize = lambda: order.append("finalize")
    chip._unmap_host_ddr.side_effect = lambda _: order.append("unmap")
    buffer = mapped_worker.create_buffer(64, device_readable=True)
    mapped_worker._host_ddr_source.acquire(1, input_args(buffer))
    mapped_worker._chip_runs[1] = SimpleNamespace(wait=lambda _: order.append("wait") or True)
    handle = RunHandle(mapped_worker, 1, ())
    mapped_worker._accepted_run_handles.add(handle)
    mapped_worker.close()
    assert order.index("wait") < order.index("unmap") < order.index("finalize")
    assert order.count("wait") == 1
    assert handle.done
    assert buffer.closed


def test_reported_lane_failure_does_not_retain_an_unused_host_mapping(mapped_worker):
    chip = mapped_worker._chip_worker
    chip.workspace_report = lambda: ("disabled", {})
    chip._close_chip_run_lane = Mock(side_effect=RuntimeError("already reported failure"))
    chip.finalize = Mock()
    buffer = mapped_worker.create_buffer(64, device_readable=True)
    mapped_worker.close()
    assert buffer.closed
    chip._unmap_host_ddr.assert_called_once()
    chip.finalize.assert_called_once()


def test_unregister_failure_retains_owner_for_retry(mapped_worker):
    from simpler.host_ddr import _live_sources  # noqa: PLC0415

    buffer = mapped_worker.create_buffer(64, device_readable=True)
    mapped_worker._chip_worker._unmap_host_ddr.side_effect = RuntimeError("unregister failed")
    with pytest.raises(RuntimeError, match="unregister failed"):
        mapped_worker.release_buffer(buffer)
    assert not buffer.closed
    assert buffer.shm is not None
    assert buffer.identity.buffer_id in mapped_worker._buffers
    assert mapped_worker._host_ddr_source in _live_sources
    mapped_worker._chip_worker._unmap_host_ddr.side_effect = None
    mapped_worker.release_buffer(buffer)
    assert not mapped_worker._buffers
    assert mapped_worker._host_ddr_source not in _live_sources


@pytest.mark.parametrize("table_count", [1, 2])
def test_interrupt_after_host_acquire_rolls_back_all_tables(mapped_worker, monkeypatch, table_count):
    from simpler.task_interface import CallConfig  # noqa: PLC0415

    tables = [mapped_worker.create_buffer(64, device_readable=True) for _ in range(table_count)]
    args = TaskArgs()
    for table in tables:
        args.add_tensor(table.tensor((4,), DataType.INT32))
    chip = mapped_worker._chip_worker
    chip._execution_environment = None
    chip._submit_chip_run_direct = Mock()
    mapped_worker._callable_registry[1] = SimpleNamespace(sig_count=table_count, sig=lambda _: ArgDirection.IN)
    source = mapped_worker._host_ddr_source
    acquire = source.acquire

    def interrupted(*args):
        acquire(*args)
        raise KeyboardInterrupt("after reserve, before return")

    monkeypatch.setattr(source, "acquire", interrupted)
    with pytest.raises(KeyboardInterrupt, match="after reserve"):
        mapped_worker._submit_l2_locked(1, args, CallConfig())
    chip._submit_chip_run_direct.assert_not_called()
    assert not mapped_worker._chip_run_touched_identities
    for table in tables:
        mapped_worker.release_buffer(table)
        assert table.closed


@pytest.mark.parametrize("method", ["close", "unlink"])
def test_fork_shm_cleanup_rejects_before_inherited_lock(mapped_worker, method):
    import os  # noqa: PLC0415
    import signal  # noqa: PLC0415
    import threading  # noqa: PLC0415

    table = mapped_worker.create_buffer(64, device_readable=True)
    source = mapped_worker._host_ddr_source
    locked = threading.Event()
    unlock = threading.Event()

    def hold_lock():
        with source._lock:
            locked.set()
            unlock.wait()

    thread = threading.Thread(target=hold_lock)
    thread.start()
    assert locked.wait(5)
    try:
        pid = os.fork()
        if pid == 0:
            signal.alarm(3)
            try:
                getattr(table.shm, method)()
            except ValueError as error:
                os._exit(0 if "another process" in str(error) else 2)
            except BaseException:
                os._exit(3)
            os._exit(4)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0, "fork child blocked on the inherited mapping lock"
    finally:
        unlock.set()
        thread.join(5)
    mapped_worker.release_buffer(table)


def test_interrupted_multi_table_selection_keeps_existing_run(mapped_worker):
    first = mapped_worker.create_buffer(64, device_readable=True)
    second = mapped_worker.create_buffer(64, device_readable=True)
    source = mapped_worker._host_ddr_source
    source.acquire(1, input_args(first))
    view = second.tensor((4,), DataType.INT32)

    def tensor(index):
        if index == 1:
            raise KeyboardInterrupt("second table selection")
        return view

    with pytest.raises(KeyboardInterrupt, match="second table"):
        source.acquire(2, SimpleNamespace(tensor_count=lambda: 2, tensor=tensor))
    source.release(2)
    mapped_worker.release_buffer(second)
    with pytest.raises(RuntimeError, match="in-flight"):
        mapped_worker.release_buffer(first)
    source.release(1)
    mapped_worker.release_buffer(first)


@pytest.mark.parametrize("refused", [False, True])
def test_native_failure_only_retires_proven_host_uses(mapped_worker, refused):
    from simpler.task_interface import CallConfig, RunPreparationRefused  # noqa: PLC0415

    table = mapped_worker.create_buffer(64, device_readable=True)
    chip = mapped_worker._chip_worker
    chip._execution_environment = None
    error = RunPreparationRefused("not submitted") if refused else KeyboardInterrupt("native outcome unknown")
    chip._submit_chip_run_direct = Mock(side_effect=error)
    mapped_worker._callable_registry[1] = SimpleNamespace(sig_count=1, sig=lambda _: ArgDirection.IN)
    with pytest.raises(type(error)):
        mapped_worker._submit_l2_locked(1, input_args(table), CallConfig())
    if refused:
        mapped_worker.release_buffer(table)
        assert table.closed
    else:
        assert mapped_worker._source_context_failed
        with pytest.raises(RuntimeError, match="unproven"):
            mapped_worker.release_buffer(table)
        assert not table.closed
        chip._unmap_host_ddr.assert_not_called()


def test_shared_table_stays_reserved_until_every_run_retires(mapped_worker):
    table = mapped_worker.create_buffer(64, device_readable=True)
    source = mapped_worker._host_ddr_source
    args = input_args(table)
    source.acquire(1, args)
    source.acquire(2, args)
    source.release(1)
    source.release(1)
    with pytest.raises(RuntimeError, match="in-flight"):
        mapped_worker.release_buffer(table)
    source.release(2)
    mapped_worker.release_buffer(table)
    assert table.closed


@pytest.mark.parametrize("acquire_first", [True, False])
def test_acquire_and_close_serialize_on_mapping_lifetime(mapped_worker, acquire_first):
    import threading  # noqa: PLC0415
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    table = mapped_worker.create_buffer(64, device_readable=True)
    source = mapped_worker._host_ddr_source
    args = input_args(table)
    view = args.tensor(0)
    entered = threading.Event()
    proceed = threading.Event()
    contended = threading.Event()
    lock = source._lock

    class ObservedLock:
        def __enter__(self):
            if lock.locked():
                contended.set()
            lock.acquire()

        def __exit__(self, *_):
            lock.release()

    source._lock = ObservedLock()

    def pause():
        entered.set()
        assert proceed.wait(5), "test did not release the mapping operation"

    def tensor(_):
        pause()
        return view

    if acquire_first:
        args = SimpleNamespace(tensor_count=lambda: 1, tensor=tensor)
    else:
        mapped_worker._chip_worker._unmap_host_ddr.side_effect = lambda _: pause()

    def acquire():
        source.acquire(1, args)

    first, second = (acquire, table.close) if acquire_first else (table.close, acquire)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_result = pool.submit(first)
        try:
            assert entered.wait(5)
            second_result = pool.submit(second)
            assert contended.wait(5), "second operation did not contend for the mapping lock"
            assert not first_result.done()
            assert not second_result.done()
        finally:
            proceed.set()
        first_result.result(timeout=5)
        if acquire_first:
            with pytest.raises(RuntimeError, match="in-flight"):
                second_result.result(timeout=5)
            assert not table.closed
            mapped_worker._chip_worker._unmap_host_ddr.assert_not_called()
            source.release(1)
            table.close()
        else:
            with pytest.raises(ValueError, match="not owned by this L2"):
                second_result.result(timeout=5)
            assert not source._uses
    assert table.closed
    mapped_worker._chip_worker._unmap_host_ddr.assert_called_once()
