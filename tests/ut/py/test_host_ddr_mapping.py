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
from simpler.buffer import AccessMode, re_export
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
    if worker._chip_import_registry is not None and worker._chip_import_registry.host_mappings is not None:
        for run_id in tuple(worker._chip_import_registry.host_mappings._uses):
            worker._chip_import_registry.host_mappings.release(run_id)
    if worker._chip_worker is not None:
        worker._chip_worker._unmap_host_ddr.side_effect = None
    worker._release_all_buffers()
    worker._chip_worker = None
    worker.close()


def input_args(buffer):
    args = TaskArgs()
    args.add_tensor(buffer._test_view, transfer=TensorTransfer.MAP)
    return args


def mapped_table(worker):
    table = worker.create_buffer(64)
    table._test_view = table.tensor((4,), DataType.INT32, byte_offset=16)
    registry = worker._l2_import_registry().host_mappings
    registry.acquire(1000, input_args(table))
    registry.release(1000)
    return table


def materialize(worker, args):
    registry = worker._l2_import_registry().host_mappings
    registry.acquire(1001, args)
    try:
        return worker._materialize_l2_args(args)
    finally:
        registry.release(1001)


def test_native_and_mapped_arguments_resolve_independently(mapped_worker):
    table = mapped_table(mapped_worker)
    assert table.access == AccessMode.READWRITE
    native = table.tensor((4,), DataType.INT32, byte_offset=16)
    mapped = table._test_view
    assert native.buffer == mapped.buffer
    assert native == mapped
    args = TaskArgs()
    args.add_tensor(native, transfer=TensorTransfer.NONE)
    args.add_tensor(mapped, transfer=TensorTransfer.MAP)
    args.add_tensor(native, transfer=TensorTransfer.H2D)
    chip_args = materialize(mapped_worker, args)
    assert chip_args.tensor(0).data != 0x40000010
    assert chip_args.tensor(1).data == 0x40000010
    assert chip_args.tensor(0).data == chip_args.tensor(2).data
    assert not chip_args.tensor(0).child_memory
    assert chip_args.tensor(1).child_memory
    materialize(mapped_worker, args)
    mapped_worker._chip_worker._map_host_ddr.assert_called_once_with(table.base, 64)
    descriptor = table.to_descriptor()
    assert _decode_buffer_descriptor_wire(_encode_buffer_descriptor_wire(descriptor)) == descriptor
    assert re_export(descriptor).to_descriptor() == descriptor


def test_mapping_reuse_across_calls_and_views(mapped_worker):
    table = mapped_table(mapped_worker)
    args = TaskArgs()
    args.add_tensor(table.tensor((8,), DataType.INT32), transfer=TensorTransfer.MAP)
    assert materialize(mapped_worker, args).tensor(0).data == 0x40000000
    assert materialize(mapped_worker, input_args(table)).tensor(0).data == 0x40000010
    mapped_worker._chip_worker._map_host_ddr.assert_called_once_with(table.base, 64)
    mapped_worker._chip_worker._unmap_host_ddr.assert_not_called()
    mapped_worker.release_buffer(table)
    mapped_worker._chip_worker._unmap_host_ddr.assert_called_once()


def test_failed_unmap_closes_admission_but_keeps_cleanup_retry(mapped_worker):
    table = mapped_table(mapped_worker)
    registry = mapped_worker._chip_import_registry.host_mappings
    args = input_args(table)
    chip = mapped_worker._chip_worker
    chip._unmap_host_ddr.side_effect = RuntimeError("unregister failed")
    with pytest.raises(RuntimeError, match="unregister failed"):
        mapped_worker.release_buffer(table)
    with pytest.raises(ValueError, match="closing"):
        registry.acquire(1, args)
    assert not table.closed
    chip._unmap_host_ddr.side_effect = None
    mapped_worker.release_buffer(table)
    assert table.closed


def test_map_does_not_add_view_lifecycle_api(mapped_worker):
    assert not hasattr(mapped_worker, "map_view")
    assert not hasattr(mapped_worker, "unmap_view")
    tensor = mapped_worker.create_buffer(64).tensor((4,), DataType.INT32)
    assert not hasattr(tensor, "_mapping_ref")
    mapped_worker._chip_worker._map_host_ddr.assert_not_called()


def test_materialization_requires_preparation(mapped_worker):
    table = mapped_worker.create_buffer(64)
    args = TaskArgs()
    args.add_tensor(table.tensor((4,), DataType.INT32), transfer=TensorTransfer.MAP)
    with pytest.raises(ValueError, match="prepared|owning"):
        mapped_worker._materialize_l2_args(args)
    mapped_worker._chip_worker._map_host_ddr.assert_not_called()


def test_host_export_pins_backing_until_released(mapped_worker):
    table = mapped_worker.create_buffer(64)
    view = table.host_view()
    view[:4] = b"data"
    with pytest.raises(BufferError):
        table.close()
    assert bytes(view[:4]) == b"data"
    view.release()
    table.close()


@pytest.mark.parametrize(
    "tag", [TensorArgType.OUTPUT, TensorArgType.OUTPUT_EXISTING, TensorArgType.INOUT, TensorArgType.NO_DEP]
)
def test_mapped_table_supports_normal_dependency_tags(mapped_worker, tag):
    buffer = mapped_table(mapped_worker)
    args = TaskArgs()
    args.add_tensor(buffer._test_view, tag, transfer=TensorTransfer.MAP)
    assert args.tag(0) == tag
    assert args.transfer(0) == TensorTransfer.MAP
    assert materialize(mapped_worker, args).tensor(0).data == 0x40000010


@pytest.mark.parametrize("direction", [None, ArgDirection.SCALAR])
def test_requires_explicit_tensor_direction(mapped_worker, direction):
    buffer = mapped_table(mapped_worker)
    target = SimpleNamespace(sig_count=1, sig=lambda _: direction) if direction is not None else None
    with pytest.raises(ValueError, match="grant does not cover"):
        _validate_host_control_access(target, input_args(buffer))
    _validate_host_control_access(SimpleNamespace(sig_count=1, sig=lambda _: ArgDirection.IN), input_args(buffer))


@pytest.mark.parametrize("access", [AccessMode.READ, AccessMode.WRITE, AccessMode.READWRITE])
@pytest.mark.parametrize("direction", [ArgDirection.IN, ArgDirection.OUT, ArgDirection.INOUT])
def test_mapping_preserves_source_grant_for_callable_direction(mapped_worker, access, direction):
    buffer = mapped_worker.create_buffer(64)
    buffer.access = access
    native = buffer.tensor((4,), DataType.INT32)
    mapped = native
    assert mapped.buffer == native.buffer
    args = TaskArgs()
    # NO_DEP must not bypass the callable's actual access requirements.
    args.add_tensor(mapped, TensorArgType.NO_DEP, transfer=TensorTransfer.MAP)
    target = SimpleNamespace(sig_count=1, sig=lambda _: direction)
    needed = {
        ArgDirection.IN: AccessMode.READ,
        ArgDirection.OUT: AccessMode.WRITE,
        ArgDirection.INOUT: AccessMode.READWRITE,
    }[direction]
    if access in (needed, AccessMode.READWRITE):
        _validate_host_control_access(target, args)
        assert materialize(mapped_worker, args).tensor(0).data == 0x40000000
    else:
        with pytest.raises(ValueError, match="grant does not cover"):
            _validate_host_control_access(target, args)


@pytest.mark.parametrize("tag", [TensorArgType.OUTPUT_EXISTING, TensorArgType.INOUT])
def test_mapped_writer_retains_backing_until_completion(mapped_worker, tag):
    buffer = mapped_table(mapped_worker)
    args = TaskArgs()
    args.add_tensor(buffer._test_view, tag, transfer=TensorTransfer.MAP)
    registry = mapped_worker._chip_import_registry.host_mappings
    registry.acquire(1, args)
    for release in (buffer.close, lambda: mapped_worker.release_buffer(buffer)):
        with pytest.raises(RuntimeError, match="in-flight"):
            release()
    registry.release(1)
    buffer.close()


def test_same_tensor_switches_map_h2d_and_native_per_call(mapped_worker):
    table = mapped_table(mapped_worker)
    native = table._test_view
    for transfer in (TensorTransfer.MAP, TensorTransfer.H2D, TensorTransfer.NONE, TensorTransfer.MAP):
        args = TaskArgs()
        args.add_tensor(native, transfer=transfer)
        actual = materialize(mapped_worker, args)
        assert args.transfer(0) == transfer
        if transfer == TensorTransfer.MAP:
            assert actual.tensor(0).data == 0x40000010
            assert actual.transfer(0) == TensorTransfer.NONE
        else:
            assert actual.tensor(0).data != 0x40000010
            assert not actual.tensor(0).child_memory
            assert actual.transfer(0) == transfer
    mapped_worker._chip_worker._map_host_ddr.assert_called_once()


def test_direct_close_and_worker_release_refuse_reserved_use(mapped_worker):
    buffer = mapped_table(mapped_worker)
    source = mapped_worker._chip_import_registry.host_mappings
    source.acquire(1, input_args(buffer))
    try:
        for release in (
            buffer.close,
            buffer.shm.close,
            buffer.shm.unlink,
            lambda: mapped_worker.release_buffer(buffer),
        ):
            with pytest.raises(RuntimeError, match="in-flight|registered"):
                release()
        assert not buffer.closed
        mapped_worker._chip_worker._unmap_host_ddr.assert_not_called()
    finally:
        source.release(1)
    mapped_worker.release_buffer(buffer)
    mapped_worker._chip_worker._unmap_host_ddr.assert_called_once()
    assert buffer.closed


def test_descriptor_mutation_and_wrong_context_rejected(mapped_worker):
    buffer = mapped_table(mapped_worker)
    original = input_args(buffer)
    buffer.nbytes = 128
    with pytest.raises(ValueError, match="descriptor"):
        mapped_worker._chip_import_registry.host_mappings.acquire(1, input_args(buffer))
    buffer.nbytes = 64
    other = Worker(level=2)
    try:
        with pytest.raises(ValueError, match="owning L2"):
            other._materialize_l2_args(original)
    finally:
        other.close()


def test_mapping_failure_retains_ordinary_backing(mapped_worker):
    mapped_worker._chip_worker._map_host_ddr.side_effect = RuntimeError("mapping refused")
    with pytest.raises(RuntimeError, match="mapping refused"):
        mapped_table(mapped_worker)
    buffer = next(iter(mapped_worker._buffers.values()))
    mapped_worker.release_buffer(buffer)
    mapped_worker._chip_worker._unmap_host_ddr.assert_called_once()


def test_failed_mapping_and_failed_rollback_keep_backing_owned(mapped_worker):
    chip = mapped_worker._chip_worker
    chip._map_host_ddr.side_effect = RuntimeError("mapping failed after registration")
    chip._unmap_host_ddr.side_effect = RuntimeError("rollback unregister failed")
    with pytest.raises(RuntimeError, match="mapping failed after registration"):
        mapped_table(mapped_worker)
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
    buffer = mapped_table(mapped_worker)
    mapped_worker._chip_import_registry.host_mappings.acquire(1, input_args(buffer))
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
    buffer = mapped_table(mapped_worker)
    mapped_worker.close()
    assert buffer.closed
    chip._unmap_host_ddr.assert_called_once()
    chip.finalize.assert_called_once()


def test_unregister_failure_retains_owner_for_retry(mapped_worker):
    from simpler.host_ddr import _live_registries  # noqa: PLC0415

    buffer = mapped_table(mapped_worker)
    mapped_worker._chip_worker._unmap_host_ddr.side_effect = RuntimeError("unregister failed")
    with pytest.raises(RuntimeError, match="unregister failed"):
        mapped_worker.release_buffer(buffer)
    assert not buffer.closed
    assert buffer.shm is not None
    assert buffer.identity.buffer_id in mapped_worker._buffers
    assert mapped_worker._chip_import_registry.host_mappings in _live_registries
    mapped_worker._chip_worker._unmap_host_ddr.side_effect = None
    mapped_worker.release_buffer(buffer)
    assert not mapped_worker._buffers
    assert mapped_worker._chip_import_registry.host_mappings not in _live_registries


@pytest.mark.parametrize("table_count", [1, 2])
def test_interrupt_after_host_acquire_rolls_back_all_tables(mapped_worker, monkeypatch, table_count):
    from simpler.task_interface import CallConfig  # noqa: PLC0415

    tables = [mapped_table(mapped_worker) for _ in range(table_count)]
    args = TaskArgs()
    for table in tables:
        args.add_tensor(table._test_view, transfer=TensorTransfer.MAP)
    chip = mapped_worker._chip_worker
    chip._execution_environment = None
    chip._submit_chip_run_direct = Mock()
    mapped_worker._callable_registry[1] = SimpleNamespace(sig_count=table_count, sig=lambda _: ArgDirection.IN)
    source = mapped_worker._chip_import_registry.host_mappings
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

    table = mapped_table(mapped_worker)
    source = mapped_worker._chip_import_registry.host_mappings
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
    first = mapped_table(mapped_worker)
    second = mapped_table(mapped_worker)
    source = mapped_worker._chip_import_registry.host_mappings
    source.acquire(1, input_args(first))
    view = second._test_view

    def tensor(index):
        if index == 1:
            raise KeyboardInterrupt("second table selection")
        return view

    with pytest.raises(KeyboardInterrupt, match="second table"):
        source.acquire(2, SimpleNamespace(tensor_count=lambda: 2, tensor=tensor, transfer=lambda _: TensorTransfer.MAP))
    source.release(2)
    mapped_worker.release_buffer(second)
    with pytest.raises(RuntimeError, match="in-flight"):
        mapped_worker.release_buffer(first)
    source.release(1)
    mapped_worker.release_buffer(first)


@pytest.mark.parametrize("refused", [False, True])
def test_native_failure_only_retires_proven_host_uses(mapped_worker, refused):
    from simpler.task_interface import CallConfig, RunPreparationRefused  # noqa: PLC0415

    table = mapped_table(mapped_worker)
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
    table = mapped_table(mapped_worker)
    source = mapped_worker._chip_import_registry.host_mappings
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


def test_native_host_sharing_survives_device_mapping(mapped_worker):
    import os  # noqa: PLC0415

    from simpler.buffer import ImportContext, ImportRegistry  # noqa: PLC0415
    from simpler.comm_endpoints import HOST_CPU  # noqa: PLC0415

    table = mapped_table(mapped_worker)
    with table.host_view() as view:
        view[:4] = b"host"
    descriptor = table.to_descriptor()
    pid = os.fork()
    if pid == 0:
        registry = ImportRegistry(ImportContext(deployment=HOST_CPU))
        try:
            imported = registry.materialize(descriptor)
            assert imported.shm is not None and imported.shm.buf is not None
            assert bytes(imported.shm.buf[:4]) == b"host"
            imported.shm.buf[:4] = b"peer"
            registry.close()
        except BaseException:
            os._exit(1)
        os._exit(0)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    with table.host_view() as view:
        assert bytes(view[:4]) == b"peer"
    assert materialize(mapped_worker, input_args(table)).tensor(0).data == 0x40000010


def test_interrupted_mapping_publication_keeps_cleanup_owner(mapped_worker, monkeypatch):
    import simpler.host_ddr as mappings  # noqa: PLC0415

    class InterruptedRetention(set):
        def add(self, value):
            raise KeyboardInterrupt("mapping publication")

    monkeypatch.setattr(mappings, "_live_registries", InterruptedRetention())
    table = mapped_worker.create_buffer(64)
    with pytest.raises(KeyboardInterrupt, match="mapping publication"):
        args = TaskArgs()
        args.add_tensor(table.tensor((4,), DataType.INT32), transfer=TensorTransfer.MAP)
        mapped_worker._l2_import_registry().host_mappings.acquire(1, args)
    mapped_worker._chip_worker._map_host_ddr.assert_not_called()
    mapped_worker.release_buffer(table)
    assert table.closed


@pytest.mark.parametrize("transfer", [TensorTransfer.NONE, TensorTransfer.H2D])
def test_native_reader_also_retains_registered_backing(mapped_worker, transfer):
    table = mapped_table(mapped_worker)
    registry = mapped_worker._chip_import_registry.host_mappings
    args = TaskArgs()
    args.add_tensor(table.tensor((4,), DataType.INT32), transfer=transfer)
    registry.acquire(1, args)
    with pytest.raises(RuntimeError, match="in-flight"):
        table.close()
    with pytest.raises(RuntimeError, match="in-flight"):
        mapped_worker.release_buffer(table)
    registry.release(1)
    mapped_worker.release_buffer(table)
    assert table.closed


def test_raw_shm_guard_uses_registered_identity(mapped_worker):
    table = mapped_table(mapped_worker)
    original = table.identity
    other = mapped_worker.create_buffer(64)
    table.identity = other.identity
    try:
        with pytest.raises(RuntimeError, match="registered"):
            table.shm.close()
    finally:
        table.identity = original


def test_host_view_obeys_native_grant(mapped_worker):
    table = mapped_worker.create_buffer(64)
    table.access = AccessMode.READ
    with table.host_view() as view:
        assert view.readonly
        with pytest.raises(TypeError):
            view[0] = 1
    table.access = AccessMode.WRITE
    with pytest.raises(ValueError, match="readable Host grant"):
        table.host_view()


@pytest.mark.parametrize("field", ["nbytes", "base"])
def test_mapping_validates_actual_owned_host_storage(mapped_worker, field):
    table = mapped_worker.create_buffer(64)
    assert table.shm is not None
    # Shared memory can be larger than requested on platforms that round to pages.
    value = table.shm.size + 1 if field == "nbytes" else 0x1000
    setattr(table, field, value)
    with pytest.raises(ValueError, match="Host descriptor|Host base"):
        args = TaskArgs()
        args.add_tensor(table.tensor((4,), DataType.INT32), transfer=TensorTransfer.MAP)
        mapped_worker._l2_import_registry().host_mappings.acquire(1, args)
    mapped_worker._chip_worker._map_host_ddr.assert_not_called()


def test_owner_grant_mutation_does_not_authorize_old_mapping(mapped_worker):
    table = mapped_table(mapped_worker)
    table.access = AccessMode.WRITE
    with pytest.raises(ValueError, match="descriptor differs"):
        materialize(mapped_worker, input_args(table))


@pytest.mark.parametrize("same_buffer", [False, True])
def test_concurrent_first_prepare_uses_one_registry_and_registration_per_buffer(
    mapped_worker, same_buffer, monkeypatch
):
    import threading  # noqa: PLC0415
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    import simpler.worker as worker_module  # noqa: PLC0415

    first = mapped_worker.create_buffer(64)
    second = first if same_buffer else mapped_worker.create_buffer(64)
    start = threading.Barrier(2)
    constructing = threading.Event()
    contended = threading.Event()
    proceed = threading.Event()
    lock = mapped_worker._registry_lock
    constructor = worker_module.ImportRegistry
    constructed = []

    class ObservedLock:
        def __enter__(self):
            if not lock.acquire(blocking=False):
                contended.set()
                lock.acquire()
            return self

        def __exit__(self, *_):
            lock.release()

    def construct(*args, **kwargs):
        constructed.append(1)
        constructing.set()
        assert proceed.wait(5)
        return constructor(*args, **kwargs)

    monkeypatch.setattr(mapped_worker, "_registry_lock", ObservedLock())
    monkeypatch.setattr(worker_module, "ImportRegistry", construct)

    def prepare(run_id, table):
        args = TaskArgs()
        args.add_tensor(table.tensor((4,), DataType.INT32), transfer=TensorTransfer.MAP)
        start.wait(timeout=5)
        registry = mapped_worker._l2_import_registry().host_mappings
        registry.acquire(run_id, args)
        assert mapped_worker._materialize_l2_args(args).tensor(0).data == 0x40000000
        return registry

    with ThreadPoolExecutor(max_workers=2) as pool:
        runs = [pool.submit(prepare, run_id, table) for run_id, table in ((1, first), (2, second))]
        try:
            assert constructing.wait(5)
            assert contended.wait(5), "concurrent first preparation did not share the registry lock"
        finally:
            proceed.set()
        registries = [run.result(timeout=5) for run in runs]
    assert len(constructed) == 1
    assert registries[0] is registries[1]
    assert mapped_worker._chip_worker._map_host_ddr.call_count == (1 if same_buffer else 2)
    assert set(registries[0]._uses) == {1, 2}
    registries[0].release(1)
    registries[0].release(2)


def test_close_racing_first_prepare_waits_for_reservation(mapped_worker):
    import threading  # noqa: PLC0415
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    table = mapped_worker.create_buffer(64)
    args = TaskArgs()
    args.add_tensor(table.tensor((4,), DataType.INT32), transfer=TensorTransfer.MAP)
    registry = mapped_worker._l2_import_registry().host_mappings
    mapping_entered = threading.Event()
    close_contended = threading.Event()
    proceed = threading.Event()
    lock = table._host_lock

    class ObservedLock:
        def __enter__(self):
            if not lock.acquire(blocking=False):
                close_contended.set()
                lock.acquire()
            return self

        def __exit__(self, *_):
            lock.release()

    table._host_lock = ObservedLock()

    def map_host(*_):
        mapping_entered.set()
        assert proceed.wait(5)
        return 0x40000000

    mapped_worker._chip_worker._map_host_ddr.side_effect = map_host
    with ThreadPoolExecutor(max_workers=2) as pool:
        prepared = pool.submit(registry.acquire, 1, args)
        try:
            assert mapping_entered.wait(5)
            closed = pool.submit(table.close)
            assert close_contended.wait(5), "close did not contend with first preparation"
            assert not closed.done()
        finally:
            proceed.set()
        prepared.result(timeout=5)
        with pytest.raises(RuntimeError, match="in-flight"):
            closed.result(timeout=5)
    assert not table.closed
    mapped_worker._chip_worker._unmap_host_ddr.assert_not_called()
    registry.release(1)
    table.close()
    mapped_worker._chip_worker._unmap_host_ddr.assert_called_once()


def test_closed_buffer_cannot_be_prepared_for_first_mapping(mapped_worker):
    table = mapped_worker.create_buffer(64)
    args = TaskArgs()
    args.add_tensor(table.tensor((4,), DataType.INT32), transfer=TensorTransfer.MAP)
    table.close()
    with pytest.raises(ValueError, match="live backing|closed"):
        mapped_worker._l2_import_registry().host_mappings.acquire(1, args)
    mapped_worker._chip_worker._map_host_ddr.assert_not_called()


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_partial_registration_rolls_back_all_new_tables(mapped_worker, cleanup_fails):
    tables = [mapped_worker.create_buffer(64) for _ in range(2)]
    args = TaskArgs()
    for table in tables:
        args.add_tensor(table.tensor((4,), DataType.INT32), transfer=TensorTransfer.MAP)
    registry = mapped_worker._l2_import_registry().host_mappings
    chip = mapped_worker._chip_worker
    chip._map_host_ddr.side_effect = [0x40000000, RuntimeError("second registration failed")]
    if cleanup_fails:
        chip._unmap_host_ddr.side_effect = RuntimeError("unregister failed")
    with pytest.raises(RuntimeError, match="second registration failed"):
        registry.acquire(1, args)
    assert not registry._uses
    assert chip._unmap_host_ddr.call_count == 2
    for table in tables:
        assert not table.closed
        assert table.identity.buffer_id in mapped_worker._buffers
    if cleanup_fails:
        assert len(registry._records) == 2
        with pytest.raises(ValueError, match="closing"):
            registry.acquire(2, args)
        chip._unmap_host_ddr.side_effect = None
    else:
        assert not registry._records
    for table in tables:
        mapped_worker.release_buffer(table)
        assert table.closed


@pytest.mark.parametrize("transfer", [TensorTransfer.NONE, TensorTransfer.H2D])
def test_native_call_retains_owned_host_backing_without_mapping(mapped_worker, transfer):
    table = mapped_worker.create_buffer(64)
    args = TaskArgs()
    args.add_tensor(table.tensor((4,), DataType.INT32), transfer=transfer)
    registry = mapped_worker._l2_import_registry().host_mappings
    registry.acquire(1, args)
    for release in (table.close, table.shm.close, table.shm.unlink):
        with pytest.raises(RuntimeError, match="in-flight|registered"):
            release()
    mapped_worker._chip_worker._map_host_ddr.assert_not_called()
    registry.release(1)
    mapped_worker.release_buffer(table)
    mapped_worker._chip_worker._unmap_host_ddr.assert_not_called()


@pytest.mark.parametrize("direction", [ArgDirection.IN, ArgDirection.OUT, ArgDirection.INOUT])
@pytest.mark.parametrize("overlap", [False, True])
def test_mixed_h2d_map_aliases_reject_overlapping_writers(mapped_worker, direction, overlap):
    table = mapped_worker.create_buffer(64)
    args = TaskArgs()
    args.add_tensor(table.tensor((4,), DataType.INT32), TensorArgType.NO_DEP, transfer=TensorTransfer.MAP)
    args.add_tensor(table.tensor((4,), DataType.INT32, byte_offset=8 if overlap else 16), transfer=TensorTransfer.H2D)
    target = SimpleNamespace(sig_count=2, sig=lambda i: direction if i == 0 else ArgDirection.IN)
    if overlap and direction != ArgDirection.IN:
        with pytest.raises(ValueError, match="overlapping H2D/MAP"):
            mapped_worker._validate_map_aliases(target, args)
    else:
        mapped_worker._validate_map_aliases(target, args)
    mapped_worker._chip_worker._map_host_ddr.assert_not_called()


@pytest.mark.parametrize("export", ["ndarray", "slice", "frombuffer"])
def test_numpy_export_from_temporary_host_view_pins_mapping(mapped_worker, export):
    import gc  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    table = mapped_table(mapped_worker)
    if export == "frombuffer":
        view = np.frombuffer(table.host_view(), dtype=np.uint8)
    else:
        view = np.ndarray((64,), dtype=np.uint8, buffer=table.host_view())
        if export == "slice":
            view = view[8:16]
    view[:] = 7
    gc.collect()
    with pytest.raises(BufferError):
        mapped_worker.release_buffer(table)
    assert not table.closed
    np.testing.assert_array_equal(view, 7)
    mapped_worker._chip_worker._unmap_host_ddr.assert_not_called()
    del view
    gc.collect()
    mapped_worker.release_buffer(table)
    mapped_worker._chip_worker._unmap_host_ddr.assert_called_once()


@pytest.mark.parametrize("transfer", [TensorTransfer.NONE, TensorTransfer.H2D])
def test_terminal_native_close_retires_host_only_uses(mapped_worker, transfer):
    chip = mapped_worker._chip_worker
    chip.workspace_report = lambda: ("disabled", {})
    chip._close_chip_run_lane = lambda: None
    table = mapped_worker.create_buffer(64)
    args = TaskArgs()
    args.add_tensor(table.tensor((4,), DataType.INT32), transfer=transfer)
    mappings = mapped_worker._l2_import_registry().host_mappings
    mappings.acquire(1, args)
    mapped_worker._ordered_cleanup_error = RuntimeError("abandoned native Host run")

    def finalize():
        assert mappings._uses
        assert not table.closed

    chip.finalize = finalize
    mapped_worker.close()
    assert not mappings._uses
    assert table.closed
    chip._unmap_host_ddr.assert_not_called()
