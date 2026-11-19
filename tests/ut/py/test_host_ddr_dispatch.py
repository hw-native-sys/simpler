# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Worker and orchestration boundaries for collective Host mappings, without hardware."""

import itertools
import threading
from multiprocessing.shared_memory import SharedMemory
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
from simpler.buffer import ImportContext, ImportRegistry
from simpler.callable_identity import CallableHandle
from simpler.comm_endpoints import DEVICE_AICPU
from simpler.host_ddr_l3 import ChildHostMappings, _decode
from simpler.orchestrator import Orchestrator
from simpler.task_interface import DataType, TaskArgs, TensorTransfer, _materialize_chip_task_args
from simpler.worker import _COMM_INIT_HEADER, _CTRL_HOST_DDR, Worker, WorkerType, _callback_frame_for, _Lifecycle


def _args(tensor):
    args = TaskArgs()
    args.add_tensor(tensor, transfer=TensorTransfer.MAP)
    return args


@pytest.fixture
def l3(monkeypatch):
    worker = Worker(level=3, device_ids=[0, 1], platform="a5", runtime="tensormap_and_ringbuffer", num_sub_workers=0)
    worker._lifecycle = _Lifecycle.READY
    worker._chip_shms = [None, None]
    events = []
    imports, children, chips = [], [], []
    for rank in range(2):
        registry = ImportRegistry(
            ImportContext(deployment=DEVICE_AICPU, device_owner_instance_id=worker._owner_instance_id)
        )
        chip = SimpleNamespace(
            _host_ddr_mapping_supported=True,
            _quiesce_host_uses=Mock(return_value=False),
            _validate_execution_context=Mock(),
            _map_host_ddr=Mock(return_value=0x1000000 + rank * 0x10000),
            _prepare_host_urma=Mock(return_value=0x2000000 + rank * 0x10000),
            _release_host_urma=Mock(side_effect=lambda r=rank: events.append(("release", r))),
            _unmap_host_ddr=Mock(side_effect=lambda *_args, r=rank: events.append(("unmap", r))),
            finalize=Mock(side_effect=lambda r=rank: events.append(("finalize", r))),
        )
        child = ChildHostMappings(
            chip, registry, owner_instance_id=worker._owner_instance_id, rank=rank, rank_count=2, alive=lambda: True
        )
        registry.host_mappings = child
        imports.append(registry)
        children.append(child)
        chips.append(chip)
    transport = MagicMock()
    native_orch = MagicMock()
    native_orch._begin_run.side_effect = itertools.count(1).__next__
    native_orch._wait_run_for.return_value = True
    worker._worker = transport
    worker._orch = Orchestrator(native_orch, worker)

    def comm_init(rank, request_name):
        request = SharedMemory(name=request_name)
        try:
            memory = request.buf
            assert memory is not None
            assert _COMM_INIT_HEADER.unpack_from(memory) == (rank, 2)
            assert memory[-1] == 0
            events.append(("comm", rank))
        finally:
            request.close()

    def control(worker_type, rank, command, payload, timeout):
        assert worker_type == WorkerType.NEXT_LEVEL
        assert command == _CTRL_HOST_DDR
        assert timeout > 0
        action = _decode(payload)[0]
        events.append((action.name, rank))
        return children[rank].handle(payload)

    def close_transport():
        for registry in imports:
            registry.close()
        events.append(("transport_close", -1))

    transport.control_comm_init.side_effect = comm_init
    transport.control_payload.side_effect = control
    transport.close.side_effect = close_transport
    # No processes/mailboxes are allocated by this fixture; only native and OS
    # transport boundaries are replaced. Python ownership/admission is real.
    monkeypatch.setattr(worker, "_broadcast_child_shutdown", lambda _shms: None)
    monkeypatch.setattr(worker, "_reclaim_child_groups", lambda _deadline: worker._chip_shms.clear())

    def resolve_handle(handle, **_kwargs):
        return SimpleNamespace(
            digest=handle.digest,
            kind=handle.kind,
            target_namespace=handle.target_namespace,
            eligible_worker_ids=(0, 1),
            target=SimpleNamespace(ccu_module_count=0),
        )

    monkeypatch.setattr(worker, "_resolve_handle", resolve_handle)
    dispatches = {}
    local_args = []

    def submit(_digest, _kind, _namespace, args, _config, rank, *_rest):
        frame = _callback_frame_for(worker)
        assert frame is not None
        run_id = frame.run_id
        key = (run_id, len(dispatches.get(run_id, ())))
        children[rank].acquire(key, args)
        local_args.append((rank, _materialize_chip_task_args(args, imports[rank])))
        dispatches.setdefault(run_id, []).append((rank, key))

    def submit_group(digest, kind, namespace, args_list, config, ranks, *_rest):
        for args, rank in zip(args_list, ranks, strict=True):
            submit(digest, kind, namespace, args, config, rank)

    def release_run(run_id):
        for rank, key in dispatches.pop(run_id, ()):
            children[rank].release(key)

    native_orch.submit_next_level.side_effect = submit
    native_orch.submit_next_level_group.side_effect = submit_group
    native_orch._release_run.side_effect = release_run
    table = worker.create_buffer(128)
    state = SimpleNamespace(
        worker=worker,
        table=table,
        native=table.tensor((4,), DataType.INT32, byte_offset=16),
        children=children,
        chips=chips,
        imports=imports,
        transport=transport,
        native_orch=native_orch,
        events=events,
        local_args=local_args,
        handle=CallableHandle("sha256:" + "00" * 32, "CHIP_CALLABLE", "LOCAL_CHIP"),
    )
    yield state
    for chip, child in zip(chips, children, strict=True):
        chip._release_host_urma.side_effect = None
        chip._unmap_host_ddr.side_effect = None
        child._uses.clear()
    worker._accepted_run_handles.clear()
    if worker._host_group_mappings is not None:
        worker._host_group_mappings._uses.clear()
    worker._ordered_cleanup_error = None
    worker.close()
    for registry in imports:
        registry.close()


def dispatch_both(l3):
    def callback(orch, _outer, config):
        orch.submit_next_level_group(l3.handle, [_args(l3.native)] * 2, config, workers=[0, 1])

    return callback


@pytest.mark.parametrize("outer_none", [False, True])
def test_child_map_does_not_require_outer_map(l3, outer_none):
    outer = TaskArgs()
    if outer_none:
        outer.add_tensor(l3.native, transfer=TensorTransfer.NONE)

    def callback(orch, _outer, config):
        assert not any(chip._map_host_ddr.called for chip in l3.chips)
        orch.submit_next_level(l3.handle, _args(l3.native), config, worker=0)

    pending = l3.worker.submit(callback, outer)
    l3.native_orch.submit_next_level.assert_called_once()
    assert l3.chips[0]._map_host_ddr.call_count == 1
    l3.chips[1]._map_host_ddr.assert_not_called()
    assert l3.local_args[0][1].tensor(0).data == 0x1000010
    l3.transport.control_payload.assert_not_called()
    l3.transport.control_comm_init.assert_not_called()
    for chip in l3.chips:
        chip._prepare_host_urma.assert_not_called()
    for release in (l3.table.close, lambda: l3.worker.release_buffer(l3.table)):
        with pytest.raises(RuntimeError):
            release()
    pending.wait()
    l3.worker.release_buffer(l3.table)
    assert l3.table.closed and l3.table.unlinked
    assert [(action, rank) for action, rank in l3.events if action in ("QUIESCE", "RELEASE")] == [
        ("QUIESCE", 0),
        ("RELEASE", 0),
    ]


def test_cached_maps_allow_callback_subviews_and_group_dispatch(l3):
    def callback(orch, args, config):
        view = l3.table.tensor((2,), DataType.INT32, byte_offset=24)
        orch.submit_next_level_group(l3.handle, [_args(view), _args(view)], config, workers=[0, 1])

    for _ in range(2):
        l3.worker.submit(callback, TaskArgs()).wait()
    assert all(chip._map_host_ddr.call_count == 1 for chip in l3.chips)
    l3.transport.control_payload.assert_not_called()
    assert [local.tensor(0).data for _, local in l3.local_args] == [0x1000018, 0x1010018] * 2


def test_new_backing_maps_with_a_previous_run_still_accepted(l3):
    first = l3.worker.submit(dispatch_both(l3), TaskArgs())
    second_table = l3.worker.create_buffer(128)
    second_args = _args(second_table.tensor((4,), DataType.INT32))

    def callback(orch, _outer, config):
        assert not first._cleanup_published
        orch.submit_next_level(l3.handle, second_args, config, worker=0)

    second = l3.worker.submit(callback, TaskArgs())
    assert l3.chips[0]._map_host_ddr.call_count == 2
    assert l3.chips[1]._map_host_ddr.call_count == 1
    l3.transport.control_payload.assert_not_called()
    first.wait()
    second.wait()
    l3.worker.release_buffer(second_table)


def test_failed_mapping_keeps_partial_setup_for_coordinated_cleanup(l3):
    l3.chips[1]._map_host_ddr.side_effect = RuntimeError("registration failed")

    def callback(orch, _outer, config):
        orch.submit_next_level(l3.handle, _args(l3.native), config, worker=0)
        assert l3.children[0]._uses
        orch.submit_next_level(l3.handle, _args(l3.native), config, worker=1)

    with pytest.raises(RuntimeError, match="registration failed"):
        l3.worker.submit(callback, TaskArgs())
    assert not l3.worker._accepted_run_handles
    assert l3.worker._host_group_mappings.holds_resources
    assert not l3.table.closed
    l3.worker.release_buffer(l3.table)
    assert all(chip._unmap_host_ddr.call_count == 1 for chip in l3.chips)


def test_outer_host_buffer_is_retained_without_mapping_or_child_dispatch(l3):
    outer = TaskArgs()
    outer.add_tensor(l3.native, transfer=TensorTransfer.NONE)
    pending = l3.worker.submit(lambda *_args: None, outer)
    assert l3.table.identity in pending._resources.touched_identities
    for release in (l3.table.close, lambda: l3.worker.release_buffer(l3.table)):
        with pytest.raises(RuntimeError):
            release()
    assert not any(chip._map_host_ddr.called for chip in l3.chips)
    pending.wait()
    l3.table.close()
    l3.transport.control_payload.assert_not_called()


def test_completed_calls_keep_registration_until_worker_close(l3):
    l3.worker.submit(dispatch_both(l3), TaskArgs()).wait()
    assert all(not chip._unmap_host_ddr.called for chip in l3.chips)
    l3.worker.close()
    assert l3.table.closed and l3.table.unlinked
    assert l3.events[-1] == ("transport_close", -1)
    assert all(chip._unmap_host_ddr.call_count == 1 for chip in l3.chips)


@pytest.mark.parametrize("transfer", [TensorTransfer.NONE, TensorTransfer.H2D, TensorTransfer.MAP])
def test_admitted_callback_can_submit_host_args_while_close_drains(l3, monkeypatch, transfer):
    entered, closing = threading.Event(), threading.Event()
    errors = []
    worker = l3.worker
    wait = worker._hierarchical_start_cv.wait

    def closing_wait(*args, **kwargs):
        assert worker._lifecycle is _Lifecycle.CLOSED
        closing.set()
        return wait(*args, **kwargs)

    monkeypatch.setattr(worker._hierarchical_start_cv, "wait", closing_wait)

    def callback(orch, _outer, config):
        entered.set()
        assert closing.wait(10)
        args = TaskArgs()
        args.add_tensor(l3.native, transfer=transfer)
        orch.submit_next_level(l3.handle, args, config, worker=0)

    def submit():
        try:
            worker.submit(callback, TaskArgs())
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    thread = threading.Thread(target=submit)
    thread.start()
    try:
        assert entered.wait(10)
        worker.close()
    finally:
        closing.set()
        thread.join(10)
    assert not thread.is_alive()
    assert not errors
    l3.native_orch.submit_next_level.assert_called_once()
    assert l3.table.closed


def test_mapping_manager_refuses_a_replaced_native_context(l3):
    mappings = l3.worker._l3_host_mappings()
    l3.worker._worker = MagicMock()
    try:
        with pytest.raises(ValueError, match="unavailable Worker context"):
            mappings.acquire(1, _args(l3.native), 0)
    finally:
        l3.worker._worker = l3.transport


def test_failed_close_keeps_backing_and_transport_for_retry(l3):
    l3.worker.submit(dispatch_both(l3), TaskArgs()).wait()
    l3.chips[1]._unmap_host_ddr.side_effect = RuntimeError("unregister failed")
    with pytest.raises(RuntimeError, match="RELEASE failed"):
        l3.worker.close()
    assert not l3.table.closed and not l3.table.unlinked
    l3.transport.close.assert_not_called()
    l3.chips[1]._unmap_host_ddr.side_effect = None
    l3.worker.close()
    assert l3.table.closed and l3.table.unlinked
    l3.transport.close.assert_called_once()


@pytest.mark.parametrize("transfer", [TensorTransfer.NONE, TensorTransfer.H2D, TensorTransfer.MAP])
@pytest.mark.parametrize("needs_owner_recovery", [False, True])
def test_close_recovers_unknown_child_submission_before_releasing_backing(l3, transfer, needs_owner_recovery):
    args = TaskArgs()
    args.add_tensor(l3.native, transfer=transfer)
    mappings = l3.worker._l3_host_mappings()
    mappings.acquire(1, args, 0)
    l3.children[0].acquire((1, 2, 3), args)
    # TASK_FAILED retires the parent fence but supplies no child last-use proof.
    mappings.release(1)
    chip = l3.chips[0]

    def recover():
        assert not l3.table.closed
        assert l3.children[0]._uses
        chip._unmap_host_ddr.assert_not_called()
        l3.events.append(("quiesce", 0))
        return needs_owner_recovery

    chip._quiesce_host_uses = Mock(side_effect=RuntimeError("device drain unproven"))
    with pytest.raises(RuntimeError):
        l3.worker.close()
    assert l3.children[0]._uses
    assert not l3.table.closed
    l3.transport.close.assert_not_called()
    chip._quiesce_host_uses.side_effect = recover
    l3.worker.close()
    assert chip._quiesce_host_uses.call_count == 2
    assert not l3.children[0]._uses
    assert l3.table.closed and l3.table.unlinked
    assert chip.finalize.call_count == int(needs_owner_recovery)
    if transfer == TensorTransfer.MAP:
        assert l3.events.index(("quiesce", 0)) < l3.events.index(("unmap", 0))


def test_idle_cache_does_not_block_unrelated_cleanup_recovery(l3):
    l3.worker.submit(dispatch_both(l3), TaskArgs()).wait()
    l3.worker._ordered_cleanup_error = RuntimeError("unrelated failure")
    l3.worker._close_host_group_mappings()
    assert not l3.worker._host_group_mappings.holds_resources
    l3.table.close()


def test_raw_close_in_callback_refuses_without_poisoning_mapping(l3):
    def callback(orch, args, config):
        orch.submit_next_level(l3.handle, _args(l3.native), config, worker=0)
        with pytest.raises(RuntimeError, match="outside orchestration"):
            l3.table.close()
        assert not l3.table.closed

    l3.worker.submit(callback, TaskArgs()).wait()
    l3.table.close()


@pytest.mark.parametrize(
    "platform,runtime",
    [("a5", "host_build_graph"), ("a5sim", "tensormap_and_ringbuffer"), ("a2a3", "tensormap_and_ringbuffer")],
)
def test_unsupported_l3_child_map_is_refused_before_dispatch(l3, platform, runtime):
    l3.worker._config["platform"] = platform
    l3.worker._config["runtime"] = runtime
    with pytest.raises(RuntimeError, match="L3 MAP requires"):
        l3.worker.submit(dispatch_both(l3), TaskArgs())
    l3.native_orch.submit_next_level_group.assert_not_called()
    l3.transport.control_payload.assert_not_called()
