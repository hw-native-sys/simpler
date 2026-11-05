# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Worker and orchestration boundaries for collective Host mappings, without hardware."""

import contextlib
import itertools
import threading
from multiprocessing.shared_memory import SharedMemory
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
from simpler.buffer import ImportContext, ImportRegistry
from simpler.callable_identity import CallableHandle
from simpler.comm_endpoints import DEVICE_AICPU
from simpler.host_ddr_l3 import ChildHostMappings, _Action, _decode
from simpler.orchestrator import Orchestrator, _callback_run
from simpler.task_interface import DataType, TaskArgs
from simpler.worker import _COMM_INIT_HEADER, _CTRL_HOST_DDR, Worker, WorkerType, _Lifecycle, _RunResources


def _args(tensor):
    args = TaskArgs()
    args.add_tensor(tensor)
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
            _host_urma_supported=True,
            _map_host_ddr=Mock(return_value=0x1000000 + rank * 0x10000),
            _prepare_host_urma=Mock(return_value=0x2000000 + rank * 0x10000),
            _release_host_urma=Mock(side_effect=lambda r=rank: events.append(("release", r))),
            _unmap_host_ddr=Mock(side_effect=lambda *_args, r=rank: events.append(("unmap", r))),
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
        return children[rank].handle(payload, base_comm_handle=0xAABB)

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
        )

    monkeypatch.setattr(worker, "_resolve_handle", resolve_handle)
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
        handle=CallableHandle("sha256:" + "00" * 32, "CHIP_CALLABLE", "LOCAL_CHIP"),
    )
    yield state
    for chip, child in zip(chips, children, strict=True):
        chip._release_host_urma.side_effect = None
        chip._unmap_host_ddr.side_effect = None
        child._uses.clear()
    worker._accepted_run_handles.clear()
    worker._ordered_cleanup_error = None
    worker.close()
    for registry in imports:
        registry.close()


def test_worker_map_bootstraps_children_and_reuses_complete_backing(l3):
    mapped = l3.worker.map_view(l3.native)
    second = l3.worker.map_view(l3.table.tensor((8,), DataType.INT32))
    assert l3.transport.control_comm_init.call_count == 2
    assert all(chip._map_host_ddr.call_count == 1 for chip in l3.chips)
    l3.table.shm.buf[16:20] = b"same"
    for registry in l3.imports:
        imported = registry.require(l3.table.identity)
        assert imported.shm.name == l3.table.shm.name
        assert bytes(imported.shm.buf[16:20]) == b"same"
    for phase in (_Action.MAP, _Action.PREPARE):
        assert {(name, rank) for name, rank in l3.events if name == phase.name} == {(phase.name, 0), (phase.name, 1)}
    l3.worker.unmap_view(mapped)
    l3.worker.unmap_view(second)
    assert not l3.worker._host_group_mappings.holds_resources
    assert not l3.table.closed


def test_direct_buffer_close_releases_all_children_before_unlink(l3):
    l3.worker.map_view(l3.native)
    l3.table.close()
    assert l3.table.closed and l3.table.unlinked
    for chip in l3.chips:
        chip._release_host_urma.assert_called_once()
        chip._unmap_host_ddr.assert_called_once()


def test_worker_close_releases_mapping_before_buffers_and_transport(l3):
    l3.worker.map_view(l3.native)
    l3.worker.close()
    assert l3.table.closed and l3.table.unlinked
    assert l3.events[-1] == ("transport_close", -1)
    assert sum(name == "unmap" for name, _ in l3.events) == 2
    l3.worker.close()
    l3.table.close()


def test_worker_close_failed_host_gate_retains_owner_and_transport_for_retry(l3):
    l3.worker.map_view(l3.native)
    l3.chips[1]._release_host_urma.side_effect = RuntimeError("native registration still live")
    with pytest.raises(RuntimeError, match="RELEASE failed"):
        l3.worker.close()
    assert l3.worker._lifecycle is _Lifecycle.CLOSED
    assert not l3.table.closed and not l3.table.unlinked
    l3.transport.close.assert_not_called()
    assert l3.worker._worker is l3.transport
    l3.chips[1]._unmap_host_ddr.assert_not_called()
    l3.chips[1]._release_host_urma.side_effect = None
    l3.worker.close()
    assert l3.table.closed and l3.table.unlinked
    l3.transport.close.assert_called_once()


def test_already_closed_mapping_does_not_gate_unrelated_cleanup(l3):
    mapped = l3.worker.map_view(l3.native)
    l3.worker.unmap_view(mapped)
    l3.worker._ordered_cleanup_error = RuntimeError("unrelated cleanup failure")
    l3.worker._host_group_mappings.close_all()
    l3.table.close()
    assert l3.table.closed


def test_graph_callback_refuses_map_unmap_and_raw_owner_close(l3):
    mapped = l3.worker.map_view(l3.native)
    before = l3.transport.control_payload.call_count
    with _callback_run(10, l3.worker):
        for action in (lambda: l3.worker.map_view(l3.native), lambda: l3.worker.unmap_view(mapped), l3.table.close):
            with pytest.raises(RuntimeError, match="outside orchestration"):
                action()
    assert l3.transport.control_payload.call_count == before
    assert not l3.table.closed


def test_outstanding_run_fences_mapping_even_without_tensor_dependencies(l3):
    mapped = l3.worker.map_view(l3.native)
    handle = l3.worker.submit(lambda *_args: None, TaskArgs())
    assert not handle._resources.touched_identities
    before = l3.transport.control_payload.call_count
    for action in (lambda: l3.worker.map_view(l3.native), lambda: l3.worker.unmap_view(mapped), l3.table.close):
        with pytest.raises(RuntimeError, match="still in flight"):
            action()
    assert l3.transport.control_payload.call_count == before
    handle.wait()
    l3.worker.unmap_view(mapped)


def test_captured_tensor_dispatch_is_validated_and_retained_by_actual_submit(l3):
    mapped = l3.worker.map_view(l3.native)

    def graph(orch, _args, _cfg):
        orch.submit_next_level(l3.handle, _args_for_capture(), worker=1)

    def _args_for_capture():
        return _args(mapped)

    handle = l3.worker.submit(graph, TaskArgs())
    assert l3.table.identity in handle._resources.touched_identities
    assert l3.native_orch.submit_next_level.call_args.args[5] == 1
    handle.wait()
    l3.worker.unmap_view(mapped)
    l3.native_orch.submit_next_level.reset_mock()
    with pytest.raises(ValueError, match="stale"):
        l3.worker.submit(graph, TaskArgs())
    l3.native_orch.submit_next_level.assert_not_called()


def test_group_checks_all_members_before_native_admission_or_retention(l3):
    mapped = l3.worker.map_view(l3.native)
    forged = l3.table.tensor((4,), DataType.INT32, byte_offset=32)._with_mapping_ref(mapped._mapping_ref)
    resources = _RunResources()
    l3.worker._building_run_resources = resources
    try:
        with pytest.raises(ValueError, match="stale"):
            l3.worker._orch.submit_next_level_group(l3.handle, [_args(mapped), _args(forged)], workers=[0, 1])
        assert not resources.touched_identities
        l3.native_orch.submit_next_level_group.assert_not_called()
        l3.worker._orch.submit_next_level_group(l3.handle, [_args(mapped), _args(mapped)], workers=[0, 1])
        assert resources.touched_identities == {l3.table.identity}
        l3.native_orch.submit_next_level_group.assert_called_once()
    finally:
        l3.worker._building_run_resources = None


@pytest.mark.parametrize("namespace", ["LOCAL_PYTHON", "REMOTE_TASK_DISPATCHER"])
def test_mapped_tensor_refused_for_host_or_remote_next_level(l3, namespace):
    mapped = l3.worker.map_view(l3.native)
    handle = CallableHandle("sha256:" + "11" * 32, "PYTHON_IMPORT", namespace)
    with pytest.raises(ValueError, match="registered local L2"):
        l3.worker._orch.submit_next_level(handle, _args(mapped), worker=0)
    l3.native_orch.submit_next_level.assert_not_called()


def test_mapped_tensor_refused_for_sub_worker_and_l4(l3):
    mapped = l3.worker.map_view(l3.native)
    handle = CallableHandle("sha256:" + "22" * 32, "PYTHON_IMPORT", "LOCAL_PYTHON")
    with pytest.raises(ValueError, match="registered device endpoint"):
        l3.worker._orch.submit_sub(handle, _args(mapped))
    l3.native_orch.submit_sub.assert_not_called()
    higher = Worker(level=4, platform="a5", runtime="tensormap_and_ringbuffer")
    try:
        with pytest.raises(TypeError, match="level 2 or level 3"):
            higher.map_view(l3.native)
        with pytest.raises(ValueError, match="registered local L2"):
            higher._validate_mapped_host_args(_args(mapped), 0)
    finally:
        higher.close()


def test_individual_child_import_release_cannot_bypass_registration(l3):
    mapped = l3.worker.map_view(l3.native)
    for registry in l3.imports:
        with pytest.raises(RuntimeError, match="registered"):
            registry.unregister(l3.table.identity)
        assert registry.require(l3.table.identity).shm is not None
    l3.worker.unmap_view(mapped)
    for registry in l3.imports:
        registry.unregister(l3.table.identity)
        with pytest.raises(KeyError):
            registry.require(l3.table.identity)


@pytest.mark.parametrize(
    ("platform", "runtime", "devices", "message"),
    [
        ("a5sim", "tensormap_and_ringbuffer", (0, 1), "A5 onboard"),
        ("a2a3", "tensormap_and_ringbuffer", (0, 1), "A5 onboard"),
        ("a5", "host_build_graph", (0, 1), "A5 onboard"),
        ("a5", "tensormap_and_ringbuffer", (0,), "at least two"),
    ],
)
def test_unsupported_mapping_scope_refused_before_communicator_effects(l3, platform, runtime, devices, message):
    l3.worker._config.update(platform=platform, runtime=runtime, device_ids=list(devices))
    with pytest.raises((ValueError, RuntimeError), match=message):
        l3.worker.map_view(l3.native)
    assert l3.worker._host_group_mappings is None
    l3.transport.control_comm_init.assert_not_called()
    l3.transport.control_payload.assert_not_called()


def test_close_drains_actual_accepted_run_before_unregistering(l3):
    l3.worker.map_view(l3.native)
    handle = l3.worker.submit(lambda *_args: None, TaskArgs())

    def completed(_run_id, _timeout):
        assert all(chip._release_host_urma.call_count == 0 for chip in l3.chips)
        return True

    l3.native_orch._wait_run_for.side_effect = completed
    l3.worker.close()
    assert handle._cleanup_published
    assert l3.table.closed
    assert all(chip._release_host_urma.call_count == 1 for chip in l3.chips)


def test_owner_close_does_not_hold_host_lock_while_waiting_for_graph(l3, monkeypatch):
    l3.worker.map_view(l3.native)
    entered_graph = threading.Event()
    close_at_admission = threading.Event()
    original = l3.worker._host_group_mappings._admission
    errors = {"submit": [], "close": []}

    @contextlib.contextmanager
    def admission():
        if threading.current_thread().name == "host-owner-close":
            close_at_admission.set()
        with original():
            yield

    monkeypatch.setattr(l3.worker._host_group_mappings, "_admission", admission)

    def graph(*_args):
        entered_graph.set()
        assert close_at_admission.wait(3)
        # A bounded acquire makes the lock-order regression fail and unwind
        # rather than leave both test threads permanently deadlocked.
        assert l3.table._host_lock.acquire(timeout=1), "owner close holds Host lock while awaiting graph admission"
        try:
            with l3.table.host_view() as memory:
                assert len(memory) == 128
        finally:
            l3.table._host_lock.release()

    def submit():
        try:
            l3.worker.submit(graph, TaskArgs()).wait()
        except BaseException as exc:  # noqa: BLE001
            errors["submit"].append(exc)

    def close():
        try:
            l3.table.close()
        except BaseException as exc:  # noqa: BLE001
            errors["close"].append(exc)

    submitting = threading.Thread(target=submit, daemon=True)
    closing = threading.Thread(target=close, name="host-owner-close", daemon=True)
    submitting.start()
    assert entered_graph.wait(3)
    closing.start()
    submitting.join(5)
    closing.join(5)
    assert not submitting.is_alive() and not closing.is_alive()
    assert not errors["submit"], errors["submit"]
    # Close can reach admission before the accepted RunHandle has retired.
    assert all(isinstance(exc, RuntimeError) and "still in flight" in str(exc) for exc in errors["close"])
    l3.table.close()
    assert l3.table.closed


def test_owner_close_rechecks_mapping_installed_after_deregistration(l3, monkeypatch):
    old_view = l3.worker.map_view(l3.native)
    registry = l3.worker._host_group_mappings
    original = registry.close
    replacement = []

    def remap_after_close(identity):
        original(identity)
        if not replacement:
            replacement.append(l3.worker.map_view(l3.native))

    monkeypatch.setattr(registry, "close", remap_after_close)
    with pytest.raises(RuntimeError, match="registered"):
        l3.table.close()
    assert not l3.table.closed and not l3.table.unlinked
    registry.validate(replacement[0])
    with pytest.raises(ValueError, match="stale"):
        registry.validate(old_view)
    monkeypatch.setattr(registry, "close", original)
    l3.worker.unmap_view(replacement[0])
    l3.table.close()
