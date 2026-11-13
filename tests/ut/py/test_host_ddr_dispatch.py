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
from multiprocessing.shared_memory import SharedMemory
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
from simpler.buffer import ImportContext, ImportRegistry
from simpler.callable_identity import CallableHandle
from simpler.comm_endpoints import DEVICE_AICPU
from simpler.host_ddr_l3 import ChildHostMappings, _decode
from simpler.orchestrator import Orchestrator
from simpler.task_interface import DataType, TaskArgs, TensorTransfer
from simpler.worker import _COMM_INIT_HEADER, _CTRL_HOST_DDR, Worker, WorkerType, _Lifecycle


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
            _validate_execution_context=Mock(),
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

    def reap_children(_deadline):
        worker._chip_shms.clear()
        worker._chip_pids.clear()

    monkeypatch.setattr(worker, "_reclaim_child_groups", reap_children)

    def resolve_handle(handle, **_kwargs):
        return SimpleNamespace(
            digest=handle.digest,
            kind=handle.kind,
            target_namespace=handle.target_namespace,
            eligible_worker_ids=(0, 1),
            target=SimpleNamespace(ccu_module_count=0),
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
    if worker._host_group_mappings is not None:
        worker._host_group_mappings._uses.clear()
        worker._host_group_mappings._declared.clear()
    worker._ordered_cleanup_error = None
    worker.close()
    for registry in imports:
        registry.close()


@pytest.mark.parametrize("platform", ["a5", "a2a3"])
def test_outer_prepare_precedes_callback_without_communication_setup(l3, platform):
    l3.worker._config["platform"] = platform

    def callback(orch, args, config):
        assert all(chip._map_host_ddr.call_count == 1 for chip in l3.chips)
        assert all(child.holds_resources for child in l3.children)
        orch.submit_next_level(l3.handle, args, config, worker=0)

    pending = l3.worker.submit(callback, _args(l3.native))
    l3.transport.control_comm_init.assert_not_called()
    for chip in l3.chips:
        chip._prepare_host_urma.assert_not_called()
    with pytest.raises(RuntimeError):
        l3.worker.release_buffer(l3.table)
    pending.wait()
    l3.worker.release_buffer(l3.table)
    assert l3.table.closed and l3.table.unlinked


def test_cached_outer_maps_allow_subviews_and_group_dispatch(l3):
    def callback(orch, args, config):
        view = l3.table.tensor((2,), DataType.INT32, byte_offset=24)
        orch.submit_next_level_group(l3.handle, [_args(view), _args(view)], config, workers=[0, 1])

    for _ in range(2):
        l3.worker.submit(callback, _args(l3.native)).wait()
    assert all(chip._map_host_ddr.call_count == 1 for chip in l3.chips)
    assert l3.transport.control_payload.call_count == 2


def test_callback_first_time_mapping_is_rejected_without_preparation(l3):
    def callback(orch, _args_unused, config):
        orch.submit_next_level(l3.handle, _args(l3.native), config, worker=0)

    with pytest.raises(ValueError, match="declared"):
        l3.worker.submit(callback, TaskArgs())
    l3.transport.control_payload.assert_not_called()


def test_failed_preparation_never_enters_callback(l3):
    l3.chips[1]._map_host_ddr.side_effect = RuntimeError("registration failed")
    callback = Mock()
    with pytest.raises(RuntimeError, match="PREPARE failed"):
        l3.worker.submit(callback, _args(l3.native))
    callback.assert_not_called()
    assert not l3.worker._accepted_run_handles
    assert not l3.worker._host_group_mappings.holds_resources


def test_parent_retains_outer_buffer_even_without_child_dispatch(l3):
    pending = l3.worker.submit(lambda *_args: None, _args(l3.native))
    assert l3.table.identity in pending._resources.touched_identities
    for release in (l3.table.close, lambda: l3.worker.release_buffer(l3.table)):
        with pytest.raises(RuntimeError):
            release()
    pending.wait()
    l3.table.close()


def test_completed_calls_keep_registration_until_worker_close(l3):
    l3.worker.submit(lambda *_args: None, _args(l3.native)).wait()
    assert all(not chip._unmap_host_ddr.called for chip in l3.chips)
    l3.worker.close()
    assert l3.table.closed and l3.table.unlinked
    assert l3.events[-1] == ("transport_close", -1)
    assert all(chip._unmap_host_ddr.call_count == 1 for chip in l3.chips)


def test_failed_close_keeps_backing_and_transport_for_retry(l3):
    l3.worker.submit(lambda *_args: None, _args(l3.native)).wait()
    l3.chips[1]._unmap_host_ddr.side_effect = RuntimeError("unregister failed")
    with l3.table.host_view() as view:
        view[0] = 7
        with pytest.raises(RuntimeError, match="RELEASE failed"):
            l3.worker.close()
        with pytest.raises(BufferError, match="new CPU exports"):
            l3.table.host_view()
        assert view[0] == 7
    assert not l3.table.closed and not l3.table.unlinked
    l3.transport.close.assert_not_called()
    l3.chips[1]._unmap_host_ddr.side_effect = None
    l3.worker.close()
    assert l3.table.closed and l3.table.unlinked
    l3.transport.close.assert_called_once()


def test_unconfirmed_release_waits_for_clean_child_exit(l3):
    worker = l3.worker
    worker._chip_pids = [101, 102]
    worker.submit(lambda *_args: None, _args(l3.native)).wait()
    control = l3.transport.control_payload.side_effect
    unreachable = False

    def lose_release_reply(worker_type, rank, command, payload, timeout):
        nonlocal unreachable
        if rank == 1 and (_decode(payload)[0].name == "RELEASE" or unreachable):
            unreachable = True
            raise RuntimeError("control channel unavailable")
        return control(worker_type, rank, command, payload, timeout)

    def close_transport():
        for child, imports in zip(l3.children, l3.imports, strict=True):
            child.close_idle()
            imports.close()

    l3.transport.control_payload.side_effect = lose_release_reply
    l3.transport.close.side_effect = close_transport
    with pytest.raises(RuntimeError, match="cleanup is unproven"):
        worker.close()
    assert not l3.table.unlinked
    assert all(chip._unmap_host_ddr.call_count == 1 for chip in l3.chips)
    l3.transport.close.assert_called_once()
    worker._clean_l3_host_map_exits.add(102)
    worker.close()
    assert l3.table.closed and l3.table.unlinked
    l3.transport.close.assert_called_once()


def test_idle_cache_does_not_block_unrelated_cleanup_recovery(l3):
    l3.worker.submit(lambda *_args: None, _args(l3.native)).wait()
    l3.worker._ordered_cleanup_error = RuntimeError("unrelated failure")
    l3.worker._close_host_group_mappings()
    assert not l3.worker._host_group_mappings.holds_resources
    l3.table.close()


def test_raw_close_in_callback_refuses_without_poisoning_mapping(l3):
    def callback(*_args):
        with pytest.raises(RuntimeError, match="outside orchestration"):
            l3.table.close()
        with pytest.raises(RuntimeError, match="outside orchestration"):
            l3.worker.release_buffer(l3.table)
        assert not l3.table.closed

    l3.worker.submit(callback, _args(l3.native)).wait()
    l3.table.close()


@pytest.mark.parametrize(
    "platform,runtime",
    [("a5", "host_build_graph"), ("a5sim", "tensormap_and_ringbuffer"), ("a2a3sim", "tensormap_and_ringbuffer")],
)
def test_unsupported_l3_map_is_refused_before_callback(l3, platform, runtime):
    l3.worker._config["platform"] = platform
    l3.worker._config["runtime"] = runtime
    callback = Mock()
    with pytest.raises(RuntimeError, match="L3 MAP requires"):
        l3.worker.submit(callback, _args(l3.native))
    callback.assert_not_called()
    l3.transport.control_payload.assert_not_called()
