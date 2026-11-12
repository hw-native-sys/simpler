# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared Host MAP preparation, reuse and uncertain cleanup without a device."""

import contextlib
import itertools
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from simpler.buffer import ImportContext, ImportRegistry, create_host_shared_buffer, mint_owner_instance_id
from simpler.comm_endpoints import DEVICE_AICPU
from simpler.host_ddr_l3 import ChildHostMappings, ParentHostMappings, _Action, _decode, _encode
from simpler.task_interface import AddressSpace, DataType, TaskArgs, TensorTransfer, _materialize_chip_task_args


def args_for(tensor, transfer=TensorTransfer.MAP):
    args = TaskArgs()
    args.add_tensor(tensor, transfer=transfer)
    return args


@pytest.fixture
def group():
    owner = mint_owner_instance_id()
    table = create_host_shared_buffer(128, owner, 1)
    buffers = {table.identity: table}
    events, imports, children, chips = [], [], [], []
    for rank in range(2):
        registry = ImportRegistry(ImportContext(deployment=DEVICE_AICPU, device_owner_instance_id=owner))
        chip = SimpleNamespace(
            _validate_execution_context=Mock(),
            _host_ddr_mapping_supported=True,
            _map_host_ddr=Mock(side_effect=lambda *_args, r=rank: events.append(("map", r)) or 0x400000 + r * 0x10000),
            _unmap_host_ddr=Mock(side_effect=lambda *_args, r=rank: events.append(("unmap", r))),
        )
        child = ChildHostMappings(chip, registry, owner_instance_id=owner, rank=rank, rank_count=2, alive=lambda: True)
        registry.host_mappings = child
        imports.append(registry)
        children.append(child)
        chips.append(chip)
    state = SimpleNamespace(busy=False, fail_reply=set(), requests=[])

    @contextlib.contextmanager
    def admission():
        if state.busy:
            raise RuntimeError("runs still in flight")
        yield

    def dispatch(rank, payload):
        state.requests.append((rank, payload))
        result = children[rank].handle(payload)
        if (_decode(payload)[0], rank) in state.fail_reply:
            raise RuntimeError("reply lost")
        return result

    def fanout(items, target):
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(target, items))

    parent = ParentHostMappings(
        members=(0, 1),
        alive=lambda: True,
        admission=admission,
        allocate_group=itertools.count(1).__next__,
        lookup=buffers.get,
        dispatch=dispatch,
        fanout=fanout,
    )
    yield SimpleNamespace(
        parent=parent,
        table=table,
        native=table.tensor((4,), DataType.INT32, byte_offset=16),
        buffers=buffers,
        children=children,
        imports=imports,
        chips=chips,
        events=events,
        state=state,
    )
    state.busy = False
    state.fail_reply.clear()
    parent._uses.clear()
    parent._declared.clear()
    for chip, child in zip(chips, children):
        chip._unmap_host_ddr.side_effect = None
        chip._validate_execution_context.side_effect = None
        child._uses.clear()
    parent.close_all()
    for buffer in buffers.values():
        buffer.close()
    for registry in imports:
        registry.close()


def test_same_backing_endpoint_local_addresses_and_native_transfers(group):
    args = args_for(group.native)
    args.add_tensor(group.native, transfer=TensorTransfer.NONE)
    args.add_tensor(group.native, transfer=TensorTransfer.H2D)
    group.parent.prepare(args)
    group.parent.acquire(1, args)
    with group.table.host_view() as memory:
        memory[16:20] = b"same"
    for rank, child in enumerate(group.children):
        child.acquire(1, args)
        local = _materialize_chip_task_args(args, group.imports[rank])
        imported = group.imports[rank].require(group.table.identity)
        assert imported.shm.name == group.table.shm.name
        assert bytes(imported.shm.buf[16:20]) == b"same"
        assert local.tensor(0).data == 0x400010 + rank * 0x10000
        assert local.tensor(0).address_space == AddressSpace.DEVICE
        assert local.tensor(1).data == imported.base + 16
        assert local.tensor(2).data == imported.base + 16
        child.release(1)
    group.parent.release(1)


def test_cache_reuse_multiple_views_and_cpu_updates(group):
    args = args_for(group.native)
    group.parent.prepare(args)
    requests = len(group.state.requests)
    for run_id in (1, 2):
        view = group.table.tensor((2,), DataType.INT32, byte_offset=32)
        child_args = args_for(view)
        group.parent.prepare(args)
        group.parent.acquire(run_id, args)
        with group.table.host_view() as memory:
            memory[32] = run_id
        for rank, child in enumerate(group.children):
            group.parent.validate_args(child_args, rank, run_id)
            child.acquire(run_id, child_args)
            assert child.resolve(view)[0] == 0x400000 + rank * 0x10000
            assert group.imports[rank].require(group.table.identity).shm.buf[32] == run_id
            child.release(run_id)
        group.parent.release(run_id)
    assert len(group.state.requests) == requests
    assert all(chip._map_host_ddr.call_count == 1 for chip in group.chips)


def test_multiple_backings_are_independent(group):
    second = create_host_shared_buffer(256, group.table.identity.owner_instance_id, 2)
    group.buffers[second.identity] = second
    args = args_for(group.native)
    args.add_tensor(second.tensor((8,), DataType.INT32), transfer=TensorTransfer.MAP)
    group.parent.prepare(args)
    assert len(group.parent._records) == 2
    group.table.close()
    assert not second.closed
    assert all(len(child._records) == 1 for child in group.children)
    second.close()
    assert all(chip._map_host_ddr.call_count == chip._unmap_host_ddr.call_count == 2 for chip in group.chips)
    assert all(not registry._by_identity for registry in group.imports)


@pytest.mark.parametrize("transfer", [TensorTransfer.NONE, TensorTransfer.H2D, TensorTransfer.MAP])
def test_child_native_and_mapped_uses_block_release(group, transfer):
    group.parent.prepare(args_for(group.native))
    args = args_for(group.native, transfer)
    for child in group.children:
        child.acquire(10, args)
        child.acquire(11, args)
        child.release(10)
    with pytest.raises(RuntimeError, match="QUIESCE failed"):
        group.parent.close(group.table.identity)
    assert not group.table.closed
    assert all(chip._unmap_host_ddr.call_count == 0 for chip in group.chips)
    for child in group.children:
        child.release(11)
    group.parent.close(group.table.identity)


def test_parent_busy_close_does_not_poison_admission(group):
    args = args_for(group.native)
    group.parent.prepare(args)
    group.parent.acquire(1, args)
    with pytest.raises(RuntimeError, match="in-flight"):
        group.table.close()
    assert not group.parent._records[group.table.identity].closing
    group.parent.acquire(2, args)
    group.parent.release(1)
    group.parent.release(2)
    group.table.close()


def test_cached_mapping_reuses_while_runs_are_active(group):
    args = args_for(group.native)
    group.parent.prepare(args)
    group.parent.acquire(1, args)
    group.state.busy = True
    group.parent.prepare(args)
    group.parent.acquire(2, args)
    assert all(chip._map_host_ddr.call_count == 1 for chip in group.chips)
    second = create_host_shared_buffer(128, group.table.identity.owner_instance_id, 2)
    group.buffers[second.identity] = second
    with pytest.raises(RuntimeError, match="still in flight"):
        group.parent.prepare(args_for(second.tensor((4,), DataType.INT32)))


@pytest.mark.parametrize("phase", [_Action.PREPARE, _Action.QUIESCE, _Action.RELEASE])
def test_lost_reply_retains_or_unwinds_every_attempted_child(group, phase):
    args = args_for(group.native)
    if phase == _Action.PREPARE:
        group.state.fail_reply.add((phase, 1))
        with pytest.raises(RuntimeError, match="PREPARE failed"):
            group.parent.prepare(args)
        assert not group.parent.holds_resources
        assert not any(child.holds_resources for child in group.children)
    else:
        group.parent.prepare(args)
        group.state.fail_reply.add((phase, 1))
        with pytest.raises(RuntimeError, match=f"{phase.name} failed"):
            group.table.close()
        assert not group.table.closed and not group.table.unlinked
        assert group.parent.holds_resources
        with pytest.raises(ValueError, match="closing"):
            group.parent.prepare(args)
        group.state.fail_reply.clear()
        group.table.close()
        assert group.table.closed and group.table.unlinked


def test_failed_prepare_and_failed_rollback_retain_backing(group):
    group.chips[1]._map_host_ddr.side_effect = RuntimeError("register failed")
    group.chips[1]._unmap_host_ddr.side_effect = RuntimeError("unregister failed")
    with pytest.raises(RuntimeError, match="PREPARE failed") as error:
        group.parent.prepare(args_for(group.native))
    assert "retained" in "\n".join(error.value.__notes__)
    assert group.parent.holds_resources
    assert not group.table.closed and not group.table.unlinked
    group.chips[1]._unmap_host_ddr.side_effect = None
    group.table.close()


def test_duplicate_prepare_and_release_do_not_repeat_native_work(group):
    group.parent.prepare(args_for(group.native))
    rank, payload = group.state.requests[0]
    group.children[rank].handle(payload)
    group.parent.close(group.table.identity)
    releases = [(r, p) for r, p in group.state.requests if _decode(p)[0] == _Action.RELEASE]
    for r, p in releases:
        group.children[r].handle(p)
    assert all(chip._map_host_ddr.call_count == chip._unmap_host_ddr.call_count == 1 for chip in group.chips)
    with pytest.raises(ValueError, match="retired"):
        group.children[rank].handle(payload)


def test_undeclared_callback_map_refuses_even_cached_backing(group):
    args = args_for(group.native)
    group.parent.prepare(args)
    group.parent.acquire(1, TaskArgs())
    with pytest.raises(ValueError, match="declared"):
        group.parent.validate_args(args, 0, 1)
    with pytest.raises(ValueError, match="outside"):
        group.parent.validate_args(args, 9, 1)


def test_resolution_requires_accepted_use(group):
    group.parent.prepare(args_for(group.native))
    for child in group.children:
        with pytest.raises(ValueError, match="accepted use"):
            child.resolve(group.native)


@pytest.mark.parametrize("mutation", ["owner", "rank", "count", "group", "length", "version"])
def test_invalid_control_is_rejected_before_native_mapping(group, mutation):
    descriptor = group.native.buffer
    if mutation == "owner":
        foreign = create_host_shared_buffer(128, mint_owner_instance_id(), 1)
        try:
            descriptor = foreign.to_descriptor()
        finally:
            foreign.close()
    payload = _encode(
        _Action.PREPARE,
        0 if mutation == "group" else 1,
        1 if mutation == "rank" else 0,
        3 if mutation == "count" else 2,
        descriptor,
    )
    if mutation == "length":
        payload = payload[:-1]
    if mutation == "version":
        payload = payload[:4] + b"\x01" + payload[5:]
    with pytest.raises(ValueError):
        group.children[0].handle(payload)
    group.chips[0]._map_host_ddr.assert_not_called()


def test_stale_child_context_retains_parent_cleanup_debt(group):
    group.parent.prepare(args_for(group.native))
    group.chips[1]._validate_execution_context.side_effect = RuntimeError("context gone")
    with pytest.raises(RuntimeError, match="QUIESCE failed"):
        group.table.close()
    assert not group.table.closed and not group.table.unlinked
    group.chips[0]._unmap_host_ddr.assert_not_called()


def test_live_cpu_export_refuses_before_child_teardown(group):
    group.parent.prepare(args_for(group.native))
    view = group.table.host_view()
    try:
        with pytest.raises(BufferError):
            group.table.close()
        assert all(not chip._unmap_host_ddr.called for chip in group.chips)
    finally:
        view.release()
    group.table.close()


def test_child_cpu_import_close_failure_retries_without_second_unmap(group, monkeypatch):
    group.parent.prepare(args_for(group.native))
    registry = group.imports[1]
    original = registry.unregister
    monkeypatch.setattr(registry, "unregister", Mock(side_effect=BufferError("live CPU export")))
    with pytest.raises(RuntimeError, match="RELEASE failed"):
        group.table.close()
    assert group.parent.holds_resources
    assert not group.table.closed and not group.table.unlinked
    assert group.children[1].holds_resources
    assert registry.require(group.table.identity) is not None
    monkeypatch.setattr(registry, "unregister", original)
    group.table.close()
    assert all(chip._unmap_host_ddr.call_count == 1 for chip in group.chips)
    assert not any(child.holds_resources for child in group.children)


def test_group_cannot_be_reused_for_another_live_backing(group):
    group.parent.prepare(args_for(group.native))
    second = create_host_shared_buffer(128, group.table.identity.owner_instance_id, 2)
    group.buffers[second.identity] = second
    payload = _encode(_Action.PREPARE, 1, 0, 2, second.to_descriptor())
    with pytest.raises(ValueError, match="conflicting backing"):
        group.children[0].handle(payload)
    assert group.chips[0]._map_host_ddr.call_count == 1


def test_child_close_releases_host_use_only_after_native_retry_proves_cleanup(group):
    args = args_for(group.native)
    group.parent.prepare(args)
    child = group.children[0]
    child.acquire(1, args)
    run = SimpleNamespace(_retry_owner_cleanup=Mock(side_effect=[False, True]))
    child.defer_release(1, run)

    child.close_idle()
    group.chips[0]._unmap_host_ddr.assert_not_called()
    child.close_idle()
    group.chips[0]._unmap_host_ddr.assert_called_once()
    child.require_closed()
