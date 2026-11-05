# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Distributed mapping authority and failure ownership without an NPU."""

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


@pytest.fixture
def group():
    owner = mint_owner_instance_id()
    table = create_host_shared_buffer(128, owner, 1)
    events = []
    imports = [ImportRegistry(ImportContext(deployment=DEVICE_AICPU, device_owner_instance_id=owner)) for _ in range(2)]
    chips = []
    children = []
    for rank in range(2):
        chip = SimpleNamespace(
            _host_urma_supported=True,
            _map_host_ddr=Mock(
                side_effect=lambda *_args, r=rank: events.append(("map", r)) or (0x400000 + r * 0x10000)
            ),
            _prepare_host_urma=Mock(side_effect=lambda *_args, r=rank: events.append(("prepare", r)) or 0x900000),
            _release_host_urma=Mock(side_effect=lambda r=rank: events.append(("release", r))),
            _unmap_host_ddr=Mock(side_effect=lambda *_args, r=rank: events.append(("unmap", r))),
        )
        child = ChildHostMappings(
            chip, imports[rank], owner_instance_id=owner, rank=rank, rank_count=2, alive=lambda: True
        )
        imports[rank].host_mappings = child
        children.append(child)
        chips.append(chip)

    def fanout(items, target):
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(target, items))

    state = SimpleNamespace(busy=False, fail_reply=set())

    @contextlib.contextmanager
    def admission():
        if state.busy:
            raise RuntimeError("runs still in flight")
        yield

    def dispatch(rank, payload):
        result = children[rank].handle(payload, base_comm_handle=123)
        if (_decode(payload)[0], rank) in state.fail_reply:
            raise RuntimeError("reply lost")
        return result

    parent = ParentHostMappings(
        members=(0, 1),
        alive=lambda: True,
        admission=admission,
        allocate_group=itertools.count(1).__next__,
        ensure_comm=Mock(),
        dispatch=dispatch,
        fanout=fanout,
    )
    result = SimpleNamespace(
        parent=parent,
        children=children,
        chips=chips,
        imports=imports,
        table=table,
        native=table.tensor((4,), DataType.INT32, byte_offset=16),
        events=events,
        state=state,
    )
    yield result
    state.busy = False
    state.fail_reply.clear()
    for chip, child in zip(chips, children, strict=True):
        chip._release_host_urma.side_effect = None
        chip._unmap_host_ddr.side_effect = None
        child._uses.clear()
    parent.close_all()
    table.close()
    for registry in imports:
        registry.close()


def test_all_imports_precede_collective_and_resolve_same_backing(group):
    mapped = group.parent.map_view(group.table, group.native)
    assert set(group.events[:2]) == {("map", 0), ("map", 1)}
    assert set(group.events[2:]) == {("prepare", 0), ("prepare", 1)}
    group.table.shm.buf[16:20] = b"same"
    children: list[ChildHostMappings] = group.children
    registries: list[ImportRegistry] = group.imports
    assert len(children) == len(registries)
    for rank, child in enumerate(children):
        imports = registries[rank]
        imported = imports.require(group.table.identity)
        assert imported.shm is not None
        memory = imported.shm.buf
        assert memory is not None
        assert imported.shm.name == group.table.shm.name
        assert bytes(memory[16:20]) == b"same"
        group.chips[rank]._prepare_host_urma.assert_called_once_with(123, 1, imported.base, 128)
        args = TaskArgs()
        args.add_tensor(group.native, transfer=TensorTransfer.NONE)
        args.add_tensor(mapped)
        args.add_tensor(group.native, transfer=TensorTransfer.H2D)
        local = _materialize_chip_task_args(args, imports)
        assert local.tensor(0).data == imported.base + 16
        assert local.tensor(1).data == 0x400010 + rank * 0x10000
        assert local.tensor(1).address_space == AddressSpace.DEVICE
        assert local.tensor(2).data == imported.base + 16
        group.parent.validate(mapped, rank)
        child.resolve(mapped)


def test_views_share_registration_and_revocation_is_exact(group):
    first = group.parent.map_view(group.table, group.native)
    second = group.parent.map_view(group.table, group.table.tensor((8,), DataType.INT32))
    group.parent.unmap_view(first)
    for child, chip in zip(group.children, group.chips, strict=True):
        with pytest.raises(ValueError, match="stale"):
            child.resolve(first)
        child.resolve(second)
        assert chip._map_host_ddr.call_count == 1
        chip._unmap_host_ddr.assert_not_called()
    group.parent.unmap_view(second)
    assert not group.parent.holds_resources
    assert not group.table.closed
    for chip in group.chips:
        chip._release_host_urma.assert_called_once()
        chip._unmap_host_ddr.assert_called_once()
    fresh = group.parent.map_view(group.table, group.native)
    assert fresh._mapping_ref not in (first._mapping_ref, second._mapping_ref)
    with pytest.raises(ValueError, match="stale"):
        group.parent.validate(first)


def test_forged_range_and_wrong_target_do_not_gain_authority(group):
    mapped = group.parent.map_view(group.table, group.native)
    forged = group.table.tensor((4,), DataType.INT32, byte_offset=32)._with_mapping_ref(mapped._mapping_ref)
    with pytest.raises(ValueError, match="outside"):
        group.parent.validate(mapped, 4)
    for registry in (group.parent, *group.children):
        check = registry.validate if registry is group.parent else registry.resolve
        with pytest.raises(ValueError, match="stale"):
            check(forged)


def test_second_backing_is_rejected_before_native_work(group):
    group.parent.map_view(group.table, group.native)
    other = create_host_shared_buffer(128, mint_owner_instance_id(), 2)
    try:
        with pytest.raises(ValueError, match="another Host"):
            group.parent.map_view(other, other.tensor((4,), DataType.INT32))
        assert all(chip._map_host_ddr.call_count == 1 for chip in group.chips)
    finally:
        other.close()


def test_failed_import_mapping_prevents_any_collective_and_rolls_back(group):
    group.chips[1]._map_host_ddr.side_effect = RuntimeError("mapping failed after ownership")
    with pytest.raises(RuntimeError, match="MAP failed"):
        group.parent.map_view(group.table, group.native)
    assert not group.parent.holds_resources
    for chip in group.chips:
        chip._prepare_host_urma.assert_not_called()
        chip._unmap_host_ddr.assert_called_once()


def test_lost_prepare_reply_keeps_backing_when_rollback_fails(group):
    group.state.fail_reply.add((_Action.PREPARE, 1))
    group.chips[1]._release_host_urma.side_effect = RuntimeError("release failed")
    with pytest.raises(RuntimeError, match="PREPARE failed") as caught:
        group.parent.map_view(group.table, group.native)
    assert "backing retained" in caught.value.__notes__[0]
    assert group.parent.holds_resources
    assert group.children[1].holds_resources
    group.chips[1]._unmap_host_ddr.assert_not_called()
    with pytest.raises(RuntimeError):
        group.table.close()
    assert not group.table.closed
    with pytest.raises(RuntimeError, match="registered"):
        group.table.shm.unlink()
    with pytest.raises(RuntimeError, match="coordinated"):
        group.children[1].require_closed()
    group.chips[1]._release_host_urma.side_effect = None
    group.state.fail_reply.clear()
    group.parent.close_all()
    assert not group.parent.holds_resources


@pytest.mark.parametrize("existing_notes", [None, ["earlier failure detail"]])
def test_rollback_preserves_original_error_without_add_note(group, monkeypatch, existing_notes):
    class LegacyError(RuntimeError):
        add_note = None

    original = LegacyError("PREPARE failed")
    if existing_notes is not None:
        original.__notes__ = list(existing_notes)
    phase = group.parent._phase

    def fail_after_prepare(*args, **kwargs):
        phase(*args, **kwargs)
        if args[1] == _Action.PREPARE:
            raise original

    monkeypatch.setattr(group.parent, "_phase", fail_after_prepare)
    group.chips[1]._release_host_urma.side_effect = RuntimeError("release failed")
    with pytest.raises(LegacyError, match="PREPARE failed") as caught:
        group.parent.map_view(group.table, group.native)
    assert caught.value is original
    notes = getattr(original, "__notes__", [])
    assert notes[:-1] == (existing_notes or [])
    assert "Host backing retained: rollback failed: Host mapping RELEASE failed" in notes[-1]
    assert group.parent.holds_resources
    assert group.children[1].holds_resources
    group.chips[1]._unmap_host_ddr.assert_not_called()
    assert not group.table.closed
    group.chips[1]._release_host_urma.side_effect = None
    group.parent.close_all()
    assert not group.parent.holds_resources


def test_lost_release_reply_is_retryable_without_double_native_release(group):
    mapped = group.parent.map_view(group.table, group.native)
    group.state.fail_reply.add((_Action.RELEASE, 1))
    with pytest.raises(RuntimeError, match="RELEASE failed"):
        group.parent.unmap_view(mapped)
    assert group.parent.holds_resources
    with pytest.raises(ValueError, match="stale"):
        group.parent.validate(mapped)
    group.state.fail_reply.clear()
    group.parent.unmap_view(mapped)
    assert not group.parent.holds_resources
    for chip in group.chips:
        chip._release_host_urma.assert_called_once()


def test_native_workspace_failure_precedes_unmap_and_retains_import(group):
    group.chips[1]._prepare_host_urma.side_effect = RuntimeError("partially created workspace")
    group.chips[1]._release_host_urma.side_effect = RuntimeError("workspace still live")
    with pytest.raises(RuntimeError, match="PREPARE failed"):
        group.parent.map_view(group.table, group.native)
    with pytest.raises(RuntimeError, match="registered"):
        group.children[1].check_unregistered(group.table.identity)
    group.chips[1]._unmap_host_ddr.assert_not_called()
    assert group.imports[1].require(group.table.identity).shm is not None


def test_workspace_is_pinned_even_for_dispatch_without_mapped_args(group):
    mapped = group.parent.map_view(group.table, group.native)
    child = group.children[0]
    key = (1, 2, 3, 4)
    child.acquire(key, TaskArgs())
    with pytest.raises(RuntimeError, match="already reserved"):
        child.acquire(key, TaskArgs())
    with pytest.raises(RuntimeError, match="QUIESCE failed"):
        group.parent.unmap_view(mapped)
    for chip in group.chips:
        chip._release_host_urma.assert_not_called()
        chip._unmap_host_ddr.assert_not_called()
    child.release(key)
    group.parent.unmap_view(mapped)


def test_buffer_close_obeys_parent_run_admission(group):
    group.parent.map_view(group.table, group.native)
    group.state.busy = True
    with pytest.raises(RuntimeError, match="in flight"):
        group.table.close()
    assert not group.table.closed
    for chip in group.chips:
        chip._release_host_urma.assert_not_called()


def test_control_codec_bounds_authority_and_retired_group(group):
    mapped = group.parent.map_view(group.table, group.native)
    payload = _encode(_Action.ADD_VIEW, 1, 0, 2, mapped)
    for invalid in (payload[:-1], payload + b"x", b"BAD!" + payload[4:], payload[:4] + b"\x02" + payload[5:]):
        with pytest.raises(ValueError):
            group.children[0].handle(invalid, base_comm_handle=123)
    with pytest.raises(ValueError, match="membership"):
        group.children[1].handle(payload, base_comm_handle=123)
    group.parent.unmap_view(mapped)
    with pytest.raises(ValueError, match="retired"):
        group.children[0].handle(_encode(_Action.MAP, 1, 0, 2, mapped), base_comm_handle=123)


def test_collective_preparation_can_wait_for_all_peers(group):
    from threading import Barrier  # noqa: PLC0415

    rendezvous = Barrier(2, timeout=2)

    def prepare(*_args):
        rendezvous.wait()
        return 0x900000

    for chip in group.chips:
        chip._prepare_host_urma.side_effect = prepare
    mapped = group.parent.map_view(group.table, group.native)
    group.parent.validate(mapped)


def test_interruption_after_dispatch_rolls_back_before_propagation(group):
    fanout = group.parent._fanout
    calls = 0

    def interrupted(items, target):
        nonlocal calls
        calls += 1
        fanout(items, target)
        if calls == 1:
            raise KeyboardInterrupt("interrupted after all imports")

    group.parent._fanout = interrupted
    with pytest.raises(KeyboardInterrupt):
        group.parent.map_view(group.table, group.native)
    assert not group.parent.holds_resources
    for chip in group.chips:
        chip._prepare_host_urma.assert_not_called()
        chip._unmap_host_ddr.assert_called_once()


def test_partial_view_publication_revokes_existing_group_safely(group):
    first = group.parent.map_view(group.table, group.native)
    group.state.fail_reply.add((_Action.ADD_VIEW, 1))
    with pytest.raises(RuntimeError, match="ADD_VIEW failed"):
        group.parent.map_view(group.table, group.table.tensor((8,), DataType.INT32))
    assert not group.parent.holds_resources
    for child in group.children:
        with pytest.raises(ValueError, match="stale"):
            child.resolve(first)


def test_dropped_reference_cannot_be_republished_by_replayed_control(group):
    first = group.parent.map_view(group.table, group.native)
    second = group.parent.map_view(group.table, group.table.tensor((8,), DataType.INT32))
    group.parent.unmap_view(first)
    replay = _encode(_Action.ADD_VIEW, 1, 0, 2, first)
    with pytest.raises(ValueError, match="retired"):
        group.children[0].handle(replay, base_comm_handle=123)
    group.children[0].resolve(second)
    with pytest.raises(ValueError, match="stale"):
        group.children[0].resolve(first)
