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
            _quiesce_host_uses=Mock(return_value=False),
            _map_host_ddr=Mock(side_effect=lambda *_args, r=rank: events.append(("map", r)) or 0x400000 + r * 0x10000),
            _unmap_host_ddr=Mock(side_effect=lambda *_args, r=rank: events.append(("unmap", r))),
            finalize=Mock(side_effect=lambda r=rank: events.append(("finalize", r))),
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
    for chip, child in zip(chips, children):
        chip._unmap_host_ddr.side_effect = None
        chip._validate_execution_context.side_effect = None
        child._uses.clear()
    parent.close_all()
    for buffer in buffers.values():
        buffer.close()
    for registry in imports:
        registry.close()


def accept(group, args, run_id=1, ranks=(0, 1)):
    for rank in ranks:
        group.parent.acquire(run_id, args, rank)
        group.children[rank].acquire(run_id, args)


def retire(group, run_id=1):
    for child in group.children:
        child.release(run_id)
    group.parent.release(run_id)


def test_same_backing_endpoint_local_addresses_and_native_transfers(group):
    args = args_for(group.native)
    args.add_tensor(group.native, transfer=TensorTransfer.NONE)
    args.add_tensor(group.native, transfer=TensorTransfer.H2D)
    with group.table.host_view() as memory:
        memory[16:20] = b"same"
    accept(group, args)
    assert not group.state.requests
    for rank in range(2):
        local = _materialize_chip_task_args(args, group.imports[rank])
        imported = group.imports[rank].require(group.table.identity)
        assert imported.shm.name == group.table.shm.name
        assert bytes(imported.shm.buf[16:20]) == b"same"
        assert local.tensor(0).data == 0x400010 + rank * 0x10000
        assert local.tensor(0).address_space == AddressSpace.DEVICE
        assert local.tensor(1).data == imported.base + 16
        assert local.tensor(2).data == imported.base + 16
    retire(group)


def test_cache_reuse_multiple_views_and_cpu_updates(group):
    for run_id in (1, 2):
        view = group.table.tensor((2,), DataType.INT32, byte_offset=32)
        with group.table.host_view() as memory:
            memory[32] = run_id
        accept(group, args_for(view), run_id)
        for rank, child in enumerate(group.children):
            assert child.resolve(view)[0] == 0x400000 + rank * 0x10000
            assert group.imports[rank].require(group.table.identity).shm.buf[32] == run_id
        retire(group, run_id)
    assert not group.state.requests
    assert all(chip._map_host_ddr.call_count == 1 for chip in group.chips)


def test_multiple_backings_are_independent(group):
    second = create_host_shared_buffer(256, group.table.identity.owner_instance_id, 2)
    group.buffers[second.identity] = second
    args = args_for(group.native)
    args.add_tensor(second.tensor((8,), DataType.INT32), transfer=TensorTransfer.MAP)
    accept(group, args)
    retire(group)
    group.table.close()
    assert not second.closed
    assert all(len(child._records) == 1 for child in group.children)
    second.close()
    assert all(chip._map_host_ddr.call_count == chip._unmap_host_ddr.call_count == 2 for chip in group.chips)
    assert all(not registry._by_identity for registry in group.imports)


@pytest.mark.parametrize("transfer", [TensorTransfer.NONE, TensorTransfer.H2D, TensorTransfer.MAP])
def test_uses_predating_first_map_block_child_release(group, transfer):
    args = args_for(group.native, transfer)
    accept(group, args, 10)
    accept(group, args_for(group.native), 11)
    retire(group, 11)
    # Simulate a parent run fence with no child-side safe-last-use proof.
    group.parent.release(10)
    with pytest.raises(RuntimeError, match="QUIESCE failed"):
        group.parent.close(group.table.identity)
    assert not group.table.closed
    assert all(chip._unmap_host_ddr.call_count == 0 for chip in group.chips)
    retire(group, 10)
    group.parent.close(group.table.identity)


def test_parent_busy_close_does_not_poison_admission(group):
    args = args_for(group.native)
    accept(group, args)
    with pytest.raises(RuntimeError, match="in-flight"):
        group.table.close()
    assert not group.parent._records[group.table.identity].closing
    accept(group, args, 2)
    retire(group)
    retire(group, 2)
    group.table.close()


@pytest.mark.parametrize("transfer", [TensorTransfer.NONE, TensorTransfer.H2D])
def test_native_host_use_blocks_import_release_before_any_map(group, transfer):
    args = args_for(group.native, transfer)
    accept(group, args)
    for rank in range(2):
        _materialize_chip_task_args(args, group.imports[rank])
        with pytest.raises(RuntimeError, match="in-flight"):
            group.imports[rank].unregister(group.table.identity)
    retire(group)
    group.table.close()
    assert all(not registry._by_identity for registry in group.imports)
    assert not any(chip._map_host_ddr.called for chip in group.chips)


def test_cold_mapping_on_one_child_while_other_calls_are_active(group):
    accept(group, args_for(group.native), ranks=(0,))
    group.state.busy = True
    second = create_host_shared_buffer(128, group.table.identity.owner_instance_id, 2)
    group.buffers[second.identity] = second
    accept(group, args_for(second.tensor((4,), DataType.INT32)), 2, ranks=(1,))
    assert group.chips[0]._map_host_ddr.call_count == 1
    assert group.chips[1]._map_host_ddr.call_count == 1
    assert not group.state.requests
    assert second.identity not in group.children[0]._records
    assert group.table.identity not in group.children[1]._records


@pytest.mark.parametrize("phase", [_Action.QUIESCE, _Action.RELEASE])
def test_lost_cleanup_reply_retains_every_attempted_child(group, phase):
    args = args_for(group.native)
    accept(group, args)
    retire(group)
    group.state.fail_reply.add((phase, 1))
    with pytest.raises(RuntimeError, match=f"{phase.name} failed"):
        group.table.close()
    assert not group.table.closed and not group.table.unlinked
    assert group.parent.holds_resources
    with pytest.raises(ValueError, match="closing"):
        group.parent.acquire(2, args, 0)
    group.state.fail_reply.clear()
    group.table.close()
    assert group.table.closed and group.table.unlinked


def test_failed_child_map_retains_partial_registration_and_other_live_use(group):
    args = args_for(group.native)
    accept(group, args, ranks=(0,))
    group.parent.acquire(1, args, 1)
    group.chips[1]._map_host_ddr.side_effect = RuntimeError("register failed")
    group.chips[1]._unmap_host_ddr.side_effect = RuntimeError("unregister failed")
    with pytest.raises(RuntimeError, match="register failed"):
        group.children[1].acquire(1, args)
    assert not group.children[1]._uses
    assert group.children[1].holds_resources
    assert group.children[0]._uses
    with pytest.raises(RuntimeError, match="in-flight"):
        group.table.close()
    retire(group)
    with pytest.raises(RuntimeError, match="RELEASE failed"):
        group.table.close()
    assert not group.table.closed and not group.table.unlinked
    group.chips[1]._unmap_host_ddr.side_effect = None
    group.table.close()


def test_parent_records_cleanup_even_when_child_publication_is_unknown(group):
    args = args_for(group.native)
    group.parent.acquire(1, args, 0)
    group.parent.release(1)
    group.table.close()
    assert [(_decode(payload)[0], rank) for rank, payload in group.state.requests] == [
        (_Action.QUIESCE, 0),
        (_Action.RELEASE, 0),
    ]
    assert not any(chip._map_host_ddr.called for chip in group.chips)
    with pytest.raises(ValueError, match="retired"):
        group.children[0].acquire(2, args)


def test_duplicate_release_is_idempotent_and_retired_backing_refuses_tasks(group):
    args = args_for(group.native)
    accept(group, args)
    retire(group)
    group.parent.close(group.table.identity)
    for rank, payload in group.state.requests:
        group.children[rank].handle(payload)
    assert all(chip._map_host_ddr.call_count == chip._unmap_host_ddr.call_count == 1 for chip in group.chips)
    with pytest.raises(ValueError, match="retired"):
        group.children[0].acquire(2, args)


def test_terminal_recovery_reply_loss_retries_without_repeating_native_drain(group):
    args = args_for(group.native)
    accept(group, args, ranks=(0,))
    group.parent.release(1)
    chip = group.chips[0]
    chip._quiesce_host_uses = Mock(return_value=True)
    group.state.fail_reply.add((_Action.RECOVER, 0))
    with pytest.raises(RuntimeError, match="RECOVER failed"):
        group.parent.close_all(terminal=True)
    assert not group.table.closed
    assert group.parent.holds_resources and group.children[0].holds_resources
    chip._unmap_host_ddr.assert_not_called()
    with pytest.raises(RuntimeError, match="terminally closed"):
        group.children[0].acquire(2, args)
    group.state.fail_reply.clear()
    group.parent.close_all(terminal=True)
    chip._quiesce_host_uses.assert_called_once()
    chip._unmap_host_ddr.assert_called_once()
    group.table.close()


@pytest.mark.parametrize("lost_reply", [False, True])
def test_terminal_finalization_failure_keeps_parent_backing_and_retries(group, lost_reply):
    accept(group, args_for(group.native), ranks=(0,))
    group.parent.release(1)
    chip = group.chips[0]
    chip._quiesce_host_uses = Mock(return_value=True)
    if lost_reply:
        group.state.fail_reply.add((_Action.FINALIZE_RECOVERY, 0))
    else:
        chip.finalize.side_effect = RuntimeError("owner teardown unknown")
    with pytest.raises(RuntimeError, match="FINALIZE_RECOVERY failed"):
        group.parent.close_all(terminal=True)
    assert group.parent.holds_resources and not group.table.closed
    assert not group.children[0].pending_cleanup
    group.state.fail_reply.clear()
    chip.finalize.side_effect = None
    group.parent.close_all(terminal=True)
    assert chip.finalize.call_count == (1 if lost_reply else 2)
    chip._quiesce_host_uses.assert_called_once()
    chip._unmap_host_ddr.assert_called_once()
    group.table.close()


def test_terminal_recovery_unregisters_every_backing_before_owner_finalize(group):
    second = create_host_shared_buffer(128, group.table.identity.owner_instance_id, 2)
    group.buffers[second.identity] = second
    args = args_for(group.native)
    args.add_tensor(second.tensor((4,), DataType.INT32), transfer=TensorTransfer.MAP)
    accept(group, args, ranks=(0,))
    group.parent.release(1)
    chip = group.chips[0]
    chip._quiesce_host_uses = Mock(return_value=True)
    chip._unmap_host_ddr.side_effect = [None, RuntimeError("second unregister failed")]
    with pytest.raises(RuntimeError, match="RELEASE failed"):
        group.parent.close_all(terminal=True)
    chip.finalize.assert_not_called()
    assert len(group.parent._records) == 2
    assert not group.table.closed and not second.closed

    def finalize():
        assert not group.children[0].pending_cleanup
        for identity in (group.table.identity, second.identity):
            with pytest.raises(KeyError):
                group.imports[0].require(identity)

    chip._unmap_host_ddr.side_effect = None
    chip.finalize.side_effect = finalize
    group.parent.close_all(terminal=True)
    chip._quiesce_host_uses.assert_called_once()
    chip.finalize.assert_called_once()
    assert chip._unmap_host_ddr.call_count == 3


def test_idle_mapping_does_not_hide_unrelated_native_recovery(group):
    accept(group, args_for(group.native), ranks=(0,))
    retire(group)
    chip = group.chips[0]
    chip._quiesce_host_uses.return_value = True
    group.parent.close_all(terminal=True)
    chip._quiesce_host_uses.assert_called_once()
    chip._unmap_host_ddr.assert_called_once()
    chip.finalize.assert_called_once()
    assert group.events.index(("unmap", 0)) < group.events.index(("finalize", 0))


def test_resolution_requires_accepted_use(group):
    accept(group, args_for(group.native))
    retire(group)
    for child in group.children:
        with pytest.raises(ValueError, match="accepted use"):
            child.resolve(group.native)


@pytest.mark.parametrize("mutation", ["owner", "rank", "count", "length", "version"])
def test_invalid_cleanup_control_is_rejected(group, mutation):
    descriptor = group.native.buffer
    if mutation == "owner":
        foreign = create_host_shared_buffer(128, mint_owner_instance_id(), 1)
        try:
            descriptor = foreign.to_descriptor()
        finally:
            foreign.close()
    payload = _encode(_Action.QUIESCE, 1 if mutation == "rank" else 0, 3 if mutation == "count" else 2, descriptor)
    if mutation == "length":
        payload = payload[:-1]
    if mutation == "version":
        payload = payload[:4] + b"\x01" + payload[5:]
    with pytest.raises(ValueError):
        group.children[0].handle(payload)
    group.chips[0]._map_host_ddr.assert_not_called()


def test_foreign_owner_and_target_are_rejected_before_mapping(group):
    foreign = create_host_shared_buffer(128, mint_owner_instance_id(), 1)
    try:
        args = args_for(foreign.tensor((4,), DataType.INT32))
        with pytest.raises(ValueError, match="owned"):
            group.parent.acquire(1, args, 0)
        with pytest.raises(ValueError, match="owned"):
            group.children[0].acquire(1, args)
    finally:
        foreign.close()
    with pytest.raises(ValueError, match="outside"):
        group.parent.acquire(1, args_for(group.native), 9)
    group.chips[0]._map_host_ddr.assert_not_called()


def test_stale_child_context_retains_parent_cleanup_debt(group):
    accept(group, args_for(group.native))
    retire(group)
    group.chips[1]._validate_execution_context.side_effect = RuntimeError("context gone")
    with pytest.raises(RuntimeError, match="QUIESCE failed"):
        group.table.close()
    assert not group.table.closed and not group.table.unlinked
    group.chips[0]._unmap_host_ddr.assert_not_called()


def test_live_cpu_export_refuses_before_child_teardown(group):
    accept(group, args_for(group.native))
    retire(group)
    view = group.table.host_view()
    try:
        with pytest.raises(BufferError):
            group.table.close()
        assert all(not chip._unmap_host_ddr.called for chip in group.chips)
    finally:
        view.release()
    group.table.close()


def test_child_cpu_import_close_failure_retries_without_second_unmap(group, monkeypatch):
    accept(group, args_for(group.native))
    retire(group)
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


def test_failed_import_retains_cleanup_record_without_native_unmap(group, monkeypatch):
    args = args_for(group.native)
    group.parent.acquire(1, args, 0)
    monkeypatch.setattr(group.imports[0], "materialize", Mock(side_effect=ValueError("bad import")))
    with pytest.raises(ValueError, match="bad import"):
        group.children[0].acquire(1, args)
    group.parent.release(1)
    group.table.close()
    group.chips[0]._unmap_host_ddr.assert_not_called()


@pytest.mark.parametrize("outcome", ["accepted", "refused", "unknown", "materialization-failed"])
def test_actual_child_submit_maps_before_native_admission_and_preserves_proofs(group, outcome):
    import ast  # noqa: PLC0415
    import inspect  # noqa: PLC0415

    from simpler import worker as worker_module  # noqa: PLC0415

    tree = ast.parse(inspect.getsource(worker_module._run_chip_main_loop))
    helper = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "submit_run")
    child = group.children[0]
    args = args_for(group.native)
    group.parent.acquire(1, args, 0)
    key = (3, 1, 0, 1, 1, 0, 0, 1)
    staged = SimpleNamespace(
        identity=key,
        pending_args=args,
        cid=1,
        config=SimpleNamespace(output_prefix=""),
        provenance=SimpleNamespace(_asdict=lambda: {}),
        entry_addr=0,
        entry_buf=None,
        activated=False,
    )

    def submit(_cid, chip_args, *_args, **_kwargs):
        assert group.chips[0]._map_host_ddr.call_count == 1
        assert group.table.identity in child._uses[key]
        assert chip_args.tensor(0).data == 0x400010
        if outcome == "refused":
            raise worker_module.RunPreparationRefused("refused safely")
        if outcome == "unknown":
            raise RuntimeError("native admission unknown")
        return SimpleNamespace(preparation_disposition=worker_module._NATIVE_PREPARED)

    native = Mock(side_effect=submit)
    namespace = dict(vars(worker_module))
    namespace.update(
        _StagedRun=SimpleNamespace,
        import_registry=group.imports[0],
        registry={1: SimpleNamespace(sig_count=1, sig=lambda _index: worker_module.ArgDirection.IN)},
        cw=SimpleNamespace(_submit_chip_run_materialized=native),
        _mailbox_store_i32=Mock(),
        _write_entry_error=Mock(),
    )
    if outcome == "materialization-failed":
        namespace["_materialize_chip_task_args"] = Mock(side_effect=ValueError("materialization failed"))
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "<actual-child-submit>", "exec"), namespace)
    if outcome == "accepted":
        namespace["submit_run"](staged)
    else:
        with pytest.raises((RuntimeError, ValueError)):
            namespace["submit_run"](staged)
    assert (key in child._uses) == (outcome in ("accepted", "unknown"))
    assert child.holds_resources
    if outcome == "materialization-failed":
        native.assert_not_called()
    else:
        native.assert_called_once()
    if outcome == "unknown":
        # The caller has no ChipRun to observe, but terminal cleanup reaches
        # the native owner and preserves backing until it proves last use.
        group.chips[0]._quiesce_host_uses = Mock(return_value=True)
        group.parent.release(1)
        group.parent.close_all(terminal=True)
        group.chips[0]._quiesce_host_uses.assert_called_once()
        assert not child._uses
