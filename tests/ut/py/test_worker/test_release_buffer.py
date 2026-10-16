# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""``Worker.release_buffer()``: rejects release while an in-flight run still references the
buffer's identity, and how that identity gets tracked in the first place.

L3+ tracking hooks the four dispatch entry points that hand a Tensor arg to another process,
``Orchestrator.submit_next_level`` / ``.submit_next_level_group`` / ``.submit_sub`` /
``.submit_sub_group``, device-free via the same fake-C++-orchestrator harness
``test_child_addr_guard.py`` already established. L2's direct-chip dispatch
(``_submit_l2_locked``) is a separate run-id namespace that never touches
``_accepted_run_handles``/``_submit_mu`` at all, so it gets its own parallel tracking dict
(``_chip_run_touched_identities``, mirroring ``_chip_runs``' own add/remove lifecycle) and its
own independent check in ``release_buffer``.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from _task_interface import DataType, TensorArgType
from simpler.buffer import AccessMode, create_host_shared_buffer, mint_owner_instance_id, wrap_device_malloc
from simpler.orchestrator import Orchestrator
from simpler.task_interface import CallConfig, TaskArgs
from simpler.worker import RunHandle, Worker, _RunResources

from ._harness import ObservedCondition, fake_chip_l3

_F32 = 0  # DataType.FLOAT32 value
_OID = mint_owner_instance_id()


def _l3() -> Worker:
    return Worker(level=3, num_sub_workers=0, platform="a2a3sim", runtime="tensormap_and_ringbuffer")


def _buffer_args(buf) -> TaskArgs:
    args = TaskArgs()
    args.add_tensor(buf.tensor(shapes=(16,), dtype=DataType.FLOAT32), TensorArgType.INPUT)
    return args


def _fake_orchestrator(w: Worker, monkeypatch: pytest.MonkeyPatch) -> Orchestrator:
    import simpler.orchestrator as orch_mod  # noqa: PLC0415

    def _fake_require_handle(callable_handle, *, kind, expected_namespace=None, **_kw):
        # Per-API metadata, not one blanket tuple: a SUB submit resolves a LOCAL_PYTHON callable
        # and a NEXT_LEVEL submit a LOCAL_CHIP one, so a fake that answered NEXT_LEVEL/LOCAL_CHIP
        # for both would let a SUB test pass over an impossible dispatch contract.
        if kind.startswith("orch.submit_sub"):
            return (b"d" * 32, "SUB", "LOCAL_PYTHON", ())
        return (b"d" * 32, "NEXT_LEVEL", "LOCAL_CHIP", (0,))

    monkeypatch.setattr(orch_mod, "_require_handle", _fake_require_handle)
    return Orchestrator(MagicMock(), w)


class TestTouchedIdentityTracking:
    def test_submit_next_level_records_touched_identity(self, monkeypatch):
        w = _l3()
        o = _fake_orchestrator(w, monkeypatch)
        w._building_run_resources = _RunResources()
        buf = create_host_shared_buffer(64, _OID, buffer_id=1)
        try:
            o.submit_next_level(object(), _buffer_args(buf), None, worker=0)
            assert buf.identity in w._building_run_resources.touched_identities
        finally:
            buf.close()

    def test_submit_next_level_group_records_touched_identity_per_member(self, monkeypatch):
        w = _l3()
        w._chip_shms = [object(), object()]
        o = _fake_orchestrator(w, monkeypatch)
        w._building_run_resources = _RunResources()
        buf_a = create_host_shared_buffer(64, _OID, buffer_id=2)
        buf_b = create_host_shared_buffer(64, _OID, buffer_id=3)
        try:
            o.submit_next_level_group(object(), [_buffer_args(buf_a), _buffer_args(buf_b)], None, workers=[0, 1])
            touched = w._building_run_resources.touched_identities
            assert buf_a.identity in touched
            assert buf_b.identity in touched
        finally:
            buf_a.close()
            buf_b.close()

    def test_submit_sub_records_touched_identity(self, monkeypatch):
        # A SUB task maps the identity into a sub-worker process, so it retains the Buffer for
        # the same reason a NEXT_LEVEL dispatch does: release_buffer() unlinking the backing
        # mid-flight leaves the child faulting on a segment that no longer has a name.
        w = _l3()
        o = _fake_orchestrator(w, monkeypatch)
        w._building_run_resources = _RunResources()
        buf = create_host_shared_buffer(64, _OID, buffer_id=5)
        try:
            o.submit_sub(object(), _buffer_args(buf))
            assert buf.identity in w._building_run_resources.touched_identities
        finally:
            buf.close()

    def test_submit_sub_group_records_touched_identity_per_member(self, monkeypatch):
        w = _l3()
        o = _fake_orchestrator(w, monkeypatch)
        w._building_run_resources = _RunResources()
        buf_a = create_host_shared_buffer(64, _OID, buffer_id=6)
        buf_b = create_host_shared_buffer(64, _OID, buffer_id=7)
        try:
            o.submit_sub_group(object(), [_buffer_args(buf_a), _buffer_args(buf_b)])
            touched = w._building_run_resources.touched_identities
            assert buf_a.identity in touched
            assert buf_b.identity in touched
        finally:
            buf_a.close()
            buf_b.close()

    def test_submit_sub_group_records_nothing_when_a_member_is_rejected(self, monkeypatch):
        # A rejected member means no member is dispatched, so recording the earlier members would
        # refuse a release of buffers no task ever received. Validation is one pass, recording the
        # next -- the same two-pass shape submit_next_level_group already uses.
        w = _l3()
        o = _fake_orchestrator(w, monkeypatch)
        w._building_run_resources = _RunResources()
        buf = create_host_shared_buffer(64, _OID, buffer_id=8)
        device = wrap_device_malloc(0xDEAD0000, 4096, _OID, buffer_id=9, access=AccessMode.READWRITE)
        try:
            with pytest.raises(ValueError, match="DEVICE-space"):
                o.submit_sub_group(object(), [_buffer_args(buf), _buffer_args(device)])
            assert w._building_run_resources.touched_identities == set()
        finally:
            buf.close()

    def test_submit_sub_with_no_args_records_nothing(self, monkeypatch):
        w = _l3()
        o = _fake_orchestrator(w, monkeypatch)
        w._building_run_resources = _RunResources()
        o.submit_sub(object())  # tag-less task -- must not raise on the default TaskArgs()
        assert w._building_run_resources.touched_identities == set()

    def test_no_current_run_is_a_no_op(self, monkeypatch):
        # submit_next_level / submit_sub run fine with no _building_run_resources open (e.g. a
        # direct call outside a run's orchestration callback) -- tracking is opportunistic, not
        # required.
        w = _l3()
        o = _fake_orchestrator(w, monkeypatch)
        assert w._building_run_resources is None
        buf = create_host_shared_buffer(64, _OID, buffer_id=4)
        try:
            o.submit_next_level(object(), _buffer_args(buf), None, worker=0)  # must not raise
            o.submit_sub(object(), _buffer_args(buf))  # must not raise
        finally:
            buf.close()


def _in_flight_handle(w: Worker, identity, *, done: bool) -> RunHandle:
    resources = _RunResources()
    resources.touched_identities.add(identity)
    handle = RunHandle(w, run_id=1, keepalive=())
    handle._resources = resources
    handle._cleanup_published = done
    return handle


def _l3_with_registered_buffer(nbytes: int = 64):
    """An L3 Worker (one fake chip child, so create_buffer's topology gate passes) plus one
    buffer already registered in w._buffers, the way Worker.create_buffer() would leave it."""
    w = _l3()
    w._chip_shms = [object()]
    buf = w._create_buffer_locked(nbytes)
    return w, buf


class TestReleaseBuffer:
    def test_rejects_a_buffer_a_submitted_sub_task_still_names(self, monkeypatch):
        # End to end over the pair, rather than each half alone: submit_sub records into the open
        # run's resources, and release_buffer reads them off the accepted handle. The recording has
        # to land before admission for this to hold at every point after submit_sub returns.
        w, buf = _l3_with_registered_buffer()
        o = _fake_orchestrator(w, monkeypatch)
        handle = RunHandle(w, run_id=1, keepalive=())
        resources = _RunResources()
        handle._resources = resources
        w._building_run_resources = resources
        with w._hierarchical_start_cv:
            w._accepted_run_handles.add(handle)
        o.submit_sub(object(), _buffer_args(buf))
        with pytest.raises(RuntimeError, match="in-flight run"):
            w.release_buffer(buf)
        assert not buf.closed
        handle._cleanup_published = True
        w.release_buffer(buf)
        assert buf.closed

    def test_rejects_while_an_abandoned_run_still_names_the_buffer(self):
        # _publish_abandoned_run sets _cleanup_published and drops the handle from
        # _accepted_run_handles, but the run stays in _abandoned_run_handles until native teardown
        # drains it -- so for an abandoned run that flag no longer means the device is done with the
        # backing, and the accepted-set check alone would let the release through.
        w, buf = _l3_with_registered_buffer()
        handle = _in_flight_handle(w, buf.identity, done=True)
        w._abandoned_run_handles.append(handle)
        with pytest.raises(RuntimeError, match="abandoned run"):
            w.release_buffer(buf)
        assert not buf.closed
        assert w._buffers.get(int(buf.identity.buffer_id)) is buf
        w._abandoned_run_handles.clear()
        w.release_buffer(buf)  # cleanup

    def test_rejects_while_in_flight(self):
        w, buf = _l3_with_registered_buffer()
        w._accepted_run_handles.add(_in_flight_handle(w, buf.identity, done=False))
        with pytest.raises(RuntimeError, match="in-flight run"):
            w.release_buffer(buf)
        assert not buf.closed
        buffer_id = int(buf.identity.buffer_id)
        assert w._buffers.get(buffer_id) is buf  # rejection must not half-apply the registry drop
        w._accepted_run_handles.clear()
        w.release_buffer(buf)  # cleanup

    def test_succeeds_once_run_completed(self):
        w, buf = _l3_with_registered_buffer()
        w._accepted_run_handles.add(_in_flight_handle(w, buf.identity, done=True))
        w.release_buffer(buf)
        assert buf.closed
        assert int(buf.identity.buffer_id) not in w._buffers

    def test_succeeds_with_no_in_flight_runs(self):
        w, buf = _l3_with_registered_buffer()
        w.release_buffer(buf)
        assert buf.closed
        assert int(buf.identity.buffer_id) not in w._buffers

    def test_is_idempotent(self):
        w, buf = _l3_with_registered_buffer()
        w.release_buffer(buf)
        w.release_buffer(buf)  # must not raise
        assert buf.closed

    def test_serializes_with_a_racing_orchestration_callback(self, monkeypatch):
        # _submit_l3_locked adds a run's handle to _accepted_run_handles (worker.py:9910) BEFORE
        # its orchestration callback runs -- the callback is what eventually records
        # touched_identities via submit_next_level. Without taking _submit_mu exclusively (the same
        # serializer graph construction takes,
        # already serializes graph construction, worker.py:9741), release_buffer could observe the
        # handle with an empty touched_identities mid-callback and release a Buffer the callback is
        # about to dispatch a Tensor over. This drives that exact window directly.
        w, buf = _l3_with_registered_buffer()
        resources = _RunResources()
        handle = RunHandle(w, run_id=1, keepalive=())
        handle._resources = resources
        handle._cleanup_published = False

        entered_callback = threading.Event()
        proceed = threading.Event()
        release_returned = threading.Event()

        def fake_orchestration_callback():
            with w._submit_mu.exclusive():
                with w._hierarchical_start_cv:
                    w._accepted_run_handles.add(handle)
                entered_callback.set()
                proceed.wait(timeout=5)  # simulates the callback still building its graph
                # simulates submit_next_level's _record_touched_identities -- _submit_mu held
                # continuously from before the handle is visible until this point, exactly as
                # _submit_l3_locked holds it across the whole orchestration callback.
                resources.touched_identities.add(buf.identity)

        callback_thread = threading.Thread(target=fake_orchestration_callback)
        callback_thread.start()
        assert entered_callback.wait(timeout=5)
        release_waiting = threading.Event()
        monkeypatch.setattr(
            w._submit_mu,
            "_cv",
            ObservedCondition(w._submit_mu._cv, release_waiting),
        )

        def try_release():
            try:
                w.release_buffer(buf)
            except RuntimeError:
                pass
            finally:
                release_returned.set()

        releaser = threading.Thread(target=try_release)
        releaser.start()
        # release_buffer must block behind _submit_mu, not race past it while touched_identities
        # is still empty.
        assert release_waiting.wait(timeout=5)
        assert not release_returned.is_set()

        proceed.set()
        callback_thread.join(timeout=5)
        releaser.join(timeout=5)
        assert release_returned.is_set()
        assert not buf.closed  # correctly rejected once it could finally check -- not released mid-race


class _FakeChipImpl:
    """Stands in for ChipWorker._impl: records the dispatch, hands back a run object whose only
    used surface is .wait() (never exercised by these tests, kept for interface completeness)."""

    def __init__(self):
        self.calls: list[tuple[int, object, object]] = []
        self._execution_environment = None

    def _submit_chip_run_direct(self, callable_id, chip_args, cfg):
        self.calls.append((callable_id, chip_args, cfg))
        return SimpleNamespace(wait=lambda timeout: True)

    def _validate_execution_context(self):
        pass


def _l2_with_registered_buffer(nbytes: int = 64):
    """An L2 Worker (its own chip, no fork needed) plus one buffer already registered in
    w._buffers, the way Worker.create_buffer() would leave it."""
    w = Worker(level=2)
    w._chip_worker = _FakeChipImpl()
    buf = w._create_buffer_locked(nbytes)
    return w, buf


class TestL2TouchedIdentityTracking:
    def test_submit_l2_locked_records_touched_identity(self):
        w, buf = _l2_with_registered_buffer()
        handle = w._submit_l2_locked(3, _buffer_args(buf), CallConfig())
        assert handle._run_id is not None
        assert w._chip_run_touched_identities[handle._run_id] == {buf.identity}

    def test_submit_l2_locked_with_no_args_is_empty(self):
        w, _buf = _l2_with_registered_buffer()
        handle = w._submit_l2_locked(3, None, CallConfig())
        assert handle._run_id is not None
        assert w._chip_run_touched_identities[handle._run_id] == set()

    def test_finalize_run_handle_clears_l2_touched_identities(self):
        w, buf = _l2_with_registered_buffer()
        handle = w._submit_l2_locked(3, _buffer_args(buf), CallConfig())
        run_id = handle._run_id
        assert run_id in w._chip_runs
        assert run_id in w._chip_run_touched_identities
        w._finalize_run_handle(handle, run_id, None)
        assert run_id not in w._chip_runs
        assert run_id not in w._chip_run_touched_identities

    def test_publishes_touched_identities_before_dispatch_not_after(self):
        # _submit_l2_locked must publish _chip_run_touched_identities BEFORE calling
        # _submit_chip_run_direct, not after -- otherwise a release_buffer() landing in the
        # window between dispatch and publication would see no entry for a run already
        # running on the chip. Block dispatch mid-call and confirm release_buffer already
        # sees (and rejects on) the identity at that point, not only after dispatch returns.
        w, buf = _l2_with_registered_buffer()
        entered_dispatch = threading.Event()
        proceed = threading.Event()

        class _BlockingChipImpl(_FakeChipImpl):
            def _submit_chip_run_direct(self, callable_id, chip_args, cfg):
                entered_dispatch.set()
                proceed.wait(timeout=5)
                return super()._submit_chip_run_direct(callable_id, chip_args, cfg)

        w._chip_worker = _BlockingChipImpl()

        submit_thread = threading.Thread(target=lambda: w._submit_l2_locked(3, _buffer_args(buf), CallConfig()))
        submit_thread.start()
        assert entered_dispatch.wait(timeout=5)

        try:
            with pytest.raises(RuntimeError, match="in-flight L2 run"):
                w.release_buffer(buf)
        finally:
            proceed.set()
            submit_thread.join(timeout=5)
        assert not buf.closed

    def test_publishes_touched_identities_before_materializing_not_after(self, monkeypatch):
        # _submit_l2_locked must publish _chip_run_touched_identities BEFORE calling
        # _materialize_l2_args, not just before dispatch -- _materialize_l2_args is what
        # populates self._chip_import_registry, the very cache release_buffer()'s broadcast
        # pops. A release racing the window between materialize and publication would see no
        # entry, pass its check, and pop a mapping a dispatch already in progress just cached.
        # Block materialize mid-call and confirm release_buffer already rejects at that point.
        w, buf = _l2_with_registered_buffer()
        entered_materialize = threading.Event()
        proceed = threading.Event()
        real_materialize = Worker._materialize_l2_args

        def _blocking_materialize(self, args):
            entered_materialize.set()
            proceed.wait(timeout=5)
            return real_materialize(self, args)

        monkeypatch.setattr(Worker, "_materialize_l2_args", _blocking_materialize)

        submit_thread = threading.Thread(target=lambda: w._submit_l2_locked(3, _buffer_args(buf), CallConfig()))
        submit_thread.start()
        assert entered_materialize.wait(timeout=5)

        try:
            with pytest.raises(RuntimeError, match="in-flight L2 run"):
                w.release_buffer(buf)
        finally:
            proceed.set()
            submit_thread.join(timeout=5)
        assert not buf.closed


class TestReleaseBufferL2InFlight:
    def test_rejects_while_an_l2_run_is_in_flight(self):
        w, buf = _l2_with_registered_buffer()
        w._chip_run_touched_identities[1] = {buf.identity}
        with pytest.raises(RuntimeError, match="in-flight L2 run"):
            w.release_buffer(buf)
        assert not buf.closed
        buffer_id = int(buf.identity.buffer_id)
        assert w._buffers.get(buffer_id) is buf  # rejection must not half-apply the registry drop

    def test_succeeds_once_the_l2_run_is_gone(self):
        w, buf = _l2_with_registered_buffer()
        w._chip_run_touched_identities[1] = {buf.identity}
        del w._chip_run_touched_identities[1]
        w.release_buffer(buf)
        assert buf.closed
        assert int(buf.identity.buffer_id) not in w._buffers


class TestReleaseBufferReexport:
    def test_release_drops_the_forwarding_handle_for_the_identity(self):
        # _reexport memoizes a forwarding handle per source backing, and that handle keeps answering
        # to_descriptor() after its backing is gone -- a later forward of the same identity would
        # hand a child a descriptor for a name the owner has already unlinked.
        w, buf = _l3_with_registered_buffer()
        handle = w._reexport(buf.to_descriptor())
        assert w._reexport_by_source[buf.identity] is handle
        w.release_buffer(buf)
        assert buf.identity not in w._reexport_by_source

    def test_close_clears_the_forwarding_handles(self, monkeypatch):
        # The same cache at teardown: it holds an entry per backing ever forwarded, so leaving it
        # out of close()'s cleanup table keeps every one of them alive for the Worker's lifetime.
        with fake_chip_l3(monkeypatch, device_ids=(0,)) as w:
            buf = w.create_buffer(64)
            w._reexport(buf.to_descriptor())
            assert w._reexport_by_source
        assert not w._reexport_by_source


class TestReleaseBufferImportBroadcast:
    def test_broadcasts_to_a_real_forked_chip_child_without_error(self, monkeypatch, capsys):
        # Exercises the real wire round-trip against a live forked chip child (device-free via the
        # fake-chip-L3 harness, but _run_chip_main_loop and its ImportRegistry are production code,
        # not faked): the new _CTRL_IMPORT_RELEASE sub_cmd must reach the child's handle_control and
        # return cleanly, proving the sub_cmd numbering and CanonicalIdentity wire packing agree on
        # both ends rather than just in the mocked unit tests above.
        with fake_chip_l3(monkeypatch, device_ids=(0,)) as w:
            buf = w.create_buffer(64)
            w.release_buffer(buf)
        assert "reported errors" not in capsys.readouterr().err
