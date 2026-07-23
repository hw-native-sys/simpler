# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""G2 — device-address guard ②: kind4 (child_memory) pointer provenance.

Device-free tests that pin the guard's contract: a child device pointer is
tracked by its exact ``(worker_id, ptr)`` provenance so it is never freed,
copied, or dispatched to the wrong next-level worker, and a stale (freed)
pointer is rejected before reuse. Covers every real entry — ``Worker.malloc``
(L2), ``orch.malloc/copy/free`` (L3, the path that bypasses ``Worker.malloc``),
``submit_next_level`` / ``submit_next_level_group`` dispatch, and CommDomain
window pointers — plus the explicit boundary that strict ABA is NOT covered.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import simpler.orchestrator as orch_mod
from _task_interface import DataType, Tensor, TensorArgType
from simpler.orchestrator import Orchestrator
from simpler.task_interface import TaskArgs
from simpler.worker import Worker, _ChildProvEntry, _Lifecycle


def _l3() -> Worker:
    return Worker(level=3, num_sub_workers=0, platform="a2a3sim", runtime="tensormap_and_ringbuffer")


def _child_args(ptr: int, *, n: int = 16) -> TaskArgs:
    args = TaskArgs()
    args.add_tensor(Tensor.make(ptr, (n,), DataType.FLOAT32, child_memory=True), TensorArgType.OUTPUT_EXISTING)
    return args


def _record_malloc(w: Worker, worker_id: int, ptr: int) -> None:
    with w._child_prov_lock:
        w._child_prov_record_malloc(worker_id, ptr)


# ----------------------------------------------------------------------------
# Provenance table — typed exact (worker_id, ptr) entries
# ----------------------------------------------------------------------------


class TestProvenanceTable:
    def test_malloc_base_is_live_then_freed(self):
        w = _l3()
        with w._child_prov_lock:
            w._child_prov_record_malloc(0, 0x1000)
            w._child_prov_require_live(0, 0x1000, api="copy_to")  # no raise
            w._child_prov_require_malloc_base(0, 0x1000, api="free")  # no raise
            w._child_prov_clear_malloc(0, 0x1000)
            with pytest.raises(ValueError, match="not a live allocation"):
                w._child_prov_require_live(0, 0x1000, api="copy_to")

    def test_free_of_unknown_pointer_rejected(self):
        w = _l3()
        with w._child_prov_lock, pytest.raises(ValueError, match="not a live malloc base"):
            w._child_prov_require_malloc_base(0, 0xDEAD, api="free")

    def test_double_free_stale_before_reuse_rejected(self):
        w = _l3()
        with w._child_prov_lock:
            w._child_prov_record_malloc(0, 0x1000)
            w._child_prov_clear_malloc(0, 0x1000)
            with pytest.raises(ValueError, match="already-freed/stale"):
                w._child_prov_require_malloc_base(0, 0x1000, api="free")

    def test_interior_pointer_is_not_a_live_entry(self):
        w = _l3()
        with w._child_prov_lock:
            w._child_prov_record_malloc(0, 0x1000)
            # A pointer physically inside the allocation has no exact entry.
            with pytest.raises(ValueError, match="interior"):
                w._child_prov_require_live(0, 0x1008, api="copy_to")

    def test_same_numeric_va_on_two_workers_is_independent(self):
        # A raw device VA is not globally unique; the composite key keeps two
        # chips' identical numeric addresses independent.
        w = _l3()
        with w._child_prov_lock:
            w._child_prov_record_malloc(0, 0x4000)
            w._child_prov_record_malloc(1, 0x4000)
            w._child_prov_clear_malloc(0, 0x4000)
            with pytest.raises(ValueError, match="not a live allocation"):
                w._child_prov_require_live(0, 0x4000, api="copy_to")
            w._child_prov_require_live(1, 0x4000, api="copy_to")  # survives

    def test_domain_pointer_is_not_freeable_but_is_dispatchable(self):
        w = _l3()
        with w._child_prov_lock:
            w._child_prov_record_domain(0, 0x8000, allocation_id=7)
            # usable for copy / dispatch
            w._child_prov_require_live(0, 0x8000, api="copy_to")
            # but not free-able — domains are revoked by release, not free()
            with pytest.raises(ValueError, match="CommDomain buffer"):
                w._child_prov_require_malloc_base(0, 0x8000, api="free")
            w._child_prov_drop_domain(7)
            with pytest.raises(ValueError, match="not a live allocation"):
                w._child_prov_require_live(0, 0x8000, api="copy_to")

    def test_malloc_and_domain_alias_same_pointer(self):
        # A malloc base and a domain buffer can alias the same (worker, ptr).
        # Clearing the malloc role leaves it live via the domain; only dropping
        # the domain removes it.
        w = _l3()
        with w._child_prov_lock:
            w._child_prov_record_malloc(0, 0x9000)
            w._child_prov_record_domain(0, 0x9000, allocation_id=3)
            w._child_prov_clear_malloc(0, 0x9000)
            w._child_prov_require_live(0, 0x9000, api="copy_to")  # still live via domain
            w._child_prov_drop_domain(3)
            with pytest.raises(ValueError, match="not a live allocation"):
                w._child_prov_require_live(0, 0x9000, api="copy_to")

    def test_empty_entry_is_not_live_fail_closed(self):
        # A role-less entry (e.g. one left by an interrupted revoke) must never be
        # treated as live — every check is fail-closed on the roles, not on the
        # key's mere presence.
        w = _l3()
        w._child_alloc_prov[(0, 0x1000)] = _ChildProvEntry()  # both roles empty
        with w._child_prov_lock:
            with pytest.raises(ValueError, match="not a live allocation"):
                w._child_prov_require_live(0, 0x1000, api="copy_to")
            with pytest.raises(ValueError, match="not a live malloc base"):
                w._child_prov_require_malloc_base(0, 0x1000, api="free")
            with pytest.raises(ValueError, match="not a live allocation on target worker 0"):
                w._child_prov_check_dispatch([(0x1000, 0)], 0, api="submit_next_level")

    def test_drop_last_role_deletes_entry_no_empty_state(self):
        # Dropping the last role deletes the key outright — it never leaves a
        # role-less entry behind (which a later live check could not tell from a
        # live one without the fail-closed guard).
        w = _l3()
        with w._child_prov_lock:
            w._child_prov_record_domain(0, 0x5000, allocation_id=1)
            w._child_prov_drop_domain(1)
            assert (0, 0x5000) not in w._child_alloc_prov
            w._child_prov_record_malloc(0, 0x6000)
            w._child_prov_clear_malloc(0, 0x6000)
            assert (0, 0x6000) not in w._child_alloc_prov

    def test_strict_aba_is_explicitly_not_covered(self):
        # Documentation test: after free + re-malloc of the same numeric VA, the
        # (worker, ptr) becomes live again and an old handle is indistinguishable
        # from the new allocation. Strict ABA is deferred to P1 generation
        # handles; this guard only catches stale-BEFORE-reuse.
        w = _l3()
        with w._child_prov_lock:
            w._child_prov_record_malloc(0, 0x1000)
            w._child_prov_clear_malloc(0, 0x1000)
            w._child_prov_record_malloc(0, 0x1000)  # device handed back the same VA
            w._child_prov_require_live(0, 0x1000, api="copy_to")  # NOT rejected — by design


# ----------------------------------------------------------------------------
# Target resolution + dispatch check
# ----------------------------------------------------------------------------


class TestDispatchResolution:
    def test_unique_target_and_live_passes(self):
        w = _l3()
        _record_malloc(w, 0, 0x1000)
        with w._child_prov_lock:
            w._child_prov_check_dispatch([(0x1000, 0)], 0, api="submit_next_level")

    def test_unique_target_but_wrong_worker_rejected(self):
        w = _l3()
        _record_malloc(w, 0, 0x1000)  # lives on worker 0
        with w._child_prov_lock, pytest.raises(ValueError, match="not a live allocation on target worker 1"):
            w._child_prov_check_dispatch([(0x1000, 0)], 1, api="submit_next_level")


# ----------------------------------------------------------------------------
# Orchestrator malloc / free / copy — the L3 choke (also the orch.* bypass)
# ----------------------------------------------------------------------------


class TestOrchestratorMemoryOps:
    def test_malloc_records_then_free_clears(self):
        w = _l3()
        fake = MagicMock()
        fake.malloc.return_value = 0x1000
        o = Orchestrator(fake, w)
        ptr = o.malloc(0, 64)
        assert ptr == 0x1000
        assert (0, 0x1000) in w._child_alloc_prov
        o.free(0, ptr)
        fake.free.assert_called_once_with(0, 0x1000)
        assert (0, 0x1000) not in w._child_alloc_prov

    def test_free_wrong_worker_rejected_without_native_free(self):
        w = _l3()
        fake = MagicMock()
        fake.malloc.return_value = 0x1000
        o = Orchestrator(fake, w)
        o.malloc(1, 64)  # allocated on worker 1
        with pytest.raises(ValueError, match="not a live malloc base"):
            o.free(0, 0x1000)  # freed on worker 0
        fake.free.assert_not_called()

    def test_copy_to_requires_live_device_dst(self):
        w = _l3()
        fake = MagicMock()
        fake.malloc.return_value = 0x2000
        o = Orchestrator(fake, w)
        with pytest.raises(ValueError, match="not a live allocation"):
            o.copy_to(0, 0x2000, 0xABCD, 64)
        fake.copy_to.assert_not_called()
        o.malloc(0, 64)
        o.copy_to(0, 0x2000, 0xABCD, 64)
        fake.copy_to.assert_called_once()

    def test_copy_from_requires_live_device_src(self):
        w = _l3()
        fake = MagicMock()
        fake.malloc.return_value = 0x3000
        o = Orchestrator(fake, w)
        with pytest.raises(ValueError, match="not a live allocation"):
            o.copy_from(0, 0xABCD, 0x3000, 64)
        fake.copy_from.assert_not_called()

    def test_isolated_orchestrator_has_no_guard(self):
        # An Orchestrator constructed without a Worker back-ref (test isolation)
        # must not attempt provenance tracking.
        fake = MagicMock()
        fake.malloc.return_value = 0x1000
        o = Orchestrator(fake, None)
        assert o.malloc(0, 64) == 0x1000
        o.free(0, 0x1000)  # no raise, no table


# ----------------------------------------------------------------------------
# submit_next_level / group dispatch guard
# ----------------------------------------------------------------------------


@pytest.fixture
def _fake_handle(monkeypatch):
    """Patch _require_handle so submit_* can run device-free with a chosen
    eligible set; returns a setter for the eligible worker ids."""
    state = {"eligible": (0,)}

    def _fake(callable_handle, **_kwargs):
        return (b"d" * 32, "NEXT_LEVEL", "LOCAL_CHIP", state["eligible"])

    monkeypatch.setattr(orch_mod, "_require_handle", _fake)
    return state


class TestSubmitDispatchGuard:
    def test_child_arg_to_correct_worker_passes(self, _fake_handle):
        w = _l3()
        _record_malloc(w, 0, 0x1000)
        fake = MagicMock()
        o = Orchestrator(fake, w)
        o.submit_next_level(object(), _child_args(0x1000), None, worker=0)
        fake.submit_next_level.assert_called_once()

    def test_child_arg_to_wrong_worker_rejected(self, _fake_handle):
        w = _l3()
        w._chip_shms = [object(), object()]
        _record_malloc(w, 0, 0x1000)
        fake = MagicMock()
        o = Orchestrator(fake, w)
        with pytest.raises(ValueError, match="not a live allocation on target worker 1"):
            o.submit_next_level(object(), _child_args(0x1000), None, worker=1)
        fake.submit_next_level.assert_not_called()

    def test_host_only_args_are_not_guarded(self, _fake_handle):
        # A submit with no child_memory tensor never touches provenance.
        w = _l3()
        fake = MagicMock()
        o = Orchestrator(fake, w)
        args = TaskArgs()
        args.add_tensor(Tensor.make(0, (16,), DataType.FLOAT32, child_memory=False), TensorArgType.INPUT)
        o.submit_next_level(object(), args, None, worker=0)
        fake.submit_next_level.assert_called_once()

    def test_group_member_child_arg_wrong_worker_rejected(self, _fake_handle):
        w = _l3()
        w._chip_shms = [object(), object()]
        _record_malloc(w, 0, 0x1000)
        fake = MagicMock()
        o = Orchestrator(fake, w)
        # member 0 carries the child ptr live on worker 0, but is pinned to worker 1
        with pytest.raises(ValueError, match="not a live allocation on target worker 1"):
            o.submit_next_level_group(object(), [_child_args(0x1000), TaskArgs()], None, workers=[1, 0])
        fake.submit_next_level_group.assert_not_called()

    def test_group_member_child_arg_correct_worker_passes(self, _fake_handle):
        w = _l3()
        w._chip_shms = [object(), object()]
        _record_malloc(w, 1, 0x1000)
        fake = MagicMock()
        o = Orchestrator(fake, w)
        o.submit_next_level_group(object(), [TaskArgs(), _child_args(0x1000)], None, workers=[0, 1])
        fake.submit_next_level_group.assert_called_once()

    def test_group_child_member_rejects_mismatched_workers_length(self, _fake_handle):
        # A non-empty workers list must be one-per-member; a short list must NOT
        # be silently padded (that would bypass the C++ length check).
        w = _l3()
        w._chip_shms = [object(), object()]
        _record_malloc(w, 0, 0x1000)
        fake = MagicMock()
        o = Orchestrator(fake, w)
        with pytest.raises(ValueError, match="workers length must match"):
            o.submit_next_level_group(object(), [_child_args(0x1000), TaskArgs()], None, workers=[0])
        fake.submit_next_level_group.assert_not_called()

    def test_local_callable_rejects_remote_worker_target(self, _fake_handle):
        # A LOCAL_CHIP callable pinned to a remote worker id would enqueue on the
        # remote endpoint, whose manifest lacks the local digest -> async unknown
        # hashid. The Python guard rejects it up front; a local chip id passes.
        w = _l3()
        w._chip_shms = [object()]
        w._remote_worker_ids = [7]
        fake = MagicMock()
        o = Orchestrator(fake, w)
        with pytest.raises(ValueError, match="remote NEXT_LEVEL worker"):
            o.submit_next_level(object(), TaskArgs(), None, worker=7)
        fake.submit_next_level.assert_not_called()
        o.submit_next_level(object(), TaskArgs(), None, worker=0)
        fake.submit_next_level.assert_called_once()

    def test_local_group_rejects_remote_worker_target(self, _fake_handle):
        w = _l3()
        w._chip_shms = [object()]
        w._remote_worker_ids = [7]
        fake = MagicMock()
        o = Orchestrator(fake, w)
        with pytest.raises(ValueError, match="remote NEXT_LEVEL worker"):
            o.submit_next_level_group(object(), [TaskArgs(), TaskArgs()], None, workers=[0, 7])
        fake.submit_next_level_group.assert_not_called()

    def test_domain_pointer_dispatch_to_owner_then_rejected_after_release(self, _fake_handle):
        # CommDomain window pointers enter provenance so they dispatch as kind4
        # to their owning chip; a release revokes them.
        w = _l3()
        w._chip_shms = [object(), object()]
        with w._child_prov_lock:
            w._child_prov_record_domain(0, 0x5000, allocation_id=42)
        fake = MagicMock()
        o = Orchestrator(fake, w)
        o.submit_next_level(object(), _child_args(0x5000), None, worker=0)
        fake.submit_next_level.assert_called_once()
        # wrong chip is rejected
        with pytest.raises(ValueError, match="target worker 1"):
            o.submit_next_level(object(), _child_args(0x5000), None, worker=1)
        # after release the pointer is dead everywhere
        with w._child_prov_lock:
            w._child_prov_drop_domain(42)
        with pytest.raises(ValueError, match="not a live allocation"):
            o.submit_next_level(object(), _child_args(0x5000), None, worker=0)


# ----------------------------------------------------------------------------
# L2 Worker.malloc/free/copy path (direct to the single chip)
# ----------------------------------------------------------------------------


class TestL2WorkerPath:
    def _l2(self) -> tuple[Worker, MagicMock]:
        w = Worker(level=2, platform="a2a3sim", runtime="tensormap_and_ringbuffer", device_id=0)
        chip = MagicMock()
        chip.malloc.return_value = 0x2000
        w._chip_worker = chip
        w._lifecycle = _Lifecycle.READY
        return w, chip

    def test_l2_malloc_records_and_free_clears(self):
        w, chip = self._l2()
        ptr = w.malloc(64)
        assert (0, ptr) in w._child_alloc_prov
        w.free(ptr)
        chip.free.assert_called_once_with(0x2000)
        assert (0, ptr) not in w._child_alloc_prov

    def test_l2_free_stale_rejected_without_native_free(self):
        w, chip = self._l2()
        w.malloc(64)
        w.free(0x2000)
        chip.free.reset_mock()
        with pytest.raises(ValueError, match="already-freed/stale"):
            w.free(0x2000)
        chip.free.assert_not_called()

    def test_l2_free_revokes_before_native_free(self):
        # L2 mirrors the L3 commit barrier: revoke before the native free.
        w, chip = self._l2()
        w.malloc(64)
        seen = {}
        chip.free.side_effect = lambda p: seen.__setitem__("live_at_native", (0, 0x2000) in w._child_alloc_prov)
        w.free(0x2000)
        assert seen["live_at_native"] is False

    def test_l2_copy_to_requires_live_dst(self):
        w, chip = self._l2()
        with pytest.raises(ValueError, match="not a live allocation"):
            w.copy_to(0x2000, 0xABCD, 64)
        chip.copy_to.assert_not_called()
        w.malloc(64)
        w.copy_to(0x2000, 0xABCD, 64)
        chip.copy_to.assert_called_once()


# ----------------------------------------------------------------------------
# Provenance transaction failures (record after alloc success; free revokes
# before the native free — safety-first, terminal-leak on failure)
# ----------------------------------------------------------------------------


class TestProvenanceTransactions:
    def test_orch_malloc_native_error_records_nothing(self):
        # Provenance is recorded only after the backend malloc succeeds.
        w = _l3()
        fake = MagicMock()
        fake.malloc.side_effect = RuntimeError("device OOM")
        o = Orchestrator(fake, w)
        with pytest.raises(RuntimeError, match="device OOM"):
            o.malloc(0, 64)
        assert w._child_alloc_prov == {}

    def test_orch_free_revokes_before_native_free(self):
        # Safety-first commit barrier: provenance is revoked BEFORE the native
        # free, so an async unwind after a successful free cannot leave a freed
        # address live.
        w = _l3()
        fake = MagicMock()
        fake.malloc.return_value = 0x1000
        o = Orchestrator(fake, w)
        o.malloc(0, 64)
        seen = {}
        fake.free.side_effect = lambda wid, p: seen.__setitem__("live_at_native", (0, 0x1000) in w._child_alloc_prov)
        o.free(0, 0x1000)
        assert seen["live_at_native"] is False  # already revoked when native free runs

    def test_orch_free_native_error_revokes_provenance_safe_first(self):
        # A native free that fails becomes a terminal leak — provenance is
        # revoked, never re-authorized. No retry (the address is no longer a
        # live malloc base).
        w = _l3()
        fake = MagicMock()
        fake.malloc.return_value = 0x1000
        o = Orchestrator(fake, w)
        o.malloc(0, 64)
        fake.free.side_effect = RuntimeError("free failed")
        with pytest.raises(RuntimeError, match="free failed"):
            o.free(0, 0x1000)
        assert (0, 0x1000) not in w._child_alloc_prov  # revoked (terminal leak)
        fake.free.side_effect = None
        with pytest.raises(ValueError, match="not a live malloc base"):
            o.free(0, 0x1000)

    def test_free_holds_lock_across_native_free(self):
        # Deterministic mutual-exclusion check: the native free runs while
        # _child_prov_lock is held, so a concurrent free/copy/dispatch cannot
        # interleave with a half-completed free.
        w = _l3()
        fake = MagicMock()
        fake.malloc.return_value = 0x1000
        o = Orchestrator(fake, w)
        o.malloc(0, 64)
        held = {}

        def _sf(wid, p):
            acquired = w._child_prov_lock.acquire(blocking=False)
            held["locked_during_native"] = not acquired
            if acquired:
                w._child_prov_lock.release()

        fake.free.side_effect = _sf
        o.free(0, 0x1000)
        assert held["locked_during_native"] is True

    def test_capture_refs_after_provenance_analysis(self, _fake_handle, monkeypatch):
        # blocker: the kind4 provenance analysis must run BEFORE remote slot refs
        # are captured, so an analysis failure cannot strand captured refs outside
        # the rollback try (deferring a remote free forever).
        w = _l3()
        o = Orchestrator(MagicMock(), w)
        monkeypatch.setattr(w, "_child_ptrs_in_args", MagicMock(side_effect=RuntimeError("boom")))
        cap = MagicMock(return_value=[])
        monkeypatch.setattr(w, "_capture_remote_sidecar_refs", cap)
        with pytest.raises(RuntimeError, match="boom"):
            o.submit_next_level(object(), _child_args(0x1000), None, worker=0)
        cap.assert_not_called()

    def test_l2_malloc_native_error_records_nothing(self):
        w = Worker(level=2, platform="a2a3sim", runtime="tensormap_and_ringbuffer", device_id=0)
        chip = MagicMock()
        chip.malloc.side_effect = RuntimeError("device OOM")
        w._chip_worker = chip
        w._lifecycle = _Lifecycle.READY
        with pytest.raises(RuntimeError, match="device OOM"):
            w.malloc(64)
        assert w._child_alloc_prov == {}


# ----------------------------------------------------------------------------
# CommDomain physical release — revoke before backend free (commit barrier)
# ----------------------------------------------------------------------------


class TestDomainReleaseOrdering:
    def _domain_worker(self):
        """A level-3 Worker with a recorded domain and a native worker present."""
        w = _l3()
        w._worker = MagicMock()  # non-None so _release_domain_now proceeds
        with w._child_prov_lock:
            w._child_prov_record_domain(0, 0x5000, allocation_id=9)
            w._child_prov_record_domain(1, 0x6000, allocation_id=9)
        handle = SimpleNamespace(name="d", workers=(0, 1), allocation_id=9)
        return w, handle

    def test_release_revokes_provenance_before_backend_free(self, monkeypatch):
        w, handle = self._domain_worker()
        seen_live_at_dispatch = {}

        def _fake_dispatch(**kwargs):
            # At physical-free time the pointers must already be revoked.
            seen_live_at_dispatch["still_live"] = (0, 0x5000) in w._child_alloc_prov

        monkeypatch.setattr(w, "_dispatch_control_domain", _fake_dispatch)
        w._release_domain_now(handle)  # type: ignore[arg-type]
        assert seen_live_at_dispatch["still_live"] is False
        assert (0, 0x5000) not in w._child_alloc_prov
        assert (1, 0x6000) not in w._child_alloc_prov

    def test_release_vs_dispatch_rejects_during_native_release(self, monkeypatch):
        # Deterministic domain-release-vs-dispatch: by the time the backend free
        # runs, the pointer is already revoked, so a dispatch that lands during
        # the native release is rejected — no freed-but-live window.
        w, handle = self._domain_worker()
        outcome = {}

        def _fake_dispatch(**kwargs):
            try:
                with w._child_prov_lock:
                    w._child_prov_check_dispatch([(0x5000, 0)], 0, api="submit_next_level")
                outcome["dispatch"] = "allowed"
            except ValueError:
                outcome["dispatch"] = "rejected"

        monkeypatch.setattr(w, "_dispatch_control_domain", _fake_dispatch)
        w._release_domain_now(handle)  # type: ignore[arg-type]
        assert outcome["dispatch"] == "rejected"

    def test_release_backend_failure_leaves_provenance_dropped(self, monkeypatch):
        # A partial/failed backend release must not restore the pointers to live
        # (a leak is safe; a use-after-free is not).
        w, handle = self._domain_worker()

        def _boom(**kwargs):
            raise RuntimeError("release failed on one chip")

        monkeypatch.setattr(w, "_dispatch_control_domain", _boom)
        with pytest.raises(RuntimeError, match="release failed"):
            w._release_domain_now(handle)  # type: ignore[arg-type]
        assert (0, 0x5000) not in w._child_alloc_prov
        assert (1, 0x6000) not in w._child_alloc_prov
