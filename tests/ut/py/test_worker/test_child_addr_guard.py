# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""G2 — device-allocation guard ②: which allocation an operand is allowed to name.

Device-free tests that pin the guard's contract: a device allocation is named by the canonical
identity of the ``Buffer`` the Worker minted for it, and every decision is read off that registered
Buffer — which chip it lives on (``owner_worker_id``), how far it extends (``nbytes``), what may be
done to it (``access``), and whether ``free`` or a domain release reclaims it (``backend_kind``).
An address is the *result* of resolving an identity, never the key: a handle rebuilt from an address
the caller happens to know names nothing, and a freed identity stays dead even after the allocator
hands its address to a new allocation. Covers every real entry — ``Worker.malloc`` (L2),
``Worker.alloc_child_tensor`` / ``copy_to`` / ``copy_from`` / ``free`` (L3, which ``orch.*``
delegates to), ``submit_next_level`` / ``submit_next_level_group`` dispatch, and CommDomain carved
buffers.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import simpler.orchestrator as orch_mod
from _task_interface import DataType, TensorArgType
from simpler.buffer import (
    AccessMode,
    AddressSpace,
    BackendKind,
    Buffer,
    BufferCapability,
    create_host_shared_buffer,
    host_ptr_nbytes,
    mint_owner_instance_id,
    wrap_device_malloc,
    wrap_fork_inherited,
    wrap_vmm_window,
)
from simpler.orchestrator import Orchestrator
from simpler.task_interface import TaskArgs
from simpler.worker import Worker, _Lifecycle

_F32 = 0  # DataType.FLOAT32 value
_OID = mint_owner_instance_id()
# The host end of an L3 copy is a POSIX-shm Buffer: the forked child reaches it by name.
_HOSTSRC = create_host_shared_buffer(64, _OID, buffer_id=0xF0)


@pytest.fixture(scope="module", autouse=True)
def _release_module_host_buffer():
    yield
    _HOSTSRC.close()


def _l3() -> Worker:
    return Worker(level=3, num_sub_workers=0, platform="a2a3sim", runtime="tensormap_and_ringbuffer")


def _l3_ready(malloc_ret: int = 0x1000, *, chips: int = 2) -> tuple[Worker, MagicMock]:
    """An uninitialized L3 Worker forced READY with a mocked native worker, so the leased device-mem
    ops run in isolation. Returns ``(worker, native_worker_mock)``; the mock's ``malloc`` returns
    ``malloc_ret`` (the fabricated device ptr)."""
    w = _l3()
    w._lifecycle = _Lifecycle.READY
    w._chip_shms = [None] * chips  # type: ignore[list-item]
    nw = MagicMock()
    nw.malloc.return_value = malloc_ret
    w._worker = nw
    return w, nw


@contextlib.contextmanager
def _host_buf(nbytes: int, *, buffer_id: int = 0xF1):
    """A POSIX-shm host ``Buffer`` of ``nbytes``, unlinked on exit."""
    buf = create_host_shared_buffer(nbytes, _OID, buffer_id=buffer_id)
    try:
        yield buf
    finally:
        buf.close()


def _dev_handle(ptr: int, *, wid: int = 0, nbytes: int = 64, oid: bytes = _OID) -> Buffer:
    """A DEVICE_MALLOC handle whose identity is derived from ``ptr``.

    ``buffer_id=ptr`` makes a handle built here compare equal to the one ``_record_malloc``
    registered for the same address, so a test can name a registered allocation without holding the
    object the fixture created.
    """
    # `owner_worker_id` is not part of an identity, so the worker is folded into `buffer_id` here:
    # production's `_next_buffer_id()` gives every allocation a distinct id whatever chip it lands
    # on, and a fixture keyed on the address alone could not express two chips allocating at one.
    return wrap_device_malloc(ptr, nbytes, oid, buffer_id=(wid << 48) | ptr, owner_worker_id=wid)


def _device_ptr(desc) -> int:
    """The device pointer a DEVICE_MALLOC / VMM_WINDOW descriptor carries in its backend body."""
    return int.from_bytes(desc.body, "little")


def _child_args(handle: Buffer, *, n: int = 16) -> TaskArgs:
    """``TaskArgs`` naming ``handle`` as an output — one device operand for the dispatch guard.

    The tensor embeds the handle's descriptor, so the identity the guard resolves is the one the
    Worker registered; naming an allocation by an address it happens to sit at is what the guard
    exists to refuse.
    """
    args = TaskArgs()
    args.add_tensor(handle.tensor(shapes=(n,), dtype=_F32), TensorArgType.OUTPUT_EXISTING)
    return args


def _record_malloc(w: Worker, worker_id: int, ptr: int, size: int = 64) -> Buffer:
    """Register a live device allocation at ``ptr`` and return its handle."""
    handle = _dev_handle(ptr, wid=worker_id, nbytes=size, oid=w._owner_instance_id)
    with w._child_prov_lock:
        w._record_device_alloc(handle)
    return handle


def _is_live(w: Worker, ptr: int, wid: int = 0) -> bool:
    """Whether a live allocation sits at ``(wid, ptr)``.

    Production mints each identity from ``_next_buffer_id()``, so a handle rebuilt here does not
    name the allocation the Worker registered; the address is what a test knows about it.
    """
    return any(int(h.base) == ptr and int(h.owner_worker_id) == wid for h in w._child_alloc.values())


# ----------------------------------------------------------------------------
# Device-allocation table — keyed by the identity that names one allocation
# ----------------------------------------------------------------------------


class TestDeviceAllocationTable:
    def test_a_registered_allocation_is_live_until_dropped(self):
        w = _l3()
        handle = _record_malloc(w, 0, 0x1000)
        assert w.require(handle.identity, BufferCapability.COPY_TO, 64, api="copy_to") is handle
        with w._child_prov_lock:
            w._drop_device_alloc(handle.identity)
        with pytest.raises(ValueError, match="not a live device allocation"):
            w.require(handle.identity, BufferCapability.COPY_TO, 64, api="copy_to")

    def test_an_identity_never_registered_is_refused(self):
        w = _l3()
        with pytest.raises(ValueError, match="not a live device allocation"):
            w.require(_dev_handle(0xDEAD).identity, BufferCapability.COPY_TO, 8, api="copy_to")

    def test_extent_comes_from_the_registered_handle(self):
        # A copy may cover any prefix of the allocation, and nothing past it -- and the bound is the
        # registered handle's, so a caller cannot widen it by presenting a larger handle.
        w = _l3()
        handle = _record_malloc(w, 0, 0x1000, size=64)
        w.require(handle.identity, BufferCapability.COPY_TO, 16, api="copy_to")
        w.require(handle.identity, BufferCapability.COPY_TO, 64, api="copy_to")  # exact fill
        with pytest.raises(ValueError, match="overruns"):
            w.require(handle.identity, BufferCapability.COPY_TO, 65, api="copy_to")
        with pytest.raises(ValueError, match="non-negative"):
            w.require(handle.identity, BufferCapability.COPY_TO, -1, api="copy_to")

    def test_two_workers_hold_separate_allocations_at_one_address(self):
        # Two chips can hand back the same address; the identities differ, so dropping one leaves
        # the other live.
        w = _l3()
        a = _record_malloc(w, 0, 0x4000)
        b = _record_malloc(w, 1, 0x4000)
        assert a.identity != b.identity
        with w._child_prov_lock:
            w._drop_device_alloc(a.identity)
        with pytest.raises(ValueError, match="not a live device allocation"):
            w.require(a.identity, BufferCapability.COPY_TO, 64, api="copy_to")
        assert w.require(b.identity, BufferCapability.COPY_TO, 64, api="copy_to") is b

    def test_only_delegated_copy_rights_are_conferred(self):
        # A Worker never dereferences a device allocation it hands out -- the in-process ChipWorker
        # (L2) or the forked chip child (L3+) performs every copy. So the mechanism is
        # OWNER_DELEGATED_COPY and the direct-dereference rights are absent, whatever the
        # allocation's AccessMode says: a DIRECT_MAP here would name a load no caller on this side
        # can issue.
        w = _l3()
        handle = _record_malloc(w, 0, 0x1000)
        for capability in (BufferCapability.DIRECT_LOAD, BufferCapability.DIRECT_STORE):
            with pytest.raises(ValueError, match=capability.value):
                w.require(handle.identity, capability, 8, api="copy_to")
        w.require(handle.identity, BufferCapability.COPY_TO, 8, api="copy_to")
        w.require(handle.identity, BufferCapability.COPY_FROM, 8, api="copy_from")

    def test_access_mode_narrows_what_the_allocation_grants(self):
        w = _l3()
        ro = wrap_device_malloc(0x9000, 64, w._owner_instance_id, buffer_id=0x9000, access=AccessMode.READ)
        with w._child_prov_lock:
            w._record_device_alloc(ro)
        w.require(ro.identity, BufferCapability.COPY_FROM, 8, api="copy_from")
        with pytest.raises(ValueError, match="COPY_TO"):
            w.require(ro.identity, BufferCapability.COPY_TO, 8, api="copy_to")


class TestDomainAllocations:
    def _domain_buffer(self, w: Worker, ptr: int, *, nbytes: int = 64) -> Buffer:
        return wrap_vmm_window(ptr, nbytes, w._owner_instance_id, buffer_id=ptr, owner_worker_id=0)

    def test_a_domain_release_revokes_every_buffer_it_minted(self):
        w = _l3()
        first, second = self._domain_buffer(w, 0x8000), self._domain_buffer(w, 0x8040)
        with w._child_prov_lock:
            for buf in (first, second):
                w._record_device_alloc(buf, domain_allocation_id=7)
        w.require(first.identity, BufferCapability.COPY_TO, 64, api="copy_to")
        with w._child_prov_lock:
            w._drop_domain_allocs(7)
        for buf in (first, second):
            with pytest.raises(ValueError, match="not a live device allocation"):
                w.require(buf.identity, BufferCapability.COPY_TO, 64, api="copy_to")

    def test_a_domain_buffer_is_not_freeable(self):
        # Domain memory belongs to the comm backend, which reclaims the whole window collectively.
        w = _l3()
        buf = self._domain_buffer(w, 0x8000)
        with w._child_prov_lock:
            w._record_device_alloc(buf, domain_allocation_id=7)
            with pytest.raises(ValueError, match="released with its domain"):
                w._require_freeable(buf, api="free")

    def test_a_carved_buffer_and_a_sibling_at_one_base_keep_their_own_extents(self):
        # A buffer carved at the window's offset 0 shares its address with the window, and a
        # pointer-keyed table had to widen one entry to cover both. Identities keep them apart, so
        # the small buffer's own extent is what bounds a copy into it.
        w = _l3()
        small = self._domain_buffer(w, 0x8000, nbytes=4)
        with w._child_prov_lock:
            w._record_device_alloc(small, domain_allocation_id=7)
        w.require(small.identity, BufferCapability.COPY_TO, 4, api="copy_to")
        with pytest.raises(ValueError, match="overruns"):
            w.require(small.identity, BufferCapability.COPY_TO, 1024, api="copy_to")


# ----------------------------------------------------------------------------
# Target resolution + dispatch check
# ----------------------------------------------------------------------------


class TestDispatchResolution:
    def test_unique_target_and_live_passes(self):
        w = _l3()
        handle = _record_malloc(w, 0, 0x1000)
        w._child_prov_check_dispatch([(handle.identity, 0)], 0, api="submit_next_level")

    def test_unique_target_but_wrong_worker_rejected(self):
        w = _l3()
        handle = _record_malloc(w, 0, 0x1000)  # lives on worker 0
        with pytest.raises(ValueError, match="not a live allocation on target worker 1"):
            w._child_prov_check_dispatch([(handle.identity, 0)], 1, api="submit_next_level")


# ----------------------------------------------------------------------------
# Worker device memory — alloc_child_tensor / free / copy on a next-level worker
# (the Worker is the sole allocator; orch.* are thin delegates)
# ----------------------------------------------------------------------------


class TestChildDeviceMemoryOps:
    def test_alloc_child_records_then_free_clears(self):
        w, nw = _l3_ready(0x1000)
        h = w.alloc_child_tensor(0, (64,), DataType.UINT8)
        assert h.base == 0x1000
        assert _is_live(w, 0x1000, 0)
        w.free(h)
        nw.free.assert_called_once_with(0, 0x1000)
        assert not _is_live(w, 0x1000, 0)

    def test_free_wrong_worker_rejected_without_native_free(self):
        w, nw = _l3_ready(0x1000)
        w.alloc_child_tensor(1, (64,), DataType.UINT8)  # allocated on worker 1
        with pytest.raises(ValueError, match="not a live device allocation"):
            w.free(_dev_handle(0x1000, wid=0))  # freed on worker 0
        nw.free.assert_not_called()

    def test_copy_to_rejects_a_stale_handle_whose_pointer_was_reused(self):
        # free + re-malloc hands back the same numeric VA, so the (worker, ptr) key is live again and
        # the freed handle is indistinguishable from the new allocation *by pointer*. Its identity is
        # not: buffer_id never repeats, so authorizing the copy on identity refuses the freed handle
        # while the new one still works. This is the ABA case the pointer guard cannot see.
        w, nw = _l3_ready(0x1000)
        stale = w.alloc_child_tensor(0, (16,), DataType.FLOAT32)
        w.free(stale)
        fresh = w.alloc_child_tensor(0, (16,), DataType.FLOAT32)
        assert int(fresh.base) == int(stale.base)  # the device handed back the same VA
        assert fresh.identity != stale.identity
        nw.copy_to.reset_mock()
        with _host_buf(64) as src, pytest.raises(ValueError, match="not a live device allocation"):
            w.copy_to(stale, src)
        nw.copy_to.assert_not_called()
        with _host_buf(64) as src:
            w.copy_to(fresh, src)
        nw.copy_to.assert_called_once()

    def test_copy_to_requires_live_device_dst(self):
        w, nw = _l3_ready(0x2000)
        with pytest.raises(ValueError, match="not a live device allocation"):
            w.copy_to(_dev_handle(0x2000, wid=0), _HOSTSRC)
        nw.copy_to.assert_not_called()
        h = w.alloc_child_tensor(0, (64,), DataType.UINT8)
        w.copy_to(h, _HOSTSRC)
        nw.copy_to.assert_called_once()

    def test_copy_to_fills_any_prefix_of_the_allocation(self):
        # A partial update is expressed as a shorter transfer into the allocation the handle names
        # (issue #1537); the copy lands at its base.
        w, nw = _l3_ready(0x2000)
        h = w.alloc_child_tensor(0, (64,), DataType.UINT8)
        with _host_buf(16) as src:
            w.copy_to(h, src)
        nw.copy_to.assert_called_once()
        # copy_to(worker_id, dst_descriptor, src_descriptor, nbytes)
        assert _device_ptr(nw.copy_to.call_args.args[1]) == 0x2000
        assert nw.copy_to.call_args.args[3] == 16

    def test_copy_to_rejects_an_address_inside_a_live_allocation(self):
        # ``base + offset`` is a different identity, which nothing was registered under, so it names
        # no allocation at all rather than resolving back to the one that contains it. A sub-range is
        # named by the allocation plus `dst_offset`, never by a handle built part-way into it.
        w, nw = _l3_ready(0x2000)
        w.alloc_child_tensor(0, (64,), DataType.UINT8)
        for probe in (0x2020, 0x2040):  # inside the allocation, and one past its end
            with _host_buf(2) as src, pytest.raises(ValueError, match="not a live device allocation"):
                w.copy_to(_dev_handle(probe, wid=0), src)
        nw.copy_to.assert_not_called()

    def test_copy_to_offset_names_a_subrange_of_the_allocation(self):
        # The offset travels to the child, which applies it to the base its own ImportRegistry
        # resolved -- the parent never advances an address on the child's behalf, because the
        # parent's mapping of that backing means nothing there.
        w, nw = _l3_ready(0x2000)
        h = w.alloc_child_tensor(0, (64,), DataType.UINT8)
        with _host_buf(64) as src:
            w.copy_to(h, src, dst_offset=32, src_offset=8, nbytes=16)
        (wid, dst_desc, _src_desc, size, dst_off, src_off) = nw.copy_to.call_args.args
        assert (wid, size, dst_off, src_off) == (0, 16, 32, 8)
        assert _device_ptr(dst_desc) == 0x2000  # the allocation's own base, not base + 32

    def test_copy_to_offset_is_bounded_by_the_registered_allocation(self):
        # The bound is the registered handle's extent, and it covers offset + length together, so an
        # offset cannot walk a legal-looking length past the end.
        w, nw = _l3_ready(0x2000)
        h = w.alloc_child_tensor(0, (64,), DataType.UINT8)
        with _host_buf(64) as src:
            w.copy_to(h, src, dst_offset=48, nbytes=16)  # exactly reaches the end
            with pytest.raises(ValueError, match="exceeds the 64-byte backing"):
                w.copy_to(h, src, dst_offset=48, nbytes=17)
            with pytest.raises(ValueError, match="exceeds the 64-byte backing"):
                w.copy_to(h, src, dst_offset=65, nbytes=0)
            with pytest.raises(ValueError, match="dst_offset must be non-negative"):
                w.copy_to(h, src, dst_offset=-1, nbytes=1)
        assert nw.copy_to.call_count == 1

    def test_copy_to_length_defaults_to_the_rest_of_the_host_side(self):
        # The length has always come from the host side; an offset only moves where that side
        # starts, so the no-nbytes call still means "the whole host backing".
        w, nw = _l3_ready(0x2000)
        h = w.alloc_child_tensor(0, (64,), DataType.UINT8)
        with _host_buf(64) as src:
            w.copy_to(h, src, src_offset=16)
        assert nw.copy_to.call_args.args[3] == 48  # 64 - 16
        with _host_buf(8) as src, pytest.raises(ValueError, match="src range .* exceeds the 8-byte backing"):
            w.copy_to(h, src, src_offset=9)

    def test_copy_to_a_domain_buffer_is_bounded_by_its_own_extent(self):
        # Two buffers carved from one window can share an address -- the one at offset 0 has the
        # window base. Each is its own identity with its own extent, so a copy into either is
        # bounded by that buffer alone: the big one is not narrowed by its small sibling, and the
        # small one is not widened by the big one.
        w, nw = _l3_ready()
        oid = w._owner_instance_id
        big = wrap_vmm_window(0x4000, 2 << 20, oid, buffer_id=0x4000, owner_worker_id=0)
        small = wrap_vmm_window(0x4000, 4, oid, buffer_id=0x4001, owner_worker_id=0)
        with w._child_prov_lock:
            for buf in (big, small):
                w._record_device_alloc(buf, domain_allocation_id=7)
        with _host_buf(1 << 20) as chunk:
            w.copy_to(big, chunk)  # 1 MiB into a 2 MiB buffer
        nw.copy_to.assert_called_once()
        assert nw.copy_to.call_args.args[3] == 1 << 20
        with _host_buf(8) as oversized, pytest.raises(ValueError, match="exceeds the 4-byte backing"):
            w.copy_to(small, oversized)
        assert nw.copy_to.call_count == 1

    def test_copy_from_requires_live_device_src(self):
        w, nw = _l3_ready(0x3000)
        with pytest.raises(ValueError, match="not a live device allocation"):
            w.copy_from(_HOSTSRC, _dev_handle(0x3000, wid=0))
        nw.copy_from.assert_not_called()

    def test_orch_delegates_to_worker(self):
        # orch.alloc_child_tensor / free are thin wrappers over the bound Worker.
        w, nw = _l3_ready(0x1000)
        o = Orchestrator(MagicMock(), w)
        h = o.alloc_child_tensor(0, (64,), DataType.UINT8)
        assert _is_live(w, 0x1000, 0)
        o.free(h)
        nw.free.assert_called_once_with(0, 0x1000)
        assert not _is_live(w, 0x1000, 0)

    def test_orch_without_worker_rejects_memory_ops(self):
        # A worker-less Orchestrator can't allocate/free/copy — the impl lives on the Worker.
        o = Orchestrator(MagicMock(), None)
        with pytest.raises(RuntimeError, match="requires a Worker context"):
            o.free(_dev_handle(0x1000))


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
        h = _record_malloc(w, 0, 0x1000)
        fake = MagicMock()
        o = Orchestrator(fake, w)
        o.submit_next_level(object(), _child_args(h), None, worker=0)
        fake.submit_next_level.assert_called_once()

    def test_child_arg_to_wrong_worker_rejected(self, _fake_handle):
        w = _l3()
        w._chip_shms = [object(), object()]
        h = _record_malloc(w, 0, 0x1000)  # lives on worker 0
        fake = MagicMock()
        o = Orchestrator(fake, w)
        with pytest.raises(ValueError, match="not a live allocation on target worker 1"):
            o.submit_next_level(object(), _child_args(h), None, worker=1)
        fake.submit_next_level.assert_not_called()

    def test_host_only_args_are_not_guarded(self, _fake_handle):
        # A submit with no device (DEVICE_MALLOC) ref never touches provenance.
        w = _l3()
        fake = MagicMock()
        o = Orchestrator(fake, w)
        args = TaskArgs()
        host = wrap_fork_inherited(0x9000, 64, _OID, buffer_id=0x9000)
        args.add_tensor(host.tensor(shapes=(16,), dtype=_F32), TensorArgType.INPUT)
        o.submit_next_level(object(), args, None, worker=0)
        fake.submit_next_level.assert_called_once()

    def test_group_member_child_arg_wrong_worker_rejected(self, _fake_handle):
        w = _l3()
        w._chip_shms = [object(), object()]
        h = _record_malloc(w, 0, 0x1000)
        fake = MagicMock()
        o = Orchestrator(fake, w)
        # member 0 carries the allocation live on worker 0, but is pinned to worker 1
        with pytest.raises(ValueError, match="not a live allocation on target worker 1"):
            o.submit_next_level_group(object(), [_child_args(h), TaskArgs()], None, workers=[1, 0])
        fake.submit_next_level_group.assert_not_called()

    def test_group_member_child_arg_correct_worker_passes(self, _fake_handle):
        w = _l3()
        w._chip_shms = [object(), object()]
        h = _record_malloc(w, 1, 0x1000)
        fake = MagicMock()
        o = Orchestrator(fake, w)
        o.submit_next_level_group(object(), [TaskArgs(), _child_args(h)], None, workers=[0, 1])
        fake.submit_next_level_group.assert_called_once()

    def test_group_child_member_rejects_mismatched_workers_length(self, _fake_handle):
        # A non-empty workers list must be one-per-member; a short list must NOT
        # be silently padded (that would bypass the C++ length check).
        w = _l3()
        w._chip_shms = [object(), object()]
        h = _record_malloc(w, 0, 0x1000)
        fake = MagicMock()
        o = Orchestrator(fake, w)
        with pytest.raises(ValueError, match="workers length must match"):
            o.submit_next_level_group(object(), [_child_args(h), TaskArgs()], None, workers=[0])
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

    def test_domain_buffer_dispatch_to_owner_then_rejected_after_release(self, _fake_handle):
        # A CommDomain carved buffer is a device allocation like any other: it dispatches to its
        # owning chip and nowhere else, and its domain's release revokes it.
        w = _l3()
        w._chip_shms = [object(), object()]
        buf = wrap_vmm_window(0x5000, 64, w._owner_instance_id, buffer_id=0x5000, owner_worker_id=0)
        with w._child_prov_lock:
            w._record_device_alloc(buf, domain_allocation_id=42)
        fake = MagicMock()
        o = Orchestrator(fake, w)
        o.submit_next_level(object(), _child_args(buf), None, worker=0)
        fake.submit_next_level.assert_called_once()
        # wrong chip is rejected
        with pytest.raises(ValueError, match="target worker 1"):
            o.submit_next_level(object(), _child_args(buf), None, worker=1)
        # after release the buffer is dead everywhere
        with w._child_prov_lock:
            w._drop_domain_allocs(42)
        with pytest.raises(ValueError, match="not a live allocation"):
            o.submit_next_level(object(), _child_args(buf), None, worker=0)


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
        h = w.malloc(64)
        assert h.base == 0x2000
        assert _is_live(w, 0x2000, 0)
        w.free(h)
        chip.free.assert_called_once_with(0x2000)
        assert not _is_live(w, 0x2000, 0)

    def test_l2_free_stale_rejected_without_native_free(self):
        w, chip = self._l2()
        h = w.malloc(64)
        w.free(h)
        chip.free.reset_mock()
        with pytest.raises(ValueError, match="not a live device allocation"):
            w.free(h)
        chip.free.assert_not_called()

    def test_l2_free_revokes_before_native_free(self):
        # L2 mirrors the L3 commit barrier: revoke before the native free.
        w, chip = self._l2()
        h = w.malloc(64)
        seen = {}
        chip.free.side_effect = lambda p: seen.__setitem__("live_at_native", _is_live(w, 0x2000, 0))
        w.free(h)
        assert seen["live_at_native"] is False

    def test_l2_copy_to_requires_live_dst(self):
        w, chip = self._l2()
        with pytest.raises(ValueError, match="not a live device allocation"):
            w.copy_to(_dev_handle(0x2000, wid=0), _HOSTSRC)
        chip.copy_to.assert_not_called()
        h = w.malloc(64)
        w.copy_to(h, _HOSTSRC)
        chip.copy_to.assert_called_once()

    def test_l2_copy_to_fills_a_prefix_but_not_an_interior_address(self):
        # The single chip resolves an identity the same way L3 does: a shorter transfer into the
        # allocation is a partial update (issue #1537), an interior address names nothing.
        w, chip = self._l2()
        h = w.malloc(64)  # chip.malloc returns 0x2000
        w.copy_to(h, bytearray(16))
        chip.copy_to.assert_called_once()
        assert chip.copy_to.call_args.args[0] == 0x2000
        assert chip.copy_to.call_args.args[2] == 16
        with pytest.raises(ValueError, match="not a live device allocation"):
            w.copy_to(_dev_handle(0x2020, wid=0), bytearray(16))
        chip.copy_to.assert_called_once()

    def test_l2_copy_to_applies_both_offsets_in_process(self):
        # No fork here, so the Worker advances both bases itself -- but through the same helper the
        # chip child uses at L3+, so the two tiers cannot disagree on what an offset means.
        w, chip = self._l2()
        h = w.malloc(64)  # chip.malloc returns 0x2000
        src = bytearray(64)
        src_addr = host_ptr_nbytes(src)[0]
        w.copy_to(h, src, dst_offset=32, src_offset=8, nbytes=16)
        (dptr, sptr, size) = chip.copy_to.call_args.args
        assert (dptr, sptr, size) == (0x2000 + 32, src_addr + 8, 16)

    def test_l2_copy_from_applies_both_offsets_in_process(self):
        w, chip = self._l2()
        h = w.malloc(64)
        dst = bytearray(64)
        dst_addr = host_ptr_nbytes(dst)[0]
        w.copy_from(dst, h, dst_offset=8, src_offset=32, nbytes=16)
        (dptr, sptr, size) = chip.copy_from.call_args.args
        assert (dptr, sptr, size) == (dst_addr + 8, 0x2000 + 32, 16)

    def test_l2_free_of_interior_pointer_rejected(self):
        w, chip = self._l2()
        w.malloc(64)
        with pytest.raises(ValueError, match="not a live device allocation"):
            w.free(_dev_handle(0x2020, wid=0))
        chip.free.assert_not_called()


# ----------------------------------------------------------------------------
# Provenance transaction failures (record after alloc success; free revokes
# before the native free — safety-first, terminal-leak on failure)
# ----------------------------------------------------------------------------


class TestProvenanceTransactions:
    def test_alloc_child_native_error_records_nothing(self):
        # Provenance is recorded only after the backend malloc succeeds.
        w, nw = _l3_ready()
        nw.malloc.side_effect = RuntimeError("device OOM")
        with pytest.raises(RuntimeError, match="device OOM"):
            w.alloc_child_tensor(0, (64,), DataType.UINT8)
        assert w._child_alloc == {}

    def test_free_revokes_before_native_free(self):
        # Safety-first commit barrier: provenance is revoked BEFORE the native
        # free, so an async unwind after a successful free cannot leave a freed
        # address live.
        w, nw = _l3_ready(0x1000)
        h = w.alloc_child_tensor(0, (64,), DataType.UINT8)
        seen = {}
        nw.free.side_effect = lambda wid, p: seen.__setitem__("live_at_native", _is_live(w, 0x1000, 0))
        w.free(h)
        assert seen["live_at_native"] is False  # already revoked when native free runs

    def test_free_native_error_revokes_provenance_safe_first(self):
        # A native free that fails becomes a terminal leak — provenance is
        # revoked, never re-authorized. No retry (the address is no longer a
        # live malloc base).
        w, nw = _l3_ready(0x1000)
        h = w.alloc_child_tensor(0, (64,), DataType.UINT8)
        nw.free.side_effect = RuntimeError("free failed")
        with pytest.raises(RuntimeError, match="free failed"):
            w.free(h)
        assert not _is_live(w, 0x1000, 0)  # revoked (terminal leak)
        nw.free.side_effect = None
        with pytest.raises(ValueError, match="not a live device allocation"):
            w.free(h)

    def test_free_holds_lock_across_native_free(self):
        # Deterministic mutual-exclusion check, in the narrower form this exclusion
        # now takes: the native free runs under *that worker's* lock, so a
        # concurrent free/copy/dispatch on the same chip still cannot interleave
        # with a half-completed free — but `_child_prov_lock` is released, so a
        # slow free on one chip no longer blocks provenance work for another.
        # Safe because the revoke commits first: a concurrent dispatch reading
        # the table under `_child_prov_lock` finds this identity already gone,
        # or is about a different chip entirely.
        w, nw = _l3_ready(0x1000)
        h = w.alloc_child_tensor(0, (64,), DataType.UINT8)
        seen = {}

        def _sf(wid, p):
            worker_lock = w._child_prov_worker_lock(wid)
            free_to_take = worker_lock.acquire(blocking=False)
            seen["worker_lock_held"] = not free_to_take
            if free_to_take:
                worker_lock.release()
            shared_free = w._child_prov_lock.acquire(blocking=False)
            seen["shared_lock_released"] = shared_free
            if shared_free:
                w._child_prov_lock.release()
            seen["revoke_committed"] = not _is_live(w, p, wid)

        nw.free.side_effect = _sf
        w.free(h)
        assert seen["worker_lock_held"] is True
        assert seen["shared_lock_released"] is True
        assert seen["revoke_committed"] is True

    def test_remote_refs_adopted_after_provenance_analysis(self, _fake_handle, monkeypatch):
        # blocker: the device-arg analysis must run BEFORE remote slot refs are adopted, so an
        # analysis failure cannot strand adopted refs outside the rollback try (deferring a remote
        # free forever).
        w = _l3()
        o = Orchestrator(MagicMock(), w)
        monkeypatch.setattr(w, "_device_identities_in_args", MagicMock(side_effect=RuntimeError("boom")))
        adopt = MagicMock()
        monkeypatch.setattr(w, "_adopt_remote_sidecar_refs", adopt)
        with pytest.raises(RuntimeError, match="boom"):
            o.submit_next_level(object(), _child_args(_dev_handle(0x1000)), None, worker=0)
        adopt.assert_not_called()

    def test_l2_malloc_native_error_records_nothing(self):
        w = Worker(level=2, platform="a2a3sim", runtime="tensormap_and_ringbuffer", device_id=0)
        chip = MagicMock()
        chip.malloc.side_effect = RuntimeError("device OOM")
        w._chip_worker = chip
        w._lifecycle = _Lifecycle.READY
        with pytest.raises(RuntimeError, match="device OOM"):
            w.malloc(64)
        assert w._child_alloc == {}


# ----------------------------------------------------------------------------
# CommDomain physical release — revoke before backend free (commit barrier)
# ----------------------------------------------------------------------------


class TestDomainReleaseOrdering:
    def _domain_worker(self) -> tuple[Worker, SimpleNamespace, tuple[Buffer, Buffer]]:
        """A level-3 Worker holding one domain allocation's two carved buffers, plus a native worker."""
        w = _l3()
        w._worker = MagicMock()  # non-None so _release_domain_now proceeds
        bufs = (
            wrap_vmm_window(0x5000, 64, w._owner_instance_id, buffer_id=0x5000, owner_worker_id=0),
            wrap_vmm_window(0x6000, 64, w._owner_instance_id, buffer_id=0x6000, owner_worker_id=1),
        )
        with w._child_prov_lock:
            for buf in bufs:
                w._record_device_alloc(buf, domain_allocation_id=9)
        handle = SimpleNamespace(
            name="d",
            workers=(0, 1),
            allocation_id=9,
            _domain_size=2,
            _domain_ranks={0: 0, 1: 1},
        )
        return w, handle, bufs

    def test_release_revokes_provenance_before_backend_free(self, monkeypatch):
        w, handle, _bufs = self._domain_worker()
        seen_live_at_dispatch = {}

        def _fake_dispatch(**kwargs):
            # At physical-free time the buffers must already be revoked.
            seen_live_at_dispatch["still_live"] = _is_live(w, 0x5000, 0)

        monkeypatch.setattr(w, "_dispatch_control_domain", _fake_dispatch)
        w._release_domain_now(handle)  # type: ignore[arg-type]
        assert seen_live_at_dispatch["still_live"] is False
        assert not _is_live(w, 0x5000, 0)
        assert not _is_live(w, 0x6000, 1)

    def test_release_vs_dispatch_rejects_during_native_release(self, monkeypatch):
        # Deterministic domain-release-vs-dispatch: by the time the backend free
        # runs, the buffer is already revoked, so a dispatch that lands during
        # the native release is rejected — no freed-but-live window.
        w, handle, bufs = self._domain_worker()
        outcome = {}

        def _fake_dispatch(**kwargs):
            # `_child_prov_check_dispatch` takes `_child_prov_lock` itself, and that lock is not
            # re-entrant, so this runs outside it.
            try:
                w._child_prov_check_dispatch([(bufs[0].identity, 0)], 0, api="submit_next_level")
                outcome["dispatch"] = "allowed"
            except ValueError:
                outcome["dispatch"] = "rejected"

        monkeypatch.setattr(w, "_dispatch_control_domain", _fake_dispatch)
        w._release_domain_now(handle)  # type: ignore[arg-type]
        assert outcome["dispatch"] == "rejected"

    def test_release_backend_failure_leaves_provenance_dropped(self, monkeypatch):
        # A partial/failed backend release must not restore the buffers to live
        # (a leak is safe; a use-after-free is not).
        w, handle, _bufs = self._domain_worker()

        def _boom(**kwargs):
            raise RuntimeError("release failed on one chip")

        monkeypatch.setattr(w, "_dispatch_control_domain", _boom)
        with pytest.raises(RuntimeError, match="release failed"):
            w._release_domain_now(handle)  # type: ignore[arg-type]
        assert not _is_live(w, 0x5000, 0)
        assert not _is_live(w, 0x6000, 1)


# ----------------------------------------------------------------------------
# copy_to / copy_from handle validation
#
# The transfer length comes from the *host* object, so without these checks a host buffer larger
# than the device backing writes past it, and a READ-only backing accepts a write.
# ----------------------------------------------------------------------------


class TestCopyHandleValidation:
    def test_rejects_host_handle(self):
        w, nw = _l3_ready(0x1000)
        host = create_host_shared_buffer(64, _OID, buffer_id=1)
        try:
            with pytest.raises(ValueError, match="expected a DEVICE handle"):
                w.copy_to(host, _HOSTSRC)
            nw.copy_to.assert_not_called()
        finally:
            host.close()

    def test_rejects_transfer_larger_than_the_backing(self):
        w, nw = _l3_ready(0x2000)
        w.alloc_child_tensor(0, (32,), DataType.UINT8)  # a 32-byte backing
        with pytest.raises(ValueError, match="exceeds the 32-byte backing"):
            w.copy_to(_dev_handle(0x2000, wid=0, nbytes=32), _HOSTSRC)  # _HOSTSRC is 64 B
        nw.copy_to.assert_not_called()

    def test_rejects_write_to_a_read_only_backing(self):
        w, nw = _l3_ready(0x2000)
        ro = wrap_device_malloc(
            0x2000, 64, w._owner_instance_id, buffer_id=0x2000, owner_worker_id=0, access=AccessMode.READ
        )
        with w._child_prov_lock:
            w._record_device_alloc(ro)
        with pytest.raises(ValueError, match="needs WRITE"):
            w.copy_to(ro, _HOSTSRC)
        nw.copy_to.assert_not_called()
        w.copy_from(_HOSTSRC, ro)  # reading it is fine
        nw.copy_from.assert_called_once()

    def test_l3_copy_sends_both_ends_as_handles(self):
        # The owner's mapped address means nothing in the child, which resolves each descriptor
        # through its own ImportRegistry — so an L3 copy sends handles, never an address.
        w, nw = _l3_ready(0x2000)
        h = w.alloc_child_tensor(0, (64,), DataType.UINT8)
        w.copy_to(h, _HOSTSRC)
        (wid, dst_desc, src_desc, size, dst_off, src_off) = nw.copy_to.call_args.args
        assert (wid, size, dst_off, src_off) == (0, _HOSTSRC.nbytes, 0, 0)
        assert _device_ptr(dst_desc) == 0x2000
        assert src_desc.identity == _HOSTSRC.identity
        assert src_desc.address_space == AddressSpace.HOST


class TestCopyHandleTransport:
    """Both ends of a copy travel as descriptors the child resolves for itself."""

    def test_shm_handle_src_carries_its_own_backing(self):
        w, nw = _l3_ready(0x2000)
        h = w.alloc_child_tensor(0, (64,), DataType.UINT8)
        host = create_host_shared_buffer(64, _OID, buffer_id=1)
        host_shm = host.shm
        assert host_shm is not None
        try:
            w.copy_to(h, host)
            (_wid, dst_desc, src_desc, size, _dst_off, _src_off) = nw.copy_to.call_args.args
            assert src_desc.body.decode() == host_shm.name  # the caller's own shm, not a duplicate
            assert src_desc.backend_kind == BackendKind.POSIX_SHM
            assert (_device_ptr(dst_desc), size) == (0x2000, 64)
        finally:
            host.close()

    def test_rejects_raw_host_memory_at_l3(self):
        # Raw host memory has only an address, and an L3 copy runs in a forked child where that
        # address means nothing. There is no fallback that copies it somewhere reachable.
        w, nw = _l3_ready(0x2000)
        h = w.alloc_child_tensor(0, (64,), DataType.UINT8)
        with pytest.raises(TypeError, match="must be a Buffer from create_buffer"):
            w.copy_to(h, bytearray(64))
        with pytest.raises(TypeError, match="must be a Buffer from create_buffer"):
            w.copy_from(bytearray(64), h)
        nw.copy_to.assert_not_called()
        nw.copy_from.assert_not_called()

    def test_rejects_a_device_handle_on_the_host_side(self):
        w, _nw = _l3_ready(0x2000)
        w.alloc_child_tensor(0, (64,), DataType.UINT8)
        with pytest.raises(ValueError, match="host side must be a HOST handle"):
            w.copy_to(_dev_handle(0x2000, wid=0), _dev_handle(0x2000, wid=0))

    def test_rejects_writing_into_a_read_only_host_handle(self):
        w, _nw = _l3_ready(0x3000)
        w.alloc_child_tensor(0, (64,), DataType.UINT8)
        ro = create_host_shared_buffer(64, _OID, buffer_id=2, access=AccessMode.READ)
        try:
            with pytest.raises(ValueError, match="host backing grants READ"):
                w.copy_from(ro, _dev_handle(0x3000, wid=0))
        finally:
            ro.close()
