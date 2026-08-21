# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for simpler.buffer: identity/descriptor construction + create/import round trip.

The three wire types are the C++ structs of buffer.h bound directly, so what is pinned here is the
Python-visible contract over them — construction rejects what `validate_buffer_descriptor` rejects,
and equality and hashing ignore wire padding. There is deliberately no `pack`/`unpack`: no Python
path turns these types into bytes or back, which is what keeps construction the only way in.
Imports come from `simpler.buffer` because that is where a caller reaches them, alongside the
registry and the Buffer constructors that are genuinely defined there.
"""

import ctypes
import re
from multiprocessing.shared_memory import SharedMemory
from unittest.mock import patch

import pytest
from _task_interface import OWNER_INSTANCE_ID_BYTES, DataType
from simpler.buffer import (
    AccessMode,
    AddressSpace,
    BackendKind,
    BufferCapability,
    BufferDescriptor,
    CanonicalIdentity,
    ImportContext,
    ImportRegistry,
    MappedArg,
    Tensor,
    capabilities_for_adapter,
    create_host_shared_buffer,
    intern_worker_path,
    mint_owner_instance_id,
    re_export,
    select_adapter,
    wrap_device_malloc,
    wrap_fork_inherited,
    wrap_vmm_window,
)
from simpler.comm_endpoints import AdapterKind, AdapterProfile, RegionAccessReasonCode
from simpler.task_interface import ChipTensor

_OID = bytes(range(0xA0, 0xA0 + OWNER_INSTANCE_ID_BYTES))


def test_wire_tensor_and_device_pod_are_distinct_types():
    # The whole point of the two names: one address-free argument, one GM-address-bearing POD. An
    # alias here would give `Tensor` a second meaning and every later cutover would have to keep it.
    assert ChipTensor is not Tensor


def test_task_args_takes_the_wire_tensor():
    # `TaskArgs.add_tensor` takes the wire `Tensor`, and simpler.task_interface re-exports it: the
    # public submit surface names the type its own submit call accepts.
    import simpler.task_interface as ti  # noqa: PLC0415

    assert ti.Tensor is Tensor

    h = create_host_shared_buffer(64, mint_owner_instance_id(), buffer_id=1)
    try:
        args = ti.TaskArgs()
        args.add_tensor(h.tensor(shapes=(16,), dtype=DataType.FLOAT32))
        assert args.tensor_count() == 1
    finally:
        h.close()


def _identity(oid=_OID, buffer_id=7, generation=2):
    return CanonicalIdentity(oid, buffer_id, generation)


def _legal_body(backend: BackendKind) -> bytes:
    """A body that satisfies ``backend``'s schema, so a rejection is attributable to something else."""
    if backend == BackendKind.POSIX_SHM:
        return b"psm_legal"
    if backend == BackendKind.REMOTE_SIDECAR:
        return b""
    return (0x1000).to_bytes(8, "little")  # the four address-bearing backends


def test_identity_rejects_bad_oid_width():
    with pytest.raises(ValueError):
        CanonicalIdentity(b"\x00" * (OWNER_INSTANCE_ID_BYTES + 1), 1, 1)


def test_identity_distinguishes_generation_and_incarnation():
    a = _identity()
    assert a != _identity(generation=a.generation + 1)  # ABA
    assert a != _identity(oid=bytes(range(1, 1 + OWNER_INSTANCE_ID_BYTES)))  # different incarnation
    assert a != _identity(buffer_id=a.buffer_id + 1)


def _descriptor_with_path_id(path_id: int) -> BufferDescriptor:
    return BufferDescriptor(
        identity=_identity(),
        address_space=AddressSpace.HOST,
        access=AccessMode.READWRITE,
        backend_kind=BackendKind.POSIX_SHM,
        nbytes=8,
        body=b"psm_diag",  # POSIX_SHM's body is the name a consumer opens
        owner_worker_path_id=path_id,
    )


def test_worker_path_is_diagnostic_and_survives_an_unknown_id():
    h = _descriptor_with_path_id(intern_worker_path("L4/L3[2]"))
    assert h.owner_worker_path == "L4/L3[2]"
    # An id minted in another process has no local text, and that is not an error.
    assert _descriptor_with_path_id(99_999).owner_worker_path == "<path#99999>"
    # The path takes no part in identity: two descriptors differing only by path are one backing.
    assert _descriptor_with_path_id(0).identity == h.identity


def test_descriptor_rejects_oversized_body():
    with pytest.raises(ValueError, match="body"):
        BufferDescriptor(
            identity=_identity(),
            address_space=AddressSpace.HOST,
            access=AccessMode.READWRITE,
            backend_kind=BackendKind.POSIX_SHM,
            nbytes=8,
            body=b"x" * 200,  # > DESC_MAX_BYTES
        )


def test_create_export_import_resolve_zero_copy():
    oid = mint_owner_instance_id()
    buffer = create_host_shared_buffer(nbytes=256, owner_instance_id=oid, buffer_id=1, owner_worker_path="L4")
    reg = ImportRegistry(ImportContext(is_host_endpoint=True))
    try:
        assert buffer.backend_kind == BackendKind.POSIX_SHM
        imported = reg.materialize(buffer.to_descriptor())
        assert reg.resolve(buffer.identity).base == imported.base
        assert reg.materialize(buffer.to_descriptor()).base == imported.base  # map-once: same mapping
        assert imported.nbytes == 256
        owner_shm = buffer.shm
        consumer_shm = imported.shm
        assert owner_shm is not None
        assert consumer_shm is not None
        owner_buf = owner_shm.buf
        consumer_buf = consumer_shm.buf
        assert owner_buf is not None
        assert consumer_buf is not None
        owner_buf[:4] = b"\xde\xad\xbe\xef"
        assert bytes(consumer_buf[:4]) == b"\xde\xad\xbe\xef"
    finally:
        reg.close()
        buffer.close()


def test_materialize_records_host_mechanism_and_capabilities():
    buffer = create_host_shared_buffer(nbytes=64, owner_instance_id=mint_owner_instance_id(), buffer_id=1)
    reg = ImportRegistry(ImportContext(is_host_endpoint=True))
    try:
        imported = reg.materialize(buffer.to_descriptor())
        assert imported.profile is AdapterProfile.HOST_SHM_MAP
        assert imported.capabilities == capabilities_for_adapter(AdapterKind.DIRECT_MAP)
        assert BufferCapability.DIRECT_LOAD in imported.capabilities
        assert BufferCapability.DIRECT_STORE in imported.capabilities
    finally:
        reg.close()
        buffer.close()


def test_materialize_records_fork_and_device_mechanisms():
    data = ctypes.create_string_buffer(64)
    forked = wrap_fork_inherited(
        ctypes.addressof(data),
        len(data),
        mint_owner_instance_id(),
        buffer_id=1,
        backend_kind=BackendKind.FORK_SHM,
        access=AccessMode.READWRITE,
    )
    host_reg = ImportRegistry(ImportContext(is_host_endpoint=True))
    assert host_reg.materialize(forked.to_descriptor()).profile is AdapterProfile.FORK_INHERITED_VA

    owner = mint_owner_instance_id()
    device = wrap_device_malloc(0xDEAD0000, 64, owner, buffer_id=2)
    chip_reg = ImportRegistry(ImportContext(is_host_endpoint=False, device_owner_instance_id=owner))
    imported = chip_reg.materialize(device.to_descriptor())
    assert imported.profile is AdapterProfile.DEVICE_LOCAL
    assert imported.capabilities == capabilities_for_adapter(AdapterKind.DIRECT_MAP)


def test_capabilities_are_access_rights_not_adapter_names():
    direct = capabilities_for_adapter(AdapterKind.DIRECT_MAP)
    delegated = capabilities_for_adapter(AdapterKind.OWNER_DELEGATED_COPY)
    peer = capabilities_for_adapter(AdapterKind.DEVICE_PEER)
    assert delegated < direct
    assert BufferCapability.DIRECT_LOAD not in delegated
    assert BufferCapability.DEVICE_PEER_ACCESS in peer
    assert capabilities_for_adapter(AdapterKind.EXPLICIT_TRANSFER) == frozenset()
    assert capabilities_for_adapter("future-adapter") == frozenset()


def test_access_mode_narrows_what_the_mechanism_affords():
    # The mechanism sets the ceiling, the backing's own AccessMode lowers it. A mechanism that can
    # dereference both ways still grants no write right over a READ backing -- otherwise the
    # capability set would claim a direction MappedArg.buffer already refuses to hand out.
    direct = capabilities_for_adapter(AdapterKind.DIRECT_MAP)
    read_only = capabilities_for_adapter(AdapterKind.DIRECT_MAP, AccessMode.READ)
    write_only = capabilities_for_adapter(AdapterKind.DIRECT_MAP, AccessMode.WRITE)
    assert read_only == {BufferCapability.DIRECT_LOAD, BufferCapability.COPY_FROM}
    assert write_only == {BufferCapability.DIRECT_STORE, BufferCapability.COPY_TO}
    assert capabilities_for_adapter(AdapterKind.DIRECT_MAP, AccessMode.READWRITE) == direct


def test_a_read_only_fork_backing_grants_no_write_right():
    # FORK_COW is dereferenceable in the child, but a store there splits into a private copy the
    # owner never sees -- which is exactly what its READ access mode records.
    data = ctypes.create_string_buffer(64)
    cow = wrap_fork_inherited(
        ctypes.addressof(data),
        len(data),
        mint_owner_instance_id(),
        buffer_id=1,
        backend_kind=BackendKind.FORK_COW,
        access=AccessMode.READ,
    )
    reg = ImportRegistry(ImportContext(is_host_endpoint=True))
    imported = reg.materialize(cow.to_descriptor())
    assert imported.profile is AdapterProfile.FORK_INHERITED_VA
    assert BufferCapability.DIRECT_LOAD in imported.capabilities
    assert BufferCapability.DIRECT_STORE not in imported.capabilities
    assert BufferCapability.COPY_TO not in imported.capabilities


def test_every_materialize_refusal_carries_a_region_access_reason_code():
    # Both evaluation moments of the capability judgment answer in one vocabulary, refusals
    # included: a caller can tell an endpoint-relation verdict from an unsupported backing without
    # parsing prose.
    oid = mint_owner_instance_id()
    host_reg = ImportRegistry(ImportContext(is_host_endpoint=True))
    device = wrap_device_malloc(0xDEAD0000, 64, oid, buffer_id=1)
    with pytest.raises(ValueError, match=re.escape(RegionAccessReasonCode.UNSUPPORTED_ENDPOINT_RELATION.value)):
        host_reg.materialize(device.to_descriptor())

    sidecar = BufferDescriptor(
        identity=_identity(),
        address_space=AddressSpace.HOST,
        access=AccessMode.READWRITE,
        backend_kind=BackendKind.REMOTE_SIDECAR,
        nbytes=8,
    )
    with pytest.raises(ValueError, match=re.escape(RegionAccessReasonCode.UNSUPPORTED_BACKEND_KIND.value)):
        host_reg.materialize(sidecar)


def _delegating_host_context(owner: bytes) -> ImportContext:
    """A Worker: a host endpoint that owns ``owner``'s chip children, so it can delegate a copy."""
    return ImportContext(is_host_endpoint=True, device_owner_instance_id=owner)


def test_a_device_backing_at_a_delegating_host_endpoint_is_reachable_not_refused():
    # Being unable to MAP a backing is not being unable to touch it. A Worker holds no device VA,
    # but it drives the mailbox of the chip that owns the allocation, so the copy rights are real.
    owner = mint_owner_instance_id()
    device = wrap_device_malloc(0xDEAD0000, 64, owner, buffer_id=1)

    kind, profile = select_adapter(device.to_descriptor(), _delegating_host_context(owner))
    assert (kind, profile) == (AdapterKind.OWNER_DELEGATED_COPY, AdapterProfile.OWNER_DEVICE_COPY)

    granted = capabilities_for_adapter(kind, device.access)
    assert BufferCapability.COPY_TO in granted and BufferCapability.COPY_FROM in granted
    assert BufferCapability.DIRECT_LOAD not in granted  # no address in this process to load from


def test_a_vmm_window_at_a_delegating_host_endpoint_matches_the_region_planner():
    # The planner already gives a host region consumer OWNER_DELEGATED_COPY/HOST_VMM_COPY
    # (`_adapter_candidates`); the per-tensor moment must not disagree with it about the same
    # relation.
    owner = mint_owner_instance_id()
    window = wrap_vmm_window(0xBEEF0000, 4096, owner, buffer_id=2)
    assert select_adapter(window.to_descriptor(), _delegating_host_context(owner)) == (
        AdapterKind.OWNER_DELEGATED_COPY,
        AdapterProfile.HOST_VMM_COPY,
    )


def test_materialize_refuses_a_delegated_mechanism_for_having_no_address():
    # `materialize` returns a mapping, so it narrows the judgment to the mechanisms that produce
    # one -- and its refusal names the mechanism that DOES reach the backing, rather than reporting
    # no relation. That distinction is the whole point: the endpoint may touch this backing, just
    # not by holding an address to it.
    owner = mint_owner_instance_id()
    device = wrap_device_malloc(0xDEAD0000, 64, owner, buffer_id=1)
    reg = ImportRegistry(_delegating_host_context(owner))
    with pytest.raises(ValueError, match=re.escape(RegionAccessReasonCode.NO_IMPLEMENTED_DIRECT_MAP_PROBE.value)):
        reg.materialize(device.to_descriptor())
    with pytest.raises(ValueError, match="OWNER_DELEGATED_COPY"):
        reg.materialize(device.to_descriptor())


def test_a_host_endpoint_with_no_delegation_channel_still_has_no_relation():
    # A SUB child is the other host endpoint: it cannot map a device backing and owns no chip that
    # could copy for it, so there is genuinely no mechanism -- distinct from the Worker's case.
    owner = mint_owner_instance_id()
    device = wrap_device_malloc(0xDEAD0000, 64, owner, buffer_id=1)
    with pytest.raises(ValueError, match=re.escape(RegionAccessReasonCode.UNSUPPORTED_ENDPOINT_RELATION.value)):
        select_adapter(device.to_descriptor(), ImportContext(is_host_endpoint=True))
    # Nor a different Worker's device backing, even for an endpoint that can delegate to its own.
    with pytest.raises(ValueError, match=re.escape(RegionAccessReasonCode.UNSUPPORTED_ENDPOINT_RELATION.value)):
        select_adapter(device.to_descriptor(), _delegating_host_context(mint_owner_instance_id()))


def test_a_mappable_backing_still_maps_once_and_carries_the_direct_rights():
    # The mapping subset is unchanged by the judgment being able to name non-mapping mechanisms:
    # a host shm backing still resolves to one cached mapping with the direct-map rights.
    buffer = create_host_shared_buffer(nbytes=64, owner_instance_id=mint_owner_instance_id(), buffer_id=1)
    reg = ImportRegistry(ImportContext(is_host_endpoint=True))
    try:
        kind, profile = select_adapter(buffer.to_descriptor(), ImportContext(is_host_endpoint=True))
        assert (kind, profile) == (AdapterKind.DIRECT_MAP, AdapterProfile.HOST_SHM_MAP)
        imported = reg.materialize(buffer.to_descriptor())
        assert imported.capabilities == capabilities_for_adapter(AdapterKind.DIRECT_MAP, buffer.access)
        assert reg.materialize(buffer.to_descriptor()) is imported  # map-once
    finally:
        reg.close()
        buffer.close()


def test_require_distinguishes_missing_and_insufficient_capabilities():
    buffer = create_host_shared_buffer(nbytes=64, owner_instance_id=mint_owner_instance_id(), buffer_id=1)
    reg = ImportRegistry(ImportContext(is_host_endpoint=True))
    try:
        with pytest.raises(KeyError):
            reg.require(buffer.identity, BufferCapability.COPY_TO)
        reg.materialize(buffer.to_descriptor())
        assert reg.require(buffer.identity, BufferCapability.COPY_TO).identity == buffer.identity
        with pytest.raises(ValueError, match="DEVICE_PEER_ACCESS"):
            reg.require(buffer.identity, BufferCapability.DEVICE_PEER_ACCESS)
    finally:
        reg.close()
        buffer.close()


def test_close_unlinks_even_when_shm_close_raises():
    # A close() failure must not skip owner unlink: the shm's named backing is the actual leak risk,
    # not the local close() call, so unlink must run regardless of whether close() itself succeeded.
    buffer = create_host_shared_buffer(nbytes=64, owner_instance_id=mint_owner_instance_id(), buffer_id=1)
    shm = buffer.shm
    assert shm is not None
    with (
        patch.object(shm, "close", side_effect=OSError("injected close failure")),
        patch.object(shm, "unlink") as unlink,
    ):
        with pytest.raises(OSError, match="injected close failure"):
            buffer.close()
        unlink.assert_called_once()
    assert buffer.closed
    shm.unlink()  # the mock above swallowed the real unlink; do it for real so the test leaves no /dev/shm litter


def test_close_retries_the_unlink_that_failed():
    # The named backing is what outlives the process, so an unlink that fails must stay pending: the
    # cleanup journal keeps a failed buffer's registry entry precisely so close() runs again, and a
    # retry that returns without re-attempting the unlink leaves the name in /dev/shm forever.
    buffer = create_host_shared_buffer(nbytes=64, owner_instance_id=mint_owner_instance_id(), buffer_id=1)
    shm = buffer.shm
    assert shm is not None
    name = shm.name
    with patch.object(shm, "unlink", side_effect=OSError("injected unlink failure")) as unlink:
        with pytest.raises(OSError, match="injected unlink failure"):
            buffer.close()
        unlink.assert_called_once()
    assert buffer.closed  # the derivation gate shuts on the first attempt, successful release or not
    assert not buffer.unlinked
    buffer.close()
    assert buffer.unlinked
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=name)


def test_close_does_not_unlink_twice_after_a_failed_close():
    # The retry above must not re-run an unlink that already succeeded: a close() failure leaves the
    # name already removed by the finally, so a second unlink would raise FileNotFoundError over a
    # backing that is simply gone -- turning a recoverable state into a permanent error.
    buffer = create_host_shared_buffer(nbytes=64, owner_instance_id=mint_owner_instance_id(), buffer_id=1)
    shm = buffer.shm
    assert shm is not None
    with patch.object(shm, "close", side_effect=OSError("injected close failure")):
        with pytest.raises(OSError, match="injected close failure"):
            buffer.close()
    assert buffer.unlinked
    buffer.close()  # the retry: closes the mapping for real, and leaves the unlink alone
    assert buffer.shm is None


def test_closed_buffer_refuses_to_derive_a_tensor():
    # A released Buffer's identity may already be unlinked, so deriving a Tensor from it would embed
    # a descriptor for memory that no longer exists.
    buffer = create_host_shared_buffer(nbytes=64, owner_instance_id=mint_owner_instance_id(), buffer_id=1)
    buffer.close()
    assert buffer.closed
    with pytest.raises(ValueError, match="released buffer"):
        buffer.to_descriptor()
    with pytest.raises(ValueError, match="released buffer"):
        buffer.tensor(shapes=(16,), dtype=DataType.FLOAT32)


def test_resolve_unregistered_raises():
    reg = ImportRegistry(ImportContext(is_host_endpoint=True))
    with pytest.raises(KeyError):
        reg.resolve(_identity())


def test_unregister_drops_a_materialized_mapping():
    oid = mint_owner_instance_id()
    buffer = create_host_shared_buffer(nbytes=64, owner_instance_id=oid, buffer_id=1)
    reg = ImportRegistry(ImportContext(is_host_endpoint=True))
    try:
        reg.materialize(buffer.to_descriptor())
        reg.unregister(buffer.identity)
        with pytest.raises(KeyError):
            reg.resolve(buffer.identity)
        # Re-materializing after unregister must re-open the shm fresh, not fail or hit a stale
        # cache entry — the same identity mapped, closed, and mapped again.
        reg.materialize(buffer.to_descriptor())
    finally:
        reg.close()
        buffer.close()


def test_unregister_is_a_no_op_for_an_identity_never_materialized():
    reg = ImportRegistry(ImportContext(is_host_endpoint=True))
    reg.unregister(_identity())  # must not raise


def test_unregister_keeps_a_mapping_it_could_not_close():
    # shm.close() raises BufferError while a consumer still holds a memoryview derived from the
    # mapping. An entry dropped at that point names a mapping nothing can reach again, so this
    # endpoint's close() could never retry it -- the entry has to survive the failure.
    oid = mint_owner_instance_id()
    buffer = create_host_shared_buffer(nbytes=64, owner_instance_id=oid, buffer_id=1)
    reg = ImportRegistry(ImportContext(is_host_endpoint=True))
    try:
        imported = reg.materialize(buffer.to_descriptor())
        assert imported.shm is not None
        with patch.object(imported.shm, "close", side_effect=BufferError("cannot close exported pointers exist")):
            with pytest.raises(BufferError):
                reg.unregister(buffer.identity)
        assert reg.resolve(buffer.identity) is imported
        reg.unregister(buffer.identity)  # the retry, once the view is gone
        with pytest.raises(KeyError):
            reg.resolve(buffer.identity)
    finally:
        reg.close()
        buffer.close()


def test_close_attempts_every_mapping_and_reports_the_first_failure():
    # One endpoint holding an exported view must not strand the mappings behind it in iteration
    # order: every mapping is attempted, the ones that closed are dropped, and the caller still
    # learns about the leak instead of seeing a silent success.
    oid = mint_owner_instance_id()
    first = create_host_shared_buffer(nbytes=64, owner_instance_id=oid, buffer_id=1)
    second = create_host_shared_buffer(nbytes=64, owner_instance_id=oid, buffer_id=2)
    reg = ImportRegistry(ImportContext(is_host_endpoint=True))
    try:
        imported_first = reg.materialize(first.to_descriptor())
        imported_second = reg.materialize(second.to_descriptor())
        assert imported_first.shm is not None
        assert imported_second.shm is not None
        with (
            patch.object(imported_first.shm, "close", side_effect=BufferError("injected close failure")),
            patch.object(imported_second.shm, "close") as second_close,
        ):
            with pytest.raises(BufferError, match="injected close failure"):
                reg.close()
            second_close.assert_called_once()
        assert reg.resolve(first.identity) is imported_first  # the one that failed is kept
        with pytest.raises(KeyError):
            reg.resolve(second.identity)
    finally:
        first.close()
        second.close()


def test_materialize_after_release_says_the_owner_released_it():
    # A missing shm name is the expected shape of "this identity was released", and the bare
    # FileNotFoundError the OS raises names only /psm_xxxx -- which identity, and why it is gone,
    # are exactly what the reader needs.
    oid = mint_owner_instance_id()
    buffer = create_host_shared_buffer(nbytes=64, owner_instance_id=oid, buffer_id=1)
    descriptor = buffer.to_descriptor()
    buffer.close()
    reg = ImportRegistry(ImportContext(is_host_endpoint=True))
    with pytest.raises(FileNotFoundError, match="released the buffer"):
        reg.materialize(descriptor)


def test_tensor_full_view_is_contiguous():
    oid = mint_owner_instance_id()
    h = create_host_shared_buffer(nbytes=1024, owner_instance_id=oid, buffer_id=1)
    try:
        # buffer.tensor(shape, dtype) is a contiguous full view: row-major strides, zero offset.
        v = h.tensor(shapes=(4, 8), dtype=DataType.FLOAT32)
        assert v.shapes == (4, 8)
        assert v.strides == (8, 1)
        assert v.ndims == 2
        assert v.byte_offset == 0
        # An explicit stride is carried verbatim; a singleton dim is never normalized away.
        strided = h.tensor(shapes=(4, 1), dtype=DataType.FLOAT32, strides=(8, 3))
        assert strided.strides == (8, 3)
    finally:
        h.close()


def test_re_export_preserves_identity_same_backing_no_map():
    # Frozen model §5/§8: canonical identity is invariant across every edge. Re-exporting an L4-owned
    # backing for forwarding keeps the SOURCE identity (owner_instance_id / path / buffer_id /
    # generation) and the same backing, only stripping the mapping.
    l4 = mint_owner_instance_id()
    src = create_host_shared_buffer(64, l4, buffer_id=7, owner_worker_path="L4")
    try:
        sdesc = src.to_descriptor()
        hp = re_export(sdesc)
        assert hp.identity == src.identity  # identity invariant across the edge
        assert hp.backend_kind == BackendKind.POSIX_SHM
        assert hp.body == sdesc.body and hp.nbytes == 64  # same backing
        assert hp.shm is None and hp.base == 0  # no map (lazy — a compute leaf maps)
        # a tensor built from H' carries the source identity + the same shm body, so L2 can materialize it
        r = hp.tensor(shapes=(16,), dtype=DataType.FLOAT32)
        assert r.buffer.identity == src.identity
        assert r.buffer.body == sdesc.body
    finally:
        src.close()


def test_device_malloc_wrap_materialize():
    # A device pointer (from orch.malloc) wrapped as DEVICE_MALLOC: materializes to the pointer with
    # no map, address_space DEVICE (-> a child_memory Tensor).
    oid = mint_owner_instance_id()
    h = wrap_device_malloc(0xDEAD0000, 4096, oid, buffer_id=3, owner_worker_path="L3")
    assert h.backend_kind == BackendKind.DEVICE_MALLOC
    assert h.address_space == AddressSpace.DEVICE
    assert h.shm is None and h.base == 0xDEAD0000
    reg = ImportRegistry(ImportContext(is_host_endpoint=False, device_owner_instance_id=oid))
    imp = reg.materialize(h.to_descriptor())
    assert imp.base == 0xDEAD0000
    assert imp.address_space == AddressSpace.DEVICE
    assert imp.shm is None


def test_materialize_args_scopes_the_returned_map_to_this_calls_tensors():
    # materialize_args must not hand back an endpoint's entire materialize-once history — only the
    # identities the tensors in THIS call touched. A dispatch that reuses a registry across many
    # tasks (the real chip/L2-leaf usage) would otherwise pay O(every identity ever seen) per call.
    import simpler.task_interface as ti  # noqa: PLC0415

    oid = mint_owner_instance_id()
    reg = ImportRegistry(ImportContext(is_host_endpoint=False, device_owner_instance_id=oid))
    h1 = wrap_device_malloc(0xDEAD0000, 4096, oid, buffer_id=1)
    h2 = wrap_device_malloc(0xBEEF0000, 4096, oid, buffer_id=2)

    args1 = ti.TaskArgs()
    args1.add_tensor(h1.tensor(shapes=(16,), dtype=DataType.FLOAT32))
    resolved1 = reg.materialize_args(args1)
    assert set(resolved1) == {h1.identity}

    args2 = ti.TaskArgs()
    args2.add_tensor(h2.tensor(shapes=(16,), dtype=DataType.FLOAT32))
    resolved2 = reg.materialize_args(args2)
    # h1 is still live in the registry's own history (map-once), but this call's args never
    # referenced it, so it must not appear in this call's returned map.
    assert set(resolved2) == {h2.identity}


def test_materialize_remote_sidecar_rejected():
    desc = BufferDescriptor(
        identity=_identity(),
        address_space=AddressSpace.HOST,
        access=AccessMode.READWRITE,
        backend_kind=BackendKind.REMOTE_SIDECAR,
        nbytes=8,
    )
    reg = ImportRegistry(ImportContext(is_host_endpoint=True))
    with pytest.raises(ValueError, match="REMOTE_SIDECAR"):
        reg.materialize(desc)


# --- map-once is a cache, not a trust boundary ------------------------------------------------


def _device_descriptor(oid: bytes, buffer_id: int, nbytes: int, ptr: int = 0xDEAD0000) -> BufferDescriptor:
    return wrap_device_malloc(ptr, nbytes, oid, buffer_id=buffer_id).to_descriptor()


def test_materialize_rejects_a_conflicting_descriptor_for_a_live_identity():
    # Identity says WHICH allocation, not how big it is or how to reach it. A second descriptor
    # carrying the same identity and a different nbytes describes something the existing mapping is
    # not, so returning that mapping would hand back a base under a size nothing stands behind.
    oid = mint_owner_instance_id()
    reg = ImportRegistry(ImportContext(is_host_endpoint=False, device_owner_instance_id=oid))
    first = reg.materialize(_device_descriptor(oid, 1, nbytes=4096))

    for conflicting in (
        _device_descriptor(oid, 1, nbytes=8192),  # same identity, bigger claim
        _device_descriptor(oid, 1, nbytes=4096, ptr=0xBEEF0000),  # same identity, other backing
    ):
        with pytest.raises(ValueError, match="different descriptor"):
            reg.materialize(conflicting)

    # The mapping already handed out survives: callers may hold addresses into it, so the conflict
    # is refused rather than resolved by replacing it.
    assert reg.resolve(first.identity) is first
    assert reg.materialize(_device_descriptor(oid, 1, nbytes=4096)) is first


def test_conflict_error_names_only_the_fields_that_differ():
    # A whole repr of both descriptors would carry a 32-byte body twice for what is usually a
    # one-field disagreement, and leave the reader to diff them by eye.
    oid = mint_owner_instance_id()
    reg = ImportRegistry(ImportContext(is_host_endpoint=False, device_owner_instance_id=oid))
    reg.materialize(_device_descriptor(oid, 1, nbytes=4096))
    with pytest.raises(ValueError) as excinfo:
        reg.materialize(_device_descriptor(oid, 1, nbytes=8192))
    message = str(excinfo.value)
    assert "nbytes: 4096 -> 8192" in message
    assert "backend_kind" not in message  # unchanged fields stay out of it


def test_materialize_rejects_an_shm_object_shorter_than_its_descriptor():
    # Every view bound check is `byte_offset + extent <= nbytes`, so an unverified nbytes makes all
    # of them vacuous. The object's real size is the one thing here the owner cannot overstate.
    oid = mint_owner_instance_id()
    buffer = create_host_shared_buffer(nbytes=128, owner_instance_id=oid, buffer_id=1)
    reg = ImportRegistry(ImportContext(is_host_endpoint=True))
    try:
        honest = buffer.to_descriptor()
        assert reg.materialize(honest).nbytes == 128  # the truthful one maps

        overstated = BufferDescriptor(
            identity=CanonicalIdentity(oid, 2, 1),  # a fresh identity, so this is not the conflict path
            address_space=AddressSpace.HOST,
            access=AccessMode.READWRITE,
            backend_kind=BackendKind.POSIX_SHM,
            nbytes=1 << 20,
            body=honest.body,  # the same 128-byte object
        )
        with pytest.raises(ValueError, match="short of the"):
            reg.materialize(overstated)
    finally:
        reg.close()
        buffer.close()


def test_owner_instance_ids_are_distinct():
    ids = {mint_owner_instance_id() for _ in range(64)}
    assert len(ids) == 64
    assert all(len(i) == OWNER_INSTANCE_ID_BYTES for i in ids)


@pytest.mark.parametrize(
    "space,backend",
    [
        (AddressSpace.HOST, BackendKind.VMM_WINDOW),
        (AddressSpace.HOST, BackendKind.DEVICE_MALLOC),
        (AddressSpace.DEVICE, BackendKind.FORK_SHM),
        (AddressSpace.DEVICE, BackendKind.POSIX_SHM),
    ],
)
def test_descriptor_rejects_bad_capability_combo(space, backend):
    # §4.1 capability matrix: an unsupported address_space×backend_kind fails at construction (before
    # dispatch, before it can ride the wire). The body is a legal one for the backend, so the
    # rejection is attributable to the combination and not to the body schema.
    with pytest.raises(ValueError, match="capability"):
        BufferDescriptor(
            identity=_identity(),
            address_space=space,
            access=AccessMode.READWRITE,
            backend_kind=backend,
            nbytes=64,
            body=_legal_body(backend),
        )


def test_descriptor_accepts_legal_combos():
    for space, backend in [
        (AddressSpace.HOST, BackendKind.FORK_SHM),
        (AddressSpace.HOST, BackendKind.POSIX_SHM),
        (AddressSpace.DEVICE, BackendKind.VMM_WINDOW),
        (AddressSpace.DEVICE, BackendKind.DEVICE_MALLOC),
        (AddressSpace.HOST, BackendKind.REMOTE_SIDECAR),
        (AddressSpace.DEVICE, BackendKind.REMOTE_SIDECAR),
    ]:
        BufferDescriptor(_identity(), space, AccessMode.READWRITE, backend, 64, _legal_body(backend))


# --- G2: the body must fit the reading its backend_kind implies --------------------------------
#
# The exhaustive per-backend cases live in tests/ut/cpp/types/test_buffer.cpp, which can build the
# malformed bytes Python cannot (a body_len that disagrees with the body, a non-zero reserved tail).
# What is checked here is that the gate is reachable from construction.


def test_construction_rejects_an_address_body_that_is_not_eight_bytes():
    # A short body reads as a truncated pointer with nothing to distinguish it from a real one.
    for bad in (b"", b"\x00\x01\x02", b"\x00" * 7, b"\x00" * 9):
        with pytest.raises(ValueError, match="exactly 8 bytes"):
            BufferDescriptor(_identity(), AddressSpace.DEVICE, AccessMode.READWRITE, BackendKind.DEVICE_MALLOC, 64, bad)


def test_construction_rejects_a_null_address_body():
    # Nothing is mapped or allocated at 0, so a zero base is an unfilled body, not a location.
    with pytest.raises(ValueError, match="null base"):
        BufferDescriptor(
            _identity(), AddressSpace.DEVICE, AccessMode.READWRITE, BackendKind.DEVICE_MALLOC, 64, b"\x00" * 8
        )


def test_construction_rejects_a_shm_name_outside_printable_ascii():
    # The name is decoded as UTF-8 before the shm open, so the wire schema restricts it to a UTF-8
    # subset: bytes that pass validation always decode. A NUL would additionally truncate the name
    # at the first C API it reaches, letting two distinct names open one object.
    for bad in (b"psm\x00abc", b"psm abc", b"psm/abc", b"psm\xffabc", b"psm\x7fabc"):
        with pytest.raises(ValueError, match="printable ASCII"):
            BufferDescriptor(_identity(), AddressSpace.HOST, AccessMode.READWRITE, BackendKind.POSIX_SHM, 64, bad)


def test_construction_rejects_a_remote_sidecar_body():
    # The authoritative descriptor rides in the per-task sidecar; a body here is a second source of
    # truth that nothing reads.
    with pytest.raises(ValueError, match="no body"):
        BufferDescriptor(
            _identity(), AddressSpace.HOST, AccessMode.READWRITE, BackendKind.REMOTE_SIDECAR, 64, b"\x01" * 8
        )


# --- the shared validator, on the paths Python can reach --------------------------------------
#
# `validate_tensor` guards two boundaries: construction (here) and blob decode, which after the wire
# flip is `TaskArgsView::tensors(i)` in task_args.h. Both are C++, and Python has no way to turn
# bytes into a Tensor at all — so the malformed-bytes cases (bad magic, unknown backend tag,
# generation 0, body_len past the array, a view that does not fit) are exercised where they can be
# built: tests/ut/cpp/types/test_buffer.cpp. What is reachable from here is the construction gate.


def test_construction_rejects_a_view_past_the_backing():
    h = create_host_shared_buffer(64, mint_owner_instance_id(), buffer_id=1)
    try:
        h.tensor(shapes=(16,), dtype=DataType.FLOAT32)  # exactly 64 B: fits
        with pytest.raises(ValueError, match="past the backing"):
            h.tensor(shapes=(17,), dtype=DataType.FLOAT32)
        with pytest.raises(ValueError, match="past the backing"):
            h.tensor(shapes=(16,), dtype=DataType.FLOAT32, byte_offset=4)
    finally:
        h.close()


def test_construction_rejects_a_zero_stride():
    # strides are element strides and strictly > 0: broadcast and negative step are unsupported, and
    # a 0 would make two coordinates alias without the overlap map ever seeing it.
    h = create_host_shared_buffer(256, mint_owner_instance_id(), buffer_id=1)
    try:
        with pytest.raises(ValueError, match="stride"):
            h.tensor(shapes=(4, 8), dtype=DataType.FLOAT32, strides=(8, 0))
    finally:
        h.close()


def test_construction_rejects_an_extent_that_overflows():
    # shapes/strides are u32 each, so one (shape-1)*stride product alone approaches 2^64. An extent
    # summed without saturation wraps to a small value and the view passes as "in bounds" while
    # really spanning exabytes — reachable straight from here.
    h = create_host_shared_buffer(64, mint_owner_instance_id(), buffer_id=1)
    try:
        with pytest.raises(ValueError, match="overflows 64 bits"):
            h.tensor(shapes=(2147483649,), dtype=DataType.FLOAT32, strides=(2147483648,))
    finally:
        h.close()


def test_fork_backend_is_stated_not_inferred_from_access():
    # FORK_SHM and FORK_COW are opposite kernel write semantics, so the caller states which mmap it
    # holds; deriving it from `access` makes a read-only MAP_SHARED backing inexpressible.
    oid = mint_owner_instance_id()
    shared_ro = wrap_fork_inherited(
        0x1000, 64, oid, buffer_id=1, access=AccessMode.READ, backend_kind=BackendKind.FORK_SHM
    )
    assert shared_ro.backend_kind == BackendKind.FORK_SHM
    assert shared_ro.to_descriptor().access == AccessMode.READ

    cow = wrap_fork_inherited(0x1000, 64, oid, buffer_id=2)  # the safe default pair
    assert cow.backend_kind == BackendKind.FORK_COW
    assert cow.base == 0x1000

    # A write grant over copy-on-write would be silently unobservable by the owner, so the
    # descriptor's validator refuses it rather than letting the pair exist.
    bad = wrap_fork_inherited(
        0x1000, 64, oid, buffer_id=3, access=AccessMode.READWRITE, backend_kind=BackendKind.FORK_COW
    )
    with pytest.raises(ValueError, match="FORK_COW"):
        bad.to_descriptor()


def test_mapped_arg_buffer_is_read_only_for_a_read_access_descriptor():
    # FORK_COW's whole contract is that a write is invisible to the owner (copy-on-write splits the
    # page privately), so `MappedArg.buffer` must not hand back a writable view for it. `.cast("B")`
    # is what makes the write attempt itself meaningful: the raw `<c` format memoryview this property
    # returns for a ctypes-backed mapping does not support slice assignment at all (readonly or not),
    # so asserting through the native format would prove nothing about the readonly flag specifically.
    data = bytearray(16)
    addr = ctypes.addressof((ctypes.c_char * 16).from_buffer(data))
    oid = mint_owner_instance_id()
    buf = wrap_fork_inherited(addr, 16, oid, buffer_id=1)  # default: access=READ, backend=FORK_COW
    reg = ImportRegistry(ImportContext(is_host_endpoint=True))
    imported = reg.materialize(buf.to_descriptor())
    arg = MappedArg(imported, byte_offset=0, shapes=(16,), strides=(1,), dtype=DataType.UINT8)

    view = arg.buffer
    assert view.readonly
    with pytest.raises(TypeError):
        view.cast("B")[0:4] = b"\x01\x02\x03\x04"
