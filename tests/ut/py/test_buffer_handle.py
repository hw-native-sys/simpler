# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for simpler.buffer_handle: identity/descriptor pack-unpack + create/import round trip."""

import ctypes

import pytest
from _task_interface import (
    BUFFER_HANDLE_DESCRIPTOR_BYTES,
    CANONICAL_IDENTITY_BYTES,
    DataType,
    materialize_bufferref_blob,
    read_args_from_blob,
)
from simpler.buffer_handle import (
    BUFFER_REF_FLAG_MANUAL_DEP,
    AccessMode,
    AddressSpace,
    BackendKind,
    BufferHandleDescriptor,
    BufferRef,
    CanonicalIdentity,
    ImportRegistry,
    Visibility,
    create_host_shared_buffer,
    mint_owner_instance_id,
    pack_bufferref_blob,
)


def _identity(oid=0xA1A2A3A4A5A6A7A8, buffer_id=7, path=(3, 5), generation=2):
    return CanonicalIdentity(oid, buffer_id, path, generation)


def test_identity_roundtrip_various_depths():
    for path in [(), (1,), (3, 5), (1, 2, 3, 4)]:
        ident = _identity(path=path)
        raw = ident.pack()
        assert len(raw) == CANONICAL_IDENTITY_BYTES
        assert CanonicalIdentity.unpack(raw) == ident


def test_identity_depth_over_limit_rejected():
    with pytest.raises(ValueError):
        CanonicalIdentity(1, 1, (0, 0, 0, 0, 0), 0)  # depth 5 > MAX_WORKER_PATH_DEPTH (4)


def test_identity_distinguishes_generation_owner_path():
    a = _identity()
    assert a != _identity(generation=a.generation + 1)  # ABA
    assert a != _identity(oid=a.owner_instance_id + 1)  # different incarnation
    assert a != _identity(path=(3, 6))  # different path


def test_descriptor_roundtrip_host_and_device():
    host = BufferHandleDescriptor(
        identity=_identity(),
        address_space=AddressSpace.HOST,
        visibility=Visibility.SHARED,
        access=AccessMode.READWRITE,
        backend_kind=BackendKind.POSIX_SHM,
        nbytes=4096,
        token="psm_deadbeef",
    )
    raw = host.pack()
    assert len(raw) == BUFFER_HANDLE_DESCRIPTOR_BYTES
    assert BufferHandleDescriptor.unpack(raw) == host

    dev = BufferHandleDescriptor(
        identity=_identity(buffer_id=99),
        address_space=AddressSpace.DEVICE,
        visibility=Visibility.PRIVATE,
        access=AccessMode.READ,
        backend_kind=BackendKind.VMM_WINDOW,
        nbytes=1 << 20,
        backend_handle=0x7F00ABCD,
    )
    assert BufferHandleDescriptor.unpack(dev.pack()) == dev


def test_descriptor_rejects_unknown_version():
    raw = bytearray(
        BufferHandleDescriptor(
            identity=_identity(),
            address_space=AddressSpace.HOST,
            visibility=Visibility.SHARED,
            access=AccessMode.READWRITE,
            backend_kind=BackendKind.POSIX_SHM,
            nbytes=8,
        ).pack()
    )
    raw[0] = raw[0] + 1  # bump abi_version (offset 0)
    with pytest.raises(ValueError, match="abi_version"):
        BufferHandleDescriptor.unpack(bytes(raw))


def test_descriptor_rejects_oversized_token():
    with pytest.raises(ValueError, match="token"):
        BufferHandleDescriptor(
            identity=_identity(),
            address_space=AddressSpace.HOST,
            visibility=Visibility.SHARED,
            access=AccessMode.READWRITE,
            backend_kind=BackendKind.POSIX_SHM,
            nbytes=8,
            token="x" * 64,  # >= BACKEND_TOKEN_BYTES
        ).pack()


def test_create_export_import_resolve_zero_copy():
    oid = mint_owner_instance_id()
    handle = create_host_shared_buffer(nbytes=256, owner_instance_id=oid, buffer_id=1, owner_worker_path=(0,))
    reg = ImportRegistry()
    try:
        assert handle.backend_kind == BackendKind.POSIX_SHM
        imported = reg.register(handle.to_descriptor().pack())
        # Same identity resolves to the imported mapping.
        assert reg.resolve(handle.identity).base == imported.base
        assert imported.nbytes == 256
        # Zero-copy: owner and consumer map the same physical shm — a write on one side is visible
        # on the other even though the base VAs differ (independent mappings of one POSIX shm).
        owner_shm = handle.shm
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
        handle.close()


def test_resolve_unregistered_raises():
    reg = ImportRegistry()
    with pytest.raises(KeyError):
        reg.resolve(_identity())


def test_register_remote_sidecar_rejected():
    desc = BufferHandleDescriptor(
        identity=_identity(),
        address_space=AddressSpace.HOST,
        visibility=Visibility.SHARED,
        access=AccessMode.READWRITE,
        backend_kind=BackendKind.REMOTE_SIDECAR,
        nbytes=8,
    )
    reg = ImportRegistry()
    with pytest.raises(ValueError, match="REMOTE_SIDECAR"):
        reg.register(desc)


def test_owner_instance_ids_are_distinct():
    ids = {mint_owner_instance_id() for _ in range(64)}
    assert len(ids) == 64  # 64-bit nonce, collisions astronomically unlikely


def test_materialize_bufferref_blob_to_tensors():
    oid = mint_owner_instance_id()
    h_host = create_host_shared_buffer(nbytes=64, owner_instance_id=oid, buffer_id=1)
    h_dev = create_host_shared_buffer(nbytes=128, owner_instance_id=oid, buffer_id=2, generation=3)
    reg = ImportRegistry()
    try:
        reg.register(h_host.to_descriptor().pack())
        reg.register(h_dev.to_descriptor().pack())

        # A BufferRef blob referencing both handles (contiguous views) plus a scalar.
        ref0 = BufferRef(h_host.identity, byte_offset=0, shapes=(4,), strides=(1,), dtype=DataType.FLOAT32.value)
        ref1 = BufferRef(
            h_dev.identity,
            byte_offset=8,
            shapes=(2, 4),
            strides=(4, 1),
            dtype=DataType.FLOAT16.value,
            flags=BUFFER_REF_FLAG_MANUAL_DEP,
        )
        blob = pack_bufferref_blob([ref0, ref1], scalars=(42,))
        src = ctypes.create_string_buffer(blob, len(blob))

        tensor_blob = materialize_bufferref_blob(ctypes.addressof(src), len(blob), reg.materialization_map())
        dst = ctypes.create_string_buffer(tensor_blob, len(tensor_blob))
        args = read_args_from_blob(ctypes.addressof(dst))

        assert args.tensor_count() == 2
        assert args.scalar_count() == 1
        # Each tensor resolves to its handle's local base + byte_offset, with the ref's view.
        assert args.tensor(0).data == reg.resolve(h_host.identity).base + 0
        assert args.tensor(0).shapes == (4,)
        assert args.tensor(0).child_memory is False  # HOST
        assert args.tensor(1).data == reg.resolve(h_dev.identity).base + 8
        assert args.tensor(1).shapes == (2, 4)
        assert args.scalar(0) == 42
    finally:
        reg.close()
        h_host.close()
        h_dev.close()


def test_materialize_rejects_unregistered_identity():
    oid = mint_owner_instance_id()
    handle = create_host_shared_buffer(nbytes=32, owner_instance_id=oid, buffer_id=9)
    try:
        ref = BufferRef(handle.identity, byte_offset=0, shapes=(8,), strides=(1,), dtype=DataType.INT32.value)
        blob = pack_bufferref_blob([ref])
        src = ctypes.create_string_buffer(blob, len(blob))
        with pytest.raises(RuntimeError, match="identity"):
            materialize_bufferref_blob(ctypes.addressof(src), len(blob), {})  # empty registry
    finally:
        handle.close()
