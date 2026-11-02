# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import ctypes
import os
import struct
import threading
from dataclasses import FrozenInstanceError, replace
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import pytest
from _task_interface import _encode_buffer_descriptor_wire
from simpler.buffer import (
    AccessMode,
    AddressSpace,
    BackendKind,
    BufferCapability,
    BufferDescriptor,
    CanonicalIdentity,
    ImportContext,
    ImportRegistry,
    buffer_reference,
)
from simpler.comm_domain import CommDomainRef, DomainConfig
from simpler.comm_endpoints import DEVICE_AICPU, HOST_CPU
from simpler.global_comm_domain import (
    GLOBAL_DOMAIN_DESCRIPTOR_BYTES,
    GLOBAL_DOMAIN_VERSION,
    GlobalDomainBuffer,
    GlobalDomainCommand,
    GlobalDomainDescriptor,
    GlobalDomainHostWindow,
    GlobalDomainMember,
    GlobalDomainPhase,
    GlobalDomainReleaseCommand,
    GlobalDomainReleaseStage,
    decode_descriptor_table,
    decode_domain_buffer_refs,
    decode_domain_command,
    decode_release_command,
    encode_descriptor_table,
    encode_domain_buffer_refs,
    encode_domain_command,
    encode_release_command,
)
from simpler.task_interface import (
    ChipDomainContext,
    CommDomainHandle,
    DataType,
    HostWindowSpec,
    TaskArgs,
    TensorArgType,
    TensorTransfer,
)
from simpler.worker import _host_domain_buffers


@pytest.fixture
def native_domain_lib(arch, runtime):
    root = Path(__file__).resolve().parents[3] / "build/lib"
    ctypes.CDLL(str(root / "libcpu_sim_context.so"), mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL(str(root / arch / "sim" / runtime / "libhost_runtime.so"))
    u64, u32, size, ptr = ctypes.c_uint64, ctypes.c_uint32, ctypes.c_size_t, ctypes.c_void_p
    for name, args in {
        "prepare": [u64, u32, u32, size, u32, u32, ptr, ptr],
        "import": [u64, ptr, size, ptr],
        "window": [u64, u32, ptr, ptr],
        "copy": [u64, u64, u64, size, ptr],
        "release": [u64, u32],
    }.items():
        function = getattr(lib, "comm_global_domain_" + name)
        function.argtypes, function.restype = args, ctypes.c_int
    return lib


def _command():
    return GlobalDomainCommand(
        GlobalDomainPhase.PREPARE_EXPORT,
        7,
        3,
        "mixed",
        "sim",
        4096,
        (GlobalDomainMember(2, 0, 0, 0), GlobalDomainMember(5, 0, 1, 1)),
        (GlobalDomainBuffer("payload", 64),),
        owner_instance_id=b"creator1",
        host_windows=(
            GlobalDomainHostWindow(2, HostWindowSpec(0, 96), 12),
            GlobalDomainHostWindow(5, HostWindowSpec(1, 128), 13),
        ),
    )


def _exported_command():
    command = _command()
    return replace(
        command,
        phase=GlobalDomainPhase.IMPORT,
        descriptors=tuple(
            GlobalDomainDescriptor(GLOBAL_DOMAIN_VERSION, 1, slot, 2, size, b"backing")
            for slot, size in enumerate((4096, 4096, 96, 128))
        ),
        backings=_backing_descriptors(),
    )


def _hbm_descriptor():
    return BufferDescriptor(
        CanonicalIdentity(b"hbmnode2", 10, 1),
        AddressSpace.DEVICE,
        AccessMode.READWRITE,
        BackendKind.REMOTE_SIDECAR,
        64,
    )


def _backing_descriptors():
    return tuple(
        replace(
            buffer_reference(_hbm_descriptor()), identity=CanonicalIdentity(b"hbmnode2", 10 + rank, 1)
        ).to_descriptor()
        for rank in range(2)
    )


def _prepared_parent_state(worker, command):
    import simpler.worker as module  # noqa: PLC0415

    members = tuple(replace(member, buffers={}) for member in command.members)
    backings = list(command.backings)
    for member in members:
        if member.node_worker_id != 2:
            continue
        for index, spec in enumerate(command.buffers):
            descriptor = BufferDescriptor(
                worker._burn_buffer_identity(),
                AddressSpace.DEVICE,
                AccessMode.READWRITE,
                BackendKind.REMOTE_SIDECAR,
                spec.nbytes,
            )
            member.buffers[spec.name] = buffer_reference(descriptor)
            backings[member.domain_rank * len(command.buffers) + index] = descriptor
    state = module._GlobalNodeDomainState(
        command=replace(command, members=members, phase=GlobalDomainPhase.PREPARE_EXPORT, descriptors=(), backings=()),
        prepared_domain_ranks={0},
        descriptors={0: command.descriptors[0], 2: command.descriptors[2]},
        local_window_bases={0: 4096},
        mapping_sizes={0: 4096},
        node_worker_id=2,
    )
    return state, replace(command, backings=tuple(backings))


def _committed_parent_state(worker, command, **kwargs):
    import simpler.worker as module  # noqa: PLC0415

    state = module._GlobalNodeDomainState(command, phase=GlobalDomainPhase.IMPORT, **kwargs)
    sources = module._domain_buffer_descriptors(command)
    for descriptor in sources:
        state.imports.register_borrowed(descriptor, 0, live=lambda: state.phase is GlobalDomainPhase.COMMIT)
    worker._domain_members.setdefault(worker._global_domain_provenance_id(command.domain_id), set()).update(
        descriptor.identity for descriptor in sources
    )
    state.phase = GlobalDomainPhase.COMMIT
    return state


def _local_host_part(worker, domain, node):
    from tests.st.worker.comm_domain.host_domain_matrix import local_host_part  # noqa: PLC0415

    return local_host_part(worker, domain, node)


def _host_handle(command, local_rank=0):
    member = command.members[local_rank]
    members = (
        tuple(
            replace(
                item,
                buffers={
                    spec.name: buffer_reference(command.backings[item.domain_rank * len(command.buffers) + index])
                    for index, spec in enumerate(command.buffers)
                },
            )
            for item in command.members
        )
        if command.backings
        else command.members
    )
    return CommDomainHandle(
        name=command.name,
        members=members,
        contexts={
            member.local_worker_id: ChipDomainContext(command.name, local_rank, len(command.members), 1, 4096, 4096, {})
        },
        buffers=command.buffers,
        domain_id=command.domain_id,
        generation=command.generation,
        host_buffers=_host_domain_buffers(command),
        ref=CommDomainRef(command.owner_instance_id, command.domain_id, command.generation),
        _release_fn=lambda _handle: None,
    )


def test_host_configuration_is_normalized_immutable_and_identity_free():
    request = {5: HostWindowSpec(1, 128), 2: HostWindowSpec(0, 96)}
    config = DomainConfig.capture(((2, 0), (5, 0)), 4096, (), True, (), request)
    assert config.host_windows == tuple(reversed(tuple(request.items())))
    assert config == DomainConfig.capture(config.members, 4096, (), True, (), dict(config.host_windows))
    request[2] = HostWindowSpec(0, 64)
    assert config.host_windows[0][1].nbytes == 96
    with pytest.raises(FrozenInstanceError):
        config.host_windows[0][1].nbytes = 8
    changed = DomainConfig.capture(config.members, 4096, (), True, (), request)
    assert config.difference(changed).startswith("host_windows:")
    with pytest.raises(ValueError, match="exactly"):
        DomainConfig.capture(config.members, 4096, (), True, (), {2: HostWindowSpec(0, 96)})
    with pytest.raises(ValueError, match="L4"):
        DomainConfig.capture((0, 1), 4096, (), True, (), request)


@pytest.mark.parametrize(("numa", "size"), [(-1, 32), (0, 0), (0, -1), (2**31, 32), (0, 2**64)])
def test_host_spec_rejects_invalid_location_and_extent(numa, size):
    with pytest.raises(ValueError):
        HostWindowSpec(numa, size)


@pytest.mark.parametrize(
    "phase", [GlobalDomainPhase.PREPARE_EXPORT, GlobalDomainPhase.IMPORT, GlobalDomainPhase.COMMIT]
)
def test_host_command_and_logical_buffers_preserve_identity(phase):
    command = _command()
    if phase != GlobalDomainPhase.PREPARE_EXPORT:
        command = replace(_exported_command(), phase=phase)
    received = decode_domain_command(encode_domain_command(command))
    assert received == command
    logical = _host_domain_buffers(command)[5]
    mapped = replace(_host_domain_buffers(received)[5], base=8192)
    assert logical.to_descriptor() == mapped.to_descriptor() and mapped.base == 8192
    assert logical.address_space == mapped.address_space == AddressSpace.HOST
    assert logical.backend_kind == BackendKind.REMOTE_SIDECAR and not logical.body
    assert logical.identity != _host_domain_buffers(command)[2].identity
    with pytest.raises(KeyError):
        _host_domain_buffers(command)[99]
    exports = _exported_command().descriptors
    assert decode_descriptor_table(encode_descriptor_table(exports)) == exports
    with pytest.raises(ValueError, match="trailing"):
        decode_descriptor_table(encode_descriptor_table(exports) + b"x")


@pytest.mark.parametrize("phase", [GlobalDomainPhase.IMPORT, GlobalDomainPhase.COMMIT])
def test_host_command_preserves_peer_backing_identity_and_placement(phase):
    command = replace(_exported_command(), phase=phase, backings=_backing_descriptors())
    received = decode_domain_command(encode_domain_command(command))
    assert received == command and received.backings == command.backings
    assert received.backings[0].identity != received.backings[1].identity
    assert all(backing.address_space == AddressSpace.DEVICE and not backing.body for backing in received.backings)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "host", "extent", "access", "unprepared"])
def test_host_command_rejects_unconfirmed_or_inconsistent_peer_backings(mutation):
    command = replace(_exported_command(), backings=_backing_descriptors())
    backings = list(command.backings)
    if mutation == "missing":
        backings.pop()
    elif mutation == "duplicate":
        backings[1] = backings[0]
    elif mutation == "unprepared":
        command = replace(command, phase=GlobalDomainPhase.PREPARE_EXPORT, descriptors=())
    else:
        backings[1] = replace(
            buffer_reference(backings[1]),
            **{
                "host": {"address_space": AddressSpace.HOST},
                "extent": {"nbytes": 32},
                "access": {"access": AccessMode.READ},
            }[mutation],
        ).to_descriptor()
    with pytest.raises(ValueError):
        encode_domain_command(replace(command, backings=tuple(backings)))


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("kind", ["host", "peer-hbm"])
def test_host_map_view_preserves_backing_and_original_tensor(rank, kind):
    handle = _host_handle(replace(_exported_command(), backings=_backing_descriptors()), rank)
    backing = handle.host(2) if kind == "host" else handle.member(rank ^ 1).buffers["payload"]
    source = backing.tensor([8], DataType.UINT8, byte_offset=4)
    mapped = handle.map_view(source, domain_rank=rank)
    assert mapped is not source and mapped.buffer == source.buffer
    assert mapped.buffer.address_space == backing.address_space
    assert (mapped.shapes, mapped.strides, mapped.dtype, mapped.byte_offset) == (
        source.shapes,
        source.strides,
        source.dtype,
        source.byte_offset,
    )
    assert mapped.mapping_ref == rank + 1
    args = TaskArgs()
    args.add_tensor(mapped, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
    assert args.tensor(0).mapping_ref == rank + 1 and args.transfer(0) == TensorTransfer.NONE
    assert source.mapping_ref == 0 and source.buffer == backing.to_descriptor()


@pytest.mark.parametrize("local", [False, True])
def test_host_map_view_requires_existing_local_context(local):
    handle = _host_handle(_exported_command())
    source = handle.host(2).tensor([8], DataType.UINT8)
    if not local:
        handle.contexts.clear()
    with pytest.raises(KeyError):
        handle.map_view(source, domain_rank=1 if local else 0)


@pytest.mark.parametrize("mutation", ["rank", "foreign", "extent", "hbm", "released"])
def test_host_map_view_rejects_ungranted_or_released_views(mutation):
    handle = _host_handle(_exported_command())
    buffer = handle.host(2)
    rank = 0
    if mutation == "rank":
        rank = len(handle.members)
    elif mutation == "foreign":
        buffer = replace(buffer, identity=CanonicalIdentity(b"foreign1", 12, 1))
    elif mutation == "extent":
        buffer = replace(buffer, nbytes=64)
    elif mutation == "hbm":
        buffer = replace(buffer_reference(_hbm_descriptor()), identity=CanonicalIdentity(b"foreign1", 99, 1))
    else:
        handle.release()
    with pytest.raises((ValueError, IndexError, RuntimeError)):
        handle.map_view(buffer.tensor([8], DataType.UINT8), domain_rank=rank)


@pytest.mark.parametrize("mutation", ["order", "collision", "missing", "unprepared", "creator"])
def test_host_wire_rejects_conflicting_or_incomplete_metadata(mutation):
    command = _command()
    if mutation == "order":
        command = replace(command, host_windows=command.host_windows[::-1])
    elif mutation == "collision":
        command = replace(
            command, host_windows=(replace(command.host_windows[0], buffer_id=13), command.host_windows[1])
        )
    elif mutation == "missing":
        command = replace(command, host_windows=command.host_windows[:1])
    elif mutation == "unprepared":
        command = replace(command, descriptors=_exported_command().descriptors)
    else:
        command = replace(command, owner_instance_id=b"")
    with pytest.raises(ValueError):
        encode_domain_command(command)


def test_release_stage_and_old_wire_are_strict():
    for stage in GlobalDomainReleaseStage:
        command = GlobalDomainReleaseCommand(7, 3, stage)
        assert decode_release_command(encode_release_command(command)) == command
    with pytest.raises(ValueError, match="size"):
        decode_release_command(struct.pack("<QQ", 7, 3))
    with pytest.raises(ValueError):
        decode_release_command(struct.pack("<QQI", 7, 3, 99))
    encoded = encode_domain_command(_command())
    with pytest.raises(ValueError, match="format"):
        decode_domain_command(encoded.replace(b"SDC1", b"SDC2"))
    with pytest.raises(ValueError, match="trailing"):
        decode_domain_command(encoded + b"x")


@pytest.mark.parametrize(
    "slot,field,value",
    [
        (3, "domain_rank", 2),
        (3, "domain_rank", 4),
        (3, "mapping_size", 127),
        (1, "mapping_size", 8192),
        (3, "rank_count", 3),
        (3, "version", 2),
        (3, "profile_id", 2),
        (3, "handle", b""),
        (3, "handle", b"x" * 257),
        (3, "mapping_size", 0),
        (3, "mapping_size", 2**64),
    ],
)
def test_window_exports_reject_invalid_metadata(slot, field, value):
    command = _exported_command()
    descriptors = list(command.descriptors)
    descriptors[slot] = replace(descriptors[slot], **{field: value})
    with pytest.raises(ValueError):
        encode_domain_command(replace(command, descriptors=tuple(descriptors)))


@pytest.mark.parametrize("field,value", [("node", -1), ("identity", 0)])
def test_host_config_rejects_invalid_wire(field, value):
    wire = bytearray(encode_domain_command(_exported_command()))
    first_host = len(wire) - 4 - 88 * len(_exported_command().backings) - 2 * struct.calcsize("<iiQQ")
    struct.pack_into("<i" if field == "node" else "<Q", wire, first_host + (0 if field == "node" else 16), value)
    with pytest.raises(ValueError):
        decode_domain_command(bytes(wire))


def test_window_exports_reject_missing_and_excess_entries():
    command = _exported_command()
    with pytest.raises(ValueError):
        encode_domain_command(replace(command, descriptors=command.descriptors[:-1]))
    with pytest.raises(ValueError):
        encode_descriptor_table(command.descriptors * 33)


@pytest.mark.parametrize("mutation", ["wrong-node", "duplicate-slot", "missing-host"])
def test_exports_collector_checks_node_ownership(monkeypatch, mutation):
    from simpler.worker import Worker  # noqa: PLC0415

    command = _exported_command()
    exports = command.descriptors
    replies: dict[int, tuple[GlobalDomainDescriptor, ...]] = {2: (exports[0], exports[2]), 5: (exports[1], exports[3])}
    if mutation == "wrong-node":
        replies[2], replies[5] = replies[5], replies[2]
    elif mutation == "duplicate-slot":
        replies[2] = (exports[0], exports[0])
    else:
        replies[5] = replies[5][:1]
    worker = Worker(level=4, num_sub_workers=0)
    monkeypatch.setattr(
        worker,
        "_global_domain_control",
        lambda node, *_args: encode_descriptor_table(replies[node])
        + encode_domain_buffer_refs([(command.backings[0 if node == 2 else 1], 0, 0)]),
    )
    try:
        expected_error = ValueError if mutation == "missing-host" else RuntimeError
        with pytest.raises(expected_error, match="trailing" if mutation == "missing-host" else "node windows"):
            worker._collect_global_domain_buffers(
                replace(command, phase=GlobalDomainPhase.PREPARE_EXPORT, descriptors=(), backings=()), (2, 5), []
            )
    finally:
        worker.close()


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "missing",
        "extra",
        "duplicate",
        "space",
        "access",
        "backend",
        "extent",
        "owner",
        "id",
        "path",
        "ref",
        "generation",
        "truncated",
        "trailing",
    ],
)
def test_prepare_collector_preserves_owner_refs_and_rejects_unconfirmed_records(monkeypatch, mutation):
    from simpler.worker import Worker  # noqa: PLC0415

    command = _exported_command()
    exports = command.descriptors
    refs = [bytearray(encode_domain_buffer_refs([(descriptor, 0, 0)])) for descriptor in command.backings]
    if mutation == "missing":
        refs[1] = bytearray()
    elif mutation == "extra":
        refs[1] = bytearray(bytes(refs[1]) * 2)
    elif mutation == "duplicate":
        refs[1][8:40] = refs[0][8:40]
    elif mutation == "owner":
        refs[1][8:16] = bytes(8)
    elif mutation in ("space", "access", "backend", "extent", "id", "path", "ref", "generation"):
        offset, fmt, value = {
            "space": (2, "B", int(AddressSpace.HOST)),
            "access": (3, "B", int(AccessMode.READ)),
            "backend": (4, "B", int(BackendKind.DEVICE_MALLOC)),
            "extent": (40, "Q", 32),
            "id": (16, "Q", 0),
            "path": (48, "I", 1),
            "ref": (88, "Q", 7),
            "generation": (96, "Q", 3),
        }[mutation]
        struct.pack_into("<" + fmt, refs[1], offset, value)
    elif mutation == "truncated":
        refs[1] = refs[1][:-1]
    elif mutation == "trailing":
        refs[1] += b"x"
    replies = {
        2: encode_descriptor_table((exports[0], exports[2])) + bytes(refs[0]),
        5: encode_descriptor_table((exports[1], exports[3])) + bytes(refs[1]),
    }
    worker = Worker(level=4, num_sub_workers=0)
    monkeypatch.setattr(worker, "_remote_like_worker_ids", lambda: {5})
    monkeypatch.setattr(worker, "_global_domain_control", lambda node, *_args: replies[node])
    prepared = []
    prepare = replace(command, phase=GlobalDomainPhase.PREPARE_EXPORT, descriptors=(), backings=())
    try:
        if mutation:
            with pytest.raises((ValueError, RuntimeError)):
                worker._collect_global_domain_buffers(prepare, (2, 5), prepared)
        else:
            descriptors, members = worker._collect_global_domain_buffers(prepare, (2, 5), prepared)
            assert descriptors == exports
            assert tuple(members[rank]["payload"].to_descriptor() for rank in range(2)) == command.backings
            assert all(members[rank]["payload"]._remote_ref is None for rank in range(2))
        assert prepared == [2, 5]
        assert not worker._child_alloc and not worker._live_global_domains
    finally:
        worker.close()


@pytest.mark.parametrize("mutation", [None, "identity", "extent", "locator", "generation"])
def test_commit_collector_preserves_prepared_canonical_sources(monkeypatch, mutation):
    from simpler.worker import Worker  # noqa: PLC0415

    command = replace(_exported_command(), phase=GlobalDomainPhase.COMMIT)
    refs = [
        bytearray(encode_domain_buffer_refs([(descriptor, rank + 1, 3)]))
        for rank, descriptor in enumerate(command.backings)
    ]
    if mutation:
        offset, fmt, value = {
            "identity": (16, "Q", 99),
            "extent": (40, "Q", 32),
            "locator": (88, "Q", 0),
            "generation": (96, "Q", 0),
        }[mutation]
        struct.pack_into("<" + fmt, refs[1], offset, value)
    worker = Worker(level=4, num_sub_workers=0)
    monkeypatch.setattr(worker, "_remote_like_worker_ids", lambda: {2, 5})
    monkeypatch.setattr(worker, "_global_domain_control", lambda node, *_args: bytes(refs[0 if node == 2 else 1]))
    try:
        if mutation:
            with pytest.raises(RuntimeError, match="conflicting Buffer"):
                worker._collect_global_domain_buffers(command, (2, 5))
        else:
            descriptors, members = worker._collect_global_domain_buffers(command, (2, 5))
            assert descriptors == command.descriptors
            for rank in range(2):
                assert members[rank]["payload"].to_descriptor() == command.backings[rank]
                assert members[rank]["payload"]._remote_ref == (rank + 1, 3)
    finally:
        worker.close()


@pytest.mark.parametrize("length", [0, 127, 129])
def test_host_fabric_export_requires_exact_token_size(length):
    command = _exported_command()
    exports = tuple(
        replace(d, profile_id=2, handle=b"x" * (length if d.domain_rank == 3 else 128)) for d in command.descriptors
    )
    with pytest.raises(ValueError):
        encode_domain_command(replace(command, profile="a3-fabric-v1", descriptors=exports))


@pytest.mark.parametrize("arch", ["a2a3", "a5"])
@pytest.mark.parametrize("runtime", ["host_build_graph", "tensormap_and_ringbuffer"])
@pytest.mark.parametrize("oversized", [False, True])
def test_native_host_views_copy_and_two_stage_release(native_domain_lib, oversized):
    lib, u64, size = native_domain_lib, ctypes.c_uint64, ctypes.c_size_t
    domain_id = int.from_bytes(os.urandom(8), "little") or 1
    owner = SharedMemory(create=True, size=128)
    try:
        assert owner.buf is not None
        owner.buf[:128] = bytes(range(128))
        descriptor, local_base = ctypes.create_string_buffer(288), u64()
        assert lib.comm_global_domain_prepare(domain_id, 0, 1, 128, 1, 1, descriptor, ctypes.byref(local_base)) == 0
        token = ("/" + owner.name).encode()
        host = struct.pack(
            "<IIIIQII256s",
            GLOBAL_DOMAIN_VERSION,
            1,
            1,
            1,
            owner.size + 4096 if oversized else 128,
            len(token),
            0,
            token,
        )
        context = u64()
        table = ctypes.create_string_buffer(descriptor.raw + host)
        rc = getattr(lib, "comm_global_domain_import")(domain_id, table, 2, ctypes.byref(context))
        if oversized:
            assert rc != 0
            return
        assert rc == 0
        mapped, mapped_size = u64(), size()
        assert lib.comm_global_domain_window(domain_id, 1, ctypes.byref(mapped), ctypes.byref(mapped_size)) == 0
        assert mapped_size.value == 128
        assert ctypes.string_at(mapped.value, 128) == bytes(range(128))
        from _task_interface import read_args_from_blob  # noqa: PLC0415
        from simpler.task_interface import _materialize_chip_task_args  # noqa: PLC0415
        from simpler.worker import Worker  # noqa: PLC0415

        from tests.ut.py.test_worker._wire_blob import encode_blob  # noqa: PLC0415

        source = BufferDescriptor(
            CanonicalIdentity(b"hostnode", domain_id, 1),
            AddressSpace.HOST,
            AccessMode.READWRITE,
            BackendKind.REMOTE_SIDECAR,
            128,
        )
        registry = ImportRegistry(ImportContext(DEVICE_AICPU))
        registry.register_borrowed(source, mapped.value, mapping=1)
        original = source.tensor([8], DataType.UINT8, byte_offset=16)
        tensor = original._with_mapping_ref(1)
        args = TaskArgs()
        args.add_tensor(tensor, transfer=TensorTransfer.NONE)
        direct = Worker(level=2, num_sub_workers=0)
        direct._chip_import_registry = registry
        blob = ctypes.create_string_buffer(encode_blob([tensor], transfers=[TensorTransfer.NONE]))
        try:
            for bound in (
                direct._materialize_l2_args(args),
                _materialize_chip_task_args(read_args_from_blob(ctypes.addressof(blob), len(blob)), registry),
            ):
                assert bound.tensor(0).data == mapped.value + 16
                assert bound.tensor(0).address_space == AddressSpace.DEVICE
                assert ctypes.string_at(bound.tensor(0).data, 8) == bytes(range(16, 24))
            assert original.mapping_ref == 0 and tensor.buffer == source
            assert registry.require(source, mapping=1).descriptor == source
        finally:
            direct.close()
        assert lib.comm_global_domain_copy(domain_id, local_base.value + 32, mapped.value + 16, 32, None) == 0
        assert lib.comm_global_domain_copy(domain_id, mapped.value + 64, local_base.value + 32, 32, None) == 0
        expected = bytearray(range(128))
        expected[64:96] = bytes(range(16, 48))
        assert bytes(owner.buf[:128]) == expected  # Includes both unchanged guards.
        assert lib.comm_global_domain_copy(domain_id, mapped.value + 16, mapped.value + 20, 32, None) != 0
        assert lib.comm_global_domain_copy(domain_id, mapped.value + 120, local_base.value, 16, None) != 0
        assert lib.comm_global_domain_copy(domain_id, mapped.value + 64, local_base.value + 120, 16, None) != 0
        unrelated = ctypes.create_string_buffer(128)
        unrelated_address = ctypes.addressof(unrelated)
        assert lib.comm_global_domain_copy(domain_id, unrelated_address, mapped.value, 16, None) != 0
        assert lib.comm_global_domain_copy(domain_id, mapped.value, unrelated_address, 16, None) != 0
        assert lib.comm_global_domain_copy(domain_id, 2**64 - 8, mapped.value, 16, None) != 0
        assert lib.comm_global_domain_copy(domain_id, mapped.value, 2**64 - 8, 16, None) != 0
        assert bytes(owner.buf[:128]) == expected
        assert lib.comm_global_domain_release(domain_id, 0) != 0
        assert lib.comm_global_domain_release(domain_id, 2) != 0
        assert lib.comm_global_domain_release(domain_id, 1) == 0
        assert lib.comm_global_domain_window(domain_id, 1, ctypes.byref(mapped), ctypes.byref(mapped_size)) != 0
        assert lib.comm_global_domain_copy(domain_id, mapped.value + 64, local_base.value + 32, 32, None) != 0
        assert ctypes.string_at(local_base.value, 16) == bytes(16)  # HBM owner still alive after VIEWS.
        assert lib.comm_global_domain_release(domain_id, 2) == 0
        assert lib.comm_global_domain_release(domain_id, 2) == 0
    finally:
        lib.comm_global_domain_release(domain_id, 1)
        lib.comm_global_domain_release(domain_id, 2)
        owner.close()
        owner.unlink()


@pytest.mark.parametrize("arch", ["a2a3", "a5"])
@pytest.mark.parametrize("runtime", ["host_build_graph", "tensormap_and_ringbuffer"])
def test_native_partial_import_keeps_views_unpublished_until_release(native_domain_lib):
    lib, u64, size = native_domain_lib, ctypes.c_uint64, ctypes.c_size_t
    domain_ids = tuple(int.from_bytes(os.urandom(8), "little") or 1 for _ in range(2))
    owner = SharedMemory(create=True, size=128)
    try:
        assert owner.buf is not None
        expected = bytes(range(128))
        owner.buf[:128] = expected
        descriptors, local_bases = [], []
        for rank, domain_id in enumerate(domain_ids):
            descriptor, base = ctypes.create_string_buffer(288), u64()
            assert lib.comm_global_domain_prepare(domain_id, rank, 2, 128, 1, 1, descriptor, ctypes.byref(base)) == 0
            descriptors.append(descriptor.raw)
            local_bases.append(base.value)
        host = GlobalDomainDescriptor(GLOBAL_DOMAIN_VERSION, 1, 2, 2, 128, ("/" + owner.name).encode())
        missing = replace(host, domain_rank=3, handle=("/missing_" + os.urandom(8).hex()).encode())
        table = ctypes.create_string_buffer(b"".join(descriptors) + host.encode() + missing.encode())
        context = u64()
        assert getattr(lib, "comm_global_domain_import")(domain_ids[0], table, 4, ctypes.byref(context)) != 0
        assert context.value == 0
        assert getattr(lib, "comm_global_domain_import")(domain_ids[0], table, 4, ctypes.byref(context)) != 0
        for slot in range(4):
            mapped, mapped_size = u64(), size()
            assert (
                lib.comm_global_domain_window(domain_ids[0], slot, ctypes.byref(mapped), ctypes.byref(mapped_size)) != 0
            )
        host_base = ctypes.addressof(ctypes.c_char.from_buffer(owner.buf))
        assert lib.comm_global_domain_copy(domain_ids[0], host_base + 32, host_base, 16, None) != 0
        assert lib.comm_global_domain_copy(domain_ids[0], local_bases[0] + 32, host_base, 16, None) != 0
        assert bytes(owner.buf[:128]) == expected
        assert lib.comm_global_domain_release(domain_ids[0], 2) != 0
        assert lib.comm_global_domain_release(domain_ids[0], 1) == 0
        assert ctypes.string_at(local_bases[0], 128) == bytes(128)
        assert bytes(owner.buf[:128]) == expected
        assert lib.comm_global_domain_release(domain_ids[0], 2) == 0
    finally:
        for domain_id in domain_ids:
            lib.comm_global_domain_release(domain_id, 1)
            lib.comm_global_domain_release(domain_id, 2)
        owner.close()
        owner.unlink()


def test_two_node_host_cpu_copy_reuse_and_stale_access():
    from functools import partial  # noqa: PLC0415

    from simpler.comm_domain import CommDomainRef  # noqa: PLC0415
    from simpler.orchestrator import Orchestrator  # noqa: PLC0415
    from simpler.task_interface import CallConfig, CommBufferSpec, TaskArgs  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    saved_parts = []
    nodes = (0, 1)

    def node_task(orch, args, _config):
        node, phase = int(args.scalar(0)), int(args.scalar(1))
        if phase == 2:
            with pytest.raises((KeyError, RuntimeError)):
                saved_parts[0].read(0, bytearray(1))
            return
        domain = orch.get_domain(
            CommDomainRef(int(args.scalar(2)).to_bytes(8, "little"), int(args.scalar(3)), int(args.scalar(4)))
        )
        assert all(buffer.address_space != AddressSpace.HOST for buffer in orch._worker._child_alloc.values())
        part = _local_host_part(orch._worker, domain, node)
        local = domain.host(node)

        def copy(dst, src, chip, nbytes, dst_offset=0, src_offset=0):
            orch._worker._copy_domain_buffers(
                domain.ref,
                worker=chip,
                dst=dst,
                src=src,
                nbytes=nbytes,
                dst_offset=dst_offset,
                src_offset=src_offset,
            )

        if phase == 0:
            saved_parts.append(part)
            data = bytes([node + 11]) * 128 + bytes([0xCC]) * (1536 - 128)
            part.write(0, data)
            for member in domain.members:
                if member.node_worker_id == node:
                    copy(member.buffers["payload"], local, member.local_worker_id, 512)
            return
        pools = [(domain.host(n), n) for n in nodes]
        pools += [(member.buffers["payload"], member.node_worker_id) for member in domain.members]
        for chip in (0, 1):
            for pool, owner in pools:
                copy(local, pool, chip, 128, dst_offset=512)
                result = bytearray(128)
                part.read(512, result)
                assert result == bytes([owner + 11]) * 128
                copy(pool, local, chip, 64, dst_offset=128)
                copy(local, pool, chip, 512, dst_offset=512)
                result = bytearray(512)
                part.read(512, result)
                assert result == bytes([owner + 11]) * 128 + bytes([node + 11]) * 64 + bytes([0xCC]) * 320
        with pytest.raises(ValueError, match="overlap"):
            copy(local, local, 0, 64, dst_offset=16)
        with pytest.raises(ValueError, match="range"):
            copy(local, local, 0, 64, dst_offset=1530)
        hbm = domain.member(0).buffers["payload"]
        for dst, src in ((hbm, domain.member(1).buffers["payload"]), (hbm, domain.host(node ^ 1))):
            with pytest.raises(ValueError, match="this node's Host"):
                copy(dst, src, 0, 64)
        for count, offset in ((0, 0), (-1, 0), (64, -1)):
            with pytest.raises(ValueError):
                copy(local, hbm, 0, count, dst_offset=offset)
        with pytest.raises((ValueError, KeyError, RuntimeError)):
            copy(local, replace(hbm, identity=CanonicalIdentity(b"foreign1", 999, 1)), 0, 64)
        with pytest.raises((ValueError, KeyError, RuntimeError)):
            copy(local, hbm, 999, 64)
        with pytest.raises((ValueError, KeyError, RuntimeError)):
            orch._worker._copy_domain_buffers(
                replace(domain.ref, generation=domain.ref.generation + 1),
                worker=0,
                dst=local,
                src=hbm,
                nbytes=64,
            )

    root = Worker(level=4, num_sub_workers=0)
    task = root.register(node_task)
    captured = []
    try:
        for node in nodes:
            assert (
                root.add_worker(
                    Worker(
                        level=3,
                        platform="a2a3sim",
                        runtime="tensormap_and_ringbuffer",
                        device_ids=[0, 1],
                        num_sub_workers=0,
                        comm_profile="sim",
                        global_device_ranks=[2 * node, 2 * node + 1],
                    )
                )
                == node
            )
        root.init()
        request = dict(
            name="mixed",
            members=tuple((n, c) for n in nodes for c in (0, 1)),
            window_size=512,
            buffers=[CommBufferSpec("payload", "uint8", 512, 512)],
            retain_after_run=True,
            host_windows={n: HostWindowSpec(0, 1536) for n in nodes},
        )

        def allocate(orch, _args, _config):
            domain = orch.get_or_allocate_domain(**request)
            assert orch.get_or_allocate_domain(**request) is domain
            captured.append(domain)

        root.run(allocate)
        domain = captured[0]
        assert len({b.identity for b in domain._host_buffers.values()}) == 2

        def submit_step(
            orch: Orchestrator,
            _args: TaskArgs | None,
            _config: CallConfig,
            *,
            node: int,
            phase: int,
        ) -> None:
            args = TaskArgs()
            if phase != 2:
                current = orch.get_domain(domain.ref)
            args.add_scalar(node)
            args.add_scalar(phase)
            if phase != 2:
                args.add_scalar(int.from_bytes(current.ref.owner_instance_id, "little"))
                args.add_scalar(current.ref.domain_id)
                args.add_scalar(current.ref.generation)
            orch.submit_next_level(task, args, worker=node)

        for phase in (0, 1, 2):
            if phase == 2:
                assert domain.released and domain.freed
            for node in nodes:
                step = partial(submit_step, node=node, phase=phase)
                if phase == 1 and node == nodes[-1]:
                    pending = root.submit(step)
                    assert pending._resources.live_global_domains.get(domain.name) is domain
                    assert pending._resources.requires_ordered_cleanup
                    domain.release()
                    assert domain.released and not domain.freed
                    pending.wait()
                    assert domain.freed
                else:
                    root.run(step)
        assert not root.live_domains
    finally:
        root.close()


def _chip_import_payload(command, hosts=None):
    from unittest.mock import patch  # noqa: PLC0415

    import simpler.global_comm_domain as codec  # noqa: PLC0415
    import simpler.worker as module  # noqa: PLC0415

    if hosts is not None:
        body = b"".join(descriptor.encode() for descriptor in command.descriptors) + b"".join(
            _encode_buffer_descriptor_wire(descriptor) for descriptor in (*hosts, *command.backings)
        )
    else:
        # Malformed tables deliberately cross the wire; the receiver uses the real validators.
        with patch.object(codec, "validate_backing_table"), patch.object(codec, "validate_descriptor_table"):
            body = encode_domain_command(replace(command, phase=GlobalDomainPhase.IMPORT, attachments=()))
    return (
        module.LOCAL_IMPORT_REQUEST.pack(
            module.LOCAL_DOMAIN_MAGIC, GLOBAL_DOMAIN_VERSION, command.domain_id, command.generation, len(body)
        )
        + body
    )


def _control_payload(data, padding=0):
    shm = SharedMemory(create=True, size=len(data) + padding)
    shm.unlink()
    assert shm.buf is not None
    shm.buf[: len(data)] = data
    return shm, shm.buf, len(data)


@pytest.mark.parametrize("failure", [None, (1, 2), (1, 5), (2, 2), (2, 5)])
def test_owner_release_requires_every_node_view_ack(monkeypatch, failure):
    from simpler.remote_l3_protocol import ControlName  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    worker, calls = Worker(level=4, num_sub_workers=0), []

    def control(node, operation, payload):
        assert operation == ControlName.RELEASE_DOMAIN
        command = decode_release_command(payload)
        calls.append((int(command.stage), node))
        if calls[-1] == failure:
            raise RuntimeError("lost ACK")
        return b""

    monkeypatch.setattr(worker, "_global_domain_control", control)
    monkeypatch.setattr(
        worker, "_mpi_group_for_involved_nodes", lambda _nodes: pytest.fail("Host release must use per-node control")
    )
    try:
        if failure:
            with pytest.raises(RuntimeError, match="unconfirmed"):
                worker._release_host_domain_nodes(7, 3, (2, 5))
        else:
            worker._release_host_domain_nodes(7, 3, (2, 5))
        assert calls == ([(1, 2), (1, 5)] if failure and failure[0] == 1 else [(1, 2), (1, 5), (2, 2), (2, 5)])
    finally:
        worker.close()


def test_native_prepare_failure_remains_reachable_for_staged_release(monkeypatch):
    import simpler.worker as module  # noqa: PLC0415

    store, calls = module._L2GlobalDomainStore(), []

    class Chip:
        def comm_global_domain_prepare(self, *_args):
            raise RuntimeError("physical allocation reply lost")

        def comm_global_domain_release(self, identity, stage):
            calls.append((identity, stage))

    data = bytearray(module.LOCAL_PREPARE_REPLY.size + module.GLOBAL_DOMAIN_DESCRIPTOR_BYTES)
    module.LOCAL_PREPARE_REQUEST.pack_into(
        data, 0, module.LOCAL_DOMAIN_MAGIC, GLOBAL_DOMAIN_VERSION, 7, 3, 0, 1, 1, 128, True
    )
    monkeypatch.setattr(module, "_open_global_domain_payload", lambda _buf: _control_payload(data))
    with pytest.raises(RuntimeError, match="reply lost"):
        module._handle_ctrl_global_domain_prepare(Chip(), memoryview(bytearray()), store)
    assert store.domains[7].descriptor is None and store.domains[7].host_enabled
    for stage in (1, 2):
        data = module.LOCAL_RELEASE_REQUEST.pack(module.LOCAL_DOMAIN_MAGIC, GLOBAL_DOMAIN_VERSION, 7, 3, stage)
        module._handle_ctrl_global_domain_release(Chip(), memoryview(bytearray()), store)
        assert (7 in store.domains) == (stage == 1)
    assert calls == [(7, 1), (7, 2)]


@pytest.mark.parametrize("mode", ["node", "cached-root", "untracked-root"])
def test_host_close_cannot_bypass_global_release(monkeypatch, mode):
    import simpler.worker as module  # noqa: PLC0415
    from simpler.task_interface import CommDomainHandle  # noqa: PLC0415

    worker, shutdown = module.Worker(level=3 if mode == "node" else 4, num_sub_workers=0), []
    state = module._GlobalNodeDomainState(_command())
    owner = object()
    state.host_owner = owner
    if mode == "node":
        worker._global_node_domains[7] = state
    elif mode == "cached-root":
        handle = CommDomainHandle(
            name="mixed",
            _release_fn=lambda _handle: None,
            domain_id=7,
            generation=3,
            members=_command().members,
            host_buffers={2: _host_domain_buffers(_command())[2]},
        )
        worker._live_global_domains["mixed"] = handle
        worker._global_domain_free_results[7] = RuntimeError("lost view ACK")
        worker._defer_domain_retirement(handle)
    else:
        worker._record_unreclaimable("Host rollback unknown", keep_transport=True)
    monkeypatch.setattr(worker, "_release_remote_sessions", lambda: shutdown.append(True))
    try:
        with pytest.raises(RuntimeError):
            worker.close()
        assert state.host_owner is owner and not shutdown
        if mode == "node":
            assert worker._global_node_domains[7] is state
        elif mode == "cached-root":
            assert not worker._cleanup_journal.empty and not handle.freed
            assert worker._global_domain_free_results[7] is not None
    finally:
        # Only synthetic debts: no backing, native context or child process was created.
        worker._global_node_domains.clear()
        worker._live_global_domains.clear()
        worker._global_domain_free_results.clear()
        worker._cleanup_journal = type(worker._cleanup_journal)()
        with pytest.raises(RuntimeError):  # A failed close preserves its published error.
            worker.close()


@pytest.mark.parametrize(
    "mutation",
    [
        "valid",
        "missing",
        "revoked",
        "generation",
        "truncated",
        "native-failure",
        "unregistered",
        "descriptor",
        "source-access",
        "destination-access",
        "source-range",
        "overflow",
        "overlap",
    ],
)
def test_buffer_copy_reuses_domain_registry_and_revokes_unknown_completion(monkeypatch, mutation):
    import simpler.worker as module  # noqa: PLC0415

    entry = module._L2GlobalDomain(7, 3, 0, 2, None, 0, 4096, 4096, device_ctx=123, host_enabled=True)
    store = module._L2GlobalDomainStore({7: entry} if mutation != "missing" else {})
    registry = ImportRegistry(ImportContext(DEVICE_AICPU))
    destination = _host_domain_buffers(_command())[2].to_descriptor()
    source = _hbm_descriptor()
    if mutation == "source-access":
        source = replace(buffer_reference(source), access=AccessMode.WRITE).to_descriptor()
    elif mutation == "destination-access":
        destination = replace(buffer_reference(destination), access=AccessMode.READ).to_descriptor()
    elif mutation == "overlap":
        source = destination
    entry.buffer_descriptors = (destination, source)
    registry.register_borrowed(destination, 4096, mapping=1, live=lambda: not entry.revoked)
    registry.register_borrowed(
        source, 4096 if source == destination else 8192, mapping=1, live=lambda: not entry.revoked
    )
    if mutation == "revoked":
        entry.revoked = True
    elif mutation == "generation":
        entry.generation += 1
    elif mutation == "unregistered":
        registry.unregister(source.identity)
    elif mutation == "descriptor":
        source = replace(buffer_reference(source), nbytes=32).to_descriptor()
    src_offset = 60 if mutation == "source-range" else 2**64 - 1 if mutation == "overflow" else 4
    payload = module.LOCAL_POOL_COPY.pack(
        module.LOCAL_DOMAIN_MAGIC, GLOBAL_DOMAIN_VERSION, 7, 3, 8, src_offset, 8
    ) + b"".join(_encode_buffer_descriptor_wire(descriptor) for descriptor in (destination, source))
    if mutation == "truncated":
        payload = payload[:-1]
    monkeypatch.setattr(module, "_open_global_domain_payload", lambda _buf: _control_payload(payload))
    calls = []

    class Chip:
        def comm_global_domain_copy(self, *args):
            calls.append(args)
            if mutation == "native-failure":
                raise RuntimeError("completion unknown")

    try:
        if mutation == "valid":
            module._handle_ctrl_domain_pool_copy(Chip(), memoryview(bytearray()), store, registry)
            assert calls == [(7, 4104, 8196, 8)]
        elif mutation == "native-failure":
            with pytest.raises(RuntimeError, match="unknown"):
                module._handle_ctrl_domain_pool_copy(Chip(), memoryview(bytearray()), store, registry)
            assert entry.revoked
            with pytest.raises(ValueError, match="not live"):
                registry.materialize(source, mapping=1)
            with pytest.raises(ValueError):
                module._handle_ctrl_domain_pool_copy(Chip(), memoryview(bytearray()), store, registry)
            assert calls == [(7, 4104, 8196, 8)]
        else:
            with pytest.raises((ValueError, KeyError)):
                module._handle_ctrl_domain_pool_copy(Chip(), memoryview(bytearray()), store, registry)
            assert not calls and entry.revoked == (mutation == "revoked")
    finally:
        registry.close()


@pytest.mark.parametrize("field,value", [("handle", b"changed"), ("mapping_size", 8192)])
def test_import_freezes_this_nodes_exports_before_dispatch(field, value):
    import simpler.worker as module  # noqa: PLC0415

    command = _exported_command()
    worker = module.Worker(level=3, num_sub_workers=0)
    state = module._GlobalNodeDomainState(
        replace(command, phase=GlobalDomainPhase.PREPARE_EXPORT, descriptors=(), backings=()), node_worker_id=2
    )
    state.descriptors = {0: command.descriptors[0], 2: command.descriptors[2]}
    worker._global_node_domains[7] = state
    worker._worker = object()
    changed = list(command.descriptors)
    changed[2] = replace(changed[2], **{field: value})
    try:
        with pytest.raises(RuntimeError, match="exported backing"):
            worker._import_global_domain_node(replace(command, descriptors=tuple(changed)), 2)
    finally:
        worker._worker = None
        worker._global_node_domains.clear()
        worker.close()


@pytest.mark.parametrize("field,value", [("handle", b"changed"), ("mapping_size", 8192)])
def test_commit_freezes_complete_host_exports(field, value):
    import simpler.worker as module  # noqa: PLC0415
    from simpler.task_interface import GlobalCommDomainView  # noqa: PLC0415

    command = _exported_command()
    worker = module.Worker(level=3, num_sub_workers=0)
    view = GlobalCommDomainView(
        name="mixed", members=command.members, contexts={}, domain_id=7, generation=3, mapping_size=4096
    )
    state = module._GlobalNodeDomainState(command, phase=GlobalDomainPhase.IMPORT, view=view)
    worker._global_node_domains[7] = state
    try:
        commit = replace(command, phase=GlobalDomainPhase.COMMIT)
        changed = replace(
            commit, descriptors=(*commit.descriptors[:-1], replace(commit.descriptors[-1], **{field: value}))
        )
        with pytest.raises(RuntimeError, match="conflicts"):
            worker._commit_global_domain_node(changed)
        assert not view.committed
        worker._commit_global_domain_node(commit)
        assert view.committed
    finally:
        worker._global_node_domains.clear()
        worker.close()


def test_shared_host_targets_preserve_exact_descriptor_and_hbm_chip_checks():
    import simpler.worker as module  # noqa: PLC0415

    worker = module.Worker(level=3, num_sub_workers=0)
    command = _exported_command()
    host = _host_domain_buffers(command)[2]
    handle = _host_handle(command)
    hbm = buffer_reference(_hbm_descriptor(), owner_worker_id=0)
    worker._global_node_domains[7] = _committed_parent_state(worker, command, node_worker_id=2)
    try:
        with worker._child_prov_lock:
            assert host.identity not in worker._child_alloc
            worker._record_device_alloc(hbm)
            args = TaskArgs()
            args.add_tensor(
                handle.map_view(host.tensor([8], DataType.UINT8), domain_rank=0), transfer=TensorTransfer.NONE
            )
            worker._prepare_buffer_references(args, None, 0)
            worker._child_prov_check_dispatch_locked(args, 0, api="test")
            with pytest.raises(ValueError):
                worker._prepare_buffer_references(args, None, 999)
            changed = TaskArgs()
            changed.add_tensor(
                replace(host, nbytes=64).tensor([8], DataType.UINT8)._with_mapping_ref(1),
                transfer=TensorTransfer.NONE,
            )
            with pytest.raises(ValueError):
                worker._prepare_buffer_references(changed, None, 0)
            wrong_rank = TaskArgs()
            wrong_rank.add_tensor(host.tensor([8], DataType.UINT8)._with_mapping_ref(2), transfer=TensorTransfer.NONE)
            with pytest.raises(ValueError):
                worker._prepare_buffer_references(wrong_rank, None, 0)
            hbm_args = TaskArgs()
            hbm_args.add_tensor(hbm.tensor([8], DataType.UINT8))
            worker._prepare_buffer_references(hbm_args, None, 0)
            with pytest.raises(ValueError):
                worker._prepare_buffer_references(hbm_args, None, 1)
        with pytest.raises(ValueError, match="not a live device allocation"):
            worker._device_worker_for(host.identity, api="test")
    finally:
        worker._global_node_domains.clear()
        worker._clear_child_prov()
        worker.close()


def test_borrowed_host_identity_has_endpoint_local_mappings_and_revocation():
    descriptor = _host_domain_buffers(_command())[2].to_descriptor()
    registries = [ImportRegistry(ImportContext(DEVICE_AICPU)) for _ in range(2)]
    host = ImportRegistry(ImportContext(HOST_CPU))
    try:
        for mapping, (registry, base) in enumerate(zip(registries, (4096, 8192)), start=1):
            registry.register_borrowed(descriptor, base, mapping=mapping)
            registry.register_borrowed(descriptor, base, mapping=mapping)
            imported = registry.materialize(descriptor, mapping=mapping)
            assert imported.base == base and imported.address_space == AddressSpace.DEVICE
            assert imported.descriptor == descriptor and descriptor.address_space == AddressSpace.HOST
            with pytest.raises(ValueError):
                registry.materialize(descriptor)
            with pytest.raises(ValueError):
                registry.materialize(descriptor, mapping=3)
            with pytest.raises(ValueError, match="different target"):
                registry.register_borrowed(descriptor, base, mapping=3)
            with pytest.raises(ValueError, match="conflict"):
                registry.register_borrowed(descriptor, base + 64, mapping=mapping)
            changed = replace(buffer_reference(descriptor), nbytes=64).to_descriptor()
            with pytest.raises(ValueError, match="conflict"):
                registry.register_borrowed(changed, base, mapping=mapping)
            with pytest.raises(ValueError, match="different descriptor"):
                registry.materialize(changed, mapping=mapping)
        with pytest.raises(ValueError, match="Fabric binding"):
            host.register_borrowed(descriptor, 4096, mapping=1)
        host.register_borrowed(descriptor, 0)
        for need in (BufferCapability.COPY_TO, BufferCapability.COPY_FROM):
            assert host.require(descriptor, need, offset=8, nbytes=16).base == 0
        for need in (BufferCapability.DIRECT_LOAD, BufferCapability.DIRECT_STORE):
            with pytest.raises(ValueError):
                host.require(descriptor, need)
        with pytest.raises(ValueError):
            host.materialize(descriptor)
        registries[0].unregister(descriptor.identity)
        with pytest.raises(KeyError, match="no buffer registered"):
            registries[0].require(descriptor.identity)
        assert registries[1].materialize(descriptor, mapping=2).base == 8192
        with pytest.raises(ValueError):
            registries[0].materialize(descriptor, mapping=1)
    finally:
        for registry in (*registries, host):
            registry.close()


def test_host_mapping_rejects_nested_worker_forwarding(monkeypatch):
    import simpler.worker as module  # noqa: PLC0415

    worker = module.Worker(level=3, num_sub_workers=0)
    handle = _host_handle(_exported_command(), 1)
    args = TaskArgs()
    args.add_tensor(
        handle.map_view(handle.host(2).tensor([8], DataType.UINT8), domain_rank=1),
        TensorArgType.INPUT,
        transfer=TensorTransfer.NONE,
    )
    monkeypatch.setattr(module, "_buffer_field_addr", lambda *_: 0)
    monkeypatch.setattr(module, "read_args_from_blob", lambda *_: args)
    try:
        with pytest.raises(ValueError, match="mapping"):
            module._reexport_args_from_mailbox(None, worker)
    finally:
        worker.close()


@pytest.mark.parametrize("kind", ["host", "hbm"])
@pytest.mark.parametrize("state", ["live", "revoked", "closed-wrapper"])
def test_domain_buffer_cannot_bypass_collective_release(monkeypatch, kind, state):
    from simpler.buffer import wrap_vmm_window  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    worker = Worker(level=3, num_sub_workers=0)
    buffer = (
        _host_domain_buffers(_command())[2]
        if kind == "host"
        else wrap_vmm_window(4096, 64, worker._owner_instance_id, 41, owner_worker_id=0)
    )
    calls = []
    monkeypatch.setattr(worker, "_release_import_recursive", lambda *_: calls.append("unregister"))
    try:
        with worker._child_prov_lock:
            worker._record_device_alloc(buffer, domain_allocation_id=7)
        if state == "revoked":
            worker._drop_domain_allocs(7)
        elif state == "closed-wrapper":
            buffer.close()
        with pytest.raises(ValueError):
            worker.release_buffer(buffer)
        assert not calls
    finally:
        worker._clear_child_prov()
        worker.close()


@pytest.mark.parametrize("profile", ["sim", "a3-fabric-v1"])
def test_borrowed_host_access_outside_callback_does_not_reenter_serializer(monkeypatch, profile):
    import simpler.worker as module  # noqa: PLC0415
    from simpler.comm_domain import CommDomainRef  # noqa: PLC0415
    from simpler.task_interface import GlobalCommDomainView  # noqa: PLC0415

    command = replace(_exported_command(), profile=profile)
    worker = module.Worker(level=3, num_sub_workers=0)
    owner = SharedMemory(create=True, size=96)
    assert owner.buf is not None
    owner.buf[:] = bytes(len(owner.buf))
    base = ctypes.addressof(ctypes.c_char.from_buffer(owner.buf))
    calls = []

    class HostOwner:
        @property
        def base(self):
            assert worker._child_prov_lock.locked()
            calls.append("base")
            return base

    view = GlobalCommDomainView(
        name=command.name, members=command.members, contexts={}, domain_id=7, generation=3, mapping_size=4096
    )
    state = _committed_parent_state(
        worker,
        command,
        view=view,
        host_owner=owner if profile == "sim" else HostOwner(),
        node_worker_id=2,
    )
    worker._global_node_domains[7] = state
    worker._lifecycle = module._Lifecycle.READY
    errors = []
    completed = []
    memmove = ctypes.memmove

    def direct_copy(dst, src, size):
        assert isinstance(dst, int) and isinstance(src, int)
        return memmove(dst, src, size)

    def access():
        try:
            assert module._callback_frame_for(worker) is None
            ref = CommDomainRef(command.owner_instance_id, 7, 3)
            handle = worker._get_domain(ref)
            assert handle.borrowed and worker._get_domain(ref) is handle
            part = _local_host_part(worker, handle, 2)
            source = bytearray(b"head" + bytes(range(96)) + b"tail")
            result = bytearray([0xCC] * 80)
            with monkeypatch.context() as copy_patch:
                copy_patch.setattr(module.ctypes, "memmove", direct_copy)
                copy_patch.setattr(module.ctypes, "string_at", lambda *_: pytest.fail("unexpected bytes staging"))
                part.write(8, memoryview(source)[4:-4], 64)
                part.read(8, memoryview(result)[8:72])
                with pytest.raises(ValueError, match="exceeds part"):
                    part.write(80, source, 32)
                with pytest.raises(ValueError, match="writable"):
                    part.read(0, bytes(4))
            assert result == bytes([0xCC] * 8) + bytes(range(64)) + bytes([0xCC] * 8)
            assert source == b"head" + bytes(range(96)) + b"tail"
            assert owner.buf is not None
            assert bytes(owner.buf[:96]) == bytes(8) + bytes(range(64)) + bytes(24)
            assert not any(owner.buf[96:])
            assert calls == ([] if profile == "sim" else ["base", "base"])
            assert worker._get_domain(ref).host(2).identity == _host_domain_buffers(state.command)[2].identity
            completed.append(True)
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    thread = threading.Thread(target=access, daemon=True)
    try:
        thread.start()
        thread.join(3)
        assert not thread.is_alive(), "external Host access deadlocked in serializer/control admission"
        if errors:
            raise errors[0]
        assert completed == [True]
    finally:
        if not thread.is_alive():
            worker._global_node_domains.clear()
            worker.close()
        owner.close()
        owner.unlink()


@pytest.mark.parametrize("writing", [False, True])
@pytest.mark.parametrize("pointer", [0, -1, 1 << 64, 1.5])
def test_host_byte_access_refuses_invalid_pointer_before_memcpy(monkeypatch, writing, pointer):
    import simpler.worker as module  # noqa: PLC0415
    from simpler.task_interface import GlobalCommDomainView  # noqa: PLC0415

    worker = module.Worker(level=3, num_sub_workers=0)
    command = replace(_exported_command(), profile="a3-fabric-v1")
    view = GlobalCommDomainView(
        name="mixed", members=command.members, contexts={}, domain_id=7, generation=3, mapping_size=4096
    )
    state = _committed_parent_state(worker, command, view=view, host_owner=object(), node_worker_id=2)
    worker._global_node_domains[7] = state
    worker._lifecycle = module._Lifecycle.READY
    monkeypatch.setattr(module.ctypes, "memmove", lambda *_: pytest.fail("invalid pointer reached memcpy"))
    try:
        with pytest.raises((TypeError, ValueError)):
            worker._host_domain_bytes(
                CommDomainRef(command.owner_instance_id, 7, 3),
                _host_domain_buffers(command)[2].to_descriptor(),
                0,
                pointer,
                8,
                writing=writing,
            )
        assert state.phase is GlobalDomainPhase.COMMIT
    finally:
        worker._global_node_domains.clear()
        worker.close()


def test_native_host_owner_rejects_address_without_a_live_mapping():
    from _task_interface import _DomainHostMemory  # noqa: PLC0415

    with pytest.raises(ValueError, match="no live mapping"):
        _ = _DomainHostMemory().base


@pytest.mark.parametrize("partial_failure", [False, True])
def test_host_import_leaves_local_hbm_for_ordinary_vmm_materialization(monkeypatch, partial_failure):
    import simpler.worker as module  # noqa: PLC0415

    command = _exported_command()
    entry = module._L2GlobalDomain(7, 3, 0, 2, command.descriptors[0], 4096, 4096, 4096, host_enabled=True)
    store = module._L2GlobalDomainStore({7: entry})
    registry = ImportRegistry(ImportContext(DEVICE_AICPU, device_owner_instance_id=b"hbmnode2"))
    payload = _chip_import_payload(command)
    monkeypatch.setattr(module, "_open_global_domain_payload", lambda _buf: _control_payload(payload))

    calls = []

    class Chip:
        def comm_global_domain_import(self, domain_id, table):
            calls.append((domain_id, table))
            return 123

        def comm_global_domain_window(self, _domain_id, slot):
            return 4096 * (slot + 1), command.descriptors[slot].mapping_size

    try:
        if partial_failure:
            register = registry.register_borrowed
            registrations = []

            def register_one(descriptor, base, *, mapping, live):
                if registrations:
                    raise RuntimeError("second mapping refused")
                register(descriptor, base, mapping=mapping, live=live)
                registrations.append(descriptor)

            monkeypatch.setattr(registry, "register_borrowed", register_one)
            with pytest.raises(RuntimeError, match="second mapping refused"):
                module._handle_ctrl_global_domain_import(Chip(), memoryview(bytearray()), store, registry)
            assert entry.buffer_descriptors == module._domain_buffer_descriptors(command)
            assert len(calls) == len(registrations) == 1
            assert entry.revoked
            assert registry._by_identity[(registrations[0].identity, 1)].base != 0
            with pytest.raises(ValueError, match="not live"):
                registry.materialize(registrations[0], mapping=1)
            payload = module.LOCAL_RELEASE_REQUEST.pack(
                module.LOCAL_DOMAIN_MAGIC, GLOBAL_DOMAIN_VERSION, 7, 3, int(GlobalDomainReleaseStage.VIEWS)
            )

            class Release:
                def comm_global_domain_release(self, *_):
                    assert entry.revoked
                    for descriptor in registrations:
                        with pytest.raises((KeyError, ValueError)):
                            registry.materialize(descriptor, mapping=1)
                    calls.append("release")

            module._handle_ctrl_global_domain_release(Release(), memoryview(bytearray()), store, registry)
            assert calls[-1] == "release"
            return

        for _ in range(2):
            module._handle_ctrl_global_domain_import(Chip(), memoryview(bytearray()), store, registry)
        assert calls == [(7, b"".join(descriptor.encode() for descriptor in command.descriptors))]
        assert entry.descriptor_table == calls[0][1]
        assert entry.buffer_descriptors == module._domain_buffer_descriptors(command)
        changed = replace(
            command, host_windows=(replace(command.host_windows[0], buffer_id=99), command.host_windows[1])
        )
        payload = _chip_import_payload(changed)
        with pytest.raises(RuntimeError, match="changes its descriptor table"):
            module._handle_ctrl_global_domain_import(Chip(), memoryview(bytearray()), store, registry)
        assert entry.descriptor_table == calls[0][1] and len(calls) == 1
        for descriptor in (_hbm_descriptor(),):
            with pytest.raises(KeyError, match="no buffer registered"):
                registry.require(descriptor.identity)
        for slot, host in enumerate(_host_domain_buffers(command).values(), start=2):
            assert registry.materialize(host.to_descriptor(), mapping=entry.domain_rank + 1).base == 4096 * (slot + 1)
            with pytest.raises(ValueError):
                registry.materialize(host.to_descriptor())
        local_hbm = replace(
            buffer_reference(_hbm_descriptor()),
            backend_kind=BackendKind.VMM_WINDOW,
            body=(4096).to_bytes(8, "little"),
        ).to_descriptor()
        assert registry.materialize(local_hbm).base == 4096
        assert registry.require(local_hbm.identity).descriptor == local_hbm
    finally:
        registry.close()


@pytest.mark.parametrize("sizes", [(64,), (64, 32)])
@pytest.mark.parametrize("padding", [0, 256])
def test_chip_import_installs_complete_backings_once_and_freezes_replays(monkeypatch, sizes, padding):
    import simpler.worker as module  # noqa: PLC0415

    command = replace(
        _exported_command(),
        buffers=tuple(GlobalDomainBuffer(f"payload{index}", size) for index, size in enumerate(sizes)),
        backings=tuple(
            replace(
                buffer_reference(_hbm_descriptor()),
                identity=CanonicalIdentity(b"hbmnode2", 10 + rank * len(sizes) + index, 1),
                nbytes=size,
            ).to_descriptor()
            for rank in range(2)
            for index, size in enumerate(sizes)
        ),
    )
    entry = module._L2GlobalDomain(7, 3, 0, 2, command.descriptors[0], 4096, 4096, 4096, host_enabled=True)
    store = module._L2GlobalDomainStore({7: entry})
    registry = ImportRegistry(ImportContext(DEVICE_AICPU))
    payload = _chip_import_payload(command)
    monkeypatch.setattr(module, "_open_global_domain_payload", lambda _buf: _control_payload(payload, padding))
    calls = []

    class Chip:
        def comm_global_domain_import(self, domain_id, table):
            calls.append((domain_id, table))
            return 123

        def comm_global_domain_window(self, _domain_id, slot):
            return 4096 * (slot + 1), command.descriptors[slot].mapping_size

    try:
        module._handle_ctrl_global_domain_import(Chip(), memoryview(bytearray()), store, registry)
        expected = module._domain_buffer_descriptors(command)
        assert entry.buffer_descriptors == expected and len(calls) == 1
        for _ in range(2):
            module._handle_ctrl_global_domain_import(Chip(), memoryview(bytearray()), store, registry)
        assert entry.buffer_descriptors == expected and len(calls) == 1
        for index, descriptor in enumerate(command.backings):
            rank, part = divmod(index, len(sizes))
            imported = registry.require(descriptor, BufferCapability.COPY_FROM, mapping=1, offset=4, nbytes=8)
            assert imported.base == 4096 * (rank + 1) + sum(sizes[:part]) and imported.descriptor == descriptor
            assert registry.materialize(descriptor, mapping=1) is imported
            with pytest.raises(ValueError):
                registry.materialize(descriptor, mapping=2)
        for changed in (
            replace(command, backings=()),
            replace(command, backings=command.backings[::-1]),
            replace(command, backings=command.backings[:1]),
            replace(
                command,
                backings=(
                    replace(buffer_reference(command.backings[0]), nbytes=32).to_descriptor(),
                    *command.backings[1:],
                ),
            ),
        ):
            payload = _chip_import_payload(changed)
            with pytest.raises((ValueError, RuntimeError)):
                module._handle_ctrl_global_domain_import(Chip(), memoryview(bytearray()), store, registry)
            assert entry.buffer_descriptors == expected and len(calls) == 1
        payload = module.LOCAL_RELEASE_REQUEST.pack(
            module.LOCAL_DOMAIN_MAGIC, GLOBAL_DOMAIN_VERSION, 7, 3, int(GlobalDomainReleaseStage.VIEWS)
        )

        class Release:
            def comm_global_domain_release(self, *_):
                assert entry.revoked
                for descriptor in expected:
                    with pytest.raises(KeyError):
                        registry.require(descriptor, mapping=1)

        module._handle_ctrl_global_domain_release(Release(), memoryview(bytearray()), store, registry)
        assert not registry._by_identity
    finally:
        registry.close()


def test_concurrent_domain_revocations_both_drain_an_authorized_hbm_copy(monkeypatch):
    from contextlib import contextmanager  # noqa: PLC0415

    from simpler.buffer import BufferCapability, wrap_vmm_window  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    worker = Worker(level=3, num_sub_workers=0)
    hbm = wrap_vmm_window(4096, 64, worker._owner_instance_id, 41, owner_worker_id=0)
    with worker._child_prov_lock:
        worker._record_device_alloc(hbm, domain_allocation_id=7)
    fence = worker._child_prov_worker_lock(0)
    attempts = {name: threading.Event() for name in ("first-drop", "second-drop")}
    second_boundary = threading.Event()
    completed, errors = [], []

    @contextmanager
    def observed_fence(target):
        assert target == 0
        name = threading.current_thread().name
        attempts[name].set()
        if name == "second-drop":
            second_boundary.set()
        with fence:
            yield

    def revoke():
        try:
            worker._drop_domain_allocs(7)
            completed.append(threading.current_thread().name)
        except BaseException as error:  # noqa: BLE001
            errors.append(error)
        finally:
            if threading.current_thread().name == "second-drop":
                second_boundary.set()

    fence.acquire()
    with worker._child_prov_lock:
        worker._require_device_capability_locked(hbm.identity, BufferCapability.COPY_TO, 64, api="copy_to")
    monkeypatch.setattr(worker, "_child_prov_worker_lock", observed_fence)
    threads = [threading.Thread(target=revoke, name=name, daemon=True) for name in attempts]
    try:
        threads[0].start()
        assert attempts["first-drop"].wait(3)
        threads[1].start()
        assert second_boundary.wait(3)
        assert attempts["second-drop"].is_set(), "a second revoke bypassed the authorized copy's fence"
        assert not completed
    finally:
        fence.release()
        for thread in threads:
            if thread.ident is not None:
                thread.join(3)
        worker._clear_child_prov()
        worker.close()
    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert sorted(completed) == ["first-drop", "second-drop"]


@pytest.mark.parametrize("sizes", [(), (32, 64)])
@pytest.mark.parametrize("failure", [None, "prepare", "import"])
def test_prepare_owner_refs_stay_private_until_one_complete_import_and_commit(sizes, failure):
    import simpler.worker as module  # noqa: PLC0415

    prepare = replace(_command(), buffers=tuple(GlobalDomainBuffer(f"payload{i}", n) for i, n in enumerate(sizes)))
    worker = module.Worker(level=3, device_ids=(0,), num_sub_workers=0)
    calls = []

    class Chip:
        def control_payload(self, _kind, chip, control, _payload, _timeout):
            calls.append((chip, control))
            if control == module.CTRL_GLOBAL_DOMAIN_PREPARE:
                if failure == "prepare":
                    raise RuntimeError("prepare completion unknown")
                return (
                    module.LOCAL_PREPARE_REPLY.pack(module.LOCAL_DOMAIN_MAGIC, GLOBAL_DOMAIN_VERSION, 7, 3, 4096, 4096)
                    + _exported_command().descriptors[0].encode()
                )
            if control == module.CTRL_GLOBAL_DOMAIN_IMPORT:
                if failure == "import":
                    raise RuntimeError("import completion unknown")
                return module.LOCAL_IMPORT_REPLY.pack(
                    module.LOCAL_DOMAIN_MAGIC, GLOBAL_DOMAIN_VERSION, 7, 3, 123, 4096, 4096
                )
            assert control == module.CTRL_GLOBAL_DOMAIN_RELEASE
            return b""

    worker._worker = Chip()
    try:
        if failure == "prepare":
            with pytest.raises(RuntimeError, match="prepare completion unknown"):
                worker._prepare_global_domain_reply(prepare, 2)
            state = worker._global_node_domains[7]
            assert state.phase is GlobalDomainPhase.ABORT
            assert state.prepared_domain_ranks == {0} and state.host_owner is not None
            assert not state.contexts and not worker._child_alloc and not state.imports._by_identity
            with pytest.raises(ValueError, match="revoked"):
                worker._prepare_global_domain_reply(prepare, 2)
            return

        reply = worker._prepare_global_domain_reply(prepare, 2)
        prefix_size = 4 + 2 * GLOBAL_DOMAIN_DESCRIPTOR_BYTES
        exports = decode_descriptor_table(reply[:prefix_size])
        records = decode_domain_buffer_refs(reply[prefix_size:])
        state = worker._global_node_domains[7]
        prepared = state.command.members[0].buffers
        assert len(records) == len(sizes) and {d.domain_rank for d in exports} == {0, 2}
        assert all(not member.buffers for member in prepare.members)
        assert tuple(prepared) == tuple(spec.name for spec in prepare.buffers)
        for (descriptor, ref, generation), spec in zip(records, prepare.buffers):
            assert descriptor.identity.owner_instance_id == worker._owner_instance_id
            assert descriptor == prepared[spec.name].to_descriptor()
            assert descriptor.backend_kind == BackendKind.REMOTE_SIDECAR
            assert not descriptor.body and prepared[spec.name].base == 0
            assert descriptor.nbytes == spec.nbytes and not ref and not generation
            with pytest.raises(ValueError, match="not a live device allocation"):
                worker._device_worker_for(descriptor.identity, api="test")
        assert not state.contexts and state.view is None and not worker._child_alloc
        assert not state.imports._by_identity and not worker._live_global_domains
        with pytest.raises(KeyError):
            worker._get_global_domain(7)
        assert worker._prepare_global_domain_reply(prepare, 2) == reply
        assert calls == [(0, module.CTRL_GLOBAL_DOMAIN_PREPARE)]

        descriptors = list(_exported_command().descriptors)
        for descriptor in exports:
            descriptors[descriptor.domain_rank] = descriptor
        peer = tuple(
            BufferDescriptor(
                CanonicalIdentity(b"peernode", i + 1, 1),
                AddressSpace.DEVICE,
                AccessMode.READWRITE,
                BackendKind.REMOTE_SIDECAR,
                nbytes,
            )
            for i, nbytes in enumerate(sizes)
        )
        imported = replace(
            prepare,
            phase=GlobalDomainPhase.IMPORT,
            descriptors=tuple(descriptors),
            backings=tuple(record[0] for record in records) + peer,
        )
        if failure == "import":
            with pytest.raises(RuntimeError, match="import completion unknown"):
                worker._import_global_domain_node(imported, 2)
            assert state.phase is GlobalDomainPhase.ABORT
            assert state.prepared_domain_ranks == {0} and state.host_owner is not None
            assert not state.contexts and not worker._child_alloc and not state.imports._by_identity
            with pytest.raises(ValueError, match="revoked"):
                worker._import_global_domain_node(imported, 2)
            return

        worker._import_global_domain_node(imported, 2)
        worker._import_global_domain_node(imported, 2)
        assert calls.count((0, module.CTRL_GLOBAL_DOMAIN_IMPORT)) == 1
        assert state.contexts[0].buffers is state.command.members[0].buffers is prepared
        for spec, (descriptor, *_locator) in zip(prepare.buffers, records):
            owner = prepared[spec.name]
            assert owner.identity == descriptor.identity and owner.backend_kind == BackendKind.VMM_WINDOW
            assert buffer_reference(owner).to_descriptor() == descriptor
        assert not state.view.committed
        for descriptor in module._domain_buffer_descriptors(imported):
            with pytest.raises(ValueError, match="not live"):
                state.imports.require(descriptor, BufferCapability.COPY_FROM)
        owners = worker._commit_global_domain_node(replace(imported, phase=GlobalDomainPhase.COMMIT))
        assert tuple(buffer_reference(owner).to_descriptor() for owner in owners) == tuple(
            record[0] for record in records
        )
        assert state.view.committed
        assert worker._prepare_global_domain_reply(prepare, 2) == reply
    finally:
        for stage in (GlobalDomainReleaseStage.VIEWS, GlobalDomainReleaseStage.OWNERS):
            worker._release_global_domain_node(GlobalDomainReleaseCommand(7, 3, stage))
        worker._worker = None
        worker._clear_child_prov()
        worker.close()


def test_partial_host_import_revokes_owner_access_and_keeps_staged_cleanup_facts():
    import simpler.worker as module  # noqa: PLC0415

    prepare = replace(
        _command(),
        members=(GlobalDomainMember(2, 0, 0, 0), GlobalDomainMember(2, 1, 1, 1)),
        host_windows=_command().host_windows[:1],
    )
    worker = module.Worker(level=3, device_ids=(0, 1), num_sub_workers=0)
    releases = []

    class Chip:
        def control_payload(self, _kind, chip, control, payload, _timeout):
            if control == module.CTRL_GLOBAL_DOMAIN_PREPARE:
                return (
                    module.LOCAL_PREPARE_REPLY.pack(
                        module.LOCAL_DOMAIN_MAGIC, GLOBAL_DOMAIN_VERSION, 7, 3, 4096 * (chip + 1), 4096
                    )
                    + replace(_exported_command().descriptors[chip], domain_rank=chip).encode()
                )
            if control == module.CTRL_GLOBAL_DOMAIN_IMPORT:
                if chip == 1:
                    raise RuntimeError("second local import completion unknown")
                return module.LOCAL_IMPORT_REPLY.pack(
                    module.LOCAL_DOMAIN_MAGIC, GLOBAL_DOMAIN_VERSION, 7, 3, 123, 4096, 4096
                )
            assert control == module.CTRL_GLOBAL_DOMAIN_RELEASE
            releases.append((chip, module.LOCAL_RELEASE_REQUEST.unpack(payload)[-1]))
            return b""

    worker._worker = Chip()
    try:
        exports = worker._prepare_global_domain_node(prepare, 2)
        state = worker._global_node_domains[7]
        sources = tuple(member.buffers["payload"].to_descriptor() for member in state.command.members)
        imported = replace(prepare, phase=GlobalDomainPhase.IMPORT, descriptors=exports, backings=sources)
        with pytest.raises(RuntimeError, match="second local import completion unknown"):
            worker._import_global_domain_node(imported, 2)
        assert state.phase is GlobalDomainPhase.ABORT and worker._global_node_domains[7] is state
        assert state.prepared_domain_ranks == {0, 1} and state.host_owner is not None
        assert not releases and not worker._child_alloc and state.view is None
        for source in sources:
            with pytest.raises(ValueError, match="not a live device allocation"):
                worker._require_device_capability(source.identity, BufferCapability.COPY_FROM, 8, api="test")
        with pytest.raises(ValueError, match="revoked"):
            worker._import_global_domain_node(imported, 2)
        with pytest.raises(RuntimeError, match="IMPORT completion"):
            worker._commit_global_domain_node(replace(imported, phase=GlobalDomainPhase.COMMIT))
    finally:
        for stage in (GlobalDomainReleaseStage.VIEWS, GlobalDomainReleaseStage.OWNERS):
            worker._release_global_domain_node(GlobalDomainReleaseCommand(7, 3, stage))
        worker._worker = None
        worker._clear_child_prov()
        worker.close()
    assert releases == [(0, 1), (1, 1), (0, 2), (1, 2)]


@pytest.mark.parametrize("changed_base", [False, True])
def test_import_preserves_local_hbm_backing(changed_base):
    import simpler.worker as module  # noqa: PLC0415

    worker = module.Worker(level=3, device_ids=(0,), num_sub_workers=0)
    state, command = _prepared_parent_state(worker, _exported_command())
    prepared = state.command.members[0].buffers
    identity = prepared["payload"].identity
    worker._global_node_domains[7] = state
    calls = []

    class Chip:
        def control_payload(self, _kind, chip, control, _payload, _timeout):
            calls.append((chip, control))
            assert control == module.CTRL_GLOBAL_DOMAIN_IMPORT
            return module.LOCAL_IMPORT_REPLY.pack(
                module.LOCAL_DOMAIN_MAGIC,
                GLOBAL_DOMAIN_VERSION,
                7,
                3,
                123,
                8192 if changed_base else 4096,
                4096,
            )

    worker._worker = Chip()
    try:
        if changed_base:
            with pytest.raises(RuntimeError, match="local backing mismatch"):
                worker._import_global_domain_node(command, 2)
            assert not state.contexts
        else:
            for _ in range(2):
                worker._import_global_domain_node(command, 2)
            buffer = state.contexts[0].buffers["payload"]
            assert bytes(buffer.identity.owner_instance_id) == worker._owner_instance_id
            assert buffer.identity == identity
            assert state.contexts[0].buffers is state.command.members[0].buffers is prepared
            assert buffer.base == 4096 and buffer.backend_kind == BackendKind.VMM_WINDOW
            assert len(calls) == 1
    finally:
        worker._worker = None
        worker._global_node_domains.clear()
        worker._clear_child_prov()
        worker.close()


@pytest.mark.parametrize("invalid_peer", ["size", "access", "duplicate", "missing", "local-owner"])
def test_parent_import_installs_copy_sources_and_publishes_exact_peer_backings_after_commit(invalid_peer):
    import simpler.worker as module  # noqa: PLC0415

    worker = module.Worker(level=3, device_ids=(0,), num_sub_workers=0)
    state, command = _prepared_parent_state(worker, _exported_command())
    prepared = state.command.members[0].buffers
    local = prepared["payload"].to_descriptor()
    worker._global_node_domains[7] = state
    calls = []

    class Chip:
        def control_payload(self, _kind, chip, control, payload, _timeout):
            assert chip == 0 and control == module.CTRL_GLOBAL_DOMAIN_IMPORT
            calls.append(bytes(payload))
            return module.LOCAL_IMPORT_REPLY.pack(
                module.LOCAL_DOMAIN_MAGIC, GLOBAL_DOMAIN_VERSION, 7, 3, 123, 4096, 4096
            )

    worker._worker = Chip()
    try:
        invalid = (
            local
            if invalid_peer == "duplicate"
            else replace(
                buffer_reference(command.backings[1]),
                **({"nbytes": 32} if invalid_peer == "size" else {"access": AccessMode.READ}),
            ).to_descriptor()
        )
        invalid_backings = (local, invalid)
        if invalid_peer == "missing":
            invalid_backings = ()
        elif invalid_peer == "local-owner":
            invalid_backings = (command.backings[1], local)
        with pytest.raises(ValueError):
            worker._import_global_domain_node(replace(command, backings=invalid_backings), 2)
        assert not calls and not state.contexts and not worker._child_alloc
        assert state.phase is GlobalDomainPhase.PREPARE_EXPORT
        for _ in range(2):
            worker._import_global_domain_node(command, 2)
        host = _host_domain_buffers(command)[2].to_descriptor()
        assert len(calls) == 1 and state.command.backings == command.backings
        assert state.contexts[0].buffers is state.command.members[0].buffers is prepared
        for source in (host, *command.backings):
            with pytest.raises(ValueError, match="not live"):
                state.imports.require(source, BufferCapability.COPY_FROM)
        with pytest.raises(KeyError):
            worker._get_global_domain(7)
        commit = replace(command, phase=GlobalDomainPhase.COMMIT)
        owners = worker._commit_global_domain_node(commit)
        assert len(owners) == 1 and owners[0] is worker._child_alloc.get(local.identity)
        assert buffer_reference(owners[0]).to_descriptor() == local
        assert worker._commit_global_domain_node(commit) == owners
        for source in (host, *command.backings):
            for need in (BufferCapability.COPY_FROM, BufferCapability.COPY_TO):
                assert state.imports.require(source, need, offset=4, nbytes=8).base == 0
            with pytest.raises(ValueError):
                state.imports.materialize(source)
        ref = CommDomainRef(command.owner_instance_id, 7, 3)
        worker._lifecycle = module._Lifecycle.READY
        handle = worker._get_domain(ref)
        peer = handle.member(1).buffers["payload"]
        assert peer.to_descriptor() == command.backings[1] and peer.base == 0
        mapped = handle.map_view(peer.tensor([8], DataType.UINT8), domain_rank=0)
        args = TaskArgs()
        args.add_tensor(mapped, transfer=TensorTransfer.NONE)
        with worker._child_prov_lock:
            worker._prepare_buffer_references(args, None, 0)
        with pytest.raises(RuntimeError, match="descriptor table"):
            worker._import_global_domain_node(replace(command, backings=command.backings[::-1]), 2)
        assert len(calls) == 1
        worker._drop_domain_allocs(worker._global_domain_provenance_id(7))
        with pytest.raises(ValueError, match="not live"):
            state.imports.require(command.backings[1], BufferCapability.COPY_FROM)
    finally:
        worker._worker = None
        worker._global_node_domains.clear()
        worker._clear_child_prov()
        worker.close()


@pytest.mark.parametrize("writing", [False, True])
def test_host_byte_access_holds_release_fence_through_cpu_copy(monkeypatch, writing):
    import simpler.worker as module  # noqa: PLC0415
    from simpler.task_interface import GlobalCommDomainView  # noqa: PLC0415

    worker = module.Worker(level=3, num_sub_workers=0)
    entered, proceed, releasing, freed = (threading.Event() for _ in range(4))
    errors = []
    command = replace(_exported_command(), profile="a3-fabric-v1")

    storage, operand = ctypes.create_string_buffer(96), ctypes.create_string_buffer(8)
    memmove = ctypes.memmove

    def copy(dst, src, nbytes):
        entered.set()
        assert worker._child_prov_lock.locked()
        assert proceed.wait(3)
        assert not freed.is_set()
        return memmove(dst, src, nbytes)

    monkeypatch.setattr(module.ctypes, "memmove", copy)

    class Owner:
        base = ctypes.addressof(storage)

        def release(self):
            assert proceed.is_set()
            freed.set()

    view = GlobalCommDomainView(
        name="mixed", members=command.members, contexts={}, domain_id=7, generation=3, mapping_size=4096
    )
    state = _committed_parent_state(
        worker,
        command,
        view=view,
        host_owner=Owner(),
        node_worker_id=2,
    )
    worker._global_node_domains[7] = state
    worker._lifecycle = module._Lifecycle.READY
    worker._worker = object()

    def access():
        try:
            worker._host_domain_bytes(
                CommDomainRef(command.owner_instance_id, 7, 3),
                _host_domain_buffers(command)[2].to_descriptor(),
                0,
                ctypes.addressof(operand),
                8,
                writing=writing,
            )
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    def release():
        try:
            releasing.set()
            for stage in (GlobalDomainReleaseStage.VIEWS, GlobalDomainReleaseStage.OWNERS):
                worker._release_global_domain_node(GlobalDomainReleaseCommand(7, 3, stage))
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=target, daemon=True) for target in (access, release)]
    try:
        threads[0].start()
        assert entered.wait(3)
        threads[1].start()
        assert releasing.wait(3)
        assert worker._child_prov_lock.locked() and not freed.is_set()
    finally:
        proceed.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(3)
        worker._worker = None
        worker._global_node_domains.clear()
        worker.close()
    assert not errors and freed.is_set()
    assert all(not thread.is_alive() for thread in threads)


def test_buffer_copy_rechecks_state_after_waiting_for_actor_fence(monkeypatch):
    from contextlib import contextmanager  # noqa: PLC0415

    import simpler.worker as module  # noqa: PLC0415
    from simpler.task_interface import GlobalCommDomainView  # noqa: PLC0415

    worker = module.Worker(level=3, num_sub_workers=0)
    command = _exported_command()
    state = _committed_parent_state(
        worker,
        command,
        node_worker_id=2,
        view=GlobalCommDomainView(
            name="mixed", members=command.members, contexts={}, domain_id=7, generation=3, mapping_size=4096
        ),
        contexts={0: _host_handle(command).local_context(0)},
    )
    worker._global_node_domains[7] = state
    worker._lifecycle = module._Lifecycle.READY
    attempted, fence = threading.Event(), threading.Lock()
    calls, errors = [], []

    class Chip:
        def control_payload(self, *_):
            calls.append("copy")

    worker._worker = Chip()

    @contextmanager
    def acquire_actor(_worker):
        attempted.set()
        with fence:
            yield

    monkeypatch.setattr(worker, "_child_prov_worker_lock", acquire_actor)

    def copy():
        try:
            worker._copy_domain_buffers(
                CommDomainRef(command.owner_instance_id, 7, 3),
                worker=0,
                dst=_host_domain_buffers(command)[2],
                src=_host_domain_buffers(command)[5],
                nbytes=8,
            )
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    thread = threading.Thread(target=copy, daemon=True)
    fence.acquire()
    try:
        thread.start()
        assert attempted.wait(3)
        with worker._child_prov_lock:
            state.phase = GlobalDomainPhase.ABORT
    finally:
        fence.release()
        thread.join(3)
        worker._worker = None
        worker._global_node_domains.clear()
        worker.close()
    assert not thread.is_alive() and len(errors) == 1 and isinstance(errors[0], KeyError)
    assert not calls and worker._ordered_cleanup_error is None and worker._cleanup_journal.empty


@pytest.mark.parametrize(
    "mutation",
    [
        "truncated",
        "trailing",
        "count",
        "disabled",
        "slot",
        "profile",
        "rank_count",
        "capacity",
        "space",
        "backend",
        "zero-owner",
        "zero-id",
        "duplicate",
        "empty",
        "local-export",
    ],
)
def test_chip_import_rejects_bad_host_metadata_before_native_import(monkeypatch, mutation):
    import simpler.worker as module  # noqa: PLC0415

    command = _exported_command()
    hosts = list(buffer.to_descriptor() for buffer in _host_domain_buffers(command).values())
    descriptors = list(command.descriptors)
    entry = module._L2GlobalDomain(7, 3, 0, 2, descriptors[0], 4096, 4096, 4096, host_enabled=True)
    if mutation in ("slot", "profile", "rank_count", "local-export"):
        index, field, value = {
            "slot": (2, "domain_rank", 1),
            "profile": (2, "profile_id", 2),
            "rank_count": (2, "rank_count", 3),
            "local-export": (0, "handle", b"changed"),
        }[mutation]
        descriptors[index] = replace(descriptors[index], **{field: value})
    elif mutation in ("capacity", "space", "backend", "zero-owner", "zero-id", "empty"):
        original = hosts[0]
        hosts[0] = BufferDescriptor(
            CanonicalIdentity(
                bytes(8) if mutation == "zero-owner" else original.identity.owner_instance_id,
                0 if mutation == "zero-id" else original.identity.buffer_id,
                1,
            ),
            AddressSpace.DEVICE if mutation == "space" else AddressSpace.HOST,
            original.access,
            BackendKind.POSIX_SHM if mutation == "backend" else original.backend_kind,
            97 if mutation == "capacity" else 0 if mutation == "empty" else original.nbytes,
            b"name" if mutation == "backend" else b"",
        )
    elif mutation == "duplicate":
        hosts[1] = hosts[0]
    elif mutation == "disabled":
        entry.host_enabled = False
    legacy = mutation in ("capacity", "space", "backend", "zero-owner", "zero-id", "empty", "duplicate")
    payload = _chip_import_payload(replace(command, descriptors=tuple(descriptors)), hosts if legacy else None)
    if mutation == "truncated":
        payload = payload[:-1]
    elif mutation == "trailing":
        payload += b"x"
    elif mutation == "count":
        payload = bytearray(payload)
        struct.pack_into("<I", payload, module.LOCAL_IMPORT_REQUEST.size - 4, 1)
    monkeypatch.setattr(module, "_open_global_domain_payload", lambda _buf: _control_payload(payload))
    registry = ImportRegistry(ImportContext(DEVICE_AICPU))
    try:
        with pytest.raises((ValueError, RuntimeError)):
            module._handle_ctrl_global_domain_import(
                object(), memoryview(bytearray()), module._L2GlobalDomainStore({7: entry}), registry
            )
        assert entry.device_ctx == 0 and not entry.descriptor_table and not entry.buffer_descriptors
        assert not registry._by_identity
    finally:
        registry.close()


def test_host_mapping_uses_immutable_domain_configuration_and_revokes_without_device_registration():
    import simpler.worker as module  # noqa: PLC0415

    worker = module.Worker(level=3, num_sub_workers=0)
    command = _exported_command()
    state = _committed_parent_state(worker, command, node_worker_id=2)
    worker._global_node_domains[7] = state
    handle = _host_handle(command)
    original = handle.host(2).tensor([8], DataType.UINT8)._with_mapping_ref(1)
    args = TaskArgs()
    args.add_tensor(original, transfer=TensorTransfer.NONE)
    worker._domain_members[-7] = {original.buffer.identity}
    try:
        handle.host(2).nbytes = 8192
        forged = TaskArgs()
        forged.add_tensor(handle.host(2).tensor([8], DataType.UINT8)._with_mapping_ref(1), transfer=TensorTransfer.NONE)
        with worker._child_prov_lock:
            assert original.buffer.identity not in worker._child_alloc
            worker._prepare_buffer_references(args, None, 0)
            unknown = TaskArgs()
            unknown.add_tensor(original._with_mapping_ref(255), transfer=TensorTransfer.NONE)
            with pytest.raises(ValueError, match="outside its domain"):
                worker._prepare_buffer_references(unknown, None, 0)
            with pytest.raises(ValueError, match="different descriptor"):
                worker._prepare_buffer_references(forged, None, 0)
        worker._drop_domain_allocs(-7)
        assert state.phase is GlobalDomainPhase.ABORT and -7 not in worker._domain_members
        with worker._child_prov_lock, pytest.raises(ValueError, match="not live"):
            worker._prepare_buffer_references(args, None, 0)
    finally:
        worker._global_node_domains.clear()
        worker._clear_child_prov()
        worker.close()


@pytest.mark.parametrize("arch", ["a2a3", "a5"])
@pytest.mark.parametrize("runtime", ["host_build_graph", "tensormap_and_ringbuffer"])
@pytest.mark.parametrize("host_enabled", [False, True])
def test_import_rejects_another_valid_local_export_before_native_import(native_domain_lib, monkeypatch, host_enabled):
    import simpler.worker as module  # noqa: PLC0415
    from simpler.buffer import wrap_vmm_window  # noqa: PLC0415

    ids = [int.from_bytes(os.urandom(8), "little") or 1 for _ in range(3)]
    exports, bases, owners, calls = [], [], [], []
    allocators = [module.Worker(level=3, num_sub_workers=0) for _ in range(2)]
    try:
        for domain_id, rank in zip(ids, (0, 1, 0)):
            token, base = ctypes.create_string_buffer(288), ctypes.c_uint64()
            assert (
                native_domain_lib.comm_global_domain_prepare(
                    domain_id, rank, 2, 4096, 1, host_enabled, token, ctypes.byref(base)
                )
                == 0
            )
            exports.append(GlobalDomainDescriptor.decode(token.raw))
            bases.append(base.value)
        ctypes.memset(bases[0], 0x11, 16)
        ctypes.memset(bases[2], 0x33, 16)
        command = _exported_command()
        host_exports = []
        if host_enabled:
            for slot, window in enumerate(command.host_windows, 2):
                owner = SharedMemory(create=True, size=window.spec.nbytes)
                owners.append(owner)
                host_exports.append(
                    GlobalDomainDescriptor(GLOBAL_DOMAIN_VERSION, 1, slot, 2, owner.size, ("/" + owner.name).encode())
                )
        command = replace(
            command,
            host_windows=command.host_windows if host_enabled else (),
            backings=tuple(
                buffer_reference(
                    wrap_vmm_window(
                        bases[rank],
                        command.buffers[0].nbytes,
                        allocator._owner_instance_id,
                        allocator._burn_buffer_identity().buffer_id,
                    )
                ).to_descriptor()
                for rank, allocator in enumerate(allocators)
            )
            if host_enabled
            else (),
            descriptors=(exports[2], exports[1], *host_exports),
        )
        payload = _chip_import_payload(command)
        monkeypatch.setattr(module, "_open_global_domain_payload", lambda _buf: _control_payload(payload))
        entry = module._L2GlobalDomain(7, 3, 0, 2, exports[0], bases[0], 4096, 4096, host_enabled=host_enabled)

        class Chip:
            def comm_global_domain_import(self, _, table):
                calls.append(table)
                context = ctypes.c_uint64()
                assert (
                    native_domain_lib.comm_global_domain_import(
                        ids[0], ctypes.create_string_buffer(table), len(command.descriptors), ctypes.byref(context)
                    )
                    == 0
                )
                return context.value

        with pytest.raises(RuntimeError, match="local exported descriptor"):
            module._handle_ctrl_global_domain_import(
                Chip(), memoryview(bytearray()), module._L2GlobalDomainStore({7: entry})
            )
        assert not calls and entry.device_ctx == 0
        assert ctypes.string_at(bases[0], 16) == bytes([0x11]) * 16
        assert ctypes.string_at(bases[2], 16) == bytes([0x33]) * 16
        command = replace(command, descriptors=(exports[0], exports[1], *host_exports))
        payload = _chip_import_payload(command)
        module._handle_ctrl_global_domain_import(
            Chip(), memoryview(bytearray()), module._L2GlobalDomainStore({7: entry})
        )
        assert len(calls) == 1 and entry.device_ctx != 0
    finally:
        for stage in (1, 2):
            for domain_id in ids:
                assert native_domain_lib.comm_global_domain_release(domain_id, stage) == 0
        for owner in owners:
            owner.close()
            owner.unlink()
        for allocator in allocators:
            allocator.close()
