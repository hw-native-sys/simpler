# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
from dataclasses import replace
from types import SimpleNamespace

import pytest
from simpler.buffer import (
    AccessMode,
    AddressSpace,
    BackendKind,
    Buffer,
    CanonicalIdentity,
    ImportContext,
    ImportRegistry,
    mint_owner_instance_id,
    select_adapter,
)
from simpler.comm_endpoints import (
    DEVICE_AICORE,
    DEVICE_AICPU,
    HOST_CPU,
    AdapterKind,
    AdapterProfile,
    BufferAccessQuery,
    DefaultRegionAccessService,
    _AdapterCandidate,
)
from simpler.task_interface import (
    ArgDirection,
    DataType,
    TaskArgs,
    TensorArgType,
    TensorTransfer,
    _materialize_chip_task_args,
)

_REFS = {AccessMode.READWRITE: 1, AccessMode.READ: 2, AccessMode.WRITE: 3}


@pytest.fixture
def mapping(request):
    identity = CanonicalIdentity(mint_owner_instance_id(), 1, 1)
    buffer = Buffer(
        identity=identity,
        address_space=AddressSpace.HOST,
        access=AccessMode.READWRITE,
        backend_kind=BackendKind.REMOTE_SIDECAR,
        nbytes=64,
        base=4096,
    )
    environment = (os.getpid(), 1, 1, 0, False)
    registry = ImportRegistry(
        ImportContext(
            DEVICE_AICPU,
            platform="a2a3",
            runtime="tensormap_and_ringbuffer",
            execution_environment=environment,
            current_environment=lambda: environment,
        )
    )
    for access in getattr(request, "param", (AccessMode.READWRITE, AccessMode.READ, AccessMode.WRITE)):
        registry.register_attachment(
            buffer.to_descriptor(),
            8192,
            mapping_ref=_REFS[access],
            access=access,
            profile=AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT,
            live=lambda: True,
        )
    return buffer, registry


@pytest.mark.parametrize(
    "access,tag,direction,allowed",
    [
        (AccessMode.READ, TensorArgType.INPUT, ArgDirection.IN, True),
        (AccessMode.READ, TensorArgType.OUTPUT, ArgDirection.OUT, False),
        (AccessMode.READ, TensorArgType.INOUT, ArgDirection.INOUT, False),
        (AccessMode.WRITE, TensorArgType.OUTPUT, ArgDirection.OUT, True),
        (AccessMode.WRITE, TensorArgType.INPUT, ArgDirection.OUT, True),
        (AccessMode.WRITE, TensorArgType.INPUT, ArgDirection.IN, False),
        (AccessMode.READWRITE, TensorArgType.INOUT, ArgDirection.INOUT, True),
    ],
)
def test_selected_mapping_grant(mapping, access, tag, direction, allowed):
    buffer, registry = mapping
    native = buffer.tensor((16,), DataType.FLOAT32)
    view = native._with_mapping_ref(_REFS[access])
    args = TaskArgs()
    args.add_tensor(view, tag, transfer=TensorTransfer.NONE)
    target = SimpleNamespace(sig_count=1, sig=lambda index: direction)
    if allowed:
        assert _materialize_chip_task_args(args, registry, target).tensor(0).data == 8192
    else:
        with pytest.raises(ValueError, match="mapping grant"):
            _materialize_chip_task_args(args, registry, target)
    assert native.mapping_ref == 0 and view.buffer == native.buffer


def test_callable_direction_cannot_be_weakened_by_input_tag(mapping):
    buffer, registry = mapping
    view = buffer.tensor((16,), DataType.FLOAT32)._with_mapping_ref(_REFS[AccessMode.READ])
    args = TaskArgs()
    args.add_tensor(view, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
    target = SimpleNamespace(sig_count=1, sig=lambda index: ArgDirection.OUT)
    with pytest.raises(ValueError, match="mapping grant"):
        _materialize_chip_task_args(args, registry, target)


@pytest.mark.parametrize(
    "mapping,write_granted",
    [((AccessMode.READ,), False), ((AccessMode.READ, AccessMode.WRITE), True)],
    indirect=["mapping"],
    ids=["read-only", "read-and-write"],
)
def test_native_selector_requires_an_existing_target_grant(mapping, write_granted):
    buffer, registry = mapping
    native = buffer.tensor((16,), DataType.FLOAT32)
    read = native._with_mapping_ref(_REFS[AccessMode.READ])
    write = native._with_mapping_ref(_REFS[AccessMode.WRITE])
    args = TaskArgs()
    args.add_tensor(write, TensorArgType.OUTPUT_EXISTING, transfer=TensorTransfer.NONE)
    target = SimpleNamespace(sig_count=1, sig=lambda _index: ArgDirection.OUT)
    if write_granted:
        assert _materialize_chip_task_args(args, registry, target).tensor(0).data == 8192
    else:
        with pytest.raises(ValueError, match="no granted attachment"):
            _materialize_chip_task_args(args, registry, target)
    assert read.mapping_ref == _REFS[AccessMode.READ] and read.buffer == write.buffer


def test_mapped_tensor_cannot_replace_its_mapping_reference(mapping):
    buffer, _ = mapping
    read = buffer.tensor((16,), DataType.FLOAT32)._with_mapping_ref(_REFS[AccessMode.READ])
    assert read._with_mapping_ref(read.mapping_ref) == read
    with pytest.raises(ValueError):
        read._with_mapping_ref(_REFS[AccessMode.WRITE])


def test_mapping_rejects_grant_change_and_unregistered_reference(mapping):
    buffer, registry = mapping
    with pytest.raises(ValueError, match="conflicts"):
        registry.register_attachment(
            buffer.to_descriptor(),
            8192,
            mapping_ref=_REFS[AccessMode.READ],
            access=AccessMode.READWRITE,
            profile=AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT,
            live=lambda: True,
        )
    view = buffer.tensor((16,), DataType.FLOAT32)._with_mapping_ref(4)
    with pytest.raises(ValueError, match="no granted attachment"):
        registry.resolve_tensor(view)
    registry.unregister(buffer.identity)
    with pytest.raises(ValueError, match="no granted attachment"):
        registry.resolve_tensor(buffer.tensor((16,), DataType.FLOAT32)._with_mapping_ref(_REFS[AccessMode.READ]))


def test_read_mapping_preflight_precedes_other_argument_materialization(mapping, monkeypatch):
    buffer, registry = mapping
    args = TaskArgs()
    args.add_tensor(buffer.tensor((16,), DataType.FLOAT32), transfer=TensorTransfer.H2D)
    view = buffer.tensor((16,), DataType.FLOAT32)._with_mapping_ref(_REFS[AccessMode.READ])
    args.add_tensor(view, TensorArgType.OUTPUT, transfer=TensorTransfer.NONE)
    monkeypatch.setattr(registry, "materialize", lambda _: pytest.fail("must reject before ordinary import"))
    target = SimpleNamespace(sig_count=2, sig=lambda _index: ArgDirection.OUT)
    with pytest.raises(ValueError, match="mapping grant"):
        _materialize_chip_task_args(args, registry, target)


@pytest.mark.parametrize(
    "target", [None, SimpleNamespace(sig_count=0), SimpleNamespace(sig_count=2, sig=lambda _: None)]
)
def test_mapping_requires_callable_direction_before_materialization(mapping, monkeypatch, target):
    buffer, registry = mapping
    args = TaskArgs()
    args.add_tensor(buffer.tensor((16,), DataType.FLOAT32), transfer=TensorTransfer.H2D)
    view = buffer.tensor((16,), DataType.FLOAT32)._with_mapping_ref(_REFS[AccessMode.READ])
    args.add_tensor(view, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
    monkeypatch.setattr(registry, "materialize", lambda _: pytest.fail("must reject before ordinary import"))
    with pytest.raises(ValueError, match="declared callable direction"):
        _materialize_chip_task_args(args, registry, target)


def test_attachment_cannot_widen_source_grant(mapping):
    buffer, registry = mapping
    buffer.access = AccessMode.READ
    registry = ImportRegistry(registry._context)
    with pytest.raises(ValueError, match="exceeds source grant"):
        registry.register_attachment(
            buffer.to_descriptor(),
            8192,
            mapping_ref=5,
            access=AccessMode.READWRITE,
            profile=AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT,
            live=lambda: True,
        )


def test_host_profile_does_not_implicitly_materialize_a_sidecar(mapping):
    buffer, _ = mapping
    fresh = ImportRegistry(mapping[1]._context)
    with pytest.raises(ValueError, match="UNSUPPORTED_BACKEND_KIND"):
        fresh.materialize(buffer.to_descriptor())
    view = buffer.tensor((16,), DataType.FLOAT32)._with_mapping_ref(_REFS[AccessMode.READ])
    with pytest.raises(ValueError, match="no granted attachment"):
        fresh.resolve_tensor(view)
    fresh.register_attachment(
        buffer.to_descriptor(),
        8192,
        mapping_ref=view.mapping_ref,
        access=AccessMode.READ,
        profile=AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT,
        live=lambda: True,
    )
    assert fresh.resolve_tensor(view).base == 8192


def test_close_attachment_revokes_only_its_record_and_cannot_be_registered_again(mapping):
    buffer, registry = mapping
    native = buffer.tensor((16,), DataType.FLOAT32)
    registry.register_attachment(
        buffer.to_descriptor(),
        4096,
        mapping_ref=0,
        profile=AdapterProfile.HOST_VMM_MAP,
        live=lambda: True,
    )
    read = native._with_mapping_ref(_REFS[AccessMode.READ])
    write = native._with_mapping_ref(_REFS[AccessMode.WRITE])
    registry.close_attachment(buffer.identity, read.mapping_ref)
    registry.close_attachment(buffer.identity, read.mapping_ref)
    with pytest.raises(ValueError, match="not live"):
        registry.resolve_tensor(read)
    assert registry.resolve_tensor(write).base == 8192
    assert registry.has_active_attachments(buffer.identity)
    with pytest.raises(ValueError, match="conflicts"):
        registry.register_attachment(
            buffer.to_descriptor(),
            8192,
            mapping_ref=read.mapping_ref,
            access=AccessMode.READ,
            profile=AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT,
            live=lambda: True,
        )
    registry.close_attachment(buffer.identity, write.mapping_ref)
    registry.close_attachment(buffer.identity, _REFS[AccessMode.READWRITE])
    assert not registry.has_active_attachments(buffer.identity)
    assert registry.resolve_tensor(native).base == 4096
    registry.unregister(buffer.identity)
    assert not registry.has_active_attachments(buffer.identity)
    with pytest.raises(ValueError, match="no granted attachment"):
        registry.resolve_tensor(read)


@pytest.mark.parametrize("index", range(5))
def test_mapping_rejects_changed_target_environment(mapping, index):
    buffer, original = mapping
    environment = original._context.execution_environment
    current = [environment]
    registry = ImportRegistry(replace(original._context, current_environment=lambda: current[0]))
    registry.register_attachment(
        buffer.to_descriptor(),
        8192,
        mapping_ref=1,
        profile=AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT,
        live=lambda: True,
    )
    view = buffer.tensor((16,), DataType.FLOAT32)._with_mapping_ref(1)
    assert registry.resolve_tensor(view).execution_environment == environment
    changed = list(environment)
    changed[index] = not changed[index] if index == 4 else changed[index] + 1
    current[0] = tuple(changed)
    with pytest.raises(ValueError, match="not live"):
        registry.resolve_tensor(view)
    current[0] = environment
    with pytest.raises(ValueError, match="not live"):
        registry.resolve_tensor(view)


@pytest.mark.parametrize(
    "context_changes",
    [
        {"execution_environment": None},
        {"execution_environment": (1, 2)},
        {"current_environment": None},
        {"current_environment": lambda: None},
    ],
)
def test_mapping_registration_requires_current_target_environment(mapping, context_changes):
    buffer, original = mapping
    registry = ImportRegistry(replace(original._context, **context_changes))
    with pytest.raises(ValueError, match="execution environment"):
        registry.register_attachment(
            buffer.to_descriptor(),
            8192,
            mapping_ref=1,
            profile=AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT,
            live=lambda: True,
        )


@pytest.mark.parametrize(
    "profile,deployment,platform,supported",
    [
        (AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT, DEVICE_AICPU, "a2a3", True),
        (AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT, DEVICE_AICORE, "a2a3", True),
        (AdapterProfile.HOST_SHM_MAP, DEVICE_AICPU, "a2a3sim", True),
        (AdapterProfile.HOST_SHM_MAP, HOST_CPU, "a2a3sim", True),
        (AdapterProfile.HOST_VMM_MAP, HOST_CPU, "a2a3", True),
        (AdapterProfile.HOST_VMM_MAP, DEVICE_AICPU, "a2a3", False),
        (AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT, HOST_CPU, "a2a3", False),
        (AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT, DEVICE_AICPU, "a5", False),
        (AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT, DEVICE_AICPU, "a2a3sim", False),
        (AdapterProfile.HOST_SHM_MAP, DEVICE_AICPU, "a2a3", False),
        (AdapterProfile.REMOTE_COPY, DEVICE_AICPU, "a2a3", False),
        (AdapterProfile.DEVICE_LOCAL, DEVICE_AICPU, "a2a3", False),
    ],
)
def test_confirmed_host_adapter_uses_platform_and_consumer_policy(mapping, profile, deployment, platform, supported):
    buffer, original = mapping
    context = replace(original._context, deployment=deployment, platform=platform)
    kwargs = {"confirmed_profile": profile, "mapped": deployment is not HOST_CPU}
    if supported:
        assert select_adapter(buffer.to_descriptor(), context, **kwargs) == (AdapterKind.DIRECT_MAP, profile)
    else:
        with pytest.raises(ValueError, match="UNSUPPORTED_ENDPOINT_RELATION"):
            select_adapter(buffer.to_descriptor(), context, **kwargs)


@pytest.mark.parametrize(
    "query_changes",
    [
        {"source_address_space": AddressSpace.DEVICE},
        {"same_node": False},
        {"runtime": "unknown"},
        {"platform": "a5"},
    ],
)
def test_confirmed_profile_cannot_bypass_default_service_by_stripping_diagnostic(query_changes):
    query = BufferAccessQuery(
        BackendKind.REMOTE_SIDECAR,
        DEVICE_AICPU,
        same_node=True,
        platform="a2a3",
        runtime="tensormap_and_ringbuffer",
        confirmed_profile=AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT,
        source_address_space=AddressSpace.HOST,
    )
    candidate = _AdapterCandidate(AdapterKind.DIRECT_MAP, AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT)
    decision = DefaultRegionAccessService().evaluate_buffer_access(replace(query, **query_changes), candidate)
    assert not decision.supported
