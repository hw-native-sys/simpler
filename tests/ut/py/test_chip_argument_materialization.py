# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Accepted local and mailbox arguments share one chip-POD conversion."""

import ctypes

import pytest
from simpler.buffer import (
    AddressSpace,
    ImportContext,
    ImportRegistry,
    TensorTransfer,
    create_host_shared_buffer,
    mint_owner_instance_id,
    wrap_device_malloc,
)
from simpler.comm_endpoints import DEVICE_AICPU
from simpler.task_interface import (
    ChipStorageTaskArgs,
    DataType,
    TaskArgs,
    Tensor,
    _materialize_chip_task_args,
    read_args_from_blob,
)

from tests.ut.py.test_worker._wire_blob import encode_blob


@pytest.mark.parametrize("mailbox", [False, True], ids=["local", "mailbox"])
def test_materialization_preserves_views_transfers_and_scalar_bits(mailbox):
    owner = mint_owner_instance_id()
    host = create_host_shared_buffer(128, owner, 1)
    device = wrap_device_malloc(0x4000, 64, owner, 2)
    registry = ImportRegistry(ImportContext(deployment=DEVICE_AICPU, device_owner_instance_id=owner))
    try:
        views = [
            Tensor(host, shapes=(2, 3), strides=(4, 1), byte_offset=4, dtype=DataType.FLOAT32),
            Tensor(host, shapes=(2,), byte_offset=32, dtype=DataType.FLOAT32),
            Tensor(device, shapes=(2,), byte_offset=16, dtype=DataType.INT32),
        ]
        transfers = [TensorTransfer.NONE, TensorTransfer.H2D, TensorTransfer.NONE]
        bits = (0xFFFFFFFFFFFFFFFF, 0x8000000000000001, 0x7FC00001)
        args = TaskArgs()
        for view, transfer in zip(views, transfers, strict=True):
            args.add_tensor(view, transfer=transfer)
        for word in bits:
            args.add_scalar(word)
        if mailbox:
            blob = ctypes.create_string_buffer(encode_blob(views, bits, transfers))
            args = read_args_from_blob(ctypes.addressof(blob), len(blob))
        image = _materialize_chip_task_args(args, registry)
        imported = registry.require(host.identity)
        assert imported.shm is not None and imported.shm.buf is not None
        assert image.tensor_count() == 3 and image.scalar_count() == len(bits)
        for i, view in enumerate(views):
            tensor = image.tensor(i)
            assert tensor.data == (imported.base if i < 2 else device.base) + view.byte_offset
            assert tensor.shapes == view.shapes and tensor.strides == view.strides
            assert tensor.dtype.value == view.dtype
            assert tensor.address_space == (AddressSpace.HOST if i < 2 else AddressSpace.DEVICE)
            assert image.transfer(i) == transfers[i]
        assert tuple(image.scalar(i) for i in range(len(bits))) == bits

        args.clear()
        args.add_scalar(7)
        following = _materialize_chip_task_args(args, registry)
        assert following.tensor_count() == 0 and following.scalar(0) == 7
        assert image.tensor_count() == 3 and image.scalar(0) == bits[0]
        assert registry.require(host.identity) is imported
    finally:
        registry.close()
        host.close()


@pytest.mark.parametrize("args", [None, object(), ChipStorageTaskArgs()])
def test_address_free_type_gate_precedes_import(args, monkeypatch):
    registry = ImportRegistry(ImportContext(deployment=DEVICE_AICPU))

    def forbidden(*_):
        pytest.fail("unsupported arguments reached buffer import")

    monkeypatch.setattr(registry, "materialize_args", forbidden)
    with pytest.raises(TypeError, match="address-free TaskArgs"):
        _materialize_chip_task_args(args, registry)


def test_resolution_failure_leaves_existing_imports_owned_by_the_caller():
    owner = mint_owner_instance_id()
    host = create_host_shared_buffer(32, owner, 1)
    foreign = wrap_device_malloc(0x4000, 32, mint_owner_instance_id(), 2)
    registry = ImportRegistry(ImportContext(deployment=DEVICE_AICPU, device_owner_instance_id=owner))
    try:
        args = TaskArgs()
        args.add_tensor(Tensor(host, shapes=(2,), dtype=DataType.FLOAT32))
        _materialize_chip_task_args(args, registry)
        imported = registry.require(host.identity)
        args.add_tensor(Tensor(foreign, shapes=(2,), dtype=DataType.FLOAT32))
        with pytest.raises(ValueError):
            _materialize_chip_task_args(args, registry)
        assert registry.require(host.identity) is imported
        assert imported.shm is not None
        view = imported.shm.buf
        assert view is not None
        view[:4] = b"KEEP"
        assert host.shm is not None and host.shm.buf is not None
        assert bytes(host.shm.buf[:4]) == b"KEEP"
    finally:
        registry.close()
        host.close()
    assert imported.shm.buf is None
