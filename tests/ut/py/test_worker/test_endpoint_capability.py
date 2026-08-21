# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""endpoint x address_space capability: a DEVICE tensor may not reach a host Python sub-worker.

A sub-worker maps its args into its own (host) process and hands them to torch, so a device address
there would be dereferenced as a host pointer. The submit-time check turns that into an error naming
the argument; the materialization check behind it covers any future submit path that skips the first.
"""

import ctypes
import re

import pytest
from simpler.buffer import (
    AccessMode,
    AddressSpace,
    ImportContext,
    ImportRegistry,
    mint_owner_instance_id,
    wrap_device_malloc,
)
from simpler.comm_endpoints import RegionAccessReasonCode
from simpler.task_interface import DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from ._wire_blob import encode_blob


def _device_handle():
    return wrap_device_malloc(0xDEAD0000, 4096, mint_owner_instance_id(), buffer_id=1, access=AccessMode.READWRITE)


def test_submit_sub_rejects_device_tensor():
    ran = []
    w = Worker(level=3, num_sub_workers=1)
    sub = w.register(lambda args: None)
    w.init()
    try:
        dev = _device_handle()

        def orch(orch_, args, config):
            ta = TaskArgs()
            ta.add_tensor(dev.tensor(shapes=(16,), dtype=DataType.FLOAT32), TensorArgType.INPUT)
            ran.append(True)
            orch_.submit_sub(sub, ta)

        with pytest.raises(ValueError, match="DEVICE-space"):
            w.run(orch)
        assert ran, "the orchestration fn never ran — the assertion above would pass vacuously"
    finally:
        w.close()


def test_submit_sub_accepts_host_tensor():
    # The same call shape with a host backing goes through, so the guard is not blanket-rejecting.
    # The sub runs in a forked process, so the shared backing itself is the observable.
    def touch(args):
        args[0].buffer[:4] = b"\x2a\x00\x00\x00"

    w = Worker(level=3, num_sub_workers=1)
    sub = w.register(touch)
    w.init()
    try:
        buf = w.create_buffer(64)
        shm = buf.shm
        assert shm is not None

        def orch(orch_, args, config):
            ta = TaskArgs()
            ta.add_tensor(buf.tensor(shapes=(16,), dtype=DataType.INT32), TensorArgType.INOUT)
            orch_.submit_sub(sub, ta)

        w.run(orch)
        shm_buf = shm.buf
        assert shm_buf is not None
        assert bytes(shm_buf[:4]) == b"\x2a\x00\x00\x00"
    finally:
        w.close()


def test_sub_worker_materialization_refuses_a_device_tensor():
    # Depth behind the submit-time check: handed a blob directly, the host-side compute-leaf map
    # still refuses a device address rather than passing it to torch.
    dev = _device_handle()
    assert dev.address_space == AddressSpace.DEVICE
    blob = encode_blob([dev.tensor(shapes=(16,), dtype=DataType.FLOAT32)])
    src = ctypes.create_string_buffer(blob, len(blob))
    reg = ImportRegistry(ImportContext(is_host_endpoint=True))
    try:
        with pytest.raises(ValueError, match="DEVICE-space"):
            reg.mapped_args_from_blob(ctypes.addressof(src), len(blob))
    finally:
        reg.close()


def test_host_endpoint_materialize_refuses_a_device_tensor_directly():
    # The same backstop, exercised through materialize() itself rather than mapped_args_from_blob —
    # any entry point onto a host ImportRegistry must refuse a DEVICE backing, not just the one that
    # mapped_args_from_blob happens to guard. This endpoint carries no owner nonce, so it is a SUB
    # child: it can neither map the backing nor ask anyone to copy it. A Worker, the other host
    # endpoint, does carry one and gets a delegated attachment instead (see test_buffer.py).
    dev = _device_handle()
    reg = ImportRegistry(ImportContext(is_host_endpoint=True))
    with pytest.raises(ValueError, match="this endpoint is unrelated to"):
        reg.materialize(dev.to_descriptor())


def test_chip_materialization_refuses_a_foreign_chips_device_tensor():
    # Depth behind the submit-time provenance check (_child_prov_check_dispatch's owner_worker_id
    # match): handed a blob directly, a chip endpoint's own ImportRegistry still refuses a device
    # backing minted by a Worker it does not belong to.
    dev = _device_handle()
    foreign_chip_owner = mint_owner_instance_id()
    reg = ImportRegistry(ImportContext(is_host_endpoint=False, device_owner_instance_id=foreign_chip_owner))
    with pytest.raises(ValueError, match="this endpoint is unrelated to"):
        reg.materialize(dev.to_descriptor())


def test_device_backing_rejections_carry_the_endpoint_relation_reason_code():
    # Both materialize() rejections above are instances of the same capability-judgment question
    # the domain-scoped EndpointRegistry/RegionAccessService engine answers with
    # RegionAccessReasonCode.UNSUPPORTED_ENDPOINT_RELATION -- pinned here so the two mechanisms'
    # rejection vocabulary stays in sync rather than drifting back into two freeform strings.
    dev = _device_handle()

    host_reg = ImportRegistry(ImportContext(is_host_endpoint=True))
    with pytest.raises(ValueError, match=re.escape(RegionAccessReasonCode.UNSUPPORTED_ENDPOINT_RELATION.value)):
        host_reg.materialize(dev.to_descriptor())

    foreign_chip_owner = mint_owner_instance_id()
    chip_reg = ImportRegistry(ImportContext(is_host_endpoint=False, device_owner_instance_id=foreign_chip_owner))
    with pytest.raises(ValueError, match=re.escape(RegionAccessReasonCode.UNSUPPORTED_ENDPOINT_RELATION.value)):
        chip_reg.materialize(dev.to_descriptor())
