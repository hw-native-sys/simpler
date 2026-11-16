# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public retirement through real mailbox transport and native VMM release stages.

The existing chip harness supplies device-free storage. The endpoint, child command
loop, provenance and native VMM release ledger are the production implementations.
"""

import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from time import monotonic
from types import SimpleNamespace

import pytest
import simpler.worker as worker_module
from simpler import buffer_export
from simpler.buffer_export import BufferExportError
from simpler.task_interface import DataType

from ._harness import _FakeChipImpl, fake_chip_l3, install_fake_chip


def _shareable_storage(monkeypatch):
    def allocate(chip, size):
        base = chip.malloc(size)
        return SimpleNamespace(device_addr=base, mapping_bytes=4096, shareable_handle=base + 1)

    monkeypatch.setattr(_FakeChipImpl, "device_id", 0, raising=False)
    monkeypatch.setattr(_FakeChipImpl, "_malloc_shareable", allocate, raising=False)
    monkeypatch.setattr(_FakeChipImpl, "_vmm_borrow", lambda *args: None, raising=False)


@contextmanager
def _owner(monkeypatch, level):
    _shareable_storage(monkeypatch)
    if level == 3:
        with fake_chip_l3(monkeypatch) as owner:
            yield owner
    else:
        install_fake_chip(monkeypatch)
        owner = worker_module.Worker(level=2, device_id=0, platform="a2a3sim", runtime="tensormap_and_ringbuffer")
        try:
            owner.init()
            yield owner
        finally:
            owner.close()


@pytest.mark.parametrize("level", [2, 3])
def test_public_free_retries_cleanup_without_restoring_use(monkeypatch, level):
    attempts = multiprocessing.get_context("fork").Value("i", 0)
    free = _FakeChipImpl.free

    def fail_once(chip, ptr):
        with attempts.get_lock():
            attempts.value += 1
            if attempts.value == 1:
                raise RuntimeError("injected storage release failure")
        free(chip, ptr)

    monkeypatch.setattr(_FakeChipImpl, "free", fail_once)
    with _owner(monkeypatch, level) as owner:
        root = (
            owner.malloc(512, exportable=True)
            if level == 2
            else owner.alloc_child_tensor(0, (512,), DataType.UINT8, exportable=True)
        )
        with pytest.raises(RuntimeError, match="storage release failure"):
            owner.free(root)
        with pytest.raises(ValueError, match="live"):
            owner.export_buffer(root)
        with pytest.raises(ValueError):
            owner.copy_to(root, bytearray(512))
        with pytest.raises(ValueError, match="retired"):
            owner.free(replace(root, nbytes=256))
        # Neither mutable field can redirect the cleanup to another pointer/owner.
        owner.free(replace(root, base=root.base + 4096, owner_worker_id=7))
        assert attempts.value == 2
        with pytest.raises(ValueError, match="live"):
            owner.free(root)


@pytest.mark.parametrize("operation", [buffer_export._ControlOp.RETAIN, buffer_export._ControlOp.RELEASE])
@pytest.mark.parametrize("after_execution", [False, True])
def test_late_mailbox_reply_retires_the_original_lease(monkeypatch, operation, after_execution):
    context = multiprocessing.get_context("fork")
    entered, resume, acknowledged = (context.Event() for _ in range(3))
    calls = context.Array("i", 3)
    armed = context.Value("b", False)
    handler = buffer_export._handle_command

    def controlled(chip, payload):
        op = buffer_export._CONTROL_FRAME.unpack(payload)[0]
        if op == operation and armed.value:
            armed.value = False
            result = handler(chip, payload) if after_execution else None
            if after_execution:
                calls[op] += 1
            entered.set()
            assert resume.wait(20), "test did not release the child control handler"
            if not after_execution:
                result = handler(chip, payload)
                calls[op] += 1
            acknowledged.set()
            return result
        result = handler(chip, payload)
        calls[op] += 1
        return result

    _shareable_storage(monkeypatch)
    monkeypatch.setattr(buffer_export, "_handle_command", controlled)
    with fake_chip_l3(monkeypatch, device_ids=(0, 1)) as owner:
        root = owner.alloc_child_tensor(0, (512,), DataType.UINT8, exportable=True)
        exported = owner.export_buffer(root) if operation == buffer_export._ControlOp.RELEASE else None
        owner._py_control_timeout_s = 1.0
        armed.value = True
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                call = (
                    executor.submit(owner.release_export, exported)
                    if exported
                    else executor.submit(owner.export_buffer, root)
                )
                assert entered.wait(5)
                assert not call.done(), "child did not reach the requested fault stage before timeout"
                with pytest.raises(RuntimeError if exported else BufferExportError) as error:
                    call.result(timeout=5)
            if exported is None:
                exported = error.value.exported
            with pytest.raises(RuntimeError, match="external"):
                owner.free(root)
            with pytest.raises(RuntimeError, match="external"):
                owner.close()
            # An unrelated command must be rejected before overwriting the pending frame.
            pending_view = owner._chip_shms[0].buf
            assert pending_view is not None
            frame = bytes(pending_view[: worker_module.MAILBOX_FRAME_SIZE])
            with pytest.raises(RuntimeError, match="timed-out|reconcile"):
                owner.alloc_child_tensor(0, (64,), DataType.UINT8)
            assert bytes(pending_view[: worker_module.MAILBOX_FRAME_SIZE]) == frame
            other = owner.alloc_child_tensor(1, (64,), DataType.UINT8)
            owner.free(other)
        finally:
            resume.set()
        assert acknowledged.wait(5)
        owner._py_control_timeout_s = 5
        owner.release_export(exported)
        assert calls[1:] == [1, 1], "reconciliation replayed an already-applied RETAIN or RELEASE"
        owner.free(root)
        # The poisoned computation endpoint is never reopened for ordinary controls.
        with pytest.raises(RuntimeError, match="timed-out"):
            owner.alloc_child_tensor(0, (64,), DataType.UINT8)


def test_unreturned_allocation_keeps_payload_until_close_can_reconcile(monkeypatch):
    context = multiprocessing.get_context("fork")
    entered, resume = context.Event(), context.Event()
    handler = buffer_export._handle_command

    def controlled(chip, payload):
        result = handler(chip, payload)
        entered.set()
        assert resume.wait(20)
        return result

    _shareable_storage(monkeypatch)
    monkeypatch.setattr(buffer_export, "_handle_command", controlled)
    with fake_chip_l3(monkeypatch) as owner:
        owner._py_control_timeout_s = 1.0
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                call = executor.submit(owner.alloc_child_tensor, 0, (512,), DataType.UINT8, exportable=True)
                assert entered.wait(5)
                with pytest.raises(RuntimeError, match="timed out"):
                    call.result(timeout=5)
            with pytest.raises(RuntimeError, match="unresolved"):
                owner.close()
        finally:
            resume.set()
        deadline = monotonic() + 5
        view = owner._chip_shms[0].buf
        state = worker_module._buffer_field_addr(view, worker_module._OFF_STATE)
        while worker_module._mailbox_load_i32(state) != worker_module._CONTROL_DONE:
            assert monotonic() < deadline, "child did not publish its original completion"
        owner.close()


def test_owner_exit_does_not_fabricate_a_successful_allocation(monkeypatch):
    _shareable_storage(monkeypatch)
    monkeypatch.setattr(buffer_export, "_handle_command", lambda *args: os._exit(0))
    with fake_chip_l3(monkeypatch) as owner:
        owner._py_control_timeout_s = 1.0
        with pytest.raises(RuntimeError, match="child|timed out"):
            owner.alloc_child_tensor(0, (512,), DataType.UINT8, exportable=True)
        owner.close()
