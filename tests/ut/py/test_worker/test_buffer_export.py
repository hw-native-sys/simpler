# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public export recovery uses the same lease API as normal retirement."""

from types import SimpleNamespace

import pytest
from simpler.buffer_export import BufferExportError
from simpler.task_interface import DataType

from ._harness import _FakeChipImpl, fake_chip_l3


@pytest.mark.parametrize("failure", ["retain-before", "retain-after", "release-before", "release-after"])
def test_export_lease_lifetime_and_recovery(monkeypatch, failure):
    operations = []

    def allocate(chip, size):
        base = chip.malloc(size)
        return SimpleNamespace(device_addr=base, mapping_bytes=4096, shareable_handle=base + 1)

    monkeypatch.setattr(_FakeChipImpl, "device_id", 0, raising=False)
    monkeypatch.setattr(_FakeChipImpl, "_malloc_shareable", allocate, raising=False)
    monkeypatch.setattr(_FakeChipImpl, "_vmm_borrow", lambda *args: None, raising=False)
    with fake_chip_l3(monkeypatch) as owner:
        original = owner._vmm_request

        def record_request(worker, operation, *args, **kwargs):
            operations.append(operation)
            return original(worker, operation, *args, **kwargs)

        monkeypatch.setattr(owner, "_vmm_request", record_request)
        root = owner.alloc_child_tensor(0, (512,), DataType.UINT8, exportable=True)
        first, second = owner.export_buffer(root), owner.export_buffer(root)
        assert first.token != second.token and first.token >= 1 << 63
        owner.release_export(first)
        with pytest.raises(RuntimeError, match="external"):
            owner.free(root)
        owner.release_export(second)
        request = owner._vmm_request

        def lose_reply(worker, operation, *args, **kwargs):
            target = 2 if failure.startswith("release") else 1
            if operation == target and failure.endswith("before"):
                raise TimeoutError("not sent")
            reply = request(worker, operation, *args, **kwargs)
            if operation == target:
                raise TimeoutError("reply lost")
            return reply

        exported = owner.export_buffer(root) if failure.startswith("release") else None
        with monkeypatch.context() as patch:
            patch.setattr(owner, "_vmm_request", lose_reply)
            with pytest.raises(TimeoutError if exported else BufferExportError) as error:
                owner.release_export(exported) if exported else owner.export_buffer(root)
            if exported is None:
                exported = error.value.exported
            with pytest.raises(TimeoutError):
                owner.release_export(exported)
            with pytest.raises(RuntimeError, match="external"):
                owner.free(root)
            with pytest.raises(RuntimeError, match="external"):
                owner.close()
        operations.clear()
        owner.release_export(exported)
        assert operations == ([1, 2] if failure.startswith("retain") else [2])
        owner.free(root)
