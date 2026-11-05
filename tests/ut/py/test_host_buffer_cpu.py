# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Restricted Host registrations retain CPU imports and failed cleanup owners."""

import os

import pytest
from _task_interface import DataType
from simpler.buffer import (
    AccessMode,
    ImportContext,
    ImportRegistry,
    create_host_shared_buffer,
    mint_owner_instance_id,
    wrap_device_malloc,
)
from simpler.comm_endpoints import DEVICE_AICPU, AdapterProfile


def test_confirmed_device_mapping_keeps_restricted_grant_and_context():
    owner = mint_owner_instance_id()
    buffer = wrap_device_malloc(4096, 64, owner, 3)
    environment = (os.getpid(), 1, 1, 0, False)
    current: list[tuple[int, int, int, int, bool]] = [environment]
    registry = ImportRegistry(
        ImportContext(
            DEVICE_AICPU,
            device_owner_instance_id=owner,
            execution_environment=environment,
            current_environment=lambda: current[0],
        )
    )
    registry.register_attachment(
        buffer.to_descriptor(),
        buffer.base,
        mapping_ref=1,
        access=AccessMode.READ,
        profile=AdapterProfile.DEVICE_LOCAL,
        live=lambda: True,
    )
    mapped = buffer.tensor((16,), DataType.FLOAT32)._with_mapping_ref(1)
    registry.require_mapping_access(mapped, AccessMode.READ)
    with pytest.raises(ValueError, match="grant"):
        registry.require_mapping_access(mapped, AccessMode.WRITE)
    current[0] = (os.getpid(), 1, 2, 0, False)
    with pytest.raises(ValueError, match="not live"):
        registry.resolve_tensor(mapped)
    registry.close()


class Registration:
    attachment_profile = AdapterProfile.HOST_REGISTER_MAP
    closing = False
    releases = 0
    base = 0

    def register(self, device, imported):
        self.base = imported.base

    def release(self):
        self.closing = True
        self.releases += 1


def registry_for_host():
    environment = (os.getpid(), 1, 1, 0, False)
    return ImportRegistry(
        ImportContext(
            DEVICE_AICPU,
            platform="a2a3",
            runtime="tensormap_and_ringbuffer",
            execution_environment=environment,
            current_environment=lambda: environment,
        )
    )


def test_shared_registration_last_unmap_preserves_native_cpu_import():
    buffer = create_host_shared_buffer(64, mint_owner_instance_id(), 5)
    registry = registry_for_host()
    registrations = [Registration(), Registration()]
    try:
        for ref, access, owner in ((1, AccessMode.READ, registrations[0]), (2, AccessMode.WRITE, registrations[1])):
            registry.import_device_host(buffer.to_descriptor(), owner, device=0, mapping_ref=ref, access=access)
        first = buffer.tensor((16,), DataType.FLOAT32)._with_mapping_ref(1)
        second = buffer.tensor((16,), DataType.FLOAT32)._with_mapping_ref(2)
        assert registry.resolve_tensor(first).host_backing is registry.resolve_tensor(second).host_backing
        with pytest.raises(ValueError, match="grant"):
            registry.require_mapping_access(first, AccessMode.WRITE)
        registry.unmap_attachment(buffer.identity, 1)
        assert not registrations[0].closing
        registry.require_mapping_access(second, AccessMode.WRITE)
        registry.unmap_attachment(buffer.identity, 2)
        assert registrations[0].closing and registrations[0].releases == 1
        assert registry.resolve_tensor(buffer.tensor((16,), DataType.FLOAT32)).base > 0
    finally:
        registry.close()
        buffer.close()


def test_partial_registration_failure_retains_closed_owner_for_cleanup_retry():
    buffer = create_host_shared_buffer(64, mint_owner_instance_id(), 6)
    registry = registry_for_host()

    class FailingRegistration(Registration):
        def register(self, device, imported):
            raise RuntimeError("injected registration failure")

        def release(self):
            super().release()
            if self.releases == 1:
                raise RuntimeError("injected unregister failure")

    owner = FailingRegistration()
    try:
        with pytest.raises(RuntimeError, match="registration failure") as caught:
            registry.import_device_host(buffer.to_descriptor(), owner, device=0, mapping_ref=1, access=AccessMode.READ)
        assert "unregister failure" in str(caught.value.__cause__)
        with pytest.raises(ValueError):
            registry.resolve_tensor(buffer.tensor((16,), DataType.FLOAT32)._with_mapping_ref(1))
        registry.unmap_attachment(buffer.identity, 1)
        assert owner.releases == 2
    finally:
        registry.close()
        buffer.close()
