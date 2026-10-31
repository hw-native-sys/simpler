# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host allocation backend; resource identity and lifetime live in host_domain.

Local exports reuse the native global descriptor v2 envelope, not rank membership.
Its version/profile rules are checked by both Python and comm_attach_domain_host;
changing that envelope requires a paired Python/native migration.
"""

import os

from _task_interface import _HostDomainMemory

from .buffer import create_host_shared_buffer
from .comm_endpoints import AdapterProfile
from .global_comm_domain import (
    GLOBAL_DOMAIN_PROFILE_A3_FABRIC,
    GLOBAL_DOMAIN_PROFILE_IDS,
    GLOBAL_DOMAIN_PROFILE_SIM,
    GLOBAL_DOMAIN_VERSION,
    GlobalDomainDescriptor,
)


class SharedHostBacking:
    profile = AdapterProfile.HOST_SHM_MAP
    attachment_profile = AdapterProfile.HOST_SHM_MAP

    def __init__(self, *, host_runtime_path=None):
        self.shared = None

    @staticmethod
    def process_identity():
        return 0

    def authorize(self, identities):
        pass

    def allocate(self, spec, identity, device, rank_count):
        page_size = os.sysconf("SC_PAGE_SIZE")
        mapping_size = (spec.nbytes + page_size - 1) // page_size * page_size
        self.shared = create_host_shared_buffer(
            mapping_size,
            bytes(identity.owner_instance_id),
            identity.buffer_id,
            generation=identity.generation,
        )
        return self.shared.base, GlobalDomainDescriptor(
            GLOBAL_DOMAIN_VERSION,
            GLOBAL_DOMAIN_PROFILE_IDS[GLOBAL_DOMAIN_PROFILE_SIM],
            0,
            rank_count,
            mapping_size,
            self.shared.body,
        )

    def release(self):
        if self.shared is not None:
            self.shared.close()


class A3HostBacking:
    profile = AdapterProfile.HOST_VMM_MAP
    attachment_profile = AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT

    def __init__(self, *, host_runtime_path=None):
        if host_runtime_path is None:
            raise ValueError("Host backing requires an initialized platform runtime")
        self.native = _HostDomainMemory(str(host_runtime_path))
        self.native.require_mapping_support()

    def allocate(self, spec, identity, device, rank_count):
        size, token = self.native.allocate(device, spec.numa_id, spec.nbytes)
        return self.native.base, GlobalDomainDescriptor(
            GLOBAL_DOMAIN_VERSION,
            GLOBAL_DOMAIN_PROFILE_IDS[GLOBAL_DOMAIN_PROFILE_A3_FABRIC],
            0,
            rank_count,
            size,
            token,
        )

    def process_identity(self):
        return self.native.process_identity()

    def authorize(self, identities):
        self.native.authorize(identities)

    def release(self):
        self.native.release()


_HOST_BACKENDS = {"a2a3sim": SharedHostBacking, "a2a3": A3HostBacking}


def host_process_identity(platform, chip_worker=None):
    """Optional Host identity piggybacks on the existing HBM communicator handshake."""
    if platform == "a2a3":
        if chip_worker is None:
            raise ValueError("Host identity requires the initialized chip endpoint")
        return chip_worker.host_backing_process_identity()
    return 0


def host_backing_type(platform):
    """Select before any allocation. Unsupported platforms fail without side effects."""
    try:
        return _HOST_BACKENDS[platform]
    except KeyError:
        raise ValueError(f"Host domain backing is not implemented for {platform!r}") from None


def host_attachment_profile(descriptor: GlobalDomainDescriptor, platform: str) -> AdapterProfile:
    expected = GLOBAL_DOMAIN_PROFILE_SIM if platform == "a2a3sim" else GLOBAL_DOMAIN_PROFILE_A3_FABRIC
    backend = host_backing_type(platform)
    if descriptor.profile_id != GLOBAL_DOMAIN_PROFILE_IDS[expected]:
        raise ValueError("Host attachment descriptor does not match the platform backend")
    return backend.attachment_profile
