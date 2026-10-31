/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "host_domain_memory.h"
#include "host_backing.h"

#include <algorithm>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <stdexcept>
#include <string>

namespace simpler::a3 {
namespace {
void check(int rc, const char *op) {
    if (rc) throw std::runtime_error(std::string(op) + " failed with code " + std::to_string(rc));
}

void ensure_acl() {
    static std::once_flag once;
    std::call_once(once, [] {
        const aclError rc = aclInit(nullptr);
        if (rc != ACL_ERROR_REPEAT_INITIALIZE) check(rc, "aclInit");
    });
}

void *required_symbol(const char *name) {
    void *symbol = dlsym(RTLD_DEFAULT, name);
    if (!symbol) throw std::runtime_error(std::string("CANN ACL symbol not found: ") + name);
    return symbol;
}

void require_host_fabric() {
    const char *soc = aclrtGetSocName();
    if (!soc || std::strncmp(soc, "Ascend910_93", 12) != 0)
        throw std::runtime_error("Host Fabric backing requires an A3 device");
    (void)required_symbol("aclrtReserveMemAddressNoUCMemory");
}

void authorize_fabric(void *token, std::vector<int32_t> pids) {
    if (pids.empty() || std::any_of(pids.begin(), pids.end(), [](int32_t pid) {
            return pid <= 0;
        }))
        throw std::invalid_argument("Fabric authorization requires positive process identities");
    auto authorize = reinterpret_cast<decltype(&aclrtMemSetPidToShareableHandleV2)>(
        required_symbol("aclrtMemSetPidToShareableHandleV2")
    );
    check(authorize(token, ACL_MEM_SHARE_HANDLE_TYPE_FABRIC, pids.data(), pids.size()), "Fabric process authorization");
}
}  // namespace

std::pair<uint64_t, std::array<uint8_t, 128>>
HostDomainMemory::allocate(int device, uint32_t numa, uint64_t logical_bytes) {
    if (closing_ || handle_ || base_ || !logical_bytes) throw std::invalid_argument("Host allocation is not empty");
    ensure_acl();
    check(aclrtSetDevice(device), "aclrtSetDevice");
    require_host_fabric();
    device_ = device;
    aclrtPhysicalMemProp prop{};
    prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    prop.memAttr = ACL_MEM_P2P_HUGE;
    prop.location = {numa, ACL_MEM_LOCATION_TYPE_HOST_NUMA};
    size_t granularity = 0;
    check(
        aclrtMemGetAllocationGranularity(&prop, ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM, &granularity), "Host granularity"
    );
    if (!granularity || logical_bytes > SIZE_MAX - (granularity - 1))
        throw std::invalid_argument("Host allocation extent overflow");
    size_ = (logical_bytes + granularity - 1) / granularity * granularity;
    check(aclrtMallocPhysical(&handle_, size_, &prop, 0), "Host physical allocation");
    check(aclrtReserveMemAddress(&base_, size_, 0, nullptr, 1), "Host VA reservation");
    check(aclrtMapMem(base_, size_, 0, handle_, 0), "Host owner mapping");
    mapped_ = true;
    std::memset(base_, 0, logical_bytes);
    check(
        aclrtMemExportToShareableHandleV2(
            handle_, ACL_RT_IPC_MEM_EXPORT_FLAG_DEFAULT, ACL_MEM_SHARE_HANDLE_TYPE_FABRIC, token_.data()
        ),
        "Fabric export"
    );
    return {size_, token_};
}

void HostDomainMemory::authorize(const std::vector<int32_t> &pids) {
    if (closing_ || !mapped_ || !handle_) throw std::invalid_argument("Host owner has no live allocation");
    check(aclrtSetDevice(device_), "aclrtSetDevice");
    authorize_fabric(token_.data(), pids);
}

uint64_t HostDomainMemory::base() const {
    if (closing_ || !mapped_ || !base_) throw std::invalid_argument("Host owner has no live mapping");
    return reinterpret_cast<uint64_t>(base_);
}

void HostDomainMemory::release() {
    if (!base_ && !handle_) return;
    closing_ = true;
    check(aclrtSetDevice(device_), "aclrtSetDevice");
    if (mapped_) {
        check(aclrtUnmapMem(base_), "Host unmap");
        mapped_ = false;
    }
    if (base_) {
        check(aclrtReleaseMemAddress(base_), "Host VA release");
        base_ = nullptr;
    }
    if (handle_) {
        check(aclrtFreePhysical(handle_), "Host physical free");
        handle_ = nullptr;
    }
}

int HostDomainMemory::process_identity() {
    ensure_acl();
    int32_t pid = 0;
    auto get_pid = reinterpret_cast<decltype(&aclrtDeviceGetBareTgid)>(required_symbol("aclrtDeviceGetBareTgid"));
    check(get_pid(&pid), "Fabric process identity");
    if (pid <= 0) throw std::runtime_error("Invalid Fabric process identity");
    return pid;
}

}  // namespace simpler::a3

extern "C" int host_backing_allocate(
    void **owner, int device, uint32_t numa, uint64_t bytes, uint64_t *mapping_size, uint8_t *export_data,
    size_t export_capacity, size_t *export_size, char *error, size_t error_capacity
) {
    return simpler::host::call_host_backing(error, error_capacity, [&] {
        if (!owner || !mapping_size || !export_data || export_capacity < 128 || !export_size)
            throw std::invalid_argument("Invalid Host backing allocation output");
        if (!*owner) *owner = new simpler::a3::HostDomainMemory();
        const auto [size, token] = static_cast<simpler::a3::HostDomainMemory *>(*owner)->allocate(device, numa, bytes);
        std::memcpy(export_data, token.data(), token.size());
        *export_size = token.size();
        *mapping_size = size;
    });
}

extern "C" int
host_backing_authorize(void *owner, const int32_t *identities, size_t count, char *error, size_t error_capacity) {
    return simpler::host::call_host_backing(error, error_capacity, [&] {
        if (!owner) throw std::invalid_argument("Host owner has no live allocation");
        if (!identities || !count)
            throw std::invalid_argument("Fabric authorization requires positive process identities");
        static_cast<simpler::a3::HostDomainMemory *>(owner)->authorize({identities, identities + count});
    });
}

extern "C" int host_backing_get_base(void *owner, uint64_t *base, char *error, size_t error_capacity) {
    return simpler::host::call_host_backing(error, error_capacity, [&] {
        if (!owner || !base) throw std::invalid_argument("Host owner has no live mapping");
        *base = static_cast<simpler::a3::HostDomainMemory *>(owner)->base();
    });
}

extern "C" int host_backing_release(void **owner, char *error, size_t error_capacity) {
    return simpler::host::call_host_backing(error, error_capacity, [&] {
        if (!owner) throw std::invalid_argument("Invalid Host backing owner");
        if (!*owner) return;
        auto *memory = static_cast<simpler::a3::HostDomainMemory *>(*owner);
        memory->release();
        delete memory;
        *owner = nullptr;
    });
}

extern "C" int host_backing_process_identity(int32_t *identity, char *error, size_t error_capacity) {
    return simpler::host::call_host_backing(error, error_capacity, [&] {
        if (!identity) throw std::invalid_argument("Invalid Host process identity output");
        *identity = simpler::a3::HostDomainMemory::process_identity();
    });
}
