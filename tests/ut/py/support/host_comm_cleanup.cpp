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
#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "platform_comm/comm.h"
using aclError = int;
using aclrtDrvMemHandle = void *;
constexpr int ACL_SUCCESS = 0, ACL_MEM_SHARE_HANDLE_TYPE_FABRIC = 2;
constexpr int ACL_RT_IPC_MEM_EXPORT_FLAG_DEFAULT = 0;
struct CommContext {};
struct VmmWindow {
    bool live = true;
};
struct aclrtMemFabricHandle {
    char bytes[128];
};
static std::vector<std::string> calls;
static std::string fail_at;
static int step(const char *name) {
    calls.emplace_back(name);
    return fail_at == name ? 701 : 0;
}
#define LOG_ERROR(...) ((void)0)
#define LOG_WARN(...) ((void)0)
static void record_first_error(int status, int *error) {
    if (!*error) *error = status;
}
static int aclrtUnmapMem(void *) { return step("unmap"); }
static int aclrtReleaseMemAddress(void *) { return step("release_va"); }
static int aclrtFreePhysical(void *) { return step("free_handle"); }
static int aclrtFree(void *) { return step("free_ctx"); }
static int release_vmm_window(VmmWindow *window) {
    if (!window->live) return 0;
    window->live = false;
    return step("hbm");
}
static int release_vmm_windows(std::vector<VmmWindow> *windows) {
    int status = 0;
    for (auto &window : *windows)
        record_first_error(release_vmm_window(&window), &status);
    return status;
}
static int aclrtReserveMemAddressNoUCMemory(void **out, size_t, size_t, void *address, uint64_t) {
    *out = address;
    return 0;
}
constexpr void *RTLD_DEFAULT = nullptr;
static void *dlsym(void *, const char *) { return reinterpret_cast<void *>(&aclrtReserveMemAddressNoUCMemory); }
static int aclrtMemImportFromShareableHandleV2(aclrtMemFabricHandle *, int, int, void **out) {
    *out = reinterpret_cast<void *>(2);
    return step("import");
}
static int aclrtMapMem(void *, size_t, size_t, void *, int) { return step("map"); }
// PRODUCTION_BODIES
struct CommHandle_ {
    int rank = 0;
    std::string rootinfo_path, run_token;
    std::unordered_map<uint64_t, std::unique_ptr<DomainAllocation>> domain_allocations;
};
static std::string domain_barrier_tag(uint64_t, const char *) { return "release"; }
static bool file_barrier(const std::string &, int, int, const std::string &, const std::string &) {
    return step("barrier") == 0;
}
// TEST_CASES
static std::unique_ptr<DomainAllocation> allocation() {
    auto owner = std::make_unique<DomainAllocation>();
    owner->nranks = 1;
    owner->host_base = reinterpret_cast<void *>(1);
    owner->host_handle = reinterpret_cast<void *>(2);
    owner->host_mapped = true;
    owner->device_ctx = reinterpret_cast<CommContext *>(3);
    owner->peer_windows.emplace_back();
    return owner;
}
static int count(const char *name) { return std::count(calls.begin(), calls.end(), name); }
static CommGlobalDomainDescriptor descriptor() {
    CommGlobalDomainDescriptor value{};
    value.version = COMM_GLOBAL_DOMAIN_VERSION;
    value.profile = COMM_GLOBAL_DOMAIN_PROFILE_A3_FABRIC;
    value.handle_size = sizeof(aclrtMemFabricHandle);
    value.mapping_size = 4096;
    return value;
}
int main() {
    auto exported = descriptor();
    for (const char *failure : {"unmap", "release_va", "free_handle"}) {
        CommHandle_ handle;
        handle.domain_allocations.emplace(7, allocation());
        auto &owner = *handle.domain_allocations.at(7);
        auto *context = owner.device_ctx;
        calls.clear();
        fail_at = failure;
        assert(comm_detach_domain_host(&handle, 7) == 701);
        assert(owner.host_closing && owner.host_handle);
        uint64_t base = 123;
        assert(comm_attach_domain_host(&handle, 7, &exported, &base) != 0);
        assert(base == 0 && count("import") == 0);
        assert(comm_detach_domain_host(&handle, 7) == 701);
        fail_at.clear();
        assert(comm_detach_domain_host(&handle, 7) == 0);
        assert(!owner.host_closing && !owner.host_base && !owner.host_handle && !owner.host_mapped);
        assert(owner.host_reservation == 0 && owner.device_ctx == context);
        auto completed = calls.size();
        assert(comm_detach_domain_host(&handle, 7) == 0 && calls.size() == completed);
        assert(comm_attach_domain_host(&handle, 7, &exported, &base) == 0 && base != 0);
        assert(comm_detach_domain_host(&handle, 7) == 0);
        assert(owner.device_ctx == context && owner.local_window.live && owner.peer_windows.front().live);
        assert(count("barrier") == 0 && count("hbm") == 0 && count("free_ctx") == 0);
    }
    for (const char *failure : {"unmap", "release_va", "free_handle"}) {
        CommHandle_ handle;
        handle.domain_allocations.emplace(7, allocation());
        calls.clear();
        fail_at = failure;
        assert(comm_release_domain_windows(&handle, 7, 1, 0) == 701);
        assert(comm_release_domain_windows(&handle, 7, 1, 0) == 701);
        assert(handle.domain_allocations.size() == 1);
        assert(count("barrier") == 1 && count("hbm") == 0 && count("free_ctx") == 0);
        uint64_t base = 123;
        assert(comm_attach_domain_host(&handle, 7, &exported, &base) != 0 && base == 0);
        fail_at.clear();
        assert(comm_release_domain_windows(&handle, 7, 1, 0) == 0);
        assert(handle.domain_allocations.empty());
        assert(count("barrier") == 1 && count("hbm") == 2 && count("free_ctx") == 1);
        for (const char *stage : {"unmap", "release_va", "free_handle"})
            assert(count(stage) == (std::string(stage) == failure ? 3 : 1));
    }
    for (const char *failure : {"barrier", "hbm", "free_ctx"}) {
        CommHandle_ handle;
        handle.domain_allocations.emplace(7, allocation());
        calls.clear();
        fail_at = failure;
        assert(comm_release_domain_windows(&handle, 7, 1, 0) != 0);
        assert(handle.domain_allocations.size() == 1);
        fail_at.clear();
        uint64_t base = 123;
        assert(comm_attach_domain_host(&handle, 7, &exported, &base) != 0 && base == 0);
        assert(comm_detach_domain_host(&handle, 7) == 0);
        assert(comm_attach_domain_host(&handle, 7, &exported, &base) != 0 && base == 0);
        int hbm_calls = count("hbm");
        int result = comm_release_domain_windows(&handle, 7, 1, 0);
        assert(count("barrier") == 1 && count("hbm") == hbm_calls);
        if (std::string(failure) == "free_ctx")
            assert(result == 0 && handle.domain_allocations.empty() && count("free_ctx") == 2);
        else assert(result != 0 && handle.domain_allocations.size() == 1);
    }
    calls.clear();
    { auto owner = allocation(); }
    assert(count("unmap") == 1 && count("release_va") == 1 && count("free_handle") == 1);
    calls.clear();
    fail_at = "unmap";
    { auto owner = allocation(); }
    assert(count("unmap") == 1 && count("release_va") == 0 && count("free_handle") == 0);
    fail_at.clear();
}
