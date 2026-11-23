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

#include <cassert>
#include <cstring>
#include <string>
#include <vector>

#include "../../cpp/common/platform_comm/comm_lifecycle_rank.h"

namespace {
std::string scenario;
std::vector<std::string> cleanup;
CommContext context{};
int destroy_calls = 0;

void *create_context() { return reinterpret_cast<void *>(0x101); }
void destroy_context(void *ctx) {
    assert(ctx == reinterpret_cast<void *>(0x101));
    cleanup.emplace_back("context");
}
int ensure_ready(void *, int) { return 0; }
CommHandle init_comm(int, int, void *, const char *) { return reinterpret_cast<CommHandle>(0x404); }
int alloc_windows(CommHandle, size_t, uint64_t *out) {
    *out = reinterpret_cast<uint64_t>(&context);
    return scenario == "alloc_unconfirmed" ? -1 : 0;
}
int window_base(CommHandle, uint64_t *out) {
    *out = context.windowsIn[0];
    return 0;
}
int window_size(CommHandle, size_t *out) {
    *out = context.winSize;
    return 0;
}
int barrier(CommHandle) { return scenario.find("barrier_") == 0 ? -1 : 0; }
int destroy_comm(CommHandle handle) {
    assert(handle == reinterpret_cast<CommHandle>(0x404));
    ++destroy_calls;
    if (scenario.find("unconfirmed") != std::string::npos) return COMM_RECLAMATION_UNCONFIRMED;
    return scenario == "destroy_error" ? -1 : 0;
}
}  // namespace

extern "C" void *dlopen(const char *, int) noexcept { return reinterpret_cast<void *>(0x505); }
extern "C" char *dlerror() noexcept { return nullptr; }
extern "C" void *dlsym(void *, const char *name) noexcept {
    if (std::strcmp(name, "create_device_context") == 0) return reinterpret_cast<void *>(create_context);
    if (std::strcmp(name, "destroy_device_context") == 0) return reinterpret_cast<void *>(destroy_context);
    if (std::strcmp(name, "ensure_acl_ready_ctx") == 0) return reinterpret_cast<void *>(ensure_ready);
    if (std::strcmp(name, "comm_init") == 0) return reinterpret_cast<void *>(init_comm);
    if (std::strcmp(name, "comm_alloc_windows") == 0) return reinterpret_cast<void *>(alloc_windows);
    if (std::strcmp(name, "comm_get_local_window_base") == 0) return reinterpret_cast<void *>(window_base);
    if (std::strcmp(name, "comm_get_window_size") == 0) return reinterpret_cast<void *>(window_size);
    if (std::strcmp(name, "comm_barrier") == 0) return reinterpret_cast<void *>(barrier);
    if (std::strcmp(name, "comm_destroy") == 0) return reinterpret_cast<void *>(destroy_comm);
    return nullptr;
}
extern "C" int dlclose(void *handle) noexcept {
    assert(handle == reinterpret_cast<void *>(0x505));
    cleanup.emplace_back("dlclose");
    return 0;
}
extern "C" aclError aclrtCreateStream(aclrtStream *out) {
    *out = reinterpret_cast<aclrtStream>(0x303);
    return ACL_SUCCESS;
}
extern "C" aclError aclrtDestroyStream(aclrtStream stream) {
    assert(stream == reinterpret_cast<aclrtStream>(0x303));
    cleanup.emplace_back("stream");
    return ACL_SUCCESS;
}
extern "C" aclError aclrtMemcpy(void *out, size_t, const void *in, size_t bytes, aclrtMemcpyKind) {
    std::memcpy(out, in, bytes);
    return ACL_SUCCESS;
}

int main(int argc, char **argv) {
    assert(argc == 2);
    scenario = argv[1];
    context.rankId = 0;
    context.rankNum = 2;
    context.winSize = 4096;
    context.windowsIn[0] = 0x2000;
    context.windowsIn[1] = 0x3000;
    const int rc = comm_lifecycle_test::run_rank(0, 2, 0, "unused-rootinfo");
    assert(destroy_calls == 1);
    const bool unconfirmed = scenario.find("unconfirmed") != std::string::npos;
    if (unconfirmed) {
        assert(cleanup.empty());
    } else {
        assert((cleanup == std::vector<std::string>{"stream", "context", "dlclose"}));
    }
    const int expected = scenario == "alloc_unconfirmed" ? comm_lifecycle_test::EXIT_ALLOC :
                         scenario.find("barrier_") == 0  ? comm_lifecycle_test::EXIT_BARRIER :
                         scenario.find("destroy_") == 0  ? comm_lifecycle_test::EXIT_DESTROY :
                                                           0;
    assert(rc == expected);
}
