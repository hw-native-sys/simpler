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

#include "capture_recovery.h"

#include <acl/acl.h>
#include <cstring>
#include "host_build_graph/runtime.h"

namespace simpler::capture_test {
namespace {
thread_local void *arena_binding = nullptr;
thread_local void *saved_arena = nullptr;
}  // namespace

size_t prepare_snapshot(DeviceContextHandle, int) { return 0; }
void check_prepare(DeviceContextHandle, int, size_t) {}

bool needs_publications(int scenario) { return scenario == 13; }

void check_publications(int scenario, const CapturedSubmission &submission) {
    if (!needs_publications(scenario)) return;
    arena_binding = nullptr;
    constexpr size_t offset = offsetof(DeviceRuntimeLaunchDesc, prebuilt_arena_base_);
    for (const auto &launch : submission.cpu_launches) {
        if (launch.runtime_args != 0 && launch.runtime_bytes >= offset + sizeof(void *)) {
            arena_binding = reinterpret_cast<void *>(launch.runtime_args + offset);
        }
    }
    require(arena_binding != nullptr, "observed-arena-binding");
}

void inject_boot_failure() {
    check(
        aclrtMemcpy(&saved_arena, sizeof(saved_arena), arena_binding, sizeof(saved_arena), ACL_MEMCPY_DEVICE_TO_HOST),
        "read-captured-arena-binding"
    );
    require(saved_arena != nullptr, "prepared-arena-exists");
    void *missing_arena = nullptr;
    check(
        aclrtMemcpy(
            arena_binding, sizeof(missing_arena), &missing_arena, sizeof(missing_arena), ACL_MEMCPY_HOST_TO_DEVICE
        ),
        "inject-captured-arena-binding"
    );
}

void restore_boot_failure() {
    check(
        aclrtMemcpy(arena_binding, sizeof(saved_arena), &saved_arena, sizeof(saved_arena), ACL_MEMCPY_HOST_TO_DEVICE),
        "restore-captured-arena-binding"
    );
    check(0, "restore-injected-arena-binding");
}

void check_boot_failure() {}
void check_progress_timeout() {}

}  // namespace simpler::capture_test
