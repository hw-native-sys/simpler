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
    for (const auto &launch : submission.launches) {
        for (size_t index = 0; index < launch.publication_count; ++index) {
            const auto &publication = submission.publications[index];
            if (reinterpret_cast<uintptr_t>(publication.destination) == launch.runtime_args &&
                publication.bytes >= offset + sizeof(void *)) {
                // The retained reference keeps RTS's pinned publication source alive.
                arena_binding = static_cast<char *>(const_cast<void *>(publication.source)) + offset;
            }
        }
    }
    require(arena_binding != nullptr, "observed-arena-binding");
}

void inject_boot_failure() {
    std::memcpy(&saved_arena, arena_binding, sizeof(saved_arena));
    require(saved_arena != nullptr, "prepared-arena-exists");
    void *missing_arena = nullptr;
    std::memcpy(arena_binding, &missing_arena, sizeof(missing_arena));
}

void restore_boot_failure() {
    std::memcpy(arena_binding, &saved_arena, sizeof(saved_arena));
    check(0, "restore-injected-arena-binding");
}

void check_boot_failure() {}
void check_progress_timeout() {}

}  // namespace simpler::capture_test
