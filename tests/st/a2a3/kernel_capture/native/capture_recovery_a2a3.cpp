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
#include <vector>
#include "common/hbg_execution_image.h"
#include "host_build_graph/runtime_core.h"

namespace simpler::capture_test {
namespace {
thread_local void *prefix_source = nullptr;
thread_local std::vector<unsigned char> original_prefix;
constexpr size_t prefix_capacity = CHIP_ALIGN_UP(sizeof(RuntimeContext), CHIP_ALIGN_SIZE) + sizeof(SharedMemoryHeader);
constexpr size_t capacity_offset =
    offsetof(RuntimeContext, prebuilt_layout) + offsetof(RuntimeArenaLayout, task_capacity);
}  // namespace

size_t prepare_snapshot(DeviceContextHandle, int scenario) {
    if (scenario == 13)
        check(aclrtMalloc(&prefix_source, prefix_capacity, ACL_MEM_MALLOC_HUGE_FIRST), "allocate-boot-fault-source");
    return 0;
}
void check_prepare(DeviceContextHandle, int, size_t) {}
bool needs_publications(int scenario) { return scenario == 13; }

bool record_boot_fault_prefix(const unsigned char *section, void *stream) {
    if (!prefix_source) return false;
    HbgExecutionImage image{};
    std::memcpy(&image, section, sizeof(image));
    require(image.mutable_prefix_bytes == prefix_capacity, "observed-runtime-context-prefix");
    original_prefix.assign(section + sizeof(image), section + sizeof(image) + image.mutable_prefix_bytes);
    // Each normal replay restores task_capacity from the RTS snapshot. This fault
    // fixture substitutes a controlled D2D source for precisely that mutable prefix;
    // the same AICPU task continues to bind all immutable image references.
    check(
        aclrtMemcpyAsync(
            reinterpret_cast<void *>(image.working_address), image.mutable_prefix_bytes, prefix_source,
            image.mutable_prefix_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream
        ),
        "record-boot-fault-source"
    );
    return true;
}

void check_publications(int scenario, const CapturedSubmission &) {
    if (scenario == 13) require(original_prefix.size() == prefix_capacity, "observed-boot-fault-source");
}

void inject_boot_failure() {
    uint64_t capacity = 0;
    std::memcpy(&capacity, original_prefix.data() + capacity_offset, sizeof(capacity));
    require(capacity != 0, "prepared-task-capacity");
    auto fault_prefix = original_prefix;
    std::memset(fault_prefix.data() + capacity_offset, 0, sizeof(capacity));
    check(
        aclrtMemcpy(
            prefix_source, prefix_capacity, fault_prefix.data(), fault_prefix.size(), ACL_MEMCPY_HOST_TO_DEVICE
        ),
        "inject-task-capacity"
    );
}
void restore_boot_failure() {
    check(
        aclrtMemcpy(
            prefix_source, prefix_capacity, original_prefix.data(), original_prefix.size(), ACL_MEMCPY_HOST_TO_DEVICE
        ),
        "restore-injected-task-capacity"
    );
}
void release_boot_fault_source() {
    if (prefix_source) check(aclrtFree(prefix_source), "free-boot-fault-source");
    prefix_source = nullptr;
}
void check_boot_failure() {}
void check_progress_timeout() {}

}  // namespace simpler::capture_test
