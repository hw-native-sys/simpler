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
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace simpler_dispatcher {

enum class BootstrapStatus : uint32_t { Pending, Success, Failed };
enum class BootstrapStage : uint32_t {
    Pending,
    Context,
    Mkdir,
    Create,
    Write,
    Close,
    Chmod,
    Rename,
    Complete,
    Verify,
    Link,
    Unlink,
    CacheHit
};

inline const char *bootstrap_stage_name(BootstrapStage stage) {
    switch (stage) {
    case BootstrapStage::Pending:
        return "pending";
    case BootstrapStage::Context:
        return "context";
    case BootstrapStage::Mkdir:
        return "mkdir";
    case BootstrapStage::Create:
        return "create";
    case BootstrapStage::Write:
        return "write";
    case BootstrapStage::Close:
        return "close";
    case BootstrapStage::Chmod:
        return "chmod";
    case BootstrapStage::Rename:
        return "rename";
    case BootstrapStage::Complete:
        return "complete";
    case BootstrapStage::Verify:
        return "verify";
    case BootstrapStage::Link:
        return "link";
    case BootstrapStage::Unlink:
        return "unlink";
    case BootstrapStage::CacheHit:
        return "cache-hit";
    }
    return "unknown";
}

constexpr uint32_t kBootstrapResultMagic = 0x42535450;
constexpr uint32_t kBootstrapResultVersion = 1;

// One bootstrap callback publishes this record before its stream completes.
// capacity_valid marks a successful statvfs; capacity_errno preserves a failed
// collection independently of the operation's original error_number.
// Complete means a new immutable image was published. CacheHit means an existing
// byte-identical image was verified; written_bytes counts only actual temporary
// file writes, which can be zero or nonzero when another publisher wins a race.
struct alignas(64) BootstrapResult {
    uint32_t magic{kBootstrapResultMagic};
    uint32_t version{kBootstrapResultVersion};
    BootstrapStatus status{BootstrapStatus::Pending};
    BootstrapStage stage{BootstrapStage::Pending};
    int32_t error_number{0};
    int32_t context_error{0};
    int32_t device_id{0};
    int32_t capacity_errno{0};
    uint64_t requested_bytes{0};
    uint64_t written_bytes{0};
    uint64_t block_bytes{0};
    uint64_t free_blocks{0};
    uint64_t available_blocks{0};
    uint64_t free_inodes{0};
    char path[320]{};
    uint32_t capacity_valid{0};
    uint8_t reserved[44]{};
};

// The CANN bootstrap front reads the three fields at 96/104/112. The remaining
// fields belong to the bundled dispatcher and preserve the 160-byte allocation.
struct BootstrapDeviceArgs {
    uint64_t unused[12]{};
    uint64_t aicpu_so_bin{0};
    uint64_t aicpu_so_len{0};
    uint64_t device_id{0};
    uint64_t inner_so_bin{0};
    uint64_t inner_so_len{0};
    uint64_t result_addr{0};
    uint64_t reserved[2]{};
};

static_assert(std::is_trivially_copyable_v<BootstrapResult> && std::is_standard_layout_v<BootstrapResult>);
static_assert(std::is_trivially_copyable_v<BootstrapDeviceArgs> && std::is_standard_layout_v<BootstrapDeviceArgs>);
static_assert(sizeof(BootstrapResult) == 448 && offsetof(BootstrapResult, path) == 80);
static_assert(sizeof(BootstrapDeviceArgs) == 160);
static_assert(offsetof(BootstrapDeviceArgs, aicpu_so_bin) == 96);
static_assert(offsetof(BootstrapDeviceArgs, aicpu_so_len) == 104);
static_assert(offsetof(BootstrapDeviceArgs, device_id) == 112);
static_assert(offsetof(BootstrapDeviceArgs, inner_so_bin) == 120);
static_assert(offsetof(BootstrapDeviceArgs, inner_so_len) == 128);
static_assert(offsetof(BootstrapDeviceArgs, result_addr) == 136);

}  // namespace simpler_dispatcher
