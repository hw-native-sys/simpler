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

namespace simpler::tmr {

enum class TmrCoreCommand : uint32_t { Wait = 0, Open = 1, Cancel = 2 };
enum class TmrCoreRelease : uint32_t { Wait = 0, Release = 1 };
enum class TmrCompletion : uint32_t { Pending = 0, Complete = 1 };

constexpr uint32_t kTmrHostCancel = 0xffffffffu;

// AICore writes the first cache line; AICPU writes the second. Host clears both only between
// non-overlapping invocations; architecture-specific cache maintenance is required.
struct alignas(64) TmrCoreReport {
    uint32_t physical_core_id;
    uint32_t core_type;
    uint32_t ready;
    uint32_t exited;
    uint8_t report_reserved[48];
    uint32_t command;
    uint32_t release;
    uint64_t round_epoch;
    uint8_t command_reserved[48];
};

// Host cancel occupies its own cache line. Once AICPU is submitted, Host
// cannot cancel by overwriting either the live report pool or this device line.
struct alignas(64) TmrLaunchControl {
    uint32_t host_cancel;
    uint8_t host_reserved[60];
    int32_t runtime_status;
    int32_t cleanup_status;
    uint64_t round_epoch;
    uint32_t completion;
    uint8_t device_reserved[44];
};

// Both addresses identify stable device allocations owned by the context.
struct TmrKernelAicoreArgs {
    uint64_t resident_kernel_args;
    uint64_t context_descriptor;
};

static_assert(std::is_standard_layout_v<TmrCoreReport> && std::is_trivially_copyable_v<TmrCoreReport>);
static_assert(sizeof(TmrCoreReport) == 128 && alignof(TmrCoreReport) == 64);
static_assert(offsetof(TmrCoreReport, physical_core_id) == 0);
static_assert(offsetof(TmrCoreReport, core_type) == 4);
static_assert(offsetof(TmrCoreReport, ready) == 8);
static_assert(offsetof(TmrCoreReport, exited) == 12);
static_assert(offsetof(TmrCoreReport, report_reserved) == 16);
static_assert(offsetof(TmrCoreReport, command) == 64);
static_assert(offsetof(TmrCoreReport, release) == 68);
static_assert(offsetof(TmrCoreReport, round_epoch) == 72);
static_assert(offsetof(TmrCoreReport, command_reserved) == 80);

static_assert(std::is_standard_layout_v<TmrLaunchControl> && std::is_trivially_copyable_v<TmrLaunchControl>);
static_assert(sizeof(TmrLaunchControl) == 128 && alignof(TmrLaunchControl) == 64);
static_assert(offsetof(TmrLaunchControl, host_cancel) == 0);
static_assert(offsetof(TmrLaunchControl, host_reserved) == 4);
static_assert(offsetof(TmrLaunchControl, runtime_status) == 64);
static_assert(offsetof(TmrLaunchControl, cleanup_status) == 68);
static_assert(offsetof(TmrLaunchControl, round_epoch) == 72);
static_assert(offsetof(TmrLaunchControl, completion) == 80);
static_assert(offsetof(TmrLaunchControl, device_reserved) == 84);

static_assert(std::is_standard_layout_v<TmrKernelAicoreArgs> && std::is_trivially_copyable_v<TmrKernelAicoreArgs>);
static_assert(sizeof(TmrKernelAicoreArgs) == 16 && alignof(TmrKernelAicoreArgs) == 8);
static_assert(offsetof(TmrKernelAicoreArgs, resident_kernel_args) == 0);
static_assert(offsetof(TmrKernelAicoreArgs, context_descriptor) == 8);

}  // namespace simpler::tmr
