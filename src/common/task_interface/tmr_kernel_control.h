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

#include "common/core_type.h"

namespace simpler::tmr {

enum class TmrCompletion : uint32_t { Pending = 0, Complete = 1 };

// AICore publishes identity before AICPU writes task and opens the register window.
struct alignas(64) TmrCoreReport {
    volatile uint32_t aicpu_ready;
    volatile uint32_t aicore_done;
    volatile uint64_t task;
    volatile CoreType core_type;
    volatile uint32_t physical_core_id;
    volatile uint64_t report_epoch;
};

struct alignas(64) TmrLaunchControl {
    int32_t runtime_status;
    int32_t cleanup_status;
    uint64_t round_epoch;
    uint32_t completion;
};

// Both addresses identify stable device allocations owned by the context.
struct TmrKernelAicoreArgs {
    uint64_t resident_kernel_args;
    uint64_t context_descriptor;
};

static_assert(std::is_standard_layout_v<TmrCoreReport> && std::is_trivially_copyable_v<TmrCoreReport>);
static_assert(sizeof(TmrCoreReport) == 64 && alignof(TmrCoreReport) == 64);
static_assert(offsetof(TmrCoreReport, task) == 8);
static_assert(offsetof(TmrCoreReport, physical_core_id) == 20);
static_assert(offsetof(TmrCoreReport, report_epoch) == 24);
static_assert(std::is_standard_layout_v<TmrLaunchControl> && std::is_trivially_copyable_v<TmrLaunchControl>);
static_assert(sizeof(TmrLaunchControl) == 64 && alignof(TmrLaunchControl) == 64);

static_assert(std::is_standard_layout_v<TmrKernelAicoreArgs> && std::is_trivially_copyable_v<TmrKernelAicoreArgs>);
static_assert(sizeof(TmrKernelAicoreArgs) == 16 && alignof(TmrKernelAicoreArgs) == 8);
static_assert(offsetof(TmrKernelAicoreArgs, resident_kernel_args) == 0);
static_assert(offsetof(TmrKernelAicoreArgs, context_descriptor) == 8);

}  // namespace simpler::tmr
