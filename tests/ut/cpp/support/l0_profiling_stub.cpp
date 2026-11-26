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

#include <atomic>

#include "common/l0_profiling.h"

// Run-level state for scheduler tests without hardware sampler configuration.
namespace {
std::atomic<bool> l0_enabled{false};
}

extern "C" bool is_l0_swimlane_enabled() { return l0_enabled.load(std::memory_order_acquire); }

extern "C" void set_l0_swimlane_enabled(bool enable, uint64_t /*run_epoch*/) {
    l0_enabled.store(enable, std::memory_order_release);
}
