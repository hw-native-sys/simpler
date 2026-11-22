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

#include <atomic>
#include <cstdint>
#include <variant>

struct SharedMemoryHeader;

namespace simpler::tmr {

struct OrchestrationSetupFailure {
    std::atomic<bool> &runtime_init_ready;
};
struct SharedFailure {
    enum class Source { Orchestrator, Scheduler };
    Source source;
    int32_t code;
    int32_t idle_iterations{0};
    int32_t last_progress_count{-1};
};
struct LocalSchedulerFailure {
    SharedMemoryHeader *header;
    int32_t code;
};
struct ExplicitAbort {};
struct ProgressTimeout {
    SharedMemoryHeader *header;
    int32_t idle_iterations;
    int32_t last_progress_count;
};
using Failure = std::variant<OrchestrationSetupFailure, SharedFailure, ProgressTimeout, ExplicitAbort>;

}  // namespace simpler::tmr
