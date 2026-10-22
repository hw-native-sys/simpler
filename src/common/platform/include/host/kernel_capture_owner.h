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

#include <cstdint>
#include <vector>

#include "task_interface/kernel_capture.h"

// The prepared native run retains its arena, callable bindings and fixed
// launch package until explicit graph release. Destruction makes no device
// calls: a failed enqueue or wait cannot silently free memory a model retains.
struct KernelCaptureOwner {
    simpler::tmr::KernelCaptureReference reference{};
    simpler::tmr::KernelCaptureRegistrationArgs registration{};
    uint64_t pinned_callable_hash{0};
    std::vector<unsigned char> launch_packet;
    void *caller_stream{nullptr};
    bool registered{false};
    bool enqueue_attempted{false};
    bool recorded{false};
};
