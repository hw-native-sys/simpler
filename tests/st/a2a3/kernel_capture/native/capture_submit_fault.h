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
#include <vector>

struct CaptureSubmitFaultResult {
    int accepted_core = 0;
    int cpu_calls = 0;
    int submissions_after_failure = 0;
    int boundary_rc = 0;
    int conflict_rc = 0;
    bool fired = false;
};

void arm_capture_submit_fault(void *caller, void *event);
CaptureSubmitFaultResult finish_capture_submit_fault();

namespace simpler::capture_test {

struct CapturedPublication {
    void *destination = nullptr;
    const void *source = nullptr;
    uint64_t bytes = 0;
};

struct CapturedLaunch {
    uint64_t shared_state_addr = 0;
    uint64_t worker_context_base = 0;
    uint32_t runtime_mode = 0;
    int32_t worker_count = 0;
    size_t publication_count = 0;
};

struct CapturedCpuLaunch {
    uint64_t runtime_args = 0;
    uint32_t runtime_bytes = 0;
    uint64_t image_bytes = 0;
    uint64_t working_address = 0;
};

struct CapturedSubmission {
    std::vector<CapturedPublication> publications;
    std::vector<CapturedLaunch> launches;
    std::vector<CapturedCpuLaunch> cpu_launches;
};

void begin_capture_publication_probe();
CapturedSubmission finish_capture_publication_probe();

}  // namespace simpler::capture_test
