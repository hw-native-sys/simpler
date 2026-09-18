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
/**
 * Device work for #2267's late-read retention test.
 *
 * Dependency-only tasks rather than compiled kernels: the test is about when a
 * completed run's result can be read, so the work only has to occupy the device
 * long enough to be observed mid-flight. An incore kernel would tie this source
 * to one architecture, and `rt_submit_dummy_task` is offered by every
 * architecture and runtime — so one source covers all four combinations.
 *
 * The ring sizing is left at its default, so the scope admits every task and
 * the run completes.
 */

#include <cstdint>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {

// The successor has to still be executing when the predecessor's record is
// read, and that read costs ~11 us. The margin is what the test measures in, so
// it is sized for the *fastest* silicon rather than for the one this was
// developed on: 64 tasks left a2a3 ~100 us of headroom but proved too thin on
// a5 under a loaded runner, where the successor finished before the host could
// look. Well inside the default 16384-slot ring window, so nothing here is near
// an admission limit, and dependency-only tasks make each one cheap.
constexpr int32_t kTaskCount = 1024;

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return OrchestrationConfig{
        .expected_arg_count = 0,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    (void)orch_args;

    uint32_t shape[1] = {1};
    TensorCreateInfo ci(shape, 1, DataType::INT32);

    SIMPLER_SCOPE() {
        for (int32_t i = 0; i < kTaskCount; i++) {
            CoreTaskArgs args;
            args.add_output(ci);
            rt_submit_dummy_task(args);
        }
    }
}

}  // extern "C"
