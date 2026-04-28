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

#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 0,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;

    uint32_t shape[1] = {1};
    TensorCreateInfo ci(shape, 1, DataType::FLOAT32);
    (void)alloc_tensors(ci);

    rt_report_fatal(PTO2_ERROR_EXPLICIT_ORCH_FATAL, "st injected fatal");

    // Exercise API short-circuit after fatal. These calls must become no-ops
    // instead of falling through into runtime-side asserts or extra reporting.
    Arg alloc_args;
    (void)alloc_tensors(alloc_args);

    Tensor dummy = make_tensor_external(reinterpret_cast<void *>(0x1), shape, 1);
    uint32_t indices[1] = {0};
    (void)get_tensor_data<uint64_t>(dummy, 0, indices);
    set_tensor_data<uint64_t>(dummy, 0, indices, 1U);

    rt_scope_begin();
    rt_scope_end();
}

}  // extern "C"
