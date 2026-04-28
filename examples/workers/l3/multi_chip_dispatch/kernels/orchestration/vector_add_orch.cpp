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
 * Simplest possible orchestration for the Worker API demo.
 *
 *   out = a + b
 *
 * Exactly one AIV task, no intermediate scopes. For a multi-kernel example
 * with nested scopes see examples/a2a3/tensormap_and_ringbuffer/vector_example.
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
vector_add_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 3,  // a, b, out
    };
}

__attribute__((visibility("default"))) void vector_add_orchestration(const ChipStorageTaskArgs &orch_args) {
    Tensor a = from_tensor_arg(orch_args.tensor(0));
    Tensor b = from_tensor_arg(orch_args.tensor(1));
    Tensor out = from_tensor_arg(orch_args.tensor(2));

    Arg params;
    params.add_input(a);
    params.add_input(b);
    params.add_output(out);
    rt_submit_aiv_task(0, params);  // func_id=0 -> vector_add_kernel
}

}  // extern "C"
