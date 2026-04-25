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
 * Minimal orchestration for orch_so_cache test (a2a3 aicpu_build_graph)
 *
 * Computes: f = a + b  (single AIV task)
 *
 * Args layout (3 args):
 *   [0] = a (INPUT)   - 128 x 128 float32
 *   [1] = b (INPUT)   - 128 x 128 float32
 *   [2] = f (OUTPUT)  - 128 x 128 float32
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_ADD 0  // kernel_add: args[0..2] -> f = a + b

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 3,
    };
}

__attribute__((visibility("default"))) void
aicpu_orchestration_entry(PTO2Runtime *rt, const ChipStorageTaskArgs &orch_args) {
    Tensor ext_a = from_tensor_arg(orch_args.tensor(0));
    Tensor ext_b = from_tensor_arg(orch_args.tensor(1));
    Tensor ext_f = from_tensor_arg(orch_args.tensor(2));

    PTO2_SCOPE(rt) {
        // f = a + b
        Arg args;
        args.add_input(ext_a);
        args.add_input(ext_b);
        args.add_inout(ext_f);
        pto2_rt_submit_aiv_task(rt, FUNC_ADD, args);
    }
}

}  // extern "C"
