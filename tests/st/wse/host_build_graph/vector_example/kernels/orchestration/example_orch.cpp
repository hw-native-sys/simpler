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
 * Example Orchestration Function — submit_task / TensorMap form
 *
 * Builds the task graph for: f = (a + b + 1) * (a + b + 2)
 *
 * Dependencies are discovered automatically by the TensorMap from the
 * add_input/add_output directions: c is produced by task0 and consumed by
 * task1/task2; d and e are produced by task1/task2 and consumed by task3.
 *
 * Arg layout: [a, b, f, c, d, e]. All tensors are views of fixed device SRAM.
 */

#include <stdint.h>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_ADD 0         // c = a + b
#define FUNC_ADD_SCALAR 1  // out = src + scalar
#define FUNC_MUL 2         // f = d * e

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return OrchestrationConfig{
        .expected_arg_count = 6,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const simpler::hbg::Tensor &a = orch_args.tensor(0).ref();
    const simpler::hbg::Tensor &b = orch_args.tensor(1).ref();
    const simpler::hbg::Tensor &f = orch_args.tensor(2).ref();  // external output, written in place

    const auto &c = orch_args.tensor(3).ref();
    const auto &d = orch_args.tensor(4).ref();
    const auto &e = orch_args.tensor(5).ref();

    union {
        float f32;
        uint64_t u64;
    } sconv;

    // task0: c = a + b
    CoreTaskArgs p_add;
    p_add.add_input(a);
    p_add.add_input(b);
    p_add.add_output(c);
    rt_submit_aiv_task(FUNC_ADD, p_add);

    // task1: d = c + 1
    CoreTaskArgs p_d;
    p_d.add_input(c);
    p_d.add_output(d);
    sconv.f32 = 1.0f;
    p_d.add_scalar(sconv.u64);
    rt_submit_aiv_task(FUNC_ADD_SCALAR, p_d);

    // task2: e = c + 2
    CoreTaskArgs p_e;
    p_e.add_input(c);
    p_e.add_output(e);
    sconv.f32 = 2.0f;
    p_e.add_scalar(sconv.u64);
    rt_submit_aiv_task(FUNC_ADD_SCALAR, p_e);

    // task3: f = d * e  (write into the external output tensor)
    CoreTaskArgs p_mul;
    p_mul.add_input(d);
    p_mul.add_input(e);
    p_mul.add_output(f);
    rt_submit_aiv_task(FUNC_MUL, p_mul);

    LOG_INFO("[example_orch] Submitted 4 tasks for f = (a + b + 1) * (a + b + 2)");
}

}  // extern "C"
