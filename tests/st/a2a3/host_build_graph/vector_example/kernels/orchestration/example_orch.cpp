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
 * Arg layout: [a (IN), b (IN), f (OUT)] — 3 external tensors. The intermediate
 * tensors c, d, e are allocated by the runtime from the HeapRing.
 */

#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_ADD 0         // c = a + b
#define FUNC_ADD_SCALAR 1  // out = src + scalar
#define FUNC_MUL 2         // f = d * e

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 3,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    const Tensor &a = orch_args.tensor(0).ref();
    const Tensor &b = orch_args.tensor(1).ref();
    const Tensor &f = orch_args.tensor(2).ref();  // external output, written in place

    uint32_t SIZE = a.shapes[0];
    uint32_t inter_shapes[1] = {SIZE};
    TensorCreateInfo inter_ci(inter_shapes, 1, DataType::FLOAT32);

    union {
        float f32;
        uint64_t u64;
    } sconv;

    // task0: c = a + b
    L0TaskArgs p_add;
    p_add.add_input(a);
    p_add.add_input(b);
    p_add.add_output(inter_ci);
    TaskOutputTensors c_out = rt_submit_aiv_task(FUNC_ADD, p_add);
    Tensor c = c_out.get_ref(0);

    // task1: d = c + 1
    L0TaskArgs p_d;
    p_d.add_input(c);
    p_d.add_output(inter_ci);
    sconv.f32 = 1.0f;
    p_d.add_scalar(sconv.u64);
    TaskOutputTensors d_out = rt_submit_aiv_task(FUNC_ADD_SCALAR, p_d);
    Tensor d = d_out.get_ref(0);

    // task2: e = c + 2
    L0TaskArgs p_e;
    p_e.add_input(c);
    p_e.add_output(inter_ci);
    sconv.f32 = 2.0f;
    p_e.add_scalar(sconv.u64);
    TaskOutputTensors e_out = rt_submit_aiv_task(FUNC_ADD_SCALAR, p_e);
    Tensor e = e_out.get_ref(0);

    // task3: f = d * e  (write into the external output tensor)
    L0TaskArgs p_mul;
    p_mul.add_input(d);
    p_mul.add_input(e);
    p_mul.add_output(f);
    rt_submit_aiv_task(FUNC_MUL, p_mul);

    LOG_INFO_V9("[example_orch] Submitted 4 tasks for f = (a + b + 1) * (a + b + 2)");
}

}  // extern "C"
