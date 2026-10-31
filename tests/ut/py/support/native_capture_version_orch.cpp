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
 * Builds the task graph for: f = (a + b + s0) * (a + b + s1)
 *
 * Dependencies are discovered automatically by the TensorMap from the
 * add_input/add_output directions: c is produced by task0 and consumed by
 * task1/task2; d and e are produced by task1/task2 and consumed by task3.
 *
 * Arg layout: [a (IN), b (IN), f (OUT)] — 3 external tensors and 2 scalar parameters. The intermediate
 * tensors c, d, e are allocated by the runtime from the HeapRing.
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
        .expected_arg_count = 5,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const simpler::hbg::Tensor &a = orch_args.tensor(0).ref();
    const simpler::hbg::Tensor &b = orch_args.tensor(1).ref();
    const simpler::hbg::Tensor &f = orch_args.tensor(2).ref();  // external output, written in place

    uint32_t SIZE = a.shapes[0];
    uint32_t inter_shapes[1] = {SIZE};
    TensorCreateInfo inter_ci(inter_shapes, 1, DataType::FLOAT32);

    union {
        float f32;
        uint64_t u64;
    } sconv;

    // task0: c = a + b
    CoreTaskArgs p_add;
    p_add.add_input(a);
    p_add.add_input(b);
    p_add.add_output(inter_ci);
    TaskOutputTensors c_out = rt_submit_aiv_task(FUNC_ADD, p_add);
    simpler::hbg::Tensor c = c_out.get_ref(0);

    // task1: d = c + s0
    CoreTaskArgs p_d;
    p_d.add_input(c);
    p_d.add_output(inter_ci);
    sconv.u64 = orch_args.scalar<uint64_t>(0);
    p_d.add_scalar(sconv.u64);
    TaskOutputTensors d_out = rt_submit_aiv_task(FUNC_ADD_SCALAR, p_d);
    simpler::hbg::Tensor d = d_out.get_ref(0);

    // task2: e = c + s1
    CoreTaskArgs p_e;
    p_e.add_input(c);
    p_e.add_output(inter_ci);
    sconv.u64 = orch_args.scalar<uint64_t>(1);
    p_e.add_scalar(sconv.u64);
    TaskOutputTensors e_out = rt_submit_aiv_task(FUNC_ADD_SCALAR, p_e);
    simpler::hbg::Tensor e = e_out.get_ref(0);

    // task3: f = d * e  (write into the external output tensor)
    CoreTaskArgs p_mul;
    p_mul.add_input(d);
    p_mul.add_input(e);
    p_mul.add_output(f);
    rt_submit_aiv_task(FUNC_MUL, p_mul);

    LOG_INFO("[example_orch] Submitted 4 tasks for f = (a + b + s0) * (a + b + s1)");
}

}  // extern "C"
