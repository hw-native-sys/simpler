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
 * AICPU orchestration for the vector example.
 *
 * DAG structure for formula: f = (a + b + 1) * (a + b + 2)
 *   t0: c = a + b     (func_id=0, kernel_add)
 *   t1: d = c + 1     (func_id=1, kernel_add_scalar)
 *   t2: e = c + 2     (func_id=1, kernel_add_scalar)
 *   t3: f = d * e     (func_id=2, kernel_mul)
 *   Dependencies: t0->t1, t0->t2, t1->t3, t2->t3
 *
 * Uses explicit add_dependency for all dependency edges (no TensorMap).
 * Tasks are batch-published at scope_end.
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

static uint64_t float_to_u64(float f) {
    union {
        float f32;
        uint64_t u64;
    } conv;
    conv.u64 = 0;
    conv.f32 = f;
    return conv.u64;
}

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(
    const ChipStorageTaskArgs& orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 3,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(
    PTO2Runtime* rt, const ChipStorageTaskArgs& orch_args, int orch_thread_num, int orch_thread_index) {
    (void)orch_thread_num;
    (void)orch_thread_index;

    // golden shape = kernel shape, use from_tensor_arg() directly
    Tensor ext_a = from_tensor_arg(orch_args.tensor(0));
    Tensor ext_b = from_tensor_arg(orch_args.tensor(1));
    Tensor ext_f = from_tensor_arg(orch_args.tensor(2));

    uint32_t SIZE = orch_args.tensor(0).shapes[0];

    uint32_t shapes[1] = {SIZE};

    PTO2_SCOPE(rt) {
        // t0: c = a + b
        Arg args_t0;
        args_t0.add_input(ext_a);
        args_t0.add_input(ext_b);
        args_t0.add_output(TensorCreateInfo(shapes, 1, DataType::FLOAT32));
        SubmitResult r0 = pto2_rt_submit_aiv_task(rt, 0, args_t0);

        // t1: d = c + 1.0
        Arg args_t1;
        args_t1.add_input(r0.outputs.get_ref(0));
        args_t1.add_output(TensorCreateInfo(shapes, 1, DataType::FLOAT32));
        args_t1.add_scalar(float_to_u64(1.0f));
        SubmitResult r1 = pto2_rt_submit_aiv_task(rt, 1, args_t1);
        pto2_rt_add_dependency(rt, r0.task_id, r1.task_id);

        // t2: e = c + 2.0
        Arg args_t2;
        args_t2.add_input(r0.outputs.get_ref(0));
        args_t2.add_output(TensorCreateInfo(shapes, 1, DataType::FLOAT32));
        args_t2.add_scalar(float_to_u64(2.0f));
        SubmitResult r2 = pto2_rt_submit_aiv_task(rt, 1, args_t2);
        pto2_rt_add_dependency(rt, r0.task_id, r2.task_id);

        // t3: f = d * e
        Arg args_t3;
        args_t3.add_input(r1.outputs.get_ref(0));
        args_t3.add_input(r2.outputs.get_ref(0));
        args_t3.add_inout(ext_f);
        SubmitResult r3 = pto2_rt_submit_aiv_task(rt, 2, args_t3);
        pto2_rt_add_dependency(rt, r1.task_id, r3.task_id);
        pto2_rt_add_dependency(rt, r2.task_id, r3.task_id);
    }  // scope_end: batch-publish all tasks
}

}  // extern "C"
