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
 * Example: aicpu_orchestration_entry (device-side orchestration)
 *
 * DAG structure for formula: (a + b + 1)(a + b + 2) + (a + b)
 *   t0: c = a + b     (func_id=0, kernel_add)       [outer scope]
 *   t1: d = c + 1     (func_id=1, kernel_add_scalar) [inner scope]
 *   t2: e = c + 2     (func_id=1, kernel_add_scalar) [inner scope]
 *   t3: g = d * e     (func_id=2, kernel_mul)        [inner scope]
 *   t4: f = g + c     (func_id=0, kernel_add)        [inner scope]
 *   Dependencies: t0->t1, t0->t2, t1->t3, t2->t3, t0->t4, t3->t4
 *
 * Nested scope demonstration:
 *   - Inner scope owns t1, t2, t3, t4; intermediates d, e, g release on inner scope end
 *   - Outer scope owns t0; c persists across inner scope for t1, t2, t4
 *   - c flows from outer to inner scope (outer-scope tensors are visible to inner scopes)
 *
 * This file compiles as a standalone .so with zero runtime link dependencies.
 * All runtime calls go through the PTO2RuntimeOps function-pointer table.
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

extern "C" {

/**
 * Orchestration config — the executor reads these values to set up
 * shared memory and runtime before calling aicpu_orchestration_entry.
 */
__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 3,
    };
}

/**
 * Orchestration entry — runtime is bound implicitly by the framework.
 * The executor wraps this call in PTO2_SCOPE, so we are already inside
 * the outer scope on entry.
 */
__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    // golden shape = kernel shape, use from_tensor_arg() directly
    Tensor ext_a = from_tensor_arg(orch_args.tensor(0));
    Tensor ext_b = from_tensor_arg(orch_args.tensor(1));
    Tensor ext_f = from_tensor_arg(orch_args.tensor(2));

    uint32_t SIZE = orch_args.tensor(0).shapes[0];
    LOG_INFO_V0("===============SIZE=%u", SIZE);

    uint32_t inter_shapes[1] = {SIZE};
    TensorCreateInfo inter_ci(inter_shapes, 1, DataType::FLOAT32);

    // t0: c = a + b (kernel_id=0, kernel_add) [outer scope]
    Arg params_t0;
    params_t0.add_input(ext_a);
    params_t0.add_input(ext_b);
    params_t0.add_output(inter_ci);
    TaskOutputTensors outs_t0 = rt_submit_aiv_task(0, params_t0);  // kernel_add
    const Tensor &c = outs_t0.get_ref(0);

    // Inner scope: owns t1, t2, t3, t4; intermediates d, e, g release on scope end.
    // c flows in from outer scope (outer-scope tensors are visible to inner scopes).
    PTO2_SCOPE() {
        // t1: d = c + 1 (kernel_id=1, kernel_add_scalar)
        Arg params_t1;
        params_t1.add_input(c);
        params_t1.add_output(inter_ci);
        params_t1.add_scalar(1.0f);
        params_t1.add_scalar(3u);
        TaskOutputTensors outs_t1 = rt_submit_aiv_task(1, params_t1);  // kernel_add_scalar
        const Tensor &d = outs_t1.get_ref(0);

        // t2: e = c + 2 (kernel_id=1, kernel_add_scalar)
        Arg params_t2;
        params_t2.add_input(c);
        params_t2.add_output(inter_ci);
        params_t2.add_scalar(2.0f);
        params_t2.add_scalar(3u);
        TaskOutputTensors outs_t2 = rt_submit_aiv_task(1, params_t2);  // kernel_add_scalar
        const Tensor &e = outs_t2.get_ref(0);

        // t3: g = d * e (kernel_id=2, kernel_mul)
        Arg params_t3;
        params_t3.add_input(d);
        params_t3.add_input(e);
        params_t3.add_output(inter_ci);
        params_t3.add_scalar(3u);
        TaskOutputTensors outs_t3 = rt_submit_aiv_task(2, params_t3);  // kernel_mul
        const Tensor &g = outs_t3.get_ref(0);

        // t4: f = g + c (kernel_id=0, kernel_add)
        Arg params_t4;
        params_t4.add_input(g);
        params_t4.add_input(c);
        params_t4.add_output(ext_f);
        rt_submit_aiv_task(0, params_t4);  // kernel_add
    }  // inner scope ends: releases d, e, g
}

}  // extern "C"
