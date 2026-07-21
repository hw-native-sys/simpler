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
 * Matmul diamond orchestration — submit_task / TensorMap form
 *
 * Builds: F = exp(sqrt(log(A)) @ W1 + sqrt(log(A)) @ W2)
 *
 *       t0 (sqrt(log), AIV)
 *      /  \
 *    t1    t2   (matmul, AIC)
 *      \  /
 *       t3 (add+exp, AIV)
 *
 * Dependencies are discovered by the TensorMap from the add_input/add_output
 * directions. Intermediate b is FP16 (matmul input); c, d are FP32.
 *
 * Arg layout: [a (IN, fp16), w1 (IN, fp16), w2 (IN, fp16), f (OUT, fp32)].
 */

#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_LOG_SQRT 0
#define FUNC_MATMUL 1
#define FUNC_ADD_EXP 2

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 4,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    const Tensor &a = orch_args.tensor(0).ref();
    const Tensor &w1 = orch_args.tensor(1).ref();
    const Tensor &w2 = orch_args.tensor(2).ref();
    const Tensor &f = orch_args.tensor(3).ref();  // external output, written in place

    uint32_t SIZE = a.shapes[0];
    uint32_t shapes[1] = {SIZE};
    TensorCreateInfo b_ci(shapes, 1, DataType::FLOAT16);  // sqrt(log(A)) — half
    TensorCreateInfo cd_ci(shapes, 1, DataType::FLOAT32);

    // task0: b = sqrt(log(a))
    L0TaskArgs p0;
    p0.add_input(a);
    p0.add_output(b_ci);
    TaskOutputTensors b_out = rt_submit_aiv_task(FUNC_LOG_SQRT, p0);
    Tensor b = b_out.get_ref(0);

    // task1: c = b @ w1
    L0TaskArgs p1;
    p1.add_input(b);
    p1.add_input(w1);
    p1.add_output(cd_ci);
    TaskOutputTensors c_out = rt_submit_aic_task(FUNC_MATMUL, p1);
    Tensor c = c_out.get_ref(0);

    // task2: d = b @ w2
    L0TaskArgs p2;
    p2.add_input(b);
    p2.add_input(w2);
    p2.add_output(cd_ci);
    TaskOutputTensors d_out = rt_submit_aic_task(FUNC_MATMUL, p2);
    Tensor d = d_out.get_ref(0);

    // task3: f = exp(c + d)
    L0TaskArgs p3;
    p3.add_input(c);
    p3.add_input(d);
    p3.add_output(f);
    rt_submit_aiv_task(FUNC_ADD_EXP, p3);

    LOG_INFO_V9("[matmul_orch] Submitted 4-task diamond");
}

}  // extern "C"
