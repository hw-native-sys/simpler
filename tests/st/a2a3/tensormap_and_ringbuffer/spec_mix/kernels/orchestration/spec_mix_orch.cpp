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
 * Speculative early-dispatch — MIX consumer coverage.
 *
 * DAG:
 *   t0 (AIV, FLAGGED): c = a + b
 *   t1 (MIX, depends on t0):
 *        aiv0 reads args[0..2] = [c, b, d] -> d = c + b   (= a + 2b)
 *        aiv1 reads args[3..5] = [c, b, e] -> e = c * b   (= (a + b) * b)
 *
 * While t0 runs, the MIX consumer t1 (single block, 2 subtasks) is pre-staged
 * onto an idle cluster — both its AIV cores gated on the DATA_MAIN_BASE doorbell
 * — and released the instant t0's FIN satisfies t1's fanin. Exercises the
 * multi-doorbell release path (staged_count == 2).
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_ADD 0  // kernel_add_standalone: reads args[0..2]
#define FUNC_MUL 1  // kernel_mul_standalone: reads args[3..5]

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 4,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    Tensor ext_a = from_tensor_arg(orch_args.tensor(0));
    Tensor ext_b = from_tensor_arg(orch_args.tensor(1));
    Tensor ext_d = from_tensor_arg(orch_args.tensor(2));
    Tensor ext_e = from_tensor_arg(orch_args.tensor(3));

    uint32_t SIZE = orch_args.tensor(0).shapes[0];
    uint32_t inter_shapes[1] = {SIZE};
    TensorCreateInfo inter_ci(inter_shapes, 1, DataType::FLOAT32);

    // t0: c = a + b  [flagged producer — its MIX consumer may pre-stage]
    Arg params_t0;
    params_t0.add_input(ext_a);
    params_t0.add_input(ext_b);
    params_t0.add_output(inter_ci);
    params_t0.set_allow_early_resolve();
    TaskOutputTensors outs_t0 = rt_submit_aiv_task(FUNC_ADD, params_t0);
    const Tensor &c = outs_t0.get_ref(0);

    // t1 (MIX): aiv0 add(c, b -> d), aiv1 mul(c, b -> e)
    MixedKernels mk;
    mk.aiv0_kernel_id = FUNC_ADD;
    mk.aiv1_kernel_id = FUNC_MUL;
    Arg params_t1;
    params_t1.add_input(c);       // args[0]
    params_t1.add_input(ext_b);   // args[1]
    params_t1.add_output(ext_d);  // args[2]
    params_t1.add_input(c);       // args[3]
    params_t1.add_input(ext_b);   // args[4]
    params_t1.add_output(ext_e);  // args[5]
    rt_submit_task(mk, params_t1);
}

}  // extern "C"
