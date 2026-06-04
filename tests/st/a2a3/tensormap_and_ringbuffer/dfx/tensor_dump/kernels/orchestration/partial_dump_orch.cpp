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
        .expected_arg_count = 3,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    Tensor ext_a = from_tensor_arg(orch_args.tensor(0));
    Tensor ext_b = from_tensor_arg(orch_args.tensor(1));
    Tensor ext_f = from_tensor_arg(orch_args.tensor(2));

    uint32_t size = orch_args.tensor(0).shapes[0];
    uint32_t inter_shapes[1] = {size};
    TensorCreateInfo inter_ci(inter_shapes, 1, DataType::FLOAT32);

    Arg params_t0;
    params_t0.add_input(ext_a);
    params_t0.add_input(ext_b);
    params_t0.add_output(inter_ci);
    TaskOutputTensors outs_t0 = rt_submit_aiv_task(0, params_t0);
    const Tensor &c = outs_t0.get_ref(0);

    PTO2_SCOPE() {
        Arg params_t1;
        params_t1.add_input(c);
        params_t1.add_output(inter_ci);
        params_t1.add_scalar(1.0f);
        params_t1.add_scalar(3u);
        TaskOutputTensors outs_t1 = rt_submit_aiv_task(1, params_t1);
        const Tensor &d = outs_t1.get_ref(0);

        Arg params_t2;
        params_t2.add_input(c);
        params_t2.add_output(inter_ci);
        params_t2.add_scalar(2.0f);
        params_t2.add_scalar(3u);
        TaskOutputTensors outs_t2 = rt_submit_aiv_task(1, params_t2);
        const Tensor &e = outs_t2.get_ref(0);

        Arg params_t3;
        params_t3.add_input(d);
        params_t3.add_input(e);
        params_t3.add_output(inter_ci);
        params_t3.add_scalar(3u);
        // Partial dump, tensor granularity: select specific tensors of this task
        // (input d + the output; input e is left unmarked).
        params_t3.dump(d, inter_ci);
        TaskOutputTensors outs_t3 = rt_submit_aiv_task(2, params_t3);
        const Tensor &g = outs_t3.get_ref(0);

        Arg params_t4;
        params_t4.add_input(g);
        params_t4.add_input(c);
        params_t4.add_output(ext_f);
        // Partial dump, task granularity: no-arg dump() selects the whole task
        // (every tensor arg on this Arg).
        params_t4.dump();
        rt_submit_aiv_task(0, params_t4);
    }
}

}  // extern "C"
