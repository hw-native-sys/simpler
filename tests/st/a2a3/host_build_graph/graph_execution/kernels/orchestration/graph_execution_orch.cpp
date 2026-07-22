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

#define FUNC_ADD 0
#define FUNC_ADD_SCALAR 1
#define FUNC_MUL 2

namespace {

void layer(const L2TaskArgs &args) {
    const Tensor &a = args.tensor(0).ref();
    const Tensor &b = args.tensor(1).ref();
    const Tensor &output = args.tensor(2).ref();

    uint32_t shape[1] = {a.shapes[0]};
    TensorCreateInfo intermediate(shape, 1, DataType::FLOAT32);

    L0TaskArgs add_args;
    add_args.add_input(a, b);
    add_args.add_output(intermediate);
    TaskOutputTensors add_outputs = rt_submit_aiv_task(FUNC_ADD, add_args);
    Tensor sum = add_outputs.get_ref(0);

    L0TaskArgs dynamic_args;
    dynamic_args.add_input(sum);
    dynamic_args.add_output(intermediate);
    dynamic_args.add_scalar(args.scalar(0));
    TaskOutputTensors dynamic_outputs = rt_submit_aiv_task(FUNC_ADD_SCALAR, dynamic_args);
    Tensor dynamic_sum = dynamic_outputs.get_ref(0);

    L0TaskArgs static_args;
    static_args.add_input(sum);
    static_args.add_output(intermediate);
    static_args.add_scalar(2.0F);
    PTO2TaskId explicit_dep = dynamic_outputs.task_id();
    static_args.set_dependencies(&explicit_dep, 1);
    TaskOutputTensors static_outputs = rt_submit_aiv_task(FUNC_ADD_SCALAR, static_args);
    Tensor static_sum = static_outputs.get_ref(0);

    L0TaskArgs mul_args;
    mul_args.add_input(dynamic_sum, static_sum);
    mul_args.add_output(output);
    rt_submit_aiv_task(FUNC_MUL, mul_args);
}

void submit_layer(const Tensor &a, const Tensor &b, const Tensor &output, float dynamic_scalar) {
    L2TaskArgs args;
    args.add_input(a, b);
    args.add_output(output);
    args.add_scalar(dynamic_scalar);
    rt_submit_graph(&layer, args);
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &args) {
    (void)args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 5,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const L2TaskArgs &args) {
    const Tensor &a = args.tensor(0).ref();
    const Tensor &b = args.tensor(1).ref();
    submit_layer(a, b, args.tensor(2).ref(), 1.0F);  // cache miss: record four tasks
    submit_layer(a, b, args.tensor(3).ref(), 3.0F);  // cache hit: one Graph task
    submit_layer(a, b, args.tensor(4).ref(), 5.0F);  // cache hit: one Graph task
}

}  // extern "C"
