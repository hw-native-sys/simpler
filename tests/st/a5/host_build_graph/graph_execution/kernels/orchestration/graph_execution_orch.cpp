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

#include <array>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_ADD 0
#define FUNC_ADD_SCALAR 1
#define FUNC_MUL 2

namespace {

void layer(const CoreTaskArgs &args, float left_delta, float right_delta) {
    const ChipTensor &a = args.tensor(0).ref();
    const ChipTensor &b = args.tensor(1).ref();
    const ChipTensor &output = args.tensor(2).ref();
    const float ndim_delta = b.ndims == 1 ? 0.0F : 2.0F;

    const std::array<uint32_t, 1> shape{a.shapes[0]};
    TensorCreateInfo intermediate(shape.data(), static_cast<uint32_t>(shape.size()), DataType::FLOAT32);

    CoreTaskArgs add_args;
    add_args.add_input(a, b);
    add_args.add_output(intermediate);
    const std::array<PTO2TaskId, 1> external_dep{a.owner_task_id};
    add_args.set_dependencies(external_dep.data(), static_cast<uint32_t>(external_dep.size()));
    TaskOutputTensors add_outputs = rt_submit_aiv_task(FUNC_ADD, add_args);
    ChipTensor sum = add_outputs.get_ref(0);

    CoreTaskArgs fence_args;
    fence_args.add_inout(sum);
    rt_submit_dummy_task(fence_args);

    CoreTaskArgs left_args;
    left_args.add_input(sum);
    left_args.add_output(intermediate);
    left_args.add_scalar(left_delta + ndim_delta);
    left_args.set_allow_early_resolve(true);
    TaskOutputTensors left_outputs = rt_submit_aiv_task(FUNC_ADD_SCALAR, left_args);
    ChipTensor left = left_outputs.get_ref(0);

    CoreTaskArgs right_args;
    right_args.add_input(sum);
    right_args.add_output(intermediate);
    right_args.add_scalar(right_delta + ndim_delta);
    TaskOutputTensors right_outputs = rt_submit_aiv_task(FUNC_ADD_SCALAR, right_args);
    ChipTensor right = right_outputs.get_ref(0);

    CoreTaskArgs mul_args;
    mul_args.add_input(left, right);
    mul_args.add_output(output);
    rt_submit_aiv_task(FUNC_MUL, mul_args);
}

void submit_layer(const CoreTaskArgs &args) { rt_submit_graph(&layer, args, 1.0F, 2.0F); }

}  // namespace

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 5,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const ChipTensor &a = args.tensor(0).ref();
    const ChipTensor &b = args.tensor(1).ref();

    const std::array<uint32_t, 1> shape{a.shapes[0]};
    TensorCreateInfo seeded_input_info(shape.data(), static_cast<uint32_t>(shape.size()), DataType::FLOAT32);
    CoreTaskArgs seed_args;
    seed_args.add_input(a);
    seed_args.add_output(seeded_input_info);
    seed_args.add_scalar(0.0F);
    TaskOutputTensors seed_outputs = rt_submit_aiv_task(FUNC_ADD_SCALAR, seed_args);
    ChipTensor seeded_a = seed_outputs.get_ref(0);

    for (int32_t output_index = 2; output_index < 5; ++output_index) {
        CoreTaskArgs layer_args;
        layer_args.add_input(seeded_a, b);
        layer_args.add_output(args.tensor(output_index).ref());
        submit_layer(layer_args);  // first call records; later calls submit one Graph task
    }
}

}  // extern "C"
