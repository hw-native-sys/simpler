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

#define FUNC_MATMUL 0
#define FUNC_ADD 1

namespace {

// A5-native decoder-style topology: an AIC projection followed by an AIV
// residual/bias add. The first invocation records both nodes; later invocations
// submit one outer Graph task each.
void decoder_layer(const CoreTaskArgs &args) {
    const ChipTensor &input = args.tensor(0).ref();
    const ChipTensor &weight = args.tensor(1).ref();
    const ChipTensor &bias = args.tensor(2).ref();
    const ChipTensor &output = args.tensor(3).ref();

    const std::array<uint32_t, 1> shape{input.shapes[0]};
    TensorCreateInfo projected_info(shape.data(), static_cast<uint32_t>(shape.size()), DataType::FLOAT32);

    CoreTaskArgs matmul_args;
    matmul_args.add_input(input, weight);
    matmul_args.add_output(projected_info);
    TaskOutputTensors projected_outputs = rt_submit_aic_task(FUNC_MATMUL, matmul_args);

    CoreTaskArgs add_args;
    add_args.add_input(projected_outputs.get_ref(0), bias);
    add_args.add_output(output);
    rt_submit_aiv_task(FUNC_ADD, add_args);
}

void submit_layer(const CoreTaskArgs &args) { rt_submit_graph(&decoder_layer, args); }

}  // namespace

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 6,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    for (int32_t output_index = 3; output_index < 6; ++output_index) {
        CoreTaskArgs layer_args;
        layer_args.add_input(args.tensor(0).ref(), args.tensor(1).ref(), args.tensor(2).ref());
        layer_args.add_output(args.tensor(output_index).ref());
        submit_layer(layer_args);
    }
}

}  // extern "C"
