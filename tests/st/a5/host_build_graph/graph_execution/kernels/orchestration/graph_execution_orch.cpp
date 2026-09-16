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

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_ADD 0
#define FUNC_ADD_SCALAR 1
#define FUNC_MUL 2

namespace {

void layer(const GraphTaskArgs &args, int variant) {
    const simpler::hbg::Tensor &a = args.tensor(0).ref();
    const simpler::hbg::Tensor &b = args.tensor(1).ref();
    const simpler::hbg::Tensor &output = args.tensor(2).ref();

    const std::array<uint32_t, 1> shape{a.shapes[0]};
    TensorCreateInfo intermediate(shape.data(), static_cast<uint32_t>(shape.size()), DataType::FLOAT32);

    // A dependency-only root. Its shape is DUMMY, which indexes no per-shape
    // dispatch queue, so it must never be staged by the shell's early release —
    // this body is where that is exercised on device.
    CoreTaskArgs dummy_root_args;
    rt_submit_dummy_task(dummy_root_args);

    CoreTaskArgs add_args;
    add_args.add_input(a, b);
    add_args.add_output(intermediate);
    const std::array<TaskId, 1> external_dep{a.owner_task_id};
    add_args.set_dependencies(external_dep.data(), static_cast<uint32_t>(external_dep.size()));
    TaskOutputTensors add_outputs = rt_submit_aiv_task(FUNC_ADD, add_args);
    simpler::hbg::Tensor sum = add_outputs.get_ref(0);

    CoreTaskArgs fence_args;
    fence_args.add_inout(sum);
    rt_submit_dummy_task(fence_args);

    CoreTaskArgs left_args;
    left_args.add_input(sum);
    left_args.add_output(intermediate);
    if (variant == 0) {
        left_args.add_scalar(args.scalar(0));
    } else {
        left_args.add_scalar(100.0F);
    }
    left_args.set_allow_early_resolve(true);
    TaskOutputTensors left_outputs = rt_submit_aiv_task(FUNC_ADD_SCALAR, left_args);
    simpler::hbg::Tensor left = left_outputs.get_ref(0);

    CoreTaskArgs right_args;
    right_args.add_input(sum);
    right_args.add_output(intermediate);
    right_args.add_scalar(args.scalar(variant == 0 ? 1 : 0));
    // Flagged alongside left so mul qualifies as an early-dispatch candidate on
    // both its producers, which is what puts a sorted multi-entry fanin row and
    // a pair of tracked producers on the device path. left and right are not
    // themselves candidates: the fence dummy is an unflagged producer of both.
    right_args.set_allow_early_resolve(true);
    TaskOutputTensors right_outputs = rt_submit_aiv_task(FUNC_ADD_SCALAR, right_args);
    simpler::hbg::Tensor right = right_outputs.get_ref(0);

    CoreTaskArgs mul_args;
    mul_args.add_input(left, right);
    mul_args.add_output(output);
    rt_submit_aiv_task(FUNC_MUL, mul_args);
}

void submit_layer(const GraphTaskArgs &args, int variant) { rt_submit_graph(&layer, args, variant); }

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return OrchestrationConfig{
        .expected_arg_count = 5,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const simpler::hbg::Tensor &a = args.tensor(0).ref();
    const simpler::hbg::Tensor &b = args.tensor(1).ref();

    const std::array<uint32_t, 1> shape{a.shapes[0]};
    TensorCreateInfo seeded_input_info(shape.data(), static_cast<uint32_t>(shape.size()), DataType::FLOAT32);
    CoreTaskArgs seed_args;
    seed_args.add_input(a);
    seed_args.add_output(seeded_input_info);
    seed_args.add_scalar(0.0F);
    // Flagged so the Graph shells that consume its output qualify as
    // early-dispatch candidates: a shell's release materializes its body ahead
    // of this task's completion, which is the ordinary-task-to-Graph edge.
    seed_args.set_allow_early_resolve(true);
    TaskOutputTensors seed_outputs = rt_submit_aiv_task(FUNC_ADD_SCALAR, seed_args);
    simpler::hbg::Tensor seeded_a = seed_outputs.get_ref(0);

    const std::array<float, 3> left_deltas{1.0F, 3.0F, 5.0F};
    const std::array<float, 3> right_deltas{2.0F, 4.0F, 6.0F};
    const std::array<float, 3> unused_deltas{7.0F, 8.0F, 9.0F};
    const std::array<int, 3> variants{0, 1, 0};
    for (size_t layer = 0; layer < variants.size(); ++layer) {
        GraphTaskArgs layer_args;
        layer_args.add_input(seeded_a, b);
        layer_args.add_output(args.tensor(static_cast<int32_t>(layer) + 2).ref());
        if (variants[layer] == 0) {
            layer_args.add_scalar(left_deltas[layer], right_deltas[layer], unused_deltas[layer]);
        } else {
            layer_args.add_scalar(right_deltas[layer]);
        }
        submit_layer(layer_args, variants[layer]);
    }
}

}  // extern "C"
