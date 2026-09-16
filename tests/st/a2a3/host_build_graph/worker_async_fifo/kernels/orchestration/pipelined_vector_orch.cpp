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

#include <cstdint>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {

constexpr uint64_t kDelayedAdd = 0;
constexpr uint64_t kAddScalar = 1;
constexpr int kChainLength = 512;

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return OrchestrationConfig{.expected_arg_count = 4};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const simpler::hbg::Tensor &a = args.tensor(0).ref();
    const simpler::hbg::Tensor &b = args.tensor(1).ref();
    const simpler::hbg::Tensor &out = args.tensor(2).ref();
    const uint64_t spin_iters = args.scalar<uint64_t>(0);
    uint32_t shape[1] = {a.shapes[0]};
    TensorCreateInfo temporary(shape, 1, DataType::FLOAT32);

    CoreTaskArgs add_args;
    add_args.add_input(a);
    add_args.add_input(b);
    add_args.add_output(temporary);
    add_args.add_scalar(spin_iters);
    TaskOutputTensors add_outputs = rt_submit_aiv_task(kDelayedAdd, add_args);
    simpler::hbg::Tensor current = add_outputs.get_ref(0);

    union {
        float f32;
        uint64_t u64;
    } scalar{};
    scalar.f32 = 1.0F;
    for (int i = 0; i < kChainLength; ++i) {
        CoreTaskArgs step_args;
        step_args.add_input(current);
        if (i + 1 == kChainLength) {
            step_args.add_output(out);
            step_args.add_scalar(scalar.u64);
            rt_submit_aiv_task(kAddScalar, step_args);
        } else {
            step_args.add_output(temporary);
            step_args.add_scalar(scalar.u64);
            TaskOutputTensors step_outputs = rt_submit_aiv_task(kAddScalar, step_args);
            current = step_outputs.get_ref(0);
        }
    }
}

}  // extern "C"
