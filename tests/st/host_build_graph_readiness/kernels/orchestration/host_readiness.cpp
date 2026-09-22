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
enum class Mode : uint64_t { Produce, Read, ReadWrite };
}

extern "C" {
__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 5};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const auto &source = args.tensor(0).ref();
    const auto &control = args.tensor(1).ref();
    const auto &output = args.tensor(2).ref();
    const auto mode = static_cast<Mode>(args.scalar<uint64_t>(0));
    union {
        uint64_t bits;
        float value;
    } scalar{};
    if (mode == Mode::Produce) {
        scalar.bits = args.scalar<uint64_t>(1);
    } else {
        const uint32_t index[] = {0};
        scalar.value = get_tensor_data<float>(control, 1, index);
        if (mode == Mode::ReadWrite) {
            scalar.value += 3.0F;
            set_tensor_data<float>(control, 1, index, scalar.value);
        }
    }
    CoreTaskArgs task;
    // ReadWrite consumes both the host-written element and the host-read scalar.
    task.add_input(mode == Mode::ReadWrite ? control : source);
    task.add_output(mode == Mode::Produce ? control : output);
    task.add_scalar(scalar.bits);
    rt_submit_aiv_task(0, task);
}
}
