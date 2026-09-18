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

#include "orchestration_api.h"

// HBG kernel mode must be able to prove that Host build does not read tensor
// payload bytes. This orchestration only consumes tensor metadata and scalar
// values, so it declares an empty requirements mask.
extern "C" uint64_t pypto_orchestration_requirements_v1() { return 0; }

extern "C" void kernel_eager_orchestration(const ChipTaskArgs &args) {
    CoreTaskArgs task;
    task.add_input(args.tensor(0).ref());
    task.add_output(args.tensor(1).ref());
    task.add_scalar(args.scalar(0));
    rt_submit_aiv_task(0, task);
}

extern "C" void kernel_eager_intermediate(const ChipTaskArgs &args) {
    const auto &input = args.tensor(0).ref();
    TensorCreateInfo intermediate(input.shapes, input.ndims, DataType::FLOAT32);
    CoreTaskArgs first;
    first.add_input(input);
    first.add_output(intermediate);
    first.add_scalar(args.scalar(0));
    auto output = rt_submit_aiv_task(0, first);
    CoreTaskArgs second;
    second.add_input(output.get_ref(0));
    second.add_output(args.tensor(1).ref());
    second.add_scalar(args.scalar(0));
    rt_submit_aiv_task(0, second);
}
