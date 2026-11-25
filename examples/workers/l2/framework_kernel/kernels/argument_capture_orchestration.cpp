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
#include "orchestration_api.h"  // NOLINT(build/include_subdir)

extern "C" {
__attribute__((visibility("default"))) OrchestrationConfig framework_argument_capture_config(const ChipTaskArgs &args) {
    (void)args;
    return {.expected_arg_count = 3};
}

__attribute__((visibility("default"))) void framework_argument_capture(const ChipTaskArgs &args) {
    const auto &a = args.tensor(0).ref();
    const auto &b = args.tensor(1).ref();
    const auto &out = args.tensor(2).ref();
    TensorCreateInfo intermediate(out.shapes, out.ndims, DataType::FLOAT32);
    CoreTaskArgs first;
    first.add_input(a);
    first.add_input(b);
    first.add_output(intermediate);
    first.dump(intermediate);
    auto produced = rt_submit_aiv_task(0, first);
    const auto &mid = produced.get_ref(0);
    CoreTaskArgs second;
    second.add_input(mid);
    second.add_input(b);
    second.add_output(out);
    second.dump(mid, out);
    rt_submit_aiv_task(0, second);
}
}
