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
__attribute__((visibility("default"))) OrchestrationConfig framework_scope_statistics_config(const ChipTaskArgs &args) {
    (void)args;
    return {.expected_arg_count = 3};
}

__attribute__((visibility("default"))) void framework_scope_statistics(const ChipTaskArgs &args) {
    SIMPLER_SCOPE() {
        const auto &a = args.tensor(0).ref();
        TensorCreateInfo intermediate(a.shapes, a.ndims, DataType::FLOAT32);
        CoreTaskArgs first;
        first.add_input(a);
        first.add_input(args.tensor(1).ref());
        first.add_output(intermediate);
        auto temporary = rt_submit_aiv_task(0, first);
        SIMPLER_SCOPE() {
            CoreTaskArgs second;
            second.add_input(temporary.get_ref(0));
            second.add_input(a);
            second.add_output(args.tensor(2).ref());
            rt_submit_aiv_task(0, second);
        }
    }
}
}
