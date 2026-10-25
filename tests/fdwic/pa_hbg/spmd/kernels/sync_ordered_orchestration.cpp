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

#include "cla_submit.h"

extern "C" __attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 5};
}

extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const auto &plan = args.tensor(2).ref();
    const auto &flags = args.tensor(3).ref();
    if (plan.ndims != 2 || flags.ndims != 1 || flags.shapes[0] != plan.shapes[0] || flags.dtype != DataType::INT64) {
        rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "CLA probe: invalid sync flags");
        return;
    }
    submit_cla_plan(
        args, reinterpret_cast<const int64_t *>(flags.buffer.addr) + flags.start_offset, args.scalar<int64_t>(0) != 0
    );
}
