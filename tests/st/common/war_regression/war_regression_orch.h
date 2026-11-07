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

#pragma once

#include <cstdint>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

namespace war_regression_test {
constexpr int32_t FUNC_WRITER = 0;
constexpr int32_t FUNC_READER = 1;
}  // namespace war_regression_test

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return OrchestrationConfig{.expected_arg_count = 4};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const Tensor &buffer = orch_args.tensor(0).ref();
    const Tensor &outputs = orch_args.tensor(1).ref();
    const int32_t spin_iters = static_cast<int32_t>(orch_args.scalar(0));
    const bool cross_ring = static_cast<bool>(orch_args.scalar(1));
    uint32_t row_shape[] = {1, 16};
    uint32_t row0_offset[] = {0, 0};
    uint32_t row1_offset[] = {1, 0};
    Tensor first_output = outputs.view(row_shape, row0_offset);
    Tensor second_output = outputs.view(row_shape, row1_offset);

    CoreTaskArgs first_writer_args;
    first_writer_args.add_inout(buffer);
    first_writer_args.add_scalar(1);
    rt_submit_aiv_task(war_regression_test::FUNC_WRITER, first_writer_args);

    if (cross_ring) rt_scope_begin();
    CoreTaskArgs first_reader_args;
    first_reader_args.add_input(buffer);
    first_reader_args.add_output(first_output);
    first_reader_args.add_scalar(spin_iters);
    rt_submit_aiv_task(war_regression_test::FUNC_READER, first_reader_args);
    if (cross_ring) rt_scope_end();

    CoreTaskArgs second_writer_args;
    second_writer_args.add_inout(buffer);
    second_writer_args.add_scalar(2);
    rt_submit_aiv_task(war_regression_test::FUNC_WRITER, second_writer_args);

    CoreTaskArgs second_reader_args;
    second_reader_args.add_input(buffer);
    second_reader_args.add_output(second_output);
    second_reader_args.add_scalar(0);
    rt_submit_aiv_task(war_regression_test::FUNC_READER, second_reader_args);
}

}  // extern "C"
