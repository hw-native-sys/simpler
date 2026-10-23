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
#include <vector>

#include "cla_plan.h"
#include "orchestration_api.h"

extern "C" __attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 3};
}

extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const auto &host_plan = orch_args.tensor(2).ref();
    if (host_plan.ndims != 2 || host_plan.shapes[1] != cla_contract::kPlanWidth || host_plan.dtype != DataType::INT64) {
        rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "CLA probe: invalid plan shape");
        return;
    }
    const auto *rows = reinterpret_cast<const int64_t *>(host_plan.buffer.addr) + host_plan.start_offset;
    const uint32_t count = host_plan.shapes[0];
    std::vector<TaskId> handles;
    handles.reserve(count);
    // Dependency ownership is explicit; the audit's read of the full table adds no artificial data edge.
    auto records = orch_args.tensor(0).ref();
    records.manual_dep = true;
    for (uint32_t task = 0; task < count; ++task) {
        const auto *row = rows + task * cla_contract::kPlanWidth;
        const int64_t blocks = row[static_cast<uint32_t>(cla_contract::Field::Blocks)];
        const int64_t raw_mask = row[static_cast<uint32_t>(cla_contract::Field::Mask)];
        const int64_t raw_dependencies = row[static_cast<uint32_t>(cla_contract::Field::DependencyCount)];
        const int64_t checks = row[static_cast<uint32_t>(cla_contract::Field::CheckCount)];
        const int64_t base = row[static_cast<uint32_t>(cla_contract::Field::Base)];
        const int64_t record_capacity = records.shapes[0] / 16;
        if (blocks <= 0 || blocks > INT16_MAX || raw_mask < 0 || raw_mask > 7 ||
            blocks * __builtin_popcount(static_cast<uint32_t>(raw_mask)) > INT16_MAX || raw_dependencies < 0 ||
            raw_dependencies > cla_contract::kMaxDependencies || checks < 0 ||
            checks > cla_contract::kMaxDependencies || base < 0 || base > record_capacity ||
            blocks * 3 + 1 > record_capacity - base) {
            rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "CLA probe: invalid plan bounds");
            return;
        }
        const auto n = static_cast<int16_t>(blocks);
        const auto mask = static_cast<uint32_t>(raw_mask);
        const auto dependencies = static_cast<uint32_t>(raw_dependencies);
        for (int64_t i = 0; i < checks; ++i) {
            const int64_t predecessor = row[cla_contract::kCheckOffset + i];
            if (predecessor < 0 || predecessor >= task ||
                rows[predecessor * cla_contract::kPlanWidth + static_cast<uint32_t>(cla_contract::Field::Mask)] == 0) {
                rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "CLA probe: invalid audit predecessor");
                return;
            }
        }
        std::vector<TaskId> fanins;
        fanins.reserve(dependencies);
        for (uint32_t i = 0; i < dependencies; ++i) {
            const int64_t predecessor = row[cla_contract::kDependencyOffset + i];
            if (predecessor < 0 || predecessor >= task) {
                rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "CLA probe: invalid predecessor");
                return;
            }
            fanins.push_back(handles[predecessor]);
        }
        CoreTaskArgs args;
        args.set_dependencies(fanins.data(), static_cast<uint32_t>(fanins.size()));
        if (mask == 0) {
            handles.push_back(rt_submit_dummy_task(args).task_id());
            continue;
        }
        const uint32_t shape[1] = {(static_cast<uint32_t>(n) * 3 + 1) * 16};
        const uint32_t offset[1] = {static_cast<uint32_t>(row[static_cast<uint32_t>(cla_contract::Field::Base)]) * 16};
        auto state = records.view(shape, offset, true);
        args.add_inout(state);
        args.add_input(records, orch_args.tensor(1).ref());
        args.add_scalar(static_cast<int64_t>(task));
        args.launch_spec.set_block_num(n);
        MixedKernels kernels;
        kernels.aic_kernel_id = (mask & 1U) ? 0 : INVALID_KERNEL_ID;
        kernels.aiv0_kernel_id = (mask & 2U) ? 1 : INVALID_KERNEL_ID;
        kernels.aiv1_kernel_id = (mask & 4U) ? 2 : INVALID_KERNEL_ID;
        handles.push_back(rt_submit_task(kernels, args).task_id());
    }
}
