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

/**
 * scalar_access_deps orchestration scenes.
 *
 * The scene selector arrives in scalar slot 0; args are [X, Y, W].
 *
 *   case=1 (read in a manual scope): the manual-scope producer P writes
 *     X[0] = 5.0. No TensorMap entry exists for X, so get_tensor_data can only
 *     see the write through the declared dependency. Y[0] receives the read
 *     value; the host expects 5.0.
 *
 *   case=2 (WAR in a manual scope): P writes X[0] = 5.0 and consumer C,
 *     wired to P by set_dependencies, copies X[0] to W[0]. set_tensor_data
 *     overwrites X[0] with 9.0 declaring P as its dependency, so the write
 *     waits for P to complete and for C to finish reading. The host expects
 *     X[0] = 9.0 and W[0] = 5.0.
 *
 *   case=3: declare a task id that was never submitted.
 *   case=4: pass a null deps array with dep_count = 1.
 *     Both must fail the run with INVALID_ARGS before touching memory.
 */

#include <cstdint>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_WRITE_FIVE 0
#define FUNC_COPY_FIRST 1

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return OrchestrationConfig{
        .expected_arg_count = 4,  // 3 tensors + 1 case scalar
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const simpler::tmr::Tensor &ext_X = orch_args.tensor(0).ref();
    const simpler::tmr::Tensor &ext_Y = orch_args.tensor(1).ref();
    const simpler::tmr::Tensor &ext_W = orch_args.tensor(2).ref();

    uint32_t index[1] = {0};
    uint64_t case_id = orch_args.scalar(0);

    if (case_id == 1) {
        rt_scope_begin(ScopeMode::MANUAL);
        CoreTaskArgs producer;
        producer.add_inout(ext_X);
        TaskId dep = rt_submit_aiv_task(FUNC_WRITE_FIVE, producer).task_id();
        float value = get_tensor_data<float>(ext_X, 1, index, &dep, 1);
        rt_scope_end();
        if (rt_is_fatal()) return;
        set_tensor_data(ext_Y, 1, index, value);
    } else if (case_id == 2) {
        rt_scope_begin(ScopeMode::MANUAL);
        CoreTaskArgs producer;
        producer.add_inout(ext_X);
        TaskId dep = rt_submit_aiv_task(FUNC_WRITE_FIVE, producer).task_id();

        CoreTaskArgs reader;
        reader.set_dependencies(&dep, 1);
        reader.add_input(ext_X);
        reader.add_inout(ext_W);
        rt_submit_aiv_task(FUNC_COPY_FIRST, reader);
        rt_scope_end();

        set_tensor_data(ext_X, 1, index, 9.0f, &dep, 1);
    } else if (case_id == 3) {
        // Ring 0 has no submitted task yet, so local id 7 names nothing.
        TaskId never_submitted = TaskId::make(0, 7);
        get_tensor_data<float>(ext_X, 1, index, &never_submitted, 1);
    } else if (case_id == 4) {
        get_tensor_data<float>(ext_X, 1, index, nullptr, 1);
    } else {
        rt_report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, "unsupported case_id=%llu", static_cast<unsigned long long>(case_id)
        );
    }
}

}  // extern "C"
