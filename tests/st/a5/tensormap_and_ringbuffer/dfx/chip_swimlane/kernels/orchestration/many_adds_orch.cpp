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
 * Pre-seal short single-block adds (orch-side wait → idle exact drain → `release`
 * phase), then a burst of long multi-block SPMD tasks submitted without waiting
 * so orchestration can seal while they still run and their deferred releases
 * are elided.
 */
#include <stdint.h>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_ADD 0
#define FUNC_SPMD_SLOW 1

constexpr int32_t kShortAddTasks = 16;
constexpr int32_t kLongSpmdTasks = 4;
constexpr int16_t kSpmdBlocks = 8;
// Long enough to outlast orch exit+seal on a5sim; far shorter than the 1s blocker.
constexpr int64_t kSpmdSpinIters = 200000;
constexpr int32_t kFloatsPerCacheLine = 16;

static TaskId submit_short_add(
    const simpler::tmr::Tensor &ext_a, const simpler::tmr::Tensor &ext_b, const TensorCreateInfo &inter_ci,
    const simpler::tmr::Tensor *ext_out
) {
    CoreTaskArgs params;
    params.add_input(ext_a);
    params.add_input(ext_b);
    if (ext_out != nullptr) {
        params.add_output(*ext_out);
    } else {
        params.add_output(inter_ci);
    }
    return rt_submit_aiv_task(FUNC_ADD, params).task_id();
}

// Wait until every listed producer has completed, then force one more full
// scheduler loop with only dummy work so an idle drain (exact release) runs
// before orchestration seals.
static void wait_preseal_idle_release(const TaskId *producers, int32_t producer_count) {
    CoreTaskArgs completion_args;
    uint32_t fence_shape[1] = {1};
    TensorCreateInfo fence_info(fence_shape, 1, DataType::INT32);
    completion_args.add_output(fence_info);
    completion_args.set_dependencies(producers, static_cast<uint32_t>(producer_count));
    TaskOutputTensors completion_out = rt_submit_dummy_task(completion_args);
    uint32_t index[1] = {0};
    (void)get_tensor_data<int32_t>(completion_out.get_ref(0), 1, index);

    CoreTaskArgs first_args;
    TaskId first = rt_submit_dummy_task(first_args).task_id();

    CoreTaskArgs second_args;
    second_args.add_output(fence_info);
    TaskId deps[1] = {first};
    second_args.set_dependencies(deps, 1);
    TaskOutputTensors loop_out = rt_submit_dummy_task(second_args);
    (void)get_tensor_data<int32_t>(loop_out.get_ref(0), 1, index);
}

static void submit_long_spmd_burst(const simpler::tmr::Tensor &spmd_ws) {
    for (int32_t i = 0; i < kLongSpmdTasks; ++i) {
        CoreTaskArgs args;
        args.add_inout(spmd_ws);
        args.add_scalar(static_cast<int64_t>(i) * kSpmdBlocks);  // base_cl
        args.add_scalar(kSpmdSpinIters);
        args.launch_spec.set_block_num(kSpmdBlocks);
        (void)rt_submit_aiv_task(FUNC_SPMD_SLOW, args);
    }
}

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return OrchestrationConfig{
        .expected_arg_count = 3,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const simpler::tmr::Tensor &ext_a = orch_args.tensor(0).ref();
    const simpler::tmr::Tensor &ext_b = orch_args.tensor(1).ref();
    const simpler::tmr::Tensor &ext_out = orch_args.tensor(2).ref();

    uint32_t size = ext_a.shapes[0];
    uint32_t inter_shapes[1] = {size};
    TensorCreateInfo inter_ci(inter_shapes, 1, DataType::FLOAT32);

    // Phase 1: short single-block adds + orch-side wait → guaranteed pre-seal release.
    TaskId short_ids[kShortAddTasks];
    for (int32_t i = 0; i < kShortAddTasks; ++i) {
        const bool last_short = (i + 1 == kShortAddTasks);
        short_ids[i] = submit_short_add(ext_a, ext_b, inter_ci, last_short ? &ext_out : nullptr);
    }
    wait_preseal_idle_release(short_ids, kShortAddTasks);

    // Phase 2: long multi-block SPMD burst with no wait — orch returns and seals
    // while these still run; their deferred releases are elided.
    uint32_t spmd_elems =
        static_cast<uint32_t>(kLongSpmdTasks) * static_cast<uint32_t>(kSpmdBlocks) * kFloatsPerCacheLine;
    uint32_t spmd_shape[1] = {spmd_elems};
    TensorCreateInfo spmd_ci(spmd_shape, 1, DataType::FLOAT32);
    TaskOutputTensors spmd_buf = alloc_tensors(spmd_ci);
    submit_long_spmd_burst(spmd_buf.get_ref(0));
}

}  // extern "C"
