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
 * Issue #1548: early sync-start cohort that fits one scheduler.
 *
 * Mode 0 uses a one-block flagged AIC producer. Its dependent AIV
 * require_sync_start consumer fits one scheduler's local idle AIV capacity,
 * including a four-block cohort when two peer schedulers are active.
 *
 * Mode 1 fills every AIV running slot with a blocker. Scheduler-loop fences
 * first retire every blocker dispatch ACK, then give the speculative consumer
 * one full loop to stage while the one-second blockers remain active.
 *
 * The slow producer and the consumer write disjoint cache lines:
 *   P: AIC block_num=1, base_cl=0, allow_early_resolve=true
 *   C: AIV block_num=consumer_blocks, base_cl=1, require_sync_start=true, dep=[P]
 */

#include <stddef.h>
#include <stdint.h>

#include "aicpu/cache_maintenance.h"
#include "arg_with_deps.h"      // NOLINT(build/include_subdir)
#include "orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_SLOW_PRODUCER_AIC 0
#define FUNC_SYNC_CONSUMER_AIV 1
#define FUNC_SPIN_BLOCKER_AIV 2

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return OrchestrationConfig{
        .expected_arg_count = 3,
    };
}

static constexpr int16_t PRODUCER_BLOCKS = 1;
static constexpr int32_t FLOATS_PER_CACHE_LINE = 16;
static constexpr int32_t BLOCKER_STATUS_BASE_CL = 9;
static constexpr int32_t BLOCKER_STATUS_CAPACITY = 72;
// Matches the existing early-dispatch regression's delay. It is long enough
// for the scheduler to pop the consumer before the producer completes.
static constexpr int64_t PRODUCER_SPIN_ITERS = 10000000;

static bool wait_for_blockers_started(const simpler::tmr::Tensor &out, int32_t blocker_count) {
    volatile float *data = out.data_as<float>() + out.start_offset;
    volatile float *statuses = data + BLOCKER_STATUS_BASE_CL * FLOATS_PER_CACHE_LINE;
    while (!rt_is_fatal()) {
        cache_invalidate_range(
            const_cast<float *>(statuses), static_cast<size_t>(blocker_count) * FLOATS_PER_CACHE_LINE * sizeof(float)
        );
        bool all_started = true;
        for (int32_t block_idx = 0; block_idx < blocker_count; block_idx++) {
            float status = statuses[block_idx * FLOATS_PER_CACHE_LINE];
            if (status != 1.0F) {
                all_started = false;
                break;
            }
        }
        if (all_started) break;
        SPIN_WAIT_HINT();
    }
    return !rt_is_fatal();
}

static bool wait_for_scheduler_loop_fence() {
    CoreTaskArgs first_args;
    TaskId first = rt_submit_dummy_task(first_args).task_id();

    uint32_t fence_shape[1] = {1};
    TensorCreateInfo fence_info(fence_shape, 1, DataType::INT32);
    CoreTaskArgs second_args;
    second_args.add_output(fence_info);
    TaskId deps[1] = {first};
    second_args.set_dependencies(deps, 1);
    TaskOutputTensors outputs = rt_submit_dummy_task(second_args);

    // D2 cannot be popped in the batch that completes D1. Observing D2's
    // output therefore proves that one complete scheduler loop elapsed after
    // this fence was submitted.
    const simpler::tmr::Tensor &fence = outputs.get_ref(0);
    uint32_t index[1] = {0};
    (void)get_tensor_data<int32_t>(fence, 1, index);
    return !rt_is_fatal();
}

static TaskId submit_producer(const simpler::tmr::Tensor &out) {
    CoreTaskArgs args;
    args.add_inout(out);
    args.add_scalar(0);  // base cache line
    args.add_scalar(PRODUCER_SPIN_ITERS);
    args.launch_spec.set_block_num(PRODUCER_BLOCKS);
    args.set_allow_early_resolve(true);
    return rt_submit_aic_task(FUNC_SLOW_PRODUCER_AIC, args).task_id();
}

static TaskId submit_aiv_blocker(const simpler::tmr::Tensor &out, int32_t blocker_count) {
    CoreTaskArgs args;
    args.add_inout(out);
    args.add_scalar(BLOCKER_STATUS_BASE_CL);
    args.add_scalar(blocker_count);
    args.launch_spec.set_block_num(static_cast<int16_t>(blocker_count));
    args.set_allow_early_resolve(true);
    return rt_submit_aiv_task(FUNC_SPIN_BLOCKER_AIV, args).task_id();
}

static void submit_consumer(const simpler::tmr::Tensor &out, TaskId producer, int16_t consumer_blocks) {
    CoreTaskArgsWithDeps<1> args;
    args.add_inout(out);
    args.add_scalar(PRODUCER_BLOCKS);  // base cache line
    args.launch_spec.set_block_num(consumer_blocks);
    args.launch_spec.set_require_sync_start(true);
    args.add_dep(producer);
    rt_submit_aiv_task(FUNC_SYNC_CONSUMER_AIV, args);
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const simpler::tmr::Tensor &output = orch_args.tensor(0).ref();
    const bool use_pending = orch_args.scalar(0) != 0;
    const int16_t consumer_blocks = orch_args.scalar<int16_t>(1);
    const int32_t blocker_count = static_cast<int32_t>(rt_available_aiv_count());

    if (use_pending && (blocker_count <= 0 || blocker_count > BLOCKER_STATUS_CAPACITY || consumer_blocks <= 0 ||
                        consumer_blocks > blocker_count)) {
        rt_report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, "invalid pending geometry: blockers=%d consumer=%d", blocker_count,
            consumer_blocks
        );
        return;
    }

    rt_scope_begin(ScopeMode::MANUAL);
    TaskId producer = use_pending ? submit_aiv_blocker(output, blocker_count) : submit_producer(output);
    if (use_pending && (!wait_for_blockers_started(output, blocker_count) || !wait_for_scheduler_loop_fence())) return;
    submit_consumer(output, producer, consumer_blocks);
    if (use_pending && !wait_for_scheduler_loop_fence()) return;
    rt_scope_end();
}

}  // extern "C"
