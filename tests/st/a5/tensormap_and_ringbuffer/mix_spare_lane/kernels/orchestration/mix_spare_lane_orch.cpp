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
#include "arg_with_deps.h"

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 8};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const auto &producer = orch_args.tensor(0).ref();
    const auto &cohort = orch_args.tensor(1).ref();
    const auto &vector = orch_args.tensor(2).ref();
    const auto &trigger = orch_args.tensor(3).ref();
    const auto &layout = orch_args.tensor(4).ref();
    int32_t mask = static_cast<int32_t>(orch_args.scalar(0));
    bool sync = orch_args.scalar(1) != 0;
    int64_t spin = orch_args.scalar(2);
    int32_t clusters = rt_available_cluster_count();
    int32_t producer_blocks = sync ? clusters - 1 : clusters * 4;

    MixedKernels kernels;
    kernels.aic_kernel_id = 0;
    if (mask & 2) kernels.aiv0_kernel_id = 1;
    if (mask & 4) kernels.aiv1_kernel_id = 1;

    if (sync) {
        CoreTaskArgs producer_args;
        producer_args.add_inout(producer);
        producer_args.add_scalar(0);
        producer_args.add_scalar(spin);
        producer_args.launch_spec.set_block_num(static_cast<int16_t>(producer_blocks));
        rt_submit_task(kernels, producer_args);
    } else {
        // Keep multiple ready entries present while owners inspect their batches.
        // A single SPMD entry can be temporarily absent during pop/claim/requeue.
        uint32_t shape[] = {3 * 16};
        for (int32_t block = 0; block < producer_blocks; ++block) {
            uint32_t offset[] = {static_cast<uint32_t>(block) * 3 * 16};
            simpler::tmr::Tensor block_output = producer.view(shape, offset);
            CoreTaskArgs producer_args;
            producer_args.add_inout(block_output);
            producer_args.add_scalar(0);
            producer_args.add_scalar(spin);
            rt_submit_task(kernels, producer_args);
        }
    }

    TaskId trigger_id = TaskId::invalid();
    if (sync) {
        // Leave one C available for the trigger. Its completion makes the
        // whole-device cohort ready while the other C/V pairs are occupied.
        CoreTaskArgs trigger_args;
        trigger_args.add_inout(trigger);
        trigger_args.add_scalar(0);
        trigger_args.add_scalar(0);
        trigger_id = rt_submit_aic_task(0, trigger_args).task_id();

        CoreTaskArgsWithDeps<1> cohort_args;
        cohort_args.add_inout(cohort);
        cohort_args.add_scalar(0);
        cohort_args.add_scalar(0);
        cohort_args.add_dep(trigger_id);
        cohort_args.launch_spec.set_block_num(static_cast<int16_t>(clusters));
        cohort_args.launch_spec.set_require_sync_start(true);
        rt_submit_task(kernels, cohort_args);
    }

    // Publish independent work after the initial MIX admission. A zero-delay
    // publication can race an in-flight MIX pop and never exercise residual
    // priority. DFX must still verify that MIX is running at vector release.
    volatile int32_t release_tick = 0;
    for (int32_t i = 0; i < 50000; ++i) {
        ++release_tick;
    }
    (void)release_tick;

    CoreTaskArgsWithDeps<1> vector_args;
    vector_args.add_inout(vector);
    vector_args.add_scalar(0);
    // Keep the independent V on the critical path so lost overlap is observable.
    vector_args.add_scalar(spin * 4);
    vector_args.launch_spec.set_block_num(static_cast<int16_t>(clusters));
    if (sync) vector_args.add_dep(trigger_id);
    rt_submit_aiv_task(2, vector_args);

    int32_t values[] = {clusters, producer_blocks, sync ? clusters : 0, clusters};
    for (uint32_t i = 0; i < 4; ++i) {
        uint32_t index[] = {i};
        set_tensor_data<int32_t>(layout, 1, index, values[i]);
    }
}
}  // extern "C"
