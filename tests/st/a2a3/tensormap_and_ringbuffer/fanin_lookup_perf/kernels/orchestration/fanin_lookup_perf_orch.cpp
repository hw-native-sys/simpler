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
 * Explicit 64x64 fanin DAG validation scene.
 *
 * Args layout:
 *   tensor[0]: producer outputs, split into disjoint producer slices
 *   tensor[1]: consumer outputs, split into disjoint consumer slices
 *   scalar[0]: producer_count
 *   scalar[1]: consumer_count
 *   scalar[2]: use_real_kernels
 *
 * The scene submits producer_count independent producers, then consumer_count
 * independent consumers where every consumer explicitly depends on every
 * producer. Real-kernel mode writes disjoint tensor slices so tensormap
 * auto-deps do not add producer chains or consumer chains.
 *
 * When use_real_kernels is false, the same dependency shape is submitted with
 * dummy tasks to isolate orchestrator fanin lookup cost.
 */

#include <cstdint>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

static constexpr int32_t MAX_PRODUCERS = 64;
static constexpr int32_t MAX_CONSUMERS = 64;
static constexpr uint32_t SLOT_ELEMS = 16;

#define FUNC_WRITE_CONST 0

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 5,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    const Tensor &producer_outputs = orch_args.tensor(0).ref();
    const Tensor &consumer_outputs = orch_args.tensor(1).ref();
    int32_t producer_count = static_cast<int32_t>(orch_args.scalar(0));
    int32_t consumer_count = static_cast<int32_t>(orch_args.scalar(1));
    bool use_real_kernels = orch_args.scalar(2) != 0;
    if (producer_count < 1 || producer_count > MAX_PRODUCERS || consumer_count < 1 || consumer_count > MAX_CONSUMERS) {
        rt_report_fatal(
            PTO2_ERROR_INVALID_ARGS,
            "producer_count=%d consumer_count=%d exceed supported range producers=[1, %d] consumers=[1, %d]",
            producer_count, consumer_count, MAX_PRODUCERS, MAX_CONSUMERS
        );
        return;
    }

    PTO2TaskId producer_ids[MAX_PRODUCERS];
    uint32_t slot_shape[1] = {SLOT_ELEMS};
    for (int32_t i = 0; i < producer_count; i++) {
        L0TaskArgs args;
        if (use_real_kernels) {
            uint32_t offset[1] = {static_cast<uint32_t>(i) * SLOT_ELEMS};
            Tensor producer_out = producer_outputs.view(slot_shape, offset);
            args.add_inout(producer_out);
            producer_ids[i] = rt_submit_aic_task(FUNC_WRITE_CONST, args).task_id();
        } else {
            producer_ids[i] = rt_submit_dummy_task(args).task_id();
        }
    }

    for (int32_t c = 0; c < consumer_count; c++) {
        L0TaskArgs args;
        args.set_dependencies(producer_ids, static_cast<uint32_t>(producer_count));
        if (use_real_kernels) {
            uint32_t offset[1] = {static_cast<uint32_t>(c) * SLOT_ELEMS};
            Tensor consumer_out = consumer_outputs.view(slot_shape, offset);
            args.add_inout(consumer_out);
            rt_submit_aic_task(FUNC_WRITE_CONST, args);
        } else {
            rt_submit_dummy_task(args);
        }
    }
}

}  // extern "C"
