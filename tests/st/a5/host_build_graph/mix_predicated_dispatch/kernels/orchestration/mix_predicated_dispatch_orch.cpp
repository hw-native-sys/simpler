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
 * mix_predicated_dispatch orchestration: one dispatch predicate governs all
 * subtasks of one Mix task.
 *
 *   gate_producer (WRITE_GATE)  writes gate[0] = gate_value (INT32)
 *   mix_task      (MixedKernels per mask) carries set_predicate(gate[0] > 0)
 *                 and depends on gate_producer. Each active subtask writes its
 *                 own sentinel slot: AIC -> s_aic[0]=11, AIV0 -> s_aiv0[0]=22,
 *                 AIV1 -> s_aiv1[0]=33.
 *
 * A Mix task's subtasks must execute or skip together. Because each subtask
 * writes a distinct slot, a split decision leaves a state that equals neither
 * "all skipped" nor "all executed", which the exact-match golden rejects.
 *
 * Args layout: [s_aic, s_aiv0, s_aiv1, gate] + {gate_value, mask} scalars.
 */

#include <cstdint>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_MIX_AIC 0
#define FUNC_MIX_AIV0 1
#define FUNC_MIX_AIV1 2
#define FUNC_WRITE_GATE 3

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return OrchestrationConfig{
        .expected_arg_count = 6,  // 4 tensors + 2 scalars
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const simpler::hbg::Tensor &s_aic = orch_args.tensor(0).ref();
    const simpler::hbg::Tensor &s_aiv0 = orch_args.tensor(1).ref();
    const simpler::hbg::Tensor &s_aiv1 = orch_args.tensor(2).ref();
    const simpler::hbg::Tensor &ext_gate = orch_args.tensor(3).ref();

    const int64_t gate_value = orch_args.scalar<int64_t>(0);
    const int64_t mask_value = orch_args.scalar<int64_t>(1);
    if (gate_value != 0 && gate_value != 1) {
        rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "unsupported gate value %ld", gate_value);
        return;
    }
    // A single-lane Mix needs at least two active subtasks; mask 1/2/4 would be
    // an ordinary task and would not exercise the Mix predicate contract.
    if (mask_value < 3 || mask_value > 7) {
        rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "unsupported mix mask %ld", mask_value);
        return;
    }
    const uint8_t mask = static_cast<uint8_t>(mask_value);

    TaskId gate_tid;
    {
        CoreTaskArgs args;
        args.add_inout(ext_gate);
        args.add_scalar(gate_value);
        gate_tid = rt_submit_aic_task(FUNC_WRITE_GATE, args).task_id();
    }

    {
        CoreTaskArgs args;
        args.add_inout(s_aic);
        args.add_inout(s_aiv0);
        args.add_inout(s_aiv1);
        TaskId deps[] = {gate_tid};
        args.set_dependencies(deps, 1);

        CoreTaskPredicate pred;
        pred.operand.tensor = &ext_gate;
        pred.operand.ndims = 1;
        pred.operand.indices[0] = 0;
        pred.op = PredicateOp::GT;
        pred.target = 0;
        args.set_predicate(pred);

        MixedKernels kernels;
        kernels.aic_kernel_id = (mask & 1U) != 0 ? FUNC_MIX_AIC : INVALID_KERNEL_ID;
        kernels.aiv0_kernel_id = (mask & 2U) != 0 ? FUNC_MIX_AIV0 : INVALID_KERNEL_ID;
        kernels.aiv1_kernel_id = (mask & 4U) != 0 ? FUNC_MIX_AIV1 : INVALID_KERNEL_ID;
        rt_submit_task(kernels, args);
    }
}

}  // extern "C"
