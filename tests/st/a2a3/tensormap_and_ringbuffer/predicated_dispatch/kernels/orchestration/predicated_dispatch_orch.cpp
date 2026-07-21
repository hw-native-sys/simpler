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
 * predicated_dispatch orchestration: delayed-evaluation dispatch predicate.
 *
 * The predicate value is produced by a prior task and read by the scheduler at
 * the dispatch point — never in orchestration — so the orchestrator does not
 * stall on it:
 *
 *   gate_producer (WRITE_GATE)  writes gate[0] = gate_value   (INT32)
 *   x_producer    (WRITE_CONST) writes X[0]   = 42.0
 *   clobber       (CLOBBER)     would write X[0] = 999.0, but carries
 *                               set_predicate(gate[0] > 0) and depends on
 *                               gate_producer. The scheduler reads gate[0] when
 *                               the clobber becomes ready (gate already written)
 *                               and dispatches only if gate[0] > 0; otherwise the
 *                               task is retired inline without dispatch.
 *   consumer      (COPY_FIRST)  copies X[0] -> Y[0]
 *
 *   case=1: gate_value = 0 -> predicate FALSE -> clobber not dispatched ->
 *           X stays 42.0 -> Y = 42.0. Proves non-dispatch + consumer still unlocks.
 *   case=2: gate_value = 1 -> predicate TRUE  -> clobber dispatched ->
 *           X = 999.0 -> Y = 999.0. Proves the dispatch path is taken.
 *
 * Args layout: [X, Y, gate] + case scalar.
 */

#include <cstdint>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_WRITE_CONST 0
#define FUNC_COPY_FIRST 1
#define FUNC_CLOBBER 2
#define FUNC_WRITE_GATE 3

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 4,  // 3 tensors + 1 case scalar
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    const Tensor &ext_X = orch_args.tensor(0).ref();
    const Tensor &ext_Y = orch_args.tensor(1).ref();
    const Tensor &ext_gate = orch_args.tensor(2).ref();

    uint64_t case_id = orch_args.scalar(0);
    if (case_id != 1 && case_id != 2) {
        rt_report_fatal(PTO2_ERROR_INVALID_ARGS, "unsupported case_id=%llu", static_cast<unsigned long long>(case_id));
        return;
    }
    // case 1 => gate 0 (predicate FALSE, skip); case 2 => gate 1 (predicate TRUE, dispatch).
    int64_t gate_value = (case_id == 1) ? 0 : 1;

    // gate producer: gate[0] = gate_value
    PTO2TaskId gate_tid;
    {
        L0TaskArgs args;
        args.add_inout(ext_gate);
        args.add_scalar(gate_value);
        gate_tid = rt_submit_aic_task(FUNC_WRITE_GATE, args).task_id();
    }

    // x producer: X[0] = 42.0
    {
        L0TaskArgs args;
        args.add_inout(ext_X);
        rt_submit_aic_task(FUNC_WRITE_CONST, args);
    }

    // predicated clobber: would write X[0] = 999.0 if dispatched. Depends on the
    // gate producer so gate[0] is written by the time this task is ready; the
    // scheduler reads gate[0] at the dispatch point and dispatches only if > 0.
    {
        L0TaskArgs args;
        args.add_inout(ext_X);
        PTO2TaskId deps[] = {gate_tid};
        args.set_dependencies(deps, 1);
        // predicate: gate[0] > 0  (operand op target), built level by level.
        L0TaskPredicate pred;
        pred.operand.tensor = &ext_gate;
        pred.operand.ndims = 1;
        pred.operand.indices[0] = 0;
        pred.op = PredicateOp::GT;
        pred.target = 0;
        args.set_predicate(pred);
        rt_submit_aic_task(FUNC_CLOBBER, args);
    }

    // consumer: Y[0] = X[0]
    {
        L0TaskArgs args;
        args.add_input(ext_X);
        args.add_inout(ext_Y);
        rt_submit_aic_task(FUNC_COPY_FIRST, args);
    }
}

}  // extern "C"
