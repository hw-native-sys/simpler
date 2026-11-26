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

namespace {
constexpr int WRITE_CONST = 0, COPY_FIRST = 1, CLOBBER = 2, WRITE_GATE = 3;
}
extern "C" OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) { return {.expected_arg_count = 4}; }
extern "C" void aicpu_orchestration_entry(const ChipTaskArgs &a) {
    const auto &x = a.tensor(0).ref();
    const auto &y = a.tensor(1).ref();
    const auto &gate = a.tensor(2).ref();
    const auto case_id = a.scalar<uint64_t>(0);
    if (case_id != 1 && case_id != 2) {
        rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "unsupported predicate case");
        return;
    }
    // Each writer runs on the Core that owns its output SRAM.
    CoreTaskArgs write_gate;
    write_gate.add_inout(gate);
    write_gate.add_scalar(case_id - 1);
    TaskId dependencies[] = {rt_submit_wse_task(WRITE_GATE, write_gate, {.placement = {{2}}}).task_id()};

    CoreTaskArgs write_x;
    write_x.add_inout(x);
    rt_submit_wse_task(WRITE_CONST, write_x, {.placement = {{0}}});

    CoreTaskPredicate predicate;
    predicate.operand.tensor = &gate;
    predicate.operand.ndims = 1;
    predicate.operand.indices[0] = 0;
    predicate.op = PredicateOp::GT;
    predicate.target = 0;
    CoreTaskArgs clobber;
    clobber.add_inout(x);
    clobber.set_dependencies(dependencies, 1);
    clobber.set_predicate(predicate);
    rt_submit_wse_task(CLOBBER, clobber, {.placement = {{0}}});

    CoreTaskArgs copy;
    copy.add_input(x);
    copy.add_inout(y);
    rt_submit_wse_task(COPY_FIRST, copy, {.placement = {{1}}});
}
