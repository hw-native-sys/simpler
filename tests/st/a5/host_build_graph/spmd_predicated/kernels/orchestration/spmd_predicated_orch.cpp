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
 * A dispatch predicate on a task of many blocks.
 *
 * A predicate is read once, at the point the task is dispatched, and an SPMD
 * task is dispatched once per block by whichever cores claim them. The two
 * features have never met in a graph here, and this is the shape that puts
 * them together.
 *
 * Args layout: [gate, out] + {block_num, gate_value}.
 */

#include <cstdint>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {
constexpr int32_t kWriteGateFuncId = 0;
constexpr int32_t kStampBlockFuncId = 1;
}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return OrchestrationConfig{.expected_arg_count = 4};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const simpler::hbg::Tensor &gate = args.tensor(0).ref();
    const simpler::hbg::Tensor &out = args.tensor(1).ref();
    const int64_t block_num = args.scalar<int64_t>(0);
    const int64_t gate_value = args.scalar<int64_t>(1);

    CoreTaskArgs gate_args;
    gate_args.add_inout(gate);
    gate_args.add_scalar(gate_value);
    const TaskId gate_task = rt_submit_aic_task(kWriteGateFuncId, gate_args).task_id();

    CoreTaskArgs blocks;
    blocks.add_inout(out);
    blocks.set_dependencies(&gate_task, 1);
    blocks.launch_spec.set_block_num(static_cast<int32_t>(block_num));
    CoreTaskPredicate pred;
    pred.operand.tensor = &gate;
    pred.operand.ndims = 1;
    pred.operand.indices[0] = 0;
    pred.op = PredicateOp::GT;
    pred.target = 0;
    blocks.set_predicate(pred);
    (void)rt_submit_aic_task(kStampBlockFuncId, blocks);
}

}  // extern "C"
