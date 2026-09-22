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
 * IBing AllReduce orchestration — kernel shim.
 *
 * Three simpler::tmr::Tensor args plus two scalars:
 *
 *   tensor(0) input   INPUT           (plain device mem, staged in by bootstrap)
 *   tensor(1) output  OUTPUT_EXISTING (plain device mem, flushed by bootstrap)
 *   tensor(2) scratch INOUT           (HCCL window; IBing interleaved RS+AG in AIV kernel)
 *   scalar(0) nranks
 *   scalar(1) CommContext device pointer
 *
 * INOUT on scratch expresses that the kernel both writes (stage-in, push
 * to neighbours) and is updated by neighbours (remote atomic pushes
 * land directly in the local chunks[] array).
 */

#include <stdint.h>

#include "orchestration_api.h"

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig
allreduce_ibing_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return OrchestrationConfig{
        .expected_arg_count = 6,  // 3 tensors + 3 scalars
    };
}

__attribute__((visibility("default"))) void allreduce_ibing_orchestration(const ChipTaskArgs &orch_args) {
    const simpler::tmr::Tensor &input = orch_args.tensor(0).ref();
    const simpler::tmr::Tensor &output = orch_args.tensor(1).ref();
    const simpler::tmr::Tensor &scratch = orch_args.tensor(2).ref();

    // TPUT<AtomicAdd> only supports Sum reduction (CollectiveReduceOp::kSum == 0
    // in simpler_setup/incore/collectives_reduce_op.hpp); the AIV kernel silently
    // skips writing `output` for any other op, so reject before submission.
    uint64_t reduce_op = orch_args.scalar(2);
    if (reduce_op != 0) {
        rt_report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, "allreduce_ibing_orch: reduce_op=%llu unsupported, only Sum (0)",
            static_cast<unsigned long long>(reduce_op)
        );
        return;
    }

    CoreTaskArgs params;
    params.add_input(input);
    params.add_output(output);
    params.add_inout(scratch);
    params.add_scalar(orch_args.scalar(0));  // nranks
    params.add_scalar(orch_args.scalar(1));  // CommContext
    params.add_scalar(orch_args.scalar(2));  // reduce_op
    rt_submit_aiv_task(0, params);
}

}  // extern "C"
