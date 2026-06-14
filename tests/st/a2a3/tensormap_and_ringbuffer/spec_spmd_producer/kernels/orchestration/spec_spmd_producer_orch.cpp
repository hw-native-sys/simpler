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
 * Speculative early-dispatch — SPMD producer coverage.
 *
 * DAG:
 *   t0 (SPMD AIV, block_num=B, FLAGGED): block i writes out_p[i] = i + 1
 *   t1 (single-block AIV, depends on t0): out_c[i] = 2 * out_p[i]
 *
 * t0 is a multi-block producer. Its single fanin edge into t1 is released only
 * when ALL B blocks complete, so the completion-path doorbell must fire t1
 * exactly once — after the last block. t1 is pre-staged onto an idle core while
 * t0's blocks run. A premature release would leave some out_p[i] == 0 and the
 * result would mismatch the golden [2, 4, ..., 2B].
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_FILL 0    // kernel_spmd_fill (SPMD producer)
#define FUNC_DOUBLE 1  // kernel_double (single-block consumer)

static constexpr int16_t BLOCK_NUM = 4;

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 1,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    Tensor ext_out = from_tensor_arg(orch_args.tensor(0));  // out_c

    // Stride the SPMD producer's output by one cache line per block (16 floats
    // = 64B on a2a3) so no two cores write the same line. See
    // docs/aicore-kernel-programming.md "Each block must write to its own cache line".
    uint32_t inter_shapes[1] = {static_cast<uint32_t>(BLOCK_NUM) * 16};
    TensorCreateInfo inter_ci(inter_shapes, 1, DataType::FLOAT32);

    // t0: SPMD producer, block i writes out_p[i] = i + 1  [flagged]
    Arg params_t0;
    params_t0.add_output(inter_ci);
    params_t0.launch_spec.set_block_num(BLOCK_NUM);
    params_t0.set_allow_early_resolve();
    TaskOutputTensors outs_t0 = rt_submit_aiv_task(FUNC_FILL, params_t0);
    const Tensor &out_p = outs_t0.get_ref(0);

    // t1: single-block consumer, out_c[i] = 2 * out_p[i]
    Arg params_t1;
    params_t1.add_input(out_p);
    params_t1.add_output(ext_out);
    rt_submit_aiv_task(FUNC_DOUBLE, params_t1);
}

}  // extern "C"
