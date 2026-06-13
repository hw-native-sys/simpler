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
 * Speculative early-dispatch — SPMD consumer coverage.
 *
 * DAG:
 *   t0 (single-block AIV, FLAGGED): out_p[i] = i + 1
 *   t1 (SPMD AIV, block_num=B): block i computes out_c[i] = out_p[i] + 10
 *
 * t1 is a multi-block consumer. Block-by-block pre-staging stages as many of its
 * blocks as fit on idle cores and in the doorbell budget (PTO2_SPEC_MAX_DOORBELLS
 * == 3); with B == 4, three blocks are gated + released by doorbell and the
 * remaining block dispatches normally off the ready queue. Golden: out_c[i] =
 * i + 11.
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_FILL 0  // kernel_fill_all (single-block producer)
#define FUNC_ADDK 1  // kernel_spmd_addk (SPMD consumer)

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

    uint32_t inter_shapes[1] = {static_cast<uint32_t>(BLOCK_NUM)};
    TensorCreateInfo inter_ci(inter_shapes, 1, DataType::FLOAT32);

    // t0: single-block producer out_p[i] = i + 1  [flagged]
    Arg params_t0;
    params_t0.add_output(inter_ci);
    params_t0.set_allow_early_resolve();
    TaskOutputTensors outs_t0 = rt_submit_aiv_task(FUNC_FILL, params_t0);
    const Tensor &out_p = outs_t0.get_ref(0);

    // t1: SPMD consumer, block i: out_c[i] = out_p[i] + 10
    Arg params_t1;
    params_t1.add_input(out_p);
    params_t1.add_output(ext_out);
    params_t1.launch_spec.set_block_num(BLOCK_NUM);
    rt_submit_aiv_task(FUNC_ADDK, params_t1);
}

}  // extern "C"
