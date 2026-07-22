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
 * Merge mode: one block_num=8 SPMD task (merge_pipeline, func 0) runs all three
 * stages with an intra-task barrier between segments. s1/s2 are runtime scratch;
 * sync is a caller-provided zero-initialized barrier buffer.
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{.expected_arg_count = 4};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    const Tensor &ext_x = orch_args.tensor(0).ref();
    const Tensor &ext_sync = orch_args.tensor(1).ref();
    const Tensor &ext_out = orch_args.tensor(2).ref();
    const Tensor &ext_timing = orch_args.tensor(3).ref();

    uint32_t SIZE = ext_x.shapes[0];
    uint32_t sh[1] = {SIZE};
    TensorCreateInfo ci(sh, 1, DataType::FLOAT32);

    L0TaskArgs p;
    p.add_input(ext_x);        // args[0]
    p.add_input(ext_sync);     // args[1]
    p.add_output(ci);          // args[2] s1 scratch
    p.add_output(ci);          // args[3] s2 scratch
    p.add_output(ext_out);     // args[4]
    p.add_output(ext_timing);  // args[5] per-block timing (ticks)
    p.launch_spec.set_block_num(8);
    rt_submit_aiv_task(0, p);
}

}  // extern "C"
