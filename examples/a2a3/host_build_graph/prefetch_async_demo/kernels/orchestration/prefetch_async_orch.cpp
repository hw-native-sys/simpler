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
 * Orchestration for the single-card TPREFETCH_ASYNC demo.
 *
 * Only user data (in, out) is threaded: the SDMA workspace is a runtime-owned
 * device resource the kernel reads via get_dma_workspace(args, DMA_WORKSPACE_SDMA),
 * so it is neither an orchestration arg nor copied in H2D.
 */

#include <stdint.h>

#include "orchestration_api.h"

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig
prefetch_async_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return OrchestrationConfig{.expected_arg_count = 2};
}

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    return prefetch_async_orchestration_config(orch_args);
}

__attribute__((visibility("default"))) void prefetch_async_orchestration(const ChipTaskArgs &orch_args) {
    if (orch_args.tensor_count() != 2 || orch_args.scalar_count() != 0) {
        LOG_ERROR("prefetch_async_demo: expected 2 tensor args (in, out) and no scalars");
        return;
    }

    const simpler::hbg::Tensor &in = orch_args.tensor(0).ref();
    const simpler::hbg::Tensor &out = orch_args.tensor(1).ref();
    if (in.numel() != 128 || out.numel() != 128 || in.dtype != DataType::FLOAT32 || out.dtype != DataType::FLOAT32 ||
        !in.is_contiguous || !out.is_contiguous) {
        LOG_ERROR("prefetch_async_demo: expected contiguous FP32 tensors with 128 elements");
        return;
    }

    CoreTaskArgs task_args;
    task_args.add_input(in);
    task_args.add_output(out);
    rt_submit_aiv_task(0, task_args);
}

}  // extern "C"
