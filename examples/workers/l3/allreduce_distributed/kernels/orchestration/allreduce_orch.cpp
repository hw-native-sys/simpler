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
 * AllReduce orchestration — all-scalar args path.
 *
 * The kernel reads raw uint64 values from args[] (device pointers into the
 * HCCL window + a few ints) and does its own CommRemotePtr math. Wrapping
 * the pointers as tensors would force the framework to rewrite them as
 * Tensor-struct pointers, breaking that math. So every arg goes through
 * add_scalar, and the orchestration forwards them 1:1.
 *
 * scalar layout (from Python orch_fn via ChipStorageTaskArgs):
 *   [0] input device pointer   (HCCL window, remote-addressable)
 *   [1] output device pointer  (HCCL window, local write)
 *   [2] nranks
 *   [3] root rank              (unused in symmetric allreduce, kept for ABI)
 *   [4] CommContext device pointer
 */

#include <stdint.h>

#include "pto_orchestration_api.h"

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
allreduce_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 5,
    };
}

__attribute__((visibility("default"))) void allreduce_orchestration(const ChipStorageTaskArgs &orch_args) {
    Arg params;
    params.add_scalar(orch_args.scalar(0));
    params.add_scalar(orch_args.scalar(1));
    params.add_scalar(orch_args.scalar(2));
    params.add_scalar(orch_args.scalar(3));
    params.add_scalar(orch_args.scalar(4));
    pto2_rt_submit_aiv_task(0, params);
}

}  // extern "C"
