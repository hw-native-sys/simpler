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
 * Simple MatMul Orchestration Function (tensormap_and_ringbuffer Runtime)
 *
 * Builds the task graph (one node) for tiled matrix multiplication: C = A @ B
 *
 * Configuration read from scalar args (set in golden.py):
 *   - matrix_size: matrix multiplication size (M=N=K)
 *   - batch_size: batch size (number of independent matmuls)
 *
 * Memory layout (tile-first, flattened):
 *   A: [batch_size, matrix_size, matrix_size]
 *   B: [batch_size, matrix_size, matrix_size]
 *   C: [batch_size, matrix_size, matrix_size]
 *
 * Arg layout: [A, B, C, config]
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_SIMPLE_MATMUL 0

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 4,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    // Tensor args
    Tensor ext_A = from_tensor_arg(orch_args.tensor(0));
    Tensor ext_B = from_tensor_arg(orch_args.tensor(1));
    Tensor ext_C = from_tensor_arg(orch_args.tensor(2));
    Tensor ext_config = from_tensor_arg(orch_args.tensor(3));

    // Read config from tensor data: [batch_size, matrix_size]
    int64_t *host_config = orch_args.tensor(3).data_as<int64_t>();
    int batch_dim = static_cast<int>(host_config[0]);
    int matrix_size = static_cast<int>(host_config[1]);

    int grid_m = 1;
    int grid_n = 1;

    LOG_INFO_V0("[simple_matmul_orch] batch_dim: %d, matrix_size: %d", batch_dim, matrix_size);

    Arg params_gemm;
    params_gemm.add_input(ext_A);
    params_gemm.add_input(ext_B);
    params_gemm.add_output(ext_C);
    params_gemm.add_input(ext_config);
    TaskOutputTensors gemm_outs = rt_submit_aic_task(FUNC_SIMPLE_MATMUL, params_gemm);
}

}  // extern "C"
