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
 * Triangular Inverse Orchestration (tensormap_and_ringbuffer Runtime)
 *
 * Builds the task graph for batch triangular matrix inversion.
 *
 * Arg layout (set in test_triangular_inverse.py):
 *   tensor(0) = M      (INPUT)  fp16 triangular matrices [num_matrices * N * N]
 *   tensor(1) = I_neg  (INPUT)  fp16 negative identity   [N * N]
 *   tensor(2) = M_inv  (OUTPUT) fp16 result              [num_matrices * N * N]
 *   tensor(4) = config (INPUT)  int64[4]: [matrix_size, num_matrices, is_lower, block_dim]
 *
 * The single AIC task (func_id=0) receives these four args in the same order
 * and dispatches to run_tri_inv_rec_unroll_per_num_matrices.
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_TRI_INV 0

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 4,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    Tensor ext_M = from_tensor_arg(orch_args.tensor(0));
    Tensor ext_I_neg = from_tensor_arg(orch_args.tensor(1));
    Tensor ext_M_inv = from_tensor_arg(orch_args.tensor(2));
    Tensor ext_config = from_tensor_arg(orch_args.tensor(3));

    int64_t *host_config = orch_args.tensor(3).data_as<int64_t>();
    int matrix_size = static_cast<int>(host_config[0]);
    int num_matrices = static_cast<int>(host_config[1]);
    int is_lower = static_cast<int>(host_config[2]);
    int block_dim = static_cast<int>(host_config[3]);

    LOG_INFO_V0(
        "[tri_inv_orch] matrix_size: %d, num_matrices: %d, is_lower: %d, block_dim: %d", matrix_size, num_matrices,
        is_lower, block_dim
    );

    Arg params;
    params.add_input(ext_M);
    params.add_input(ext_I_neg);
    params.add_output(ext_M_inv);
    params.add_input(ext_config);
    rt_submit_aic_task(FUNC_TRI_INV, params);
}

}  // extern "C"
