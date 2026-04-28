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
 * BGEMM Orchestration Function (tensormap_and_ringbuffer Runtime)
 *
 * Builds the task graph for tiled matrix multiplication: C = A @ B
 *
 * Configuration:
 *   - Tile size: 64 x 64
 *   - Grid: 4 x 4 x 4 (GRID_M x GRID_K x GRID_N)
 *   - Batch: 2
 *
 * Memory layout (tile-first, 5D flattened):
 *   A: [BATCH, GRID_M, GRID_K, TILE, TILE]
 *   B: [BATCH, GRID_K, GRID_N, TILE, TILE]
 *   C: [BATCH, GRID_M, GRID_N, TILE, TILE]
 *
 * Task graph per output tile C[batch, m, n]:
 *   for k in [0, GRID_K):
 *     P = A[m,k] @ B[k,n]    (matmul on Cube core, func_id=0)
 *     C[m,n] = C[m,n] + P    (accumulate on Vector core, func_id=1)
 *
 * Each iteration submits a MixedKernels task (AIC + AIV0 + AIV1).
 * Intermediate result P flows via TPUSH/TPOP (VEC_FIFO), bypassing GM.
 * Dependencies are automatic via TensorMap overlap detection.
 *
 * Arg layout: [A, B, C]  — shape/dtype/size in ContinuousTensor metadata
 */

#include <cstdint>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_GEMM_TILE 0
#define FUNC_TILE_ADD 1

// Grid and tile constants
static constexpr int TILE = 64;
static constexpr int GRID_M = 4;
static constexpr int GRID_K = 4;
static constexpr int GRID_N = 4;
static constexpr int BATCH = 2;

static constexpr uint32_t TILE_ELEMS = TILE * TILE;  // 4096 elements

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 3,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    // 1D external tensors for the full A, B, C arrays
    Tensor ext_A = from_tensor_arg(orch_args.tensor(0));
    Tensor ext_B = from_tensor_arg(orch_args.tensor(1));
    Tensor ext_C = from_tensor_arg(orch_args.tensor(2));

    LOG_INFO("[bgemm_orch] Grid: %dx%dx%d, Batch: %d, Tile: %d", GRID_M, GRID_K, GRID_N, BATCH, TILE);

    uint32_t tile_shapes[1] = {TILE_ELEMS};

    for (int batch = 0; batch < BATCH; batch++) {
        for (int m_idx = 0; m_idx < GRID_M; m_idx++) {
            for (int n_idx = 0; n_idx < GRID_N; n_idx++) {
                PTO2_SCOPE() {
                    uint32_t c_elem_offset = (static_cast<uint32_t>(batch) * GRID_M * GRID_N +
                                              static_cast<uint32_t>(m_idx) * GRID_N + static_cast<uint32_t>(n_idx)) *
                                             TILE_ELEMS;
                    uint32_t c_view_offsets[1] = {c_elem_offset};
                    Tensor C_view = ext_C.view(tile_shapes, c_view_offsets);

                    for (int k_idx = 0; k_idx < GRID_K; k_idx++) {
                        uint32_t a_elem_offset =
                            (static_cast<uint32_t>(batch) * GRID_M * GRID_K + static_cast<uint32_t>(m_idx) * GRID_K +
                             static_cast<uint32_t>(k_idx)) *
                            TILE_ELEMS;
                        uint32_t b_elem_offset =
                            (static_cast<uint32_t>(batch) * GRID_K * GRID_N + static_cast<uint32_t>(k_idx) * GRID_N +
                             static_cast<uint32_t>(n_idx)) *
                            TILE_ELEMS;

                        uint32_t a_view_offsets[1] = {a_elem_offset};
                        Tensor A_view = ext_A.view(tile_shapes, a_view_offsets);
                        uint32_t b_view_offsets[1] = {b_elem_offset};
                        Tensor B_view = ext_B.view(tile_shapes, b_view_offsets);

                        // P = A[m,k] @ B[k,n], then C[m,n] += P
                        Arg args;
                        args.add_input(A_view);
                        args.add_input(B_view);
                        args.add_inout(C_view);

                        MixedKernels mk;
                        mk.aic_kernel_id = FUNC_GEMM_TILE;
                        mk.aiv0_kernel_id = FUNC_TILE_ADD;
                        mk.aiv1_kernel_id = FUNC_TILE_ADD;
                        rt_submit_task(mk, args);
                    }
                }
            }
        }
    }

    LOG_INFO(
        "[bgemm_orch] Submitted tasks for %d batches, %dx%d output tiles, %d K steps each", BATCH, GRID_M, GRID_N,
        GRID_K
    );
}

}  // extern "C"
