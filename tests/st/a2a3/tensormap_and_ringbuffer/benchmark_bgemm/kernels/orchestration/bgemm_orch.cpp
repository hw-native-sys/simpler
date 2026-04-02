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
 * Configuration read from scalar args (set in golden.py):
 *   - tile_size: tile dimension (tile_size x tile_size per tile)
 *   - grid_k: number of K-dimension partitions
 *   - num_groups: number of independent groups (= matmul_add_task_num / grid_k)
 *   - incore_loop: number of tiles per group
 *
 * Memory layout (tile-first, flattened):
 *   A: [num_groups, grid_k, incore_loop, tile_size, tile_size]
 *   B: [num_groups, grid_k, incore_loop, tile_size, tile_size]
 *   C: [incore_loop * num_groups, tile_size, tile_size]
 *
 * Arg layout: [A, B, C, config]
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_GEMM_TILE 0
#define FUNC_TILE_ADD 1

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 4,
    };
}

__attribute__((visibility("default"))) void
aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args, int orch_thread_num, int orch_thread_index) {
    (void)orch_thread_num;    // NOLINT(readability/casting)
    (void)orch_thread_index;  // NOLINT(readability/casting)

    // Tensor args
    Tensor ext_A = from_tensor_arg(orch_args.tensor(0));
    Tensor ext_B = from_tensor_arg(orch_args.tensor(1));
    Tensor ext_C = from_tensor_arg(orch_args.tensor(2));
    Tensor ext_config = from_tensor_arg(orch_args.tensor(3));

    // Read config from tensor data: [tile_size, grid_k, num_groups, incore_loop]
    int64_t *host_config = orch_args.tensor(3).data_as<int64_t>();
    int tile_size = static_cast<int>(host_config[0]);
    int grid_k = static_cast<int>(host_config[1]);
    int num_groups = static_cast<int>(host_config[2]);
    int incore_loop = static_cast<int>(host_config[3]);
    uint64_t tile_elems = static_cast<uint64_t>(tile_size) * tile_size;

    int grid_m = 1;
    int grid_n = 1;

    LOG_INFO(
        "[bgemm_orch] tile_size: %d, grid_m: %d, grid_n: %d, grid_k: %d, num_groups: %d, incore_loop: %d", tile_size,
        grid_m, grid_n, grid_k, num_groups, incore_loop
    );

    uint32_t tile_shapes[1] = {static_cast<uint32_t>(tile_elems)};
    uint64_t group_tile_elems = static_cast<uint64_t>(incore_loop) * tile_elems;
    uint32_t group_shapes[1] = {static_cast<uint32_t>(group_tile_elems)};
    TensorCreateInfo group_ci(group_shapes, 1, DataType::FLOAT32);

    int total_gemm = 0;
    int total_add = 0;

    // A/B layout: [num_groups, grid_k, incore_loop, tile_size, tile_size]
    // C layout:   [incore_loop * num_groups, tile_size, tile_size]
    for (int group_idx = 0; group_idx < num_groups; group_idx++) {
        PTO2_SCOPE_GUARD();

        uint32_t c_elem_offset = static_cast<uint32_t>(static_cast<uint64_t>(group_idx) * group_tile_elems);
        uint32_t c_view_offsets[1] = {c_elem_offset};
        Tensor C_view = ext_C.view(group_shapes, c_view_offsets);

        for (int k_idx = 0; k_idx < grid_k; k_idx++) {
            // In layout [num_groups, grid_k, incore_loop, tile_size, tile_size],
            // offset = (group_idx * grid_k + k_idx) * incore_loop * tile_elems
            uint64_t ab_offset =
                (static_cast<uint64_t>(group_idx) * grid_k + static_cast<uint64_t>(k_idx)) * group_tile_elems;

            uint32_t a_view_offsets[1] = {static_cast<uint32_t>(ab_offset)};
            Tensor A_view = ext_A.view(group_shapes, a_view_offsets);
            uint32_t b_view_offsets[1] = {static_cast<uint32_t>(ab_offset)};
            Tensor B_view = ext_B.view(group_shapes, b_view_offsets);
            Arg params_gemm;
            params_gemm.add_input(A_view);
            params_gemm.add_input(B_view);
            params_gemm.add_output(group_ci);
            params_gemm.add_input(ext_config);
            TaskOutputTensors gemm_outs = pto2_rt_submit_aic_task(FUNC_GEMM_TILE, params_gemm);
            total_gemm++;

            Arg params_add;
            params_add.add_inout(C_view);
            params_add.add_input(gemm_outs.get_ref(0));
            params_add.add_input(ext_config);
            pto2_rt_submit_aiv_task(FUNC_TILE_ADD, params_add);
            total_add++;
        }
    }

    LOG_INFO(
        "[bgemm_orch] Submitted %d gemm tasks and %d add tasks (%d total)", total_gemm, total_add,
        total_gemm + total_add
    );
}

}  // extern "C"
