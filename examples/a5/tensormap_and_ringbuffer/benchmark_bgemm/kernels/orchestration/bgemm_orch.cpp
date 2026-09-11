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
 * A5 benchmark BGEMM orchestration.
 *
 * The topology parameters are scalar orchestration arguments. The fixed-size
 * A5 incore kernels derive incore_loop from their tensor views.
 *
 * Arg layout: [A, B, C, tile_size, grid_k, num_groups, incore_loop]
 */

#include <stddef.h>
#include <stdint.h>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_GEMM_TILE 0
#define FUNC_TILE_ADD 1

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return OrchestrationConfig{
        .expected_arg_count = 7,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const simpler::tmr::Tensor &ext_A = orch_args.tensor(0).ref();
    const simpler::tmr::Tensor &ext_B = orch_args.tensor(1).ref();
    const simpler::tmr::Tensor &ext_C = orch_args.tensor(2).ref();

    int tile_size = orch_args.scalar<int>(0);
    int grid_k = orch_args.scalar<int>(1);
    int num_groups = orch_args.scalar<int>(2);
    int incore_loop = orch_args.scalar<int>(3);
    uint64_t tile_elems = static_cast<uint64_t>(tile_size) * tile_size;

    LOG_INFO(
        "[bgemm_orch] tile_size: %d, grid_k: %d, num_groups: %d, incore_loop: %d", tile_size, grid_k, num_groups,
        incore_loop
    );

    uint64_t group_tile_elems = static_cast<uint64_t>(incore_loop) * tile_elems;
    uint32_t group_shapes[1] = {static_cast<uint32_t>(group_tile_elems)};
    TensorCreateInfo group_ci(group_shapes, 1, DataType::FLOAT32);

    int total_gemm = 0;
    int total_add = 0;

    for (int group_idx = 0; group_idx < num_groups; group_idx++) {
        SIMPLER_SCOPE_GUARD();

        uint32_t c_elem_offset = static_cast<uint32_t>(static_cast<uint64_t>(group_idx) * group_tile_elems);
        uint32_t c_view_offsets[1] = {c_elem_offset};
        simpler::tmr::Tensor C_view = ext_C.view(group_shapes, c_view_offsets);

        for (int k_idx = 0; k_idx < grid_k; k_idx++) {
            uint64_t ab_offset =
                (static_cast<uint64_t>(group_idx) * grid_k + static_cast<uint64_t>(k_idx)) * group_tile_elems;

            uint32_t a_view_offsets[1] = {static_cast<uint32_t>(ab_offset)};
            simpler::tmr::Tensor A_view = ext_A.view(group_shapes, a_view_offsets);
            uint32_t b_view_offsets[1] = {static_cast<uint32_t>(ab_offset)};
            simpler::tmr::Tensor B_view = ext_B.view(group_shapes, b_view_offsets);

            CoreTaskArgs params_gemm;
            params_gemm.add_input(A_view);
            params_gemm.add_input(B_view);
            params_gemm.add_output(group_ci);
            TaskOutputTensors gemm_outs = rt_submit_aic_task(FUNC_GEMM_TILE, params_gemm);
            total_gemm++;

            CoreTaskArgs params_add;
            params_add.add_inout(C_view);
            params_add.add_input(gemm_outs.get_ref(0));
            rt_submit_aiv_task(FUNC_TILE_ADD, params_add);
            total_add++;
        }
    }

    LOG_INFO(
        "[bgemm_orch] Submitted %d gemm tasks and %d add tasks (%d total)", total_gemm, total_add,
        total_gemm + total_add
    );
}

}  // extern "C"
