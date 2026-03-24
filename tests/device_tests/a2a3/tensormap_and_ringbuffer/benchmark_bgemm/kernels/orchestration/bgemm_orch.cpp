/**
 * BGEMM Orchestration Function (tensormap_and_ringbuffer Runtime)
 *
 * Builds the task graph for tiled matrix multiplication: C = A @ B
 *
 * Configuration read from scalar OrchArgs (set in golden.py):
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
 * Args layout: [A, B, C, config]
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"

#define FUNC_GEMM_TILE 0
#define FUNC_TILE_ADD  1

extern "C" {

__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(OrchArg* orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 4,
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(OrchArg* orch_args, int orch_thread_num, int orch_thread_index) {
    (void)orch_thread_num;
    (void)orch_thread_index;

    // Tensor args
    Tensor ext_A = orch_args[0].to_tensor();
    Tensor ext_B = orch_args[1].to_tensor();
    Tensor ext_C = orch_args[2].to_tensor();
    Tensor ext_config = orch_args[3].to_tensor();

    // Read config from tensor data: [tile_size, grid_k, num_groups, incore_loop]
    int64_t* host_config = orch_args[3].data<int64_t>();
    int tile_size = (int)host_config[0];
    int grid_k = (int)host_config[1];
    int num_groups = (int)host_config[2];
    int incore_loop = (int)host_config[3];
    uint64_t tile_elems = (uint64_t)tile_size * tile_size;

    int grid_m = 1;
    int grid_n = 1;

    LOG_INFO("[bgemm_orch] tile_size: %d, grid_m: %d, grid_n: %d, grid_k: %d, num_groups: %d, incore_loop: %d",
             tile_size, grid_m, grid_n, grid_k, num_groups, incore_loop);

    uint32_t tile_shapes[1] = {(uint32_t)tile_elems};
    uint64_t group_tile_elems = (uint64_t)incore_loop * tile_elems;
    uint32_t group_shapes[1] = {(uint32_t)group_tile_elems};

    int total_gemm = 0;
    int total_add = 0;

    // A/B layout: [num_groups, grid_k, incore_loop, tile_size, tile_size]
    // C layout:   [incore_loop * num_groups, tile_size, tile_size]
    for (int group_idx = 0; group_idx < num_groups; group_idx++) {
        PTO2_SCOPE_GUARD();

        uint32_t c_elem_offset = (uint32_t)((uint64_t)group_idx * group_tile_elems);
        uint32_t c_view_offsets[1] = {c_elem_offset};
        Tensor C_view = ext_C.view(group_shapes, c_view_offsets);

        for (int k_idx = 0; k_idx < grid_k; k_idx++) {
            // In layout [num_groups, grid_k, incore_loop, tile_size, tile_size],
            // offset = (group_idx * grid_k + k_idx) * incore_loop * tile_elems
            uint64_t ab_offset =
                ((uint64_t)group_idx * grid_k + (uint64_t)k_idx) * group_tile_elems;

            uint32_t a_view_offsets[1] = {(uint32_t)ab_offset};
            Tensor A_view = ext_A.view(group_shapes, a_view_offsets);
            uint32_t b_view_offsets[1] = {(uint32_t)ab_offset};
            Tensor B_view = ext_B.view(group_shapes, b_view_offsets);
            Tensor P = make_tensor(group_shapes, 1, DataType::FLOAT32);

            PTOParam params_gemm;
            params_gemm.add_input(A_view);
            params_gemm.add_input(B_view);
            params_gemm.add_output(P);
            params_gemm.add_input(ext_config);
            pto2_rt_submit_aic_task(FUNC_GEMM_TILE, params_gemm);
            total_gemm++;

            PTOParam params_add;
            params_add.add_inout(C_view);
            params_add.add_input(P);
            params_add.add_input(ext_config);
            pto2_rt_submit_aiv_task(FUNC_TILE_ADD, params_add);
            total_add++;
        }
    }

    LOG_INFO("[bgemm_orch] Submitted %d gemm tasks and %d add tasks (%d total)",
             total_gemm, total_add, total_gemm + total_add);
}

}  // extern "C"
