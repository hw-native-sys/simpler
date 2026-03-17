/**
 * BGEMM Orchestration Function (tensormap_and_ringbuffer Runtime)
 *
 * Builds the task graph for tiled matrix multiplication: C = A @ B
 *
 * Configuration read from config tensor (set in golden.py):
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
 * Task graph per group:
 *   for k in [0, grid_k):
 *     P[0..incore_loop-1] = A[group,k,0..incore_loop-1] @ B[group,k,0..incore_loop-1]
 *     C[group*incore_loop..] += P[0..incore_loop-1]
 *
 * Dependencies are automatic via TensorMap overlap detection.
 *
 * This file compiles as a standalone .so with zero runtime link dependencies.
 * All runtime calls go through the PTO2RuntimeOps function-pointer table.
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"

#define FUNC_GEMM_TILE 0
#define FUNC_TILE_ADD  1

// Args layout: [ptr_A, ptr_B, ptr_C, ptr_config, size_A, size_B, size_C]
#define ARG_PTR_A      0
#define ARG_PTR_B      1
#define ARG_PTR_C      2
#define ARG_PTR_CONFIG 3
#define ARG_SIZE_A     4
#define ARG_SIZE_B     5
#define ARG_SIZE_C     6

extern "C" {

__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(uint64_t* args, int arg_count) {
    (void)args;
    (void)arg_count;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 7,
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    (void)arg_count;

    void* dev_A = (void*)(uintptr_t)args[ARG_PTR_A];
    void* dev_B = (void*)(uintptr_t)args[ARG_PTR_B];
    void* dev_C = (void*)(uintptr_t)args[ARG_PTR_C];
    void* dev_config = (void*)(uintptr_t)args[ARG_PTR_CONFIG];
    size_t size_A = (size_t)args[ARG_SIZE_A];
    size_t size_B = (size_t)args[ARG_SIZE_B];
    size_t size_C = (size_t)args[ARG_SIZE_C];

    // Read config from golden.py
    int64_t* host_config = (int64_t*)(uintptr_t)args[ARG_PTR_CONFIG];
    int tile_size = (int)host_config[0];
    int grid_k = (int)host_config[1];
    int num_groups = (int)host_config[2];
    int incore_loop = (int)host_config[3];
    uint64_t tile_elems = (uint64_t)tile_size * tile_size;

    int grid_m = 1;
    int grid_n = 1;

    LOG_INFO(rt, "[bgemm_orch] tile_size: %d, grid_m: %d, grid_n: %d, grid_k: %d, num_groups: %d, incore_loop: %d",
             tile_size, grid_m, grid_n, grid_k, num_groups, incore_loop);

    // Create 1D external tensors for the full A, B, C arrays
    uint32_t ext_A_shapes[1] = {(uint32_t)(size_A / sizeof(float))};
    Tensor ext_A = make_tensor_external(dev_A, ext_A_shapes, 1, DataType::FLOAT32);
    uint32_t ext_B_shapes[1] = {(uint32_t)(size_B / sizeof(float))};
    Tensor ext_B = make_tensor_external(dev_B, ext_B_shapes, 1, DataType::FLOAT32);
    uint32_t ext_C_shapes[1] = {(uint32_t)(size_C / sizeof(float))};
    Tensor ext_C = make_tensor_external(dev_C, ext_C_shapes, 1, DataType::FLOAT32);

    // Wrap config as a device tensor so AICore kernels can read tile_size directly
    uint32_t config_shapes[1] = {4};  // [tile_size, grid_k, num_groups, incore_loop]
    Tensor ext_config = make_tensor_external(dev_config, config_shapes, 1, DataType::INT64);

    uint32_t tile_shapes[1] = {(uint32_t)tile_elems};
    uint64_t group_tile_elems = (uint64_t)incore_loop * tile_elems;
    uint32_t group_shapes[1] = {(uint32_t)group_tile_elems};

    int total_gemm = 0;
    int total_add = 0;

    // A/B layout: [num_groups, grid_k, incore_loop, tile_size, tile_size]
    // C layout:   [incore_loop * num_groups, tile_size, tile_size]
    for (int group_idx = 0; group_idx < num_groups; group_idx++) {
        PTO2_SCOPE_GUARD(rt);

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
            pto2_rt_submit_aic_task(rt, FUNC_GEMM_TILE, params_gemm);
            total_gemm++;

            PTOParam params_add;
            params_add.add_inout(C_view);
            params_add.add_input(P);
            params_add.add_input(ext_config);
            pto2_rt_submit_aiv_task(rt, FUNC_TILE_ADD, params_add);
            total_add++;
        }
    }

    LOG_INFO(rt, "[bgemm_orch] Submitted %d gemm tasks and %d add tasks (%d total)",
             total_gemm, total_add, total_gemm + total_add);
}

}  // extern "C"
