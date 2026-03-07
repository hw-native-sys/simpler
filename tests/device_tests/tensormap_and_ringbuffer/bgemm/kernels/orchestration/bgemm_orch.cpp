/**
 * BGEMM Orchestration Function (tensormap_and_ringbuffer Runtime)
 *
 * Builds the task graph for tiled matrix multiplication: C = A @ B
 *
 * Configuration read from config tensor (set in golden.py):
 *   - tile_size: tile dimension (tile_size x tile_size per tile)
 *   - grid_k: number of K-dimension partitions
 *   - batch_size: number of independent batch elements
 *
 * Memory layout (tile-first, flattened):
 *   A: [batch_size, grid_k, tile_size, tile_size]
 *   B: [batch_size, grid_k, tile_size, tile_size]
 *   C: [batch_size, tile_size, tile_size]
 *
 * Task graph per output tile C[batch]:
 *   for k in [0, grid_k):
 *     P = A[batch,k] @ B[batch,k]  (gemm_tile on Cube core, func_id=0)
 *     C[batch] = C[batch] + P      (tile_add on Vector core, func_id=1)
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
    pto2_rt_init_tensor_pool(rt);

    void* dev_A = (void*)(uintptr_t)args[ARG_PTR_A];
    void* dev_B = (void*)(uintptr_t)args[ARG_PTR_B];
    void* dev_C = (void*)(uintptr_t)args[ARG_PTR_C];
    size_t size_A = (size_t)args[ARG_SIZE_A];
    size_t size_B = (size_t)args[ARG_SIZE_B];
    size_t size_C = (size_t)args[ARG_SIZE_C];

    // Read config from golden.py
    int64_t* host_config = (int64_t*)(uintptr_t)args[ARG_PTR_CONFIG];
    int tile_size = (int)host_config[0];
    int grid_k = (int)host_config[1];
    int batch_size = (int)host_config[2];
    uint64_t tile_elems = (uint64_t)tile_size * tile_size;

    int grid_m = 1;
    int grid_n = 1;

    LOG_INFO(rt, "[bgemm_orch] tile_size: %d, grid_m: %d, grid_n: %d, grid_k: %d, batch_size: %d",
             tile_size, grid_m, grid_n, grid_k, batch_size);

    // Create 1D external tensors for the full A, B, C arrays
    uint64_t ext_A_shapes[1] = {size_A / sizeof(float)};
    Tensor ext_A = make_tensor_external(dev_A, ext_A_shapes, 1, DataType::FLOAT32);
    uint64_t ext_B_shapes[1] = {size_B / sizeof(float)};
    Tensor ext_B = make_tensor_external(dev_B, ext_B_shapes, 1, DataType::FLOAT32);
    uint64_t ext_C_shapes[1] = {size_C / sizeof(float)};
    Tensor ext_C = make_tensor_external(dev_C, ext_C_shapes, 1, DataType::FLOAT32);

    uint64_t tile_shapes[1] = {tile_elems};

    int total_gemm = 0;
    int total_add = 0;

    for (int batch = 0; batch < batch_size; batch++) {
        PTO2_SCOPE(rt) {
            uint64_t c_elem_offset = (uint64_t)batch * tile_elems;
            uint64_t c_view_offsets[1] = {c_elem_offset};
            Tensor C_view = ext_C.view(tile_shapes, c_view_offsets);

            for (int k_idx = 0; k_idx < grid_k; k_idx++) {
                uint64_t a_elem_offset =
                    ((uint64_t)batch * grid_k + (uint64_t)k_idx) * tile_elems;
                uint64_t b_elem_offset =
                    ((uint64_t)batch * grid_k + (uint64_t)k_idx) * tile_elems;

                uint64_t a_view_offsets[1] = {a_elem_offset};
                Tensor A_view = ext_A.view(tile_shapes, a_view_offsets);
                uint64_t b_view_offsets[1] = {b_elem_offset};
                Tensor B_view = ext_B.view(tile_shapes, b_view_offsets);
                Tensor P = make_tensor(tile_shapes, 1, DataType::FLOAT32);

                // P = A[batch,k] @ B[batch,k]
                PTOParam params_gemm[] = {
                    make_input_param(A_view),
                    make_input_param(B_view),
                    make_output_param(P),
                };
                pto2_rt_submit_task(rt, FUNC_GEMM_TILE, PTO2_WORKER_CUBE,
                                   params_gemm, 3); // gemm
                total_gemm++;

                // C[batch] += P
                PTOParam params_add[] = {
                    make_inout_param(C_view),
                    make_input_param(P),
                };
                pto2_rt_submit_task(rt, FUNC_TILE_ADD, PTO2_WORKER_VECTOR,
                                   params_add, 2); // add
                total_add++;
            }
        }
    }

    LOG_INFO(rt, "[bgemm_orch] Submitted %d gemm tasks and %d add tasks (%d total)",
             total_gemm, total_add, total_gemm + total_add);
}

}  // extern "C"
