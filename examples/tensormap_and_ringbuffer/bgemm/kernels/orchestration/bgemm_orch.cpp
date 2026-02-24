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
 *     P = A[m,k] @ B[k,n]    (gemm_tile on Cube core, func_id=0)
 *     C[m,n] = C[m,n] + P    (tile_add on Vector core, func_id=1)
 *
 * Dependencies are automatic via TensorMap overlap detection.
 *
 * This file compiles as a standalone .so with zero runtime link dependencies.
 * All runtime calls go through the PTO2RuntimeOps function-pointer table.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "pto_orchestration_api.h"

#define FUNC_GEMM_TILE 0
#define FUNC_TILE_ADD  1

// Grid and tile constants
static constexpr int TILE = 64;
static constexpr int GRID_M = 4;
static constexpr int GRID_K = 4;
static constexpr int GRID_N = 4;
static constexpr int BATCH = 2;

static constexpr uint64_t TILE_ELEMS = TILE * TILE;           // 4096 elements
static constexpr uint64_t TILE_BYTES = TILE_ELEMS * sizeof(float);  // 16384 bytes

// Args layout: [ptr_A, ptr_B, ptr_C, size_A, size_B, size_C, elem_count]
#define ARG_PTR_A   0
#define ARG_PTR_B   1
#define ARG_PTR_C   2
#define ARG_SIZE_A  3
#define ARG_SIZE_B  4
#define ARG_SIZE_C  5

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
    size_t size_A = (size_t)args[ARG_SIZE_A];
    size_t size_B = (size_t)args[ARG_SIZE_B];
    size_t size_C = (size_t)args[ARG_SIZE_C];

    printf("[bgemm_orch] Grid: %dx%dx%d, Batch: %d, Tile: %d\n",
           GRID_M, GRID_K, GRID_N, BATCH, TILE);

    // Create 1D external tensors for the full A, B, C arrays
    Tensor ext_A = make_tensor_external(dev_A, size_A);
    Tensor ext_B = make_tensor_external(dev_B, size_B);
    Tensor ext_C = make_tensor_external(dev_C, size_C);

    for (int batch = 0; batch < BATCH; batch++) {
        for (int m_idx = 0; m_idx < GRID_M; m_idx++) {
            for (int n_idx = 0; n_idx < GRID_N; n_idx++) {
                PTO2_SCOPE(rt) {
                    uint64_t c_elem_offset =
                        ((uint64_t)batch * GRID_M * GRID_N +
                         (uint64_t)m_idx * GRID_N +
                         (uint64_t)n_idx) * TILE_ELEMS;
                    Tensor C_view = ext_C.view({TILE_ELEMS}, {c_elem_offset});

                    for (int k_idx = 0; k_idx < GRID_K; k_idx++) {
                        uint64_t a_elem_offset =
                            ((uint64_t)batch * GRID_M * GRID_K +
                             (uint64_t)m_idx * GRID_K +
                             (uint64_t)k_idx) * TILE_ELEMS;
                        uint64_t b_elem_offset =
                            ((uint64_t)batch * GRID_K * GRID_N +
                             (uint64_t)k_idx * GRID_N +
                             (uint64_t)n_idx) * TILE_ELEMS;

                        Tensor A_view = ext_A.view({TILE_ELEMS}, {a_elem_offset});
                        Tensor B_view = ext_B.view({TILE_ELEMS}, {b_elem_offset});
                        Tensor P = make_tensor(TILE_BYTES);

                        // P = A[m,k] @ B[k,n]
                        PTOParam params_gemm[] = {
                            make_input_param(A_view),
                            make_input_param(B_view),
                            make_output_param(P),
                        };
                        pto2_rt_submit_task(rt, FUNC_GEMM_TILE, PTO2_WORKER_CUBE,
                                           params_gemm, 3); // gemm

                        // C[m,n] += P
                        PTOParam params_add[] = {
                            make_inout_param(C_view),
                            make_input_param(P),
                        };
                        pto2_rt_submit_task(rt, FUNC_TILE_ADD, PTO2_WORKER_VECTOR,
                                           params_add, 2); // add
                    }
                }
            }
        }
    }

    printf("[bgemm_orch] Submitted tasks for %d batches, %dx%d output tiles, %d K steps each\n",
           BATCH, GRID_M, GRID_N, GRID_K);
}

}  // extern "C"
