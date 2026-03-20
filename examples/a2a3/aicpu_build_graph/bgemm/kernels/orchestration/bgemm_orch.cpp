/**
 * BGEMM Orchestration Function (aicpu_build_graph Runtime)
 *
 * Builds the task graph for tiled matrix multiplication: C = A @ B
 *
 * Configuration:
 *   - Tile size: 64 x 64
 *   - Grid: 4 x 4 x 4 (GRID_M x GRID_K x GRID_N)
 *   - Batch: 1
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
 * Dependencies are explicit via pto2_rt_add_dependency:
 *   - gemm(k) -> add(k): add reads P which gemm produces
 *   - add(k-1) -> add(k): add reads/writes C_view (K accumulation chain)
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"

#define FUNC_GEMM_TILE 0
#define FUNC_TILE_ADD  1

static constexpr int TILE = 64;
static constexpr int GRID_M = 4;
static constexpr int GRID_K = 4;
static constexpr int GRID_N = 4;
static constexpr int BATCH = 1;

static constexpr uint32_t TILE_ELEMS = TILE * TILE;
static constexpr uint64_t TILE_BYTES = TILE_ELEMS * sizeof(float);

// Args layout: [ptr_A, ptr_B, ptr_C, size_A, size_B, size_C]
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
        .expected_arg_count = 6,
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(PTO2Runtime* rt, uint64_t* args, int arg_count, int orch_thread_num, int orch_thread_index) {
    (void)arg_count;
    (void)orch_thread_num;
    (void)orch_thread_index;

    void* dev_A = (void*)(uintptr_t)args[ARG_PTR_A];
    void* dev_B = (void*)(uintptr_t)args[ARG_PTR_B];
    void* dev_C = (void*)(uintptr_t)args[ARG_PTR_C];
    size_t size_A = (size_t)args[ARG_SIZE_A];
    size_t size_B = (size_t)args[ARG_SIZE_B];
    size_t size_C = (size_t)args[ARG_SIZE_C];

    LOG_INFO(rt, "[bgemm_orch] Grid: %dx%dx%d, Batch: %d, Tile: %d",
                  GRID_M, GRID_K, GRID_N, BATCH, TILE);

    uint32_t ext_A_shapes[1] = {(uint32_t)(size_A / sizeof(float))};
    Tensor ext_A = make_tensor_external(dev_A, ext_A_shapes, 1, DataType::FLOAT32);
    uint32_t ext_B_shapes[1] = {(uint32_t)(size_B / sizeof(float))};
    Tensor ext_B = make_tensor_external(dev_B, ext_B_shapes, 1, DataType::FLOAT32);
    uint32_t ext_C_shapes[1] = {(uint32_t)(size_C / sizeof(float))};
    Tensor ext_C = make_tensor_external(dev_C, ext_C_shapes, 1, DataType::FLOAT32);

    uint32_t tile_shapes[1] = {TILE_ELEMS};

    for (int batch = 0; batch < BATCH; batch++) {
        for (int m_idx = 0; m_idx < GRID_M; m_idx++) {
            for (int n_idx = 0; n_idx < GRID_N; n_idx++) {
                PTO2_SCOPE(rt) {
                    uint32_t c_elem_offset =
                        ((uint32_t)batch * GRID_M * GRID_N +
                         (uint32_t)m_idx * GRID_N +
                         (uint32_t)n_idx) * TILE_ELEMS;
                    uint32_t c_view_offsets[1] = {c_elem_offset};
                    Tensor C_view = ext_C.view(tile_shapes, c_view_offsets);

                    PTO2TaskId last_add_task = PTO2TaskId{};
                    bool has_last_add = false;

                    for (int k_idx = 0; k_idx < GRID_K; k_idx++) {
                        uint32_t a_elem_offset =
                            ((uint32_t)batch * GRID_M * GRID_K +
                             (uint32_t)m_idx * GRID_K +
                             (uint32_t)k_idx) * TILE_ELEMS;
                        uint32_t b_elem_offset =
                            ((uint32_t)batch * GRID_K * GRID_N +
                             (uint32_t)k_idx * GRID_N +
                             (uint32_t)n_idx) * TILE_ELEMS;

                        uint32_t a_view_offsets[1] = {a_elem_offset};
                        Tensor A_view = ext_A.view(tile_shapes, a_view_offsets);
                        uint32_t b_view_offsets[1] = {b_elem_offset};
                        Tensor B_view = ext_B.view(tile_shapes, b_view_offsets);
                        Tensor P = make_tensor(tile_shapes, 1, DataType::FLOAT32);

                        // P = A[m,k] @ B[k,n]
                        PTOParam params_gemm;
                        params_gemm.add_input(A_view);
                        params_gemm.add_input(B_view);
                        params_gemm.add_output(P);
                        PTO2TaskId t_gemm = pto2_rt_submit_aic_task(rt, FUNC_GEMM_TILE, params_gemm);

                        // C[m,n] += P
                        PTOParam params_add;
                        params_add.add_inout(C_view);
                        params_add.add_input(P);
                        PTO2TaskId t_add = pto2_rt_submit_aiv_task(rt, FUNC_TILE_ADD, params_add);

                        // gemm -> add: add reads P which gemm produces
                        pto2_rt_add_dependency(rt, t_gemm, t_add);
                        // K accumulation chain: previous add -> current add
                        if (has_last_add) {
                            pto2_rt_add_dependency(rt, last_add_task, t_add);
                        }

                        last_add_task = t_add;
                        has_last_add = true;
                    }
                }
            }
        }
    }

    LOG_INFO(rt, "[bgemm_orch] Submitted tasks for %d batches, %dx%d output tiles, %d K steps each",
                  BATCH, GRID_M, GRID_N, GRID_K);
}

}  // extern "C"
