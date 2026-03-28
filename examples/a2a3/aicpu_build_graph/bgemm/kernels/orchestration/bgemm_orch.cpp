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
 *
 * Arg layout: [A, B, C]  — shape/dtype/size in TaskArg metadata
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

extern "C" {

__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(TaskArg* orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 3,
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(PTO2Runtime* rt, TaskArg* orch_args, int orch_thread_num, int orch_thread_index) {
    (void)orch_thread_num;
    (void)orch_thread_index;

    Tensor ext_A = from_task_arg(orch_args[0]);
    Tensor ext_B = from_task_arg(orch_args[1]);
    Tensor ext_C = from_task_arg(orch_args[2]);

    LOG_INFO(rt, "[bgemm_orch] Grid: %dx%dx%d, Batch: %d, Tile: %d",
                  GRID_M, GRID_K, GRID_N, BATCH, TILE);

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
                        Arg args_gemm;
                        args_gemm.add_input(A_view);
                        args_gemm.add_input(B_view);
                        args_gemm.add_output(P);
                        PTO2TaskId t_gemm = pto2_rt_submit_aic_task(rt, FUNC_GEMM_TILE, args_gemm);

                        // C[m,n] += P
                        Arg args_add;
                        args_add.add_inout(C_view);
                        args_add.add_input(P);
                        PTO2TaskId t_add = pto2_rt_submit_aiv_task(rt, FUNC_TILE_ADD, args_add);

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
