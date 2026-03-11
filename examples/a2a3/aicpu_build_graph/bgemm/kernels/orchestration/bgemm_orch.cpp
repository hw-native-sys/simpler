/**
 * BGEMM Orchestration Function (AICPU Build Graph Runtime)
 *
 * Builds the task graph for tiled matrix multiplication: C = A @ B
 *
 * Configuration:
 *   - Tile size: 64 x 64
 *   - Grid: 4 x 4 x 4 (GRID_M x GRID_K x GRID_N)
 *
 * Memory layout (tile-first):
 *   A: [BATCH, GRID_M, GRID_K, TILE_M, TILE_K]
 *   B: [BATCH, GRID_K, GRID_N, TILE_K, TILE_N]
 *   C: [BATCH, GRID_M, GRID_N, TILE_M, TILE_N]
 *
 * Task graph per output tile:
 *   for k in [0, GRID_K):
 *     P = A[m,k] @ B[k,n]    (gemm_tile on Cube core)
 *     C[m,n] = C[m,n] + P    (tile_add on Vector core)
 */

#include <cstdint>

#include "runtime.h"

namespace {
constexpr int TILE = 64;
constexpr int GRID_M = 4;
constexpr int GRID_K = 4;
constexpr int GRID_N = 4;
constexpr int BATCH = 1;

constexpr size_t TILE_BYTES = TILE * TILE * sizeof(float);
constexpr int NUM_P_BUFFERS = BATCH * GRID_M * GRID_N;

constexpr int DEV_A = 0;
constexpr int DEV_B = 1;
constexpr int DEV_C = 2;
constexpr int ARG_SIZE = 2;
}  // namespace

extern "C" int orchestration(Runtime* runtime) {
    if (runtime == nullptr) {
        return -1;
    }

    if (runtime->orch_argc < ARG_SIZE + 1) {
        return -1;
    }

    const uint64_t dev_A = runtime->orch_args[DEV_A];
    const uint64_t dev_B = runtime->orch_args[DEV_B];
    const uint64_t dev_C = runtime->orch_args[DEV_C];

    if (dev_A == 0 || dev_B == 0 || dev_C == 0) {
        return -1;
    }

    const AicpuBuildApi& api = runtime->aicpu_build_api;
    if (api.add_task == nullptr || api.add_successor_conditional == nullptr ||
        api.publish_task == nullptr || api.device_malloc == nullptr) {
        return -1;
    }

    // Allocate intermediate P buffers (one per C tile)
    void* dev_P[NUM_P_BUFFERS];
    for (int i = 0; i < NUM_P_BUFFERS; i++) {
        dev_P[i] = api.device_malloc(TILE_BYTES);
        if (dev_P[i] == nullptr) {
            return -1;
        }
    }

    // Track last add task for each C tile (for K accumulation dependency)
    int last_add_task[NUM_P_BUFFERS];
    for (int i = 0; i < NUM_P_BUFFERS; i++) {
        last_add_task[i] = -1;
    }

    // Build task graph: 4-level tiling loop
    for (int batch = 0; batch < BATCH; batch++) {
        for (int m_idx = 0; m_idx < GRID_M; m_idx++) {
            for (int n_idx = 0; n_idx < GRID_N; n_idx++) {
                for (int k_idx = 0; k_idx < GRID_K; k_idx++) {
                    // Calculate tile offsets
                    size_t A_offset = (batch * GRID_M * GRID_K + m_idx * GRID_K + k_idx) * TILE_BYTES;
                    size_t B_offset = (batch * GRID_K * GRID_N + k_idx * GRID_N + n_idx) * TILE_BYTES;
                    size_t C_offset = (batch * GRID_M * GRID_N + m_idx * GRID_N + n_idx) * TILE_BYTES;

                    int c_tile_idx = batch * GRID_M * GRID_N + m_idx * GRID_N + n_idx;

                    if (k_idx == 0) {
                        // First k iteration: C[m,n] = A[m,0] @ B[0,n]
                        // Write gemm result directly to C (no accumulation needed)
                        uint64_t args_gemm[6];
                        args_gemm[0] = dev_A + A_offset;
                        args_gemm[1] = dev_B + B_offset;
                        args_gemm[2] = dev_C + C_offset;  // Write directly to C
                        args_gemm[3] = TILE;
                        args_gemm[4] = TILE;
                        args_gemm[5] = TILE;
                        int t_gemm = api.add_task(runtime, args_gemm, 6, 0, CoreType::AIC, 0);
                        if (t_gemm < 0) return -1;

                        api.publish_task(runtime, t_gemm);
                        last_add_task[c_tile_idx] = t_gemm;
                    } else {
                        // Subsequent k iterations: C[m,n] = C[m,n] + A[m,k] @ B[k,n]
                        // Task 1: P = A[m,k] @ B[k,n] (gemm_tile on Cube core)
                        uint64_t args_gemm[6];
                        args_gemm[0] = dev_A + A_offset;
                        args_gemm[1] = dev_B + B_offset;
                        args_gemm[2] = reinterpret_cast<uint64_t>(dev_P[c_tile_idx]);
                        args_gemm[3] = TILE;
                        args_gemm[4] = TILE;
                        args_gemm[5] = TILE;
                        int t_gemm = api.add_task(runtime, args_gemm, 6, 0, CoreType::AIC, 0);
                        if (t_gemm < 0) return -1;

                        // Task 2: C[m,n] = C[m,n] + P (tile_add on Vector core)
                        uint64_t args_add[5];
                        args_add[0] = dev_C + C_offset;
                        args_add[1] = reinterpret_cast<uint64_t>(dev_P[c_tile_idx]);
                        args_add[2] = dev_C + C_offset;
                        args_add[3] = TILE;
                        args_add[4] = TILE;
                        int t_add = api.add_task(runtime, args_add, 5, 1, CoreType::AIV, 0);
                        if (t_add < 0) return -1;

                        // Dependency: gemm must complete before add
                        api.add_successor_conditional(runtime, t_gemm, t_add);

                        // Dependency: previous task must complete before current gemm
                        if (last_add_task[c_tile_idx] >= 0) {
                            api.add_successor_conditional(runtime, last_add_task[c_tile_idx], t_gemm);
                        }

                        // Publish tasks after all dependencies are set
                        api.publish_task(runtime, t_gemm);
                        api.publish_task(runtime, t_add);

                        last_add_task[c_tile_idx] = t_add;
                    }
                }
            }
        }
    }

    if (runtime->kernel_addrs[0] == 0 || runtime->kernel_addrs[1] == 0) {
        return -1;
    }

    return 0;
}
