/**
 * BGEMM Orchestration Function (Host Build Graph Runtime)
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

#include "runtime.h"
#include <iostream>
#include <vector>

extern "C" {

constexpr int TILE = 64;
constexpr int GRID_M = 4;
constexpr int GRID_K = 4;
constexpr int GRID_N = 4;
constexpr int BATCH = 1;

constexpr size_t TILE_BYTES = TILE * TILE * sizeof(float);

int build_bgemm_graph(Runtime* runtime, uint64_t* args, int arg_count) {
    if (arg_count < 6) {
        std::cerr << "build_bgemm_graph: Expected at least 6 args, got " << arg_count << '\n';
        return -1;
    }

    void* host_A = reinterpret_cast<void*>(args[0]);
    void* host_B = reinterpret_cast<void*>(args[1]);
    void* host_C = reinterpret_cast<void*>(args[2]);
    size_t size_A = static_cast<size_t>(args[3]);
    size_t size_B = static_cast<size_t>(args[4]);
    size_t size_C = static_cast<size_t>(args[5]);

    std::cout << "\n=== build_bgemm_graph ===" << '\n';
    std::cout << "Grid: " << GRID_M << " x " << GRID_K << " x " << GRID_N << '\n';

    // Allocate device memory and copy inputs
    void* dev_A = runtime->host_api.device_malloc(size_A);
    if (!dev_A) return -1;
    runtime->host_api.copy_to_device(dev_A, host_A, size_A);

    void* dev_B = runtime->host_api.device_malloc(size_B);
    if (!dev_B) {
        runtime->host_api.device_free(dev_A);
        return -1;
    }
    runtime->host_api.copy_to_device(dev_B, host_B, size_B);

    void* dev_C = runtime->host_api.device_malloc(size_C);
    if (!dev_C) {
        runtime->host_api.device_free(dev_A);
        runtime->host_api.device_free(dev_B);
        return -1;
    }
    runtime->host_api.copy_to_device(dev_C, host_C, size_C);
    runtime->record_tensor_pair(host_C, dev_C, size_C);

    // Allocate intermediate P buffers (one per C tile)
    constexpr int NUM_P_BUFFERS = BATCH * GRID_M * GRID_N;
    std::vector<void*> dev_P(NUM_P_BUFFERS, nullptr);
    for (int i = 0; i < NUM_P_BUFFERS; i++) {
        dev_P[i] = runtime->host_api.device_malloc(TILE_BYTES);
        if (!dev_P[i]) {
            for (int j = 0; j < i; j++) {
                runtime->host_api.device_free(dev_P[j]);
            }
            runtime->host_api.device_free(dev_A);
            runtime->host_api.device_free(dev_B);
            runtime->host_api.device_free(dev_C);
            return -1;
        }
    }

    // Track last add task for each C tile (for K accumulation dependency)
    std::vector<int> last_add_task(BATCH * GRID_M * GRID_N, -1);

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

                    // Task 1: P = A[m,k] @ B[k,n] (gemm_tile on Cube core)
                    uint64_t args_gemm[6];
                    args_gemm[0] = reinterpret_cast<uint64_t>(static_cast<char*>(dev_A) + A_offset);
                    args_gemm[1] = reinterpret_cast<uint64_t>(static_cast<char*>(dev_B) + B_offset);
                    args_gemm[2] = reinterpret_cast<uint64_t>(dev_P[c_tile_idx]);
                    args_gemm[3] = TILE;
                    args_gemm[4] = TILE;
                    args_gemm[5] = TILE;
                    int t_gemm = runtime->add_task(args_gemm, 6, 0, CoreType::AIC);

                    // Task 2: C[m,n] = C[m,n] + P (tile_add on Vector core)
                    uint64_t args_add[5];
                    args_add[0] = reinterpret_cast<uint64_t>(static_cast<char*>(dev_C) + C_offset);
                    args_add[1] = reinterpret_cast<uint64_t>(dev_P[c_tile_idx]);
                    args_add[2] = reinterpret_cast<uint64_t>(static_cast<char*>(dev_C) + C_offset);
                    args_add[3] = TILE;
                    args_add[4] = TILE;
                    int t_add = runtime->add_task(args_add, 5, 1, CoreType::AIV);

                    // Dependency: gemm must complete before add
                    runtime->add_successor(t_gemm, t_add);

                    // Dependency: previous add must complete before current gemm (K accumulation)
                    if (last_add_task[c_tile_idx] >= 0) {
                        runtime->add_successor(last_add_task[c_tile_idx], t_gemm);
                    }
                    last_add_task[c_tile_idx] = t_add;
                }
            }
        }
    }

    std::cout << "Created " << runtime->get_task_count() << " tasks\n";
    return 0;
}

}  // extern "C"
