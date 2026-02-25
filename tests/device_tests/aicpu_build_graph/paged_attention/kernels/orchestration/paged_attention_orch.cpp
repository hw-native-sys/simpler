/**
 * Paged Attention Orchestration - AICPU Build Graph Version
 *
 * Runs on AICPU. The framework has already allocated device memory for I/O
 * tensors and populated orch_args[] with device pointers and scalar values.
 *
 * orch_args[] layout (from TENSOR_ORDER in golden.py):
 *   orch_args[0]  = dev_query         (device ptr, bf16)
 *   orch_args[1]  = dev_key_cache     (device ptr, bf16)
 *   orch_args[2]  = dev_value_cache   (device ptr, bf16)
 *   orch_args[3]  = dev_block_table   (device ptr, int32)
 *   orch_args[4]  = dev_context_lens  (device ptr, int32)
 *   orch_args[5]  = dev_out           (device ptr, float32)
 *   orch_args[6]  = dev_config        (device ptr, int64)
 *   orch_args[7]  = query_nbytes      (scalar)
 *   orch_args[8]  = key_cache_nbytes  (scalar)
 *   orch_args[9]  = value_cache_nbytes(scalar)
 *   orch_args[10] = block_table_nbytes(scalar)
 *   orch_args[11] = context_lens_nbytes(scalar)
 *   orch_args[12] = out_nbytes        (scalar)
 *   orch_args[13] = config_nbytes     (scalar)
 *   orch_args[14] = element_count     (scalar, element count of first tensor)
 *
 * AICPU is on-device and can directly read dev_config, dev_context_lens,
 * and dev_block_table from HBM to determine graph structure.
 *
 * Supports production-scale paged attention with:
 *   Query: (batch, q_head_num, head_dim) bf16
 *   Key:   (total_blocks, block_size, kv_head_num, head_dim) bf16
 *   Value: (total_blocks, block_size, kv_head_num, head_dim) bf16
 *   Output: (batch * q_head_num, head_dim) float32
 *
 * Head tiling: q_tile_size = min(num_heads, 128)
 * GQA: kv_head_num can differ from q_head_num
 */

#include <cstdint>

#include "runtime.h"

#define FUNC_QK_MATMUL       0
#define FUNC_SOFTMAX_PREPARE 1
#define FUNC_PV_MATMUL       2
#define FUNC_ONLINE_UPDATE   3

namespace {

// orch_args[] index constants
constexpr int IDX_QUERY         = 0;
constexpr int IDX_KEY_CACHE     = 1;
constexpr int IDX_VALUE_CACHE   = 2;
constexpr int IDX_BLOCK_TABLE   = 3;
constexpr int IDX_CONTEXT_LENS  = 4;
constexpr int IDX_OUT           = 5;
constexpr int IDX_CONFIG        = 6;

inline int min_int(int a, int b) { return (a < b) ? a : b; }

}  // namespace

extern "C" int orchestration(Runtime* runtime) {
    if (runtime == nullptr) {
        return -1;
    }

    if (runtime->orch_argc < 14) {
        return -1;
    }

    const AicpuBuildApi& api = runtime->aicpu_build_api;
    if (api.add_task == nullptr || api.add_successor_conditional == nullptr ||
        api.publish_task == nullptr || api.device_malloc == nullptr) {
        return -1;
    }

    // Device pointers (already allocated and populated by the framework)
    uint8_t* dev_query       = reinterpret_cast<uint8_t*>(runtime->orch_args[IDX_QUERY]);
    uint8_t* dev_key_cache   = reinterpret_cast<uint8_t*>(runtime->orch_args[IDX_KEY_CACHE]);
    uint8_t* dev_value_cache = reinterpret_cast<uint8_t*>(runtime->orch_args[IDX_VALUE_CACHE]);
    int* dev_block_table     = reinterpret_cast<int*>(runtime->orch_args[IDX_BLOCK_TABLE]);
    int* dev_context_lens    = reinterpret_cast<int*>(runtime->orch_args[IDX_CONTEXT_LENS]);
    uint8_t* dev_out         = reinterpret_cast<uint8_t*>(runtime->orch_args[IDX_OUT]);
    int64_t* dev_config      = reinterpret_cast<int64_t*>(runtime->orch_args[IDX_CONFIG]);

    // Read config from device memory (AICPU can access HBM directly)
    int batch          = static_cast<int>(dev_config[0]);
    int num_heads      = static_cast<int>(dev_config[1]);
    int kv_head_num    = static_cast<int>(dev_config[2]);
    int head_dim       = static_cast<int>(dev_config[3]);
    int block_size     = static_cast<int>(dev_config[4]);
    int max_num_blocks = static_cast<int>(dev_config[5]);
    uint64_t scale_value_bits = static_cast<uint64_t>(dev_config[6]);

    int q_tile_size     = min_int(num_heads, 128);
    int num_head_tiles  = (num_heads + q_tile_size - 1) / q_tile_size;

    // Buffer sizes for per-block intermediates
    size_t sij_size    = static_cast<size_t>(q_tile_size) * block_size * sizeof(float);
    size_t pij_size    = static_cast<size_t>(q_tile_size) * block_size * sizeof(uint16_t);
    size_t mij_size    = static_cast<size_t>(q_tile_size) * sizeof(float);
    size_t lij_size    = mij_size;
    size_t oi_new_size = static_cast<size_t>(q_tile_size) * head_dim * sizeof(float);

    // Allocate per-block intermediate buffers on device (HBM)
    int total_buffers = batch * max_num_blocks;
    void** dev_sij_arr    = new void*[total_buffers];
    void** dev_pij_arr    = new void*[total_buffers];
    void** dev_mij_arr    = new void*[total_buffers];
    void** dev_lij_arr    = new void*[total_buffers];
    void** dev_oi_new_arr = new void*[total_buffers];

    for (int i = 0; i < total_buffers; i++) {
        dev_sij_arr[i]    = api.device_malloc(sij_size);
        dev_pij_arr[i]    = api.device_malloc(pij_size);
        dev_mij_arr[i]    = api.device_malloc(mij_size);
        dev_lij_arr[i]    = api.device_malloc(lij_size);
        dev_oi_new_arr[i] = api.device_malloc(oi_new_size);
    }

    // Per-(batch, head_tile) accumulators
    int total_accums = batch * num_head_tiles;
    size_t mi_size = static_cast<size_t>(q_tile_size) * sizeof(float);
    size_t li_size = mi_size;
    size_t oi_size = static_cast<size_t>(q_tile_size) * head_dim * sizeof(float);

    void** dev_mi_arr = new void*[total_accums];
    void** dev_li_arr = new void*[total_accums];
    void** dev_oi_arr = new void*[total_accums];

    for (int i = 0; i < total_accums; i++) {
        dev_mi_arr[i] = api.device_malloc(mi_size);
        dev_li_arr[i] = api.device_malloc(li_size);
        dev_oi_arr[i] = api.device_malloc(oi_size);
    }

    // Build the task graph
    for (int b_idx = 0; b_idx < batch; b_idx++) {
        int cur_seq = dev_context_lens[b_idx];
        int bn_this_batch = (cur_seq + block_size - 1) / block_size;

        for (int ht = 0; ht < num_head_tiles; ht++) {
            int cur_offset = ht * q_tile_size;

            // Query: (batch, q_head_num, head_dim) bf16
            uint8_t* qi_ptr = dev_query
                + static_cast<int64_t>(b_idx * num_heads + cur_offset) * head_dim * sizeof(uint16_t);

            // Output: (batch * q_head_num, head_dim) float32
            uint8_t* out_ptr = dev_out
                + static_cast<int64_t>(b_idx * num_heads + cur_offset) * head_dim * sizeof(float);

            // GQA: which kv_head this head tile maps to
            int kv_head_idx = cur_offset / (num_heads / kv_head_num);

            // Per-(batch, head_tile) accumulators
            int accum_idx = b_idx * num_head_tiles + ht;
            void* dev_mi = dev_mi_arr[accum_idx];
            void* dev_li = dev_li_arr[accum_idx];
            void* dev_oi = dev_oi_arr[accum_idx];

            int t_up_prev = -1;

            for (int bn = 0; bn < bn_this_batch; bn++) {
                int cur_block_idx = dev_block_table[b_idx * max_num_blocks + bn];
                int valid_len = min_int(block_size, cur_seq - bn * block_size);

                // Key: (total_blocks, block_size, kv_head_num, head_dim) bf16
                uint8_t* kj_ptr = dev_key_cache
                    + (static_cast<int64_t>(cur_block_idx) * block_size * kv_head_num + kv_head_idx)
                      * head_dim * sizeof(uint16_t);

                // Value: same layout as key
                uint8_t* vj_ptr = dev_value_cache
                    + (static_cast<int64_t>(cur_block_idx) * block_size * kv_head_num + kv_head_idx)
                      * head_dim * sizeof(uint16_t);

                int buf_idx = b_idx * max_num_blocks + bn;
                void* dev_sij    = dev_sij_arr[buf_idx];
                void* dev_pij    = dev_pij_arr[buf_idx];
                void* dev_mij    = dev_mij_arr[buf_idx];
                void* dev_lij    = dev_lij_arr[buf_idx];
                void* dev_oi_new = dev_oi_new_arr[buf_idx];

                // QK: qi(M, K) @ kj.T(K, N) -> sij(M, N)
                uint64_t qk_args[6] = {
                    reinterpret_cast<uint64_t>(qi_ptr),
                    reinterpret_cast<uint64_t>(kj_ptr),
                    reinterpret_cast<uint64_t>(dev_sij),
                    static_cast<uint64_t>(q_tile_size),
                    static_cast<uint64_t>(head_dim),
                    static_cast<uint64_t>(block_size)
                };
                int t_qk = api.add_task(runtime, qk_args, 6, FUNC_QK_MATMUL, CoreType::AIC, 0);
                if (t_qk < 0) return -1;

                // SF: scale, rowmax, exp, rowsum -> pij, mij, lij
                uint64_t sf_args[8] = {
                    reinterpret_cast<uint64_t>(dev_sij),
                    scale_value_bits,
                    reinterpret_cast<uint64_t>(dev_pij),
                    reinterpret_cast<uint64_t>(dev_mij),
                    reinterpret_cast<uint64_t>(dev_lij),
                    static_cast<uint64_t>(q_tile_size),
                    static_cast<uint64_t>(block_size),
                    static_cast<uint64_t>(valid_len)
                };
                int t_sf = api.add_task(runtime, sf_args, 8, FUNC_SOFTMAX_PREPARE, CoreType::AIV, 0);
                if (t_sf < 0) return -1;

                // PV: pij(M, K') @ vj(K', N') -> oi_new(M, N')
                uint64_t pv_args[6] = {
                    reinterpret_cast<uint64_t>(dev_pij),
                    reinterpret_cast<uint64_t>(vj_ptr),
                    reinterpret_cast<uint64_t>(dev_oi_new),
                    static_cast<uint64_t>(q_tile_size),
                    static_cast<uint64_t>(block_size),
                    static_cast<uint64_t>(head_dim)
                };
                int t_pv = api.add_task(runtime, pv_args, 6, FUNC_PV_MATMUL, CoreType::AIC, 0);
                if (t_pv < 0) return -1;

                // Dependencies: QK -> SF -> PV
                api.add_successor_conditional(runtime, t_qk, t_sf);
                api.add_successor_conditional(runtime, t_sf, t_pv);

                // Publish QK, SF, PV
                api.publish_task(runtime, t_qk);
                api.publish_task(runtime, t_sf);
                api.publish_task(runtime, t_pv);

                // Online Update: serialized across blocks
                int is_first = (bn == 0) ? 1 : 0;
                int is_last  = (bn == bn_this_batch - 1) ? 1 : 0;

                uint64_t up_args[11] = {
                    reinterpret_cast<uint64_t>(dev_mij),
                    reinterpret_cast<uint64_t>(dev_lij),
                    reinterpret_cast<uint64_t>(dev_oi_new),
                    reinterpret_cast<uint64_t>(dev_mi),
                    reinterpret_cast<uint64_t>(dev_li),
                    reinterpret_cast<uint64_t>(dev_oi),
                    static_cast<uint64_t>(is_first),
                    static_cast<uint64_t>(is_last),
                    reinterpret_cast<uint64_t>(out_ptr),
                    static_cast<uint64_t>(q_tile_size),
                    static_cast<uint64_t>(head_dim)
                };
                int t_up = api.add_task(runtime, up_args, 11, FUNC_ONLINE_UPDATE, CoreType::AIV, 0);
                if (t_up < 0) return -1;

                // UP depends on PV completing, and on previous UP (serialized)
                api.add_successor_conditional(runtime, t_pv, t_up);
                if (t_up_prev >= 0) {
                    api.add_successor_conditional(runtime, t_up_prev, t_up);
                }
                api.publish_task(runtime, t_up);

                t_up_prev = t_up;
            }
        }
    }

    delete[] dev_sij_arr;
    delete[] dev_pij_arr;
    delete[] dev_mij_arr;
    delete[] dev_lij_arr;
    delete[] dev_oi_new_arr;
    delete[] dev_mi_arr;
    delete[] dev_li_arr;
    delete[] dev_oi_arr;

    return 0;
}
