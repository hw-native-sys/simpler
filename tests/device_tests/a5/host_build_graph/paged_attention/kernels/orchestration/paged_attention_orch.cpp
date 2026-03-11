/**
 * Paged Attention Orchestration - Production Scale
 *
 * Supports production-scale paged attention with:
 *   Query: (batch, q_head_num, head_dim) bf16
 *   Key:   (total_blocks, block_size, kv_head_num, head_dim) bf16 (NOT transposed)
 *   Value: (total_blocks, block_size, kv_head_num, head_dim) bf16
 *   Output: (batch * q_head_num, head_dim) float32
 *
 * Head tiling: q_tile_size = min(num_heads, 128)
 * GQA: kv_head_num can differ from q_head_num
 */

#include "runtime.h"
#include <iostream>
#include <algorithm>
#include <cstring>

#define FUNC_QK_MATMUL       0
#define FUNC_SOFTMAX_PREPARE 1
#define FUNC_PV_MATMUL       2
#define FUNC_ONLINE_UPDATE   3

extern "C" {

int build_paged_attention_graph(Runtime* runtime, uint64_t* args, int arg_count) {
    if (arg_count < 14) {
        std::cerr << "Expected at least 14 args, got " << arg_count << '\n';
        return -1;
    }

    void* host_query = reinterpret_cast<void*>(args[0]);
    void* host_key_cache = reinterpret_cast<void*>(args[1]);
    void* host_value_cache = reinterpret_cast<void*>(args[2]);
    int* host_block_table = reinterpret_cast<int*>(args[3]);
    int* host_context_lens = reinterpret_cast<int*>(args[4]);
    void* host_out = reinterpret_cast<void*>(args[5]);
    int64_t* host_config = reinterpret_cast<int64_t*>(args[6]);

    size_t query_size = static_cast<size_t>(args[7]);
    size_t key_cache_size = static_cast<size_t>(args[8]);
    size_t value_cache_size = static_cast<size_t>(args[9]);
    size_t block_table_size = static_cast<size_t>(args[10]);
    size_t context_lens_size = static_cast<size_t>(args[11]);
    size_t out_size = static_cast<size_t>(args[12]);
    size_t config_size = static_cast<size_t>(args[13]);

    int batch = static_cast<int>(host_config[0]);
    int num_heads = static_cast<int>(host_config[1]);
    int kv_head_num = static_cast<int>(host_config[2]);
    int head_dim = static_cast<int>(host_config[3]);
    int block_size = static_cast<int>(host_config[4]);
    int max_num_blocks = static_cast<int>(host_config[5]);
    uint64_t scale_value_bits = static_cast<uint64_t>(host_config[6]);

    int q_tile_size = std::min(num_heads, 128);
    int num_head_tiles = (num_heads + q_tile_size - 1) / q_tile_size;

    std::cout << "\n=== build_paged_attention_graph ===" << '\n';
    std::cout << "batch=" << batch << ", num_heads=" << num_heads
              << ", kv_head_num=" << kv_head_num << ", head_dim=" << head_dim << '\n';
    std::cout << "block_size=" << block_size << ", max_num_blocks=" << max_num_blocks << '\n';
    std::cout << "q_tile_size=" << q_tile_size << ", num_head_tiles=" << num_head_tiles << '\n';

    // Allocate device memory for inputs/outputs
    void* dev_query = runtime->host_api.device_malloc(query_size);
    void* dev_key_cache = runtime->host_api.device_malloc(key_cache_size);
    void* dev_value_cache = runtime->host_api.device_malloc(value_cache_size);
    void* dev_out = runtime->host_api.device_malloc(out_size);

    if (!dev_query || !dev_key_cache || !dev_value_cache || !dev_out) {
        std::cerr << "Error: Failed to allocate device memory\n";
        return -1;
    }

    runtime->host_api.copy_to_device(dev_query, host_query, query_size);
    runtime->host_api.copy_to_device(dev_key_cache, host_key_cache, key_cache_size);
    runtime->host_api.copy_to_device(dev_value_cache, host_value_cache, value_cache_size);
    runtime->record_tensor_pair(host_out, dev_out, out_size);

    // Buffer sizes depend on q_tile_size and block_size
    size_t sij_size    = static_cast<size_t>(q_tile_size) * block_size * sizeof(float);
    size_t pij_size    = static_cast<size_t>(q_tile_size) * block_size * sizeof(uint16_t);
    size_t mij_size    = static_cast<size_t>(q_tile_size) * sizeof(float);
    size_t lij_size    = mij_size;
    size_t oi_new_size = static_cast<size_t>(q_tile_size) * head_dim * sizeof(float);

    // Per-batch-per-block intermediate buffers
    int total_buffers = batch * max_num_blocks;
    void** dev_sij_arr    = new void*[total_buffers];
    void** dev_pij_arr    = new void*[total_buffers];
    void** dev_mij_arr    = new void*[total_buffers];
    void** dev_lij_arr    = new void*[total_buffers];
    void** dev_oi_new_arr = new void*[total_buffers];

    for (int i = 0; i < total_buffers; i++) {
        dev_sij_arr[i]    = runtime->host_api.device_malloc(sij_size);
        dev_pij_arr[i]    = runtime->host_api.device_malloc(pij_size);
        dev_mij_arr[i]    = runtime->host_api.device_malloc(mij_size);
        dev_lij_arr[i]    = runtime->host_api.device_malloc(lij_size);
        dev_oi_new_arr[i] = runtime->host_api.device_malloc(oi_new_size);
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
        dev_mi_arr[i] = runtime->host_api.device_malloc(mi_size);
        dev_li_arr[i] = runtime->host_api.device_malloc(li_size);
        dev_oi_arr[i] = runtime->host_api.device_malloc(oi_size);
    }

    std::cout << "Allocated " << total_buffers << " per-block buffers\n";
    std::cout << "Allocated " << total_accums << " per-(batch,head_tile) accumulators\n";

    int total_tasks = 0;

    for (int b_idx = 0; b_idx < batch; b_idx++) {
        int cur_seq = host_context_lens[b_idx];
        int bn_this_batch = (cur_seq + block_size - 1) / block_size;

        for (int ht = 0; ht < num_head_tiles; ht++) {
            int cur_offset = ht * q_tile_size;

            // Query: (batch, q_head_num, head_dim) bf16
            // qi points to heads [cur_offset .. cur_offset+q_tile_size) for batch b_idx
            uint8_t* qi_ptr = reinterpret_cast<uint8_t*>(dev_query)
                + static_cast<int64_t>(b_idx * num_heads + cur_offset) * head_dim * sizeof(uint16_t);

            // Output: (batch * q_head_num, head_dim) float32
            uint8_t* out_ptr = reinterpret_cast<uint8_t*>(dev_out)
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
                int cur_block_idx = host_block_table[b_idx * max_num_blocks + bn];
                int valid_len = std::min(block_size, cur_seq - bn * block_size);

                // Key: (total_blocks, block_size, kv_head_num, head_dim) bf16
                // Stride to block: cur_block_idx * (block_size * kv_head_num * head_dim)
                // Then offset to kv_head: kv_head_idx * head_dim (within each token row)
                // But since we want contiguous (block_size, head_dim), and kv_head_num=1 makes it simple:
                uint8_t* kj_ptr = reinterpret_cast<uint8_t*>(dev_key_cache)
                    + (static_cast<int64_t>(cur_block_idx) * block_size * kv_head_num + kv_head_idx)
                      * head_dim * sizeof(uint16_t);

                // Value: (total_blocks, block_size, kv_head_num, head_dim) bf16 - same layout as key
                uint8_t* vj_ptr = reinterpret_cast<uint8_t*>(dev_value_cache)
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
                int t_qk = runtime->add_task(qk_args, 6, FUNC_QK_MATMUL, CoreType::AIC);
                total_tasks++;

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
                int t_sf = runtime->add_task(sf_args, 8, FUNC_SOFTMAX_PREPARE, CoreType::AIV);
                total_tasks++;

                // PV: pij(M, K') @ vj(K', N') -> oi_new(M, N')
                uint64_t pv_args[6] = {
                    reinterpret_cast<uint64_t>(dev_pij),
                    reinterpret_cast<uint64_t>(vj_ptr),
                    reinterpret_cast<uint64_t>(dev_oi_new),
                    static_cast<uint64_t>(q_tile_size),
                    static_cast<uint64_t>(block_size),
                    static_cast<uint64_t>(head_dim)
                };
                int t_pv = runtime->add_task(pv_args, 6, FUNC_PV_MATMUL, CoreType::AIC);
                total_tasks++;

                runtime->add_successor(t_qk, t_sf);
                runtime->add_successor(t_sf, t_pv);

                // Online Update: serialized across blocks (each depends on previous)
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
                int t_up = runtime->add_task(up_args, 11, FUNC_ONLINE_UPDATE, CoreType::AIV);
                total_tasks++;

                runtime->add_successor(t_pv, t_up);
                if (t_up_prev >= 0) {
                    runtime->add_successor(t_up_prev, t_up);
                }
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

    std::cout << "Created " << total_tasks << " tasks\n";
    runtime->print_runtime();

    return 0;
}

}
