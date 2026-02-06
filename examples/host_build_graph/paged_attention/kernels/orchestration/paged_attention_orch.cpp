/**
 * Paged Attention Orchestration Function - 16x16 Version
 *
 * Simplified for 16x16 framework-generated matmul kernels.
 * Each block processes a single 16x16 matmul operation.
 *
 * Memory Layout:
 *   Query: (batch, 16, 16) - one 16x16 tile per batch
 *   Key:   (total_blocks, 16, 16) - stored as K^T for direct matmul
 *   Value: (total_blocks, 16, 16) - direct format
 */

#include "runtime.h"
#include <iostream>
#include <cstring>

#define FUNC_QK_MATMUL       0
#define FUNC_SOFTMAX_PREPARE 1
#define FUNC_PV_MATMUL       2
#define FUNC_ONLINE_UPDATE   3

#define MAX_BLOCKS 64

// Fixed 16x16 tile dimensions
constexpr int kTileSize = 16;
constexpr int kTileElements = kTileSize * kTileSize;  // 256

extern "C" {

int build_paged_attention_graph(Runtime* runtime, uint64_t* args, int arg_count) {
    if (arg_count < 15) {
        std::cerr << "Expected at least 15 args, got " << arg_count << '\n';
        return -1;
    }

    // Extract pointers (first 7)
    void* host_query = reinterpret_cast<void*>(args[0]);
    void* host_key_cache = reinterpret_cast<void*>(args[1]);
    void* host_value_cache = reinterpret_cast<void*>(args[2]);
    int* host_block_table = reinterpret_cast<int*>(args[3]);
    int* host_context_lens = reinterpret_cast<int*>(args[4]);
    void* host_out = reinterpret_cast<void*>(args[5]);
    int64_t* host_config = reinterpret_cast<int64_t*>(args[6]);

    // Extract sizes (next 7)
    size_t query_size = static_cast<size_t>(args[7]);
    size_t key_cache_size = static_cast<size_t>(args[8]);
    size_t value_cache_size = static_cast<size_t>(args[9]);
    size_t block_table_size = static_cast<size_t>(args[10]);
    size_t context_lens_size = static_cast<size_t>(args[11]);
    size_t out_size = static_cast<size_t>(args[12]);
    size_t config_size = static_cast<size_t>(args[13]);

    // Extract config parameters
    int batch = static_cast<int>(host_config[0]);
    int num_heads = static_cast<int>(host_config[1]);
    int kv_head_num = static_cast<int>(host_config[2]);
    int head_dim = static_cast<int>(host_config[3]);
    int block_size = static_cast<int>(host_config[4]);
    int block_num = static_cast<int>(host_config[5]);
    uint64_t scale_value_bits = static_cast<uint64_t>(host_config[6]);

    std::cout << "\n=== build_paged_attention_graph (16x16 framework version) ===" << '\n';
    std::cout << "batch=" << batch << ", num_heads=" << num_heads
              << ", kv_head_num=" << kv_head_num << ", head_dim=" << head_dim << '\n';
    std::cout << "block_size=" << block_size << ", block_num=" << block_num << '\n';

    // Allocate device memory
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

    // Buffer sizes for 16x16 tiles
    size_t tile_size = kTileElements * sizeof(float);  // 256 * 4 = 1024 bytes
    size_t scalar_size = kTileSize * sizeof(float);    // 16 * 4 = 64 bytes

    // Per-batch-per-block intermediate buffers
    int total_buffers = batch * block_num;
    void** dev_sij_arr    = new void*[total_buffers];
    void** dev_pij_arr    = new void*[total_buffers];
    void** dev_mij_arr    = new void*[total_buffers];
    void** dev_lij_arr    = new void*[total_buffers];
    void** dev_oi_new_arr = new void*[total_buffers];

    for (int i = 0; i < total_buffers; i++) {
        dev_sij_arr[i]    = runtime->host_api.device_malloc(tile_size);
        dev_pij_arr[i]    = runtime->host_api.device_malloc(tile_size);
        dev_mij_arr[i]    = runtime->host_api.device_malloc(scalar_size);
        dev_lij_arr[i]    = runtime->host_api.device_malloc(scalar_size);
        dev_oi_new_arr[i] = runtime->host_api.device_malloc(tile_size);
    }

    // Per-batch accumulators
    void** dev_mi_arr = new void*[batch];
    void** dev_li_arr = new void*[batch];
    void** dev_oi_arr = new void*[batch];

    for (int i = 0; i < batch; i++) {
        dev_mi_arr[i] = runtime->host_api.device_malloc(scalar_size);
        dev_li_arr[i] = runtime->host_api.device_malloc(scalar_size);
        dev_oi_arr[i] = runtime->host_api.device_malloc(tile_size);
    }

    std::cout << "Allocated " << total_buffers << " per-batch-per-block buffers\n";
    std::cout << "Allocated " << batch << " per-batch accumulators\n";

    int total_tasks = 0;

    for (int b_idx = 0; b_idx < batch; b_idx++) {
        int cur_seq = host_context_lens[b_idx];
        int bn_this_batch = (cur_seq + block_size - 1) / block_size;

        // Query pointer: each batch has one 16x16 tile
        float* qi_ptr = reinterpret_cast<float*>(dev_query) + b_idx * kTileElements;

        // Output pointer: each batch has one 16x16 tile
        float* out_ptr = reinterpret_cast<float*>(dev_out) + b_idx * kTileElements;

        // Per-batch accumulators
        void* dev_mi = dev_mi_arr[b_idx];
        void* dev_li = dev_li_arr[b_idx];
        void* dev_oi = dev_oi_arr[b_idx];

        int t_pv_arr[MAX_BLOCKS];

        // Phase 1: Create parallel QK -> SF -> PV chains
        for (int bn = 0; bn < bn_this_batch; bn++) {
            int cur_block_idx = host_block_table[b_idx * block_num + bn];

            // Key/Value pointers: each block has one 16x16 tile
            float* kj_ptr = reinterpret_cast<float*>(dev_key_cache) + cur_block_idx * kTileElements;
            float* vj_ptr = reinterpret_cast<float*>(dev_value_cache) + cur_block_idx * kTileElements;

            int buf_idx = b_idx * block_num + bn;
            void* dev_sij    = dev_sij_arr[buf_idx];
            void* dev_pij    = dev_pij_arr[buf_idx];
            void* dev_mij    = dev_mij_arr[buf_idx];
            void* dev_lij    = dev_lij_arr[buf_idx];
            void* dev_oi_new = dev_oi_new_arr[buf_idx];

            // QK MatMul: Q (16x16) @ K^T (16x16) = S (16x16)
            uint64_t qk_args[3] = {
                reinterpret_cast<uint64_t>(qi_ptr),
                reinterpret_cast<uint64_t>(kj_ptr),
                reinterpret_cast<uint64_t>(dev_sij)
            };
            int t_qk = runtime->add_task(qk_args, 3, FUNC_QK_MATMUL, CoreType::AIC);
            total_tasks++;

            // Softmax Prepare
            uint64_t sf_args[5] = {
                reinterpret_cast<uint64_t>(dev_sij),
                scale_value_bits,
                reinterpret_cast<uint64_t>(dev_pij),
                reinterpret_cast<uint64_t>(dev_mij),
                reinterpret_cast<uint64_t>(dev_lij)
            };
            int t_sf = runtime->add_task(sf_args, 5, FUNC_SOFTMAX_PREPARE, CoreType::AIV);
            total_tasks++;

            // PV MatMul: P (16x16) @ V (16x16) = O (16x16)
            uint64_t pv_args[3] = {
                reinterpret_cast<uint64_t>(dev_pij),
                reinterpret_cast<uint64_t>(vj_ptr),
                reinterpret_cast<uint64_t>(dev_oi_new)
            };
            int t_pv = runtime->add_task(pv_args, 3, FUNC_PV_MATMUL, CoreType::AIC);
            total_tasks++;

            runtime->add_successor(t_qk, t_sf);
            runtime->add_successor(t_sf, t_pv);

            t_pv_arr[bn] = t_pv;
        }

        // Phase 2: Serialized Online Update chain
        int t_up_prev = -1;
        for (int bn = 0; bn < bn_this_batch; bn++) {
            int is_first = (bn == 0) ? 1 : 0;
            int is_last  = (bn == bn_this_batch - 1) ? 1 : 0;

            int buf_idx = b_idx * block_num + bn;
            void* dev_mij    = dev_mij_arr[buf_idx];
            void* dev_lij    = dev_lij_arr[buf_idx];
            void* dev_oi_new = dev_oi_new_arr[buf_idx];

            uint64_t up_args[9] = {
                reinterpret_cast<uint64_t>(dev_mij),
                reinterpret_cast<uint64_t>(dev_lij),
                reinterpret_cast<uint64_t>(dev_oi_new),
                reinterpret_cast<uint64_t>(dev_mi),
                reinterpret_cast<uint64_t>(dev_li),
                reinterpret_cast<uint64_t>(dev_oi),
                static_cast<uint64_t>(is_first),
                static_cast<uint64_t>(is_last),
                reinterpret_cast<uint64_t>(out_ptr)
            };
            int t_up = runtime->add_task(up_args, 9, FUNC_ONLINE_UPDATE, CoreType::AIV);
            total_tasks++;

            runtime->add_successor(t_pv_arr[bn], t_up);

            if (t_up_prev >= 0) {
                runtime->add_successor(t_up_prev, t_up);
            }

            t_up_prev = t_up;
        }
    }

    // Cleanup
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
