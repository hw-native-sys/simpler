/**
 * Paged Attention Orchestration Function (Multi-Head per Task)
 *
 * Each task processes ALL heads for a given (batch, block).
 * This dramatically reduces task count and improves NPU utilization.
 *
 * Task count: batch * block_num * 4  (instead of batch * num_heads * block_num * 4)
 *
 * Parallelism:
 *   - Different batches run in parallel
 *   - Different blocks: QK->SF->PV chains run in parallel
 *   - Only UP within same batch is serialized (accumulator dependency)
 *
 * Buffer allocation:
 *   - Per-batch-per-block: sij, pij, mij, lij, oi_new  (all heads)
 *   - Per-batch accumulators: mi, li, oi  (all heads)
 */

#include "runtime.h"
#include <iostream>
#include <cstring>

#define FUNC_QK_MATMUL       0
#define FUNC_SOFTMAX_PREPARE 1
#define FUNC_PV_MATMUL       2
#define FUNC_ONLINE_UPDATE   3

#define MAX_BLOCKS 64

extern "C" {

int build_paged_attention_graph(Runtime* runtime, uint64_t* args, int arg_count) {
    // Expected args layout from CodeRunner:
    // [ptr_query, ptr_key_cache, ptr_value_cache, ptr_block_table, ptr_context_lens, ptr_out, ptr_config,
    //  size_query, size_key_cache, size_value_cache, size_block_table, size_context_lens, size_out, size_config,
    //  count]
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

    // Extract config parameters from config array
    // config = [batch, num_heads, kv_head_num, head_dim, block_size, block_num, scale_bits]
    int batch = static_cast<int>(host_config[0]);
    int num_heads = static_cast<int>(host_config[1]);
    int kv_head_num = static_cast<int>(host_config[2]);
    int head_dim = static_cast<int>(host_config[3]);
    int block_size = static_cast<int>(host_config[4]);
    int block_num = static_cast<int>(host_config[5]);
    uint64_t scale_value_bits = static_cast<uint64_t>(host_config[6]);

    std::cout << "\n=== build_paged_attention_graph (multi-head per task) ===" << '\n';
    std::cout << "batch=" << batch << ", num_heads=" << num_heads
              << ", kv_head_num=" << kv_head_num << ", head_dim=" << head_dim << '\n';
    std::cout << "block_size=" << block_size << ", block_num=" << block_num << '\n';

    // Allocate device memory for inputs
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

    // Buffer sizes (all heads packed together)
    size_t sij_size    = num_heads * block_size * sizeof(float);   // (num_heads, block_size)
    size_t scalar_size = num_heads * sizeof(float);                // (num_heads,)
    size_t vec_size    = num_heads * head_dim * sizeof(float);     // (num_heads, head_dim)

    // Per-batch-per-block intermediate buffers
    // Index: [b_idx * block_num + bn]
    int total_buffers = batch * block_num;
    void** dev_sij_arr    = new void*[total_buffers];
    void** dev_pij_arr    = new void*[total_buffers];
    void** dev_mij_arr    = new void*[total_buffers];
    void** dev_lij_arr    = new void*[total_buffers];
    void** dev_oi_new_arr = new void*[total_buffers];

    for (int i = 0; i < total_buffers; i++) {
        dev_sij_arr[i]    = runtime->host_api.device_malloc(sij_size);
        dev_pij_arr[i]    = runtime->host_api.device_malloc(sij_size);
        dev_mij_arr[i]    = runtime->host_api.device_malloc(scalar_size);
        dev_lij_arr[i]    = runtime->host_api.device_malloc(scalar_size);
        dev_oi_new_arr[i] = runtime->host_api.device_malloc(vec_size);
    }

    // Per-batch accumulators (mi, li, oi) — all heads packed
    void** dev_mi_arr = new void*[batch];
    void** dev_li_arr = new void*[batch];
    void** dev_oi_arr = new void*[batch];

    for (int i = 0; i < batch; i++) {
        dev_mi_arr[i] = runtime->host_api.device_malloc(scalar_size);
        dev_li_arr[i] = runtime->host_api.device_malloc(scalar_size);
        dev_oi_arr[i] = runtime->host_api.device_malloc(vec_size);
    }

    std::cout << "Allocated " << total_buffers << " per-batch-per-block buffers (all heads packed)\n";
    std::cout << "Allocated " << batch << " per-batch accumulators (all heads packed)\n";

    int total_tasks = 0;

    for (int b_idx = 0; b_idx < batch; b_idx++) {
        int cur_seq = host_context_lens[b_idx];
        int bn_this_batch = (cur_seq + block_size - 1) / block_size;

        // Query pointer for this batch (all heads): (num_heads, head_dim)
        float* qi_ptr = reinterpret_cast<float*>(dev_query)
                      + b_idx * num_heads * head_dim;

        // Output pointer for this batch (all heads): (num_heads, head_dim)
        float* out_ptr = reinterpret_cast<float*>(dev_out)
                       + b_idx * num_heads * head_dim;

        // Per-batch accumulators
        void* dev_mi = dev_mi_arr[b_idx];
        void* dev_li = dev_li_arr[b_idx];
        void* dev_oi = dev_oi_arr[b_idx];

        int t_pv_arr[MAX_BLOCKS];

        // Phase 1: Create parallel QK -> SF -> PV chains for all blocks
        for (int bn = 0; bn < bn_this_batch; bn++) {
            int cur_block_idx = host_block_table[b_idx * block_num + bn];

            int valid_len = (bn == bn_this_batch - 1)
                          ? (cur_seq - bn * block_size)
                          : block_size;

            // K/V block base pointer: (block_size, kv_head_num, head_dim)
            float* kj_ptr = reinterpret_cast<float*>(dev_key_cache)
                          + cur_block_idx * block_size * kv_head_num * head_dim;
            float* vj_ptr = reinterpret_cast<float*>(dev_value_cache)
                          + cur_block_idx * block_size * kv_head_num * head_dim;

            // Per-batch-per-block buffers
            int buf_idx = b_idx * block_num + bn;
            void* dev_sij    = dev_sij_arr[buf_idx];
            void* dev_pij    = dev_pij_arr[buf_idx];
            void* dev_mij    = dev_mij_arr[buf_idx];
            void* dev_lij    = dev_lij_arr[buf_idx];
            void* dev_oi_new = dev_oi_new_arr[buf_idx];

            // QK MatMul (AIC) — all heads
            uint64_t qk_args[7] = {
                reinterpret_cast<uint64_t>(qi_ptr),
                reinterpret_cast<uint64_t>(kj_ptr),
                reinterpret_cast<uint64_t>(dev_sij),
                static_cast<uint64_t>(num_heads),
                static_cast<uint64_t>(kv_head_num),
                static_cast<uint64_t>(block_size),
                static_cast<uint64_t>(head_dim)
            };
            int t_qk = runtime->add_task(qk_args, 7, FUNC_QK_MATMUL, CoreType::AIC);
            total_tasks++;

            // Softmax Prepare (AIV) — all heads
            uint64_t sf_args[8] = {
                reinterpret_cast<uint64_t>(dev_sij),
                scale_value_bits,
                reinterpret_cast<uint64_t>(dev_pij),
                reinterpret_cast<uint64_t>(dev_mij),
                reinterpret_cast<uint64_t>(dev_lij),
                static_cast<uint64_t>(num_heads),
                static_cast<uint64_t>(block_size),
                static_cast<uint64_t>(valid_len)
            };
            int t_sf = runtime->add_task(sf_args, 8, FUNC_SOFTMAX_PREPARE, CoreType::AIV);
            total_tasks++;

            // PV MatMul (AIC) — all heads
            uint64_t pv_args[7] = {
                reinterpret_cast<uint64_t>(dev_pij),
                reinterpret_cast<uint64_t>(vj_ptr),
                reinterpret_cast<uint64_t>(dev_oi_new),
                static_cast<uint64_t>(num_heads),
                static_cast<uint64_t>(kv_head_num),
                static_cast<uint64_t>(block_size),
                static_cast<uint64_t>(head_dim)
            };
            int t_pv = runtime->add_task(pv_args, 7, FUNC_PV_MATMUL, CoreType::AIC);
            total_tasks++;

            // Dependencies: QK -> SF -> PV
            runtime->add_successor(t_qk, t_sf);
            runtime->add_successor(t_sf, t_pv);

            t_pv_arr[bn] = t_pv;
        }

        // Phase 2: Create serialized UP chain (within this batch)
        int t_up_prev = -1;
        for (int bn = 0; bn < bn_this_batch; bn++) {
            int is_first = (bn == 0) ? 1 : 0;
            int is_last  = (bn == bn_this_batch - 1) ? 1 : 0;

            int buf_idx = b_idx * block_num + bn;
            void* dev_mij    = dev_mij_arr[buf_idx];
            void* dev_lij    = dev_lij_arr[buf_idx];
            void* dev_oi_new = dev_oi_new_arr[buf_idx];

            // Online Update (AIV) — all heads
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
                static_cast<uint64_t>(num_heads),
                static_cast<uint64_t>(head_dim)
            };
            int t_up = runtime->add_task(up_args, 11, FUNC_ONLINE_UPDATE, CoreType::AIV);
            total_tasks++;

            // UP[bn] depends on PV[bn]
            runtime->add_successor(t_pv_arr[bn], t_up);

            // UP[bn] depends on UP[bn-1] (within same batch)
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
