/**
 * Paged Attention Orchestration Function (Full Parallel)
 *
 * Parallelism:
 *   - Different batches run in parallel (independent buffers per batch)
 *   - Different heads run in parallel (independent buffers per head)
 *   - Different blocks within a head: QK→SF→PV run in parallel
 *   - Only UP within same (batch, head) is serialized (accumulator dependency)
 *
 * Buffer allocation:
 *   - Per-batch-per-head-per-block buffers: sij, pij, mij, lij, oi_new
 *   - Per-batch-per-head accumulators: mi, li, oi
 */

#include "runtime.h"
#include <iostream>

#define FUNC_QK_MATMUL       0
#define FUNC_SOFTMAX_PREPARE 1
#define FUNC_PV_MATMUL       2
#define FUNC_ONLINE_UPDATE   3

#define MAX_BLOCKS 64

extern "C" {

int build_paged_attention_graph(Runtime* runtime, uint64_t* args, int arg_count) {
    if (arg_count < 19) {
        std::cerr << "Expected 19 args, got " << arg_count << '\n';
        return -1;
    }

    void* host_query = reinterpret_cast<void*>(args[0]);
    void* host_key_cache = reinterpret_cast<void*>(args[1]);
    void* host_value_cache = reinterpret_cast<void*>(args[2]);
    int* host_block_table = reinterpret_cast<int*>(args[3]);
    int* host_context_lens = reinterpret_cast<int*>(args[4]);
    void* host_out = reinterpret_cast<void*>(args[5]);

    size_t query_size = static_cast<size_t>(args[6]);
    size_t key_cache_size = static_cast<size_t>(args[7]);
    size_t value_cache_size = static_cast<size_t>(args[8]);
    size_t out_size = static_cast<size_t>(args[11]);

    int batch = static_cast<int>(args[12]);
    int num_heads = static_cast<int>(args[13]);
    int kv_head_num = static_cast<int>(args[14]);
    int head_dim = static_cast<int>(args[15]);
    int block_size = static_cast<int>(args[16]);
    int block_num = static_cast<int>(args[17]);
    uint64_t scale_value_bits = args[18];

    int heads_per_kv = num_heads / kv_head_num;
    int q_tile = 1;

    std::cout << "\n=== build_paged_attention_graph (full parallel) ===" << '\n';
    std::cout << "batch=" << batch << ", num_heads=" << num_heads
              << ", head_dim=" << head_dim << '\n';
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

    // Buffer sizes
    size_t sij_size = q_tile * block_size * sizeof(float);
    size_t scalar_size = q_tile * sizeof(float);
    size_t vec_size = q_tile * head_dim * sizeof(float);

    // Per-batch-per-head-per-block intermediate buffers (for full parallelism)
    // Index: [(b_idx * num_heads + h_idx) * block_num + bn]
    int total_buffers = batch * num_heads * block_num;
    void** dev_sij_arr = new void*[total_buffers];
    void** dev_pij_arr = new void*[total_buffers];
    void** dev_mij_arr = new void*[total_buffers];
    void** dev_lij_arr = new void*[total_buffers];
    void** dev_oi_new_arr = new void*[total_buffers];

    for (int i = 0; i < total_buffers; i++) {
        dev_sij_arr[i] = runtime->host_api.device_malloc(sij_size);
        dev_pij_arr[i] = runtime->host_api.device_malloc(sij_size);
        dev_mij_arr[i] = runtime->host_api.device_malloc(scalar_size);
        dev_lij_arr[i] = runtime->host_api.device_malloc(scalar_size);
        dev_oi_new_arr[i] = runtime->host_api.device_malloc(vec_size);
    }

    // Per-batch-per-head accumulators (mi, li, oi)
    int total_accumulators = batch * num_heads;
    void** dev_mi_arr = new void*[total_accumulators];
    void** dev_li_arr = new void*[total_accumulators];
    void** dev_oi_arr = new void*[total_accumulators];

    for (int i = 0; i < total_accumulators; i++) {
        dev_mi_arr[i] = runtime->host_api.device_malloc(scalar_size);
        dev_li_arr[i] = runtime->host_api.device_malloc(scalar_size);
        dev_oi_arr[i] = runtime->host_api.device_malloc(vec_size);
    }

    std::cout << "Allocated " << total_buffers << " per-batch-per-head-per-block buffers\n";
    std::cout << "Allocated " << total_accumulators << " per-batch-per-head accumulators\n";

    int total_tasks = 0;

    for (int b_idx = 0; b_idx < batch; b_idx++) {
        int cur_seq = host_context_lens[b_idx];
        int bn_this_batch = (cur_seq + block_size - 1) / block_size;

        for (int h_idx = 0; h_idx < num_heads; h_idx++) {
            int kv_h_idx = h_idx / heads_per_kv;

            float* qi_ptr = reinterpret_cast<float*>(dev_query)
                          + (b_idx * num_heads + h_idx) * head_dim;

            float* out_ptr = reinterpret_cast<float*>(dev_out)
                           + (b_idx * num_heads + h_idx) * head_dim;

            // Per-batch-per-head accumulators
            int acc_idx = b_idx * num_heads + h_idx;
            void* dev_mi = dev_mi_arr[acc_idx];
            void* dev_li = dev_li_arr[acc_idx];
            void* dev_oi = dev_oi_arr[acc_idx];

            int t_pv_arr[MAX_BLOCKS];  // Store PV task IDs

            // Phase 1: Create parallel QK → SF → PV chains for all blocks
            for (int bn = 0; bn < bn_this_batch; bn++) {
                int cur_block_idx = host_block_table[b_idx * block_num + bn];

                int valid_len = (bn == bn_this_batch - 1)
                              ? (cur_seq - bn * block_size)
                              : block_size;

                float* kj_ptr = reinterpret_cast<float*>(dev_key_cache)
                              + cur_block_idx * block_size * kv_head_num * head_dim
                              + kv_h_idx * head_dim;
                float* vj_ptr = reinterpret_cast<float*>(dev_value_cache)
                              + cur_block_idx * block_size * kv_head_num * head_dim
                              + kv_h_idx * head_dim;

                // Per-batch-per-head-per-block buffers
                int buf_idx = (b_idx * num_heads + h_idx) * block_num + bn;
                void* dev_sij = dev_sij_arr[buf_idx];
                void* dev_pij = dev_pij_arr[buf_idx];
                void* dev_mij = dev_mij_arr[buf_idx];
                void* dev_lij = dev_lij_arr[buf_idx];
                void* dev_oi_new = dev_oi_new_arr[buf_idx];

                // QK MatMul (AIC)
                uint64_t qk_args[6] = {
                    reinterpret_cast<uint64_t>(qi_ptr),
                    reinterpret_cast<uint64_t>(kj_ptr),
                    reinterpret_cast<uint64_t>(dev_sij),
                    static_cast<uint64_t>(q_tile),
                    static_cast<uint64_t>(block_size),
                    static_cast<uint64_t>(head_dim)
                };
                int t_qk = runtime->add_task(qk_args, 6, FUNC_QK_MATMUL, 0);
                total_tasks++;

                // Softmax Prepare (AIV)
                uint64_t sf_args[8] = {
                    reinterpret_cast<uint64_t>(dev_sij),
                    scale_value_bits,
                    reinterpret_cast<uint64_t>(dev_pij),
                    reinterpret_cast<uint64_t>(dev_mij),
                    reinterpret_cast<uint64_t>(dev_lij),
                    static_cast<uint64_t>(q_tile),
                    static_cast<uint64_t>(block_size),
                    static_cast<uint64_t>(valid_len)
                };
                int t_sf = runtime->add_task(sf_args, 8, FUNC_SOFTMAX_PREPARE, 1);
                total_tasks++;

                // PV MatMul (AIC)
                uint64_t pv_args[6] = {
                    reinterpret_cast<uint64_t>(dev_pij),
                    reinterpret_cast<uint64_t>(vj_ptr),
                    reinterpret_cast<uint64_t>(dev_oi_new),
                    static_cast<uint64_t>(q_tile),
                    static_cast<uint64_t>(block_size),
                    static_cast<uint64_t>(head_dim)
                };
                int t_pv = runtime->add_task(pv_args, 6, FUNC_PV_MATMUL, 0);
                total_tasks++;

                // Dependencies: QK → SF → PV
                runtime->add_successor(t_qk, t_sf);
                runtime->add_successor(t_sf, t_pv);

                // No cross-batch or cross-head dependency! Each (batch, head) has independent buffers.

                t_pv_arr[bn] = t_pv;
            }

            // Phase 2: Create serialized UP chain (within this (batch, head) only)
            int t_up_prev = -1;
            for (int bn = 0; bn < bn_this_batch; bn++) {
                int is_first = (bn == 0) ? 1 : 0;
                int is_last  = (bn == bn_this_batch - 1) ? 1 : 0;

                int buf_idx = (b_idx * num_heads + h_idx) * block_num + bn;
                void* dev_mij = dev_mij_arr[buf_idx];
                void* dev_lij = dev_lij_arr[buf_idx];
                void* dev_oi_new = dev_oi_new_arr[buf_idx];

                // Online Update (AIV)
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
                    static_cast<uint64_t>(q_tile),
                    static_cast<uint64_t>(head_dim)
                };
                int t_up = runtime->add_task(up_args, 11, FUNC_ONLINE_UPDATE, 1);
                total_tasks++;

                // UP[bn] depends on PV[bn]
                runtime->add_successor(t_pv_arr[bn], t_up);

                // UP[bn] depends on UP[bn-1] (within same (batch, head))
                if (t_up_prev >= 0) {
                    runtime->add_successor(t_up_prev, t_up);
                }

                t_up_prev = t_up;
            }
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

    std::cout << "Created " << total_tasks << " tasks (full parallel)\n";
    runtime->print_runtime();

    return 0;
}

}
