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

#include <chrono>
#include <cstdint>
#include <cstring>

#include "pto_orchestration_api.h"

#define FUNC_QK_MATMUL 0
#define FUNC_SOFTMAX_PREPARE 1
#define FUNC_PV_MATMUL 2
#define FUNC_ONLINE_UPDATE 3
#define FUNC_AIC_HUB 4
#define FUNC_AIV_HUB 5

// Helper to encode float as uint64_t for scalar params
static uint64_t float_to_u64(float f) {
    union {
        float f32;
        uint64_t u64;
    } conv;
    conv.u64 = 0;  // Clear upper bits
    conv.f32 = f;
    return conv.u64;
}

extern "C" {
/**
 * Orchestration config — the executor reads these values to set up
 * shared memory and runtime before calling aicpu_orchestration_entry.
 */
__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(
    uint64_t* args, int arg_count) {
    (void)args;
    (void)arg_count;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 15,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    int submit_task_count = 0;
    uint64_t submit_task_total_ns = 0;

#define TIMED_SUBMIT_TASK(rt, func, worker, name, params, params_count)                                     \
    do {                                                                                                    \
        auto _t0 = std::chrono::high_resolution_clock::now();                                               \
        pto2_rt_submit_task(rt, func, worker, name, params, params_count);                                  \
        auto _t1 = std::chrono::high_resolution_clock::now();                                               \
        submit_task_total_ns +=                                                                             \
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(_t1 - _t0).count()); \
        submit_task_count++;                                                                                \
    } while (0)

    // Extract device pointers
    // Extract pointers (first 7)
    void* host_query = reinterpret_cast<void*>(args[0]);        // [batch, num_heads, head_dim]
    void* host_key_cache = reinterpret_cast<void*>(args[1]);    // [batch, block_num, block_size, head_dim]
    void* host_value_cache = reinterpret_cast<void*>(args[2]);  // [batch, block_num, block_size, head_dim]
    int* host_block_table = reinterpret_cast<int*>(args[3]);    // [batch, block_num]
    int* host_context_lens = reinterpret_cast<int*>(args[4]);   // [batch]
    void* host_out = reinterpret_cast<void*>(args[5]);          // [batch, num_heads, head_dim]
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
    uint64_t batch = static_cast<uint64_t>(static_cast<int>(host_config[0]));
    uint64_t num_heads = static_cast<uint64_t>(static_cast<int>(host_config[1]));
    int kv_head_num = static_cast<int>(host_config[2]);
    uint64_t head_dim = static_cast<uint64_t>(static_cast<int>(host_config[3]));
    uint64_t block_size = static_cast<uint64_t>(static_cast<int>(host_config[4]));
    uint64_t block_num = static_cast<uint64_t>(static_cast<int>(host_config[5]));
    union {
        uint32_t u;
        float f;
    } scale_conv;
    scale_conv.u = static_cast<uint32_t>(host_config[6]);
    float scale_value = scale_conv.f;
    uint64_t q_head_num = num_heads;
    uint64_t q_tile = std::min(num_heads, 128UL);
    uint64_t q_loop = (q_head_num + q_tile - 1) / q_tile;
    DataType data_type = DataType::BFLOAT16;  // 用例是float32的，这个考虑要如何扩展成其他类型

    printf("batch = %lu\n", batch);

    // query_size = batch * num_heads * head_dim * data_type
    // key_cache_size = batch * block_num * block_size * head_dim * data_type
    // value_cache_size = batch * block_num * block_size * head_dim * data_type
    // out = batch * num_heads * head_dim * data_type
    uint64_t query_shapes[2] = {batch * num_heads, head_dim};
    uint64_t key_cache_shapes[2] = {batch * block_num * block_size, head_dim};
    uint64_t value_cache_shapes[2] = {batch * block_num * block_size, head_dim};
    uint64_t out_shapes[2] = {batch * num_heads, head_dim};
    Tensor query = make_tensor_external(host_query, query_shapes, 2, data_type);
    Tensor key_cache = make_tensor_external(host_key_cache, key_cache_shapes, 2, data_type);
    Tensor value_cache = make_tensor_external(host_value_cache, value_cache_shapes, 2, data_type);
    // Tensor block_table = make_tensor_external(host_block_table, block_table_size);
    // Tensor context_lens = make_tensor_external(host_context_lens, context_lens_size);
    Tensor out = make_tensor_external(host_out, out_shapes, 2, DataType::FLOAT32);
    printf("query=%s\n", query.dump().c_str());
    printf("key_cache=%s\n", key_cache.dump().c_str());
    printf("value_cache=%s\n", value_cache.dump().c_str());
    printf("out=%s\n", out.dump().c_str());

    int total_tasks = 0;

    for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
        uint64_t cur_seq = host_context_lens[b_idx];
        uint64_t bn_this_batch = (cur_seq + block_size - 1) / block_size;
        for (uint64_t q_idx = 0; q_idx < q_loop; q_idx++) {
            PTO2_SCOPE(rt) {
                uint64_t cur_offset = b_idx * q_head_num + q_idx * q_tile;
                uint64_t oi_shapes[2] = {q_tile, head_dim};
                uint64_t li_shapes[1] = {q_tile};
                uint64_t mi_shapes[1] = {q_tile};
                Tensor oi = make_tensor(oi_shapes, 2, DataType::FLOAT32);
                Tensor li_update = make_tensor(li_shapes, 1, DataType::FLOAT32);
                Tensor mi_update = make_tensor(mi_shapes, 1, DataType::FLOAT32);

                PTOParam params_inplace[] = {
                    make_output_param(oi),
                    make_output_param(li_update),
                    make_output_param(mi_update),
                };
                TIMED_SUBMIT_TASK(rt, FUNC_AIV_HUB, PTO2_WORKER_VECTOR, "create_inplace", params_inplace, 3);

                for (uint64_t bn = 0; bn < bn_this_batch; bn++) {
                    Tensor qi = query.view({q_tile, head_dim}, {cur_offset, 0});
                    uint64_t cur_block_idx = host_block_table[b_idx * block_num + bn];
                    uint64_t valid_len = std::min(block_size, cur_seq - bn * block_size);
                    Tensor kj = key_cache.view({block_size, head_dim}, {cur_block_idx * block_size, 0});
                    Tensor vj = value_cache.view({block_size, head_dim}, {cur_block_idx * block_size, 0});

                    uint64_t sij_shapes[2] = {q_tile, block_size};
                    Tensor sij = make_tensor(sij_shapes, 2, DataType::FLOAT32);
                    Tensor pij_f16 = make_tensor(sij_shapes, 2, data_type);

                    PTOParam params_qk[] = {
                        make_input_param(qi),
                        make_input_param(kj),
                        make_output_param(sij),
                    };
                    TIMED_SUBMIT_TASK(rt, FUNC_QK_MATMUL, PTO2_WORKER_CUBE, "c1", params_qk, 3);

                    Tensor sij_valid = sij.view({q_tile, valid_len}, {0, 0});
                    Tensor li = make_tensor(li_shapes, 1, DataType::FLOAT32);
                    Tensor mi = make_tensor(mi_shapes, 1, DataType::FLOAT32);
                    PTOParam params_sf[] = {
                        make_input_param(sij_valid),
                        make_scalar_param(float_to_u64(scale_value)),
                        make_output_param(pij_f16),
                        make_output_param(mi),
                        make_output_param(li),
                    };
                    TIMED_SUBMIT_TASK(rt, FUNC_SOFTMAX_PREPARE, PTO2_WORKER_VECTOR, "v1", params_sf, 5);

                    uint64_t oi_tmp_shapes[2] = {q_tile, head_dim};
                    Tensor oi_tmp = make_tensor(oi_tmp_shapes, 2, DataType::FLOAT32);

                    PTOParam params_pv[] = {
                        make_input_param(pij_f16),
                        make_input_param(vj),
                        make_output_param(oi_tmp),
                    };
                    TIMED_SUBMIT_TASK(rt, FUNC_PV_MATMUL, PTO2_WORKER_CUBE, "c2", params_pv, 3);

                    uint64_t is_first = (bn == 0) ? 1 : 0;
                    uint64_t is_last = (bn == bn_this_batch - 1) ? 1 : 0;

                    Tensor out_view = out.view({q_tile, head_dim}, {cur_offset, 0});
                    PTOParam params_up[] = {
                        make_input_param(mi),
                        make_input_param(li),
                        make_input_param(oi_tmp),
                        make_inout_param(mi_update),
                        make_inout_param(li_update),
                        make_inout_param(oi),
                        make_output_param(out_view),
                        make_scalar_param(is_first),
                        make_scalar_param(is_last),
                    };
                    TIMED_SUBMIT_TASK(rt, FUNC_ONLINE_UPDATE, PTO2_WORKER_VECTOR, "v2", params_up, 9);
                }
            }
        }
    }

    printf(
        "[orch stats] pto2_submit_task called %d times, total cost %lu ns\n", submit_task_count, submit_task_total_ns);

#undef TIMED_SUBMIT_TASK
}

}  // extern "C"