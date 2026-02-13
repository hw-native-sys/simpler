/**
 * Paged Attention Orchestration Function - 16x16 Version
 *
 * Simplified for 16x16 framework-generated matmul kernels.
 * Each block processes a single 16x16 matmul operation.
 *
 * Memory Layout:
 *   Query: (batch, 16, 16) - one 16x16 tile per batch fp16
 *   Key:   (total_blocks, 16, 16) - stored as K^T for direct matmul fp16
 *   Value: (total_blocks, 16, 16) - direct format fp16
 *
 * This file compiles as a standalone .so with zero runtime link dependencies.
 * All runtime calls go through the PTO2RuntimeOps function-pointer table.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

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
__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(uint64_t* args, int arg_count) {
    (void)args;
    (void)arg_count;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 7,
    };
}

/**
 * Orchestration entry — receives a PTO2Runtime* with ops table populated.
 * The executor wraps this call in PTO2_SCOPE, so we are already inside
 * the outer scope on entry.
 */
__attribute__((visibility("default")))
void aicpu_orchestration_entry(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    (void)arg_count;

    // Extract device pointers (first 7)
    void* host_query = (void*)(uintptr_t)args[0];           // [batch, num_heads, head_dim]
    void* host_key_cache = (void*)(uintptr_t)args[1];       // [batch, block_num, block_size, head_dim]
    void* host_value_cache = (void*)(uintptr_t)args[2];     // [batch, block_num, block_size, head_dim]
    int* host_block_table = (int*)(uintptr_t)args[3];       // [batch, block_num]
    int* host_context_lens = (int*)(uintptr_t)args[4];      // [batch]
    void* host_out = (void*)(uintptr_t)args[5];             // [batch, num_heads, head_dim]
    int64_t* host_config = (int64_t*)(uintptr_t)args[6];

    // Extract sizes (next 7 args after pointers)
    size_t query_size = (size_t)args[7];
    size_t key_cache_size = (size_t)args[8];
    size_t value_cache_size = (size_t)args[9];

    // Extract config parameters
    uint64_t batch = (uint64_t)(int)host_config[0];
    uint64_t num_heads = (uint64_t)(int)host_config[1];
    int kv_head_num = (int)host_config[2];
    uint64_t head_dim = (uint64_t)(int)host_config[3];
    uint64_t block_size = (uint64_t)(int)host_config[4];
    uint64_t block_num = (uint64_t)(int)host_config[5];
    // Reinterpret scale_bits as float (golden.py packs float via struct.pack)
    union { uint32_t u; float f; } scale_conv;
    scale_conv.u = (uint32_t)host_config[6];
    float scale_value = scale_conv.f;
    uint64_t q_head_num = num_heads;
    uint64_t q_tile = 16;
    uint64_t q_loop = (q_head_num + q_tile - 1) / q_tile;
    DataType data_type = DataType::FLOAT16;
    uint64_t elem_size = get_element_size(data_type);

    (void)kv_head_num;

    printf("batch = %lu\n", (unsigned long)batch);

    // Compute actual tensor shapes from buffer sizes (not from max block_num)
    uint64_t query_shapes[2] = {batch * num_heads, head_dim};
    uint64_t kv_total_rows = key_cache_size / (head_dim * elem_size);
    uint64_t key_cache_shapes[2] = {kv_total_rows, head_dim};
    uint64_t value_cache_shapes[2] = {kv_total_rows, head_dim};
    uint64_t out_shapes[2] = {batch * num_heads, head_dim};
    Tensor query = make_tensor_external(host_query, query_shapes, 2, data_type);
    Tensor key_cache = make_tensor_external(host_key_cache, key_cache_shapes, 2, data_type);
    Tensor value_cache = make_tensor_external(host_value_cache, value_cache_shapes, 2, data_type);
    Tensor out = make_tensor_external(host_out, out_shapes, 2, DataType::FLOAT32);
    printf("query=%s\n", query.dump().c_str());
    printf("key_cache=%s\n", key_cache.dump().c_str());
    printf("value_cache=%s\n", value_cache.dump().c_str());
    printf("out=%s\n", out.dump().c_str());

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
                pto2_rt_submit_task(rt, FUNC_AIV_HUB, PTO2_WORKER_VECTOR, "create_inplace", params_inplace, 3);

                for (uint64_t bn = 0; bn < bn_this_batch; bn++) {
                    Tensor qi = query.view({q_tile, head_dim}, {cur_offset, 0});
                    uint64_t cur_block_idx = host_block_table[b_idx * block_num + bn];
                    uint64_t valid_len = block_size < (cur_seq - bn * block_size) ? block_size : (cur_seq - bn * block_size);
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
                    pto2_rt_submit_task(rt, FUNC_QK_MATMUL, PTO2_WORKER_CUBE, "c1", params_qk, 3);

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
                    pto2_rt_submit_task(rt, FUNC_SOFTMAX_PREPARE, PTO2_WORKER_VECTOR, "v1", params_sf, 5);

                    uint64_t oi_tmp_shapes[2] = {q_tile, head_dim};
                    Tensor oi_tmp = make_tensor(oi_tmp_shapes, 2, DataType::FLOAT32);

                    PTOParam params_pv[] = {
                        make_input_param(pij_f16),
                        make_input_param(vj),
                        make_output_param(oi_tmp),
                    };
                    pto2_rt_submit_task(rt, FUNC_PV_MATMUL, PTO2_WORKER_CUBE, "c2", params_pv, 3);

                    uint64_t is_first = (bn == 0) ? 1 : 0;
                    uint64_t is_last = (bn == bn_this_batch - 1) ? 1 : 0;

                    Tensor out_view = out.view({q_tile, head_dim}, {cur_offset, 0});
                    PTOParam params_up[] = {
                        make_input_param(mi),
                        make_input_param(li),
                        make_input_param(oi_tmp),
                        make_inout_param(mi_update),
                        make_inout_param(li_update),
                        make_output_param(oi),
                        make_output_param(out_view),
                        make_scalar_param(is_first),
                        make_scalar_param(is_last),
                    };
                    pto2_rt_submit_task(rt, FUNC_ONLINE_UPDATE, PTO2_WORKER_VECTOR, "v2", params_up, 9);
                }
            }
        }
    }

    printf("[orch] tasks submitted for batch=%lu, num_heads=%lu\n",
           (unsigned long)batch, (unsigned long)num_heads);
}

}  // extern "C"
