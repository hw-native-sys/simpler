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
        .expected_arg_count = 10,
    };
}

/**
 * Orchestration entry — receives a PTO2Runtime* with ops table populated.
 * The executor wraps this call in PTO2_SCOPE, so we are already inside
 * the outer scope on entry.
 */
__attribute__((visibility("default")))
void aicpu_orchestration_entry(PTO2Runtime* rt, uint64_t* args, int arg_count, int orch_thread_num, int orch_thread_index) {
    (void)arg_count;

    // Extract device pointers (first 7)
    void* host_query = (void*)(uintptr_t)args[0];           // [batch, num_heads, head_dim]
    void* host_key_cache = (void*)(uintptr_t)args[1];       // [batch, block_num, block_size, head_dim]
    void* host_value_cache = (void*)(uintptr_t)args[2];     // [batch, block_num, block_size, head_dim]
    int* host_block_table = (int*)(uintptr_t)args[3];       // [batch, block_num]
    int* host_context_lens = (int*)(uintptr_t)args[4];      // [batch]
    void* host_out = (void*)(uintptr_t)args[5];             // [batch, num_heads, head_dim]
    int64_t* host_config = (int64_t*)(uintptr_t)args[6];

    // Extract sizes (next 3 args after pointers)
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

    // Partition batch across orchestrators
    uint64_t b_start = batch * orch_thread_index / orch_thread_num;
    uint64_t b_end = batch * (orch_thread_index + 1) / orch_thread_num;

    LOG_INFO(rt, "orch_idx=%d/%d batch=%lu b_range=[%lu,%lu)",
             orch_thread_index, orch_thread_num,
             (unsigned long)batch, (unsigned long)b_start, (unsigned long)b_end);

    // Compute actual tensor shapes from buffer sizes (not from max block_num)
    uint32_t query_shapes[2] = {(uint32_t)(batch * num_heads), (uint32_t)head_dim};
    uint64_t kv_total_rows = key_cache_size / (head_dim * elem_size);
    uint32_t key_cache_shapes[2] = {(uint32_t)kv_total_rows, (uint32_t)head_dim};
    uint32_t value_cache_shapes[2] = {(uint32_t)kv_total_rows, (uint32_t)head_dim};
    uint32_t out_shapes[2] = {(uint32_t)(batch * num_heads), (uint32_t)head_dim};
    Tensor query = make_tensor_external(host_query, query_shapes, 2, data_type);
    Tensor key_cache = make_tensor_external(host_key_cache, key_cache_shapes, 2, data_type);
    Tensor value_cache = make_tensor_external(host_value_cache, value_cache_shapes, 2, data_type);
    Tensor out = make_tensor_external(host_out, out_shapes, 2, DataType::FLOAT32);
    LOG_DEBUG(rt, "query=%s", query.dump().c_str());
    LOG_DEBUG(rt, "key_cache=%s", key_cache.dump().c_str());
    LOG_DEBUG(rt, "value_cache=%s", value_cache.dump().c_str());
    LOG_DEBUG(rt, "out=%s", out.dump().c_str());

    for (uint64_t b_idx = b_start; b_idx < b_end; b_idx++) {
        uint64_t cur_seq = host_context_lens[b_idx];
        uint64_t bn_this_batch = (cur_seq + block_size - 1) / block_size;
        for (uint64_t q_idx = 0; q_idx < q_loop; q_idx++) {
            PTO2_SCOPE(rt) {
                uint32_t cur_offset = (uint32_t)(b_idx * q_head_num + q_idx * q_tile);
                uint32_t oi_shapes[2] = {(uint32_t)q_tile, (uint32_t)head_dim};
                uint32_t li_shapes[1] = {(uint32_t)q_tile};
                uint32_t mi_shapes[1] = {(uint32_t)q_tile};
                Tensor oi = make_tensor(oi_shapes, 2, DataType::FLOAT32);
                Tensor li_update = make_tensor(li_shapes, 1, DataType::FLOAT32);
                Tensor mi_update = make_tensor(mi_shapes, 1, DataType::FLOAT32);

                uint32_t qi_shapes[2] = {(uint32_t)q_tile, (uint32_t)head_dim};
                uint32_t qi_offsets[2] = {cur_offset, 0};
                Tensor qi = query.view(qi_shapes, qi_offsets);
                uint32_t out_view_shapes[2] = {(uint32_t)q_tile, (uint32_t)head_dim};
                uint32_t out_view_offsets[2] = {cur_offset, 0};
                Tensor out_view = out.view(out_view_shapes, out_view_offsets);

                PTOParam params_inplace;
                params_inplace.add_output(oi);
                params_inplace.add_output(li_update);
                params_inplace.add_output(mi_update);
                pto2_rt_submit_aiv_task(rt, FUNC_AIV_HUB, params_inplace); // create_inplace

                for (uint64_t bn = 0; bn < bn_this_batch; bn++) {
                    uint64_t cur_block_idx = host_block_table[b_idx * block_num + bn];
                    uint64_t valid_len = block_size < (cur_seq - bn * block_size) ? block_size : (cur_seq - bn * block_size);
                    uint32_t kv_shapes[2] = {(uint32_t)block_size, (uint32_t)head_dim};
                    uint32_t kv_offsets[2] = {(uint32_t)(cur_block_idx * block_size), 0};
                    Tensor kj = key_cache.view(kv_shapes, kv_offsets);
                    Tensor vj = value_cache.view(kv_shapes, kv_offsets);

                    uint32_t sij_shapes[2] = {(uint32_t)q_tile, (uint32_t)block_size};
                    Tensor sij = make_tensor(sij_shapes, 2, DataType::FLOAT32);
                    Tensor pij_f16 = make_tensor(sij_shapes, 2, data_type);

                    PTOParam params_qk;
                    params_qk.add_input(qi);
                    params_qk.add_input(kj);
                    params_qk.add_output(sij);
                    pto2_rt_submit_aic_task(rt, FUNC_QK_MATMUL, params_qk); // c1

                    uint32_t sij_valid_shapes[2] = {(uint32_t)q_tile, (uint32_t)valid_len};
                    uint32_t sij_valid_offsets[2] = {0, 0};
                    Tensor sij_valid = sij.view(sij_valid_shapes, sij_valid_offsets);
                    Tensor li = make_tensor(li_shapes, 1, DataType::FLOAT32);
                    Tensor mi = make_tensor(mi_shapes, 1, DataType::FLOAT32);
                    PTOParam params_sf;
                    params_sf.add_input(sij_valid);
                    params_sf.add_output(pij_f16);
                    params_sf.add_output(mi);
                    params_sf.add_output(li);
                    params_sf.add_scalar(float_to_u64(scale_value));
                    pto2_rt_submit_aiv_task(rt, FUNC_SOFTMAX_PREPARE, params_sf); // v1

                    uint32_t oi_tmp_shapes[2] = {(uint32_t)q_tile, (uint32_t)head_dim};
                    Tensor oi_tmp = make_tensor(oi_tmp_shapes, 2, DataType::FLOAT32);

                    PTOParam params_pv;
                    params_pv.add_input(pij_f16);
                    params_pv.add_input(vj);
                    params_pv.add_output(oi_tmp);
                    pto2_rt_submit_aic_task(rt, FUNC_PV_MATMUL, params_pv); // c2

                    uint64_t is_first = (bn == 0) ? 1 : 0;
                    uint64_t is_last = (bn == bn_this_batch - 1) ? 1 : 0;

                    PTOParam params_up;
                    params_up.add_input(mi);
                    params_up.add_input(li);
                    params_up.add_input(oi_tmp);
                    params_up.add_inout(mi_update);
                    params_up.add_inout(li_update);
                    params_up.add_inout(oi);
                    params_up.add_output(out_view);
                    params_up.add_scalar(is_first);
                    params_up.add_scalar(is_last);
                    pto2_rt_submit_aiv_task(rt, FUNC_ONLINE_UPDATE, params_up); // v2
                }
            }
        }
    }

    LOG_INFO(rt, "orch_idx=%d: tasks submitted for batch=[%lu,%lu), num_heads=%lu",
                  orch_thread_index, (unsigned long)b_start, (unsigned long)b_end,
                  (unsigned long)num_heads);
}

}  // extern "C"
