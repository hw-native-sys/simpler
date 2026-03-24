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

extern "C" {

__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(OrchArg* orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 7,
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(OrchArg* orch_args, int orch_thread_num, int orch_thread_index) {
    // Read dimensions from OrchArg tensor metadata
    // query: shape=[batch, num_heads, head_dim]
    uint64_t batch     = orch_args[0].tensor.shapes[0];
    uint64_t num_heads = orch_args[0].tensor.shapes[1];
    uint64_t head_dim  = orch_args[0].tensor.shapes[2];
    DataType data_type = orch_args[0].tensor.dtype;

    // key_cache: shape=[total_blocks, block_size, kv_head_num, head_dim]
    uint64_t block_size  = orch_args[1].tensor.shapes[1];

    // block_table: shape=[batch, max_num_blocks_per_req]
    uint64_t block_num = orch_args[3].tensor.shapes[1];

    // scale from scalar arg
    uint64_t scale_value = orch_args[6].scalar;

    uint64_t q_head_num = num_heads;
    uint64_t q_tile = 16;
    uint64_t q_loop = (q_head_num + q_tile - 1) / q_tile;
    uint64_t elem_size = get_element_size(data_type);

    // Partition batch across orchestrators
    uint64_t b_start = batch * orch_thread_index / orch_thread_num;
    uint64_t b_end = batch * (orch_thread_index + 1) / orch_thread_num;

    LOG_INFO("orch_idx=%d/%d batch=%lu b_range=[%lu,%lu)",
             orch_thread_index, orch_thread_num,
             (unsigned long)batch, (unsigned long)b_start, (unsigned long)b_end);

    // Reshape tensors for kernel consumption (2D flattened)
    void* query_ptr = orch_args[0].data<void>();
    void* kc_ptr    = orch_args[1].data<void>();
    void* vc_ptr    = orch_args[2].data<void>();
    void* out_ptr   = orch_args[5].data<void>();

    // Compute kv_total_rows from key_cache tensor metadata
    uint64_t total_blocks_count = orch_args[1].tensor.shapes[0];
    uint64_t kv_total_rows = total_blocks_count * block_size;

    uint32_t query_shapes[2] = {(uint32_t)(batch * num_heads), (uint32_t)head_dim};
    uint32_t key_cache_shapes[2] = {(uint32_t)kv_total_rows, (uint32_t)head_dim};
    uint32_t value_cache_shapes[2] = {(uint32_t)kv_total_rows, (uint32_t)head_dim};
    uint32_t out_shapes[2] = {(uint32_t)(batch * num_heads), (uint32_t)head_dim};
    Tensor query = make_tensor_external(query_ptr, query_shapes, 2, data_type);
    Tensor key_cache = make_tensor_external(kc_ptr, key_cache_shapes, 2, data_type);
    Tensor value_cache = make_tensor_external(vc_ptr, value_cache_shapes, 2, data_type);
    Tensor out = make_tensor_external(out_ptr, out_shapes, 2, DataType::FLOAT32);
    LOG_DEBUG("query=%s", query.dump().c_str());
    LOG_DEBUG("key_cache=%s", key_cache.dump().c_str());
    LOG_DEBUG("value_cache=%s", value_cache.dump().c_str());
    LOG_DEBUG("out=%s", out.dump().c_str());

    int* host_block_table  = orch_args[3].data<int>();
    int* host_context_lens = orch_args[4].data<int>();

    for (uint64_t b_idx = b_start; b_idx < b_end; b_idx++) {
        uint64_t cur_seq = host_context_lens[b_idx];
        uint64_t bn_this_batch = (cur_seq + block_size - 1) / block_size;
        for (uint64_t q_idx = 0; q_idx < q_loop; q_idx++) {
            PTO2_SCOPE() {
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
                pto2_rt_submit_aiv_task(FUNC_AIV_HUB, params_inplace); // create_inplace

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
                    pto2_rt_submit_aic_task(FUNC_QK_MATMUL, params_qk); // c1

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
                    params_sf.add_scalar(scale_value);
                    pto2_rt_submit_aiv_task(FUNC_SOFTMAX_PREPARE, params_sf); // v1

                    uint32_t oi_tmp_shapes[2] = {(uint32_t)q_tile, (uint32_t)head_dim};
                    Tensor oi_tmp = make_tensor(oi_tmp_shapes, 2, DataType::FLOAT32);

                    PTOParam params_pv;
                    params_pv.add_input(pij_f16);
                    params_pv.add_input(vj);
                    params_pv.add_output(oi_tmp);
                    pto2_rt_submit_aic_task(FUNC_PV_MATMUL, params_pv); // c2

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
                    pto2_rt_submit_aiv_task(FUNC_ONLINE_UPDATE, params_up); // v2
                }
            }
        }
    }

    LOG_INFO("orch_idx=%d: tasks submitted for batch=[%lu,%lu), num_heads=%lu",
                  orch_thread_index, (unsigned long)b_start, (unsigned long)b_end,
                  (unsigned long)num_heads);
}

}  // extern "C"
