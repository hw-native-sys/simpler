/**
 * Batch Paged Attention Orchestration Function - 16x16 Version
 *
 * Chunked batched architecture: the full batch is split into chunks of
 * IN_CORE_BATCH size. Each chunk's QK/SF/PV/UP tasks are independent
 * and can be scheduled to different cores in parallel.
 *
 * Task count = num_chunks * (1 + max_bn * 4), where
 *   num_chunks = ceil(batch / IN_CORE_BATCH)
 *
 * For batch <= IN_CORE_BATCH, behavior is identical to the non-chunked version.
 *
 * Memory Layout:
 *   Query: (batch * num_heads, head_dim) fp16
 *   Key:   (total_blocks, block_size, head_dim) fp16 (stored as K^T for QK)
 *   Value: (total_blocks, block_size, head_dim) fp16
 *
 * Per-chunk intermediate tensors (contiguous across chunk_bc dimension):
 *   sij:     (chunk_bc * q_tile, block_size)  fp32
 *   pij:     (chunk_bc * q_tile, block_size)  fp16
 *   mij/lij: (chunk_bc * q_tile)              fp32
 *   oi_new:  (chunk_bc * q_tile, head_dim)    fp32
 *   oi:      (chunk_bc * q_tile, head_dim)    fp32  accumulator
 *   mi/li:   (chunk_bc * q_tile)              fp32  accumulator
 *
 * Kernels receive global tensors + scalar metadata (including batch_start)
 * and compute per-batch addresses internally.
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

static uint64_t float_to_u64(float f) {
    union {
        float f32;
        uint64_t u64;
    } conv;
    conv.u64 = 0;
    conv.f32 = f;
    return conv.u64;
}

extern "C" {

__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(uint64_t* args, int arg_count) {
    (void)args;
    (void)arg_count;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 10,
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(uint64_t* args, int arg_count, int orch_thread_num, int orch_thread_index) {
    (void)arg_count;

    void* host_query = (void*)(uintptr_t)args[0];
    void* host_key_cache = (void*)(uintptr_t)args[1];
    void* host_value_cache = (void*)(uintptr_t)args[2];
    int* host_block_table = (int*)(uintptr_t)args[3];
    int* host_context_lens = (int*)(uintptr_t)args[4];
    void* host_out = (void*)(uintptr_t)args[5];
    int64_t* host_config = (int64_t*)(uintptr_t)args[6];

    size_t key_cache_size = (size_t)args[8];

    uint64_t batch = (uint64_t)(int)host_config[0];
    uint64_t num_heads = (uint64_t)(int)host_config[1];
    uint64_t head_dim = (uint64_t)(int)host_config[3];
    uint64_t block_size = (uint64_t)(int)host_config[4];
    uint64_t block_num = (uint64_t)(int)host_config[5];
    union { uint32_t u; float f; } scale_conv;
    scale_conv.u = (uint32_t)host_config[6];
    float scale_value = scale_conv.f;

    uint64_t q_tile = 16;
    uint64_t q_loop = (num_heads + q_tile - 1) / q_tile;
    DataType data_type = DataType::FLOAT16;
    uint64_t elem_size = get_element_size(data_type);

    LOG_INFO("batch_paged_attention: batch=%lu, num_heads=%lu",
             (unsigned long)batch, (unsigned long)num_heads);

    uint64_t max_bn = 0;
    for (uint64_t b = 0; b < batch; b++) {
        uint64_t cur_seq = host_context_lens[b];
        uint64_t bn_b = (cur_seq + block_size - 1) / block_size;
        if (bn_b > max_bn) max_bn = bn_b;
    }

    uint32_t query_shapes[2] = {(uint32_t)(batch * num_heads), (uint32_t)head_dim};
    uint64_t kv_total_rows = key_cache_size / (head_dim * elem_size);
    uint32_t key_cache_shapes[2] = {(uint32_t)kv_total_rows, (uint32_t)head_dim};
    uint32_t value_cache_shapes[2] = {(uint32_t)kv_total_rows, (uint32_t)head_dim};
    uint32_t out_shapes[2] = {(uint32_t)(batch * num_heads), (uint32_t)head_dim};

    Tensor query = make_tensor_external(host_query, query_shapes, 2, data_type);
    Tensor key_cache = make_tensor_external(host_key_cache, key_cache_shapes, 2, data_type);
    Tensor value_cache = make_tensor_external(host_value_cache, value_cache_shapes, 2, data_type);
    Tensor out = make_tensor_external(host_out, out_shapes, 2, DataType::FLOAT32);

    uint64_t bt_addr = (uint64_t)(uintptr_t)host_block_table;
    uint64_t cl_addr = (uint64_t)(uintptr_t)host_context_lens;

    uint64_t IN_CORE_BATCH = 16;
    uint64_t num_chunks = (batch + IN_CORE_BATCH - 1) / IN_CORE_BATCH;

    for (uint64_t q_idx = 0; q_idx < q_loop; q_idx++) {
        uint64_t q_offset = q_idx * q_tile;

        for (uint64_t chunk_idx = orch_thread_index; chunk_idx < num_chunks; chunk_idx += orch_thread_num) {
            uint64_t chunk_bc = batch - chunk_idx * IN_CORE_BATCH;
            if (chunk_bc > IN_CORE_BATCH) chunk_bc = IN_CORE_BATCH;
            uint64_t batch_start = chunk_idx * IN_CORE_BATCH;

            PTO2_SCOPE() {
                uint32_t oi_acc_shapes[2] = {(uint32_t)(chunk_bc * q_tile), (uint32_t)head_dim};
                uint32_t scalar_acc_shapes[1] = {(uint32_t)(chunk_bc * q_tile)};
                Tensor oi_batch = make_tensor(oi_acc_shapes, 2, DataType::FLOAT32);
                Tensor li_batch = make_tensor(scalar_acc_shapes, 1, DataType::FLOAT32);
                Tensor mi_batch = make_tensor(scalar_acc_shapes, 1, DataType::FLOAT32);

                PTOParam params_hub;
                params_hub.add_output(oi_batch);
                params_hub.add_output(li_batch);
                params_hub.add_output(mi_batch);
                pto2_rt_submit_aiv_task(FUNC_AIV_HUB, params_hub);

                for (uint64_t bn = 0; bn < max_bn; bn++) {
                    uint32_t sij_shapes[2] = {(uint32_t)(chunk_bc * q_tile), (uint32_t)block_size};
                    uint32_t vec_shapes[1] = {(uint32_t)(chunk_bc * q_tile)};
                    uint32_t oi_new_shapes[2] = {(uint32_t)(chunk_bc * q_tile), (uint32_t)head_dim};

                    Tensor sij_b = make_tensor(sij_shapes, 2, DataType::FLOAT32);
                    Tensor pij_b = make_tensor(sij_shapes, 2, data_type);
                    Tensor mij_b = make_tensor(vec_shapes, 1, DataType::FLOAT32);
                    Tensor lij_b = make_tensor(vec_shapes, 1, DataType::FLOAT32);
                    Tensor oi_new_b = make_tensor(oi_new_shapes, 2, DataType::FLOAT32);

                    PTOParam params_qk;
                    params_qk.add_input(query);
                    params_qk.add_input(key_cache);
                    params_qk.add_output(sij_b);
                    params_qk.add_scalar(bt_addr);
                    params_qk.add_scalar(chunk_bc);
                    params_qk.add_scalar(bn);
                    params_qk.add_scalar(q_offset);
                    params_qk.add_scalar(block_num);
                    params_qk.add_scalar(num_heads);
                    params_qk.add_scalar(batch_start);
                    pto2_rt_submit_aic_task(FUNC_QK_MATMUL, params_qk);

                    PTOParam params_sf;
                    params_sf.add_input(sij_b);
                    params_sf.add_output(pij_b);
                    params_sf.add_output(mij_b);
                    params_sf.add_output(lij_b);
                    params_sf.add_scalar(float_to_u64(scale_value));
                    params_sf.add_scalar(cl_addr);
                    params_sf.add_scalar(chunk_bc);
                    params_sf.add_scalar(bn);
                    params_sf.add_scalar(batch_start);
                    pto2_rt_submit_aiv_task(FUNC_SOFTMAX_PREPARE, params_sf);

                    PTOParam params_pv;
                    params_pv.add_input(pij_b);
                    params_pv.add_input(value_cache);
                    params_pv.add_output(oi_new_b);
                    params_pv.add_scalar(bt_addr);
                    params_pv.add_scalar(chunk_bc);
                    params_pv.add_scalar(bn);
                    params_pv.add_scalar(block_num);
                    params_pv.add_scalar(batch_start);
                    pto2_rt_submit_aic_task(FUNC_PV_MATMUL, params_pv);

                    uint64_t is_first = (bn == 0) ? 1 : 0;
                    uint64_t is_last = (bn == max_bn - 1) ? 1 : 0;
                    PTOParam params_up;
                    params_up.add_input(mij_b);
                    params_up.add_input(lij_b);
                    params_up.add_input(oi_new_b);
                    params_up.add_inout(mi_batch);
                    params_up.add_inout(li_batch);
                    params_up.add_inout(oi_batch);
                    params_up.add_output(out);
                    params_up.add_scalar(is_first);
                    params_up.add_scalar(is_last);
                    params_up.add_scalar(chunk_bc);
                    params_up.add_scalar(q_offset);
                    params_up.add_scalar(num_heads);
                    params_up.add_scalar(batch_start);
                    pto2_rt_submit_aiv_task(FUNC_ONLINE_UPDATE, params_up);
                }
            }
        }
    }

    LOG_INFO("batch_paged_attention: %lu tasks (batch=%lu, max_bn=%lu, chunks=%lu, IN_CORE_BATCH=%lu)",
             (unsigned long)(num_chunks * (1 + max_bn * 4)),
             (unsigned long)batch, (unsigned long)max_bn,
             (unsigned long)num_chunks, (unsigned long)IN_CORE_BATCH);
}

}  // extern "C"
