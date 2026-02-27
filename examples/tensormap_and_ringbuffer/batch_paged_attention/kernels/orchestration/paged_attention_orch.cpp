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
void aicpu_orchestration_entry(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    (void)arg_count;
    pto2_rt_init_tensor_pool(rt);

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

    LOG_INFO(rt, "batch_paged_attention: batch=%lu, num_heads=%lu",
             (unsigned long)batch, (unsigned long)num_heads);

    uint64_t max_bn = 0;
    for (uint64_t b = 0; b < batch; b++) {
        uint64_t cur_seq = host_context_lens[b];
        uint64_t bn_b = (cur_seq + block_size - 1) / block_size;
        if (bn_b > max_bn) max_bn = bn_b;
    }

    uint64_t query_shapes[2] = {batch * num_heads, head_dim};
    uint64_t kv_total_rows = key_cache_size / (head_dim * elem_size);
    uint64_t key_cache_shapes[2] = {kv_total_rows, head_dim};
    uint64_t value_cache_shapes[2] = {kv_total_rows, head_dim};
    uint64_t out_shapes[2] = {batch * num_heads, head_dim};

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

        for (uint64_t batch_start = 0; batch_start < batch; batch_start += IN_CORE_BATCH) {
            uint64_t chunk_bc = batch - batch_start;
            if (chunk_bc > IN_CORE_BATCH) chunk_bc = IN_CORE_BATCH;

            PTO2_SCOPE(rt) {
                uint64_t oi_acc_shapes[2] = {chunk_bc * q_tile, head_dim};
                uint64_t scalar_acc_shapes[1] = {chunk_bc * q_tile};
                Tensor oi_batch = make_tensor(oi_acc_shapes, 2, DataType::FLOAT32);
                Tensor li_batch = make_tensor(scalar_acc_shapes, 1, DataType::FLOAT32);
                Tensor mi_batch = make_tensor(scalar_acc_shapes, 1, DataType::FLOAT32);

                PTOParam params_hub[] = {
                    make_output_param(oi_batch),
                    make_output_param(li_batch),
                    make_output_param(mi_batch),
                };
                pto2_rt_submit_task(rt, FUNC_AIV_HUB, PTO2_WORKER_VECTOR, params_hub, 3);

                for (uint64_t bn = 0; bn < max_bn; bn++) {
                    uint64_t sij_shapes[2] = {chunk_bc * q_tile, block_size};
                    uint64_t vec_shapes[1] = {chunk_bc * q_tile};
                    uint64_t oi_new_shapes[2] = {chunk_bc * q_tile, head_dim};

                    Tensor sij_b = make_tensor(sij_shapes, 2, DataType::FLOAT32);
                    Tensor pij_b = make_tensor(sij_shapes, 2, data_type);
                    Tensor mij_b = make_tensor(vec_shapes, 1, DataType::FLOAT32);
                    Tensor lij_b = make_tensor(vec_shapes, 1, DataType::FLOAT32);
                    Tensor oi_new_b = make_tensor(oi_new_shapes, 2, DataType::FLOAT32);

                    PTOParam params_qk[] = {
                        make_input_param(query),
                        make_input_param(key_cache),
                        make_output_param(sij_b),
                        make_scalar_param(bt_addr),
                        make_scalar_param(chunk_bc),
                        make_scalar_param(bn),
                        make_scalar_param(q_offset),
                        make_scalar_param(block_num),
                        make_scalar_param(num_heads),
                        make_scalar_param(batch_start),
                    };
                    pto2_rt_submit_task(rt, FUNC_QK_MATMUL, PTO2_WORKER_CUBE, params_qk, 10);

                    PTOParam params_sf[] = {
                        make_input_param(sij_b),
                        make_output_param(pij_b),
                        make_output_param(mij_b),
                        make_output_param(lij_b),
                        make_scalar_param(float_to_u64(scale_value)),
                        make_scalar_param(cl_addr),
                        make_scalar_param(chunk_bc),
                        make_scalar_param(bn),
                        make_scalar_param(batch_start),
                    };
                    pto2_rt_submit_task(rt, FUNC_SOFTMAX_PREPARE, PTO2_WORKER_VECTOR, params_sf, 9);

                    PTOParam params_pv[] = {
                        make_input_param(pij_b),
                        make_input_param(value_cache),
                        make_output_param(oi_new_b),
                        make_scalar_param(bt_addr),
                        make_scalar_param(chunk_bc),
                        make_scalar_param(bn),
                        make_scalar_param(block_num),
                        make_scalar_param(batch_start),
                    };
                    pto2_rt_submit_task(rt, FUNC_PV_MATMUL, PTO2_WORKER_CUBE, params_pv, 8);

                    uint64_t is_first = (bn == 0) ? 1 : 0;
                    uint64_t is_last = (bn == max_bn - 1) ? 1 : 0;
                    PTOParam params_up[] = {
                        make_input_param(mij_b),
                        make_input_param(lij_b),
                        make_input_param(oi_new_b),
                        make_inout_param(mi_batch),
                        make_inout_param(li_batch),
                        make_output_param(oi_batch),
                        make_output_param(out),
                        make_scalar_param(is_first),
                        make_scalar_param(is_last),
                        make_scalar_param(chunk_bc),
                        make_scalar_param(q_offset),
                        make_scalar_param(num_heads),
                        make_scalar_param(batch_start),
                    };
                    pto2_rt_submit_task(rt, FUNC_ONLINE_UPDATE, PTO2_WORKER_VECTOR, params_up, 13);
                }
            }
        }
    }

    LOG_INFO(rt, "batch_paged_attention: %lu tasks (batch=%lu, max_bn=%lu, chunks=%lu, IN_CORE_BATCH=%lu)",
             (unsigned long)(num_chunks * (1 + max_bn * 4)),
             (unsigned long)batch, (unsigned long)max_bn,
             (unsigned long)num_chunks, (unsigned long)IN_CORE_BATCH);
}

}  // extern "C"
