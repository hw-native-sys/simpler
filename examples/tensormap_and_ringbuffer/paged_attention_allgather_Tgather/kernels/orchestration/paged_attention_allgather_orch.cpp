/**
 * Paged Attention + AllGather (TGATHER) orchestration for tensormap_and_ringbuffer.
 *
 * Flow: Phase 1 - Paged Attention (QK->SF->PV->UP)
 *       Phase 2 - WindowMemCopyIn -> for r in [0,n_ranks): Barrier -> Gather(root=r)
 *                 -> [rank r: WindowMemCopyOut] -> Barrier(post)
 *
 * Args (22): [0] query, [1] key_cache, [2] value_cache, [3] block_table,
 *   [4] context_lens, [5] attn_out, [6] allgather_out, [7] config,
 *   [8-15] 7 sizes, [16] device_ctx_ptr, [17] win_in_base, [18] win_out_base,
 *   [19] n_ranks, [20] root, [21] rank_id
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"

constexpr int GATHER_COUNT = 64;
constexpr size_t HCCL_WIN_SYNC_PREFIX = 64 * sizeof(int32_t);

#define FUNC_QK_MATMUL 0
#define FUNC_SOFTMAX_PREPARE 1
#define FUNC_PV_MATMUL 2
#define FUNC_ONLINE_UPDATE 3
#define FUNC_AIC_HUB 4
#define FUNC_AIV_HUB 5
#define FUNC_WIN_MEMCOPY_IN  6
#define FUNC_GATHER          7
#define FUNC_WIN_MEMCOPY_OUT 8
#define FUNC_COMM_BARRIER    9

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
        .expected_arg_count = 22,
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    (void)arg_count;
    pto2_rt_init_tensor_pool(rt);

    void* dev_query = reinterpret_cast<void*>(args[0]);
    void* dev_key_cache = reinterpret_cast<void*>(args[1]);
    void* dev_value_cache = reinterpret_cast<void*>(args[2]);
    int* dev_block_table = reinterpret_cast<int*>(args[3]);
    int* dev_context_lens = reinterpret_cast<int*>(args[4]);
    void* dev_attn_out = reinterpret_cast<void*>(args[5]);
    void* dev_allgather_out = reinterpret_cast<void*>(args[6]);
    int64_t* dev_config = reinterpret_cast<int64_t*>(args[7]);

    size_t query_size = static_cast<size_t>(args[8]);
    size_t key_cache_size = static_cast<size_t>(args[9]);
    size_t value_cache_size = static_cast<size_t>(args[10]);
    (void)args[11];
    (void)args[12];
    (void)args[13];
    (void)args[14];
    (void)args[15];

    uint64_t device_ctx_ptr = args[16];
    uint64_t win_in_base = args[17];
    (void)args[18];
    int n_ranks = static_cast<int>(args[19]);
    (void)args[20];
    int rank_id = static_cast<int>(args[21]);

    uint64_t batch = static_cast<uint64_t>(static_cast<int>(dev_config[0]));
    uint64_t num_heads = static_cast<uint64_t>(static_cast<int>(dev_config[1]));
    int kv_head_num = static_cast<int>(dev_config[2]);
    uint64_t head_dim = static_cast<uint64_t>(static_cast<int>(dev_config[3]));
    uint64_t block_size = static_cast<uint64_t>(static_cast<int>(dev_config[4]));
    uint64_t block_num = static_cast<uint64_t>(static_cast<int>(dev_config[5]));
    union { uint32_t u; float f; } scale_conv;
    scale_conv.u = static_cast<uint32_t>(dev_config[6]);
    float scale_value = scale_conv.f;

    uint64_t q_head_num = num_heads;
    uint64_t q_tile = 16;
    uint64_t q_loop = (q_head_num + q_tile - 1) / q_tile;
    DataType data_type = DataType::FLOAT16;
    uint64_t elem_size = get_element_size(data_type);

    (void)kv_head_num;

    LOG_INFO(rt, "paged_attention_allgather_Tgather: n_ranks=%d rank_id=%d batch=%lu",
             n_ranks, rank_id, (unsigned long)batch);

    uint64_t query_shapes[2] = {batch * num_heads, head_dim};
    uint64_t kv_total_rows = key_cache_size / (head_dim * elem_size);
    uint64_t key_cache_shapes[2] = {kv_total_rows, head_dim};
    uint64_t value_cache_shapes[2] = {kv_total_rows, head_dim};
    uint64_t attn_out_shapes[2] = {batch * num_heads, head_dim};

    Tensor query = make_tensor_external(dev_query, query_shapes, 2, data_type);
    Tensor key_cache = make_tensor_external(dev_key_cache, key_cache_shapes, 2, data_type);
    Tensor value_cache = make_tensor_external(dev_value_cache, value_cache_shapes, 2, data_type);
    Tensor attn_out = make_tensor_external(dev_attn_out, attn_out_shapes, 2, DataType::FLOAT32);

    /* Phase 1: Paged Attention */
    for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
        uint64_t cur_seq = static_cast<uint64_t>(dev_context_lens[b_idx]);
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

                uint64_t qi_shapes[2] = {q_tile, head_dim};
                uint64_t qi_offsets[2] = {cur_offset, 0};
                Tensor qi = query.view(qi_shapes, qi_offsets);
                uint64_t out_view_shapes[2] = {q_tile, head_dim};
                uint64_t out_view_offsets[2] = {cur_offset, 0};
                Tensor out_view = attn_out.view(out_view_shapes, out_view_offsets);

                PTOParam params_inplace[] = {
                    make_output_param(oi),
                    make_output_param(li_update),
                    make_output_param(mi_update),
                };
                pto2_rt_submit_task(rt, FUNC_AIV_HUB, PTO2_WORKER_VECTOR, params_inplace, 3);

                for (uint64_t bn = 0; bn < bn_this_batch; bn++) {
                    uint64_t cur_block_idx = static_cast<uint64_t>(dev_block_table[b_idx * block_num + bn]);
                    uint64_t valid_len = block_size < (cur_seq - bn * block_size) ? block_size : (cur_seq - bn * block_size);
                    uint64_t kv_shapes[2] = {block_size, head_dim};
                    uint64_t kv_offsets[2] = {cur_block_idx * block_size, 0};
                    Tensor kj = key_cache.view(kv_shapes, kv_offsets);
                    Tensor vj = value_cache.view(kv_shapes, kv_offsets);

                    uint64_t sij_shapes[2] = {q_tile, block_size};
                    Tensor sij = make_tensor(sij_shapes, 2, DataType::FLOAT32);
                    Tensor pij_f16 = make_tensor(sij_shapes, 2, data_type);

                    PTOParam params_qk[] = {
                        make_input_param(qi),
                        make_input_param(kj),
                        make_output_param(sij),
                    };
                    pto2_rt_submit_task(rt, FUNC_QK_MATMUL, PTO2_WORKER_CUBE, params_qk, 3);

                    uint64_t sij_valid_shapes[2] = {q_tile, valid_len};
                    uint64_t sij_valid_offsets[2] = {0, 0};
                    Tensor sij_valid = sij.view(sij_valid_shapes, sij_valid_offsets);
                    Tensor li = make_tensor(li_shapes, 1, DataType::FLOAT32);
                    Tensor mi = make_tensor(mi_shapes, 1, DataType::FLOAT32);
                    PTOParam params_sf[] = {
                        make_input_param(sij_valid),
                        make_scalar_param(float_to_u64(scale_value)),
                        make_output_param(pij_f16),
                        make_output_param(mi),
                        make_output_param(li),
                    };
                    pto2_rt_submit_task(rt, FUNC_SOFTMAX_PREPARE, PTO2_WORKER_VECTOR, params_sf, 5);

                    uint64_t oi_tmp_shapes[2] = {q_tile, head_dim};
                    Tensor oi_tmp = make_tensor(oi_tmp_shapes, 2, DataType::FLOAT32);

                    PTOParam params_pv[] = {
                        make_input_param(pij_f16),
                        make_input_param(vj),
                        make_output_param(oi_tmp),
                    };
                    pto2_rt_submit_task(rt, FUNC_PV_MATMUL, PTO2_WORKER_CUBE, params_pv, 3);

                    uint64_t is_first = (bn == 0) ? 1 : 0;
                    uint64_t is_last = (bn == bn_this_batch - 1) ? 1 : 0;

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
                    pto2_rt_submit_task(rt, FUNC_ONLINE_UPDATE, PTO2_WORKER_VECTOR, params_up, 9);
                }
            }
        }
    }

    /* Phase 2: AllGather (TGATHER) - for r in [0,n_ranks): Barrier -> Gather(root=r) -> [rank r: WindowMemCopyOut] */
    size_t barrier_size = static_cast<size_t>(n_ranks) * sizeof(int32_t);
    size_t total_barrier_bytes = barrier_size * (static_cast<size_t>(n_ranks) + 1);
    uint64_t barrier_base_0 = win_in_base + HCCL_WIN_SYNC_PREFIX;
    uint64_t win_src = barrier_base_0 + total_barrier_bytes;
    uint64_t win_dst = win_src + GATHER_COUNT * sizeof(float);

    uint64_t src_shapes[1] = {static_cast<uint64_t>(GATHER_COUNT)};
    uint64_t dst_shapes[1] = {static_cast<uint64_t>(n_ranks) * GATHER_COUNT};
    uint64_t barrier_shapes[1] = {static_cast<uint64_t>(n_ranks)};
    uint64_t sync_shapes[1] = {1};

    Tensor dev_src_t = make_tensor_external(dev_attn_out, src_shapes, 1, DataType::FLOAT32);
    Tensor dev_out_t = make_tensor_external(dev_allgather_out, dst_shapes, 1, DataType::FLOAT32);
    Tensor win_src_t = make_tensor_external(reinterpret_cast<void*>(win_src), src_shapes, 1, DataType::FLOAT32);
    Tensor win_dst_t = make_tensor_external(reinterpret_cast<void*>(win_dst), dst_shapes, 1, DataType::FLOAT32);

    PTO2_SCOPE(rt) {
        PTOParam params_wmin[] = {
            make_output_param(win_src_t),
            make_input_param(dev_src_t),
            make_scalar_param(static_cast<uint64_t>(GATHER_COUNT)),
        };
        pto2_rt_submit_task(rt, FUNC_WIN_MEMCOPY_IN, PTO2_WORKER_VECTOR, params_wmin, 3);

        for (int r = 0; r < n_ranks; r++) {
            uint64_t barrier_base_r = barrier_base_0 + static_cast<uint64_t>(r) * barrier_size;
            Tensor barrier_r_t = make_tensor_external(reinterpret_cast<void*>(barrier_base_r), barrier_shapes, 1, DataType::INT32);
            Tensor sync_r_t = make_tensor(sync_shapes, 1, DataType::INT32);

            PTOParam params_barrier[] = {
                make_input_param(barrier_r_t),
                make_scalar_param(device_ctx_ptr),
                make_scalar_param(static_cast<uint64_t>(n_ranks)),
                make_scalar_param(static_cast<uint64_t>(0)),
                make_input_param(r == 0 ? win_src_t : win_dst_t),
                make_output_param(sync_r_t),
            };
            pto2_rt_submit_task(rt, FUNC_COMM_BARRIER, PTO2_WORKER_VECTOR, params_barrier, 6);

            PTOParam params_gather[] = {
                make_output_param(win_dst_t),
                make_input_param(win_src_t),
                make_input_param(sync_r_t),
                make_scalar_param(device_ctx_ptr),
                make_scalar_param(static_cast<uint64_t>(n_ranks)),
                make_scalar_param(static_cast<uint64_t>(r)),
            };
            pto2_rt_submit_task(rt, FUNC_GATHER, PTO2_WORKER_VECTOR, params_gather, 6);

            if (rank_id == r) {
                PTOParam params_wmout[] = {
                    make_output_param(dev_out_t),
                    make_input_param(win_dst_t),
                    make_scalar_param(static_cast<uint64_t>(n_ranks * GATHER_COUNT)),
                };
                pto2_rt_submit_task(rt, FUNC_WIN_MEMCOPY_OUT, PTO2_WORKER_VECTOR, params_wmout, 3);
            }
        }

        uint64_t barrier_base_post = barrier_base_0 + static_cast<uint64_t>(n_ranks) * barrier_size;
        Tensor barrier_post_t = make_tensor_external(reinterpret_cast<void*>(barrier_base_post), barrier_shapes, 1, DataType::INT32);
        Tensor sync_post_t = make_tensor(sync_shapes, 1, DataType::INT32);
        PTOParam params_barrier_post[] = {
            make_input_param(barrier_post_t),
            make_scalar_param(device_ctx_ptr),
            make_scalar_param(static_cast<uint64_t>(n_ranks)),
            make_scalar_param(static_cast<uint64_t>(0)),
            make_input_param(win_dst_t),
            make_output_param(sync_post_t),
        };
        pto2_rt_submit_task(rt, FUNC_COMM_BARRIER, PTO2_WORKER_VECTOR, params_barrier_post, 6);
    }

    LOG_INFO(rt, "paged_attention_allgather_Tgather tasks submitted");
}

}  // extern "C"
