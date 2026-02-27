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

#include <cstdint>
#include <cstring>

#include "pto_orchestration_api.h"

#define FUNC_QK_MATMUL 0
#define FUNC_SOFTMAX_PREPARE 1
#define FUNC_PV_MATMUL 2
#define FUNC_ONLINE_UPDATE 3
#define FUNC_AIC_HUB 4
#define FUNC_AIV_HUB 5

constexpr uint64_t PLATFORM_PROF_SYS_CNT_FREQ = 50000000;  // 50 MHz

inline double cycles_to_us(uint64_t cycles) {
    return (static_cast<double>(cycles) / PLATFORM_PROF_SYS_CNT_FREQ) * 1000000.0;
};

inline uint64_t get_sys_cnt_aicpu() {
    uint64_t ticks;
    asm volatile("mrs %0, cntvct_el0" : "=r"(ticks));
    return ticks;
}

#define CYCLE_COUNT_START() uint64_t _t0 = get_sys_cnt_aicpu(), _t1
#define CYCLE_COUNT_LAP(acc) do { _t1 = get_sys_cnt_aicpu(); acc += (_t1 - _t0); _t0 = _t1; } while(0)

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
        .expected_arg_count = 10,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    pto2_rt_init_tensor_pool(rt);
    uint64_t prof_param_extract = 0;
    uint64_t prof_ext_tensor    = 0;
    uint64_t prof_make_tensor   = 0;
    uint64_t prof_tensor_view   = 0;
    uint64_t prof_param_setup   = 0;
    uint64_t prof_submit_task   = 0;
    int      prof_submit_count  = 0;
    int      prof_make_count    = 0;
    int      prof_view_count    = 0;

    CYCLE_COUNT_START();

    // Extract device pointers
    // Extract pointers (first 7)
    void* host_query = reinterpret_cast<void*>(args[0]);        // [batch, num_heads, head_dim]
    void* host_key_cache = reinterpret_cast<void*>(args[1]);    // [batch, block_num, block_size, head_dim]
    void* host_value_cache = reinterpret_cast<void*>(args[2]);  // [batch, block_num, block_size, head_dim]
    int* host_block_table = reinterpret_cast<int*>(args[3]);    // [batch, block_num]
    int* host_context_lens = reinterpret_cast<int*>(args[4]);   // [batch]
    void* host_out = reinterpret_cast<void*>(args[5]);          // [batch, num_heads, head_dim]
    int64_t* host_config = reinterpret_cast<int64_t*>(args[6]);

    // Extract sizes (next 3)
    size_t query_size = static_cast<size_t>(args[7]);
    size_t key_cache_size = static_cast<size_t>(args[8]);
    size_t value_cache_size = static_cast<size_t>(args[9]);

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
    CYCLE_COUNT_LAP(prof_param_extract);

    LOG_ALWAYS(rt, ">>>>>> batch = %lu", (unsigned long)batch);

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
    CYCLE_COUNT_LAP(prof_ext_tensor);
    LOG_DEBUG(rt, "query=%s", query.dump().c_str());
    LOG_DEBUG(rt, "key_cache=%s", key_cache.dump().c_str());
    LOG_DEBUG(rt, "value_cache=%s", value_cache.dump().c_str());
    LOG_DEBUG(rt, "out=%s", out.dump().c_str());

    int total_tasks = 0;

    for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
        uint64_t cur_seq = host_context_lens[b_idx];
        uint64_t bn_this_batch = (cur_seq + block_size - 1) / block_size;
        for (uint64_t q_idx = 0; q_idx < q_loop; q_idx++) {
            PTO2_SCOPE(rt) {
                uint64_t cur_offset = b_idx * q_head_num + q_idx * q_tile;
                CYCLE_COUNT_START();

                uint64_t oi_shapes[2] = {q_tile, head_dim};
                uint64_t li_shapes[1] = {q_tile};
                uint64_t mi_shapes[1] = {q_tile};
                Tensor oi = make_tensor(oi_shapes, 2, DataType::FLOAT32);
                Tensor li_update = make_tensor(li_shapes, 1, DataType::FLOAT32);
                Tensor mi_update = make_tensor(mi_shapes, 1, DataType::FLOAT32);
                prof_make_count += 3;
                CYCLE_COUNT_LAP(prof_make_tensor);
                uint64_t qi_shapes[2] = {q_tile, head_dim};
                uint64_t qi_offsets[2] = {cur_offset, 0};
                Tensor qi = query.view(qi_shapes, qi_offsets);
                uint64_t out_view_shapes[2] = {q_tile, head_dim};
                uint64_t out_view_offsets[2] = {cur_offset, 0};
                Tensor out_view = out.view(out_view_shapes, out_view_offsets);
                prof_view_count += 2;
                CYCLE_COUNT_LAP(prof_tensor_view);

                PTOParam params_inplace[] = {
                    make_output_param(oi),
                    make_output_param(li_update),
                    make_output_param(mi_update),
                };
                CYCLE_COUNT_LAP(prof_param_setup);
                pto2_rt_submit_task(rt, FUNC_AIV_HUB, PTO2_WORKER_VECTOR, params_inplace, 3);
                prof_submit_count++;
                CYCLE_COUNT_LAP(prof_submit_task);

                for (uint64_t bn = 0; bn < bn_this_batch; bn++) {
                    uint64_t cur_block_idx = host_block_table[b_idx * block_num + bn];
                    uint64_t valid_len = std::min(block_size, cur_seq - bn * block_size);
                    CYCLE_COUNT_LAP(prof_param_extract);

                    uint64_t kv_shapes[2] = {block_size, head_dim};
                    uint64_t kv_offsets[2] = {cur_block_idx * block_size, 0};
                    Tensor kj = key_cache.view(kv_shapes, kv_offsets);
                    Tensor vj = value_cache.view(kv_shapes, kv_offsets);
                    prof_view_count += 2;
                    CYCLE_COUNT_LAP(prof_tensor_view);

                    uint64_t sij_shapes[2] = {q_tile, block_size};
                    Tensor sij = make_tensor(sij_shapes, 2, DataType::FLOAT32);
                    Tensor pij_f16 = make_tensor(sij_shapes, 2, data_type);
                    prof_make_count += 2;
                    CYCLE_COUNT_LAP(prof_make_tensor);

                    PTOParam params_qk[] = {
                        make_input_param(qi),
                        make_input_param(kj),
                        make_output_param(sij),
                    };
                    CYCLE_COUNT_LAP(prof_param_setup);
                    pto2_rt_submit_task(rt, FUNC_QK_MATMUL, PTO2_WORKER_CUBE, params_qk, 3);
                    prof_submit_count++;
                    CYCLE_COUNT_LAP(prof_submit_task);

                    uint64_t sij_valid_shapes[2] = {q_tile, valid_len};
                    uint64_t sij_valid_offsets[2] = {0, 0};
                    Tensor sij_valid = sij.view(sij_valid_shapes, sij_valid_offsets);
                    prof_view_count += 1;
                    CYCLE_COUNT_LAP(prof_tensor_view);

                    Tensor li = make_tensor(li_shapes, 1, DataType::FLOAT32);
                    Tensor mi = make_tensor(mi_shapes, 1, DataType::FLOAT32);
                    prof_make_count += 2;
                    CYCLE_COUNT_LAP(prof_make_tensor);

                    PTOParam params_sf[] = {
                        make_input_param(sij_valid),
                        make_scalar_param(float_to_u64(scale_value)),
                        make_output_param(pij_f16),
                        make_output_param(mi),
                        make_output_param(li),
                    };
                    CYCLE_COUNT_LAP(prof_param_setup);
                    pto2_rt_submit_task(rt, FUNC_SOFTMAX_PREPARE, PTO2_WORKER_VECTOR, params_sf, 5);
                    prof_submit_count++;
                    CYCLE_COUNT_LAP(prof_submit_task);

                    uint64_t oi_tmp_shapes[2] = {q_tile, head_dim};
                    Tensor oi_tmp = make_tensor(oi_tmp_shapes, 2, DataType::FLOAT32);
                    prof_make_count += 1;
                    CYCLE_COUNT_LAP(prof_make_tensor);

                    PTOParam params_pv[] = {
                        make_input_param(pij_f16),
                        make_input_param(vj),
                        make_output_param(oi_tmp),
                    };
                    CYCLE_COUNT_LAP(prof_param_setup);
                    pto2_rt_submit_task(rt, FUNC_PV_MATMUL, PTO2_WORKER_CUBE, params_pv, 3);
                    prof_submit_count++;
                    CYCLE_COUNT_LAP(prof_submit_task);

                    uint64_t is_first = (bn == 0) ? 1 : 0;
                    uint64_t is_last = (bn == bn_this_batch - 1) ? 1 : 0;
                    CYCLE_COUNT_LAP(prof_param_extract);

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
                    CYCLE_COUNT_LAP(prof_param_setup);
                    pto2_rt_submit_task(rt, FUNC_ONLINE_UPDATE, PTO2_WORKER_VECTOR, params_up, 9);
                    prof_submit_count++;
                    CYCLE_COUNT_LAP(prof_submit_task);
                }
            }
        }
    }

    uint64_t total = prof_param_extract + prof_ext_tensor + prof_make_tensor +
                     prof_tensor_view + prof_param_setup + prof_submit_task;
    LOG_ALWAYS(rt, "=== PagedAttn Orch Profiling: %d submits, %d makes, %d views, total=%.3fus ===",
        prof_submit_count, prof_make_count, prof_view_count, cycles_to_us(total));
    if (total > 0) {
        LOG_ALWAYS(rt, "  param_extract    : %7.3fus (%5.1f%%)",
            cycles_to_us(prof_param_extract), prof_param_extract * 100.0 / total);
        LOG_ALWAYS(rt, "  ext_tensor(x4)   : %7.3fus (%5.1f%%)",
            cycles_to_us(prof_ext_tensor), prof_ext_tensor * 100.0 / total);
        LOG_ALWAYS(rt, "  make_tensor(x%d) : %7.3fus (%5.1f%%)  avg=%.3fus",
            prof_make_count, cycles_to_us(prof_make_tensor), prof_make_tensor * 100.0 / total,
            prof_make_count > 0 ? cycles_to_us(prof_make_tensor) / prof_make_count : 0.0);
        LOG_ALWAYS(rt, "  tensor_view(x%d) : %7.3fus (%5.1f%%)  avg=%.3fus",
            prof_view_count, cycles_to_us(prof_tensor_view), prof_tensor_view * 100.0 / total,
            prof_view_count > 0 ? cycles_to_us(prof_tensor_view) / prof_view_count : 0.0);
        LOG_ALWAYS(rt, "  param_setup      : %7.3fus (%5.1f%%)",
            cycles_to_us(prof_param_setup), prof_param_setup * 100.0 / total);
        LOG_ALWAYS(rt, "  submit_task(x%d) : %7.3fus (%5.1f%%)  avg=%.3fus",
            prof_submit_count, cycles_to_us(prof_submit_task), prof_submit_task * 100.0 / total,
            prof_submit_count > 0 ? cycles_to_us(prof_submit_task) / prof_submit_count : 0.0);
    }

#undef CYCLE_COUNT_START
#undef CYCLE_COUNT_LAP
}

}  // extern "C"