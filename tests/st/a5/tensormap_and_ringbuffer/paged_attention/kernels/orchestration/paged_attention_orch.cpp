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
#define CYCLE_COUNT_LAP(acc)       \
    do {                           \
        _t1 = get_sys_cnt_aicpu(); \
        acc += (_t1 - _t0);        \
        _t0 = _t1;                 \
    } while (0)

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(TaskArg* orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 7,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(
    PTO2Runtime* rt, TaskArg* orch_args, int orch_thread_num, int orch_thread_index) {
    (void)orch_thread_num;
    (void)orch_thread_index;
    uint64_t prof_param_extract = 0;
    uint64_t prof_ext_tensor = 0;
    uint64_t prof_scope = 0;
    uint64_t prof_make_tensor = 0;
    uint64_t prof_tensor_view = 0;
    uint64_t prof_param_setup = 0;
    uint64_t prof_submit_task = 0;
    int prof_submit_count = 0;
    int prof_make_count = 0;
    int prof_view_count = 0;

    CYCLE_COUNT_START();

    // Read dimensions from TaskArg tensor metadata
    uint64_t batch = orch_args[0].tensor.shapes[0];
    uint64_t num_heads = orch_args[0].tensor.shapes[1];
    uint64_t head_dim = orch_args[0].tensor.shapes[2];
    DataType data_type = orch_args[0].tensor.dtype;

    uint64_t block_size = orch_args[1].tensor.shapes[1];
    uint64_t block_num = orch_args[3].tensor.shapes[1];

    uint64_t scale_value = orch_args[6].scalar;

    uint64_t q_head_num = num_heads;
    uint64_t q_tile = std::min(num_heads, 128UL);
    uint64_t q_loop = (q_head_num + q_tile - 1) / q_tile;
    CYCLE_COUNT_LAP(prof_param_extract);

    LOG_ALWAYS(rt, ">>>>>> batch = %lu", (unsigned long)batch);

    // Reshape tensors for kernel consumption (2D flattened)
    void* query_ptr = orch_args[0].data<void>();
    void* kc_ptr = orch_args[1].data<void>();
    void* vc_ptr = orch_args[2].data<void>();
    void* out_ptr = orch_args[5].data<void>();

    uint64_t total_blocks_count = orch_args[1].tensor.shapes[0];

    uint32_t query_shapes[2] = {(uint32_t)(batch * num_heads), (uint32_t)head_dim};
    uint32_t key_cache_shapes[2] = {(uint32_t)(total_blocks_count * block_size), (uint32_t)head_dim};
    uint32_t value_cache_shapes[2] = {(uint32_t)(total_blocks_count * block_size), (uint32_t)head_dim};
    uint32_t out_shapes[2] = {(uint32_t)(batch * num_heads), (uint32_t)head_dim};
    Tensor query = make_tensor_external(query_ptr, query_shapes, 2, data_type);
    Tensor key_cache = make_tensor_external(kc_ptr, key_cache_shapes, 2, data_type);
    Tensor value_cache = make_tensor_external(vc_ptr, value_cache_shapes, 2, data_type);
    Tensor out = make_tensor_external(out_ptr, out_shapes, 2, DataType::FLOAT32);
    CYCLE_COUNT_LAP(prof_ext_tensor);

    int* host_block_table = orch_args[3].data<int>();
    int* host_context_lens = orch_args[4].data<int>();

    int total_tasks = 0;

    for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
        uint64_t cur_seq = host_context_lens[b_idx];
        uint64_t bn_this_batch = (cur_seq + block_size - 1) / block_size;
        for (uint64_t q_idx = 0; q_idx < q_loop; q_idx++) {
            PTO2_SCOPE(rt) {
                CYCLE_COUNT_LAP(prof_scope);
                uint64_t cur_offset = b_idx * q_head_num + q_idx * q_tile;

                uint32_t oi_shapes[2] = {(uint32_t)q_tile, (uint32_t)head_dim};
                uint32_t li_shapes[1] = {(uint32_t)q_tile};
                uint32_t mi_shapes[1] = {(uint32_t)q_tile};
                Tensor oi = make_tensor(oi_shapes, 2, DataType::FLOAT32);
                Tensor li_update = make_tensor(li_shapes, 1, DataType::FLOAT32);
                Tensor mi_update = make_tensor(mi_shapes, 1, DataType::FLOAT32);
                prof_make_count += 3;
                CYCLE_COUNT_LAP(prof_make_tensor);
                uint32_t qi_shapes[2] = {(uint32_t)q_tile, (uint32_t)head_dim};
                uint32_t qi_offsets[2] = {(uint32_t)cur_offset, 0};
                Tensor qi = query.view(qi_shapes, qi_offsets);
                uint32_t out_view_shapes[2] = {(uint32_t)q_tile, (uint32_t)head_dim};
                uint32_t out_view_offsets[2] = {(uint32_t)cur_offset, 0};
                Tensor out_view = out.view(out_view_shapes, out_view_offsets);
                prof_view_count += 2;
                CYCLE_COUNT_LAP(prof_tensor_view);

                PTOParam params_inplace;
                params_inplace.add_output(oi);
                params_inplace.add_output(li_update);
                params_inplace.add_output(mi_update);
                CYCLE_COUNT_LAP(prof_param_setup);
                pto2_rt_submit_aiv_task(rt, FUNC_AIV_HUB, params_inplace);
                prof_submit_count++;
                CYCLE_COUNT_LAP(prof_submit_task);

                for (uint64_t bn = 0; bn < bn_this_batch; bn++) {
                    PTO2_SCOPE_GUARD(rt);

                    uint64_t cur_block_idx = host_block_table[b_idx * block_num + bn];
                    uint64_t valid_len = std::min(block_size, cur_seq - bn * block_size);
                    CYCLE_COUNT_LAP(prof_param_extract);

                    uint32_t kv_shapes[2] = {(uint32_t)block_size, (uint32_t)head_dim};
                    uint32_t kv_offsets[2] = {(uint32_t)(cur_block_idx * block_size), 0};
                    Tensor kj = key_cache.view(kv_shapes, kv_offsets);
                    Tensor vj = value_cache.view(kv_shapes, kv_offsets);
                    prof_view_count += 2;
                    CYCLE_COUNT_LAP(prof_tensor_view);

                    uint32_t sij_shapes[2] = {(uint32_t)q_tile, (uint32_t)block_size};
                    Tensor sij = make_tensor(sij_shapes, 2, DataType::FLOAT32);
                    Tensor pij_f16 = make_tensor(sij_shapes, 2, data_type);
                    prof_make_count += 2;
                    CYCLE_COUNT_LAP(prof_make_tensor);

                    PTOParam params_qk;
                    params_qk.add_input(qi);
                    params_qk.add_input(kj);
                    params_qk.add_output(sij);
                    CYCLE_COUNT_LAP(prof_param_setup);
                    pto2_rt_submit_aic_task(rt, FUNC_QK_MATMUL, params_qk);
                    prof_submit_count++;
                    CYCLE_COUNT_LAP(prof_submit_task);

                    uint32_t sij_valid_shapes[2] = {(uint32_t)q_tile, (uint32_t)valid_len};
                    uint32_t sij_valid_offsets[2] = {0, 0};
                    Tensor sij_valid = sij.view(sij_valid_shapes, sij_valid_offsets);
                    prof_view_count += 1;
                    CYCLE_COUNT_LAP(prof_tensor_view);

                    Tensor li = make_tensor(li_shapes, 1, DataType::FLOAT32);
                    Tensor mi = make_tensor(mi_shapes, 1, DataType::FLOAT32);
                    prof_make_count += 2;
                    CYCLE_COUNT_LAP(prof_make_tensor);

                    PTOParam params_sf;
                    params_sf.add_input(sij_valid);
                    params_sf.add_output(pij_f16);
                    params_sf.add_output(mi);
                    params_sf.add_output(li);
                    params_sf.add_scalar(scale_value);
                    CYCLE_COUNT_LAP(prof_param_setup);
                    pto2_rt_submit_aiv_task(rt, FUNC_SOFTMAX_PREPARE, params_sf);
                    prof_submit_count++;
                    CYCLE_COUNT_LAP(prof_submit_task);

                    uint32_t oi_tmp_shapes[2] = {(uint32_t)q_tile, (uint32_t)head_dim};
                    Tensor oi_tmp = make_tensor(oi_tmp_shapes, 2, DataType::FLOAT32);
                    prof_make_count += 1;
                    CYCLE_COUNT_LAP(prof_make_tensor);

                    PTOParam params_pv;
                    params_pv.add_input(pij_f16);
                    params_pv.add_input(vj);
                    params_pv.add_output(oi_tmp);
                    CYCLE_COUNT_LAP(prof_param_setup);
                    pto2_rt_submit_aic_task(rt, FUNC_PV_MATMUL, params_pv);
                    prof_submit_count++;
                    CYCLE_COUNT_LAP(prof_submit_task);

                    uint64_t is_first = (bn == 0) ? 1 : 0;
                    uint64_t is_last = (bn == bn_this_batch - 1) ? 1 : 0;
                    CYCLE_COUNT_LAP(prof_param_extract);

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
                    CYCLE_COUNT_LAP(prof_param_setup);
                    pto2_rt_submit_aiv_task(rt, FUNC_ONLINE_UPDATE, params_up);
                    prof_submit_count++;
                    CYCLE_COUNT_LAP(prof_submit_task);
                }
            }
            CYCLE_COUNT_LAP(prof_scope);
        }
    }

    uint64_t total = prof_param_extract + prof_ext_tensor + prof_make_tensor + prof_tensor_view + prof_param_setup +
                     prof_submit_task + prof_scope;
    LOG_ALWAYS(rt,
        "=== PagedAttn Orch Profiling: %d submits, %d makes, %d views, total=%.3fus ===",
        prof_submit_count,
        prof_make_count,
        prof_view_count,
        cycles_to_us(total));
    if (total > 0) {
        LOG_ALWAYS(rt,
            "  param_extract    : %7.3fus (%5.1f%%)",
            cycles_to_us(prof_param_extract),
            prof_param_extract * 100.0 / total);
        LOG_ALWAYS(rt,
            "  ext_tensor(x4)   : %7.3fus (%5.1f%%)",
            cycles_to_us(prof_ext_tensor),
            prof_ext_tensor * 100.0 / total);
        LOG_ALWAYS(rt,
            "  make_tensor(x%d) : %7.3fus (%5.1f%%)  avg=%.3fus",
            prof_make_count,
            cycles_to_us(prof_make_tensor),
            prof_make_tensor * 100.0 / total,
            prof_make_count > 0 ? cycles_to_us(prof_make_tensor) / prof_make_count : 0.0);
        LOG_ALWAYS(rt,
            "  tensor_view(x%d) : %7.3fus (%5.1f%%)  avg=%.3fus",
            prof_view_count,
            cycles_to_us(prof_tensor_view),
            prof_tensor_view * 100.0 / total,
            prof_view_count > 0 ? cycles_to_us(prof_tensor_view) / prof_view_count : 0.0);
        LOG_ALWAYS(rt,
            "  param_setup      : %7.3fus (%5.1f%%)",
            cycles_to_us(prof_param_setup),
            prof_param_setup * 100.0 / total);
        LOG_ALWAYS(rt, "  scope            : %7.3fus (%5.1f%%)", cycles_to_us(prof_scope), prof_scope * 100.0 / total);
        LOG_ALWAYS(rt,
            "  submit_task(x%d) : %7.3fus (%5.1f%%)  avg=%.3fus",
            prof_submit_count,
            cycles_to_us(prof_submit_task),
            prof_submit_task * 100.0 / total,
            prof_submit_count > 0 ? cycles_to_us(prof_submit_task) / prof_submit_count : 0.0);
    }

#undef CYCLE_COUNT_START
#undef CYCLE_COUNT_LAP
}

}  // extern "C"
