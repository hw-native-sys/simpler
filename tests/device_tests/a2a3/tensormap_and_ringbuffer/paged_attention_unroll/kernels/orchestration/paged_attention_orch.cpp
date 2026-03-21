/**
 * Paged Attention Orchestration Function V2 - N_UNROLL=8, 4 Tasks Per Group
 *
 * Batches up to N_UNROLL blocks per group. Each group submits exactly 4 tasks:
 *   1. QK matmul:  qi @ K^T for n_blocks → sij_buf (q_tile, n_blocks * block_size)
 *   2. Softmax:    two-pass over sij_buf → pij_buf, mi, li
 *   3. PV matmul:  SplitK accumulated P @ V → oi_new (q_tile, head_dim)
 *   4. Update:     online softmax accumulation with group-level mi, li, oi_new
 *
 * Memory Layout:
 *   Query: (batch * num_heads, head_dim) bf16
 *   Key:   (total_blocks, block_size, head_dim) bf16 (stored as K^T for QK)
 *   Value: (total_blocks, block_size, head_dim) bf16
 */

#include <cstdint>
#include <cstring>

#include "pto_orchestration_api.h"

#define N_UNROLL 64

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

#ifdef ENABLE_PROFILING
#define CYCLE_COUNT_START() uint64_t _t0 = get_sys_cnt_aicpu(), _t1
#define CYCLE_COUNT_LAP(acc)       \
    do {                           \
        _t1 = get_sys_cnt_aicpu(); \
        acc += (_t1 - _t0);        \
        _t0 = _t1;                 \
    } while (0)
#else
#define CYCLE_COUNT_START() (void)0
#define CYCLE_COUNT_LAP(acc) (void)0
#endif

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

__attribute__((visibility("default"))) void aicpu_orchestration_entry(uint64_t* args, int arg_count, int orch_thread_num, int orch_thread_index) {
    (void)orch_thread_num;
    (void)orch_thread_index;
#ifdef ENABLE_PROFILING
    uint64_t prof_param_extract = 0;
    uint64_t prof_ext_tensor    = 0;
    uint64_t prof_make_tensor   = 0;
    uint64_t prof_tensor_view   = 0;
    uint64_t prof_param_setup   = 0;
    uint64_t prof_submit_task   = 0;
    uint64_t prof_scope_and_loop = 0;
    int      prof_submit_count  = 0;
    int      prof_make_count    = 0;
    int      prof_view_count    = 0;
#endif

    CYCLE_COUNT_START();

    // Extract device pointers (first 7)
    void* host_query = reinterpret_cast<void*>(args[0]);
    void* host_key_cache = reinterpret_cast<void*>(args[1]);
    void* host_value_cache = reinterpret_cast<void*>(args[2]);
    int* host_block_table = reinterpret_cast<int*>(args[3]);
    int* host_context_lens = reinterpret_cast<int*>(args[4]);
    void* host_out = reinterpret_cast<void*>(args[5]);
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
    DataType data_type = DataType::BFLOAT16;
    CYCLE_COUNT_LAP(prof_param_extract);

    uint32_t query_shapes[2] = {(uint32_t)(batch * num_heads), (uint32_t)head_dim};
    uint32_t key_cache_shapes[2] = {(uint32_t)(batch * block_num * block_size), (uint32_t)head_dim};
    uint32_t value_cache_shapes[2] = {(uint32_t)(batch * block_num * block_size), (uint32_t)head_dim};
    uint32_t out_shapes[2] = {(uint32_t)(batch * num_heads), (uint32_t)head_dim};
    Tensor query = make_tensor_external(host_query, query_shapes, 2, data_type, false);
    Tensor key_cache = make_tensor_external(host_key_cache, key_cache_shapes, 2, data_type, false);
    Tensor value_cache = make_tensor_external(host_value_cache, value_cache_shapes, 2, data_type, false);
    Tensor out = make_tensor_external(host_out, out_shapes, 2, DataType::FLOAT32);

#ifdef ENABLE_PROFILING
    CYCLE_COUNT_LAP(prof_ext_tensor);
#endif

    // Prefetch first batch's block table data into cache (4 cache lines = 256 bytes)
    for (int cl = 0; cl < N_UNROLL * (int)sizeof(int); cl += 64) {
        __builtin_prefetch(reinterpret_cast<char*>(host_block_table) + cl, 0, 3);
    }
    __builtin_prefetch(&host_context_lens[0], 0, 3);

    for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
        uint64_t cur_seq = host_context_lens[b_idx];
        uint64_t bn_this_batch = (cur_seq + block_size - 1) / block_size;
        // Pre-compute block table base pointer for this batch
        int* bt_base = host_block_table + b_idx * block_num;

        // Prefetch next batch's block table + context_lens while processing current batch
        if (b_idx + 1 < batch) {
            int* bt_next = host_block_table + (b_idx + 1) * block_num;
            for (int cl = 0; cl < N_UNROLL * (int)sizeof(int); cl += 64) {
                __builtin_prefetch(reinterpret_cast<char*>(bt_next) + cl, 0, 3);
            }
            __builtin_prefetch(&host_context_lens[b_idx + 1], 0, 3);
        }
        for (uint64_t q_idx = 0; q_idx < q_loop; q_idx++) {
            CYCLE_COUNT_LAP(prof_scope_and_loop);
            PTO2_SCOPE() {
                uint64_t cur_offset = b_idx * q_head_num + q_idx * q_tile;

                uint32_t oi_shapes[2] = {(uint32_t)q_tile, (uint32_t)head_dim};
                uint32_t li_shapes[1] = {(uint32_t)q_tile};
                uint32_t mi_shapes[1] = {(uint32_t)q_tile};
                Tensor oi = make_tensor(oi_shapes, 2, DataType::FLOAT32);
                Tensor li_update = make_tensor(li_shapes, 1, DataType::FLOAT32, false);
                Tensor mi_update = make_tensor(mi_shapes, 1, DataType::FLOAT32, false);
#ifdef ENABLE_PROFILING
                prof_make_count += 3;
                CYCLE_COUNT_LAP(prof_make_tensor);
#endif

                uint32_t qi_shapes[2] = {(uint32_t)q_tile, (uint32_t)head_dim};
                uint32_t qi_offsets[2] = {(uint32_t)cur_offset, 0};
                Tensor qi = query.view(qi_shapes, qi_offsets);
                uint32_t out_view_shapes[2] = {(uint32_t)q_tile, (uint32_t)head_dim};
                uint32_t out_view_offsets[2] = {(uint32_t)cur_offset, 0};
                Tensor out_view = out.view(out_view_shapes, out_view_offsets);
#ifdef ENABLE_PROFILING
                prof_view_count += 2;
                CYCLE_COUNT_LAP(prof_tensor_view);
#endif
                PTOParam params_inplace;
                params_inplace.add_output(oi);
                params_inplace.add_output(li_update);
                params_inplace.add_output(mi_update);
                CYCLE_COUNT_LAP(prof_param_setup);
                pto2_rt_submit_aiv_task(FUNC_AIV_HUB, params_inplace);
#ifdef ENABLE_PROFILING
                prof_submit_count++;
                CYCLE_COUNT_LAP(prof_submit_task);
#endif

                // Reusable PTOParam objects — reset() before each use avoids
                // repeated stack-frame construction in the inner loop.
                PTOParam params_qk, params_sf, params_pv, params_up;

                for (uint64_t bn = 0; bn < bn_this_batch; bn += N_UNROLL) {
                    uint64_t n_blocks = std::min((uint64_t)N_UNROLL, bn_this_batch - bn);

                    // Valid length for last block in this group
                    uint64_t last_block_seq_start = (bn + n_blocks - 1) * block_size;
                    uint64_t valid_len_last = std::min(block_size, cur_seq - last_block_seq_start);
                    CYCLE_COUNT_LAP(prof_param_extract);

                    // === Task 1: Batched QK matmul ===
                    uint32_t sij_buf_shapes[2] = {(uint32_t)q_tile, (uint32_t)(n_blocks * block_size)};
                    Tensor sij_buf = make_tensor(sij_buf_shapes, 2, DataType::FLOAT32);
#ifdef ENABLE_PROFILING
                    prof_make_count += 1;
                    CYCLE_COUNT_LAP(prof_make_tensor);
#endif

                    params_qk.reset();
                    params_qk.add_input(qi);
                    params_qk.add_input(key_cache);
                    params_qk.add_output(sij_buf);
                    params_qk.add_scalar(n_blocks);
                    params_qk.add_scalar(reinterpret_cast<uint64_t>(bt_base + bn));
                    CYCLE_COUNT_LAP(prof_param_setup);
                    pto2_rt_submit_aic_task(FUNC_QK_MATMUL, params_qk);
#ifdef ENABLE_PROFILING
                    prof_submit_count++;
                    CYCLE_COUNT_LAP(prof_submit_task);
#endif

                    // === Task 2: Two-pass softmax over all blocks in group ===
                    uint32_t pij_buf_shapes[2] = {(uint32_t)q_tile, (uint32_t)(n_blocks * block_size)};
                    Tensor pij_buf = make_tensor(pij_buf_shapes, 2, data_type);
                    Tensor mi = make_tensor(mi_shapes, 1, DataType::FLOAT32);
                    Tensor li = make_tensor(li_shapes, 1, DataType::FLOAT32);
#ifdef ENABLE_PROFILING
                    prof_make_count += 3;
                    CYCLE_COUNT_LAP(prof_make_tensor);
#endif

                    params_sf.reset();
                    params_sf.add_input(sij_buf);
                    params_sf.add_output(pij_buf);
                    params_sf.add_output(mi);
                    params_sf.add_output(li);
                    params_sf.add_scalar(float_to_u64(scale_value));
                    params_sf.add_scalar(n_blocks);
                    params_sf.add_scalar(valid_len_last);
                    CYCLE_COUNT_LAP(prof_param_setup);
                    pto2_rt_submit_aiv_task(FUNC_SOFTMAX_PREPARE, params_sf);
#ifdef ENABLE_PROFILING
                    prof_submit_count++;
                    CYCLE_COUNT_LAP(prof_submit_task);
#endif

                    // === Task 3: SplitK PV matmul (accumulated P @ V) ===
                    uint32_t oi_new_shapes[2] = {(uint32_t)q_tile, (uint32_t)head_dim};
                    Tensor oi_new = make_tensor(oi_new_shapes, 2, DataType::FLOAT32);
#ifdef ENABLE_PROFILING
                    prof_make_count += 1;
                    CYCLE_COUNT_LAP(prof_make_tensor);
#endif

                    params_pv.reset();
                    params_pv.add_input(pij_buf);
                    params_pv.add_input(value_cache);
                    params_pv.add_output(oi_new);
                    params_pv.add_scalar(n_blocks);
                    params_pv.add_scalar(reinterpret_cast<uint64_t>(bt_base + bn));
                    CYCLE_COUNT_LAP(prof_param_setup);
                    pto2_rt_submit_aic_task(FUNC_PV_MATMUL, params_pv);
#ifdef ENABLE_PROFILING
                    prof_submit_count++;
                    CYCLE_COUNT_LAP(prof_submit_task);
#endif

                    // === Task 4: Online update (per-group) ===
                    uint64_t is_first = (bn == 0) ? 1 : 0;
                    uint64_t is_last = (bn + n_blocks >= bn_this_batch) ? 1 : 0;

                    params_up.reset();
                    params_up.add_input(mi);
                    params_up.add_input(li);
                    params_up.add_input(oi_new);
                    params_up.add_inout(mi_update);
                    params_up.add_inout(li_update);
                    params_up.add_inout(oi);
                    params_up.add_output(out_view);
                    params_up.add_scalar(is_first);
                    params_up.add_scalar(is_last);
                    CYCLE_COUNT_LAP(prof_param_setup);
                    pto2_rt_submit_aiv_task(FUNC_ONLINE_UPDATE, params_up);
#ifdef ENABLE_PROFILING
                    prof_submit_count++;
                    CYCLE_COUNT_LAP(prof_submit_task);
#endif
                }
            }
            CYCLE_COUNT_LAP(prof_scope_and_loop);
        }
    }
    CYCLE_COUNT_LAP(prof_scope_and_loop);

#ifdef ENABLE_PROFILING
    uint64_t total = prof_param_extract + prof_ext_tensor + prof_make_tensor +
                     prof_tensor_view + prof_param_setup + prof_submit_task +
                     prof_scope_and_loop;
    LOG_ALWAYS("=== PagedAttn Orch Profiling: %d submits, %d makes, %d views, total=%.3fus ===",
        prof_submit_count, prof_make_count, prof_view_count, cycles_to_us(total));
    if (total > 0) {
        LOG_ALWAYS("  param_extract    : %7.3fus (%5.1f%%)",
            cycles_to_us(prof_param_extract), prof_param_extract * 100.0 / total);
        LOG_ALWAYS("  ext_tensor(x4)   : %7.3fus (%5.1f%%)",
            cycles_to_us(prof_ext_tensor), prof_ext_tensor * 100.0 / total);
        LOG_ALWAYS("  make_tensor(x%d) : %7.3fus (%5.1f%%)  avg=%.3fus",
            prof_make_count, cycles_to_us(prof_make_tensor), prof_make_tensor * 100.0 / total,
            prof_make_count > 0 ? cycles_to_us(prof_make_tensor) / prof_make_count : 0.0);
        LOG_ALWAYS("  tensor_view(x%d) : %7.3fus (%5.1f%%)  avg=%.3fus",
            prof_view_count, cycles_to_us(prof_tensor_view), prof_tensor_view * 100.0 / total,
            prof_view_count > 0 ? cycles_to_us(prof_tensor_view) / prof_view_count : 0.0);
        LOG_ALWAYS("  param_setup      : %7.3fus (%5.1f%%)",
            cycles_to_us(prof_param_setup), prof_param_setup * 100.0 / total);
        LOG_ALWAYS("  submit_task(x%d) : %7.3fus (%5.1f%%)  avg=%.3fus",
            prof_submit_count, cycles_to_us(prof_submit_task), prof_submit_task * 100.0 / total,
            prof_submit_count > 0 ? cycles_to_us(prof_submit_task) / prof_submit_count : 0.0);
        LOG_ALWAYS("  scope_and_loop   : %7.3fus (%5.1f%%)",
            cycles_to_us(prof_scope_and_loop), prof_scope_and_loop * 100.0 / total);
    }
#endif

#undef CYCLE_COUNT_START
#undef CYCLE_COUNT_LAP
}

}  // extern "C"
