/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
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

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "pto_orchestration_api.h"

#define N_UNROLL 64

#define FUNC_QK_MATMUL 0
#define FUNC_SOFTMAX_PREPARE 1
#define FUNC_PV_MATMUL 2
#define FUNC_ONLINE_UPDATE 3
constexpr uint64_t PLATFORM_PROF_SYS_CNT_FREQ = 50000000;  // 50 MHz

inline double cycles_to_us(uint64_t cycles) {
    return (static_cast<double>(cycles) / PLATFORM_PROF_SYS_CNT_FREQ) * 1000000.0;
}

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

extern "C" {
/**
 * Orchestration config — the executor reads these values to set up
 * shared memory and runtime before calling aicpu_orchestration_entry.
 */
__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 7,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
#ifdef ENABLE_PROFILING
    uint64_t prof_param_extract = 0;
    uint64_t prof_ext_tensor = 0;
    uint64_t prof_make_tensor = 0;
    uint64_t prof_tensor_view = 0;
    uint64_t prof_param_setup = 0;
    uint64_t prof_submit_task = 0;
    uint64_t prof_scope_and_loop = 0;
    int prof_submit_count = 0;
    int prof_make_count = 0;
    int prof_view_count = 0;
#endif

    CYCLE_COUNT_START();

    // Read dimensions from tensor metadata
    // query: shape=[batch, num_heads, head_dim]
    uint64_t batch = orch_args.tensor(0).shapes[0];
    uint64_t num_heads = orch_args.tensor(0).shapes[1];
    uint64_t head_dim = orch_args.tensor(0).shapes[2];
    DataType data_type = orch_args.tensor(0).dtype;

    // key_cache: shape=[total_blocks, block_size, kv_head_num, head_dim]
    uint64_t block_size = orch_args.tensor(1).shapes[1];

    // block_table: shape=[batch, max_num_blocks_per_req]
    uint64_t block_num = orch_args.tensor(3).shapes[1];

    // scale from scalar arg
    uint64_t scale_value = orch_args.scalar(0);
    uint64_t q_head_num = num_heads;
    uint64_t q_tile = std::min(num_heads, static_cast<uint64_t>(128));
    uint64_t q_loop = (q_head_num + q_tile - 1) / q_tile;
    CYCLE_COUNT_LAP(prof_param_extract);

    // Reshape tensors for kernel consumption (2D flattened)
    void *query_ptr = orch_args.tensor(0).data_as<void>();
    void *kc_ptr = orch_args.tensor(1).data_as<void>();
    void *vc_ptr = orch_args.tensor(2).data_as<void>();
    void *out_ptr = orch_args.tensor(5).data_as<void>();

    uint64_t total_blocks_count = orch_args.tensor(1).shapes[0];

    uint32_t query_shapes[2] = {static_cast<uint32_t>(batch * num_heads), static_cast<uint32_t>(head_dim)};
    uint32_t key_cache_shapes[2] = {
        static_cast<uint32_t>(total_blocks_count * block_size), static_cast<uint32_t>(head_dim)
    };
    uint32_t value_cache_shapes[2] = {
        static_cast<uint32_t>(total_blocks_count * block_size), static_cast<uint32_t>(head_dim)
    };
    uint32_t out_shapes[2] = {static_cast<uint32_t>(batch * num_heads), static_cast<uint32_t>(head_dim)};
    Tensor query = make_tensor_external(query_ptr, query_shapes, 2, data_type, false);
    Tensor key_cache = make_tensor_external(kc_ptr, key_cache_shapes, 2, data_type, false);
    Tensor value_cache = make_tensor_external(vc_ptr, value_cache_shapes, 2, data_type, false);
    Tensor out = make_tensor_external(out_ptr, out_shapes, 2, DataType::FLOAT32);

    uint32_t bt_shapes[2] = {static_cast<uint32_t>(batch), static_cast<uint32_t>(block_num)};
    Tensor block_table =
        make_tensor_external(orch_args.tensor(3).data_as<void>(), bt_shapes, 2, DataType::INT32, false);
    uint32_t cl_shapes[1] = {static_cast<uint32_t>(batch)};
    Tensor context_lens =
        make_tensor_external(orch_args.tensor(4).data_as<void>(), cl_shapes, 1, DataType::INT32, false);

#ifdef ENABLE_PROFILING
    CYCLE_COUNT_LAP(prof_ext_tensor);
#endif

    // Create infos are loop-invariant — shapes depend only on q_tile/head_dim
    uint32_t oi_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(head_dim)};
    uint32_t li_shapes[1] = {static_cast<uint32_t>(q_tile)};
    TensorCreateInfo tile2d_ci(oi_shapes, 2, DataType::FLOAT32);
    TensorCreateInfo scalar_ci(li_shapes, 1, DataType::FLOAT32);
#ifdef ENABLE_PROFILING
    prof_make_count += 2;
    CYCLE_COUNT_LAP(prof_make_tensor);
#endif

    for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
        uint32_t cl_idx[1] = {static_cast<uint32_t>(b_idx)};
        uint64_t cur_seq = static_cast<uint64_t>(get_tensor_data<int32_t>(context_lens, 1, cl_idx));
        uint64_t bn_this_batch = (cur_seq + block_size - 1) / block_size;

        for (uint64_t q_idx = 0; q_idx < q_loop; q_idx++) {
            CYCLE_COUNT_LAP(prof_scope_and_loop);
            PTO2_SCOPE() {
                uint64_t cur_offset = b_idx * q_head_num + q_idx * q_tile;

                uint32_t qi_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(head_dim)};
                uint32_t qi_offsets[2] = {static_cast<uint32_t>(cur_offset), 0};
                Tensor qi = query.view(qi_shapes, qi_offsets);
                uint32_t out_view_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(head_dim)};
                uint32_t out_view_offsets[2] = {static_cast<uint32_t>(cur_offset), 0};
                Tensor out_view = out.view(out_view_shapes, out_view_offsets, true);
#ifdef ENABLE_PROFILING
                prof_view_count += 2;
                CYCLE_COUNT_LAP(prof_tensor_view);
#endif
                CYCLE_COUNT_LAP(prof_param_setup);
                TaskOutputTensors alloc_outs = alloc_tensors(tile2d_ci, scalar_ci, scalar_ci);
                const Tensor &oi = alloc_outs.get_ref(0);
                const Tensor &li_update = alloc_outs.get_ref(1);
                const Tensor &mi_update = alloc_outs.get_ref(2);
#ifdef ENABLE_PROFILING
                prof_submit_count++;
                CYCLE_COUNT_LAP(prof_submit_task);
#endif

                // Reusable Arg objects — reset() before each use avoids
                // repeated stack-frame construction in the inner loop.
                Arg params_qk, params_sf, params_pv, params_up;

                for (uint64_t bn = 0; bn < bn_this_batch; bn += N_UNROLL) {
                    uint64_t n_blocks = std::min(static_cast<uint64_t>(N_UNROLL), bn_this_batch - bn);

                    // Valid length for last block in this group
                    uint64_t last_block_seq_start = (bn + n_blocks - 1) * block_size;
                    uint64_t valid_len_last = std::min(block_size, cur_seq - last_block_seq_start);
                    CYCLE_COUNT_LAP(prof_param_extract);

                    // === Task 1: Batched QK matmul ===
                    uint32_t sij_buf_shapes[2] = {
                        static_cast<uint32_t>(q_tile), static_cast<uint32_t>(n_blocks * block_size)
                    };
                    TensorCreateInfo sij_buf_ci(sij_buf_shapes, 2, DataType::FLOAT32);
#ifdef ENABLE_PROFILING
                    prof_make_count += 1;
                    CYCLE_COUNT_LAP(prof_make_tensor);
#endif

                    params_qk.reset();
                    params_qk.add_input(qi, key_cache, block_table);
                    params_qk.add_output(sij_buf_ci);
                    params_qk.add_scalar(n_blocks, b_idx * block_num + bn);
                    CYCLE_COUNT_LAP(prof_param_setup);
                    TaskOutputTensors qk_outs = rt_submit_aic_task(FUNC_QK_MATMUL, params_qk);
                    const Tensor &sij_buf = qk_outs.get_ref(0);
#ifdef ENABLE_PROFILING
                    prof_submit_count++;
                    CYCLE_COUNT_LAP(prof_submit_task);
#endif

                    // === Task 2: Two-pass softmax over all blocks in group ===
                    uint32_t pij_buf_shapes[2] = {
                        static_cast<uint32_t>(q_tile), static_cast<uint32_t>(n_blocks * block_size)
                    };
                    TensorCreateInfo pij_buf_ci(pij_buf_shapes, 2, data_type);
#ifdef ENABLE_PROFILING
                    prof_make_count += 1;
                    CYCLE_COUNT_LAP(prof_make_tensor);
#endif

                    params_sf.reset();
                    params_sf.add_input(sij_buf);
                    params_sf.add_output(pij_buf_ci, scalar_ci, scalar_ci);
                    params_sf.add_scalar(scale_value, n_blocks, valid_len_last);
                    CYCLE_COUNT_LAP(prof_param_setup);
                    TaskOutputTensors sf_outs = rt_submit_aiv_task(FUNC_SOFTMAX_PREPARE, params_sf);
                    const Tensor &pij_buf = sf_outs.get_ref(0);
                    const Tensor &mi = sf_outs.get_ref(1);
                    const Tensor &li = sf_outs.get_ref(2);
#ifdef ENABLE_PROFILING
                    prof_submit_count++;
                    CYCLE_COUNT_LAP(prof_submit_task);
#endif

                    // === Task 3: SplitK PV matmul (accumulated P @ V) ===
                    params_pv.reset();
                    params_pv.add_input(pij_buf, value_cache, block_table);
                    params_pv.add_output(tile2d_ci);
                    params_pv.add_scalar(n_blocks, b_idx * block_num + bn);
                    CYCLE_COUNT_LAP(prof_param_setup);
                    TaskOutputTensors pv_outs = rt_submit_aic_task(FUNC_PV_MATMUL, params_pv);
                    const Tensor &oi_new = pv_outs.get_ref(0);
#ifdef ENABLE_PROFILING
                    prof_submit_count++;
                    CYCLE_COUNT_LAP(prof_submit_task);
#endif

                    // === Task 4: Online update (per-group) ===
                    uint64_t is_first = (bn == 0) ? 1 : 0;
                    uint64_t is_last = (bn + n_blocks >= bn_this_batch) ? 1 : 0;

                    params_up.reset();
                    params_up.add_input(mi, li, oi_new);
                    params_up.add_inout(mi_update, li_update, oi, out_view);
                    params_up.add_scalar(is_first, is_last);
                    CYCLE_COUNT_LAP(prof_param_setup);
                    rt_submit_aiv_task(FUNC_ONLINE_UPDATE, params_up);
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
    uint64_t total = prof_param_extract + prof_ext_tensor + prof_make_tensor + prof_tensor_view + prof_param_setup +
                     prof_submit_task + prof_scope_and_loop;
    LOG_INFO_V9(
        "=== PagedAttn Orch Profiling: %d submits, %d makes, %d views, total=%.3fus ===", prof_submit_count,
        prof_make_count, prof_view_count, cycles_to_us(total)
    );
    if (total > 0) {
        LOG_INFO_V9(
            "  param_extract    : %7.3fus (%5.1f%%)", cycles_to_us(prof_param_extract),
            prof_param_extract * 100.0 / total
        );
        LOG_INFO_V9(
            "  ext_tensor(x4)   : %7.3fus (%5.1f%%)", cycles_to_us(prof_ext_tensor), prof_ext_tensor * 100.0 / total
        );
        LOG_INFO_V9(
            "  create_info(x%d) : %7.3fus (%5.1f%%)  avg=%.3fus", prof_make_count, cycles_to_us(prof_make_tensor),
            prof_make_tensor * 100.0 / total,
            prof_make_count > 0 ? cycles_to_us(prof_make_tensor) / prof_make_count : 0.0
        );
        LOG_INFO_V9(
            "  tensor_view(x%d) : %7.3fus (%5.1f%%)  avg=%.3fus", prof_view_count, cycles_to_us(prof_tensor_view),
            prof_tensor_view * 100.0 / total,
            prof_view_count > 0 ? cycles_to_us(prof_tensor_view) / prof_view_count : 0.0
        );
        LOG_INFO_V9(
            "  param_setup      : %7.3fus (%5.1f%%)", cycles_to_us(prof_param_setup), prof_param_setup * 100.0 / total
        );
        LOG_INFO_V9(
            "  submit_task(x%d) : %7.3fus (%5.1f%%)  avg=%.3fus", prof_submit_count, cycles_to_us(prof_submit_task),
            prof_submit_task * 100.0 / total,
            prof_submit_count > 0 ? cycles_to_us(prof_submit_task) / prof_submit_count : 0.0
        );
        LOG_INFO_V9(
            "  scope_and_loop   : %7.3fus (%5.1f%%)", cycles_to_us(prof_scope_and_loop),
            prof_scope_and_loop * 100.0 / total
        );
    }
#endif

#undef CYCLE_COUNT_START
#undef CYCLE_COUNT_LAP
}

}  // extern "C"
