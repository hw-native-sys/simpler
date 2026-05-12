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
 * Paged Attention Orchestration Function - manual-scope variant
 *
 * Matches the small-case paged_attention orchestration shape while replacing
 * the automatic same-scope dependency wiring with explicit task-to-task deps
 * inside PTO2_SCOPE(PTO2ScopeMode::MANUAL).
 */

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstring>

#include "pto_orchestration_api.h"

#define FUNC_QK_MATMUL 0
#define FUNC_SOFTMAX_PREPARE 1
#define FUNC_PV_MATMUL 2
#define FUNC_ONLINE_UPDATE 3
constexpr uint64_t PLATFORM_PROF_SYS_CNT_FREQ = 50000000;  // 50 MHz

inline double cycles_to_us(uint64_t cycles) {
    return (static_cast<double>(cycles) / PLATFORM_PROF_SYS_CNT_FREQ) * 1000000.0;
}

inline uint64_t get_sys_cnt_aicpu() {
#if defined(__aarch64__)
    uint64_t ticks;
    asm volatile("mrs %0, cntvct_el0" : "=r"(ticks));
    return ticks;
#elif defined(__x86_64__)
    return 0;
#else
    return 0;
#endif
}

#ifdef ENABLE_PROFILING
#define CYCLE_COUNT_START() uint64_t _t0 = get_sys_cnt_aicpu(), _t1
#define CYCLE_COUNT_LAP(acc)       \
    do {                           \
        _t1 = get_sys_cnt_aicpu(); \
        acc += (_t1 - _t0);        \
        _t0 = _t1;                 \
    } while (0)
#define PROF_INC(counter, n) (counter) += (n)
#else
#define CYCLE_COUNT_START() (void)0
#define CYCLE_COUNT_LAP(acc) (void)0
#define PROF_INC(counter, n) (void)0
#endif

extern "C" {

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
    uint64_t prof_scope = 0;
    uint64_t prof_make_tensor = 0;
    uint64_t prof_tensor_view = 0;
    uint64_t prof_param_setup = 0;
    uint64_t prof_submit_task = 0;
    int prof_submit_count = 0;
    int prof_make_count = 0;
    int prof_view_count = 0;
#endif

    CYCLE_COUNT_START();

    // Read dimensions from tensor metadata
    uint64_t batch = orch_args.tensor(0).shapes[0];
    uint64_t num_heads = orch_args.tensor(0).shapes[1];
    uint64_t head_dim = orch_args.tensor(0).shapes[2];
    DataType data_type = orch_args.tensor(0).dtype;

    uint64_t block_size = orch_args.tensor(1).shapes[1];
    uint64_t block_num = orch_args.tensor(3).shapes[1];

    uint64_t scale_value = orch_args.scalar(0);

    uint64_t q_head_num = num_heads;
    uint64_t q_tile = std::min(num_heads, static_cast<uint64_t>(128));
    uint64_t q_loop = (q_head_num + q_tile - 1) / q_tile;
    CYCLE_COUNT_LAP(prof_param_extract);

    LOG_INFO_V9(">>>>>> batch = %" PRIu64, batch);

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
    Tensor query = make_tensor_external(query_ptr, query_shapes, 2, data_type);
    Tensor key_cache = make_tensor_external(kc_ptr, key_cache_shapes, 2, data_type);
    Tensor value_cache = make_tensor_external(vc_ptr, value_cache_shapes, 2, data_type);
    Tensor out = make_tensor_external(out_ptr, out_shapes, 2, DataType::FLOAT32);
    CYCLE_COUNT_LAP(prof_ext_tensor);

    uint32_t bt_shapes[2] = {static_cast<uint32_t>(batch), static_cast<uint32_t>(block_num)};
    Tensor block_table =
        make_tensor_external(orch_args.tensor(3).data_as<void>(), bt_shapes, 2, DataType::INT32, false);
    uint32_t cl_shapes[1] = {static_cast<uint32_t>(batch)};
    Tensor context_lens =
        make_tensor_external(orch_args.tensor(4).data_as<void>(), cl_shapes, 1, DataType::INT32, false);

    // Create infos are loop-invariant — shapes depend only on q_tile/head_dim/block_size
    uint32_t tile2d_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(head_dim)};
    uint32_t scalar_shapes[1] = {static_cast<uint32_t>(q_tile)};
    uint32_t sij_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(block_size)};
    TensorCreateInfo tile2d_ci(tile2d_shapes, 2, DataType::FLOAT32);
    TensorCreateInfo scalar_ci(scalar_shapes, 1, DataType::FLOAT32);
    TensorCreateInfo sij_ci(sij_shapes, 2, DataType::FLOAT32);
    TensorCreateInfo pij_f16_ci(sij_shapes, 2, data_type);

    PROF_INC(prof_make_count, 4);
    CYCLE_COUNT_LAP(prof_make_tensor);

    for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
        uint32_t cl_idx[1] = {static_cast<uint32_t>(b_idx)};
        uint64_t cur_seq = static_cast<uint64_t>(get_tensor_data<int32_t>(context_lens, 1, cl_idx));
        uint64_t bn_this_batch = (cur_seq + block_size - 1) / block_size;
        for (uint64_t q_idx = 0; q_idx < q_loop; q_idx++) {
            PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
                CYCLE_COUNT_LAP(prof_scope);
                uint64_t cur_offset = b_idx * q_head_num + q_idx * q_tile;

                uint32_t qi_offsets[2] = {static_cast<uint32_t>(cur_offset), 0};
                Tensor qi = query.view(tile2d_shapes, qi_offsets);
                uint32_t out_view_offsets[2] = {static_cast<uint32_t>(cur_offset), 0};
                Tensor out_view = out.view(tile2d_shapes, out_view_offsets);
                PROF_INC(prof_view_count, 2);
                CYCLE_COUNT_LAP(prof_tensor_view);

                CYCLE_COUNT_LAP(prof_param_setup);
                TaskOutputTensors alloc_outs = alloc_tensors(tile2d_ci, scalar_ci, scalar_ci);
                const Tensor &oi = alloc_outs.get_ref(0);
                const Tensor &li_update = alloc_outs.get_ref(1);
                const Tensor &mi_update = alloc_outs.get_ref(2);
                PTO2TaskId alloc_task = alloc_outs.task_id();
                PTO2TaskId prev_update_task = PTO2TaskId::invalid();
                PROF_INC(prof_submit_count, 1);
                CYCLE_COUNT_LAP(prof_submit_task);

                for (uint64_t bn = 0; bn < bn_this_batch; bn++) {
                    uint32_t bt_idx[2] = {static_cast<uint32_t>(b_idx), static_cast<uint32_t>(bn)};
                    uint64_t cur_block_idx = static_cast<uint64_t>(get_tensor_data<int32_t>(block_table, 2, bt_idx));
                    uint64_t valid_len = std::min(block_size, cur_seq - bn * block_size);
                    CYCLE_COUNT_LAP(prof_param_extract);

                    uint32_t kv_shapes[2] = {static_cast<uint32_t>(block_size), static_cast<uint32_t>(head_dim)};
                    uint32_t kv_offsets[2] = {static_cast<uint32_t>(cur_block_idx * block_size), 0};
                    Tensor kj = key_cache.view(kv_shapes, kv_offsets);
                    Tensor vj = value_cache.view(kv_shapes, kv_offsets);
                    PROF_INC(prof_view_count, 2);
                    CYCLE_COUNT_LAP(prof_tensor_view);

                    Arg params_qk;
                    params_qk.add_input(qi);
                    params_qk.add_input(kj);
                    params_qk.add_output(sij_ci);
                    CYCLE_COUNT_LAP(prof_param_setup);
                    TaskOutputTensors qk_outs = rt_submit_aic_task(FUNC_QK_MATMUL, params_qk);
                    const Tensor &sij = qk_outs.get_ref(0);
                    PROF_INC(prof_submit_count, 1);
                    CYCLE_COUNT_LAP(prof_submit_task);

                    uint32_t sij_valid_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(valid_len)};
                    uint32_t sij_valid_offsets[2] = {0, 0};
                    Tensor sij_valid = sij.view(sij_valid_shapes, sij_valid_offsets);
                    PROF_INC(prof_view_count, 1);
                    CYCLE_COUNT_LAP(prof_tensor_view);

                    Arg params_sf;
                    params_sf.add_input(sij_valid);
                    params_sf.add_output(pij_f16_ci);
                    params_sf.add_output(scalar_ci);
                    params_sf.add_output(scalar_ci);
                    params_sf.add_dep(qk_outs.task_id());
                    params_sf.add_scalar(scale_value);
                    CYCLE_COUNT_LAP(prof_param_setup);
                    TaskOutputTensors sf_outs = rt_submit_aiv_task(FUNC_SOFTMAX_PREPARE, params_sf);
                    const Tensor &pij_f16 = sf_outs.get_ref(0);
                    const Tensor &mi = sf_outs.get_ref(1);
                    const Tensor &li = sf_outs.get_ref(2);
                    PROF_INC(prof_submit_count, 1);
                    CYCLE_COUNT_LAP(prof_submit_task);

                    Arg params_pv;
                    params_pv.add_input(pij_f16);
                    params_pv.add_input(vj);
                    params_pv.add_output(tile2d_ci);
                    params_pv.add_dep(sf_outs.task_id());
                    CYCLE_COUNT_LAP(prof_param_setup);
                    TaskOutputTensors pv_outs = rt_submit_aic_task(FUNC_PV_MATMUL, params_pv);
                    const Tensor &oi_tmp = pv_outs.get_ref(0);
                    PROF_INC(prof_submit_count, 1);
                    CYCLE_COUNT_LAP(prof_submit_task);

                    uint64_t is_first = (bn == 0) ? 1 : 0;
                    uint64_t is_last = (bn == bn_this_batch - 1) ? 1 : 0;
                    CYCLE_COUNT_LAP(prof_param_extract);

                    Arg params_up;
                    params_up.add_input(mi);
                    params_up.add_input(li);
                    params_up.add_input(oi_tmp);
                    params_up.add_inout(mi_update);
                    params_up.add_inout(li_update);
                    params_up.add_inout(oi);
                    params_up.add_inout(out_view);
                    // UP reads SF's mi/li, but SF -> PV -> UP already orders it; only the PV edge is explicit.
                    params_up.add_dep(pv_outs.task_id());
                    if (prev_update_task.is_valid()) {
                        params_up.add_dep(prev_update_task);
                    }
                    // alloc completes inline; this dep only keeps the scratch buffers alive until the last consumer.
                    if (is_last) {
                        params_up.add_dep(alloc_task);
                    }
                    params_up.add_scalar(is_first);
                    params_up.add_scalar(is_last);
                    CYCLE_COUNT_LAP(prof_param_setup);
                    TaskOutputTensors up_outs = rt_submit_aiv_task(FUNC_ONLINE_UPDATE, params_up);
                    prev_update_task = up_outs.task_id();
                    PROF_INC(prof_submit_count, 1);
                    CYCLE_COUNT_LAP(prof_submit_task);
                }
            }
            CYCLE_COUNT_LAP(prof_scope);
        }
    }

#ifdef ENABLE_PROFILING
    uint64_t total = prof_param_extract + prof_ext_tensor + prof_make_tensor + prof_tensor_view + prof_param_setup +
                     prof_submit_task + prof_scope;
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
        LOG_INFO_V9("  scope            : %7.3fus (%5.1f%%)", cycles_to_us(prof_scope), prof_scope * 100.0 / total);
        LOG_INFO_V9(
            "  submit_task(x%d) : %7.3fus (%5.1f%%)  avg=%.3fus", prof_submit_count, cycles_to_us(prof_submit_task),
            prof_submit_task * 100.0 / total,
            prof_submit_count > 0 ? cycles_to_us(prof_submit_task) / prof_submit_count : 0.0
        );
    }
#endif
}

}  // extern "C"
