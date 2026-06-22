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
struct ProfCounters {
    uint64_t param_extract = 0;
    uint64_t ext_tensor = 0;
    uint64_t make_tensor = 0;
    uint64_t tensor_view = 0;
    uint64_t param_setup = 0;
    uint64_t submit_task = 0;
    uint64_t scope_and_loop = 0;
    int submit_count = 0;
    int make_count = 0;
    int view_count = 0;
    // Running lap timestamps. File-global so the lap timeline stays continuous
    // across the entry/process_qtile_scope() boundary — orchestration runs on a
    // single thread, so a shared counter needs no synchronization.
    uint64_t t0 = 0;
    uint64_t t1 = 0;
};
static ProfCounters g_prof;
#define CYCLE_COUNT_START() (g_prof.t0 = get_sys_cnt_aicpu())
#define CYCLE_COUNT_LAP(acc)              \
    do {                                  \
        g_prof.t1 = get_sys_cnt_aicpu();  \
        (acc) += (g_prof.t1 - g_prof.t0); \
        g_prof.t0 = g_prof.t1;            \
    } while (0)
#else
#define CYCLE_COUNT_START() (void)0
#define CYCLE_COUNT_LAP(acc) (void)0
#endif

/**
 * Submit the QK -> softmax -> PV -> update task chain for one (batch, q-tile) unit.
 *
 * All context is passed positionally through a transport `Arg` (built by the
 * caller, never submitted — only its slots are read back here). Every tensor
 * slot is a materialized Tensor; the Arg carries no TensorCreateInfo (the
 * scope's create-infos are rebuilt locally from the q_tile/head_dim scalars):
 *   tensors: 0 query, 1 key_cache, 2 value_cache, 3 block_table (inputs),
 *            4 out (output buffer the update task writes — add_output(Tensor))
 *   scalars: 0 b_idx, 1 q_idx, 2 q_head_num, 3 q_tile, 4 head_dim,
 *            5 block_size, 6 block_num, 7 scale_value, 8 bn_this_batch,
 *            9 cur_seq, 10 data_type
 * Adding/removing a slot here must be mirrored at the caller's build site.
 *
 * Must run inside a PTO2_SCOPE: the alloc'd / submitted tensors it references
 * do not outlive that scope.
 */
static void process_qtile_scope(const Arg &ctx) {
    const Tensor &query = *ctx.tensor(0).ptr;
    const Tensor &key_cache = *ctx.tensor(1).ptr;
    const Tensor &value_cache = *ctx.tensor(2).ptr;
    const Tensor &block_table = *ctx.tensor(3).ptr;
    const Tensor &out = *ctx.tensor(4).ptr;
    uint64_t b_idx = ctx.scalar(0);
    uint64_t q_idx = ctx.scalar(1);
    uint64_t q_head_num = ctx.scalar(2);
    uint64_t q_tile = ctx.scalar(3);
    uint64_t head_dim = ctx.scalar(4);
    uint64_t block_size = ctx.scalar(5);
    uint64_t block_num = ctx.scalar(6);
    uint64_t scale_value = ctx.scalar(7);
    uint64_t bn_this_batch = ctx.scalar(8);
    uint64_t cur_seq = ctx.scalar(9);
    DataType data_type = static_cast<DataType>(ctx.scalar(10));

    CYCLE_COUNT_START();

    // Create infos for the per-scope accumulators — shapes depend only on
    // q_tile/head_dim, so build once before the block loop. Kept out of the
    // transport Arg, which carries only materialized Tensors.
    uint32_t oi_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(head_dim)};
    uint32_t li_shapes[1] = {static_cast<uint32_t>(q_tile)};
    TensorCreateInfo tile2d_ci(oi_shapes, 2, DataType::FLOAT32);
    TensorCreateInfo scalar_ci(li_shapes, 1, DataType::FLOAT32);
#ifdef ENABLE_PROFILING
    g_prof.make_count += 2;
    CYCLE_COUNT_LAP(g_prof.make_tensor);
#endif

    uint64_t cur_offset = b_idx * q_head_num + q_idx * q_tile;

    uint32_t qi_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(head_dim)};
    uint32_t qi_offsets[2] = {static_cast<uint32_t>(cur_offset), 0};
    Tensor qi = query.view(qi_shapes, qi_offsets);
    uint32_t out_view_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(head_dim)};
    uint32_t out_view_offsets[2] = {static_cast<uint32_t>(cur_offset), 0};
    Tensor out_view = out.view(out_view_shapes, out_view_offsets, true);
#ifdef ENABLE_PROFILING
    g_prof.view_count += 2;
    CYCLE_COUNT_LAP(g_prof.tensor_view);
#endif
    CYCLE_COUNT_LAP(g_prof.param_setup);
    TaskOutputTensors alloc_outs = alloc_tensors(tile2d_ci, scalar_ci, scalar_ci);
    const Tensor &oi = alloc_outs.get_ref(0);
    const Tensor &li_update = alloc_outs.get_ref(1);
    const Tensor &mi_update = alloc_outs.get_ref(2);
#ifdef ENABLE_PROFILING
    g_prof.submit_count++;
    CYCLE_COUNT_LAP(g_prof.submit_task);
#endif

    // Reusable Arg objects — reset() before each use avoids
    // repeated stack-frame construction in the inner loop.
    Arg params_qk, params_sf, params_pv, params_up;

    for (uint64_t bn = 0; bn < bn_this_batch; bn += N_UNROLL) {
        uint64_t n_blocks = std::min(static_cast<uint64_t>(N_UNROLL), bn_this_batch - bn);

        // Valid length for last block in this group
        uint64_t last_block_seq_start = (bn + n_blocks - 1) * block_size;
        uint64_t valid_len_last = std::min(block_size, cur_seq - last_block_seq_start);
        CYCLE_COUNT_LAP(g_prof.param_extract);

        // === Task 1: Batched QK matmul ===
        uint32_t sij_buf_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(n_blocks * block_size)};
        TensorCreateInfo sij_buf_ci(sij_buf_shapes, 2, DataType::FLOAT32);
#ifdef ENABLE_PROFILING
        g_prof.make_count += 1;
        CYCLE_COUNT_LAP(g_prof.make_tensor);
#endif

        params_qk.reset();
        params_qk.add_input(qi, key_cache, block_table);
        params_qk.add_output(sij_buf_ci);
        params_qk.add_scalar(n_blocks, b_idx * block_num + bn);
        CYCLE_COUNT_LAP(g_prof.param_setup);
        TaskOutputTensors qk_outs = rt_submit_aic_task(FUNC_QK_MATMUL, params_qk);
        const Tensor &sij_buf = qk_outs.get_ref(0);
#ifdef ENABLE_PROFILING
        g_prof.submit_count++;
        CYCLE_COUNT_LAP(g_prof.submit_task);
#endif

        // === Task 2: Two-pass softmax over all blocks in group ===
        uint32_t pij_buf_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(n_blocks * block_size)};
        TensorCreateInfo pij_buf_ci(pij_buf_shapes, 2, data_type);
#ifdef ENABLE_PROFILING
        g_prof.make_count += 1;
        CYCLE_COUNT_LAP(g_prof.make_tensor);
#endif

        params_sf.reset();
        params_sf.add_input(sij_buf);
        params_sf.add_output(pij_buf_ci, scalar_ci, scalar_ci);
        params_sf.add_scalar(scale_value, n_blocks, valid_len_last);
        CYCLE_COUNT_LAP(g_prof.param_setup);
        TaskOutputTensors sf_outs = rt_submit_aiv_task(FUNC_SOFTMAX_PREPARE, params_sf);
        const Tensor &pij_buf = sf_outs.get_ref(0);
        const Tensor &mi = sf_outs.get_ref(1);
        const Tensor &li = sf_outs.get_ref(2);
#ifdef ENABLE_PROFILING
        g_prof.submit_count++;
        CYCLE_COUNT_LAP(g_prof.submit_task);
#endif

        // === Task 3: SplitK PV matmul (accumulated P @ V) ===
        params_pv.reset();
        params_pv.add_input(pij_buf, value_cache, block_table);
        params_pv.add_output(tile2d_ci);
        params_pv.add_scalar(n_blocks, b_idx * block_num + bn);
        CYCLE_COUNT_LAP(g_prof.param_setup);
        TaskOutputTensors pv_outs = rt_submit_aic_task(FUNC_PV_MATMUL, params_pv);
        const Tensor &oi_new = pv_outs.get_ref(0);
#ifdef ENABLE_PROFILING
        g_prof.submit_count++;
        CYCLE_COUNT_LAP(g_prof.submit_task);
#endif

        // === Task 4: Online update (per-group) ===
        uint64_t is_first = (bn == 0) ? 1 : 0;
        uint64_t is_last = (bn + n_blocks >= bn_this_batch) ? 1 : 0;

        params_up.reset();
        params_up.add_input(mi, li, oi_new);
        params_up.add_inout(mi_update, li_update, oi, out_view);
        params_up.add_scalar(is_first, is_last);
        CYCLE_COUNT_LAP(g_prof.param_setup);
        rt_submit_aiv_task(FUNC_ONLINE_UPDATE, params_up);
#ifdef ENABLE_PROFILING
        g_prof.submit_count++;
        CYCLE_COUNT_LAP(g_prof.submit_task);
#endif
    }
}

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
    g_prof = ProfCounters{};  // reset per entry — single-threaded orchestration
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
    CYCLE_COUNT_LAP(g_prof.param_extract);

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
    CYCLE_COUNT_LAP(g_prof.ext_tensor);
#endif

    // Transport Arg reused across iterations — packs the scope's context for
    // process_qtile_scope(); see that function for the positional slot layout.
    // It carries only materialized Tensors (no TensorCreateInfo); the scope's
    // create-infos are rebuilt inside the helper from the q_tile/head_dim scalars.
    Arg ctx;

    for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
        uint32_t cl_idx[1] = {static_cast<uint32_t>(b_idx)};
        uint64_t cur_seq = static_cast<uint64_t>(get_tensor_data<int32_t>(context_lens, 1, cl_idx));
        uint64_t bn_this_batch = (cur_seq + block_size - 1) / block_size;

        for (uint64_t q_idx = 0; q_idx < q_loop; q_idx++) {
            CYCLE_COUNT_LAP(g_prof.scope_and_loop);

            ctx.reset();
            ctx.add_input(query, key_cache, value_cache, block_table);
            ctx.add_output(out);
            ctx.add_scalar(
                b_idx, q_idx, q_head_num, q_tile, head_dim, block_size, block_num, scale_value, bn_this_batch, cur_seq,
                static_cast<uint64_t>(data_type)
            );

            PTO2_SCOPE() { process_qtile_scope(ctx); }
        }
    }
    CYCLE_COUNT_LAP(g_prof.scope_and_loop);

#ifdef ENABLE_PROFILING
    uint64_t total = g_prof.param_extract + g_prof.ext_tensor + g_prof.make_tensor + g_prof.tensor_view +
                     g_prof.param_setup + g_prof.submit_task + g_prof.scope_and_loop;
    LOG_INFO_V9(
        "=== PagedAttn Orch Profiling: %d submits, %d makes, %d views, total=%.3fus ===", g_prof.submit_count,
        g_prof.make_count, g_prof.view_count, cycles_to_us(total)
    );
    if (total > 0) {
        LOG_INFO_V9(
            "  param_extract    : %7.3fus (%5.1f%%)", cycles_to_us(g_prof.param_extract),
            g_prof.param_extract * 100.0 / total
        );
        LOG_INFO_V9(
            "  ext_tensor(x4)   : %7.3fus (%5.1f%%)", cycles_to_us(g_prof.ext_tensor), g_prof.ext_tensor * 100.0 / total
        );
        LOG_INFO_V9(
            "  create_info(x%d) : %7.3fus (%5.1f%%)  avg=%.3fus", g_prof.make_count, cycles_to_us(g_prof.make_tensor),
            g_prof.make_tensor * 100.0 / total,
            g_prof.make_count > 0 ? cycles_to_us(g_prof.make_tensor) / g_prof.make_count : 0.0
        );
        LOG_INFO_V9(
            "  tensor_view(x%d) : %7.3fus (%5.1f%%)  avg=%.3fus", g_prof.view_count, cycles_to_us(g_prof.tensor_view),
            g_prof.tensor_view * 100.0 / total,
            g_prof.view_count > 0 ? cycles_to_us(g_prof.tensor_view) / g_prof.view_count : 0.0
        );
        LOG_INFO_V9(
            "  param_setup      : %7.3fus (%5.1f%%)", cycles_to_us(g_prof.param_setup),
            g_prof.param_setup * 100.0 / total
        );
        LOG_INFO_V9(
            "  submit_task(x%d) : %7.3fus (%5.1f%%)  avg=%.3fus", g_prof.submit_count, cycles_to_us(g_prof.submit_task),
            g_prof.submit_task * 100.0 / total,
            g_prof.submit_count > 0 ? cycles_to_us(g_prof.submit_task) / g_prof.submit_count : 0.0
        );
        LOG_INFO_V9(
            "  scope_and_loop   : %7.3fus (%5.1f%%)", cycles_to_us(g_prof.scope_and_loop),
            g_prof.scope_and_loop * 100.0 / total
        );
    }
#endif

#undef CYCLE_COUNT_START
#undef CYCLE_COUNT_LAP
}

}  // extern "C"
