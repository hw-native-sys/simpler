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
 * SPMD Paged Attention Orchestration with TPUSH/TPOP (fixed block_num=24)
 *
 * Submits a single MixedKernels task with hardware block_num fixed at 24.
 * total_logical_blocks = batch * q_loop logical work items are distributed
 * across the 24 hardware blocks via a stride loop inside the kernel:
 *   for (block_idx = hw_block_idx; block_idx < total_logical_blocks; block_idx += 24)
 *
 * q_tile adapts to num_heads at runtime: q_tile = min(num_heads, MAX_Q_TILE).
 * When num_heads <= MAX_Q_TILE (=64), q_loop = 1 and each block processes all heads.
 *
 * Each iteration of the stride loop processes one (batch_idx, q_tile_idx) logical
 * position, running the full AIC/AIV cooperative pipeline via TPUSH/TPOP pipes:
 *   AIC: QK matmul -> TPUSH(sij) -> TPOP(pij) -> PV matmul -> TPUSH(oi_new)
 *   AIV: TPOP(sij) -> online softmax -> TPUSH(pij) -> TPOP(oi_new) -> online update
 *
 * GM FIFO buffers for TPUSH/TPOP are sized for the 24 hardware blocks (not for
 * total_logical_blocks). Each hardware block owns its own FIFO slots and reuses
 * them across stride-loop iterations.
 */

#include <stddef.h>
#include <stdint.h>

#include <cinttypes>

#include "pto_orchestration_api.h"

#define FUNC_PA_AIC 0
#define FUNC_PA_AIV 1

static constexpr uint64_t MAX_Q_TILE = 64;
static constexpr uint64_t HEAD_DIM = 128;
static constexpr uint64_t MAX_BLOCK_SIZE = 128;
static constexpr int16_t SPMD_BLOCK_NUM = 24;

// GM FIFO slot sizes (must match kernel's PAConfig<MAX_Q_TILE> constants).
// Sized for the maximum (q_tile, block_size) so the same FIFO layout works
// for both q_tile=16 and q_tile=64 dispatch paths inside the kernel.
static constexpr uint32_t SIJ_SLOT_SIZE = MAX_Q_TILE * MAX_BLOCK_SIZE * sizeof(float);
static constexpr uint32_t PIJ_SLOT_SIZE = MAX_Q_TILE * MAX_BLOCK_SIZE * sizeof(uint16_t);
static constexpr uint32_t OI_SLOT_SIZE = MAX_Q_TILE * HEAD_DIM * sizeof(float);
static constexpr uint32_t FIFO_DEPTH = 2;

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 7,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    uint64_t batch = orch_args.tensor(0).shapes[0];
    uint64_t num_heads = orch_args.tensor(0).shapes[1];
    uint64_t head_dim = orch_args.tensor(0).shapes[2];
    DataType data_type = orch_args.tensor(0).dtype;

    uint64_t block_size = orch_args.tensor(1).shapes[1];
    uint64_t max_num_blocks_per_req = orch_args.tensor(3).shapes[1];
    uint64_t scale_value = orch_args.scalar(0);

    // q_tile adapts to num_heads: use 64 when num_heads >= 64, else 16.
    // The kernel statically dispatches on q_tile == 16 vs 64.
    uint64_t q_tile = (num_heads >= MAX_Q_TILE) ? MAX_Q_TILE : 16;
    uint64_t q_loop = (num_heads + q_tile - 1) / q_tile;
    int64_t total_logical_blocks = static_cast<int64_t>(batch * q_loop);

    LOG_INFO(
        "SPMD PA TPUSH/TPOP: batch=%" PRIu64 " heads=%" PRIu64 " hd=%" PRIu64 " bs=%" PRIu64 " q_tile=%" PRIu64
        " q_loop=%" PRIu64 " hw_blocks=%d logical_blocks=%" PRId64,
        batch, num_heads, head_dim, block_size, q_tile, q_loop, SPMD_BLOCK_NUM, total_logical_blocks
    );

    // Wrap host tensors
    void *query_ptr = orch_args.tensor(0).data_as<void>();
    void *kc_ptr = orch_args.tensor(1).data_as<void>();
    void *vc_ptr = orch_args.tensor(2).data_as<void>();
    void *out_ptr = orch_args.tensor(5).data_as<void>();

    uint64_t total_kv_blocks = orch_args.tensor(1).shapes[0];
    uint64_t kv_total_rows = total_kv_blocks * block_size;

    uint32_t query_shapes[2] = {static_cast<uint32_t>(batch * num_heads), static_cast<uint32_t>(head_dim)};
    uint32_t kv_shapes[2] = {static_cast<uint32_t>(kv_total_rows), static_cast<uint32_t>(head_dim)};
    uint32_t out_shapes[2] = {static_cast<uint32_t>(batch * num_heads), static_cast<uint32_t>(head_dim)};

    Tensor query = make_tensor_external(query_ptr, query_shapes, 2, data_type);
    Tensor key_cache = make_tensor_external(kc_ptr, kv_shapes, 2, data_type);
    Tensor value_cache = make_tensor_external(vc_ptr, kv_shapes, 2, data_type);
    Tensor out = make_tensor_external(out_ptr, out_shapes, 2, DataType::FLOAT32);

    uint32_t bt_shapes[2] = {static_cast<uint32_t>(batch), static_cast<uint32_t>(max_num_blocks_per_req)};
    Tensor block_table =
        make_tensor_external(orch_args.tensor(3).data_as<void>(), bt_shapes, 2, DataType::INT32, false);
    uint32_t cl_shapes[1] = {static_cast<uint32_t>(batch)};
    Tensor context_lens =
        make_tensor_external(orch_args.tensor(4).data_as<void>(), cl_shapes, 1, DataType::INT32, false);

    // GM FIFO buffers for TPUSH/TPOP (one set of slots per hardware block)
    uint32_t sij_fifo_total = static_cast<uint32_t>(SPMD_BLOCK_NUM) * SIJ_SLOT_SIZE * FIFO_DEPTH;
    uint32_t pij_fifo_total = static_cast<uint32_t>(SPMD_BLOCK_NUM) * PIJ_SLOT_SIZE * FIFO_DEPTH;
    uint32_t oi_fifo_total = static_cast<uint32_t>(SPMD_BLOCK_NUM) * OI_SLOT_SIZE * FIFO_DEPTH;

    // Allocate as 1D byte tensors (using INT32 for 4-byte alignment, divide by 4)
    uint32_t sij_fifo_shapes[1] = {sij_fifo_total / sizeof(int32_t)};
    uint32_t pij_fifo_shapes[1] = {pij_fifo_total / sizeof(int32_t)};
    uint32_t oi_fifo_shapes[1] = {oi_fifo_total / sizeof(int32_t)};

    TensorCreateInfo sij_fifo_ci(sij_fifo_shapes, 1, DataType::INT32);
    TensorCreateInfo pij_fifo_ci(pij_fifo_shapes, 1, DataType::INT32);
    TensorCreateInfo oi_fifo_ci(oi_fifo_shapes, 1, DataType::INT32);

    PTO2_SCOPE() {
        Arg args;
        args.add_input(query);
        args.add_input(key_cache);
        args.add_input(value_cache);
        args.add_input(block_table);
        args.add_input(context_lens);
        args.add_inout(out);
        args.add_output(sij_fifo_ci);
        args.add_output(pij_fifo_ci);
        args.add_output(oi_fifo_ci);
        args.add_scalar(scale_value);
        args.add_scalar(static_cast<int64_t>(num_heads));
        args.add_scalar(static_cast<int64_t>(head_dim));
        args.add_scalar(static_cast<int64_t>(block_size));
        args.add_scalar(static_cast<int64_t>(max_num_blocks_per_req));
        args.add_scalar(static_cast<int64_t>(q_loop));
        args.add_scalar(total_logical_blocks);
        args.add_scalar(static_cast<int64_t>(q_tile));
        args.launch_spec.set_block_num(SPMD_BLOCK_NUM);

        MixedKernels mk;
        mk.aic_kernel_id = FUNC_PA_AIC;
        mk.aiv0_kernel_id = FUNC_PA_AIV;
        mk.aiv1_kernel_id = FUNC_PA_AIV;
        rt_submit_task(mk, args);
    }

    LOG_INFO(
        "SPMD PA TPUSH/TPOP: submitted 1 MixedKernels task, hw_blocks=%d logical=%" PRId64,
        static_cast<int>(SPMD_BLOCK_NUM), total_logical_blocks
    );
}

}  // extern "C"
