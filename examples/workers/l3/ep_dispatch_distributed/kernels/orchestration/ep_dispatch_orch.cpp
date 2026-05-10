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
 * EP dispatch orchestration — passes 9 tensors + 2 scalars to the AIV kernel.
 *
 * Three-channel push (x / weight / idx) per Step 2 of decode_ep_dispatch_plan:
 *
 *   tensor(0) indices         INPUT             [T, TOPK]            INT32
 *   tensor(1) x_norm          INPUT             [T, D]               BF16
 *   tensor(2) w_padded        INPUT             [T*TOPK, W_PAD=8]    FP32
 *                                                 host pre-pads each row to
 *                                                 [weight, 0, 0, …, 0]
 *   tensor(3) idx_padded      INPUT             [T*TOPK, IDX_PAD=8]  INT32
 *                                                 host pre-packs each row to
 *                                                 [r, 0, 0, …, 0] where
 *                                                 r = t * TOPK + k
 *   tensor(4) recv_x_out      OUTPUT_EXISTING   [L, R, D]            BF16
 *   tensor(5) recv_w_out      OUTPUT_EXISTING   [L, R]               FP32
 *                                                 kernel Phase 4 compacts
 *                                                 column 0 of the [L,R,W_PAD]
 *                                                 window via TROWSUM
 *   tensor(6) recv_idx_out    OUTPUT_EXISTING   [L, R]               INT32
 *                                                 same column-0 compaction
 *                                                 (scalar copy fallback —
 *                                                 INT32 TROWSUM hangs on a2a3)
 *   tensor(7) recv_count_out  OUTPUT_EXISTING   [L, 1]               INT32
 *                                                 kernel Phase 2 emits
 *                                                 sum_s pub_counts[s][me][e]
 *   tensor(8) scratch         INOUT             HCCL window slot
 *   scalar(0) nranks
 *   scalar(1) CommContext device pointer
 */

#include <stdint.h>

#include "pto_orchestration_api.h"

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
ep_dispatch_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 11,  // 9 tensors + 2 scalars
    };
}

__attribute__((visibility("default"))) void ep_dispatch_orchestration(const ChipStorageTaskArgs &orch_args) {
    Tensor indices         = from_tensor_arg(orch_args.tensor(0));
    Tensor x_norm          = from_tensor_arg(orch_args.tensor(1));
    Tensor w_padded        = from_tensor_arg(orch_args.tensor(2));
    Tensor idx_padded      = from_tensor_arg(orch_args.tensor(3));
    Tensor recv_x_out      = from_tensor_arg(orch_args.tensor(4));
    Tensor recv_w_out      = from_tensor_arg(orch_args.tensor(5));
    Tensor recv_idx_out    = from_tensor_arg(orch_args.tensor(6));
    Tensor recv_count_out  = from_tensor_arg(orch_args.tensor(7));
    Tensor scratch         = from_tensor_arg(orch_args.tensor(8));

    Arg params;
    params.add_input(indices);
    params.add_input(x_norm);
    params.add_input(w_padded);
    params.add_input(idx_padded);
    params.add_output(recv_x_out);
    params.add_output(recv_w_out);
    params.add_output(recv_idx_out);
    params.add_output(recv_count_out);
    params.add_inout(scratch);
    params.add_scalar(orch_args.scalar(0));   // nranks
    params.add_scalar(orch_args.scalar(1));   // CommContext
    rt_submit_aiv_task(0, params);
}

}  // extern "C"
