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
 * EP dispatch + local-expert + combine orchestration.
 *
 * Three child AIV kernels chained via the runtime; cross-kernel data flows
 * through host-backed device tensors (recv_*_out / recv_y) rather than the
 * HCCL window — only routed_y_buf and combine_done need cross-rank
 * visibility, both still in the shared scratch.
 *
 *   func_id 0  dispatch.cpp        count exchange + 3-channel push
 *                                  + stage-out + recv_count emission
 *   func_id 1  local_expert.cpp    placeholder for moe_expert:
 *                                  recv_y[e, s, :] = recv_x[e, s, :] * recv_w[e, s]
 *   func_id 2  combine.cpp         TPUT recv_y rows by recv_idx_out into
 *                                  routed_y_buf (relies on HCCL window
 *                                  zero-init), barrier, reduce_sum along
 *                                  TOPK -> routed_y FP32
 *
 *   tensor(0)  indices            INPUT             [T, TOPK]            INT32
 *   tensor(1)  x_norm             INPUT             [T, D]               BF16
 *   tensor(2)  w_padded           INPUT             [T*TOPK, W_PAD=8]    FP32
 *   tensor(3)  idx_padded         INPUT             [T*TOPK, IDX_PAD=8]  INT32
 *   tensor(4)  recv_x_out         OUTPUT_EXISTING   [L, R, D]            BF16
 *   tensor(5)  recv_w_out         OUTPUT_EXISTING   [L, R]               FP32
 *   tensor(6)  recv_idx_out       OUTPUT_EXISTING   [L, R]               INT32
 *   tensor(7)  recv_count_out     OUTPUT_EXISTING   [L, 1]               INT32
 *   tensor(8)  recv_y             OUTPUT_EXISTING   [L, R, D]            BF16
 *   tensor(9)  routed_y           OUTPUT_EXISTING   [T, D]               FP32
 *   tensor(10) scratch            INOUT             HCCL window slot
 *   scalar(0)  nranks
 *   scalar(1)  CommContext device pointer
 *
 * Tasks run sequentially because rt_submit_aiv_task dispatches in order.
 * Cross-rank synchronization: dispatch's data_done barrier ends the dispatch
 * step; combine's combine_done barrier (inside combine.cpp) ends combine.
 */

#include <stdint.h>

#include "pto_orchestration_api.h"

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
ep_dispatch_combine_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 13,  // 11 tensors + 2 scalars
    };
}

__attribute__((visibility("default"))) void ep_dispatch_combine_orchestration(const ChipStorageTaskArgs &orch_args) {
    Tensor indices         = from_tensor_arg(orch_args.tensor(0));
    Tensor x_norm          = from_tensor_arg(orch_args.tensor(1));
    Tensor w_padded        = from_tensor_arg(orch_args.tensor(2));
    Tensor idx_padded      = from_tensor_arg(orch_args.tensor(3));
    Tensor recv_x_out      = from_tensor_arg(orch_args.tensor(4));
    Tensor recv_w_out      = from_tensor_arg(orch_args.tensor(5));
    Tensor recv_idx_out    = from_tensor_arg(orch_args.tensor(6));
    Tensor recv_count_out  = from_tensor_arg(orch_args.tensor(7));
    Tensor recv_y          = from_tensor_arg(orch_args.tensor(8));
    Tensor routed_y        = from_tensor_arg(orch_args.tensor(9));
    Tensor scratch         = from_tensor_arg(orch_args.tensor(10));

    // child 0: dispatch
    {
        Arg p;
        p.add_input(indices);
        p.add_input(x_norm);
        p.add_input(w_padded);
        p.add_input(idx_padded);
        p.add_output(recv_x_out);
        p.add_output(recv_w_out);
        p.add_output(recv_idx_out);
        p.add_output(recv_count_out);
        p.add_inout(scratch);
        p.add_scalar(orch_args.scalar(0));   // nranks
        p.add_scalar(orch_args.scalar(1));   // CommContext
        rt_submit_aiv_task(0, p);
    }

    // child 1: local_expert (pure local, host-backed I/O only — no scratch)
    {
        Arg p;
        p.add_input(recv_x_out);
        p.add_input(recv_w_out);
        p.add_input(recv_count_out);
        p.add_output(recv_y);
        p.add_scalar(orch_args.scalar(1));   // CommContext (only for ABI symmetry)
        rt_submit_aiv_task(1, p);
    }

    // child 2: combine (push to routed_y_buf in scratch, barrier, reduce_sum)
    {
        Arg p;
        p.add_input(recv_y);
        p.add_input(recv_idx_out);
        p.add_output(routed_y);
        p.add_inout(scratch);
        p.add_scalar(orch_args.scalar(0));
        p.add_scalar(orch_args.scalar(1));
        rt_submit_aiv_task(2, p);
    }
}

}  // extern "C"
