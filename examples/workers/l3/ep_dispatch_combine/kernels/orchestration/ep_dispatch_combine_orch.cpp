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
 * EP dispatch + moe_expert + combine orchestration.
 *
 * Three stages chained through host-backed device tensors:
 *
 *   func_id 0       dispatch.cpp        count exchange + 3-channel push + stage-out
 *                                       + recv_count emission
 *   func_id 1..17   moe_expert kernels  routed local experts (per-tile A8 matmul
 *                                       chain) + shared expert; produces recv_y
 *                                       (routed partial) and sh (shared output).
 *                                       The body below is the PyPTO-generated
 *                                       moe_expert_test orchestration, transplanted
 *                                       verbatim with task ids shifted by +1 and
 *                                       recv_expert_count read from the host-known
 *                                       tensor(4) instead of a kernel output.
 *   func_id 18      combine.cpp         TPUT recv_y rows by recv_idx_out into
 *                                       routed_y_buf, barrier, reduce_sum -> routed_y
 *
 *   tensor(0)  indices            INPUT             [T, TOPK]               INT32
 *   tensor(1)  x_norm / x_local   INPUT             [T, D]                  BF16
 *   tensor(2)  w_padded           INPUT             [T*TOPK, W_PAD=8]       FP32
 *   tensor(3)  idx_padded         INPUT             [T*TOPK, IDX_PAD=8]     INT32
 *   tensor(4)  recv_count_host    INPUT             [L, 1]                  INT32  (orch loop bounds)
 *   tensor(5)  expert_w1          INPUT             [L, MOE_INTER, D]       INT8
 *   tensor(6)  expert_w1_scale    INPUT             [L, MOE_INTER]          FP32
 *   tensor(7)  expert_w3          INPUT             [L, MOE_INTER, D]       INT8
 *   tensor(8)  expert_w3_scale    INPUT             [L, MOE_INTER]          FP32
 *   tensor(9)  expert_w2          INPUT             [L, D, MOE_INTER]       INT8
 *   tensor(10) expert_w2_scale    INPUT             [L, D]                  FP32
 *   tensor(11) shared_w1          INPUT             [MOE_INTER, D]          INT8
 *   tensor(12) shared_w1_scale    INPUT             [MOE_INTER]             FP32
 *   tensor(13) shared_w3          INPUT             [MOE_INTER, D]          INT8
 *   tensor(14) shared_w3_scale    INPUT             [MOE_INTER]             FP32
 *   tensor(15) shared_w2          INPUT             [D, MOE_INTER]          INT8
 *   tensor(16) shared_w2_scale    INPUT             [D]                     FP32
 *   tensor(17) recv_x_out         OUTPUT_EXISTING   [L, R, D]               BF16   (dispatch out / moe_expert in)
 *   tensor(18) recv_w_out         OUTPUT_EXISTING   [L, R]                  FP32   (dispatch out / moe_expert recv_weights)
 *   tensor(19) recv_idx_out       OUTPUT_EXISTING   [L, R]                  INT32  (dispatch out / combine in)
 *   tensor(20) recv_count_out     OUTPUT_EXISTING   [L, 1]                  INT32  (dispatch out / host verify)
 *   tensor(21) recv_y             OUTPUT_EXISTING   [L, R, D]               BF16   (moe_expert out / combine in)
 *   tensor(22) sh                 OUTPUT_EXISTING   [T, D]                  BF16   (moe_expert out)
 *   tensor(23) routed_y           OUTPUT_EXISTING   [T, D]                  FP32   (combine out)
 *   tensor(24) scratch            INOUT             HCCL window slot
 *   scalar(0)  nranks
 *   scalar(1)  CommContext device pointer
 */

#include "runtime.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "pto_orchestration_api.h"

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
ep_dispatch_combine_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 27,  // 25 tensors + 2 scalars
    };
}

__attribute__((visibility("default"))) void ep_dispatch_combine_orchestration(const ChipStorageTaskArgs &orch_args) {
    // Dispatch / combine wiring + the cross-kernel host-backed tensors.
    Tensor indices = from_tensor_arg(orch_args.tensor(0));
    Tensor x_norm = from_tensor_arg(orch_args.tensor(1));  // doubles as moe_expert's x_local
    Tensor w_padded = from_tensor_arg(orch_args.tensor(2));
    Tensor idx_padded = from_tensor_arg(orch_args.tensor(3));
    // tensor(4) recv_count_host is read directly via .data_as inside the moe_expert loops.
    Tensor recv_x_out = from_tensor_arg(orch_args.tensor(17));
    Tensor recv_w_out = from_tensor_arg(orch_args.tensor(18));
    Tensor recv_idx_out = from_tensor_arg(orch_args.tensor(19));
    Tensor recv_count_out = from_tensor_arg(orch_args.tensor(20));
    Tensor recv_y = from_tensor_arg(orch_args.tensor(21));
    Tensor sh = from_tensor_arg(orch_args.tensor(22));
    Tensor routed_y = from_tensor_arg(orch_args.tensor(23));
    Tensor scratch = from_tensor_arg(orch_args.tensor(24));

    // moe_expert external tensors (names preserved from the generated orch).
    Tensor ext_recv_x = recv_x_out;
    Tensor ext_recv_weights = recv_w_out;
    Tensor ext_x_local = x_norm;
    Tensor ext_expert_w1 = from_tensor_arg(orch_args.tensor(5));
    Tensor ext_expert_w1_scale = from_tensor_arg(orch_args.tensor(6));
    Tensor ext_expert_w3 = from_tensor_arg(orch_args.tensor(7));
    Tensor ext_expert_w3_scale = from_tensor_arg(orch_args.tensor(8));
    Tensor ext_expert_w2 = from_tensor_arg(orch_args.tensor(9));
    Tensor ext_expert_w2_scale = from_tensor_arg(orch_args.tensor(10));
    Tensor ext_shared_w1 = from_tensor_arg(orch_args.tensor(11));
    Tensor ext_shared_w1_scale = from_tensor_arg(orch_args.tensor(12));
    Tensor ext_shared_w3 = from_tensor_arg(orch_args.tensor(13));
    Tensor ext_shared_w3_scale = from_tensor_arg(orch_args.tensor(14));
    Tensor ext_shared_w2 = from_tensor_arg(orch_args.tensor(15));
    Tensor ext_shared_w2_scale = from_tensor_arg(orch_args.tensor(16));
    Tensor ext_recv_y = recv_y;
    Tensor ext_sh = sh;

    // ---- func_id 0: dispatch ----
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
        p.add_scalar(orch_args.scalar(0));  // nranks
        p.add_scalar(orch_args.scalar(1));  // CommContext
        rt_submit_aiv_task(0, p);
    }

    // ---- func_id 1..17: moe_expert (transplanted PyPTO-generated body) ----
        PTO2_SCOPE() {
            uint32_t x_local_i8_inline43_ci_shapes[2] = {16, 4096};
            TensorCreateInfo x_local_i8_inline43_ci(x_local_i8_inline43_ci_shapes, 2, DataType::INT8);
            uint32_t ret0__out_ci_shapes[2] = {16, 1};
            TensorCreateInfo ret0__out_ci(ret0__out_ci_shapes, 2, DataType::FLOAT32);
            uint32_t sh_tile_fp32_inline88_ci_shapes[2] = {16, 4096};
            TensorCreateInfo sh_tile_fp32_inline88_ci(sh_tile_fp32_inline88_ci_shapes, 2, DataType::FLOAT32);
            uint32_t sh_tile_i8_inline126_ci_shapes[2] = {16, 4096};
            TensorCreateInfo sh_tile_i8_inline126_ci(sh_tile_i8_inline126_ci_shapes, 2, DataType::INT8);
            uint32_t ret0__out_1_ci_shapes[2] = {16, 1};
            TensorCreateInfo ret0__out_1_ci(ret0__out_1_ci_shapes, 2, DataType::FLOAT32);
            TaskOutputTensors alloc_0 = alloc_tensors(x_local_i8_inline43_ci, ret0__out_ci, sh_tile_fp32_inline88_ci, sh_tile_i8_inline126_ci, ret0__out_1_ci);
            const Tensor& x_local_i8_inline43 = alloc_0.get_ref(0);
            const Tensor& ret0__out = alloc_0.get_ref(1);
            const Tensor& sh_tile_fp32_inline88 = alloc_0.get_ref(2);
            const Tensor& sh_tile_i8_inline126 = alloc_0.get_ref(3);
            const Tensor& ret0__out_1 = alloc_0.get_ref(4);
            uint32_t recv_y_flat_inline53_shapes[2] = {256, 4096};
            Tensor recv_y_flat_inline53 = ext_recv_y.reshape(recv_y_flat_inline53_shapes, 2);
            uint32_t recv_weights_flat_inline62_shapes[2] = {256, 1};
            Tensor recv_weights_flat_inline62 = ext_recv_weights.reshape(recv_weights_flat_inline62_shapes, 2);

            // Task 1: x_local_q
            Arg params_t0;
            params_t0.add_input(ext_x_local);
            params_t0.add_output(x_local_i8_inline43);
            params_t0.add_output(ret0__out);
            rt_submit_aiv_task(1, params_t0);
            const Tensor& x_local_i8_inline43__rv_v2 = x_local_i8_inline43;
            const Tensor& x_local_scale_dq_inline32 = ret0__out;
            for (int64_t local_i_inline67 = 0; local_i_inline67 < 8; local_i_inline67 += 1) {
                PTO2_SCOPE() {
                    size_t idx_n_rows_inline68 = local_i_inline67 * 1 + 0;
                    int32_t n_rows_inline68 = static_cast<int32_t*>(orch_args.tensor(4).data_as<void>())[idx_n_rows_inline68];
                    int64_t n_tiles_inline91 = ((static_cast<int64_t>(n_rows_inline68) + 15) / 16);
                    int64_t flat_base_inline30 = (local_i_inline67 * 32);
                    for (int64_t t_inline108 = 0; t_inline108 < n_tiles_inline91; t_inline108 += 1) {
                        PTO2_SCOPE() {
                            uint32_t recv_x_tile_i8_inline75_ci_shapes[2] = {16, 4096};
                            TensorCreateInfo recv_x_tile_i8_inline75_ci(recv_x_tile_i8_inline75_ci_shapes, 2, DataType::INT8);
                            uint32_t ret0__out_2_ci_shapes[2] = {16, 1};
                            TensorCreateInfo ret0__out_2_ci(ret0__out_2_ci_shapes, 2, DataType::FLOAT32);
                            uint32_t h_tile_fp32_inline18_ci_shapes[2] = {16, 4096};
                            TensorCreateInfo h_tile_fp32_inline18_ci(h_tile_fp32_inline18_ci_shapes, 2, DataType::FLOAT32);
                            uint32_t h_tile_i8_inline92_ci_shapes[2] = {16, 4096};
                            TensorCreateInfo h_tile_i8_inline92_ci(h_tile_i8_inline92_ci_shapes, 2, DataType::INT8);
                            uint32_t ret0__out_3_ci_shapes[2] = {16, 1};
                            TensorCreateInfo ret0__out_3_ci(ret0__out_3_ci_shapes, 2, DataType::FLOAT32);
                            TaskOutputTensors alloc_1 = alloc_tensors(recv_x_tile_i8_inline75_ci, ret0__out_2_ci, h_tile_fp32_inline18_ci, h_tile_i8_inline92_ci, ret0__out_3_ci);
                            const Tensor& recv_x_tile_i8_inline75 = alloc_1.get_ref(0);
                            const Tensor& ret0__out_2 = alloc_1.get_ref(1);
                            const Tensor& h_tile_fp32_inline18 = alloc_1.get_ref(2);
                            const Tensor& h_tile_i8_inline92 = alloc_1.get_ref(3);
                            const Tensor& ret0__out_3 = alloc_1.get_ref(4);
                            int64_t t0_inline47 = (t_inline108 * 16);
                            int64_t flat_t0_inline40 = (flat_base_inline30 + t0_inline47);
                            int64_t valid_rows_inline73 = std::min<int64_t>((static_cast<int64_t>(n_rows_inline68) - t0_inline47), 16);

                            // Task 2: recv_x_q
                            Arg params_t1;
                            params_t1.add_input(ext_recv_x);
                            params_t1.add_output(recv_x_tile_i8_inline75);
                            params_t1.add_output(ret0__out_2);
                            params_t1.add_scalar(local_i_inline67);
                            params_t1.add_scalar(t0_inline47);
                            rt_submit_aiv_task(2, params_t1);
                            const Tensor& recv_x_tile_i8_inline75__rv_v2 = recv_x_tile_i8_inline75;
                            const Tensor& recv_x_scale_dq_inline29 = ret0__out_2;
                            for (int64_t n0_inline72 = 0; n0_inline72 < 4096; n0_inline72 += 256) {
                                PTO2_SCOPE() {
                                    uint32_t ret0__out_4_ci_shapes[3] = {1, 16, 256};
                                    TensorCreateInfo ret0__out_4_ci(ret0__out_4_ci_shapes, 3, DataType::INT32);
                                    uint32_t ret1__out_ci_shapes[3] = {1, 16, 256};
                                    TensorCreateInfo ret1__out_ci(ret1__out_ci_shapes, 3, DataType::INT32);
                                    uint32_t ret0__out_5_ci_shapes[2] = {16, 256};
                                    TensorCreateInfo ret0__out_5_ci(ret0__out_5_ci_shapes, 2, DataType::FLOAT32);
                                    uint32_t ret1__out_1_ci_shapes[2] = {16, 256};
                                    TensorCreateInfo ret1__out_1_ci(ret1__out_1_ci_shapes, 2, DataType::FLOAT32);
                                    uint32_t ret0__out_6_ci_shapes[2] = {16, 256};
                                    TensorCreateInfo ret0__out_6_ci(ret0__out_6_ci_shapes, 2, DataType::FLOAT32);
                                    TaskOutputTensors alloc_2 = alloc_tensors(ret0__out_4_ci, ret1__out_ci, ret0__out_5_ci, ret1__out_1_ci, ret0__out_6_ci);
                                    const Tensor& ret0__out_4 = alloc_2.get_ref(0);
                                    const Tensor& ret1__out = alloc_2.get_ref(1);
                                    const Tensor& ret0__out_5 = alloc_2.get_ref(2);
                                    const Tensor& ret1__out_1 = alloc_2.get_ref(3);
                                    const Tensor& ret0__out_6 = alloc_2.get_ref(4);

                                    // Task 3: exp_gate_up_matmul
                                    Arg params_t2;
                                    params_t2.add_input(recv_x_tile_i8_inline75__rv_v2);
                                    params_t2.add_input(ext_expert_w1);
                                    params_t2.add_input(ext_expert_w3);
                                    params_t2.add_output(ret0__out_4);
                                    params_t2.add_output(ret1__out);
                                    params_t2.add_scalar(local_i_inline67);
                                    params_t2.add_scalar(n0_inline72);
                                    rt_submit_aic_task(3, params_t2);
                                    const Tensor& gate_acc_inline69 = ret0__out_4;
                                    const Tensor& up_acc_inline46 = ret1__out;

                                    // Task 4: exp_gate_up_dequant
                                    Arg params_t3;
                                    params_t3.add_input(gate_acc_inline69);
                                    params_t3.add_input(up_acc_inline46);
                                    params_t3.add_input(ext_expert_w1_scale);
                                    params_t3.add_input(ext_expert_w3_scale);
                                    params_t3.add_input(recv_x_scale_dq_inline29);
                                    params_t3.add_output(ret0__out_5);
                                    params_t3.add_output(ret1__out_1);
                                    params_t3.add_scalar(local_i_inline67);
                                    params_t3.add_scalar(n0_inline72);
                                    rt_submit_aiv_task(4, params_t3);
                                    const Tensor& gate_2d_v1_inline7 = ret0__out_5;
                                    const Tensor& up_2d_v1_inline78 = ret1__out_1;

                                    // Task 5: exp_swiglu
                                    Arg params_t4;
                                    params_t4.add_input(gate_2d_v1_inline7);
                                    params_t4.add_input(up_2d_v1_inline78);
                                    params_t4.add_input(recv_weights_flat_inline62);
                                    params_t4.add_output(ret0__out_6);
                                    params_t4.add_scalar(flat_t0_inline40);
                                    rt_submit_aiv_task(5, params_t4);
                                    const Tensor& h_chunk_inline89 = ret0__out_6;

                                    // Task 6: exp_swiglu_mask
                                    Arg params_t5;
                                    params_t5.add_input(h_chunk_inline89);
                                    params_t5.add_output(h_tile_fp32_inline18);
                                    params_t5.add_scalar(valid_rows_inline73);
                                    params_t5.add_scalar(n0_inline72);
                                    rt_submit_aiv_task(6, params_t5);
                                    const Tensor& h_tile_fp32_inline18__ssa_v3 = h_tile_fp32_inline18;
                                }
                            }

                            // Task 7: exp_h_q
                            Arg params_t6;
                            params_t6.add_input(h_tile_fp32_inline18);
                            params_t6.add_output(h_tile_i8_inline92);
                            params_t6.add_output(ret0__out_3);
                            rt_submit_aiv_task(7, params_t6);
                            const Tensor& h_tile_i8_inline92__rv_v2 = h_tile_i8_inline92;
                            const Tensor& h_tile_scale_dq_inline63 = ret0__out_3;
                            for (int64_t d0_inline49 = 0; d0_inline49 < 4096; d0_inline49 += 512) {
                                PTO2_SCOPE() {
                                    uint32_t ret0__out_7_ci_shapes[3] = {1, 16, 512};
                                    TensorCreateInfo ret0__out_7_ci(ret0__out_7_ci_shapes, 3, DataType::INT32);
                                    uint32_t ret0__out_8_ci_shapes[2] = {16, 512};
                                    TensorCreateInfo ret0__out_8_ci(ret0__out_8_ci_shapes, 2, DataType::FLOAT32);
                                    TaskOutputTensors alloc_3 = alloc_tensors(ret0__out_7_ci, ret0__out_8_ci);
                                    const Tensor& ret0__out_7 = alloc_3.get_ref(0);
                                    const Tensor& ret0__out_8 = alloc_3.get_ref(1);

                                    // Task 8: exp_w2_matmul
                                    Arg params_t7;
                                    params_t7.add_input(h_tile_i8_inline92__rv_v2);
                                    params_t7.add_input(ext_expert_w2);
                                    params_t7.add_output(ret0__out_7);
                                    params_t7.add_scalar(local_i_inline67);
                                    params_t7.add_scalar(d0_inline49);
                                    rt_submit_aic_task(8, params_t7);
                                    const Tensor& y_acc_inline109 = ret0__out_7;

                                    // Task 9: exp_w2_dequant
                                    Arg params_t8;
                                    params_t8.add_input(y_acc_inline109);
                                    params_t8.add_input(ext_expert_w2_scale);
                                    params_t8.add_input(h_tile_scale_dq_inline63);
                                    params_t8.add_output(ret0__out_8);
                                    params_t8.add_scalar(local_i_inline67);
                                    params_t8.add_scalar(d0_inline49);
                                    rt_submit_aiv_task(9, params_t8);
                                    const Tensor& y_2d_v1_inline17 = ret0__out_8;

                                    // Task 10: exp_recv_y_write
                                    Arg params_t9;
                                    params_t9.add_input(y_2d_v1_inline17);
                                    params_t9.add_output(recv_y_flat_inline53);
                                    params_t9.add_scalar(flat_t0_inline40);
                                    params_t9.add_scalar(d0_inline49);
                                    rt_submit_aiv_task(10, params_t9);
                                    const Tensor& recv_y_flat_inline53__ssa_v7 = recv_y_flat_inline53;
                                }
                            }
                        }
                    }
                }
            }
            for (int64_t n0_inline113 = 0; n0_inline113 < 4096; n0_inline113 += 256) {
                PTO2_SCOPE() {
                    uint32_t ret0__out_9_ci_shapes[2] = {16, 256};
                    TensorCreateInfo ret0__out_9_ci(ret0__out_9_ci_shapes, 2, DataType::INT32);
                    uint32_t ret1__out_2_ci_shapes[2] = {16, 256};
                    TensorCreateInfo ret1__out_2_ci(ret1__out_2_ci_shapes, 2, DataType::INT32);
                    uint32_t ret0__out_10_ci_shapes[2] = {16, 256};
                    TensorCreateInfo ret0__out_10_ci(ret0__out_10_ci_shapes, 2, DataType::FLOAT32);
                    uint32_t ret1__out_3_ci_shapes[2] = {16, 256};
                    TensorCreateInfo ret1__out_3_ci(ret1__out_3_ci_shapes, 2, DataType::FLOAT32);
                    TaskOutputTensors alloc_4 = alloc_tensors(ret0__out_9_ci, ret1__out_2_ci, ret0__out_10_ci, ret1__out_3_ci);
                    const Tensor& ret0__out_9 = alloc_4.get_ref(0);
                    const Tensor& ret1__out_2 = alloc_4.get_ref(1);
                    const Tensor& ret0__out_10 = alloc_4.get_ref(2);
                    const Tensor& ret1__out_3 = alloc_4.get_ref(3);

                    // Task 11: sh_gate_up_matmul
                    Arg params_t10;
                    params_t10.add_input(x_local_i8_inline43__rv_v2);
                    params_t10.add_input(ext_shared_w1);
                    params_t10.add_input(ext_shared_w3);
                    params_t10.add_output(ret0__out_9);
                    params_t10.add_output(ret1__out_2);
                    params_t10.add_scalar(n0_inline113);
                    rt_submit_aic_task(11, params_t10);
                    const Tensor& sh_gate_acc_inline95 = ret0__out_9;
                    const Tensor& sh_up_acc_inline118 = ret1__out_2;

                    // Task 12: sh_gate_up_dequant
                    Arg params_t11;
                    params_t11.add_input(ext_shared_w1_scale);
                    params_t11.add_input(ext_shared_w3_scale);
                    params_t11.add_input(sh_gate_acc_inline95);
                    params_t11.add_input(sh_up_acc_inline118);
                    params_t11.add_input(x_local_scale_dq_inline32);
                    params_t11.add_output(ret0__out_10);
                    params_t11.add_output(ret1__out_3);
                    params_t11.add_scalar(n0_inline113);
                    rt_submit_aiv_task(12, params_t11);
                    const Tensor& sh_gate_v1_inline50 = ret0__out_10;
                    const Tensor& sh_up_v1_inline9 = ret1__out_3;

                    // Task 13: sh_swiglu
                    Arg params_t12;
                    params_t12.add_input(sh_gate_v1_inline50);
                    params_t12.add_input(sh_up_v1_inline9);
                    params_t12.add_output(sh_tile_fp32_inline88);
                    params_t12.add_scalar(n0_inline113);
                    rt_submit_aiv_task(13, params_t12);
                    const Tensor& sh_tile_fp32_inline88__ssa_v3 = sh_tile_fp32_inline88;
                }
            }

            // Task 14: sh_h_q
            Arg params_t13;
            params_t13.add_input(sh_tile_fp32_inline88);
            params_t13.add_output(sh_tile_i8_inline126);
            params_t13.add_output(ret0__out_1);
            rt_submit_aiv_task(14, params_t13);
            const Tensor& sh_tile_i8_inline126__rv_v2 = sh_tile_i8_inline126;
            const Tensor& sh_tile_scale_dq_inline99 = ret0__out_1;
            for (int64_t d0_inline64 = 0; d0_inline64 < 4096; d0_inline64 += 512) {
                PTO2_SCOPE() {
                    uint32_t ret0__out_11_ci_shapes[2] = {16, 512};
                    TensorCreateInfo ret0__out_11_ci(ret0__out_11_ci_shapes, 2, DataType::INT32);
                    uint32_t ret0__out_12_ci_shapes[2] = {16, 512};
                    TensorCreateInfo ret0__out_12_ci(ret0__out_12_ci_shapes, 2, DataType::FLOAT32);
                    TaskOutputTensors alloc_5 = alloc_tensors(ret0__out_11_ci, ret0__out_12_ci);
                    const Tensor& ret0__out_11 = alloc_5.get_ref(0);
                    const Tensor& ret0__out_12 = alloc_5.get_ref(1);

                    // Task 15: sh_w2_matmul
                    Arg params_t14;
                    params_t14.add_input(sh_tile_i8_inline126__rv_v2);
                    params_t14.add_input(ext_shared_w2);
                    params_t14.add_output(ret0__out_11);
                    params_t14.add_scalar(d0_inline64);
                    rt_submit_aic_task(15, params_t14);
                    const Tensor& sh_y_acc_inline4 = ret0__out_11;

                    // Task 16: sh_w2_dequant
                    Arg params_t15;
                    params_t15.add_input(ext_shared_w2_scale);
                    params_t15.add_input(sh_y_acc_inline4);
                    params_t15.add_input(sh_tile_scale_dq_inline99);
                    params_t15.add_output(ret0__out_12);
                    params_t15.add_scalar(d0_inline64);
                    rt_submit_aiv_task(16, params_t15);
                    const Tensor& sh_y_v1_inline114 = ret0__out_12;

                    // Task 17: sh_write
                    Arg params_t16;
                    params_t16.add_input(sh_y_v1_inline114);
                    params_t16.add_output(ext_sh);
                    params_t16.add_scalar(d0_inline64);
                    rt_submit_aiv_task(17, params_t16);
                    const Tensor& sh = ext_sh;
                }
            }
        }

    // ---- func_id 18: combine ----
    {
        Arg p;
        p.add_input(recv_y);
        p.add_input(recv_idx_out);
        p.add_output(routed_y);
        p.add_inout(scratch);
        p.add_scalar(orch_args.scalar(0));  // nranks
        p.add_scalar(orch_args.scalar(1));  // CommContext
        rt_submit_aiv_task(18, p);
    }
}

}  // extern "C"
