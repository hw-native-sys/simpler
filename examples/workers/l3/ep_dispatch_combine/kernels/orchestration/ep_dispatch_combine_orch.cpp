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
 * moe_router + EP dispatch + moe_expert + combine orchestration.
 *
 *   func_id 0..17   moe_router kernels  ffn half-compress pre-mix (hc_pre) +
 *                                       RMSNorm + learned-score gate + top-k +
 *                                       weight normalize. Produces x_norm,
 *                                       indices, weights (plus post_ffn /
 *                                       comb_ffn for hc_post).
 *   func_id 18      dispatch.cpp        EP count exchange + 3-channel push;
 *                                       reads chip-produced x_norm + indices,
 *                                       plus host-packed w_padded / idx_padded.
 *   func_id 19..35  moe_expert kernels  routed local experts + shared expert
 *                                       (task ids +19; recv_expert_count read
 *                                       from host-known tensor(11)).
 *   func_id 36      combine.cpp         TPUT recv_y rows -> routed_y_buf;
 *                                       reduce_sum -> routed_y.
 *
 *   tensor(0)   x_hc              INPUT  (host) [B, S, HC_MULT, D]    BF16
 *   tensor(1)   hc_ffn_fn         INPUT  (host) [MIX_HC, HC_DIM]      FP32
 *   tensor(2)   hc_ffn_scale      INPUT  (host) [3]                   FP32  (orch reads .data_as)
 *   tensor(3)   hc_ffn_base       INPUT  (host) [MIX_HC]              FP32
 *   tensor(4)   norm_w            INPUT  (host) [D]                   FP32
 *   tensor(5)   gate_w            INPUT  (host) [N_EXPERTS, D]        FP32
 *   tensor(6)   gate_bias         INPUT  (host) [N_EXPERTS]           FP32
 *   tensor(7)   tid2eid           INPUT  (host) [VOCAB, TOPK]         INT32  (unused at LAYER_ID >= N_HASH_LAYERS)
 *   tensor(8)   input_ids         INPUT  (host) [B, S]                INT64  (ditto)
 *   tensor(9)   w_padded          INPUT  (host) [T*TOPK, W_PAD=8]     FP32   (packed from golden weights)
 *   tensor(10)  idx_padded        INPUT  (host) [T*TOPK, IDX_PAD=8]   INT32  (packed from golden indices)
 *   tensor(11)  recv_count_host   INPUT  (host) [L, 1]                INT32  (orch reads .data_as for moe_expert loop bounds)
 *   tensor(12)  expert_w1         INPUT  (host) [L, MOE_INTER, D]     INT8
 *   tensor(13)  expert_w1_scale   INPUT  (host) [L, MOE_INTER]        FP32
 *   tensor(14)  expert_w3         INPUT  (host) [L, MOE_INTER, D]     INT8
 *   tensor(15)  expert_w3_scale   INPUT  (host) [L, MOE_INTER]        FP32
 *   tensor(16)  expert_w2         INPUT  (host) [L, D, MOE_INTER]     INT8
 *   tensor(17)  expert_w2_scale   INPUT  (host) [L, D]                FP32
 *   tensor(18)  shared_w1         INPUT  (host) [MOE_INTER, D]        INT8
 *   tensor(19)  shared_w1_scale   INPUT  (host) [MOE_INTER]           FP32
 *   tensor(20)  shared_w3         INPUT  (host) [MOE_INTER, D]        INT8
 *   tensor(21)  shared_w3_scale   INPUT  (host) [MOE_INTER]           FP32
 *   tensor(22)  shared_w2         INPUT  (host) [D, MOE_INTER]        INT8
 *   tensor(23)  shared_w2_scale   INPUT  (host) [D]                   FP32
 *   tensor(24)  x_norm            OUTPUT_EXISTING [T, D]              BF16   (router out -> dispatch + moe_expert)
 *   tensor(25)  indices           OUTPUT_EXISTING [T, TOPK]           INT32  (router out -> dispatch)
 *   tensor(26)  weights           OUTPUT_EXISTING [T, TOPK]           FP32   (router out; verification only)
 *   tensor(27)  post_ffn          OUTPUT_EXISTING [B, S, HC_MULT]     FP32   (router out; unused)
 *   tensor(28)  comb_ffn          OUTPUT_EXISTING [B, S, HC_MULT, HC_MULT] FP32 (router out; unused)
 *   tensor(29)  recv_x_out        OUTPUT_EXISTING [L, R, D]           BF16
 *   tensor(30)  recv_w_out        OUTPUT_EXISTING [L, R]              FP32
 *   tensor(31)  recv_idx_out      OUTPUT_EXISTING [L, R]              INT32
 *   tensor(32)  recv_count_out    OUTPUT_EXISTING [L, 1]              INT32
 *   tensor(33)  recv_y            OUTPUT_EXISTING [L, R, D]           BF16
 *   tensor(34)  sh                OUTPUT_EXISTING [T, D]              BF16
 *   tensor(35)  routed_y          OUTPUT_EXISTING [T, D]              FP32
 *   tensor(36)  scratch           INOUT  HCCL window slot
 *   scalar(0)   nranks
 *   scalar(1)   CommContext device pointer
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
        .expected_arg_count = 39,  // 37 tensors + 2 scalars
    };
}

__attribute__((visibility("default"))) void ep_dispatch_combine_orchestration(const ChipStorageTaskArgs &orch_args) {
    // ---- bind external tensors (kept under the orch's view) ----
    Tensor ext_x_hc = from_tensor_arg(orch_args.tensor(0));
    Tensor ext_hc_ffn_fn = from_tensor_arg(orch_args.tensor(1));
    Tensor ext_hc_ffn_scale = from_tensor_arg(orch_args.tensor(2));
    Tensor ext_hc_ffn_base = from_tensor_arg(orch_args.tensor(3));
    Tensor ext_norm_w = from_tensor_arg(orch_args.tensor(4));
    Tensor ext_gate_w = from_tensor_arg(orch_args.tensor(5));
    Tensor ext_gate_bias = from_tensor_arg(orch_args.tensor(6));
    Tensor ext_tid2eid = from_tensor_arg(orch_args.tensor(7));
    Tensor ext_input_ids = from_tensor_arg(orch_args.tensor(8));
    Tensor w_padded = from_tensor_arg(orch_args.tensor(9));
    Tensor idx_padded = from_tensor_arg(orch_args.tensor(10));
    // tensor(11) recv_count_host is read directly via .data_as inside the moe_expert loops.
    Tensor x_norm = from_tensor_arg(orch_args.tensor(24));
    Tensor indices = from_tensor_arg(orch_args.tensor(25));
    Tensor recv_x_out = from_tensor_arg(orch_args.tensor(29));
    Tensor recv_w_out = from_tensor_arg(orch_args.tensor(30));
    Tensor recv_idx_out = from_tensor_arg(orch_args.tensor(31));
    Tensor recv_count_out = from_tensor_arg(orch_args.tensor(32));
    Tensor recv_y = from_tensor_arg(orch_args.tensor(33));
    Tensor sh = from_tensor_arg(orch_args.tensor(34));
    Tensor routed_y = from_tensor_arg(orch_args.tensor(35));
    Tensor scratch = from_tensor_arg(orch_args.tensor(36));

    // Router output names (PyPTO body refs them as ext_*).
    Tensor ext_x_norm = x_norm;
    Tensor ext_indices = indices;
    Tensor ext_weights = from_tensor_arg(orch_args.tensor(26));
    Tensor ext_post_ffn = from_tensor_arg(orch_args.tensor(27));
    Tensor ext_comb_ffn = from_tensor_arg(orch_args.tensor(28));

    // moe_expert external tensors (names preserved from the generated orch).
    Tensor ext_recv_x = recv_x_out;
    Tensor ext_recv_weights = recv_w_out;
    Tensor ext_x_local = x_norm;
    Tensor ext_expert_w1 = from_tensor_arg(orch_args.tensor(12));
    Tensor ext_expert_w1_scale = from_tensor_arg(orch_args.tensor(13));
    Tensor ext_expert_w3 = from_tensor_arg(orch_args.tensor(14));
    Tensor ext_expert_w3_scale = from_tensor_arg(orch_args.tensor(15));
    Tensor ext_expert_w2 = from_tensor_arg(orch_args.tensor(16));
    Tensor ext_expert_w2_scale = from_tensor_arg(orch_args.tensor(17));
    Tensor ext_shared_w1 = from_tensor_arg(orch_args.tensor(18));
    Tensor ext_shared_w1_scale = from_tensor_arg(orch_args.tensor(19));
    Tensor ext_shared_w3 = from_tensor_arg(orch_args.tensor(20));
    Tensor ext_shared_w3_scale = from_tensor_arg(orch_args.tensor(21));
    Tensor ext_shared_w2 = from_tensor_arg(orch_args.tensor(22));
    Tensor ext_shared_w2_scale = from_tensor_arg(orch_args.tensor(23));
    Tensor ext_recv_y = recv_y;
    Tensor ext_sh = sh;

    // ---- func_id 0..17: moe_router (transplanted PyPTO-generated body) ----
        PTO2_SCOPE() {
            uint32_t x_mixed_ci_shapes[3] = {16, 1, 4096};
            TensorCreateInfo x_mixed_ci(x_mixed_ci_shapes, 3, DataType::BFLOAT16);
            uint32_t x_flat_fp32_inline49_ci_shapes[2] = {16, 16384};
            TensorCreateInfo x_flat_fp32_inline49_ci(x_flat_fp32_inline49_ci_shapes, 2, DataType::FLOAT32);
            uint32_t inv_rms_inline75_ci_shapes[2] = {1, 16};
            TensorCreateInfo inv_rms_inline75_ci(inv_rms_inline75_ci_shapes, 2, DataType::FLOAT32);
            uint32_t mixes_inline34_ci_shapes[2] = {16, 32};
            TensorCreateInfo mixes_inline34_ci(mixes_inline34_ci_shapes, 2, DataType::FLOAT32);
            uint32_t comb_logits_inline19_ci_shapes[2] = {16, 16};
            TensorCreateInfo comb_logits_inline19_ci(comb_logits_inline19_ci_shapes, 2, DataType::FLOAT32);
            uint32_t ret0__out_ci_shapes[2] = {16, 8};
            TensorCreateInfo ret0__out_ci(ret0__out_ci_shapes, 2, DataType::FLOAT32);
            uint32_t ret0__out_1_ci_shapes[2] = {16, 8};
            TensorCreateInfo ret0__out_1_ci(ret0__out_1_ci_shapes, 2, DataType::FLOAT32);
            uint32_t ret0__out_2_ci_shapes[2] = {16, 8};
            TensorCreateInfo ret0__out_2_ci(ret0__out_2_ci_shapes, 2, DataType::FLOAT32);
            uint32_t ret1__out_ci_shapes[2] = {16, 16};
            TensorCreateInfo ret1__out_ci(ret1__out_ci_shapes, 2, DataType::FLOAT32);
            uint32_t inv_rms_inline126_ci_shapes[2] = {1, 16};
            TensorCreateInfo inv_rms_inline126_ci(inv_rms_inline126_ci_shapes, 2, DataType::FLOAT32);
            uint32_t x_norm_bf16_inline111_ci_shapes[2] = {16, 4096};
            TensorCreateInfo x_norm_bf16_inline111_ci(x_norm_bf16_inline111_ci_shapes, 2, DataType::BFLOAT16);
            uint32_t biased_scores_inline131_ci_shapes[2] = {16, 32};
            TensorCreateInfo biased_scores_inline131_ci(biased_scores_inline131_ci_shapes, 2, DataType::FLOAT32);
            uint32_t score_acc_buf_inline117_ci_shapes[2] = {1, 16};
            TensorCreateInfo score_acc_buf_inline117_ci(score_acc_buf_inline117_ci_shapes, 2, DataType::FLOAT32);
            uint32_t sorted_rows_inline97_ci_shapes[2] = {16, 64};
            TensorCreateInfo sorted_rows_inline97_ci(sorted_rows_inline97_ci_shapes, 2, DataType::FLOAT32);
            uint32_t topk_vals_pad_inline110_ci_shapes[2] = {16, 32};
            TensorCreateInfo topk_vals_pad_inline110_ci(topk_vals_pad_inline110_ci_shapes, 2, DataType::FLOAT32);
            uint32_t topk_idx_pad_inline104_ci_shapes[2] = {16, 32};
            TensorCreateInfo topk_idx_pad_inline104_ci(topk_idx_pad_inline104_ci_shapes, 2, DataType::INT32);
            TaskOutputTensors alloc_0 = alloc_tensors(x_mixed_ci, x_flat_fp32_inline49_ci, inv_rms_inline75_ci, mixes_inline34_ci, comb_logits_inline19_ci, ret0__out_ci, ret0__out_1_ci, ret0__out_2_ci, ret1__out_ci, inv_rms_inline126_ci, x_norm_bf16_inline111_ci, biased_scores_inline131_ci, score_acc_buf_inline117_ci, sorted_rows_inline97_ci, topk_vals_pad_inline110_ci, topk_idx_pad_inline104_ci);
            const Tensor& x_mixed = alloc_0.get_ref(0);
            const Tensor& x_flat_fp32_inline49 = alloc_0.get_ref(1);
            const Tensor& inv_rms_inline75 = alloc_0.get_ref(2);
            const Tensor& mixes_inline34 = alloc_0.get_ref(3);
            const Tensor& comb_logits_inline19 = alloc_0.get_ref(4);
            const Tensor& ret0__out = alloc_0.get_ref(5);
            const Tensor& ret0__out_1 = alloc_0.get_ref(6);
            const Tensor& ret0__out_2 = alloc_0.get_ref(7);
            const Tensor& ret1__out = alloc_0.get_ref(8);
            const Tensor& inv_rms_inline126 = alloc_0.get_ref(9);
            const Tensor& x_norm_bf16_inline111 = alloc_0.get_ref(10);
            const Tensor& biased_scores_inline131 = alloc_0.get_ref(11);
            const Tensor& score_acc_buf_inline117 = alloc_0.get_ref(12);
            const Tensor& sorted_rows_inline97 = alloc_0.get_ref(13);
            const Tensor& topk_vals_pad_inline110 = alloc_0.get_ref(14);
            const Tensor& topk_idx_pad_inline104 = alloc_0.get_ref(15);
            uint32_t weight_out_pad_inline108_ci_shapes[2] = {16, 32};
            TensorCreateInfo weight_out_pad_inline108_ci(weight_out_pad_inline108_ci_shapes, 2, DataType::FLOAT32);
            TaskOutputTensors alloc_1 = alloc_tensors(weight_out_pad_inline108_ci);
            const Tensor& weight_out_pad_inline108 = alloc_1.get_ref(0);
            uint32_t x_flat_inline36_shapes[2] = {16, 16384};
            Tensor x_flat_inline36 = ext_x_hc.reshape(x_flat_inline36_shapes, 2);
            uint32_t post_flat_inline51_shapes[1] = {64};
            Tensor post_flat_inline51 = ext_post_ffn.reshape(post_flat_inline51_shapes, 1);
            uint32_t comb_flat_inline37_shapes[1] = {256};
            Tensor comb_flat_inline37 = ext_comb_ffn.reshape(comb_flat_inline37_shapes, 1);
            for (int64_t kb_inline48 = 0; kb_inline48 < 32; kb_inline48 += 1) {
                PTO2_SCOPE() {
                    int64_t k0_inline59 = (kb_inline48 * 512);

                    // Task 0: cast_x
                    Arg params_t0;
                    params_t0.add_input(x_flat_inline36);
                    params_t0.add_output(x_flat_fp32_inline49);
                    params_t0.add_scalar(k0_inline59);
                    rt_submit_aiv_task(0, params_t0);
                    const Tensor& x_flat_fp32_inline49__ssa_v3 = x_flat_fp32_inline49;
                }
            }
            uint32_t ret0__out_3_shapes[2] = {1, 16};
            uint32_t ret0__out_3_offsets[2] = {0, 0};
            Tensor ret0__out_3 = inv_rms_inline75.view(ret0__out_3_shapes, ret0__out_3_offsets);

            // Task 1: rms
            Arg params_t1;
            params_t1.add_input(x_flat_fp32_inline49);
            params_t1.add_output(ret0__out_3);
            rt_submit_aiv_task(1, params_t1);
            uint32_t mixes_flat_inline45_shapes[1] = {512};
            Tensor mixes_flat_inline45 = mixes_inline34.reshape(mixes_flat_inline45_shapes, 1);

            // Task 2: linear
            Arg params_t2;
            params_t2.add_input(x_flat_fp32_inline49);
            params_t2.add_input(ext_hc_ffn_fn);
            params_t2.add_input(inv_rms_inline75);
            params_t2.add_output(mixes_flat_inline45);
            rt_submit_aiv_task(2, params_t2);
            uint32_t mixes_v1_inline58_shapes[2] = {16, 32};
            Tensor mixes_v1_inline58 = mixes_flat_inline45.reshape(mixes_v1_inline58_shapes, 2);
            float scale0_inline21 = static_cast<float*>(orch_args.tensor(2).data_as<void>())[0];
            float scale1_inline17 = static_cast<float*>(orch_args.tensor(2).data_as<void>())[1];
            float scale2_inline26 = static_cast<float*>(orch_args.tensor(2).data_as<void>())[2];

            // Task 3: split_pre_post
            Arg params_t3;
            params_t3.add_output(ret0__out);
            rt_submit_aiv_task(3, params_t3);
            const Tensor& ones_hc_inline79 = ret0__out;
            uint32_t t_shapes[1] = {8};
            uint32_t t_offsets[1] = {0};
            Tensor t = ext_hc_ffn_base.view(t_shapes, t_offsets);
            uint32_t pre_base_inline76_shapes[2] = {1, 8};
            Tensor pre_base_inline76 = t.reshape(pre_base_inline76_shapes, 2);
            uint32_t t__tmp_v26_shapes[2] = {16, 8};
            uint32_t t__tmp_v26_offsets[2] = {0, 0};
            Tensor t__tmp_v26 = mixes_v1_inline58.view(t__tmp_v26_shapes, t__tmp_v26_offsets);

            // Task 4: split_pre_post_0
            Arg params_t4;
            params_t4.add_input(t__tmp_v26);
            params_t4.add_input(ones_hc_inline79);
            params_t4.add_input(pre_base_inline76);
            params_t4.add_output(ret0__out_1);
            params_t4.add_scalar(to_u64(scale0_inline21));
            rt_submit_aiv_task(4, params_t4);
            const Tensor& pre_val_inline62 = ret0__out_1;
            uint32_t t__tmp_v33_shapes[1] = {8};
            uint32_t t__tmp_v33_offsets[1] = {4};
            Tensor t__tmp_v33 = ext_hc_ffn_base.view(t__tmp_v33_shapes, t__tmp_v33_offsets);
            uint32_t post_base_inline50_shapes[2] = {1, 8};
            Tensor post_base_inline50 = t__tmp_v33.reshape(post_base_inline50_shapes, 2);
            uint32_t t__tmp_v34_shapes[2] = {16, 8};
            uint32_t t__tmp_v34_offsets[2] = {0, 4};
            Tensor t__tmp_v34 = mixes_v1_inline58.view(t__tmp_v34_shapes, t__tmp_v34_offsets);

            // Task 5: split_pre_post_1
            Arg params_t5;
            params_t5.add_input(t__tmp_v34);
            params_t5.add_input(ones_hc_inline79);
            params_t5.add_input(post_base_inline50);
            params_t5.add_output(ret0__out_2);
            params_t5.add_output(ret1__out);
            params_t5.add_scalar(to_u64(scale1_inline17));
            rt_submit_aiv_task(5, params_t5);
            const Tensor& post_pad_inline66 = ret0__out_2;
            const Tensor& ones_comb_inline67 = ret1__out;
            uint32_t t__tmp_v41_shapes[1] = {16};
            uint32_t t__tmp_v41_offsets[1] = {8};
            Tensor t__tmp_v41 = ext_hc_ffn_base.view(t__tmp_v41_shapes, t__tmp_v41_offsets);
            uint32_t comb_base_inline70_shapes[2] = {1, 16};
            Tensor comb_base_inline70 = t__tmp_v41.reshape(comb_base_inline70_shapes, 2);
            uint32_t comb_mix_inline41_shapes[2] = {16, 16};
            uint32_t comb_mix_inline41_offsets[2] = {0, 8};
            Tensor comb_mix_inline41 = mixes_v1_inline58.view(comb_mix_inline41_shapes, comb_mix_inline41_offsets);
            uint32_t ret0__out_4_shapes[2] = {16, 16};
            uint32_t ret0__out_4_offsets[2] = {0, 0};
            Tensor ret0__out_4 = comb_logits_inline19.view(ret0__out_4_shapes, ret0__out_4_offsets);

            // Task 6: split_pre_post_2
            Arg params_t6;
            params_t6.add_input(comb_mix_inline41);
            params_t6.add_input(ones_comb_inline67);
            params_t6.add_input(comb_base_inline70);
            params_t6.add_output(ret0__out_4);
            params_t6.add_scalar(to_u64(scale2_inline26));
            rt_submit_aiv_task(6, params_t6);
            uint32_t post_pad_flat_inline73_shapes[1] = {128};
            Tensor post_pad_flat_inline73 = post_pad_inline66.reshape(post_pad_flat_inline73_shapes, 1);

            // Task 7: comb_sinkhorn
            Arg params_t7;
            params_t7.add_input(comb_logits_inline19);
            params_t7.add_output(comb_flat_inline37);
            rt_submit_aiv_task(7, params_t7);
            for (int64_t t_inline7 = 0; t_inline7 < 1; t_inline7 += 1) {
                PTO2_SCOPE() {

                    // Task 8: write_post
                    Arg params_t8;
                    params_t8.add_input(post_pad_flat_inline73);
                    params_t8.add_output(post_flat_inline51);
                    params_t8.add_scalar((uint64_t)0);
                    rt_submit_aiv_task(8, params_t8);
                }
            }
            uint32_t pre_val_flat_inline44_shapes[1] = {128};
            Tensor pre_val_flat_inline44 = pre_val_inline62.reshape(pre_val_flat_inline44_shapes, 1);
            uint32_t x_mixed_view_inline55_shapes[2] = {16, 4096};
            Tensor x_mixed_view_inline55 = x_mixed.reshape(x_mixed_view_inline55_shapes, 2);
            for (int64_t t_inline5 = 0; t_inline5 < 1; t_inline5 += 1) {
                PTO2_SCOPE() {

                    // Task 9: mix_x
                    Arg params_t9;
                    params_t9.add_output(x_mixed_view_inline55);
                    params_t9.add_input(pre_val_flat_inline44);
                    params_t9.add_input(x_flat_fp32_inline49);
                    params_t9.add_scalar((uint64_t)0);
                    rt_submit_aiv_task(9, params_t9);
                    const Tensor& x_mixed_view_inline55__co_l1_rv_v1 = x_mixed_view_inline55;
                }
            }
            uint32_t x_mixed_v1_inline0_shapes[3] = {16, 1, 4096};
            Tensor x_mixed_v1_inline0 = x_mixed_view_inline55.reshape(x_mixed_v1_inline0_shapes, 3);
            uint32_t x_mixed_flat_inline127_shapes[2] = {16, 4096};
            Tensor x_mixed_flat_inline127 = x_mixed.reshape(x_mixed_flat_inline127_shapes, 2);
            uint32_t ret0__out_5_shapes[2] = {1, 16};
            uint32_t ret0__out_5_offsets[2] = {0, 0};
            Tensor ret0__out_5 = inv_rms_inline126.view(ret0__out_5_shapes, ret0__out_5_offsets);

            // Task 10: ffn_norm_rms
            Arg params_t10;
            params_t10.add_input(x_mixed_flat_inline127);
            params_t10.add_output(ret0__out_5);
            rt_submit_aiv_task(10, params_t10);
            for (int64_t db_inline125 = 0; db_inline125 < 8; db_inline125 += 1) {
                PTO2_SCOPE() {

                    // Task 11: ffn_norm_apply
                    Arg params_t11;
                    params_t11.add_input(inv_rms_inline126);
                    params_t11.add_input(x_mixed_flat_inline127);
                    params_t11.add_input(ext_norm_w);
                    params_t11.add_output(x_norm_bf16_inline111);
                    params_t11.add_output(ext_x_norm);
                    params_t11.add_scalar(db_inline125);
                    rt_submit_aiv_task(11, params_t11);
                    const Tensor& x_norm_bf16_inline111__ssa_v3 = x_norm_bf16_inline111;
                    const Tensor& x_norm__ssa_v3 = ext_x_norm;
                }
            }
            uint32_t biased_flat_inline133_shapes[1] = {512};
            Tensor biased_flat_inline133 = biased_scores_inline131.reshape(biased_flat_inline133_shapes, 1);
            uint32_t ret0__out_6_shapes[2] = {16, 32};
            uint32_t ret0__out_6_offsets[2] = {0, 0};
            Tensor ret0__out_6 = biased_scores_inline131.view(ret0__out_6_shapes, ret0__out_6_offsets);

            // Task 12: gate_dot
            Arg params_t12;
            params_t12.add_output(ret0__out_6);
            rt_submit_aiv_task(12, params_t12);

            // Task 13: gate_dot_0
            Arg params_t13;
            params_t13.add_inout(score_acc_buf_inline117);
            params_t13.add_input(x_norm_bf16_inline111);
            params_t13.add_input(ext_gate_w);
            params_t13.add_input(ext_gate_bias);
            params_t13.add_output(biased_flat_inline133);
            rt_submit_aiv_task(13, params_t13);
            uint32_t biased_scores_v1_inline136_shapes[2] = {16, 32};
            Tensor biased_scores_v1_inline136 = biased_flat_inline133.reshape(biased_scores_v1_inline136_shapes, 2);

            // Task 14: route_sort_top2
            Arg params_t14;
            params_t14.add_input(biased_scores_v1_inline136);
            params_t14.add_inout(sorted_rows_inline97);
            rt_submit_aiv_task(14, params_t14);
            const Tensor& sorted_rows_inline97__ssa_v16 = sorted_rows_inline97;

            // Task 15: route_extract_top2
            Arg params_t15;
            params_t15.add_inout(topk_vals_pad_inline110);
            params_t15.add_input(sorted_rows_inline97__ssa_v16);
            params_t15.add_inout(topk_idx_pad_inline104);
            rt_submit_aiv_task(15, params_t15);
            const Tensor& topk_vals_pad_inline110__ssa_v17 = topk_vals_pad_inline110;
            const Tensor& topk_idx_pad_inline104__ssa_v16 = topk_idx_pad_inline104;

            // Task 16: route_normalize_weights
            Arg params_t16;
            params_t16.add_input(topk_vals_pad_inline110__ssa_v17);
            params_t16.add_inout(weight_out_pad_inline108);
            rt_submit_aiv_task(16, params_t16);
            const Tensor& weight_out_pad_inline108__ssa_v1 = weight_out_pad_inline108;
            uint32_t indices_flat_inline89_shapes[1] = {32};
            Tensor indices_flat_inline89 = ext_indices.reshape(indices_flat_inline89_shapes, 1);
            uint32_t weights_flat_inline88_shapes[1] = {32};
            Tensor weights_flat_inline88 = ext_weights.reshape(weights_flat_inline88_shapes, 1);
            uint32_t topk_idx_flat_inline87_shapes[1] = {512};
            Tensor topk_idx_flat_inline87 = topk_idx_pad_inline104__ssa_v16.reshape(topk_idx_flat_inline87_shapes, 1);
            uint32_t weight_out_flat_inline86_shapes[1] = {512};
            Tensor weight_out_flat_inline86 = weight_out_pad_inline108__ssa_v1.reshape(weight_out_flat_inline86_shapes, 1);

            // Task 17: write_route_outputs.
            // NOTE: declare the writes against the *same* C++ Tensor variables
            // that dispatch reads (``indices`` / ``ext_weights``'s storage),
            // not their flat reshape views. The L3 runtime tracks deps by
            // Tensor-object identity, so writes to the reshape view don't
            // establish a happens-before edge from this task to dispatch's
            // later ``add_input(indices)`` — without this dispatch races with
            // the router and reads stale (mostly-zero) indices on a fraction
            // of runs. Buffer addr / offset are unchanged so the generated
            // kernel still indexes the same memory.
            Arg params_t17;
            params_t17.add_input(topk_idx_flat_inline87);
            params_t17.add_output(indices);
            params_t17.add_input(weight_out_flat_inline86);
            params_t17.add_output(ext_weights);
            rt_submit_aiv_task(17, params_t17);
        }

    // ---- func_id 18: dispatch ----
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
        rt_submit_aiv_task(18, p);
    }

    // ---- func_id 19..35: moe_expert (transplanted PyPTO-generated body) ----
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

            // Task 19: x_local_q
            Arg params_t0;
            params_t0.add_input(ext_x_local);
            params_t0.add_output(x_local_i8_inline43);
            params_t0.add_output(ret0__out);
            rt_submit_aiv_task(19, params_t0);
            const Tensor& x_local_i8_inline43__rv_v2 = x_local_i8_inline43;
            const Tensor& x_local_scale_dq_inline32 = ret0__out;
            for (int64_t local_i_inline67 = 0; local_i_inline67 < 8; local_i_inline67 += 1) {
                PTO2_SCOPE() {
                    size_t idx_n_rows_inline68 = local_i_inline67 * 1 + 0;
                    int32_t n_rows_inline68 = static_cast<int32_t*>(orch_args.tensor(11).data_as<void>())[idx_n_rows_inline68];
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

                            // Task 20: recv_x_q
                            Arg params_t1;
                            params_t1.add_input(ext_recv_x);
                            params_t1.add_output(recv_x_tile_i8_inline75);
                            params_t1.add_output(ret0__out_2);
                            params_t1.add_scalar(local_i_inline67);
                            params_t1.add_scalar(t0_inline47);
                            rt_submit_aiv_task(20, params_t1);
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

                                    // Task 21: exp_gate_up_matmul
                                    Arg params_t2;
                                    params_t2.add_input(recv_x_tile_i8_inline75__rv_v2);
                                    params_t2.add_input(ext_expert_w1);
                                    params_t2.add_input(ext_expert_w3);
                                    params_t2.add_output(ret0__out_4);
                                    params_t2.add_output(ret1__out);
                                    params_t2.add_scalar(local_i_inline67);
                                    params_t2.add_scalar(n0_inline72);
                                    rt_submit_aic_task(21, params_t2);
                                    const Tensor& gate_acc_inline69 = ret0__out_4;
                                    const Tensor& up_acc_inline46 = ret1__out;

                                    // Task 22: exp_gate_up_dequant
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
                                    rt_submit_aiv_task(22, params_t3);
                                    const Tensor& gate_2d_v1_inline7 = ret0__out_5;
                                    const Tensor& up_2d_v1_inline78 = ret1__out_1;

                                    // Task 23: exp_swiglu
                                    Arg params_t4;
                                    params_t4.add_input(gate_2d_v1_inline7);
                                    params_t4.add_input(up_2d_v1_inline78);
                                    params_t4.add_input(recv_weights_flat_inline62);
                                    params_t4.add_output(ret0__out_6);
                                    params_t4.add_scalar(flat_t0_inline40);
                                    rt_submit_aiv_task(23, params_t4);
                                    const Tensor& h_chunk_inline89 = ret0__out_6;

                                    // Task 24: exp_swiglu_mask
                                    Arg params_t5;
                                    params_t5.add_input(h_chunk_inline89);
                                    params_t5.add_output(h_tile_fp32_inline18);
                                    params_t5.add_scalar(valid_rows_inline73);
                                    params_t5.add_scalar(n0_inline72);
                                    rt_submit_aiv_task(24, params_t5);
                                    const Tensor& h_tile_fp32_inline18__ssa_v3 = h_tile_fp32_inline18;
                                }
                            }

                            // Task 25: exp_h_q
                            Arg params_t6;
                            params_t6.add_input(h_tile_fp32_inline18);
                            params_t6.add_output(h_tile_i8_inline92);
                            params_t6.add_output(ret0__out_3);
                            rt_submit_aiv_task(25, params_t6);
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

                                    // Task 26: exp_w2_matmul
                                    Arg params_t7;
                                    params_t7.add_input(h_tile_i8_inline92__rv_v2);
                                    params_t7.add_input(ext_expert_w2);
                                    params_t7.add_output(ret0__out_7);
                                    params_t7.add_scalar(local_i_inline67);
                                    params_t7.add_scalar(d0_inline49);
                                    rt_submit_aic_task(26, params_t7);
                                    const Tensor& y_acc_inline109 = ret0__out_7;

                                    // Task 27: exp_w2_dequant
                                    Arg params_t8;
                                    params_t8.add_input(y_acc_inline109);
                                    params_t8.add_input(ext_expert_w2_scale);
                                    params_t8.add_input(h_tile_scale_dq_inline63);
                                    params_t8.add_output(ret0__out_8);
                                    params_t8.add_scalar(local_i_inline67);
                                    params_t8.add_scalar(d0_inline49);
                                    rt_submit_aiv_task(27, params_t8);
                                    const Tensor& y_2d_v1_inline17 = ret0__out_8;

                                    // Task 28: exp_recv_y_write
                                    Arg params_t9;
                                    params_t9.add_input(y_2d_v1_inline17);
                                    params_t9.add_output(recv_y_flat_inline53);
                                    params_t9.add_scalar(flat_t0_inline40);
                                    params_t9.add_scalar(d0_inline49);
                                    rt_submit_aiv_task(28, params_t9);
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

                    // Task 29: sh_gate_up_matmul
                    Arg params_t10;
                    params_t10.add_input(x_local_i8_inline43__rv_v2);
                    params_t10.add_input(ext_shared_w1);
                    params_t10.add_input(ext_shared_w3);
                    params_t10.add_output(ret0__out_9);
                    params_t10.add_output(ret1__out_2);
                    params_t10.add_scalar(n0_inline113);
                    rt_submit_aic_task(29, params_t10);
                    const Tensor& sh_gate_acc_inline95 = ret0__out_9;
                    const Tensor& sh_up_acc_inline118 = ret1__out_2;

                    // Task 30: sh_gate_up_dequant
                    Arg params_t11;
                    params_t11.add_input(ext_shared_w1_scale);
                    params_t11.add_input(ext_shared_w3_scale);
                    params_t11.add_input(sh_gate_acc_inline95);
                    params_t11.add_input(sh_up_acc_inline118);
                    params_t11.add_input(x_local_scale_dq_inline32);
                    params_t11.add_output(ret0__out_10);
                    params_t11.add_output(ret1__out_3);
                    params_t11.add_scalar(n0_inline113);
                    rt_submit_aiv_task(30, params_t11);
                    const Tensor& sh_gate_v1_inline50 = ret0__out_10;
                    const Tensor& sh_up_v1_inline9 = ret1__out_3;

                    // Task 31: sh_swiglu
                    Arg params_t12;
                    params_t12.add_input(sh_gate_v1_inline50);
                    params_t12.add_input(sh_up_v1_inline9);
                    params_t12.add_output(sh_tile_fp32_inline88);
                    params_t12.add_scalar(n0_inline113);
                    rt_submit_aiv_task(31, params_t12);
                    const Tensor& sh_tile_fp32_inline88__ssa_v3 = sh_tile_fp32_inline88;
                }
            }

            // Task 32: sh_h_q
            Arg params_t13;
            params_t13.add_input(sh_tile_fp32_inline88);
            params_t13.add_output(sh_tile_i8_inline126);
            params_t13.add_output(ret0__out_1);
            rt_submit_aiv_task(32, params_t13);
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

                    // Task 33: sh_w2_matmul
                    Arg params_t14;
                    params_t14.add_input(sh_tile_i8_inline126__rv_v2);
                    params_t14.add_input(ext_shared_w2);
                    params_t14.add_output(ret0__out_11);
                    params_t14.add_scalar(d0_inline64);
                    rt_submit_aic_task(33, params_t14);
                    const Tensor& sh_y_acc_inline4 = ret0__out_11;

                    // Task 34: sh_w2_dequant
                    Arg params_t15;
                    params_t15.add_input(ext_shared_w2_scale);
                    params_t15.add_input(sh_y_acc_inline4);
                    params_t15.add_input(sh_tile_scale_dq_inline99);
                    params_t15.add_output(ret0__out_12);
                    params_t15.add_scalar(d0_inline64);
                    rt_submit_aiv_task(34, params_t15);
                    const Tensor& sh_y_v1_inline114 = ret0__out_12;

                    // Task 35: sh_write
                    Arg params_t16;
                    params_t16.add_input(sh_y_v1_inline114);
                    params_t16.add_output(ext_sh);
                    params_t16.add_scalar(d0_inline64);
                    rt_submit_aiv_task(35, params_t16);
                    const Tensor& sh = ext_sh;
                }
            }
        }

    // ---- func_id 36: combine ----
    {
        Arg p;
        p.add_input(recv_y);
        p.add_input(recv_idx_out);
        p.add_output(routed_y);
        p.add_inout(scratch);
        p.add_scalar(orch_args.scalar(0));  // nranks
        p.add_scalar(orch_args.scalar(1));  // CommContext
        rt_submit_aiv_task(36, p);
    }
}

}  // extern "C"
