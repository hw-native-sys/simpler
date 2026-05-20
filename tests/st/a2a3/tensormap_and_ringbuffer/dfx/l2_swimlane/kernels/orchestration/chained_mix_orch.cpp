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
 * Chained MIX orchestration — three MIX tasks where each step reads the
 * previous step's output. Purpose-built for the l2_swimlane differential
 * gate: produces MIX tasks (multiple perf rows per task_id) AND non-zero
 * deps.json edges, so the ``seen_tids`` dedup in
 * ``compute_dag_stats_from_deps`` has an arithmetically observable effect.
 *
 * Each MIX task runs ``aic_matmul`` (kernel_matmul.cpp, 128x128 GEMM) and
 * ``aiv_add`` (kernel_add.cpp, elementwise add). Inputs are reshaped from
 * the flat 16384-element tensors the test allocates.
 *
 * Arg layout (8 args, all 1-D 16384 float32 except workspaces which are
 * 32768):
 *   [A, B, D, E, ws_aic, ws_aiv, aic_out, aiv_out]
 *
 * Chain (each line is one MIX task; AIC ↑ AIV ↓):
 *   step 1:  ws_aic[0:T]      ← matmul(A, B)              edges: (none)
 *            ws_aiv[0:T]      ← add(D, E)
 *   step 2:  ws_aic[T:2T]     ← matmul(ws_aic[0:T], B)    edges: 1→2 (×2 tensors)
 *            ws_aiv[T:2T]     ← add(ws_aiv[0:T], E)
 *   step 3:  aic_out          ← matmul(ws_aic[T:2T], B)   edges: 2→3 (×2 tensors)
 *            aiv_out          ← add(ws_aiv[T:2T], E)
 *
 * dep_gen collapses the per-tensor flows to unique (pred, succ) pairs,
 * so deps.json reports 2 edges: (step1, step2) and (step2, step3).
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_MATMUL 0  // AIC kernel — reads first 3 args of the MIX bundle
#define FUNC_ADD 1     // AIV kernel — reads next 3 args of the MIX bundle

static constexpr uint32_t TILE_ELEMS = 128 * 128;

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 8,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    Tensor ext_A = from_tensor_arg(orch_args.tensor(0));
    Tensor ext_B = from_tensor_arg(orch_args.tensor(1));
    Tensor ext_D = from_tensor_arg(orch_args.tensor(2));
    Tensor ext_E = from_tensor_arg(orch_args.tensor(3));
    Tensor ext_ws_aic = from_tensor_arg(orch_args.tensor(4));
    Tensor ext_ws_aiv = from_tensor_arg(orch_args.tensor(5));
    Tensor ext_aic_out = from_tensor_arg(orch_args.tensor(6));
    Tensor ext_aiv_out = from_tensor_arg(orch_args.tensor(7));

    uint32_t slot_shape[1] = {TILE_ELEMS};
    uint32_t off_slot0[1] = {0};
    uint32_t off_slot1[1] = {TILE_ELEMS};

    Tensor ws_aic_slot0 = ext_ws_aic.view(slot_shape, off_slot0);
    Tensor ws_aic_slot1 = ext_ws_aic.view(slot_shape, off_slot1);
    Tensor ws_aiv_slot0 = ext_ws_aiv.view(slot_shape, off_slot0);
    Tensor ws_aiv_slot1 = ext_ws_aiv.view(slot_shape, off_slot1);

    LOG_INFO_V0("[chained_mix_orch] launching 3-step chained MIX (AIC + AIV)");

    // Step 1: heads of both chains read external inputs.
    {
        MixedKernels mk;
        mk.aic_kernel_id = FUNC_MATMUL;
        mk.aiv0_kernel_id = FUNC_ADD;
        Arg args;
        args.add_input(ext_A);
        args.add_input(ext_B);
        args.add_output(ws_aic_slot0);
        args.add_input(ext_D);
        args.add_input(ext_E);
        args.add_output(ws_aiv_slot0);
        rt_submit_task(mk, args);
    }

    // Step 2: AIC reads ws_aic_slot0 (step 1 AIC output) and AIV reads
    // ws_aiv_slot0 (step 1 AIV output). Two tensors flow from step 1 to
    // step 2; dep_gen collapses to a single (step1, step2) edge.
    {
        MixedKernels mk;
        mk.aic_kernel_id = FUNC_MATMUL;
        mk.aiv0_kernel_id = FUNC_ADD;
        Arg args;
        args.add_input(ws_aic_slot0);
        args.add_input(ext_B);
        args.add_output(ws_aic_slot1);
        args.add_input(ws_aiv_slot0);
        args.add_input(ext_E);
        args.add_output(ws_aiv_slot1);
        rt_submit_task(mk, args);
    }

    // Step 3: writes the final user-visible outputs.
    {
        MixedKernels mk;
        mk.aic_kernel_id = FUNC_MATMUL;
        mk.aiv0_kernel_id = FUNC_ADD;
        Arg args;
        args.add_input(ws_aic_slot1);
        args.add_input(ext_B);
        args.add_output(ext_aic_out);
        args.add_input(ws_aiv_slot1);
        args.add_input(ext_E);
        args.add_output(ext_aiv_out);
        rt_submit_task(mk, args);
    }
}

}  // extern "C"
