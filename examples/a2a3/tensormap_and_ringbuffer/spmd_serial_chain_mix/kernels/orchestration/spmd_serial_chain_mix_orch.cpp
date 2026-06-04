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
 * SPMD serial-chain orchestration — MIX variant (AIC + AIV0 + AIV1).
 *
 * Same shape as the AIV-only sibling (spmd_serial_chain_spin): 4 tasks chained
 * t0 → t1 → t2 → t3, per-task input counts 0, 4, 8, 12, block_num=24. The
 * difference is that every task is submitted as a MIX bundle so each block
 * occupies a full cluster (1 AIC + 2 AIV cores). With block_num=24 each task
 * therefore occupies all 72 AICore cores in lockstep.
 *
 * AIC + AIV0 + AIV1 share the same Arg list; each subtask kernel is the
 * matching get_sys_cnt() spin and ignores the args, so the kernel signatures
 * don't need to match the args' semantics — they just need to be present.
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_SPIN_AIC 0
#define FUNC_SPIN_AIV 1

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 1,
    };
}

static inline TaskOutputTensors submit_mix(Arg &args, int16_t block_num) {
    MixedKernels mk;
    mk.aic_kernel_id = FUNC_SPIN_AIC;
    mk.aiv0_kernel_id = FUNC_SPIN_AIV;
    mk.aiv1_kernel_id = FUNC_SPIN_AIV;
    args.launch_spec.set_block_num(block_num);
    return rt_submit_task(mk, args);
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    Tensor ext_out = from_tensor_arg(orch_args.tensor(0));

    uint32_t SIZE = orch_args.tensor(0).shapes[0];
    uint32_t inter_shapes[1] = {SIZE};
    TensorCreateInfo inter_ci(inter_shapes, 1, DataType::FLOAT32);

    constexpr int16_t kBlockNum = 24;

    // t0: 0 inputs, 1 output. Chain head.
    Arg t0_args;
    t0_args.add_output(inter_ci);
    TaskOutputTensors t0_outs = submit_mix(t0_args, kBlockNum);
    const Tensor &t0_out = t0_outs.get_ref(0);

    // t1: 4 inputs (all = t0_out, collapses to a single dep edge).
    Arg t1_args;
    for (int i = 0; i < 4; ++i)
        t1_args.add_input(t0_out);
    t1_args.add_output(inter_ci);
    TaskOutputTensors t1_outs = submit_mix(t1_args, kBlockNum);
    const Tensor &t1_out = t1_outs.get_ref(0);

    // t2: 8 inputs.
    Arg t2_args;
    for (int i = 0; i < 8; ++i)
        t2_args.add_input(t1_out);
    t2_args.add_output(inter_ci);
    TaskOutputTensors t2_outs = submit_mix(t2_args, kBlockNum);
    const Tensor &t2_out = t2_outs.get_ref(0);

    // t3: 12 inputs, output = external ext_out.
    Arg t3_args;
    for (int i = 0; i < 12; ++i)
        t3_args.add_input(t2_out);
    t3_args.add_output(ext_out);
    submit_mix(t3_args, kBlockNum);
}

}  // extern "C"
