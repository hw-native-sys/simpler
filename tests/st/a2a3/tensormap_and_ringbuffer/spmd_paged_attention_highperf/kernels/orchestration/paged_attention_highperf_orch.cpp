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
#include <cinttypes>
#include <cstdint>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_PA_AIC 0
#define FUNC_PA_AIV 1

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 16,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    int64_t block_dim = static_cast<int64_t>(orch_args.scalar(0));

    LOG_INFO_V1("SPMD PA highperf: block_dim=%" PRId64, block_dim);

    Tensor query = from_tensor_arg(orch_args.tensor(0));
    Tensor key_cache = from_tensor_arg(orch_args.tensor(1));
    Tensor value_cache = from_tensor_arg(orch_args.tensor(2));
    Tensor block_table = from_tensor_arg(orch_args.tensor(3));
    Tensor out = from_tensor_arg(orch_args.tensor(4));
    Tensor s_gm = from_tensor_arg(orch_args.tensor(5));
    Tensor p_gm = from_tensor_arg(orch_args.tensor(6));
    Tensor o_tmp_gm = from_tensor_arg(orch_args.tensor(7));
    Tensor go_gm = from_tensor_arg(orch_args.tensor(8));
    Tensor o_core_tmp_gm = from_tensor_arg(orch_args.tensor(9));
    Tensor l_gm = from_tensor_arg(orch_args.tensor(10));
    Tensor gm_k16 = from_tensor_arg(orch_args.tensor(11));
    Tensor gm_v16 = from_tensor_arg(orch_args.tensor(12));
    Tensor tiling = from_tensor_arg(orch_args.tensor(13));
    Tensor null_tensor = from_tensor_arg(orch_args.tensor(14));

    Arg args;
    args.add_input(query);
    args.add_input(key_cache);
    args.add_input(value_cache);
    args.add_input(block_table);
    args.add_inout(out);
    args.add_inout(s_gm);
    args.add_inout(p_gm);
    args.add_inout(o_tmp_gm);
    args.add_inout(go_gm);
    args.add_inout(o_core_tmp_gm);
    args.add_inout(l_gm);
    args.add_inout(gm_k16);
    args.add_inout(gm_v16);
    args.add_input(tiling);
    args.add_input(null_tensor);
    args.launch_spec.set_block_num(static_cast<int16_t>(block_dim));

    MixedKernels mk;
    mk.aic_kernel_id = FUNC_PA_AIC;
    mk.aiv0_kernel_id = FUNC_PA_AIV;
    mk.aiv1_kernel_id = FUNC_PA_AIV;
    rt_submit_task(mk, args);
}

}  // extern "C"
