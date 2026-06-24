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

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 16,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    int64_t block_dim = static_cast<int64_t>(orch_args.scalar(0));

    LOG_INFO_V1("SPMD PA highperf: block_dim=%" PRId64, block_dim);

    const Tensor &query = orch_args.tensor(0).ref();
    const Tensor &key_cache = orch_args.tensor(1).ref();
    const Tensor &value_cache = orch_args.tensor(2).ref();
    const Tensor &block_table = orch_args.tensor(3).ref();
    const Tensor &out = orch_args.tensor(4).ref();
    const Tensor &s_gm = orch_args.tensor(5).ref();
    const Tensor &p_gm = orch_args.tensor(6).ref();
    const Tensor &o_tmp_gm = orch_args.tensor(7).ref();
    const Tensor &go_gm = orch_args.tensor(8).ref();
    const Tensor &o_core_tmp_gm = orch_args.tensor(9).ref();
    const Tensor &l_gm = orch_args.tensor(10).ref();
    const Tensor &gm_k16 = orch_args.tensor(11).ref();
    const Tensor &gm_v16 = orch_args.tensor(12).ref();
    const Tensor &tiling = orch_args.tensor(13).ref();
    const Tensor &null_tensor = orch_args.tensor(14).ref();

    L0TaskArgs args;
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
