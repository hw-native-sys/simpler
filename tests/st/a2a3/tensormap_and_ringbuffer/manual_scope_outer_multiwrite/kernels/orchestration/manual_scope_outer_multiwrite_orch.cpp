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

#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_ADD 0

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(
    const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 5,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(
    const ChipStorageTaskArgs &orch_args, int orch_thread_num, int orch_thread_index) {
    (void)orch_thread_num;    // NOLINT(readability/casting)
    (void)orch_thread_index;  // NOLINT(readability/casting)

    Tensor ext_a = from_tensor_arg(orch_args.tensor(0));
    Tensor ext_b = from_tensor_arg(orch_args.tensor(1));
    Tensor ext_out = from_tensor_arg(orch_args.tensor(2));
    Tensor ext_result = from_tensor_arg(orch_args.tensor(3));
    Tensor ext_check = from_tensor_arg(orch_args.tensor(4));

    uint32_t size = orch_args.tensor(0).shapes[0];
    uint32_t inter_shapes[1] = {size};
    TensorCreateInfo inter_ci(inter_shapes, 1, DataType::FLOAT32);

    PTO2_SCOPE() {
        PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
            Arg tmp0_args;
            tmp0_args.add_input(ext_a);
            tmp0_args.add_input(ext_a);
            tmp0_args.add_output(inter_ci);
            PTO2ManualSubmitResult tmp0 = pto2_rt_submit_aiv_task_manual(FUNC_ADD, tmp0_args);

            Arg write0_args;
            write0_args.add_input(tmp0.outputs.get_ref(0));
            write0_args.add_input(ext_a);
            write0_args.add_output(ext_out);
            PTO2ManualSubmitResult write0 = pto2_rt_submit_aiv_task_manual(FUNC_ADD, write0_args);
            pto2_rt_add_dependency(tmp0.task_id, write0.task_id);

            Arg tmp1_args;
            tmp1_args.add_input(ext_b);
            tmp1_args.add_input(ext_b);
            tmp1_args.add_output(inter_ci);
            PTO2ManualSubmitResult tmp1 = pto2_rt_submit_aiv_task_manual(FUNC_ADD, tmp1_args);

            Arg write1_args;
            write1_args.add_input(tmp1.outputs.get_ref(0));
            write1_args.add_input(ext_a);
            write1_args.add_output(ext_out);
            PTO2ManualSubmitResult write1 = pto2_rt_submit_aiv_task_manual(FUNC_ADD, write1_args);
            pto2_rt_add_dependency(tmp1.task_id, write1.task_id);
            pto2_rt_add_dependency(write0.task_id, write1.task_id);
        }

        Arg consumer_args;
        consumer_args.add_input(ext_out);
        consumer_args.add_input(ext_b);
        consumer_args.add_output(ext_result);
        pto2_rt_submit_aiv_task(FUNC_ADD, consumer_args);

        uint32_t idx0[1] = {0};
        uint32_t idx100[1] = {100};

        float out0 = get_tensor_data<float>(ext_out, 1, idx0);
        float result0 = get_tensor_data<float>(ext_result, 1, idx0);
        float out100 = get_tensor_data<float>(ext_out, 1, idx100);

        idx0[0] = 0;
        set_tensor_data(ext_check, 1, idx0, out0);
        idx0[0] = 1;
        set_tensor_data(ext_check, 1, idx0, result0);
        idx0[0] = 2;
        set_tensor_data(ext_check, 1, idx0, out100);
    }
}
}
