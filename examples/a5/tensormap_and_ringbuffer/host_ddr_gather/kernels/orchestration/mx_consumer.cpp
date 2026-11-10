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

#include "orchestration_api.h"

extern "C" {
__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 9};
}
__attribute__((visibility("default"))) void host_ddr_mx(const ChipTaskArgs &args) {
    CoreTaskArgs gather;
    gather.add_input(args.tensor(0).ref());
    gather.add_input(args.tensor(1).ref());
    gather.add_output(args.tensor(2).ref());
    gather.add_output(args.tensor(3).ref());
    gather.add_scalar(args.scalar<uint64_t>(0));
    rt_submit_aiv_task(0, gather);
    CoreTaskArgs matmul;
    matmul.add_input(args.tensor(2).ref());
    matmul.add_input(args.tensor(4).ref());
    matmul.add_input(args.tensor(5).ref());
    matmul.add_input(args.tensor(6).ref());
    matmul.add_output(args.tensor(7).ref());
    matmul.add_scalar(uint64_t{0});
    rt_submit_aic_task(1, matmul);
}
}
