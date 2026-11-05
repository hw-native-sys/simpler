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
    return OrchestrationConfig{.expected_arg_count = 10};
}
__attribute__((visibility("default"))) void host_ddr_urma_mx(const ChipTaskArgs &args) {
    // One task posts at most 32 rows into the bounded deferred-completion slab.
    for (uint32_t begin = 0; begin < 128; begin += 32) {
        const uint32_t id_shape[]{32}, id_offset[]{begin};
        const uint32_t out_shape[]{32, 64}, out_offset[]{begin, 0};
        // Each scalar writer owns one complete 128-byte cache line.
        const uint32_t status_shape[]{1}, status_offset[]{begin};
        auto ids = args.tensor(1).ref().view(id_shape, id_offset);
        auto output = args.tensor(2).ref().view(out_shape, out_offset);
        auto status = args.tensor(3).ref().view(status_shape, status_offset);
        CoreTaskArgs gather;
        gather.add_input(args.tensor(0).ref());
        gather.add_input(ids);
        gather.add_output(output);
        gather.add_output(status);
        gather.add_scalar(uint64_t{32});
        gather.add_scalar(args.scalar<uint64_t>(0));
        gather.add_scalar(args.scalar<uint64_t>(1));
        rt_submit_aiv_task(0, gather);
    }
    CoreTaskArgs consume;
    consume.add_input(args.tensor(2).ref());
    consume.add_input(args.tensor(4).ref());
    consume.add_input(args.tensor(5).ref());
    consume.add_input(args.tensor(6).ref());
    consume.add_output(args.tensor(7).ref());
    consume.add_scalar(uint64_t{0});
    rt_submit_aic_task(1, consume);
}
}
