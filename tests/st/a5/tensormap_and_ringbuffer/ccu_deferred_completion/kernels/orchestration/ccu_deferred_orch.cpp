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

extern "C" __attribute__((visibility("default"))) OrchestrationConfig
ccu_deferred_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 6};
}

extern "C" __attribute__((visibility("default"))) void ccu_deferred_orchestration(const ChipTaskArgs &args) {
    const auto &source = args.tensor(0).ref();
    const auto &destination = args.tensor(1).ref();
    const auto &scratch = args.tensor(2).ref();
    const auto &results = args.tensor(3).ref();
    const uint64_t domain = args.scalar<uint64_t>(0);
    const uint64_t rounds = args.scalar<uint64_t>(1);
    for (uint64_t round = 0; round < rounds; ++round) {
        SIMPLER_SCOPE() {
            CoreTaskArgs initialize;
            initialize.add_inout(source);
            initialize.add_inout(destination);
            initialize.add_inout(scratch);
            initialize.add_inout(results);
            initialize.add_scalar(domain);
            initialize.add_scalar(round);
            rt_submit_aiv_task(0, initialize);

            CoreTaskArgs send;
            send.add_input(source);
            send.add_inout(destination);
            send.add_input(scratch);
            send.add_inout(results);
            send.add_scalar(domain);
            send.add_scalar(round);
            rt_submit_aiv_task(1, send);

            // The consumer's inputs depend on the task with the deferred CCU completion.
            CoreTaskArgs verify;
            verify.add_input(source);
            verify.add_input(destination);
            verify.add_input(scratch);
            verify.add_inout(results);
            verify.add_scalar(domain);
            verify.add_scalar(round);
            rt_submit_aiv_task(2, verify);
        }
    }
}
