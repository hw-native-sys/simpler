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

/** A serial add chain used by the externally Notify-gated diagnostic scene. */

#include <cstdint>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {

constexpr int32_t kAdd = 0;
constexpr int kChainLength = 64;

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return OrchestrationConfig{
        .expected_arg_count = 3,  // 3 tensors
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const simpler::tmr::Tensor &ext_a = orch_args.tensor(0).ref();
    const simpler::tmr::Tensor &ext_b = orch_args.tensor(1).ref();
    const simpler::tmr::Tensor &ext_out = orch_args.tensor(2).ref();

    for (int i = 0; i < kChainLength; ++i) {
        const bool first = (i == 0);
        CoreTaskArgs args;
        args.add_input(ext_a);
        args.add_input(ext_b);
        args.add_inout(ext_out);
        args.add_scalar(first ? static_cast<uint64_t>(0) : static_cast<uint64_t>(1));
        // The first two tasks mark their arguments for capture, and only those
        // two. Selective capture keeps a 64-task chain's dump to a few
        // payloads, and this pair is the one worth reading: every task writes
        // the same `ext_out`, so the first task's output and the second task's
        // input are the same buffer at two different instants — which is what
        // lets a reader check an argument against its own task rather than
        // against whatever that buffer held when the run ended.
        if (i < 2) args.dump();
        (void)rt_submit_aiv_task(kAdd, args);
    }
}

}  // extern "C"
