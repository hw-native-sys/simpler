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
 * A serial chain of adds whose first task carries a bounded spin.
 *
 * Args layout: [a, b, out], then one scalar holding the first task's spin count.
 *
 * Task 0 writes `a + b` into `out`; every later task adds `b` to `out` again,
 * so `out == a + chain_length * b`. Every task names the same three caller
 * tensors and declares `out` INOUT, so the tensor map orders the whole chain
 * through `out` — no intermediate is created, which keeps the run off the ring
 * heap and away from the single-scope lifetime of a submit's output view.
 *
 * The spin belongs to the first task alone: one task sets the run's duration,
 * so the chain length only has to make the graph non-trivial.
 */

#include <cstdint>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {

constexpr int32_t kSpinAdd = 0;
constexpr int kChainLength = 64;

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return OrchestrationConfig{
        .expected_arg_count = 4,  // 3 tensors + the spin scalar
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const simpler::tmr::Tensor &ext_a = orch_args.tensor(0).ref();
    const simpler::tmr::Tensor &ext_b = orch_args.tensor(1).ref();
    const simpler::tmr::Tensor &ext_out = orch_args.tensor(2).ref();
    const uint64_t spin_iters = orch_args.scalar(0);

    for (int i = 0; i < kChainLength; ++i) {
        const bool first = (i == 0);
        CoreTaskArgs args;
        args.add_input(ext_a);
        args.add_input(ext_b);
        args.add_inout(ext_out);
        args.add_scalar(first ? spin_iters : static_cast<uint64_t>(0));
        args.add_scalar(first ? static_cast<uint64_t>(0) : static_cast<uint64_t>(1));
        (void)rt_submit_aiv_task(kSpinAdd, args);
    }
}

}  // extern "C"
