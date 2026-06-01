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
 * Many-to-one barrier via explicit set_dependencies — exercises the dep_gen
 * overflow chain wire format.
 *
 * Submits N producers each writing X[0] = 42.0, then a dummy_T whose only
 * dependency surface is set_dependencies({all N producer ids}, N), then a
 * consumer that explicit-depends on the barrier and copies X[0] -> Y[0].
 *
 * Picking N > DEP_GEN_MAX_EXPLICIT_DEPS (=64) forces the dep_gen capture to
 * spill into one or more DepGenOverflowRecord slots; picking N to span the
 * 64 + k*326 boundaries exercises both single- and multi-overflow chains.
 *
 * Args layout: [X, Y, scalar(N)]
 *   - X: every producer writes it (tensormap auto-deps the chain so the
 *        SENTINEL is preserved); consumer reads it.
 *   - Y: consumer writes it; host checks Y[0] == SENTINEL.
 *
 * Scalar: N (1 .. MAX_PRODUCERS).
 */

#include <cstdint>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_WRITE_CONST 0
#define FUNC_COPY_FIRST 1

// Stack room for producer_ids[]. 500 covers everything we expect to test;
// PTO2_DEP_LIST_POOL_SIZE (16384) is the real ceiling on a per-ring basis.
static constexpr int32_t MAX_PRODUCERS = 500;

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 3,  // X, Y, scalar(N)
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    Tensor ext_X = from_tensor_arg(orch_args.tensor(0));
    Tensor ext_Y = from_tensor_arg(orch_args.tensor(1));

    uint64_t n_raw = orch_args.scalar(0);
    int32_t n = static_cast<int32_t>(n_raw);
    if (n < 1 || n > MAX_PRODUCERS) {
        rt_report_fatal(PTO2_ERROR_INVALID_ARGS, "chain_barrier_orch: invalid n=%d", n);
        return;
    }

    PTO2TaskId producer_ids[MAX_PRODUCERS];

    // N producers each INOUT X. tensormap auto-deps them in a chain, so X[0]
    // stays at SENTINEL through all of them — the host only checks the final
    // value, which proves the barrier waited for every producer to finish.
    for (int32_t i = 0; i < n; i++) {
        Arg args;
        args.add_inout(ext_X);
        producer_ids[i] = rt_submit_aic_task(FUNC_WRITE_CONST, args).task_id();
    }

    // Dummy barrier with explicit deps on ALL N producers. dc=n > 64 forces
    // the dep_gen writer to emit base + overflow chain.
    PTO2TaskId barrier_id;
    {
        Arg args;
        args.set_dependencies(producer_ids, n);
        barrier_id = rt_submit_dummy_task(args).task_id();
    }

    // Consumer: explicit dep on barrier only, reads X, writes Y.
    {
        Arg args;
        PTO2TaskId consumer_deps[] = {barrier_id};
        args.set_dependencies(consumer_deps, 1);
        args.add_input(ext_X);
        args.add_inout(ext_Y);
        rt_submit_aic_task(FUNC_COPY_FIRST, args);
    }
}

}  // extern "C"
