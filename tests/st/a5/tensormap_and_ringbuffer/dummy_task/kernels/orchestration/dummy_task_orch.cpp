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
 * dummy_task orchestration scenes.
 *
 * Each case is selected via params["case"] in the orchestration scalar slot.
 *
 *   case=1: Single dummy via auto tensormap dep.
 *     producer (kernel_write_const) writes X[0] = 42.0
 *     dummy_T INOUTs X (no kernel)        // becomes new producer in tensormap
 *     consumer (kernel_copy_first) X -> Y
 *     expect Y[0] = 42.0
 *
 *   case=2: Long dummy chain (N dummies between producer and consumer).
 *     producer writes X[0] = 42.0
 *     dummy_T1 .. dummy_TN each INOUT X    // chained through tensormap
 *     consumer copies X -> Y
 *     expect Y[0] = 42.0 (no dummy runs a kernel; X must be undisturbed)
 *
 *   case=3: Dummy as many-to-one barrier via explicit set_dependencies.
 *     producer_A writes X[0] = 42.0
 *     producer_B writes W[0] = 7.0
 *     dummy_T explicit set_dependencies({A.id, B.id}, 2)  // pure barrier
 *     consumer explicit set_dependencies({dummy.id}, 1), copies X -> Y
 *     expect Y[0] = 42.0 (consumer waits on dummy which waits on A+B)
 *
 * Args layout: [X, Y, W]
 *   - X: producer A writes; consumer reads
 *   - Y: consumer writes; host checks
 *   - W: producer B writes (case 3 only); ignored by consumer
 *
 * Scalar:  case selector
 */

#include <cstdint>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_WRITE_CONST 0
#define FUNC_COPY_FIRST 1

static constexpr int32_t LONG_CHAIN_DUMMIES = 4;

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 4,  // 3 tensors + 1 case scalar
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    Tensor ext_X = from_tensor_arg(orch_args.tensor(0));
    Tensor ext_Y = from_tensor_arg(orch_args.tensor(1));
    Tensor ext_W = from_tensor_arg(orch_args.tensor(2));

    uint64_t case_id = orch_args.scalar(0);
    LOG_INFO_V0("[dummy_task_orch] case_id=%llu", static_cast<unsigned long long>(case_id));

    if (case_id == 1) {
        // producer writes X
        {
            Arg args;
            args.add_inout(ext_X);
            rt_submit_aic_task(FUNC_WRITE_CONST, args);
        }
        // dummy_T INOUTs X (becomes new producer)
        {
            Arg args;
            args.add_inout(ext_X);
            rt_submit_dummy_task(args);
        }
        // consumer reads X -> writes Y
        {
            Arg args;
            args.add_input(ext_X);
            args.add_inout(ext_Y);
            rt_submit_aic_task(FUNC_COPY_FIRST, args);
        }
    } else if (case_id == 2) {
        // producer writes X
        {
            Arg args;
            args.add_inout(ext_X);
            rt_submit_aic_task(FUNC_WRITE_CONST, args);
        }
        // long dummy chain
        for (int32_t i = 0; i < LONG_CHAIN_DUMMIES; i++) {
            Arg args;
            args.add_inout(ext_X);
            rt_submit_dummy_task(args);
        }
        // consumer
        {
            Arg args;
            args.add_input(ext_X);
            args.add_inout(ext_Y);
            rt_submit_aic_task(FUNC_COPY_FIRST, args);
        }
    } else if (case_id == 3) {
        // producer A writes X, producer B writes W
        PTO2TaskId a_id;
        PTO2TaskId b_id;
        {
            Arg args;
            args.add_inout(ext_X);
            a_id = rt_submit_aic_task(FUNC_WRITE_CONST, args).task_id();
        }
        {
            Arg args;
            args.add_inout(ext_W);
            b_id = rt_submit_aic_task(FUNC_WRITE_CONST, args).task_id();
        }
        // dummy barrier on A + B (no tensor args, only explicit deps)
        PTO2TaskId dummy_id;
        {
            Arg args;
            PTO2TaskId barrier_deps[] = {a_id, b_id};
            args.set_dependencies(barrier_deps, 2);
            dummy_id = rt_submit_dummy_task(args).task_id();
        }
        // consumer: explicit dep on dummy, reads X
        {
            Arg args;
            PTO2TaskId consumer_deps[] = {dummy_id};
            args.set_dependencies(consumer_deps, 1);
            args.add_input(ext_X);
            args.add_inout(ext_Y);
            rt_submit_aic_task(FUNC_COPY_FIRST, args);
        }
    } else {
        rt_report_fatal(PTO2_ERROR_INVALID_ARGS, "unsupported case_id=%llu", static_cast<unsigned long long>(case_id));
    }
}

}  // extern "C"
