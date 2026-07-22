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
// case=4 dense-dependency degree: > PTO2_DEP_DEGREE_WARN_THRESHOLD (16) so the
// producer's fanout and the final consumer's fanin both trip the diagnostic.
static constexpr int32_t DENSE_DEP_COUNT = 18;

static void dummy_task_graph(const L2TaskArgs &args) {
    const Tensor &x = args.tensor(0).ref();
    const Tensor &y = args.tensor(1).ref();

    L0TaskArgs producer_args;
    producer_args.add_inout(x);
    rt_submit_aic_task(FUNC_WRITE_CONST, producer_args);

    L0TaskArgs barrier_args;
    barrier_args.add_inout(x);
    rt_submit_dummy_task(barrier_args);

    L0TaskArgs consumer_args;
    consumer_args.add_input(x);
    consumer_args.add_inout(y);
    rt_submit_aic_task(FUNC_COPY_FIRST, consumer_args);
}

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 4,  // 3 tensors + 1 case scalar
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    const Tensor &ext_X = orch_args.tensor(0).ref();
    const Tensor &ext_Y = orch_args.tensor(1).ref();
    const Tensor &ext_W = orch_args.tensor(2).ref();

    uint64_t case_id = orch_args.scalar(0);
    LOG_INFO_V0("[dummy_task_orch] case_id=%llu", static_cast<unsigned long long>(case_id));

    if (case_id == 1) {
        // producer writes X
        {
            L0TaskArgs args;
            args.add_inout(ext_X);
            rt_submit_aic_task(FUNC_WRITE_CONST, args);
        }
        // dummy_T INOUTs X (becomes new producer)
        {
            L0TaskArgs args;
            args.add_inout(ext_X);
            rt_submit_dummy_task(args);
        }
        // consumer reads X -> writes Y
        {
            L0TaskArgs args;
            args.add_input(ext_X);
            args.add_inout(ext_Y);
            rt_submit_aic_task(FUNC_COPY_FIRST, args);
        }
    } else if (case_id == 2) {
        // producer writes X
        {
            L0TaskArgs args;
            args.add_inout(ext_X);
            rt_submit_aic_task(FUNC_WRITE_CONST, args);
        }
        // long dummy chain
        for (int32_t i = 0; i < LONG_CHAIN_DUMMIES; i++) {
            L0TaskArgs args;
            args.add_inout(ext_X);
            rt_submit_dummy_task(args);
        }
        // consumer
        {
            L0TaskArgs args;
            args.add_input(ext_X);
            args.add_inout(ext_Y);
            rt_submit_aic_task(FUNC_COPY_FIRST, args);
        }
    } else if (case_id == 3) {
        // producer A writes X, producer B writes W
        PTO2TaskId a_id;
        PTO2TaskId b_id;
        {
            L0TaskArgs args;
            args.add_inout(ext_X);
            a_id = rt_submit_aic_task(FUNC_WRITE_CONST, args).task_id();
        }
        {
            L0TaskArgs args;
            args.add_inout(ext_W);
            b_id = rt_submit_aic_task(FUNC_WRITE_CONST, args).task_id();
        }
        // dummy barrier on A + B (no tensor args, only explicit deps)
        PTO2TaskId dummy_id;
        {
            L0TaskArgs args;
            PTO2TaskId barrier_deps[] = {a_id, b_id};
            args.set_dependencies(barrier_deps, 2);
            dummy_id = rt_submit_dummy_task(args).task_id();
        }
        // consumer: explicit dep on dummy, reads X
        {
            L0TaskArgs args;
            PTO2TaskId consumer_deps[] = {dummy_id};
            args.set_dependencies(consumer_deps, 1);
            args.add_input(ext_X);
            args.add_inout(ext_Y);
            rt_submit_aic_task(FUNC_COPY_FIRST, args);
        }
    } else if (case_id == 4) {
        // Dense fanout + fanin: one producer feeds DENSE_DEP_COUNT dummy
        // barriers (producer fanout = DENSE_DEP_COUNT), then one consumer
        // depends on all of them (consumer fanin = DENSE_DEP_COUNT). Both
        // degrees exceed PTO2_DEP_DEGREE_WARN_THRESHOLD, tripping the
        // orchestrator's dense-fanout and dense-fanin diagnostics.
        PTO2TaskId a_id;
        {
            L0TaskArgs args;
            args.add_inout(ext_X);
            a_id = rt_submit_aic_task(FUNC_WRITE_CONST, args).task_id();
        }
        PTO2TaskId dummies[DENSE_DEP_COUNT];
        for (int32_t i = 0; i < DENSE_DEP_COUNT; i++) {
            L0TaskArgs args;
            PTO2TaskId dep[] = {a_id};
            args.set_dependencies(dep, 1);
            dummies[i] = rt_submit_dummy_task(args).task_id();
        }
        {
            L0TaskArgs args;
            args.set_dependencies(dummies, DENSE_DEP_COUNT);
            args.add_input(ext_X);
            args.add_inout(ext_Y);
            rt_submit_aic_task(FUNC_COPY_FIRST, args);
        }
    } else if (case_id == 5) {
        L2TaskArgs args;
        args.add_inout(ext_X);
        args.add_inout(ext_Y);
        args.add_input(ext_W);
        args.add_scalar(case_id);
        rt_submit_graph(&dummy_task_graph, args);
    } else if (case_id == 6) {
        L0TaskArgs producer_args;
        producer_args.add_inout(ext_X);
        rt_submit_aic_task(FUNC_WRITE_CONST, producer_args);
        rt_graph_boundary();

        L0TaskArgs consumer_args;
        consumer_args.add_input(ext_X);
        consumer_args.add_inout(ext_Y);
        rt_submit_aic_task(FUNC_COPY_FIRST, consumer_args);
    } else {
        rt_report_fatal(PTO2_ERROR_INVALID_ARGS, "unsupported case_id=%llu", static_cast<unsigned long long>(case_id));
    }
}

}  // extern "C"
