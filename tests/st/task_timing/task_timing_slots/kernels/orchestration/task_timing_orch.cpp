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
 * Selective task-timing-slot demo orchestration (issue #1325).
 *
 *   out = (a + b) + b
 *
 * A two-task chain that exercises the timing slots end to end:
 *   t0: c   = a + b   (func_id=0)  tagged timing slot 0  [first dispatch]
 *   t1: out = c + b   (func_id=0)  tagged timing slot 1  [last finish]
 *
 * t1 consumes t0's output, so t1's Scheduler-observed FIN necessarily follows
 * t0's dispatch. The host reads back both slots and emits
 * `simpler_run.runner_run.device_wall.task_slot_0` / `_1` on the [STRACE]
 * timeline; tooling recovers the whole-chain interval as
 * finish(slot_1) - dispatch(slot_0).
 *
 * Reuses the vector_add AIV kernel (out = src0 + src1), so func_id 0 is that
 * kernel with the (IN, IN, OUT) signature.
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

// SPMD parallelism count is spelled set_block_num on a2a3 and set_core_num on
// a5; pick whichever the launch spec exposes so this example builds on both.
template <typename Spec>
static inline auto set_spmd_count(Spec &s, int16_t n) -> decltype(s.set_block_num(n), void()) {
    s.set_block_num(n);
}
template <typename Spec>
static inline auto set_spmd_count(Spec &s, int16_t n) -> decltype(s.set_core_num(n), void()) {
    s.set_core_num(n);
}

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
task_timing_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 3,  // a, b, out
    };
}

__attribute__((visibility("default"))) void task_timing_orchestration(const L2TaskArgs &orch_args) {
    const Tensor &a = orch_args.tensor(0).ref();
    const Tensor &b = orch_args.tensor(1).ref();
    const Tensor &out = orch_args.tensor(2).ref();

    uint32_t shapes[2] = {a.shapes[0], a.shapes[1]};
    TensorCreateInfo inter_ci(shapes, 2, DataType::FLOAT32);

    // t0: c = a + b, tagged slot 0 (interval start).
    L0TaskArgs params_t0;
    params_t0.add_input(a);
    params_t0.add_input(b);
    params_t0.add_output(inter_ci);
    params_t0.set_task_timing_slot(0);
    TaskOutputTensors outs_t0 = rt_submit_aiv_task(0, params_t0);
    const Tensor &c = outs_t0.get_ref(0);

    // t1: out = c + b, tagged slot 1 (interval end). Depends on c -> runs after t0.
    L0TaskArgs params_t1;
    params_t1.add_input(c);
    params_t1.add_input(b);
    params_t1.add_output(out);
    params_t1.set_task_timing_slot(1);
    rt_submit_aiv_task(0, params_t1);
}

// Duplicate-slot variant: a three-task chain t0->t1->t2, ALL tagged slot 0.
// The scheduler folds min(dispatch)/max(finish) across the three tagged tasks,
// so the single emitted task_slot_0 span must cover the whole chain
// (dispatch of t0 .. finish of t2), not any single task's window.
__attribute__((visibility("default"))) void task_timing_dup_orchestration(const L2TaskArgs &orch_args) {
    const Tensor &a = orch_args.tensor(0).ref();
    const Tensor &b = orch_args.tensor(1).ref();
    const Tensor &out = orch_args.tensor(2).ref();

    uint32_t shapes[2] = {a.shapes[0], a.shapes[1]};
    TensorCreateInfo inter_ci(shapes, 2, DataType::FLOAT32);

    L0TaskArgs p0;
    p0.add_input(a);
    p0.add_input(b);
    p0.add_output(inter_ci);
    p0.set_task_timing_slot(0);
    TaskOutputTensors o0 = rt_submit_aiv_task(0, p0);
    const Tensor &c = o0.get_ref(0);

    L0TaskArgs p1;
    p1.add_input(c);
    p1.add_input(b);
    p1.add_output(inter_ci);
    p1.set_task_timing_slot(0);
    TaskOutputTensors o1 = rt_submit_aiv_task(0, p1);
    const Tensor &d = o1.get_ref(0);

    L0TaskArgs p2;
    p2.add_input(d);
    p2.add_input(b);
    p2.add_output(out);
    p2.set_task_timing_slot(0);
    rt_submit_aiv_task(0, p2);
}

// SPMD variant: a single task launched across 8 blocks, tagged slot 0. Blocks
// are dispatched by multiple Scheduler threads, so slot 0's window is the
// cross-thread min(dispatch)/max(finish) reduction of every participating
// block. (The vector_add kernel is not block-partitioned, so each block
// recomputes out = a + b over the whole tile — redundant but golden-correct.)
__attribute__((visibility("default"))) void task_timing_spmd_orchestration(const L2TaskArgs &orch_args) {
    const Tensor &a = orch_args.tensor(0).ref();
    const Tensor &b = orch_args.tensor(1).ref();
    const Tensor &out = orch_args.tensor(2).ref();

    L0TaskArgs params;
    params.add_input(a);
    params.add_input(b);
    params.add_output(out);
    set_spmd_count(params.launch_spec, 8);
    params.set_task_timing_slot(0);
    rt_submit_aiv_task(0, params);
}

// MIX variant: a single mixed task (AIC matmul + AIV0 add + AIV1 mul) tagged
// slot 0. The three subtasks each publish a DATA_MAIN_BASE, so slot 0's window
// is the min-dispatch/max-finish fold across all three subtask handles of the
// one task. Args: A,B,C (matmul) | D,E,F (add) | G,H,I (mul), 9 tensors.
// Reuses the mixed_example kernels: func 0 = matmul (AIC), 1 = add, 2 = mul.
__attribute__((visibility("default"))) void task_timing_mix_orchestration(const L2TaskArgs &orch_args) {
    MixedKernels mk;
    mk.aic_kernel_id = 0;   // matmul
    mk.aiv0_kernel_id = 1;  // add
    mk.aiv1_kernel_id = 2;  // mul

    L0TaskArgs args;
    args.add_input(orch_args.tensor(0).ref());   // A
    args.add_input(orch_args.tensor(1).ref());   // B
    args.add_output(orch_args.tensor(2).ref());  // C = A @ B
    args.add_input(orch_args.tensor(3).ref());   // D
    args.add_input(orch_args.tensor(4).ref());   // E
    args.add_output(orch_args.tensor(5).ref());  // F = D + E
    args.add_input(orch_args.tensor(6).ref());   // G
    args.add_input(orch_args.tensor(7).ref());   // H
    args.add_output(orch_args.tensor(8).ref());  // I = G * H
    args.set_task_timing_slot(0);
    rt_submit_task(mk, args);
}

__attribute__((visibility("default"))) PTO2OrchestrationConfig
task_timing_mix_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 9,
    };
}

}  // extern "C"
