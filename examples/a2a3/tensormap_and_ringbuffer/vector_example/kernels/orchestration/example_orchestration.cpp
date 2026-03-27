/**
 * Example: aicpu_orchestration_entry 设备端编排
 *
 * DAG structure for formula: (a + b + 1)(a + b + 2) + (a + b)
 *   t0: c = a + b     (func_id=0, kernel_add)       [outer scope]
 *   t1: d = c + 1     (func_id=1, kernel_add_scalar) [inner scope]
 *   t2: e = c + 2     (func_id=1, kernel_add_scalar) [inner scope]
 *   t3: g = d * e     (func_id=2, kernel_mul)        [inner scope]
 *   t4: f = g + c     (func_id=0, kernel_add)        [inner scope]
 *   Dependencies: t0->t1, t0->t2, t1->t3, t2->t3, t0->t4, t3->t4
 *
 * Nested scope demonstration:
 *   - Inner scope owns t1, t2, t3, t4; intermediates d, e, g release on inner scope end
 *   - Outer scope owns t0; c persists across inner scope for t1, t2, t4
 *   - c flows from outer to inner scope (outer-scope tensors are visible to inner scopes)
 *
 * This file compiles as a standalone .so with zero runtime link dependencies.
 * All runtime calls go through the PTO2RuntimeOps function-pointer table.
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"

// Helper to encode float as uint64_t for scalar params
static uint64_t float_to_u64(float f) {
    union {
        float f32;
        uint64_t u64;
    } conv;
    conv.u64 = 0;  // Clear upper bits
    conv.f32 = f;
    return conv.u64;
}

extern "C" {

/**
 * Orchestration config — the executor reads these values to set up
 * shared memory and runtime before calling aicpu_orchestration_entry.
 */
__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(TaskArg* orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 3,
    };
}

/**
 * Orchestration entry — runtime is bound implicitly by the framework.
 * The executor wraps this call in PTO2_SCOPE, so we are already inside
 * the outer scope on entry.
 */
__attribute__((visibility("default")))
void aicpu_orchestration_entry(TaskArg* orch_args, int orch_thread_num, int orch_thread_index) {
    (void)orch_thread_num;
    (void)orch_thread_index;

    // golden shape = kernel shape, use from_task_arg() directly
    Tensor ext_a = from_task_arg(orch_args[0]);
    Tensor ext_b = from_task_arg(orch_args[1]);
    Tensor ext_f = from_task_arg(orch_args[2]);

    uint32_t SIZE = orch_args[0].tensor.shapes[0];
    LOG_INFO("===============SIZE=%u", SIZE);

    uint32_t inter_shapes[1] = {SIZE};
    TensorCreateInfo inter_ci(inter_shapes, 1, DataType::FLOAT32);

    // t0: c = a + b (kernel_id=0, kernel_add) [outer scope]
    PTOParam params_t0;
    params_t0.add_input(ext_a);
    params_t0.add_input(ext_b);
    params_t0.add_output(inter_ci);
    TaskOutputTensors outs_t0 = pto2_rt_submit_aiv_task(0, params_t0); // kernel_add
    const Tensor& c = outs_t0.get_ref(0);

    // Inner scope: owns t1, t2, t3, t4; intermediates d, e, g release on scope end.
    // c flows in from outer scope (outer-scope tensors are visible to inner scopes).
    PTO2_SCOPE() {
        // t1: d = c + 1 (kernel_id=1, kernel_add_scalar)
        PTOParam params_t1;
        params_t1.add_input(c);
        params_t1.add_output(inter_ci);
        params_t1.add_scalar(float_to_u64(1.0f));
        params_t1.add_scalar((uint64_t)3);
        TaskOutputTensors outs_t1 = pto2_rt_submit_aiv_task(1, params_t1); // kernel_add_scalar
        const Tensor& d = outs_t1.get_ref(0);

        // t2: e = c + 2 (kernel_id=1, kernel_add_scalar)
        PTOParam params_t2;
        params_t2.add_input(c);
        params_t2.add_output(inter_ci);
        params_t2.add_scalar(float_to_u64(2.0f));
        params_t2.add_scalar((uint64_t)3);
        TaskOutputTensors outs_t2 = pto2_rt_submit_aiv_task(1, params_t2); // kernel_add_scalar
        const Tensor& e = outs_t2.get_ref(0);

        // t3: g = d * e (kernel_id=2, kernel_mul)
        PTOParam params_t3;
        params_t3.add_input(d);
        params_t3.add_input(e);
        params_t3.add_output(inter_ci);
        params_t3.add_scalar((uint64_t)3);
        TaskOutputTensors outs_t3 = pto2_rt_submit_aiv_task(2, params_t3); // kernel_mul
        const Tensor& g = outs_t3.get_ref(0);

        // t4: f = g + c (kernel_id=0, kernel_add)
        PTOParam params_t4;
        params_t4.add_input(g);
        params_t4.add_input(c);
        params_t4.add_inout(ext_f);
        pto2_rt_submit_aiv_task(0, params_t4); // kernel_add
    }  // inner scope ends: releases d, e, g
}

}  // extern "C"
