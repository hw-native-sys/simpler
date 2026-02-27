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

// =============================================================================
// Args layout (from code_runner.py + runtime_maker.cpp extension):
// Base args from code_runner.py: [tensors..., sizes..., SIZE]
// Extended by runtime_maker.cpp: [..., gm_heap, heap_size] (always last 2)
//
// For this example (a+b+1)(a+b+2)+(a+b):
//   [a, b, f, size_a, size_b, size_f, SIZE]
//   + [gm_heap, heap_size] appended by runtime_maker.cpp
//
// Intermediate tensors (c, d, e, g) are allocated on-device by the runtime heap.
// Generic access: gm_heap = args[arg_count - 2], heap_size = args[arg_count - 1]
// =============================================================================

// Tensor device pointers (order from code_runner.py: inputs, outputs)
#define ARG_PTR_A 0
#define ARG_PTR_B 1
#define ARG_PTR_F 2  // output

// Tensor sizes (same order as pointers)
#define ARG_SIZE_A 3
#define ARG_SIZE_B 4
#define ARG_SIZE_F 5

// Element count (scalar)
#define ARG_SIZE 6

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
PTO2OrchestrationConfig aicpu_orchestration_config(uint64_t* args, int arg_count) {
    (void)args;
    (void)arg_count;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 7,
    };
}

/**
 * Orchestration entry — receives a PTO2Runtime* with ops table populated.
 * The executor wraps this call in PTO2_SCOPE, so we are already inside
 * the outer scope on entry.
 */
__attribute__((visibility("default")))
void aicpu_orchestration_entry(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    (void)arg_count;
    pto2_rt_init_tensor_pool(rt);

    void* arg_a_ptr = (void*)(uintptr_t)args[ARG_PTR_A];
    void* arg_b_ptr = (void*)(uintptr_t)args[ARG_PTR_B];
    void* arg_f_ptr = (void*)(uintptr_t)args[ARG_PTR_F];
    int SIZE = (int)(args[ARG_SIZE] & 0x7FFFFFFF);

    LOG_INFO(rt, "===============SIZE=%d", SIZE);

    uint64_t ext_shapes[1] = {(uint64_t)SIZE};
    Tensor ext_a = make_tensor_external(arg_a_ptr, ext_shapes, 1, DataType::FLOAT32);
    Tensor ext_b = make_tensor_external(arg_b_ptr, ext_shapes, 1, DataType::FLOAT32);
    Tensor ext_f = make_tensor_external(arg_f_ptr, ext_shapes, 1, DataType::FLOAT32);

    uint64_t inter_shapes[1] = {(uint64_t)SIZE};
    Tensor c = make_tensor(inter_shapes, 1, DataType::FLOAT32);  // c = a + b

    // t0: c = a + b (kernel_id=0, kernel_add) [outer scope]
    PTOParam params_t0[] = {
        make_input_param(ext_a),
        make_input_param(ext_b),
        make_output_param(c),
    };
    pto2_rt_submit_task(rt, 0, PTO2_WORKER_VECTOR, params_t0, 3); // kernel_add

    // Inner scope: owns t1, t2, t3, t4; intermediates d, e, g release on scope end.
    // c flows in from outer scope (outer-scope tensors are visible to inner scopes).
    PTO2_SCOPE(rt) {
        Tensor d = make_tensor(inter_shapes, 1, DataType::FLOAT32);  // d = c + 1
        Tensor e = make_tensor(inter_shapes, 1, DataType::FLOAT32);  // e = c + 2
        Tensor g = make_tensor(inter_shapes, 1, DataType::FLOAT32);  // g = d * e

        // t1: d = c + 1 (kernel_id=1, kernel_add_scalar)
        PTOParam params_t1[] = {
            make_input_param(c),
            make_scalar_param(float_to_u64(1.0f)),
            make_output_param(d),
            make_scalar_param((uint64_t)3),
        };
        pto2_rt_submit_task(rt, 1, PTO2_WORKER_VECTOR, params_t1, 3); // kernel_add_scalar

        // t2: e = c + 2 (kernel_id=1, kernel_add_scalar)
        PTOParam params_t2[] = {
            make_input_param(c),
            make_scalar_param(float_to_u64(2.0f)),
            make_output_param(e),
            make_scalar_param((uint64_t)3),
        };
        pto2_rt_submit_task(rt, 1, PTO2_WORKER_VECTOR, params_t2, 3); // kernel_add_scalar

        // t3: g = d * e (kernel_id=2, kernel_mul)
        PTOParam params_t3[] = {
            make_input_param(d),
            make_input_param(e),
            make_output_param(g),
            make_scalar_param((uint64_t)3),
        };
        pto2_rt_submit_task(rt, 2, PTO2_WORKER_VECTOR, params_t3, 3); // kernel_mul

        // t4: f = g + c (kernel_id=0, kernel_add)
        PTOParam params_t4[] = {
            make_input_param(g),
            make_input_param(c),
            make_output_param(ext_f),
        };
        pto2_rt_submit_task(rt, 0, PTO2_WORKER_VECTOR, params_t4, 3); // kernel_add
    }  // inner scope ends: releases d, e, g
}

}  // extern "C"
