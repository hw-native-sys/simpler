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
 * Compiled with PTO2 runtime sources for device execution.
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_runtime2.h"
#include "pto_shared_memory.h"

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

// gm_heap and heap_size are ALWAYS the last 2 args (generic, not hardcoded index)

#ifndef PTO2_TASK_WINDOW_SIZE
#define PTO2_TASK_WINDOW_SIZE 16384
#endif
#ifndef PTO2_DEP_LIST_POOL_SIZE
#define PTO2_DEP_LIST_POOL_SIZE 65536
#endif
#ifndef PTO2_HEAP_SIZE
#define PTO2_HEAP_SIZE (256 * 1024)
#endif

// Static buffer only for simulation; real device uses host-allocated gm_heap
static char s_gm_heap_stub[PTO2_HEAP_SIZE];

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

__attribute__((visibility("default")))
void aicpu_orchestration_entry(void* sm_ptr, uint64_t* args, int arg_count) {
    PTO2OrchestrationBeginInfo begin_info{
        .sm_ptr             = sm_ptr,
        .args               = args,
        .arg_count          = arg_count,
        .expected_arg_count = 7,
        .task_window_size   = PTO2_TASK_WINDOW_SIZE,
        .dep_list_pool_size = PTO2_DEP_LIST_POOL_SIZE,
        .heap_size          = PTO2_HEAP_SIZE,
        .gm_heap_ptr        = s_gm_heap_stub,
    };

    PTO2_ORCHESTRATION(rt, begin_info) {
        // Outer scope: implicitly opened by PTO2_ORCHESTRATION macro; owns t0, t4

        void* arg_a_ptr = (void*)(uintptr_t)args[ARG_PTR_A];
        void* arg_b_ptr = (void*)(uintptr_t)args[ARG_PTR_B];
        void* arg_f_ptr = (void*)(uintptr_t)args[ARG_PTR_F];
        size_t size_a = (size_t)args[ARG_SIZE_A];
        size_t size_b = (size_t)args[ARG_SIZE_B];
        size_t size_f = (size_t)args[ARG_SIZE_F];
        int SIZE = (int)(args[ARG_SIZE] & 0x7FFFFFFF);

        printf("===============SIZE=%d\n", SIZE);

        size_t BYTES = (size_t)SIZE * sizeof(float);

        Tensor ext_a = make_tensor_external(arg_a_ptr, size_a);
        Tensor ext_b = make_tensor_external(arg_b_ptr, size_b);
        Tensor ext_f = make_tensor_external(arg_f_ptr, size_f);

        Tensor c = make_tensor(BYTES);  // c = a + b

        // t0: c = a + b (kernel_id=0, kernel_add) [outer scope]
        PTOParam params_t0[] = {
            make_input_param(ext_a),
            make_input_param(ext_b),
            make_output_param(c),
        };
        pto2_rt_submit_task(rt, 0, PTO2_WORKER_VECTOR, "kernel_add", params_t0, 3);

        // Inner scope: owns t1, t2, t3, t4; intermediates d, e, g release on scope end.
        // c flows in from outer scope (outer-scope tensors are visible to inner scopes).
        PTO2_SCOPE(rt) {
            Tensor d = make_tensor(BYTES);  // d = c + 1
            Tensor e = make_tensor(BYTES);  // e = c + 2
            Tensor g = make_tensor(BYTES);  // g = d * e

            // t1: d = c + 1 (kernel_id=1, kernel_add_scalar)
            PTOParam params_t1[] = {
                make_input_param(c),
                make_scalar_param(float_to_u64(1.0f)),
                make_output_param(d),
                make_scalar_param((uint64_t)3),
            };
            pto2_rt_submit_task(rt, 1, PTO2_WORKER_VECTOR, "kernel_add_scalar", params_t1, 3);

            // t2: e = c + 2 (kernel_id=1, kernel_add_scalar)
            PTOParam params_t2[] = {
                make_input_param(c),
                make_scalar_param(float_to_u64(2.0f)),
                make_output_param(e),
                make_scalar_param((uint64_t)3),
            };
            pto2_rt_submit_task(rt, 1, PTO2_WORKER_VECTOR, "kernel_add_scalar", params_t2, 3);

            // t3: g = d * e (kernel_id=2, kernel_mul)
            PTOParam params_t3[] = {
                make_input_param(d),
                make_input_param(e),
                make_output_param(g),
                make_scalar_param((uint64_t)3),
            };
            pto2_rt_submit_task(rt, 2, PTO2_WORKER_VECTOR, "kernel_mul", params_t3, 3);

            // t4: f = g + c (kernel_id=0, kernel_add)
            PTOParam params_t4[] = {
                make_input_param(g),
                make_input_param(c),
                make_output_param(ext_f),
            };
            pto2_rt_submit_task(rt, 0, PTO2_WORKER_VECTOR, "kernel_add", params_t4, 3);
        }  // inner scope ends: releases d, e, g
    }
}

}  // extern "C"
