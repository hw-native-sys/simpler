/**
 * AICPU orchestration for the vector example.
 *
 * DAG structure for formula: f = (a + b + 1) * (a + b + 2)
 *   t0: c = a + b     (func_id=0, kernel_add)
 *   t1: d = c + 1     (func_id=1, kernel_add_scalar)
 *   t2: e = c + 2     (func_id=1, kernel_add_scalar)
 *   t3: f = d * e     (func_id=2, kernel_mul)
 *   Dependencies: t0->t1, t0->t2, t1->t3, t2->t3
 *
 * Uses explicit add_dependency for all dependency edges (no TensorMap).
 * Tasks are batch-published at scope_end.
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"

// Args layout (from golden.py + runtime_maker.cpp extension):
//   [a, b, f, size_a, size_b, size_f, SIZE]
//   + [gm_heap, heap_size] appended by runtime_maker.cpp
#define ARG_PTR_A 0
#define ARG_PTR_B 1
#define ARG_PTR_F 2
#define ARG_SIZE  6

static uint64_t float_to_u64(float f) {
    union {
        float f32;
        uint64_t u64;
    } conv;
    conv.u64 = 0;
    conv.f32 = f;
    return conv.u64;
}

extern "C" {

__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(uint64_t* args, int arg_count) {
    (void)args;
    (void)arg_count;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 7,
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(PTO2Runtime* rt, uint64_t* args, int arg_count, int orch_thread_num, int orch_thread_index) {
    (void)arg_count;
    (void)orch_thread_num;
    (void)orch_thread_index;

    void* arg_a_ptr = (void*)(uintptr_t)args[ARG_PTR_A];
    void* arg_b_ptr = (void*)(uintptr_t)args[ARG_PTR_B];
    void* arg_f_ptr = (void*)(uintptr_t)args[ARG_PTR_F];
    int SIZE = (int)(args[ARG_SIZE] & 0x7FFFFFFF);

    uint32_t shapes[1] = {(uint32_t)SIZE};
    Tensor ext_a = make_tensor_external(arg_a_ptr, shapes, 1, DataType::FLOAT32);
    Tensor ext_b = make_tensor_external(arg_b_ptr, shapes, 1, DataType::FLOAT32);
    Tensor ext_f = make_tensor_external(arg_f_ptr, shapes, 1, DataType::FLOAT32);

    // Intermediate tensors — allocated from HeapRing by the runtime
    Tensor c = make_tensor(shapes, 1, DataType::FLOAT32);
    Tensor d = make_tensor(shapes, 1, DataType::FLOAT32);
    Tensor e = make_tensor(shapes, 1, DataType::FLOAT32);

    PTO2_SCOPE(rt) {
        // t0: c = a + b
        PTOParam p0;
        p0.add_input(ext_a);
        p0.add_input(ext_b);
        p0.add_output(c);
        PTO2TaskId t0 = pto2_rt_submit_aiv_task(rt, 0, p0);

        // t1: d = c + 1.0
        PTOParam p1;
        p1.add_input(c);
        p1.add_output(d);
        p1.add_scalar(float_to_u64(1.0f));
        PTO2TaskId t1 = pto2_rt_submit_aiv_task(rt, 1, p1);
        pto2_rt_add_dependency(rt, t0, t1);

        // t2: e = c + 2.0
        PTOParam p2;
        p2.add_input(c);
        p2.add_output(e);
        p2.add_scalar(float_to_u64(2.0f));
        PTO2TaskId t2 = pto2_rt_submit_aiv_task(rt, 1, p2);
        pto2_rt_add_dependency(rt, t0, t2);

        // t3: f = d * e
        PTOParam p3;
        p3.add_input(d);
        p3.add_input(e);
        p3.add_output(ext_f);
        PTO2TaskId t3 = pto2_rt_submit_aiv_task(rt, 2, p3);
        pto2_rt_add_dependency(rt, t1, t3);
        pto2_rt_add_dependency(rt, t2, t3);
    }  // scope_end: batch-publish all tasks
}

}  // extern "C"
