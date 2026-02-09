/**
 * Example: aicpu_orchestration_entry 设备端编排
 *
 * DAG structure for formula: (a + b + 1)(a + b + 2)
 *   t0: c = a + b     (func_id=0, kernel_add)
 *   t1: d = c + 1     (func_id=1, kernel_add_scalar)
 *   t2: e = c + 2     (func_id=1, kernel_add_scalar)
 *   t3: f = d * e     (func_id=2, kernel_mul)
 *   Dependencies: t0->t1, t0->t2, t1->t3, t2->t3
 *
 * Compiled with PTO2 runtime sources for device execution.
 */

#include <stdint.h>
#include <stddef.h>

#include "pto_runtime2.h"
#include "pto_shared_memory.h"


// =============================================================================
// Args layout (from code_runner.py + runtime_maker.cpp extension):
// Base args from code_runner.py: [tensors..., sizes..., SIZE]
// Extended by runtime_maker.cpp: [..., gm_heap, heap_size] (always last 2)
//
// For this example (a+b+1)(a+b+2):
//   [a, b, f, size_a, size_b, size_f, SIZE]
//   + [gm_heap, heap_size] appended by runtime_maker.cpp
//
// Intermediate tensors (c, d, e) are allocated on-device by the runtime heap.
// Generic access: gm_heap = args[arg_count - 2], heap_size = args[arg_count - 1]
// =============================================================================

// Tensor device pointers (order from code_runner.py: inputs, outputs)
#define ARG_PTR_A      0
#define ARG_PTR_B      1
#define ARG_PTR_F      2   // output

// Tensor sizes (same order as pointers)
#define ARG_SIZE_A     3
#define ARG_SIZE_B     4
#define ARG_SIZE_F     5

// Element count (scalar)
#define ARG_SIZE       6

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
    // Get shared memory header for proper access
    PTO2SharedMemoryHeader* header = (PTO2SharedMemoryHeader*)sm_ptr;

    // Validate inputs
    if (!sm_ptr || !args || arg_count < 7) {
        if (header) {
            header->orchestrator_done = 1;
        }
        return;
    }

    // Extract device pointers
    void* arg_a_ptr = (void*)(uintptr_t)args[ARG_PTR_A];
    void* arg_b_ptr = (void*)(uintptr_t)args[ARG_PTR_B];
    void* arg_f_ptr = (void*)(uintptr_t)args[ARG_PTR_F];
    size_t size_a = (size_t)args[ARG_SIZE_A];
    size_t size_b = (size_t)args[ARG_SIZE_B];
    size_t size_f = (size_t)args[ARG_SIZE_F];
    int SIZE = (int)(args[ARG_SIZE] & 0x7FFFFFFF);

    printf("===============SIZE=%d\n", SIZE);

    size_t BYTES = (size_t)SIZE * sizeof(float);

    // Create shared memory handle
    int32_t sm_size = pto2_sm_calculate_size(PTO2_TASK_WINDOW_SIZE, PTO2_DEP_LIST_POOL_SIZE);

    PTO2SharedMemoryHandle* sm_handle = pto2_sm_create_from_buffer(
        sm_ptr,
        sm_size,
        PTO2_TASK_WINDOW_SIZE,
        PTO2_HEAP_SIZE,
        PTO2_DEP_LIST_POOL_SIZE
    );
    if (!sm_handle) {
        header->orchestrator_done = 1;
        return;
    }

    // Get GM heap: runtime_maker.cpp appends [gm_heap, heap_size] as last 2 args
    // Use generic access: args[arg_count - 2] and args[arg_count - 1]
    // Fall back to static buffer only for simulation (when not provided)
    void* gm_heap = s_gm_heap_stub;
    int32_t heap_size = (int32_t)sizeof(s_gm_heap_stub);
    if (arg_count >= 2) {
        uint64_t gm_heap_arg = args[arg_count - 2];
        uint64_t heap_size_arg = args[arg_count - 1];
        if (gm_heap_arg != 0 && heap_size_arg != 0) {
            gm_heap = (void*)(uintptr_t)gm_heap_arg;
            heap_size = (int32_t)(heap_size_arg & 0x7FFFFFFF);
        }
    }

    // Create runtime
    PTO2Runtime* rt = pto2_runtime_create_from_sm(
        PTO2_MODE_EXECUTE,
        sm_handle,
        gm_heap,
        heap_size
    );
    if (!rt) {
        pto2_sm_destroy(sm_handle);
        header->orchestrator_done = 1;
        return;
    }

    int32_t tile = 0;
    int32_t sz = (int32_t)BYTES;
    if (sz <= 0) sz = (int32_t)size_a;

    TensorDescriptor ext_td_a = make_tensor_external(arg_a_ptr, size_a);
    TensorDescriptor ext_td_b = make_tensor_external(arg_b_ptr, size_b);
    TensorDescriptor ext_td_f = make_tensor_external(arg_f_ptr, size_f);
    TensorDescriptor td_c = make_tensor(BYTES);  // c = a + b
    TensorDescriptor td_d = make_tensor(BYTES);  // d = c + 1
    TensorDescriptor td_e = make_tensor(BYTES);  // e = c + 2

    // Use RAII scope guard for automatic scope management.
    // PTO2_SCOPE creates a scoped block where pto2_rt_scope_begin() is called
    // at the start and pto2_rt_scope_end() is called automatically at the end
    // (even in error paths). This eliminates manual cleanup and prevents bugs.
    // See src/runtime/rt2/runtime/pto_runtime2.h for alternative usage patterns.
    PTO2_SCOPE(rt) {
        // t0: c = a + b (kernel_id=0, kernel_add)
        PTOParam params_t0[] = {
            make_input_param(ext_td_a),
            make_input_param(ext_td_b),
            make_output_param(td_c),
        };
        pto2_rt_submit_task(rt, 0, PTO2_WORKER_VECTOR, "kernel_add", params_t0, 3);

        // t1: d = c + 1 (kernel_id=1, kernel_add_scalar)
        PTOParam params_t1[] = {
            make_input_param(td_c),
            make_scalar_param(float_to_u64(1.0f)),
            make_output_param(td_d),
            make_scalar_param((uint64_t)3),
        };
        pto2_rt_submit_task(rt, 1, PTO2_WORKER_VECTOR, "kernel_add_scalar", params_t1, 3);

        // t2: e = c + 2 (kernel_id=1, kernel_add_scalar)
        PTOParam params_t2[] = {
            make_input_param(td_c),
            make_scalar_param(float_to_u64(2.0f)),
            make_output_param(td_e),
            make_scalar_param((uint64_t)3),
        };
        pto2_rt_submit_task(rt, 1, PTO2_WORKER_VECTOR, "kernel_add_scalar", params_t2, 3);

        // t3: f = d * e (kernel_id=2, kernel_mul)
        PTOParam params_t3[] = {
            make_input_param(td_d),
            make_input_param(td_e),
            make_output_param(ext_td_f),
            make_scalar_param((uint64_t)3),
        };
        pto2_rt_submit_task(rt, 2, PTO2_WORKER_VECTOR, "kernel_mul", params_t3, 3);
    } // PTO2_SCOPE ends here - automatic pto2_rt_scope_end() called

    pto2_rt_orchestration_done(rt);
    pto2_runtime_destroy(rt);

    // Signal orchestration complete
    header->orchestrator_done = 1;
}

}  // extern "C"
