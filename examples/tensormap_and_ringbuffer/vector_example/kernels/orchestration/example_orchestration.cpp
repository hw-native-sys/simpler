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
//   [dev_a, dev_b, dev_f, dev_c, dev_d, dev_e, size_a, size_b, size_f, size_c, size_d, size_e, SIZE]
//   + [gm_heap, heap_size] appended by runtime_maker.cpp
//
// Generic access: gm_heap = args[arg_count - 2], heap_size = args[arg_count - 1]
// =============================================================================

// Tensor device pointers (order from code_runner.py: inputs, outputs, intermediates)
#define ARG_DEV_A      0
#define ARG_DEV_B      1
#define ARG_DEV_F      2   // output
#define ARG_DEV_C      3   // intermediate
#define ARG_DEV_D      4   // intermediate
#define ARG_DEV_E      5   // intermediate

// Tensor sizes (same order as pointers)
#define ARG_SIZE_A     6
#define ARG_SIZE_B     7
#define ARG_SIZE_F     8
#define ARG_SIZE_C     9
#define ARG_SIZE_D     10
#define ARG_SIZE_E     11

// Element count (scalar)
#define ARG_SIZE       12

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

extern "C" {

__attribute__((visibility("default")))
void aicpu_orchestration_entry(void* sm_ptr, uint64_t* args, int arg_count) {
    // Validate inputs
    if (!sm_ptr || !args || arg_count < 13) {
        if (sm_ptr) {
            *(volatile int32_t*)((char*)sm_ptr + 8) = 1;
        }
        return;
    }

    // Extract device pointers
    void* dev_a = (void*)(uintptr_t)args[ARG_DEV_A];
    void* dev_b = (void*)(uintptr_t)args[ARG_DEV_B];
    void* dev_f = (void*)(uintptr_t)args[ARG_DEV_F];
    void* dev_c = (void*)(uintptr_t)args[ARG_DEV_C];
    void* dev_d = (void*)(uintptr_t)args[ARG_DEV_D];
    void* dev_e = (void*)(uintptr_t)args[ARG_DEV_E];
    size_t size_a = (size_t)args[ARG_SIZE_A];
    size_t size_f = (size_t)args[ARG_SIZE_F];
    int SIZE = (int)(args[ARG_SIZE] & 0x7FFFFFFF);

    // Validate intermediate pointers
    if (!dev_c || !dev_d || !dev_e) {
        *(volatile int32_t*)((char*)sm_ptr + 8) = 1;
        return;
    }

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
        *(volatile int32_t*)((char*)sm_ptr + 8) = 1;
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
        *(volatile int32_t*)((char*)sm_ptr + 8) = 1;
        return;
    }

    int32_t tile = 0;
    int32_t sz = (int32_t)BYTES;
    if (sz <= 0) sz = (int32_t)size_a;

    // Use RAII scope guard for automatic scope management.
    // PTO2_SCOPE creates a scoped block where pto2_rt_scope_begin() is called
    // at the start and pto2_rt_scope_end() is called automatically at the end
    // (even in error paths). This eliminates manual cleanup and prevents bugs.
    // See src/runtime/rt2/runtime/pto_runtime2.h for alternative usage patterns.
    PTO2_SCOPE(rt) {
        // t0: c = a + b (kernel_id=0, kernel_add)
        PTO2TaskParam params_t0[] = {
            PTO2_INPUT(dev_a, tile, sz),
            PTO2_INPUT(dev_b, tile, sz),
            PTO2_OUTPUT(dev_c, tile, sz),
        };
        (void)pto2_rt_submit_task(rt, 0, PTO2_WORKER_VECTOR, "kernel_add", params_t0, 3);

        // t1: d = c + 1 (kernel_id=1, kernel_add_scalar)
        PTO2TaskParam params_t1[] = {
            PTO2_INPUT(dev_c, tile, sz),
            PTO2_OUTPUT(dev_d, tile, sz),
        };
        (void)pto2_rt_submit_task(rt, 1, PTO2_WORKER_VECTOR, "kernel_add_scalar", params_t1, 2);

        // t2: e = c + 2 (kernel_id=1, kernel_add_scalar)
        PTO2TaskParam params_t2[] = {
            PTO2_INPUT(dev_c, tile, sz),
            PTO2_OUTPUT(dev_e, tile, sz),
        };
        (void)pto2_rt_submit_task(rt, 1, PTO2_WORKER_VECTOR, "kernel_add_scalar", params_t2, 2);

        // t3: f = d * e (kernel_id=2, kernel_mul)
        PTO2TaskParam params_t3[] = {
            PTO2_INPUT(dev_d, tile, sz),
            PTO2_INPUT(dev_e, tile, sz),
            PTO2_OUTPUT(dev_f, tile, sz),
        };
        int32_t task3_id = pto2_rt_submit_task(rt, 2, PTO2_WORKER_VECTOR, "kernel_mul", params_t3, 3);

        // Set graph output pointer for host copy-back
        void* graph_out_ptr = pto2_rt_get_output(rt, task3_id, 0);
        if (graph_out_ptr && size_f > 0) {
            rt->sm_handle->header->graph_output_ptr = (uint64_t)(uintptr_t)graph_out_ptr;
            rt->sm_handle->header->graph_output_size = (int32_t)size_f;
        }
    } // PTO2_SCOPE ends here - automatic pto2_rt_scope_end() called

    pto2_rt_orchestration_done(rt);
    pto2_runtime_destroy(rt);

    // Signal orchestration complete
    *(volatile int32_t*)((char*)sm_ptr + 8) = 1;
}

}  // extern "C"
