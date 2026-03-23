/**
 * Async Completion Demo - Device-side orchestration (dual-mode)
 *
 * DAG structure:
 *   t0 (producer): out = in * 2.0  [deferred completion]
 *   t1 (consumer): result = out + 1.0  [run-to-completion]
 *   Dependency: t0 -> t1 (consumer reads producer's output tensor)
 *
 * Dual-mode dispatch:
 *   - Sim mode (no SDMA workspace): func_id=0, complete_in_future=1
 *     The sim producer kernel writes a flag directly; the scheduler polls it.
 *   - HW mode (SDMA available): func_id=2, complete_in_future=2
 *     The HW producer kernel issues TPUT_ASYNC and writes the AsyncEvent handle
 *     to a GM buffer.  The scheduler does two-level indirection to poll the
 *     SdmaEventRecord.flag.
 *
 * Args layout (from golden.py):
 *   [ptr_in, ptr_out, ptr_result, ptr_event_handle_output,
 *    size_in, size_out, size_result, size_event_handle_output, SIZE]
 *   + [gm_heap, heap_size] appended by runtime_maker.cpp
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"

#define ARG_PTR_IN                   0
#define ARG_PTR_OUT                  1
#define ARG_PTR_RESULT               2
#define ARG_PTR_EVENT_HANDLE_OUTPUT  3

#define ARG_SIZE_IN                   4
#define ARG_SIZE_OUT                  5
#define ARG_SIZE_RESULT               6
#define ARG_SIZE_EVENT_HANDLE_OUTPUT  7

#define ARG_SIZE 8

extern "C" {

__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(uint64_t* args, int arg_count) {
    (void)args;
    (void)arg_count;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 9,
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(uint64_t* args, int arg_count,
                               int orch_thread_num, int orch_thread_index) {
    (void)arg_count;
    (void)orch_thread_num;
    (void)orch_thread_index;

    void* in_ptr     = (void*)(uintptr_t)args[ARG_PTR_IN];
    void* out_ptr    = (void*)(uintptr_t)args[ARG_PTR_OUT];
    void* result_ptr = (void*)(uintptr_t)args[ARG_PTR_RESULT];
    uint64_t event_handle_output_gm = args[ARG_PTR_EVENT_HANDLE_OUTPUT];
    int SIZE = (int)(args[ARG_SIZE] & 0x7FFFFFFF);

    uint64_t sdma_workspace = pto2_rt_get_sdma_workspace();

    LOG_INFO("async_demo: SIZE=%d, event_handle_output=0x%lx, sdma_workspace=0x%lx",
             SIZE, event_handle_output_gm, sdma_workspace);

    uint32_t shapes[1] = {(uint32_t)SIZE};
    Tensor ext_in     = make_tensor_external(in_ptr, shapes, 1, DataType::FLOAT32);
    Tensor ext_out    = make_tensor_external(out_ptr, shapes, 1, DataType::FLOAT32);
    Tensor ext_result = make_tensor_external(result_ptr, shapes, 1, DataType::FLOAT32);

    if (sdma_workspace != 0) {
        // HW mode: use TPUT_ASYNC kernel (func_id=2) with SDMA workspace
        // complete_in_future=2 → scheduler uses two-level indirection
        PTOParam params_producer;
        params_producer.add_input(ext_in);
        params_producer.add_output(ext_out);
        params_producer.add_scalar(sdma_workspace);
        pto2_rt_submit_aiv_task_async_sdma(2, params_producer, event_handle_output_gm);

        LOG_INFO("async_demo: HW mode - submitted TPUT_ASYNC producer (func_id=2)");
    } else {
        // Sim mode: use simulated producer (func_id=0) with direct flag write
        // complete_in_future=1 → scheduler polls flag address directly
        PTOParam params_producer;
        params_producer.add_input(ext_in);
        params_producer.add_output(ext_out);
        pto2_rt_submit_aiv_task_async(0, params_producer, event_handle_output_gm);

        LOG_INFO("async_demo: Sim mode - submitted simulated producer (func_id=0)");
    }

    // t1 (consumer): result = out + 1.0 — normal run-to-completion
    PTOParam params_consumer;
    params_consumer.add_input(ext_out);
    params_consumer.add_output(ext_result);
    pto2_rt_submit_aiv_task(1, params_consumer);
}

}  // extern "C"
