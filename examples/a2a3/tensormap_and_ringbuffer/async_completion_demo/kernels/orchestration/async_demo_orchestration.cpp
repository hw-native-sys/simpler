/**
 * Async Completion Demo - Device-side orchestration (CQ model)
 *
 * Two execution modes share this file:
 *
 * 1. Single-card / sim mode (legacy demo):
 *    t0 (producer): out = in * 2.0  [deferred completion via CQ]
 *    t1 (consumer): result = out + 1.0  [run-to-completion]
 *
 * 2. Two-card hardware mode:
 *    both ranks submit one deferred producer task that TGET_ASYNCs the peer
 *    rank's input buffer into local out, then run the normal consumer on out.
 *
 * CQ model:
 *   Orchestration marks t0 as complete_in_future and passes a CQ address.
 *   The producer kernel decides at runtime what completions it needs and writes
 *   them into the completion queue. The scheduler reads the CQ after the kernel
 *   returns and registers completions dynamically.
 */

#include <stddef.h>
#include <stdint.h>

#include "common/comm_context.h"
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
    return PTO2OrchestrationConfig{
        .expected_arg_count = (arg_count >= 9) ? 9 : 4,
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(uint64_t* args, int arg_count,
                               int orch_thread_num, int orch_thread_index) {
    (void)arg_count;
    (void)orch_thread_num;
    if (orch_thread_index != 0) return;

    if (arg_count == 4) {
        void* in_ptr  = (void*)(uintptr_t)args[0];
        void* out_ptr = (void*)(uintptr_t)args[1];
        void* result_ptr = (void*)(uintptr_t)args[2];
        auto* comm_ctx = reinterpret_cast<CommDeviceContext*>((uintptr_t)args[3]);
        int my_rank = (int)comm_ctx->rankId;

        uint32_t shapes[1] = {128 * 128};
        Tensor ext_in  = make_tensor_external(in_ptr, shapes, 1, DataType::FLOAT32);
        Tensor ext_out = make_tensor_external(out_ptr, shapes, 1, DataType::FLOAT32);
        Tensor ext_result = make_tensor_external(result_ptr, shapes, 1, DataType::FLOAT32);

        uint64_t sdma_context = pto2_rt_get_sdma_context();
        uint64_t cq = pto2_rt_alloc_cq();
        if (sdma_context == 0 || cq == 0) {
            LOG_ERROR("async_demo 2P: rank %d failed to get SDMA context or CQ (sdma=0x%lx, cq=0x%lx)",
                      my_rank, sdma_context, cq);
            return;
        }

        PTOParam params_producer;
        params_producer.add_input(ext_in);
        params_producer.add_output(ext_out);
        params_producer.add_scalar((uint64_t)(uintptr_t)comm_ctx);
        params_producer.add_scalar(sdma_context);
        pto2_rt_submit_aiv_task_deferred(2, params_producer, cq);

        PTOParam params_consumer;
        params_consumer.add_input(ext_out);
        params_consumer.add_output(ext_result);
        pto2_rt_submit_aiv_task(1, params_consumer);

        LOG_INFO("async_demo 2P: rank %d submitted TGET_ASYNC producer with CQ", my_rank);
        return;
    }

    void* in_ptr     = (void*)(uintptr_t)args[ARG_PTR_IN];
    void* out_ptr    = (void*)(uintptr_t)args[ARG_PTR_OUT];
    void* result_ptr = (void*)(uintptr_t)args[ARG_PTR_RESULT];
    uint64_t event_handle_output_gm = args[ARG_PTR_EVENT_HANDLE_OUTPUT];
    int SIZE = (int)(args[ARG_SIZE] & 0x7FFFFFFF);

    uint64_t sdma_context = pto2_rt_get_sdma_context();
    uint64_t cq = pto2_rt_alloc_cq();

    LOG_INFO("async_demo: SIZE=%d, event_handle_output=0x%lx, sdma_context=0x%lx, cq=0x%lx",
             SIZE, event_handle_output_gm, sdma_context, cq);

    uint32_t shapes[1] = {(uint32_t)SIZE};
    Tensor ext_in     = make_tensor_external(in_ptr, shapes, 1, DataType::FLOAT32);
    Tensor ext_out    = make_tensor_external(out_ptr, shapes, 1, DataType::FLOAT32);
    Tensor ext_result = make_tensor_external(result_ptr, shapes, 1, DataType::FLOAT32);

    if (sdma_context != 0) {
        // HW mode: kernel issues async SDMA request and puts event.handle directly in CQ entry.
        PTOParam params_producer;
        params_producer.add_input(ext_in);
        params_producer.add_output(ext_out);
        params_producer.add_scalar(sdma_context);
        pto2_rt_submit_aiv_task_deferred(2, params_producer, cq);

        LOG_INFO("async_demo: HW mode - submitted async SDMA producer (func_id=2) with CQ");
    } else {
        PTOParam params_producer;
        params_producer.add_input(ext_in);
        params_producer.add_output(ext_out);
        params_producer.add_scalar(event_handle_output_gm);
        pto2_rt_submit_aiv_task_deferred(0, params_producer, cq);

        LOG_INFO("async_demo: Sim mode - submitted producer (func_id=0) with CQ");
    }

    // t1 (consumer): result = out + 1.0 — normal run-to-completion
    PTOParam params_consumer;
    params_consumer.add_input(ext_out);
    params_consumer.add_output(ext_result);
    pto2_rt_submit_aiv_task(1, params_consumer);
}

}  // extern "C"
