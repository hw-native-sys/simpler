/**
 * Async Notify Demo - Device-side orchestration
 *
 * Two-card hardware mode:
 *   t0 (producer, func_id=0): out = in * 2, then TNOTIFY(AtomicAdd) the
 *                  peer's window counter. Completes normally (RTC).
 *   t1 (notify_wait, func_id=2, deferred): registers notification counter
 *                  condition (counter >= 1) via CQ, returns immediately.
 *                  Produces dummy_notify tensor for dependency chain.
 *   t2 (consumer, func_id=1): result = out + notify_counter.
 *                  Depends on both producer (via ext_out) and notify_wait
 *                  (via dummy_notify), ensuring counter >= 1 before reading.
 *
 * The notify counter is pre-zeroed by the distributed runner input loader.
 */

#include <stdint.h>

#include "common/comm_context.h"
#include "pto_orchestration_api.h"

extern "C" {

__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(uint64_t* args, int arg_count) {
    (void)args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 5,
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(uint64_t* args, int arg_count,
                               int orch_thread_num, int orch_thread_index) {
    (void)orch_thread_num;
    if (orch_thread_index != 0) return;

    if (arg_count != 5) {
        LOG_ERROR("async_notify_demo: expected 5 args, got %d", arg_count);
        return;
    }

    void* in_ptr = (void*)(uintptr_t)args[0];
    void* out_ptr = (void*)(uintptr_t)args[1];
    void* result_ptr = (void*)(uintptr_t)args[2];
    void* notify_counter_ptr = (void*)(uintptr_t)args[3];
    auto* comm_ctx = reinterpret_cast<CommDeviceContext*>((uintptr_t)args[4]);
    int my_rank = (int)comm_ctx->rankId;

    uint32_t shapes[1] = {128 * 128};
    Tensor ext_in = make_tensor_external(in_ptr, shapes, 1, DataType::FLOAT32);
    Tensor ext_out = make_tensor_external(out_ptr, shapes, 1, DataType::FLOAT32);
    Tensor ext_result = make_tensor_external(result_ptr, shapes, 1, DataType::FLOAT32);

    uint64_t cq_notify = pto2_rt_alloc_cq();
    if (cq_notify == 0) {
        LOG_ERROR("async_notify_demo: rank %d failed CQ alloc", my_rank);
        return;
    }

    // Producer: normal run-to-completion task (sends TNOTIFY to peer)
    PTOParam params_producer;
    params_producer.add_input(ext_in);
    params_producer.add_output(ext_out);
    params_producer.add_scalar((uint64_t)(uintptr_t)notify_counter_ptr);
    params_producer.add_scalar((uint64_t)(uintptr_t)comm_ctx);
    pto2_rt_submit_aiv_task(0, params_producer);

    // NotifyWait: deferred task that waits for notification counter >= 1.
    // Produces dummy_notify so the consumer can depend on it via TensorMap.
    uint32_t dummy_shape[1] = { 1 };
    Tensor dummy_notify = make_tensor(dummy_shape, 1, DataType::INT32);

    PTOParam params_wait;
    params_wait.add_output(dummy_notify);
    params_wait.add_scalar((uint64_t)(uintptr_t)notify_counter_ptr);
    params_wait.add_scalar((uint64_t)1);
    pto2_rt_submit_aiv_task_deferred(2, params_wait, cq_notify);

    // Consumer: depends on producer (via ext_out) and notify_wait (via dummy_notify).
    // Guaranteed notify_counter >= 1 when this task runs.
    PTOParam params_consumer;
    params_consumer.add_input(dummy_notify);
    params_consumer.add_input(ext_out);
    params_consumer.add_output(ext_result);
    params_consumer.add_scalar((uint64_t)(uintptr_t)notify_counter_ptr);
    pto2_rt_submit_aiv_task(1, params_consumer);

    LOG_INFO("async_notify_demo: rank %d producer=RTC, notify_wait=deferred(counter=0x%lx), consumer=RTC",
             my_rank, (uint64_t)(uintptr_t)notify_counter_ptr);
}

}  // extern "C"
