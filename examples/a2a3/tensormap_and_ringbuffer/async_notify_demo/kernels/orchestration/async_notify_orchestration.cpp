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
PTO2OrchestrationConfig aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 5,
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    int arg_count = orch_args.tensor_count() + orch_args.scalar_count();

    if (arg_count != 5) {
        LOG_ERROR("async_notify_demo: expected 5 args, got %d", arg_count);
        return;
    }

    void *in_ptr = nullptr;
    void *out_ptr = nullptr;
    void *result_ptr = nullptr;
    void *notify_counter_ptr = nullptr;
    auto *comm_ctx = reinterpret_cast<CommDeviceContext *>(static_cast<uintptr_t>(
        orch_args.tensor_count() == 0 ? orch_args.scalar(4) : orch_args.scalar(0)));
    if (orch_args.tensor_count() == 0) {
        in_ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(orch_args.scalar(0)));
        out_ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(orch_args.scalar(1)));
        result_ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(orch_args.scalar(2)));
        notify_counter_ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(orch_args.scalar(3)));
    } else {
        in_ptr = orch_args.tensor(0).data_as<void>();
        out_ptr = orch_args.tensor(1).data_as<void>();
        result_ptr = orch_args.tensor(2).data_as<void>();
        notify_counter_ptr = orch_args.tensor(3).data_as<void>();
    }
    int my_rank = (int)comm_ctx->rankId;

    uint32_t shapes[1] = {128 * 128};
    Tensor ext_in = make_tensor_external(in_ptr, shapes, 1, DataType::FLOAT32);
    Tensor ext_out = make_tensor_external(out_ptr, shapes, 1, DataType::FLOAT32);
    Tensor ext_result = make_tensor_external(result_ptr, shapes, 1, DataType::FLOAT32);

    // Producer: normal run-to-completion task (sends TNOTIFY to peer)
    Arg params_producer;
    params_producer.add_input(ext_in);
    params_producer.add_output(ext_out);
    params_producer.add_scalar((uint64_t)(uintptr_t)notify_counter_ptr);
    params_producer.add_scalar((uint64_t)(uintptr_t)comm_ctx);
    pto2_rt_submit_aiv_task(0, params_producer);

    // Returns a dependency token tensor for downstream tasks.
    Tensor notify_token = pto2_rt_submit_notification_wait_task(2, (uint64_t)(uintptr_t)notify_counter_ptr, 1);

    // Consumer: depends on producer (via ext_out) and notify_wait (via token).
    Arg params_consumer;
    params_consumer.add_input(notify_token);
    params_consumer.add_input(ext_out);
    params_consumer.add_output(ext_result);
    params_consumer.add_scalar((uint64_t)(uintptr_t)notify_counter_ptr);
    pto2_rt_submit_aiv_task(1, params_consumer);

    LOG_INFO("async_notify_demo: rank %d producer=RTC, notify_wait=deferred(counter=0x%lx), consumer=RTC",
             my_rank, (uint64_t)(uintptr_t)notify_counter_ptr);
}

}  // extern "C"
