/**
 * PTO Notify Kernel API — notification counter abstraction for AICore kernels.
 *
 * This wraps PTO-ISA TNOTIFY and maps the local counter wait condition onto
 * the runtime's existing COUNTER deferred-completion path.
 *
 * Requires:
 *   - PTO-ISA headers included before this header
 *   - __gm__ and __aicore__ defined before this header
 */

#ifndef PTO_NOTIFY_KERNEL_API_H
#define PTO_NOTIFY_KERNEL_API_H

#include "pto_cq_kernel_api.h"
#include "aicore/pto_async_backend_kernel.h"

enum class PTO2NotifyOp : uint32_t {
    Set = 0,
    AtomicAdd = 1,
};

inline __aicore__ void pto2_send_notification(
    volatile __gm__ int32_t* remote_counter_addr,
    int32_t value = 1,
    PTO2NotifyOp op = PTO2NotifyOp::AtomicAdd)
{
    pto2_backend_send_notification(remote_counter_addr, value, static_cast<uint32_t>(op));
}

inline __aicore__ void pto2_save_expected_notification_counter(
    volatile __gm__ PTO2CompletionQueue* cq,
    volatile __gm__ int32_t* local_counter_addr,
    uint32_t expected_value)
{
    pto2_save_expected_completion(PTO2_ENGINE_SDMA, cq,
                                  (uint64_t)local_counter_addr,
                                  expected_value);
}

#endif  // PTO_NOTIFY_KERNEL_API_H
