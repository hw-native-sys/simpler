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

#include <pto/comm/pto_comm_inst.hpp>

enum class PTO2NotifyOp : uint32_t {
    Set = 0,
    AtomicAdd = 1,
};

inline __aicore__ pto::comm::NotifyOp pto2_to_notify_op(PTO2NotifyOp op) {
    return op == PTO2NotifyOp::Set
               ? pto::comm::NotifyOp::Set
               : pto::comm::NotifyOp::AtomicAdd;
}

inline __aicore__ void pto2_send_notification(
    volatile __gm__ int32_t* remote_counter_addr,
    int32_t value = 1,
    PTO2NotifyOp op = PTO2NotifyOp::AtomicAdd)
{
    pto::comm::Signal signal((__gm__ int32_t*)remote_counter_addr);
    pto::comm::TNOTIFY(signal, value, pto2_to_notify_op(op));
#if defined(PIPE_ALL)
    pipe_barrier(PIPE_ALL);
#endif
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
