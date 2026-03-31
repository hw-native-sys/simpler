/**
 * NotifyWait Kernel — register notification counter as CQ condition (func_id=3)
 *
 * Trivial deferred-completion kernel: registers a COUNTER wait condition
 * for the notification counter, then returns immediately. The scheduler
 * polls the counter via the CQ mechanism and completes this task once
 * *notify_counter >= expected_value.
 *
 * Kernel args layout:
 *   args[0] = &Tensor(dummy_notify)    — output (dependency token for downstream)
 *   args[1] = notify_counter_addr      — scalar (GM int32* to poll)
 *   args[2] = expected_value           — scalar (threshold)
 *   args[3] = cq_addr                  — scalar (auto-appended by deferred submit)
 */

#include <cstdint>

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "pto_cq_kernel_api.h"
#include "pto_notify_kernel_api.h"

extern "C" __aicore__ __attribute__((always_inline))
void kernel_entry(__gm__ int64_t* args) {
    // args[0] = dummy_notify tensor (output, unused by kernel)
    uint64_t notify_counter_addr = static_cast<uint64_t>(args[1]);
    uint32_t expected_value = static_cast<uint32_t>(args[2]);
    uint64_t cq_addr = static_cast<uint64_t>(args[3]);

    volatile __gm__ PTO2CompletionQueue* cq = pto2_cq_get(cq_addr);
    pto2_cq_reset(cq);
    pto2_save_expected_notification_counter(
        cq,
        reinterpret_cast<volatile __gm__ int32_t*>(static_cast<uintptr_t>(notify_counter_addr)),
        expected_value);
    dcci((__gm__ int32_t*)cq, cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
    dsb(DSB_DDR);
    pipe_barrier(PIPE_ALL);
}
