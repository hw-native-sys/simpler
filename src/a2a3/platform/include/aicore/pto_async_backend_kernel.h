/**
 * A2/A3 async backend helpers for AICore kernels.
 *
 * This header is the platform/backend implementation layer behind the generic
 * PTO async kernel API in runtime/.  Runtime headers should call these helpers
 * rather than hard-code PTO-ISA engine details directly.
 */

#ifndef SRC_A2A3_PLATFORM_INCLUDE_AICORE_PTO_ASYNC_BACKEND_KERNEL_H_
#define SRC_A2A3_PLATFORM_INCLUDE_AICORE_PTO_ASYNC_BACKEND_KERNEL_H_

#include <stdint.h>

#include <pto/comm/async/async_types.hpp>
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/npu/comm/async/sdma/sdma_types.hpp>

using PTO2BackendAsyncSession = pto::comm::AsyncSession;
using PTO2BackendAsyncEvent = pto::comm::AsyncEvent;

inline constexpr uint32_t pto2_backend_remote_copy_default_block_bytes() {
    return pto::comm::sdma::kDefaultSdmaBlockBytes;
}

template <typename ScratchTile>
inline __aicore__ PTO2BackendAsyncSession pto2_backend_remote_copy_open(
    uint32_t sq_id,
    ScratchTile &scratch,
    __gm__ uint8_t *context,
    uint32_t sync_id,
    uint32_t block_bytes,
    uint32_t block_offset,
    uint32_t repeat_times)
{
    PTO2BackendAsyncSession session;
    pto::comm::sdma::SdmaBaseConfig base_config{
        block_bytes != 0 ? block_bytes : pto::comm::sdma::kDefaultSdmaBlockBytes,
        block_offset,
        repeat_times,
    };
    pto::comm::BuildAsyncSession<pto::comm::DmaEngine::SDMA>(
        scratch, context, session, sync_id, base_config, sq_id);
    return session;
}

template <typename GlobalDstData, typename GlobalSrcData>
inline __aicore__ PTO2BackendAsyncEvent pto2_backend_remote_copy_put(
    GlobalDstData &dst,
    GlobalSrcData &src,
    const PTO2BackendAsyncSession &session)
{
    return pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::SDMA>(dst, src, session);
}

template <typename GlobalDstData, typename GlobalSrcData>
inline __aicore__ PTO2BackendAsyncEvent pto2_backend_remote_copy_get(
    GlobalDstData &dst,
    GlobalSrcData &src,
    const PTO2BackendAsyncSession &session)
{
    return pto::comm::TGET_ASYNC<pto::comm::DmaEngine::SDMA>(dst, src, session);
}

inline __aicore__ bool pto2_backend_async_event_valid(const PTO2BackendAsyncEvent &event) {
    return event.valid();
}

inline __aicore__ uint32_t pto2_backend_async_event_engine(const PTO2BackendAsyncEvent &event) {
    return static_cast<uint32_t>(event.engine);
}

inline __aicore__ uint64_t pto2_backend_async_event_handle(const PTO2BackendAsyncEvent &event) {
    return event.handle;
}

inline __aicore__ void pto2_backend_send_notification(
    volatile __gm__ int32_t *remote_counter_addr,
    int32_t value,
    uint32_t op)
{
    pto::comm::NotifyOp notify_op =
        op == 0 ? pto::comm::NotifyOp::Set : pto::comm::NotifyOp::AtomicAdd;
    pto::comm::Signal signal((__gm__ int32_t *)remote_counter_addr);
    pto::comm::TNOTIFY(signal, value, notify_op);
#if defined(PIPE_ALL)
    pipe_barrier(PIPE_ALL);
#endif
}

#endif  // SRC_A2A3_PLATFORM_INCLUDE_AICORE_PTO_ASYNC_BACKEND_KERNEL_H_
