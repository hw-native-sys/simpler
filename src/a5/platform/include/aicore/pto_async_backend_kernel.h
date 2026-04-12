/**
 * A5 async backend helpers for AICore kernels.
 *
 * This is currently a stub backend so the generic runtime async headers can
 * compile without depending on A2/A3-specific transport implementations.
 */

#ifndef SRC_A5_PLATFORM_INCLUDE_AICORE_PTO_ASYNC_BACKEND_KERNEL_H_
#define SRC_A5_PLATFORM_INCLUDE_AICORE_PTO_ASYNC_BACKEND_KERNEL_H_

#include <stdint.h>

struct PTO2BackendAsyncSession {
    bool valid{false};
};

struct PTO2BackendAsyncEvent {
    uint32_t engine{0};
    uint64_t handle{0};

    bool valid() const { return false; }
};

inline constexpr uint32_t pto2_backend_remote_copy_default_block_bytes() { return 0; }

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
    (void)sq_id;
    (void)scratch;
    (void)context;
    (void)sync_id;
    (void)block_bytes;
    (void)block_offset;
    (void)repeat_times;
    return {};
}

template <typename GlobalDstData, typename GlobalSrcData>
inline __aicore__ PTO2BackendAsyncEvent pto2_backend_remote_copy_put(
    GlobalDstData &dst,
    GlobalSrcData &src,
    const PTO2BackendAsyncSession &session)
{
    (void)dst;
    (void)src;
    (void)session;
    return {};
}

template <typename GlobalDstData, typename GlobalSrcData>
inline __aicore__ PTO2BackendAsyncEvent pto2_backend_remote_copy_get(
    GlobalDstData &dst,
    GlobalSrcData &src,
    const PTO2BackendAsyncSession &session)
{
    (void)dst;
    (void)src;
    (void)session;
    return {};
}

inline __aicore__ bool pto2_backend_async_event_valid(const PTO2BackendAsyncEvent &event) {
    return event.valid();
}

inline __aicore__ uint32_t pto2_backend_async_event_engine(const PTO2BackendAsyncEvent &event) {
    return event.engine;
}

inline __aicore__ uint64_t pto2_backend_async_event_handle(const PTO2BackendAsyncEvent &event) {
    return event.handle;
}

inline __aicore__ void pto2_backend_send_notification(
    volatile __gm__ int32_t *remote_counter_addr,
    int32_t value,
    uint32_t op)
{
    (void)remote_counter_addr;
    (void)value;
    (void)op;
}

#endif  // SRC_A5_PLATFORM_INCLUDE_AICORE_PTO_ASYNC_BACKEND_KERNEL_H_
