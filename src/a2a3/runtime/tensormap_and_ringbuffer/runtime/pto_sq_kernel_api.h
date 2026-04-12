/**
 * PTO SQ Kernel API — generic async remote-copy abstraction for AICore kernels.
 *
 * Two usage paths, both ending with CQ registration:
 *
 * Path 1 — High-level (send_request_entry, one-stop):
 *
 *   auto desc = pto2_remote_copy_descriptor(dst, src, scratch, context);
 *   uint64_t tag = pto2_send_request_entry(PTO2_ENGINE_SDMA, sq_id, desc);
 *   pto2_save_expected_completion(PTO2_ENGINE_SDMA, cq, tag);
 *   pto2_cq_flush();
 *
 * Path 2 — Low-level (sq_open + direct ISA instruction):
 *
 *   auto session = pto2_sq_open(PTO2_ENGINE_SDMA, sq_id, scratch, context);
 *   PTO2AsyncEvent event = pto2_backend_remote_copy_put(dst, src, session);
 *   pto2_save_expected_completion(cq, event);
 *   pto2_cq_flush();
 *
 * Layering:
 *   send_request_entry = sq_open + ISA instruction (syntactic sugar)
 *   sq_open            = session management (BuildAsyncSession wrapper)
 *
 * Requires:
 *   - PTO-ISA headers included before this header
 *   - __gm__ and __aicore__ defined before this header
 *   - HW build only (uses PTO-ISA async instructions)
 */

#ifndef PTO_SQ_KERNEL_API_H
#define PTO_SQ_KERNEL_API_H

#include "pto_cq_types.h"
#include "pto_cq_kernel_api.h"
#include "aicore/pto_async_backend_kernel.h"

// SQ engine types — aliases for the unified PTO2_ENGINE_* constants
#define PTO2_SQ_ENGINE_SDMA   PTO2_ENGINE_SDMA
// #define PTO2_SQ_ENGINE_CCU    PTO2_ENGINE_CCU    // future
// #define PTO2_SQ_ENGINE_URMA   PTO2_ENGINE_URMA   // future

#define PTO2_SQ_ID_AUTO  UINT32_MAX

using PTO2AsyncSession = PTO2BackendAsyncSession;
using PTO2AsyncEvent = PTO2BackendAsyncEvent;

struct PTO2RemoteCopyBaseConfig {
    uint32_t block_bytes{0};
    uint32_t block_offset{0};
    uint32_t repeat_times{1};
};

// ============================================================================
// pto2_sq_open — build async session for a hardware engine queue
//
// This is the foundation layer. Both send_request_entry (high-level)
// and direct ISA usage (low-level) go through this to obtain a session.
// ============================================================================

template <typename ScratchTile>
inline __aicore__ PTO2AsyncSession pto2_sq_open(
    uint32_t sq_type,
    uint32_t sq_id,
    ScratchTile& scratch,
    __gm__ uint8_t* context,
    uint32_t sync_id = 0,
    const PTO2RemoteCopyBaseConfig& base_config = {})
{
    (void)sq_type;
    return pto2_backend_remote_copy_open(
        sq_id, scratch, context, sync_id,
        base_config.block_bytes,
        base_config.block_offset,
        base_config.repeat_times);
}

// ============================================================================
// pto2_save_expected_completion — AsyncEvent overload
//
// Accepts a PTO-ISA AsyncEvent directly, auto-extracting engine and handle.
// For the low-level path where the user calls ISA instructions directly.
// ============================================================================

inline __aicore__ void pto2_save_expected_completion(
    volatile __gm__ PTO2CompletionQueue* cq,
    const PTO2AsyncEvent& event)
{
    pto2_save_expected_completion(
        pto2_backend_async_event_engine(event),
        cq,
        pto2_backend_async_event_handle(event));
}

enum class PTO2RemoteCopyRequestOp : uint32_t {
    Put = 0,
    Get = 1,
};

// ============================================================================
// SDMA descriptor + factories (for high-level path)
// ============================================================================

template <typename GlobalDstData, typename GlobalSrcData, typename ScratchTile>
struct PTO2RemoteCopyDescriptor {
    GlobalDstData& dst;
    GlobalSrcData& src;
    ScratchTile& scratch;
    __gm__ uint8_t* context;
    uint32_t sync_id;
    PTO2RemoteCopyBaseConfig base_config;
    PTO2RemoteCopyRequestOp op;
};

template <typename GlobalDstData, typename GlobalSrcData, typename ScratchTile>
inline __aicore__ PTO2RemoteCopyDescriptor<GlobalDstData, GlobalSrcData, ScratchTile>
pto2_remote_copy_descriptor(
    GlobalDstData& dst,
    GlobalSrcData& src,
    ScratchTile& scratch,
    __gm__ uint8_t* context,
    uint32_t sync_id = 0,
    const PTO2RemoteCopyBaseConfig& base_config = {})
{
    return {dst, src, scratch, context, sync_id, base_config,
            PTO2RemoteCopyRequestOp::Put};
}

template <typename GlobalDstData, typename GlobalSrcData, typename ScratchTile>
inline __aicore__ PTO2RemoteCopyDescriptor<GlobalDstData, GlobalSrcData, ScratchTile>
pto2_remote_copy_tget_descriptor(
    GlobalDstData& dst,
    GlobalSrcData& src,
    ScratchTile& scratch,
    __gm__ uint8_t* context,
    uint32_t sync_id = 0,
    const PTO2RemoteCopyBaseConfig& base_config = {})
{
    return {dst, src, scratch, context, sync_id, base_config,
            PTO2RemoteCopyRequestOp::Get};
}

using PTO2SdmaRequestOp = PTO2RemoteCopyRequestOp;
template <typename GlobalDstData, typename GlobalSrcData, typename ScratchTile>
using PTO2SdmaDescriptor = PTO2RemoteCopyDescriptor<GlobalDstData, GlobalSrcData, ScratchTile>;

template <typename GlobalDstData, typename GlobalSrcData, typename ScratchTile>
inline __aicore__ PTO2SdmaDescriptor<GlobalDstData, GlobalSrcData, ScratchTile>
pto2_sdma_descriptor(
    GlobalDstData& dst,
    GlobalSrcData& src,
    ScratchTile& scratch,
    __gm__ uint8_t* context,
    uint32_t sync_id = 0,
    const PTO2RemoteCopyBaseConfig& base_config = {})
{
    return pto2_remote_copy_descriptor(dst, src, scratch, context, sync_id, base_config);
}

template <typename GlobalDstData, typename GlobalSrcData, typename ScratchTile>
inline __aicore__ PTO2SdmaDescriptor<GlobalDstData, GlobalSrcData, ScratchTile>
pto2_sdma_tget_descriptor(
    GlobalDstData& dst,
    GlobalSrcData& src,
    ScratchTile& scratch,
    __gm__ uint8_t* context,
    uint32_t sync_id = 0,
    const PTO2RemoteCopyBaseConfig& base_config = {})
{
    return pto2_remote_copy_tget_descriptor(dst, src, scratch, context, sync_id, base_config);
}

// ============================================================================
// pto2_send_request_entry — high-level, sugar over sq_open + async ISA op
//
// Original design: tag = pto2_send_request_entry(SQ_TYPE, SQ_ID, descriptor)
// Internally: sq_open(session params from desc) → async ISA op → tag
// ============================================================================

template <typename GlobalDstData, typename GlobalSrcData, typename ScratchTile>
inline __aicore__ uint64_t pto2_send_request_entry(
    uint32_t sq_type,
    uint32_t sq_id,
    PTO2RemoteCopyDescriptor<GlobalDstData, GlobalSrcData, ScratchTile>& desc)
{
    PTO2AsyncSession session = pto2_sq_open(
        sq_type, sq_id, desc.scratch, desc.context,
        desc.sync_id, desc.base_config);
    if (!session.valid) return 0;

    PTO2AsyncEvent event;
    if (desc.op == PTO2RemoteCopyRequestOp::Get) {
        event = pto2_backend_remote_copy_get(desc.dst, desc.src, session);
    } else {
        event = pto2_backend_remote_copy_put(desc.dst, desc.src, session);
    }
    return pto2_backend_async_event_valid(event)
               ? pto2_backend_async_event_handle(event)
               : 0;
}

#endif  // PTO_SQ_KERNEL_API_H
