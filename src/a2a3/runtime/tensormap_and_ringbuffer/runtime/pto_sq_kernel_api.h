/**
 * PTO SQ Kernel API — send queue abstraction for AICore kernels.
 *
 * Two usage paths, both ending with CQ registration:
 *
 * Path 1 — High-level (send_request_entry, one-stop):
 *
 *   auto desc = pto2_sdma_descriptor(dst, src, scratch, context);
 *   uint64_t tag = pto2_send_request_entry(PTO2_ENGINE_SDMA, sq_id, desc);
 *   pto2_save_expected_completion(PTO2_ENGINE_SDMA, cq, tag);
 *   pto2_cq_flush();
 *
 * Path 2 — Low-level (sq_open + direct ISA instruction):
 *
 *   auto session = pto2_sq_open(PTO2_ENGINE_SDMA, sq_id, scratch, context);
 *   AsyncEvent event = TPUT_ASYNC(dst, src, session);  // or TGET_ASYNC
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

#include <pto/comm/pto_comm_inst.hpp>
#include <pto/comm/async/async_types.hpp>
#include <pto/npu/comm/async/sdma/sdma_types.hpp>

// SQ engine types — aliases for the unified PTO2_ENGINE_* constants
#define PTO2_SQ_ENGINE_SDMA   PTO2_ENGINE_SDMA
// #define PTO2_SQ_ENGINE_CCU    PTO2_ENGINE_CCU    // future
// #define PTO2_SQ_ENGINE_URMA   PTO2_ENGINE_URMA   // future

#define PTO2_SQ_ID_AUTO  UINT32_MAX

// ============================================================================
// pto2_sq_open — build async session for a hardware engine queue
//
// This is the foundation layer. Both send_request_entry (high-level)
// and direct ISA usage (low-level) go through this to obtain a session.
// ============================================================================

template <typename ScratchTile>
inline __aicore__ pto::comm::AsyncSession pto2_sq_open(
    uint32_t sq_type,
    uint32_t sq_id,
    ScratchTile& scratch,
    __gm__ uint8_t* context,
    uint32_t sync_id = 0,
    const pto::comm::sdma::SdmaBaseConfig& base_config =
        {pto::comm::sdma::kDefaultSdmaBlockBytes, 0, 1})
{
    pto::comm::AsyncSession session;
    pto::comm::BuildAsyncSession<pto::comm::DmaEngine::SDMA>(
        scratch, context, session, sync_id, base_config, sq_id);
    return session;
}

// ============================================================================
// pto2_save_expected_completion — AsyncEvent overload
//
// Accepts a PTO-ISA AsyncEvent directly, auto-extracting engine and handle.
// For the low-level path where the user calls ISA instructions directly.
// ============================================================================

inline __aicore__ void pto2_save_expected_completion(
    volatile __gm__ PTO2CompletionQueue* cq,
    const pto::comm::AsyncEvent& event)
{
    uint32_t engine = static_cast<uint32_t>(event.engine);
    pto2_save_expected_completion(engine, cq, event.handle);
}

enum class PTO2SdmaRequestOp : uint32_t {
    TPut = 0,
    TGet = 1,
};

// ============================================================================
// SDMA descriptor + factories (for high-level path)
// ============================================================================

template <typename GlobalDstData, typename GlobalSrcData, typename ScratchTile>
struct PTO2SdmaDescriptor {
    GlobalDstData& dst;
    GlobalSrcData& src;
    ScratchTile& scratch;
    __gm__ uint8_t* context;
    uint32_t sync_id;
    pto::comm::sdma::SdmaBaseConfig base_config;
    PTO2SdmaRequestOp op;
};

template <typename GlobalDstData, typename GlobalSrcData, typename ScratchTile>
inline __aicore__ PTO2SdmaDescriptor<GlobalDstData, GlobalSrcData, ScratchTile>
pto2_sdma_descriptor(
    GlobalDstData& dst,
    GlobalSrcData& src,
    ScratchTile& scratch,
    __gm__ uint8_t* context,
    uint32_t sync_id = 0,
    const pto::comm::sdma::SdmaBaseConfig& base_config =
        {pto::comm::sdma::kDefaultSdmaBlockBytes, 0, 1})
{
    return {dst, src, scratch, context, sync_id, base_config,
            PTO2SdmaRequestOp::TPut};
}

template <typename GlobalDstData, typename GlobalSrcData, typename ScratchTile>
inline __aicore__ PTO2SdmaDescriptor<GlobalDstData, GlobalSrcData, ScratchTile>
pto2_sdma_tget_descriptor(
    GlobalDstData& dst,
    GlobalSrcData& src,
    ScratchTile& scratch,
    __gm__ uint8_t* context,
    uint32_t sync_id = 0,
    const pto::comm::sdma::SdmaBaseConfig& base_config =
        {pto::comm::sdma::kDefaultSdmaBlockBytes, 0, 1})
{
    return {dst, src, scratch, context, sync_id, base_config,
            PTO2SdmaRequestOp::TGet};
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
    PTO2SdmaDescriptor<GlobalDstData, GlobalSrcData, ScratchTile>& desc)
{
    pto::comm::AsyncSession session = pto2_sq_open(
        sq_type, sq_id, desc.scratch, desc.context,
        desc.sync_id, desc.base_config);
    if (!session.valid) return 0;

    pto::comm::AsyncEvent event;
    if (desc.op == PTO2SdmaRequestOp::TGet) {
        event = pto::comm::TGET_ASYNC<pto::comm::DmaEngine::SDMA>(
            desc.dst, desc.src, session);
    } else {
        event = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::SDMA>(
            desc.dst, desc.src, session);
    }
    return event.valid() ? event.handle : 0;
}

#endif  // PTO_SQ_KERNEL_API_H
