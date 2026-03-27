/**
 * MOE Send Kernel — TPUT_ASYNC data + counts + TNOTIFY to all peers (func_id=1)
 *
 * All SDMA operations share the same channel (channelGroupIdx = block_idx).
 * The SDMA engine processes SQEs strictly in order within a channel, so each
 * TPUT_ASYNC appends [data SQEs][flag SQE] to the tail.  When the last flag
 * is set, all previous transfers are guaranteed complete.
 *
 * Therefore we only register ONE CQ entry — for the very last TPUT_ASYNC.
 * The AICPU scheduler polls that single flag; once it flips, the entire
 * batch (14 data + 7 count transfers) is done.
 *
 * Steps:
 *   1. 14 × TPUT_ASYNC — per-(peer, expert) token data → peer shmem_data slots
 *   2.  7 × TPUT_ASYNC — per-peer count block → peer recv_counts
 *   3. Register CQ entry for the LAST TPUT_ASYNC only (1 entry total)
 *   4.  7 × TNOTIFY    — AtomicAdd(1) → peer notify_counter
 *
 * Kernel args layout:
 *   args[0] = &Tensor(send_staging)  — input [STAGING_ELEMS] float
 *   args[1] = shmem_data_addr        — scalar
 *   args[2] = send_counts_addr       — scalar (GM int32*, [NUM_RANKS * COUNT_PAD])
 *   args[3] = recv_counts_addr       — scalar (local addr → CommRemotePtr)
 *   args[4] = notify_counter_addr    — scalar (local addr → CommRemotePtr)
 *   args[5] = CommDeviceContext*      — scalar
 *   args[6] = sdma_context           — scalar
 *   args[7] = cq_addr                — scalar (auto-appended by deferred submit)
 */

#include <cstdint>

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include <pto/pto-inst.hpp>
#include "pto/comm/pto_comm_inst.hpp"
#include "pto/npu/comm/async/sdma/sdma_types.hpp"
#include "pto/common/pto_tile.hpp"

#include "common/comm_context.h"
#include "tensor.h"

using namespace pto;

#include "pto_sq_kernel_api.h"
#include "pto_notify_kernel_api.h"

static constexpr int NUM_TOKENS = 16;
static constexpr int HIDDEN_DIM = 128;
static constexpr int NUM_RANKS = 8;
static constexpr int EXPERTS_PER_RANK = 2;
static constexpr int COUNT_PAD = 32;
static constexpr int SLOT_ELEMS = NUM_TOKENS * HIDDEN_DIM;

template <typename T>
AICORE inline __gm__ T* CommRemotePtr(
    __gm__ CommDeviceContext* ctx, __gm__ T* local_ptr, int peer_rank) {
    uint64_t local_base = ctx->windowsIn[ctx->rankId];
    uint64_t offset = reinterpret_cast<uint64_t>(local_ptr) - local_base;
    return reinterpret_cast<__gm__ T*>(ctx->windowsIn[peer_rank] + offset);
}

extern "C" __aicore__ __attribute__((always_inline))
void kernel_entry(__gm__ int64_t* args) {
    __gm__ Tensor* send_stg_t = reinterpret_cast<__gm__ Tensor*>(args[0]);

    __gm__ float* shmem_data =
        reinterpret_cast<__gm__ float*>(static_cast<uintptr_t>(args[1]));
    __gm__ int32_t* send_counts =
        reinterpret_cast<__gm__ int32_t*>(static_cast<uintptr_t>(args[2]));
    __gm__ int32_t* local_recv_counts =
        reinterpret_cast<__gm__ int32_t*>(static_cast<uintptr_t>(args[3]));
    __gm__ int32_t* local_notify_counter =
        reinterpret_cast<__gm__ int32_t*>(static_cast<uintptr_t>(args[4]));
    __gm__ CommDeviceContext* comm_ctx =
        reinterpret_cast<__gm__ CommDeviceContext*>(static_cast<uintptr_t>(args[5]));
    uint64_t sdma_context = static_cast<uint64_t>(args[6]);
    uint64_t cq_addr      = static_cast<uint64_t>(args[7]);

    __gm__ float* send_staging =
        reinterpret_cast<__gm__ float*>(send_stg_t->buffer.addr) + send_stg_t->start_offset;

    int my_rank = static_cast<int>(comm_ctx->rankId);

    volatile __gm__ PTO2CompletionQueue* cq = pto2_cq_get(cq_addr);
    pto2_cq_reset(cq);

    using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1,
                                  pto::comm::sdma::UB_ALIGN_SIZE>;
    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x20000);

    __gm__ uint8_t* context =
        reinterpret_cast<__gm__ uint8_t*>(static_cast<uintptr_t>(sdma_context));

    uint64_t last_tag = 0;

    // --- Step 1: 14 × TPUT_ASYNC token data to peer shmem_data slots ---
    {
        using SlotShape  = Shape<1, 1, 1, 1, SLOT_ELEMS>;
        using SlotStride = Stride<SLOT_ELEMS, SLOT_ELEMS, SLOT_ELEMS, SLOT_ELEMS, 1>;
        using SlotGlobal = GlobalTensor<float, SlotShape, SlotStride>;

        for (int peer = 0; peer < NUM_RANKS; peer++) {
            if (peer == my_rank) continue;
            for (int exp_off = 0; exp_off < EXPERTS_PER_RANK; exp_off++) {
                int staging_idx = peer * EXPERTS_PER_RANK + exp_off;
                __gm__ float* local_src = send_staging + staging_idx * SLOT_ELEMS;

                int slot = exp_off * NUM_RANKS + my_rank;
                __gm__ float* peer_dst = CommRemotePtr(
                    comm_ctx, shmem_data + slot * SLOT_ELEMS, peer);

                SlotGlobal dstGlobal(peer_dst);
                SlotGlobal srcGlobal(local_src);

                auto desc = pto2_sdma_descriptor(
                    dstGlobal, srcGlobal, scratchTile, context);
                last_tag = pto2_send_request_entry(
                    PTO2_ENGINE_SDMA, PTO2_SQ_ID_AUTO, desc);
                pipe_barrier(PIPE_ALL);
            }
        }
    }

    // --- Step 2: 7 × TPUT_ASYNC per-peer count blocks ---
    {
        using CountShape  = Shape<1, 1, 1, 1, COUNT_PAD>;
        using CountStride = Stride<COUNT_PAD, COUNT_PAD, COUNT_PAD, COUNT_PAD, 1>;
        using CountGlobal = GlobalTensor<int32_t, CountShape, CountStride>;

        for (int peer = 0; peer < NUM_RANKS; peer++) {
            if (peer == my_rank) continue;

            __gm__ int32_t* peer_recv = CommRemotePtr(
                comm_ctx, local_recv_counts + my_rank * COUNT_PAD, peer);

            CountGlobal dstGlobal(peer_recv);
            CountGlobal srcGlobal(send_counts + peer * COUNT_PAD);

            auto desc = pto2_sdma_descriptor(
                dstGlobal, srcGlobal, scratchTile, context);
            last_tag = pto2_send_request_entry(
                PTO2_ENGINE_SDMA, PTO2_SQ_ID_AUTO, desc);
            pipe_barrier(PIPE_ALL);
        }
    }

    // --- Step 3: Register only the LAST flag (1 CQ entry) ---
    // SDMA channel ordering: when this flag is set, all 21 preceding
    // transfers (14 data + 7 count) are guaranteed complete.
    if (last_tag != 0) {
        pto2_save_expected_completion(PTO2_ENGINE_SDMA, cq, last_tag);
    }

    // --- Step 4: 7 × TNOTIFY to peer notify_counter ---
    for (int peer = 0; peer < NUM_RANKS; peer++) {
        if (peer == my_rank) continue;
        __gm__ int32_t* peer_counter = CommRemotePtr(
            comm_ctx, local_notify_counter, peer);
        pto2_send_notification(peer_counter, 1, PTO2NotifyOp::AtomicAdd);
    }

    pto2_cq_flush(cq);
}
