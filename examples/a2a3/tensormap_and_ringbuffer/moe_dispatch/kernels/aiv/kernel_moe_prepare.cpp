/**
 * MOE Prepare Kernel — route tokens and pack per-rank staging buffers (func_id=0)
 *
 * For each token, determine the target expert/rank, then:
 *   - Local expert  -> shmem_data[slot] (in-place, no SDMA needed)
 *   - Remote expert -> send_staging[target_rank][expert_offset]
 *
 * Computes per-expert local_counts and per-(target_rank, expert_offset) send_counts.
 *
 * Kernel args layout:
 *   args[0] = &Tensor(tokens)       — input  [NUM_TOKENS * HIDDEN_DIM] float
 *   args[1] = &Tensor(expert_ids)   — input  [NUM_TOKENS] int32
 *   args[2] = &Tensor(send_staging) — output [NUM_RANKS * EXPERTS_PER_RANK * NUM_TOKENS * HIDDEN_DIM] float
 *   args[3] = &Tensor(local_counts) — output [COUNT_PAD] int32
 *   args[4] = shmem_data_addr       — scalar (GM float* base)
 *   args[5] = send_counts_addr      — scalar (GM int32* base, [NUM_RANKS * COUNT_PAD])
 *   args[6] = CommDeviceContext*     — scalar
 */

#include <cstdint>

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include "common/comm_context.h"
#include "tensor.h"

static constexpr int NUM_TOKENS = 16;
static constexpr int HIDDEN_DIM = 128;
static constexpr int NUM_RANKS = 8;
static constexpr int EXPERTS_PER_RANK = 2;
static constexpr int NUM_EXPERT_SLOTS = EXPERTS_PER_RANK * NUM_RANKS;
static constexpr int COUNT_PAD = 32;
static constexpr int SLOT_ELEMS = NUM_TOKENS * HIDDEN_DIM;

extern "C" __aicore__ __attribute__((always_inline))
void kernel_entry(__gm__ int64_t* args) {
    __gm__ Tensor* tokens_t      = reinterpret_cast<__gm__ Tensor*>(args[0]);
    __gm__ Tensor* expert_ids_t  = reinterpret_cast<__gm__ Tensor*>(args[1]);
    __gm__ Tensor* send_stg_t    = reinterpret_cast<__gm__ Tensor*>(args[2]);
    __gm__ Tensor* local_cnt_t   = reinterpret_cast<__gm__ Tensor*>(args[3]);

    __gm__ float*   shmem_data =
        reinterpret_cast<__gm__ float*>(static_cast<uintptr_t>(args[4]));
    __gm__ int32_t* send_counts =
        reinterpret_cast<__gm__ int32_t*>(static_cast<uintptr_t>(args[5]));
    __gm__ CommDeviceContext* comm_ctx =
        reinterpret_cast<__gm__ CommDeviceContext*>(static_cast<uintptr_t>(args[6]));

    __gm__ float*   tokens =
        reinterpret_cast<__gm__ float*>(tokens_t->buffer.addr) + tokens_t->start_offset;
    __gm__ int32_t* expert_ids =
        reinterpret_cast<__gm__ int32_t*>(expert_ids_t->buffer.addr) + expert_ids_t->start_offset;
    __gm__ float*   send_staging =
        reinterpret_cast<__gm__ float*>(send_stg_t->buffer.addr) + send_stg_t->start_offset;
    __gm__ int32_t* local_counts =
        reinterpret_cast<__gm__ int32_t*>(local_cnt_t->buffer.addr) + local_cnt_t->start_offset;

    int my_rank = static_cast<int>(comm_ctx->rankId);

    pipe_barrier(PIPE_ALL);

    int l_counts[EXPERTS_PER_RANK] = {};
    int s_counts[NUM_RANKS * EXPERTS_PER_RANK] = {};

    for (int i = 0; i < NUM_TOKENS; i++) {
        int eid = static_cast<int>(expert_ids[i]);
        int target_rank = eid / EXPERTS_PER_RANK;
        int expert_offset = eid % EXPERTS_PER_RANK;

        __gm__ float* src_ptr = tokens + i * HIDDEN_DIM;

        if (target_rank == my_rank) {
            int slot = expert_offset * NUM_RANKS + my_rank;
            int idx = l_counts[expert_offset];
            __gm__ float* dst_ptr = shmem_data +
                (slot * NUM_TOKENS + idx) * HIDDEN_DIM;
            for (int j = 0; j < HIDDEN_DIM; j++) {
                dst_ptr[j] = src_ptr[j];
            }
            l_counts[expert_offset]++;
        } else {
            int staging_idx = target_rank * EXPERTS_PER_RANK + expert_offset;
            int idx = s_counts[staging_idx];
            __gm__ float* dst_ptr = send_staging +
                (staging_idx * NUM_TOKENS + idx) * HIDDEN_DIM;
            for (int j = 0; j < HIDDEN_DIM; j++) {
                dst_ptr[j] = src_ptr[j];
            }
            s_counts[staging_idx]++;
        }
    }

    pipe_barrier(PIPE_ALL);

    for (int k = 0; k < EXPERTS_PER_RANK; k++) {
        local_counts[k] = static_cast<int32_t>(l_counts[k]);
    }
    for (int r = 0; r < NUM_RANKS; r++) {
        for (int e = 0; e < EXPERTS_PER_RANK; e++) {
            send_counts[r * COUNT_PAD + e] =
                static_cast<int32_t>(s_counts[r * EXPERTS_PER_RANK + e]);
        }
    }

    pipe_barrier(PIPE_ALL);
}
