/**
 * MOE RecvAssemble Kernel — cumsum + assemble expandX (func_id=2)
 *
 * Depends on NotifyWait completing (via dummy tensor input),
 * guaranteeing notify_counter >= NUM_RANKS-1 (7 peers done).
 *
 * Reads local_counts + per-source-rank recv_counts, computes cumulative
 * sums for assembly offsets, copies token data from shmem_data slots
 * into expandX, and writes expert_token_nums.
 *
 * Slot ordering: for each expert_offset, enumerate source ranks:
 *   slot = expert_offset * NUM_RANKS + src_rank
 *   count = local_counts[expert_offset]          if src_rank == my_rank
 *         = recv_counts[src_rank * COUNT_PAD + expert_offset]  otherwise
 *
 * Kernel args layout:
 *   args[0] = &Tensor(dummy_notify)      — input (dependency token from NotifyWait)
 *   args[1] = &Tensor(local_counts)      — input [COUNT_PAD] int32
 *   args[2] = &Tensor(expand_x)          — output [EXPAND_X_ROWS * HIDDEN_DIM] float
 *   args[3] = &Tensor(expert_token_nums) — output [EXPERTS_PER_RANK] int32
 *   args[4] = shmem_data_addr            — scalar (GM float* base)
 *   args[5] = recv_counts_addr           — scalar (GM int32*, [NUM_RANKS * COUNT_PAD])
 *   args[6] = CommDeviceContext*          — scalar
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

extern "C" __aicore__ __attribute__((always_inline))
void kernel_entry(__gm__ int64_t* args) {
    // args[0] = dummy_notify tensor (dependency token, unused)
    __gm__ Tensor* local_cnt_t = reinterpret_cast<__gm__ Tensor*>(args[1]);
    __gm__ Tensor* expand_x_t  = reinterpret_cast<__gm__ Tensor*>(args[2]);
    __gm__ Tensor* etn_t       = reinterpret_cast<__gm__ Tensor*>(args[3]);

    __gm__ float* shmem_data =
        reinterpret_cast<__gm__ float*>(static_cast<uintptr_t>(args[4]));
    __gm__ int32_t* recv_counts =
        reinterpret_cast<__gm__ int32_t*>(static_cast<uintptr_t>(args[5]));
    __gm__ CommDeviceContext* comm_ctx =
        reinterpret_cast<__gm__ CommDeviceContext*>(static_cast<uintptr_t>(args[6]));

    __gm__ int32_t* local_counts =
        reinterpret_cast<__gm__ int32_t*>(local_cnt_t->buffer.addr) + local_cnt_t->start_offset;
    __gm__ float* expand_x =
        reinterpret_cast<__gm__ float*>(expand_x_t->buffer.addr) + expand_x_t->start_offset;
    __gm__ int32_t* expert_token_nums =
        reinterpret_cast<__gm__ int32_t*>(etn_t->buffer.addr) + etn_t->start_offset;

    int my_rank = static_cast<int>(comm_ctx->rankId);

    pipe_barrier(PIPE_ALL);

    int slot_counts[NUM_EXPERT_SLOTS];
    for (int exp_off = 0; exp_off < EXPERTS_PER_RANK; exp_off++) {
        for (int src_rank = 0; src_rank < NUM_RANKS; src_rank++) {
            int slot = exp_off * NUM_RANKS + src_rank;
            if (src_rank == my_rank) {
                slot_counts[slot] = static_cast<int>(local_counts[exp_off]);
            } else {
                slot_counts[slot] = static_cast<int>(
                    recv_counts[src_rank * COUNT_PAD + exp_off]);
            }
        }
    }

    int cumsum[NUM_EXPERT_SLOTS + 1];
    cumsum[0] = 0;
    for (int s = 0; s < NUM_EXPERT_SLOTS; s++) {
        cumsum[s + 1] = cumsum[s] + slot_counts[s];
    }

    for (int exp_off = 0; exp_off < EXPERTS_PER_RANK; exp_off++) {
        int total = 0;
        for (int src_rank = 0; src_rank < NUM_RANKS; src_rank++) {
            int slot = exp_off * NUM_RANKS + src_rank;
            total += slot_counts[slot];
        }
        expert_token_nums[exp_off] = static_cast<int32_t>(total);
    }
    pipe_barrier(PIPE_ALL);

    for (int s = 0; s < NUM_EXPERT_SLOTS; s++) {
        int count = slot_counts[s];
        int out_offset = cumsum[s];
        for (int t = 0; t < count; t++) {
            __gm__ float* src_ptr = shmem_data + (s * NUM_TOKENS + t) * HIDDEN_DIM;
            __gm__ float* dst_ptr = expand_x + (out_offset + t) * HIDDEN_DIM;
            for (int j = 0; j < HIDDEN_DIM; j++) {
                dst_ptr[j] = src_ptr[j];
            }
        }
    }

    pipe_barrier(PIPE_ALL);
}
