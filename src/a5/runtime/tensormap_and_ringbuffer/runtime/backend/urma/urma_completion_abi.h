/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#ifndef SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_BACKEND_URMA_URMA_COMPLETION_ABI_H_
#define SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_BACKEND_URMA_URMA_COMPLETION_ABI_H_

#include <stddef.h>
#include <stdint.h>

namespace pto2::urma_backend {

inline constexpr uint32_t kHandleRankIdShift = 32;
inline constexpr uint32_t kCqeBytes = 64;
inline constexpr uint32_t kMaxCqeShift = 12;
inline constexpr uint32_t kFakeUrmaWorkspaceMagic = 0x55524D41u;  // "URMA"
inline constexpr uint32_t kFakeUrmaWorkspaceInitializing = 0x55524D30u;
inline constexpr uint32_t kFakeUrmaMaxRanks = 16;
inline constexpr uint32_t kFakeUrmaCqDepth = 64;
inline constexpr uint32_t kFakeUrmaCqeShift = 6;

enum class UrmaDbMode : int32_t {
    INVALID_DB = -1,
    HW_DB = 0,
    SW_DB = 1,
};

struct UrmaInfo {
    uint32_t qp_num;
    uint32_t local_token_id;
    uint32_t rank_count;
    uint64_t sq_ptr;
    uint64_t rq_ptr;
    uint64_t scq_ptr;
    uint64_t rcq_ptr;
    uint64_t mem_ptr;
};

struct UrmaWqCtx {
    uint32_t wqn;
    uint64_t buf_addr;
    uint32_t wqe_shift_size;
    uint32_t depth;
    uint64_t head_addr;
    uint64_t tail_addr;
    UrmaDbMode db_mode;
    uint64_t db_addr;
    uint32_t sl;
};

struct UrmaCqCtx {
    uint32_t cqn;
    uint64_t buf_addr;
    uint32_t cqe_shift_size;
    uint32_t depth;
    uint64_t head_addr;
    uint64_t tail_addr;
    UrmaDbMode db_mode;
    uint64_t db_addr;
};

struct alignas(kCqeBytes) FakeUrmaCqe {
    uint32_t dw[16];
};

struct alignas(kCqeBytes) FakeUrmaWorkspace {
    UrmaInfo info;
    UrmaWqCtx sq[kFakeUrmaMaxRanks];
    UrmaCqCtx scq[kFakeUrmaMaxRanks];
    uint32_t sq_head[kFakeUrmaMaxRanks];
    uint32_t sq_tail[kFakeUrmaMaxRanks];
    uint32_t cq_tail[kFakeUrmaMaxRanks];
    uint32_t cq_doorbell[kFakeUrmaMaxRanks];
    uint32_t sq_submit_lock[kFakeUrmaMaxRanks];
    FakeUrmaCqe scq_entries[kFakeUrmaMaxRanks][kFakeUrmaCqDepth];
    uint32_t magic;
};

static_assert(sizeof(UrmaInfo) == 56, "URMA info ABI drift");
static_assert(offsetof(UrmaInfo, sq_ptr) == 16, "URMA info ABI drift");
static_assert(sizeof(UrmaWqCtx) == 64, "URMA WQ context ABI drift");
static_assert(offsetof(UrmaWqCtx, db_addr) == 48, "URMA WQ context ABI drift");
static_assert(sizeof(UrmaCqCtx) == 56, "URMA CQ context ABI drift");
static_assert(offsetof(UrmaCqCtx, db_addr) == 48, "URMA CQ context ABI drift");
static_assert(sizeof(FakeUrmaCqe) == kCqeBytes, "fake URMA CQE must match scheduler CQE size");

inline uint64_t encode_urma_event_handle(uint32_t remote_rank, uint32_t target_head) {
    return (static_cast<uint64_t>(remote_rank) << kHandleRankIdShift) | static_cast<uint64_t>(target_head);
}

inline void decode_urma_event_handle(uint64_t handle, uint32_t &remote_rank, uint32_t &target_head) {
    remote_rank = static_cast<uint32_t>(handle >> kHandleRankIdShift);
    target_head = static_cast<uint32_t>(handle & 0xFFFFFFFFu);
}

inline bool fake_cqe_ready_owner(uint32_t cqe_seq) { return ((cqe_seq / kFakeUrmaCqDepth) & 1u) == 0; }

inline uint32_t encode_fake_cqe_dw0(uint32_t cqe_seq, uint8_t status, uint8_t substatus) {
    const uint32_t owner = fake_cqe_ready_owner(cqe_seq) ? 1u : 0u;
    return (owner << 2) | (static_cast<uint32_t>(substatus) << 16) | (static_cast<uint32_t>(status) << 24);
}

inline uint32_t encode_fake_pending_cqe_dw0(uint32_t cqe_seq) {
    const uint32_t owner = fake_cqe_ready_owner(cqe_seq) ? 0u : 1u;
    return owner << 2;
}

}  // namespace pto2::urma_backend

#endif  // SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_BACKEND_URMA_URMA_COMPLETION_ABI_H_
