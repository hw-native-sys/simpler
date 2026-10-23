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

#pragma once

#include <cstddef>
#include <cstdint>

#include "aicpu/platform_regs.h"
#include "common/urma_workspace_abi.h"
#include "completion_token.h"
#include "runtime_status.h"

namespace pto2::urma_backend {

inline constexpr uint32_t kHandleRankIdShift = 32;
inline constexpr uint32_t kCqeBytes = 64;

inline uint64_t encode_urma_event_handle(uint32_t remote_rank, uint32_t target_head) {
    return (static_cast<uint64_t>(remote_rank) << kHandleRankIdShift) | static_cast<uint64_t>(target_head);
}

inline void decode_urma_event_handle(uint64_t handle, uint32_t &remote_rank, uint32_t &target_head) {
    remote_rank = static_cast<uint32_t>(handle >> kHandleRankIdShift);
    target_head = static_cast<uint32_t>(handle & 0xFFFFFFFFu);
}

inline uintptr_t cache_line(const volatile void *addr) {
    return reinterpret_cast<uintptr_t>(addr) & ~(uintptr_t(CHIP_ALIGN_SIZE) - 1u);
}

inline void invalidate_object(const volatile void *addr, std::size_t size) {
    const uintptr_t object_addr = reinterpret_cast<uintptr_t>(addr);
    const uintptr_t begin = cache_line(addr);
    const uintptr_t end = (object_addr + size + CHIP_ALIGN_SIZE - 1u) & ~(uintptr_t(CHIP_ALIGN_SIZE) - 1u);
    cache_invalidate_range(reinterpret_cast<const void *>(begin), end - begin);
}

inline bool has_reached(uint32_t current, uint32_t target) { return static_cast<int32_t>(current - target) >= 0; }

inline bool is_power_of_two(uint32_t value) { return value != 0 && (value & (value - 1u)) == 0; }

inline uint32_t load_device_u32(uint64_t addr) {
    auto *ptr = reinterpret_cast<volatile uint32_t *>(static_cast<uintptr_t>(addr));
    return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
}

inline void store_device_u32(uint64_t addr, uint32_t value) {
    auto *ptr = reinterpret_cast<volatile uint32_t *>(static_cast<uintptr_t>(addr));
    __atomic_store_n(ptr, value, __ATOMIC_RELEASE);
    cache_flush_range(reinterpret_cast<const void *>(cache_line(ptr)), CHIP_ALIGN_SIZE);
#if defined(__aarch64__)
    __asm__ __volatile__("dsb sy" ::: "memory");
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

inline uint32_t load_cqe_dw0(uint64_t cqe_addr) {
    auto *ptr = reinterpret_cast<volatile uint32_t *>(static_cast<uintptr_t>(cqe_addr));
    cache_invalidate_range(reinterpret_cast<const void *>(cache_line(ptr)), CHIP_ALIGN_SIZE);
    return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
}

// A single scheduler dispatch thread owns CQ consumption. AIV submits only
// when both rings have room and never drains them on the supported path.
inline CompletionPollResult
poll_urma_event_handle(uint64_t event_handle, uint64_t workspace_addr, uint32_t target_cqe) {
    const auto invalid = CompletionPollResult{CompletionPollState::FAILED, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID};
    if (workspace_addr == 0) return invalid;
    uint32_t peer = 0, target_bb = 0;
    decode_urma_event_handle(event_handle, peer, target_bb);
    auto *info = reinterpret_cast<volatile UrmaInfo *>(workspace_addr);
    invalidate_object(info, sizeof(*info));
    if (info->layout != 0 || info->qp_num != 1 || info->jetties_per_core != 1 ||
        info->jetty_count != info->rank_count || peer >= info->rank_count || !info->sq_ptr || !info->scq_ptr) {
        return invalid;
    }
    auto *cq = reinterpret_cast<volatile UrmaCqCtx *>(info->scq_ptr) + peer;
    auto *wq = reinterpret_cast<volatile UrmaWqCtx *>(info->sq_ptr) + peer;
    invalidate_object(cq, sizeof(*cq));
    invalidate_object(wq, sizeof(*wq));
    if (!cq->buf_addr || !cq->tail_addr || !cq->db_addr || !wq->tail_addr || cq->cqe_shift_size != 6 ||
        !is_power_of_two(cq->depth) || !is_power_of_two(wq->depth) || cq->depth >= (1u << 31) ||
        wq->depth >= (1u << 31))
        return invalid;
    invalidate_object(reinterpret_cast<void *>(cq->tail_addr), sizeof(uint32_t));
    invalidate_object(reinterpret_cast<void *>(wq->tail_addr), sizeof(uint32_t));
    const uint32_t current_cqe = load_device_u32(cq->tail_addr);
    const uint32_t current_bb = load_device_u32(wq->tail_addr);
    if (has_reached(current_cqe, target_cqe)) {
        return {
            has_reached(current_bb, target_bb) ? CompletionPollState::READY : CompletionPollState::PENDING,
            SIMPLER_ERROR_NONE
        };
    }
    if (target_cqe - current_cqe > cq->depth || target_bb - current_bb > wq->depth) return invalid;
    for (uint32_t seq = current_cqe; seq != target_cqe; ++seq) {
        const uint64_t addr = cq->buf_addr + kCqeBytes * static_cast<uint64_t>(seq & (cq->depth - 1));
        const uint32_t dw0 = load_cqe_dw0(addr);
        const bool owner = (dw0 >> 2) & 1u;
        if (owner == (((seq / cq->depth) & 1u) != 0)) {
            return {CompletionPollState::PENDING, SIMPLER_ERROR_NONE};
        }
        // Failed prefixes remain unretired, including when a later event is polled.
        if ((dw0 >> 16) != 0) return invalid;
    }
    // Completion publication follows acquisition of every CQE in the prefix.
    store_device_u32(cq->tail_addr, target_cqe);
    store_device_u32(cq->db_addr, target_cqe & 0xFFFFFFu);
    store_device_u32(wq->tail_addr, target_bb);
    return {CompletionPollState::READY, SIMPLER_ERROR_NONE};
}

inline void retire_urma_event_handle(uint64_t /*event_handle*/, uint64_t /*workspace_addr*/) {}

}  // namespace pto2::urma_backend
