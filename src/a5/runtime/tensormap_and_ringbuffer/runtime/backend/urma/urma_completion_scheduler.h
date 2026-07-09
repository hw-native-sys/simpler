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

#ifndef SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_BACKEND_URMA_URMA_COMPLETION_SCHEDULER_H_
#define SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_BACKEND_URMA_URMA_COMPLETION_SCHEDULER_H_

#include <cstddef>
#include <cstdint>

#include "aicpu/platform_regs.h"
#include "aicore_completion_mailbox.h"
#include "pto_completion_token.h"
#include "pto_runtime_status.h"
#include "backend/urma/urma_completion_abi.h"

namespace pto2::urma_backend {

inline uintptr_t cache_line(const volatile void *addr) {
    return reinterpret_cast<uintptr_t>(addr) & ~(uintptr_t(PTO2_ALIGN_SIZE) - 1u);
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
    __asm__ __volatile__("dsb sy" ::: "memory");
}

inline uint32_t load_cqe_dw0(uint64_t cqe_addr) {
    auto *ptr = reinterpret_cast<volatile uint32_t *>(static_cast<uintptr_t>(cqe_addr));
    cache_invalidate_range(reinterpret_cast<const void *>(cache_line(ptr)), PTO2_ALIGN_SIZE);
    return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
}

inline void update_tail_info(const UrmaCqCtx &cq_ctx, const UrmaWqCtx &wq_ctx, uint32_t cur_tail) {
    if (cq_ctx.tail_addr != 0) {
        store_device_u32(cq_ctx.tail_addr, cur_tail);
    }
    if (cq_ctx.db_addr != 0) {
        store_device_u32(cq_ctx.db_addr, cur_tail & 0xFFFFFFu);
    }
    if (wq_ctx.tail_addr != 0) {
        store_device_u32(wq_ctx.tail_addr, cur_tail);
    }
}

inline CompletionPollResult poll_urma_event_handle(uint64_t event_handle, uint64_t workspace_addr) {
    if (event_handle == 0) {
        return {CompletionPollState::READY, PTO2_ERROR_NONE};
    }
    if (workspace_addr == 0) {
        return {CompletionPollState::FAILED, PTO2_ERROR_ASYNC_COMPLETION_INVALID};
    }

    uint32_t remote_rank = 0;
    uint32_t target_head = 0;
    decode_urma_event_handle(event_handle, remote_rank, target_head);
    if (target_head == 0) {
        return {CompletionPollState::READY, PTO2_ERROR_NONE};
    }

    auto *info = reinterpret_cast<volatile UrmaInfo *>(static_cast<uintptr_t>(workspace_addr));
    cache_invalidate_range(reinterpret_cast<const void *>(cache_line(info)), PTO2_ALIGN_SIZE);
    const uint32_t qp_num = __atomic_load_n(&info->qp_num, __ATOMIC_ACQUIRE);
    const uint32_t rank_count = __atomic_load_n(&info->rank_count, __ATOMIC_ACQUIRE);
    const uint64_t sq_ptr = __atomic_load_n(&info->sq_ptr, __ATOMIC_ACQUIRE);
    const uint64_t scq_ptr = __atomic_load_n(&info->scq_ptr, __ATOMIC_ACQUIRE);
    if (qp_num == 0 || remote_rank >= rank_count || sq_ptr == 0 || scq_ptr == 0) {
        return {CompletionPollState::FAILED, PTO2_ERROR_ASYNC_COMPLETION_INVALID};
    }

    const uint64_t ctx_index = static_cast<uint64_t>(remote_rank) * qp_num;
    auto *cq_entry = reinterpret_cast<volatile UrmaCqCtx *>(
        static_cast<uintptr_t>(scq_ptr + ctx_index * static_cast<uint64_t>(sizeof(UrmaCqCtx)))
    );
    auto *wq_entry = reinterpret_cast<volatile UrmaWqCtx *>(
        static_cast<uintptr_t>(sq_ptr + ctx_index * static_cast<uint64_t>(sizeof(UrmaWqCtx)))
    );
    cache_invalidate_range(reinterpret_cast<const void *>(cache_line(cq_entry)), PTO2_ALIGN_SIZE);
    cache_invalidate_range(reinterpret_cast<const void *>(cache_line(wq_entry)), PTO2_ALIGN_SIZE);

    UrmaCqCtx cq_ctx{};
    cq_ctx.buf_addr = __atomic_load_n(&cq_entry->buf_addr, __ATOMIC_ACQUIRE);
    cq_ctx.cqe_shift_size = __atomic_load_n(&cq_entry->cqe_shift_size, __ATOMIC_ACQUIRE);
    cq_ctx.depth = __atomic_load_n(&cq_entry->depth, __ATOMIC_ACQUIRE);
    cq_ctx.tail_addr = __atomic_load_n(&cq_entry->tail_addr, __ATOMIC_ACQUIRE);
    cq_ctx.db_addr = __atomic_load_n(&cq_entry->db_addr, __ATOMIC_ACQUIRE);

    UrmaWqCtx wq_ctx{};
    wq_ctx.tail_addr = __atomic_load_n(&wq_entry->tail_addr, __ATOMIC_ACQUIRE);
    if (cq_ctx.buf_addr == 0 || cq_ctx.tail_addr == 0 || cq_ctx.cqe_shift_size > kMaxCqeShift ||
        !is_power_of_two(cq_ctx.depth)) {
        return {CompletionPollState::FAILED, PTO2_ERROR_ASYNC_COMPLETION_INVALID};
    }

    const uint32_t cqe_size = 1u << cq_ctx.cqe_shift_size;
    uint32_t cur_tail = load_device_u32(cq_ctx.tail_addr);
    if (has_reached(cur_tail, target_head)) {
        return {CompletionPollState::READY, PTO2_ERROR_NONE};
    }

    uint32_t next_tail = cur_tail;
    while (!has_reached(next_tail, target_head)) {
        const uint64_t cqe_addr =
            cq_ctx.buf_addr + static_cast<uint64_t>(cqe_size) * static_cast<uint64_t>(next_tail & (cq_ctx.depth - 1));
        const uint32_t dw0 = load_cqe_dw0(cqe_addr);
        const bool valid_owner = ((next_tail / cq_ctx.depth) & 1u) != 0;
        const bool owner = ((dw0 >> 2) & 1u) != 0;
        if ((valid_owner ^ owner) == 0) {
            break;
        }

        const uint8_t substatus = static_cast<uint8_t>((dw0 >> 16) & 0xFFu);
        const uint8_t status = static_cast<uint8_t>((dw0 >> 24) & 0xFFu);
        if (status != 0 || substatus != 0) {
            return {CompletionPollState::FAILED, PTO2_ERROR_ASYNC_COMPLETION_INVALID};
        }
        next_tail++;
    }

    if (next_tail != cur_tail) {
        update_tail_info(cq_ctx, wq_ctx, next_tail);
    }
    return {
        has_reached(next_tail, target_head) ? CompletionPollState::READY : CompletionPollState::PENDING, PTO2_ERROR_NONE
    };
}

inline void retire_urma_event_handle(uint64_t /*event_handle*/, uint64_t /*workspace_addr*/) {}

}  // namespace pto2::urma_backend

#endif  // SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_BACKEND_URMA_URMA_COMPLETION_SCHEDULER_H_
