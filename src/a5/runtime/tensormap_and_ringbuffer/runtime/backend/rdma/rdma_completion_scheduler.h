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
#include "aicore_completion_mailbox.h"
#include "common/unified_log.h"
#include "completion_token.h"
#include "runtime_status.h"

namespace pto2::rdma_backend {

inline constexpr uint32_t kHandleRankIdShift = 32;
inline constexpr uint32_t kCqeBytes = 64;
inline constexpr uint32_t kCqUpdateCiMask = 0xffffffu;
inline constexpr uint32_t kRdmaWorkspaceMagic = 0x52444d41u;
inline constexpr uint32_t kRdmaWorkspaceVersion = 1u;
inline constexpr uint32_t kRdmaBackendHns1825 = 1u;
inline constexpr uint32_t kRdmaErrorRank = 0xffffffffu;

enum class RdmaDbMode : int32_t {
    INVALID_DB = -1,
    HW_DB = 0,
    SW_DB = 1,
};

struct RdmaInfo {
    uint32_t magic;
    uint32_t version;
    uint32_t backend;
    uint32_t qp_num;
    uint32_t rank_count;
    uint32_t reserved;
    uint64_t sq_ptr;
    uint64_t rq_ptr;
    uint64_t scq_ptr;
    uint64_t rcq_ptr;
    uint64_t mem_ptr;
};

struct RdmaWqCtx {
    uint32_t wqn;
    uint64_t buf_addr;
    uint32_t wqe_size;
    uint32_t depth;
    uint64_t head_addr;
    uint64_t tail_addr;
    RdmaDbMode db_mode;
    uint64_t db_addr;
    uint32_t sl;
    uint64_t amo_addr;
    uint32_t amo_lkey;
    uint64_t db_sw_addr;
    uint8_t mtu_shift;
    uint8_t db_cos;
    uint8_t reserved[6];
};

struct RdmaCqCtx {
    uint32_t cqn;
    uint64_t buf_addr;
    uint32_t cqe_size;
    uint32_t depth;
    uint64_t head_addr;
    uint64_t tail_addr;
    RdmaDbMode db_mode;
    uint64_t db_addr;
    uint64_t db_sw_addr;
};

struct Hns1825Cqe {
    uint32_t owner_id_qpn;
    uint32_t op_sr_wqebb;
    uint32_t byte_cnt;
    uint32_t imm_data;
    uint32_t rsvd_dw5;
    uint32_t wqe_num;
    uint32_t vlan_queue_index;
    uint8_t syndrome;
    uint8_t rsvd;
    uint16_t wqe_counter;
};

struct RdmaMemInfo {
    uint64_t size;
    uint64_t addr;
    uint32_t lkey;
    uint32_t rkey;
};

static_assert(sizeof(RdmaInfo) == 64, "RDMA info ABI drift");
static_assert(offsetof(RdmaInfo, sq_ptr) == 24, "RDMA info ABI drift");
static_assert(sizeof(RdmaWqCtx) == 96, "RDMA WQ context ABI drift");
static_assert(offsetof(RdmaWqCtx, db_addr) == 48, "RDMA WQ context ABI drift");
static_assert(offsetof(RdmaWqCtx, db_sw_addr) == 80, "RDMA WQ context ABI drift");
static_assert(offsetof(RdmaWqCtx, mtu_shift) == 88, "RDMA WQ context ABI drift");
static_assert(offsetof(RdmaWqCtx, db_cos) == 89, "RDMA WQ context ABI drift");
static_assert(sizeof(RdmaCqCtx) == 64, "RDMA CQ context ABI drift");
static_assert(offsetof(RdmaCqCtx, db_addr) == 48, "RDMA CQ context ABI drift");
static_assert(offsetof(RdmaCqCtx, db_sw_addr) == 56, "RDMA CQ context ABI drift");
static_assert(sizeof(Hns1825Cqe) == 32, "RDMA CQE ABI drift");
static_assert(sizeof(RdmaMemInfo) == 24, "RDMA memory-info ABI drift");

inline uint64_t encode_rdma_event_handle(uint32_t dest_rank, uint32_t target_head) {
    return (static_cast<uint64_t>(dest_rank) << kHandleRankIdShift) | static_cast<uint64_t>(target_head);
}

inline bool is_rdma_error_handle(uint64_t handle) {
    return static_cast<uint32_t>(handle >> kHandleRankIdShift) == kRdmaErrorRank;
}

inline void decode_rdma_event_handle(uint64_t handle, uint32_t &dest_rank, uint32_t &target_head) {
    dest_rank = static_cast<uint32_t>(handle >> kHandleRankIdShift);
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

inline bool is_hns1825_cqe_owner_ready(bool owner, uint32_t cur_tail, uint32_t cq_ring) {
    const bool expected_owner = (cur_tail & cq_ring) == 0;
    return owner != expected_owner;
}

inline uint32_t htobe32(uint32_t value) { return __builtin_bswap32(value); }

inline uint32_t load_device_u32(uint64_t addr) {
    auto *ptr = reinterpret_cast<volatile uint32_t *>(static_cast<uintptr_t>(addr));
    return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
}

inline uint32_t load_device_u32_or_zero(uint64_t addr) { return addr != 0 ? load_device_u32(addr) : 0; }

inline RdmaWqCtx load_wq_ctx(uint64_t wq_ptr, uint64_t ctx_index) {
    auto *wq_entry = reinterpret_cast<volatile RdmaWqCtx *>(
        static_cast<uintptr_t>(wq_ptr + ctx_index * static_cast<uint64_t>(sizeof(RdmaWqCtx)))
    );
    invalidate_object(wq_entry, sizeof(*wq_entry));

    RdmaWqCtx wq_ctx{};
    wq_ctx.wqn = __atomic_load_n(&wq_entry->wqn, __ATOMIC_ACQUIRE);
    wq_ctx.buf_addr = __atomic_load_n(&wq_entry->buf_addr, __ATOMIC_ACQUIRE);
    wq_ctx.wqe_size = __atomic_load_n(&wq_entry->wqe_size, __ATOMIC_ACQUIRE);
    wq_ctx.depth = __atomic_load_n(&wq_entry->depth, __ATOMIC_ACQUIRE);
    wq_ctx.head_addr = __atomic_load_n(&wq_entry->head_addr, __ATOMIC_ACQUIRE);
    wq_ctx.tail_addr = __atomic_load_n(&wq_entry->tail_addr, __ATOMIC_ACQUIRE);
    wq_ctx.db_addr = __atomic_load_n(&wq_entry->db_addr, __ATOMIC_ACQUIRE);
    wq_ctx.db_sw_addr = __atomic_load_n(&wq_entry->db_sw_addr, __ATOMIC_ACQUIRE);
    wq_ctx.mtu_shift = __atomic_load_n(&wq_entry->mtu_shift, __ATOMIC_ACQUIRE);
    wq_ctx.db_cos = __atomic_load_n(&wq_entry->db_cos, __ATOMIC_ACQUIRE);
    return wq_ctx;
}

inline RdmaCqCtx load_cq_ctx(uint64_t cq_ptr, uint64_t ctx_index) {
    auto *cq_entry = reinterpret_cast<volatile RdmaCqCtx *>(
        static_cast<uintptr_t>(cq_ptr + ctx_index * static_cast<uint64_t>(sizeof(RdmaCqCtx)))
    );
    invalidate_object(cq_entry, sizeof(*cq_entry));

    RdmaCqCtx cq_ctx{};
    cq_ctx.cqn = __atomic_load_n(&cq_entry->cqn, __ATOMIC_ACQUIRE);
    cq_ctx.buf_addr = __atomic_load_n(&cq_entry->buf_addr, __ATOMIC_ACQUIRE);
    cq_ctx.cqe_size = __atomic_load_n(&cq_entry->cqe_size, __ATOMIC_ACQUIRE);
    cq_ctx.depth = __atomic_load_n(&cq_entry->depth, __ATOMIC_ACQUIRE);
    cq_ctx.head_addr = __atomic_load_n(&cq_entry->head_addr, __ATOMIC_ACQUIRE);
    cq_ctx.tail_addr = __atomic_load_n(&cq_entry->tail_addr, __ATOMIC_ACQUIRE);
    cq_ctx.db_addr = __atomic_load_n(&cq_entry->db_addr, __ATOMIC_ACQUIRE);
    cq_ctx.db_sw_addr = __atomic_load_n(&cq_entry->db_sw_addr, __ATOMIC_ACQUIRE);
    return cq_ctx;
}

inline RdmaMemInfo load_mem_info(uint64_t mem_ptr, uint64_t rank) {
    auto *mem_entry = reinterpret_cast<volatile RdmaMemInfo *>(
        static_cast<uintptr_t>(mem_ptr + rank * static_cast<uint64_t>(sizeof(RdmaMemInfo)))
    );
    invalidate_object(mem_entry, sizeof(*mem_entry));

    RdmaMemInfo mem{};
    mem.size = __atomic_load_n(&mem_entry->size, __ATOMIC_ACQUIRE);
    mem.addr = __atomic_load_n(&mem_entry->addr, __ATOMIC_ACQUIRE);
    mem.lkey = __atomic_load_n(&mem_entry->lkey, __ATOMIC_ACQUIRE);
    mem.rkey = __atomic_load_n(&mem_entry->rkey, __ATOMIC_ACQUIRE);
    return mem;
}

inline bool is_cq_ctx_valid(const RdmaCqCtx &cq_ctx, uint32_t cqe_size) {
    return cqe_size >= sizeof(Hns1825Cqe) && cqe_size <= kCqeBytes && cq_ctx.buf_addr != 0 && cq_ctx.tail_addr != 0 &&
           cq_ctx.db_sw_addr != 0 && is_power_of_two(cq_ctx.depth);
}

inline void log_rdma_wq_snapshot(const char *queue_name, const RdmaWqCtx &wq_ctx, int32_t entry_idx, int32_t cond_idx) {
    LOG_INFO(
        "[ASYNC_WAIT RDMA entry=%d cond=%d] %s wqn=%u depth=%u wqe_size=%u head=%u tail=%u db_sw_be=0x%x "
        "buf=0x%llx head_addr=0x%llx tail_addr=0x%llx db_hw=0x%llx db_sw=0x%llx mtu_shift=%u",
        entry_idx, cond_idx, queue_name, wq_ctx.wqn, wq_ctx.depth, wq_ctx.wqe_size,
        load_device_u32_or_zero(wq_ctx.head_addr), load_device_u32_or_zero(wq_ctx.tail_addr),
        load_device_u32_or_zero(wq_ctx.db_sw_addr), static_cast<unsigned long long>(wq_ctx.buf_addr),
        static_cast<unsigned long long>(wq_ctx.head_addr), static_cast<unsigned long long>(wq_ctx.tail_addr),
        static_cast<unsigned long long>(wq_ctx.db_addr), static_cast<unsigned long long>(wq_ctx.db_sw_addr),
        static_cast<unsigned>(wq_ctx.mtu_shift)
    );
}

inline void log_rdma_mem_snapshot(
    const char *mem_name, const RdmaMemInfo &mem, uint32_t rank, int32_t entry_idx, int32_t cond_idx
) {
    LOG_INFO(
        "[ASYNC_WAIT RDMA entry=%d cond=%d] %s rank=%u addr=0x%llx size=%llu lkey=0x%x rkey=0x%x", entry_idx, cond_idx,
        mem_name, rank, static_cast<unsigned long long>(mem.addr), static_cast<unsigned long long>(mem.size), mem.lkey,
        mem.rkey
    );
}

inline void log_rdma_wqe_snapshot(
    const char *queue_name, const RdmaWqCtx &wq_ctx, uint32_t wqe_index, int32_t entry_idx, int32_t cond_idx
) {
    const uint32_t wqe_size = wq_ctx.wqe_size == 0 ? kCqeBytes : wq_ctx.wqe_size;
    if (wq_ctx.buf_addr == 0 || wq_ctx.depth == 0 || !is_power_of_two(wq_ctx.depth) || wqe_size < kCqeBytes) {
        return;
    }
    const uint64_t wqe_addr =
        wq_ctx.buf_addr + static_cast<uint64_t>(wqe_index & (wq_ctx.depth - 1)) * static_cast<uint64_t>(wqe_size);
    auto *wqe = reinterpret_cast<volatile uint64_t *>(static_cast<uintptr_t>(wqe_addr));
    invalidate_object(wqe, kCqeBytes);
    const uint64_t word0 = __atomic_load_n(&wqe[0], __ATOMIC_ACQUIRE);
    const uint64_t word1 = __atomic_load_n(&wqe[1], __ATOMIC_ACQUIRE);
    const uint64_t word2 = __atomic_load_n(&wqe[2], __ATOMIC_ACQUIRE);
    const uint64_t word3 = __atomic_load_n(&wqe[3], __ATOMIC_ACQUIRE);
    const uint64_t word4 = __atomic_load_n(&wqe[4], __ATOMIC_ACQUIRE);
    const uint64_t word5 = __atomic_load_n(&wqe[5], __ATOMIC_ACQUIRE);
    const uint64_t word6 = __atomic_load_n(&wqe[6], __ATOMIC_ACQUIRE);
    const uint64_t word7 = __atomic_load_n(&wqe[7], __ATOMIC_ACQUIRE);
    LOG_INFO(
        "[ASYNC_WAIT RDMA entry=%d cond=%d] %s wqe index=%u addr=0x%llx raw64="
        "0x%llx,0x%llx,0x%llx,0x%llx,0x%llx,0x%llx,0x%llx,0x%llx",
        entry_idx, cond_idx, queue_name, wqe_index, static_cast<unsigned long long>(wqe_addr),
        static_cast<unsigned long long>(word0), static_cast<unsigned long long>(word1),
        static_cast<unsigned long long>(word2), static_cast<unsigned long long>(word3),
        static_cast<unsigned long long>(word4), static_cast<unsigned long long>(word5),
        static_cast<unsigned long long>(word6), static_cast<unsigned long long>(word7)
    );
}

inline void log_rdma_cqe_snapshot(
    const char *queue_name, const RdmaCqCtx &cq_ctx, uint32_t cur_tail, uint32_t cqe_size, int32_t entry_idx,
    int32_t cond_idx
) {
    if (!is_cq_ctx_valid(cq_ctx, cqe_size)) {
        return;
    }
    const uint64_t cqe_addr =
        cq_ctx.buf_addr + static_cast<uint64_t>(cqe_size) * static_cast<uint64_t>(cur_tail & (cq_ctx.depth - 1));
    auto *cqe = reinterpret_cast<volatile Hns1825Cqe *>(static_cast<uintptr_t>(cqe_addr));
    invalidate_object(cqe, sizeof(*cqe));
    const uint32_t owner_id_qpn = __atomic_load_n(&cqe->owner_id_qpn, __ATOMIC_ACQUIRE);
    const uint32_t op_sr_wqebb = __atomic_load_n(&cqe->op_sr_wqebb, __ATOMIC_ACQUIRE);
    constexpr uint32_t kOwnerShift = 31;
    constexpr uint32_t kCqeOpcodeShift = 27;
    constexpr uint32_t kCqeOpcodeMask = 0x1f;
    const bool owner = (owner_id_qpn & (1u << kOwnerShift)) != 0;
    const uint32_t cqe_type = (op_sr_wqebb >> kCqeOpcodeShift) & kCqeOpcodeMask;
    LOG_INFO(
        "[ASYNC_WAIT RDMA entry=%d cond=%d] %s cqe addr=0x%llx owner=%u ready=%u type=0x%x "
        "owner_id_qpn=0x%x op_sr_wqebb=0x%x syndrome=0x%x",
        entry_idx, cond_idx, queue_name, static_cast<unsigned long long>(cqe_addr), static_cast<unsigned>(owner),
        static_cast<unsigned>(is_hns1825_cqe_owner_ready(owner, cur_tail, cq_ctx.depth)), cqe_type, owner_id_qpn,
        op_sr_wqebb, static_cast<unsigned>(__atomic_load_n(&cqe->syndrome, __ATOMIC_ACQUIRE))
    );
}

inline void log_rdma_cq_snapshot(
    const char *queue_name, const RdmaCqCtx &cq_ctx, uint32_t target_head, int32_t entry_idx, int32_t cond_idx
) {
    const uint32_t cqe_size = cq_ctx.cqe_size == 0 ? kCqeBytes : cq_ctx.cqe_size;
    const uint32_t cur_head = load_device_u32_or_zero(cq_ctx.head_addr);
    const uint32_t cur_tail = load_device_u32_or_zero(cq_ctx.tail_addr);
    const bool ctx_valid = is_cq_ctx_valid(cq_ctx, cqe_size);
    LOG_INFO(
        "[ASYNC_WAIT RDMA entry=%d cond=%d] %s cqn=%u depth=%u cqe_size=%u cur_head=%u cur_tail=%u target_head=%u "
        "db_sw_be=0x%x cq_buf=0x%llx cq_head=0x%llx cq_tail=0x%llx cq_db_sw=0x%llx valid=%u",
        entry_idx, cond_idx, queue_name, cq_ctx.cqn, cq_ctx.depth, cqe_size, cur_head, cur_tail, target_head,
        load_device_u32_or_zero(cq_ctx.db_sw_addr), static_cast<unsigned long long>(cq_ctx.buf_addr),
        static_cast<unsigned long long>(cq_ctx.head_addr), static_cast<unsigned long long>(cq_ctx.tail_addr),
        static_cast<unsigned long long>(cq_ctx.db_sw_addr), static_cast<unsigned>(ctx_valid)
    );
    log_rdma_cqe_snapshot(queue_name, cq_ctx, cur_tail, cqe_size, entry_idx, cond_idx);
}

inline void store_device_u32(uint64_t addr, uint32_t value) {
    auto *ptr = reinterpret_cast<volatile uint32_t *>(static_cast<uintptr_t>(addr));
    __atomic_store_n(ptr, value, __ATOMIC_RELEASE);
#if defined(__aarch64__)
    __asm__ __volatile__("dsb sy" ::: "memory");
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

inline void update_tail_info(const RdmaCqCtx &cq_ctx, const RdmaWqCtx & /*wq_ctx*/, uint32_t cur_tail) {
    if (cq_ctx.tail_addr != 0) {
        store_device_u32(cq_ctx.tail_addr, cur_tail);
    }
    if (cq_ctx.db_sw_addr != 0) {
        store_device_u32(cq_ctx.db_sw_addr, htobe32(cur_tail & kCqUpdateCiMask));
    }
}

// HNS1825 SQ hardware doorbell bit layout (mirrors pto-isa Hns1825SqDb).
// AICore posts WQEs and rings this doorbell via st_dev in RingSqDoorbell; if
// that write never reaches the NIC (head advances, software doorbell written,
// but NIC never consumes), re-issuing the 64-bit doorbell from AICPU can
// confirm whether the write path is the gap. This is a diagnostic helper: it
// reproduces pto-isa's doorbell encoding and stores it to db_addr.
inline void ring_sq_doorbell_from_aicpu(const RdmaWqCtx &wq_ctx, uint32_t cur_head) {
    constexpr uint32_t kPiHighShift = 8;
    constexpr uint32_t kPiFieldShift = 32;
    constexpr uint32_t kDbTypeSq = 21;
    constexpr uint32_t kSgidIdx = 1;
    constexpr uint32_t kCntxSize = 1;
    // Bit layout mirrors pto-isa Hns1825SqDb: qpn[19:0] cntx_size[21:20]
    // rsvd0[22] c[23] cos[26:24] type[31:27] pi[39:32] rsvd1[47:40]
    // xrc_vld[48] rsvd2[49] mtu_shift[52:50] sgid_index[59:53] sub_type[63:60].
    uint64_t dbValue = 0;
    dbValue |= static_cast<uint64_t>(wq_ctx.wqn & 0xfffffULL);                                  // qpn
    dbValue |= static_cast<uint64_t>(kCntxSize) << 20;                                          // cntx_size
    dbValue |= static_cast<uint64_t>(wq_ctx.db_cos & 0x7ULL) << 24;                             // cos (from workspace)
    dbValue |= static_cast<uint64_t>(kDbTypeSq) << 27;                                          // type
    dbValue |= static_cast<uint64_t>(wq_ctx.mtu_shift & 0x7ULL) << 50;                          // mtu_shift
    dbValue |= static_cast<uint64_t>(kSgidIdx & 0x7fULL) << 53;                                 // sgid_index
    dbValue |= ((static_cast<uint64_t>(cur_head >> kPiHighShift) & 0xffULL) << kPiFieldShift);  // PI high bits
    if (wq_ctx.db_addr != 0) {
        auto *db = reinterpret_cast<volatile uint64_t *>(static_cast<uintptr_t>(wq_ctx.db_addr));
        __atomic_store_n(db, dbValue, __ATOMIC_RELEASE);
        LOG_INFO(
            "[ASYNC_WAIT RDMA] AICPU re-ring SQ doorbell db_addr=0x%llx head=%u db_cos=%u dbValue=0x%llx",
            static_cast<unsigned long long>(wq_ctx.db_addr), cur_head, static_cast<unsigned>(wq_ctx.db_cos),
            static_cast<unsigned long long>(dbValue)
        );
    }
}

inline CompletionPollResult poll_rdma_event_handle(uint64_t event_handle, uint64_t workspace_addr) {
    if (event_handle == 0) {
        return {CompletionPollState::READY, SIMPLER_ERROR_NONE};
    }
    if (is_rdma_error_handle(event_handle) || workspace_addr == 0) {
        LOG_ERROR(
            "RDMA completion invalid: bad handle/workspace handle=0x%llx workspace=0x%llx is_error=%u",
            static_cast<unsigned long long>(event_handle), static_cast<unsigned long long>(workspace_addr),
            static_cast<unsigned>(is_rdma_error_handle(event_handle))
        );
        return {CompletionPollState::FAILED, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID};
    }

    uint32_t dest_rank = 0;
    uint32_t target_head = 0;
    decode_rdma_event_handle(event_handle, dest_rank, target_head);
    if (target_head == 0) {
        return {CompletionPollState::READY, SIMPLER_ERROR_NONE};
    }

    auto *info = reinterpret_cast<volatile RdmaInfo *>(static_cast<uintptr_t>(workspace_addr));
    invalidate_object(info, sizeof(*info));
    const uint32_t magic = __atomic_load_n(&info->magic, __ATOMIC_ACQUIRE);
    const uint32_t version = __atomic_load_n(&info->version, __ATOMIC_ACQUIRE);
    const uint32_t backend = __atomic_load_n(&info->backend, __ATOMIC_ACQUIRE);
    const uint32_t qp_num = __atomic_load_n(&info->qp_num, __ATOMIC_ACQUIRE);
    const uint32_t rank_count = __atomic_load_n(&info->rank_count, __ATOMIC_ACQUIRE);
    const uint64_t sq_ptr = __atomic_load_n(&info->sq_ptr, __ATOMIC_ACQUIRE);
    const uint64_t scq_ptr = __atomic_load_n(&info->scq_ptr, __ATOMIC_ACQUIRE);
    if (magic != kRdmaWorkspaceMagic || version != kRdmaWorkspaceVersion || backend != kRdmaBackendHns1825 ||
        qp_num == 0 || dest_rank >= rank_count || sq_ptr == 0 || scq_ptr == 0) {
        LOG_ERROR(
            "RDMA completion invalid: workspace metadata handle=0x%llx workspace=0x%llx dest_rank=%u target_head=%u "
            "magic=0x%x version=%u backend=%u qp_num=%u rank_count=%u sq=0x%llx scq=0x%llx",
            static_cast<unsigned long long>(event_handle), static_cast<unsigned long long>(workspace_addr), dest_rank,
            target_head, magic, version, backend, qp_num, rank_count, static_cast<unsigned long long>(sq_ptr),
            static_cast<unsigned long long>(scq_ptr)
        );
        return {CompletionPollState::FAILED, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID};
    }

    const uint64_t ctx_index = static_cast<uint64_t>(dest_rank) * qp_num;
    auto *cq_entry = reinterpret_cast<volatile RdmaCqCtx *>(
        static_cast<uintptr_t>(scq_ptr + ctx_index * static_cast<uint64_t>(sizeof(RdmaCqCtx)))
    );
    auto *wq_entry = reinterpret_cast<volatile RdmaWqCtx *>(
        static_cast<uintptr_t>(sq_ptr + ctx_index * static_cast<uint64_t>(sizeof(RdmaWqCtx)))
    );
    invalidate_object(cq_entry, sizeof(*cq_entry));
    invalidate_object(wq_entry, sizeof(*wq_entry));

    RdmaCqCtx cq_ctx{};
    cq_ctx.buf_addr = __atomic_load_n(&cq_entry->buf_addr, __ATOMIC_ACQUIRE);
    cq_ctx.cqe_size = __atomic_load_n(&cq_entry->cqe_size, __ATOMIC_ACQUIRE);
    cq_ctx.depth = __atomic_load_n(&cq_entry->depth, __ATOMIC_ACQUIRE);
    cq_ctx.tail_addr = __atomic_load_n(&cq_entry->tail_addr, __ATOMIC_ACQUIRE);
    cq_ctx.db_addr = __atomic_load_n(&cq_entry->db_addr, __ATOMIC_ACQUIRE);
    cq_ctx.db_sw_addr = __atomic_load_n(&cq_entry->db_sw_addr, __ATOMIC_ACQUIRE);

    RdmaWqCtx wq_ctx{};
    wq_ctx.wqn = __atomic_load_n(&wq_entry->wqn, __ATOMIC_ACQUIRE);
    wq_ctx.depth = __atomic_load_n(&wq_entry->depth, __ATOMIC_ACQUIRE);
    wq_ctx.head_addr = __atomic_load_n(&wq_entry->head_addr, __ATOMIC_ACQUIRE);
    wq_ctx.tail_addr = __atomic_load_n(&wq_entry->tail_addr, __ATOMIC_ACQUIRE);
    wq_ctx.db_addr = __atomic_load_n(&wq_entry->db_addr, __ATOMIC_ACQUIRE);
    wq_ctx.db_sw_addr = __atomic_load_n(&wq_entry->db_sw_addr, __ATOMIC_ACQUIRE);
    wq_ctx.mtu_shift = __atomic_load_n(&wq_entry->mtu_shift, __ATOMIC_ACQUIRE);
    wq_ctx.db_cos = __atomic_load_n(&wq_entry->db_cos, __ATOMIC_ACQUIRE);
    const uint32_t cqe_size = cq_ctx.cqe_size == 0 ? kCqeBytes : cq_ctx.cqe_size;
    const uint32_t cq_ring = cq_ctx.depth;
    if (cqe_size < sizeof(Hns1825Cqe) || cqe_size > kCqeBytes || cq_ctx.buf_addr == 0 || cq_ctx.tail_addr == 0 ||
        cq_ctx.db_sw_addr == 0 || !is_power_of_two(cq_ring)) {
        LOG_ERROR(
            "RDMA completion invalid: cq context handle=0x%llx workspace=0x%llx dest_rank=%u target_head=%u "
            "cqe_size=%u depth=%u cq_ring=%u cq_buf=0x%llx cq_tail=0x%llx cq_db_sw=0x%llx sq_tail=0x%llx",
            static_cast<unsigned long long>(event_handle), static_cast<unsigned long long>(workspace_addr), dest_rank,
            target_head, cqe_size, cq_ctx.depth, cq_ring, static_cast<unsigned long long>(cq_ctx.buf_addr),
            static_cast<unsigned long long>(cq_ctx.tail_addr), static_cast<unsigned long long>(cq_ctx.db_sw_addr),
            static_cast<unsigned long long>(wq_ctx.tail_addr)
        );
        return {CompletionPollState::FAILED, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID};
    }

    uint32_t cur_tail = load_device_u32(cq_ctx.tail_addr);
    if (has_reached(cur_tail, target_head)) {
        return {CompletionPollState::READY, SIMPLER_ERROR_NONE};
    }

    uint32_t next_tail = cur_tail;
    while (!has_reached(next_tail, target_head)) {
        const uint64_t cqe_addr =
            cq_ctx.buf_addr + static_cast<uint64_t>(cqe_size) * static_cast<uint64_t>(next_tail & (cq_ring - 1));
        auto *cqe = reinterpret_cast<volatile Hns1825Cqe *>(static_cast<uintptr_t>(cqe_addr));
        invalidate_object(cqe, sizeof(*cqe));
        const uint32_t owner_id_qpn = __atomic_load_n(&cqe->owner_id_qpn, __ATOMIC_ACQUIRE);
        const uint32_t op_sr_wqebb = __atomic_load_n(&cqe->op_sr_wqebb, __ATOMIC_ACQUIRE);
        constexpr uint32_t kOwnerShift = 31;
        constexpr uint32_t kCqeOpcodeShift = 27;
        constexpr uint32_t kCqeOpcodeMask = 0x1f;
        constexpr uint32_t kCqeOptypeError = 0x1e;
        constexpr uint32_t kCqeOptypeInvalid = 0x1f;
        const uint32_t cqe_type = (op_sr_wqebb >> kCqeOpcodeShift) & kCqeOpcodeMask;
        const bool owner = (owner_id_qpn & (1u << kOwnerShift)) != 0;
        if (cqe_type == kCqeOptypeInvalid || !is_hns1825_cqe_owner_ready(owner, next_tail, cq_ring)) {
            break;
        }

        if (cqe_type == kCqeOptypeError) {
            const uint8_t syndrome = __atomic_load_n(&cqe->syndrome, __ATOMIC_ACQUIRE);
            const uint32_t byte_cnt = __atomic_load_n(&cqe->byte_cnt, __ATOMIC_ACQUIRE);
            const uint32_t imm_data = __atomic_load_n(&cqe->imm_data, __ATOMIC_ACQUIRE);
            const uint32_t wqe_num = __atomic_load_n(&cqe->wqe_num, __ATOMIC_ACQUIRE);
            const uint32_t vlan_queue_index = __atomic_load_n(&cqe->vlan_queue_index, __ATOMIC_ACQUIRE);
            LOG_ERROR(
                "RDMA completion invalid: CQE error handle=0x%llx workspace=0x%llx dest_rank=%u target_head=%u "
                "cur_tail=%u next_tail=%u cqe_addr=0x%llx owner_id_qpn=0x%x op_sr_wqebb=0x%x "
                "byte_cnt=0x%x imm_data=0x%x wqe_num=0x%x vlan_queue_index=0x%x syndrome=0x%x",
                static_cast<unsigned long long>(event_handle), static_cast<unsigned long long>(workspace_addr),
                dest_rank, target_head, cur_tail, next_tail, static_cast<unsigned long long>(cqe_addr), owner_id_qpn,
                op_sr_wqebb, byte_cnt, imm_data, wqe_num, vlan_queue_index, static_cast<unsigned>(syndrome)
            );
            ++next_tail;
            update_tail_info(cq_ctx, wq_ctx, next_tail);
            return {CompletionPollState::FAILED, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID};
        }
        next_tail++;
    }

    if (next_tail != cur_tail) {
        update_tail_info(cq_ctx, wq_ctx, next_tail);
    }
    if (!has_reached(next_tail, target_head)) {
        // Diagnostic: AICore posts WQEs and rings the SQ hardware doorbell via
        // st_dev, but the NIC is observed never consuming (head>tail, CQ empty).
        // Re-issue the hardware doorbell from AICPU once per handle to test
        // whether the AICore doorbell write is the gap. Logged once per handle.
        static thread_local uint64_t last_ringed_handle = 0;
        if (last_ringed_handle != event_handle) {
            last_ringed_handle = event_handle;
            const uint32_t sq_head = load_device_u32_or_zero(wq_ctx.head_addr);
            if (sq_head != 0) {
                ring_sq_doorbell_from_aicpu(wq_ctx, sq_head);
            }
        }
    }
    return {
        has_reached(next_tail, target_head) ? CompletionPollState::READY : CompletionPollState::PENDING,
        SIMPLER_ERROR_NONE
    };
}

inline void
log_rdma_event_handle_snapshot(uint64_t event_handle, uint64_t workspace_addr, int32_t entry_idx, int32_t cond_idx) {
    uint32_t dest_rank = 0;
    uint32_t target_head = 0;
    decode_rdma_event_handle(event_handle, dest_rank, target_head);
    LOG_INFO(
        "[ASYNC_WAIT RDMA entry=%d cond=%d] handle=0x%llx workspace=0x%llx dest_rank=%u target_head=%u", entry_idx,
        cond_idx, static_cast<unsigned long long>(event_handle), static_cast<unsigned long long>(workspace_addr),
        dest_rank, target_head
    );

    if (event_handle == 0 || workspace_addr == 0 || is_rdma_error_handle(event_handle)) {
        return;
    }

    auto *info = reinterpret_cast<volatile RdmaInfo *>(static_cast<uintptr_t>(workspace_addr));
    invalidate_object(info, sizeof(*info));
    const uint32_t magic = __atomic_load_n(&info->magic, __ATOMIC_ACQUIRE);
    const uint32_t version = __atomic_load_n(&info->version, __ATOMIC_ACQUIRE);
    const uint32_t backend = __atomic_load_n(&info->backend, __ATOMIC_ACQUIRE);
    const uint32_t qp_num = __atomic_load_n(&info->qp_num, __ATOMIC_ACQUIRE);
    const uint32_t rank_count = __atomic_load_n(&info->rank_count, __ATOMIC_ACQUIRE);
    const uint64_t sq_ptr = __atomic_load_n(&info->sq_ptr, __ATOMIC_ACQUIRE);
    const uint64_t rq_ptr = __atomic_load_n(&info->rq_ptr, __ATOMIC_ACQUIRE);
    const uint64_t scq_ptr = __atomic_load_n(&info->scq_ptr, __ATOMIC_ACQUIRE);
    const uint64_t rcq_ptr = __atomic_load_n(&info->rcq_ptr, __ATOMIC_ACQUIRE);
    const uint64_t mem_ptr = __atomic_load_n(&info->mem_ptr, __ATOMIC_ACQUIRE);
    LOG_INFO(
        "[ASYNC_WAIT RDMA entry=%d cond=%d] info magic=0x%x version=%u backend=%u qp_num=%u rank_count=%u "
        "sq=0x%llx rq=0x%llx scq=0x%llx rcq=0x%llx mem=0x%llx",
        entry_idx, cond_idx, magic, version, backend, qp_num, rank_count, static_cast<unsigned long long>(sq_ptr),
        static_cast<unsigned long long>(rq_ptr), static_cast<unsigned long long>(scq_ptr),
        static_cast<unsigned long long>(rcq_ptr), static_cast<unsigned long long>(mem_ptr)
    );
    if (magic != kRdmaWorkspaceMagic || version != kRdmaWorkspaceVersion || backend != kRdmaBackendHns1825 ||
        qp_num == 0 || dest_rank >= rank_count || sq_ptr == 0 || scq_ptr == 0) {
        return;
    }

    const uint64_t ctx_index = static_cast<uint64_t>(dest_rank) * qp_num;
    RdmaWqCtx sq_ctx = load_wq_ctx(sq_ptr, ctx_index);
    log_rdma_wq_snapshot("sq", sq_ctx, entry_idx, cond_idx);
    if (rq_ptr != 0) {
        log_rdma_wq_snapshot("rq", load_wq_ctx(rq_ptr, ctx_index), entry_idx, cond_idx);
    }
    log_rdma_cq_snapshot("scq", load_cq_ctx(scq_ptr, ctx_index), target_head, entry_idx, cond_idx);
    if (rcq_ptr != 0) {
        log_rdma_cq_snapshot("rcq", load_cq_ctx(rcq_ptr, ctx_index), target_head, entry_idx, cond_idx);
    }
    if (mem_ptr != 0) {
        log_rdma_mem_snapshot("mem", load_mem_info(mem_ptr, dest_rank), dest_rank, entry_idx, cond_idx);
    }
    if (target_head != 0) {
        log_rdma_wqe_snapshot("sq", sq_ctx, 0, entry_idx, cond_idx);
        if (target_head > 1) {
            log_rdma_wqe_snapshot("sq", sq_ctx, target_head - 1, entry_idx, cond_idx);
        }
    }
}

inline void retire_rdma_event_handle(uint64_t /*event_handle*/, uint64_t /*workspace_addr*/) {}

}  // namespace pto2::rdma_backend
