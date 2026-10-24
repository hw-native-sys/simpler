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

#include <cstdint>

#include "aicpu/platform_regs.h"
#include "aicore_completion_mailbox.h"
#include "common/unified_log.h"
#include "completion_token.h"
#include "runtime_status.h"

namespace pto2::rdma_backend {

// HNS1825 CQE layout as the poll below consumes it. The owner bit and the opcode share the first two
// dwords, so one 8-byte load decides both readiness and error.
inline constexpr uint32_t kRdmaCqeOwnerBit = 31U;
inline constexpr uint32_t kRdmaCqeOpcodeShift = 32U + 27U;
inline constexpr uint32_t kRdmaCqeOpcodeMask = 0x1FU;
inline constexpr uint32_t kRdmaCqeOpcodeError = 0x1EU;
inline constexpr uint32_t kRdmaCqeOpcodeInvalid = 0x1FU;

inline uintptr_t cache_line(const volatile void *addr) {
    return reinterpret_cast<uintptr_t>(addr) & ~(uintptr_t(CHIP_ALIGN_SIZE) - 1u);
}

// Read-only completion check. The kernel computed this CQE's address and the owner value that marks
// it done; the NIC writes both the owner bit and the opcode, so nothing here needs the SQ or the CQ
// tail. CQ reclaim stays with the sender, which recycles as its SQ nears full.
inline CompletionPollResult poll_rdma_hns1825_cqe_record(uint64_t cqe_addr, uint32_t expected_owner) {
    if (cqe_addr == 0U || expected_owner > 1U) {
        return {CompletionPollState::FAILED, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID};
    }

    auto *dw01_addr = reinterpret_cast<volatile uint64_t *>(static_cast<uintptr_t>(cqe_addr));
    cache_invalidate_range(reinterpret_cast<const void *>(cache_line(dw01_addr)), CHIP_ALIGN_SIZE);
    const uint64_t dw01 = __atomic_load_n(dw01_addr, __ATOMIC_ACQUIRE);

    const uint32_t owner = static_cast<uint32_t>((dw01 >> kRdmaCqeOwnerBit) & 1U);
    const uint32_t opcode = static_cast<uint32_t>((dw01 >> kRdmaCqeOpcodeShift) & kRdmaCqeOpcodeMask);
    if (opcode == kRdmaCqeOpcodeInvalid || owner != expected_owner) {
        return {CompletionPollState::PENDING, SIMPLER_ERROR_NONE};
    }
    if (opcode == kRdmaCqeOpcodeError) {
        return {CompletionPollState::FAILED, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID};
    }
    return {CompletionPollState::READY, SIMPLER_ERROR_NONE};
}

inline void retire_rdma_hns1825_cqe_record(uint64_t /*cqe_addr*/, uint32_t /*expected_owner*/) {}

// Diagnostic mirror of the poll above, for a wait that has stopped progressing.
inline void
log_rdma_hns1825_cqe_record_snapshot(uint64_t cqe_addr, uint32_t expected_owner, int32_t entry_idx, int32_t cond_idx) {
    if (cqe_addr == 0U) {
        return;
    }
    auto *dw01_addr = reinterpret_cast<volatile uint64_t *>(static_cast<uintptr_t>(cqe_addr));
    cache_invalidate_range(reinterpret_cast<const void *>(cache_line(dw01_addr)), CHIP_ALIGN_SIZE);
    const uint64_t dw01 = __atomic_load_n(dw01_addr, __ATOMIC_ACQUIRE);

    const uint32_t owner = static_cast<uint32_t>((dw01 >> kRdmaCqeOwnerBit) & 1U);
    const uint32_t opcode = static_cast<uint32_t>((dw01 >> kRdmaCqeOpcodeShift) & kRdmaCqeOpcodeMask);
    LOG_INFO(
        "[ASYNC_WAIT RDMA entry=%d cond=%d] hns1825_cqe addr=0x%llx owner=%u expected=%u type=0x%x "
        "owner_id_qpn=0x%x op_sr_wqebb=0x%x",
        entry_idx, cond_idx, static_cast<unsigned long long>(cqe_addr), owner, expected_owner, opcode,
        static_cast<unsigned>(dw01 & 0xffffffffu), static_cast<unsigned>(dw01 >> 32u)
    );
}

}  // namespace pto2::rdma_backend
