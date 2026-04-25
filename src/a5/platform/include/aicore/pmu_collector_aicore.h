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
/**
 * @file pmu_collector_aicore.h
 * @brief AICore-side PMU gate + per-task MMIO record (a5)
 *
 * Split of duties:
 *   - AICPU programs event selectors + starts PMU_CTRL_0/1 once at init,
 *     and restores them at finalize.
 *   - AICore gates counting around each kernel execution via CTRL SPR bit 0
 *     (pmu_aicore_begin / pmu_aicore_end), reads the 10 PMU counters +
 *     PMU_CNT_TOTAL via MMIO using the handshake-supplied reg_base, and
 *     writes the snapshot into PmuBuffer::dual_issue_slots[task_id & 1].
 *   - AICPU, on COND FIN, validates the slot and copies it into
 *     PmuBuffer::records[count] while filling func_id / core_type
 *     (pmu_aicpu_complete_record).
 *
 * The two dual_issue_slots exist because AICore's dual-issue dispatch
 * can have up to two tasks in flight per core — the parity task_id & 1
 * keeps N and N+1 from colliding.
 */

#ifndef PLATFORM_AICORE_PMU_COLLECTOR_AICORE_H_
#define PLATFORM_AICORE_PMU_COLLECTOR_AICORE_H_

#include "aicore/aicore.h"
#include "common/platform_config.h"
#include "common/pmu_profiling.h"

// PMU enable bit in the AICore CTRL SPR (bit 0 = GLB_PMU_EN).
constexpr uint64_t PMU_AICORE_CTRL_ENABLE_BIT = 0x1ULL;

/**
 * Begin PMU counting window: set CTRL bit 0 so hardware counters start accruing.
 */
__aicore__ __attribute__((always_inline)) static inline void pmu_aicore_begin() {
    write_reg(RegId::CTRL, read_reg(RegId::CTRL) | PMU_AICORE_CTRL_ENABLE_BIT);
}

/**
 * End PMU counting window: clear CTRL bit 0 so counters freeze until next begin.
 */
__aicore__ __attribute__((always_inline)) static inline void pmu_aicore_end() {
    write_reg(RegId::CTRL, read_reg(RegId::CTRL) & ~PMU_AICORE_CTRL_ENABLE_BIT);
}

/**
 * Record PMU counters for one completed task into the dual-issue slot
 * (AICore-side producer half of the PMU record path).
 *
 * Must be called after pmu_aicore_end() has frozen the counters.
 * AICPU picks up the slot and commits it via pmu_aicpu_complete_record.
 *
 * Leaves func_id and core_type untouched — those are AICPU-owned fields.
 *
 * @param buf       Per-core PmuBuffer (from Handshake.pmu_buffer_addr)
 * @param reg_base  Per-core PMU MMIO base (from Handshake.pmu_reg_base)
 * @param task_id   Register dispatch token (DATA_MAIN_BASE value for this task)
 */
__aicore__ __attribute__((always_inline)) static inline void
pmu_aicore_record_task(__gm__ PmuBuffer *buf, uint64_t reg_base, uint32_t task_id) {
    if (buf == nullptr || reg_base == 0) {
        return;
    }

    __gm__ PmuRecord *slot = &buf->dual_issue_slots[task_id & 1u];

    // Read the 10 event counters + 64-bit cycle counter via the AICore MMIO
    // load intrinsic ld_dev(base, offset) — the only legal way for AICore to
    // read its own MMIO regs. CCE constrains `offset` to a 12-bit signed
    // immediate ([-2048, 2047]), so we rebase the pointer to the start of
    // the PMU CTRL/CNT block (0x4200) — relative offsets then fit within
    // [0x10, 0x64] and stay inside the immediate range.
    int32_t *pmu_base = reinterpret_cast<int32_t *>(reg_base + REG_MMIO_PMU_CTRL_0_OFFSET);
    constexpr int16_t REL = static_cast<int16_t>(REG_MMIO_PMU_CTRL_0_OFFSET);
    slot->pmu_counters[0] = static_cast<uint32_t>(ld_dev(pmu_base, REG_MMIO_PMU_CNT0_OFFSET - REL));
    slot->pmu_counters[1] = static_cast<uint32_t>(ld_dev(pmu_base, REG_MMIO_PMU_CNT1_OFFSET - REL));
    slot->pmu_counters[2] = static_cast<uint32_t>(ld_dev(pmu_base, REG_MMIO_PMU_CNT2_OFFSET - REL));
    slot->pmu_counters[3] = static_cast<uint32_t>(ld_dev(pmu_base, REG_MMIO_PMU_CNT3_OFFSET - REL));
    slot->pmu_counters[4] = static_cast<uint32_t>(ld_dev(pmu_base, REG_MMIO_PMU_CNT4_OFFSET - REL));
    slot->pmu_counters[5] = static_cast<uint32_t>(ld_dev(pmu_base, REG_MMIO_PMU_CNT5_OFFSET - REL));
    slot->pmu_counters[6] = static_cast<uint32_t>(ld_dev(pmu_base, REG_MMIO_PMU_CNT6_OFFSET - REL));
    slot->pmu_counters[7] = static_cast<uint32_t>(ld_dev(pmu_base, REG_MMIO_PMU_CNT7_OFFSET - REL));
    slot->pmu_counters[8] = static_cast<uint32_t>(ld_dev(pmu_base, REG_MMIO_PMU_CNT8_OFFSET - REL));
    slot->pmu_counters[9] = static_cast<uint32_t>(ld_dev(pmu_base, REG_MMIO_PMU_CNT9_OFFSET - REL));
    uint64_t lo = static_cast<uint32_t>(ld_dev(pmu_base, REG_MMIO_PMU_CNT_TOTAL0_OFFSET - REL));
    uint64_t hi = static_cast<uint32_t>(ld_dev(pmu_base, REG_MMIO_PMU_CNT_TOTAL1_OFFSET - REL));
    slot->pmu_total_cycles = lo | (hi << 32);

    // Publish task_id last so AICPU can validate the slot is ready.
    OUT_OF_ORDER_STORE_BARRIER();
    slot->task_id = static_cast<uint64_t>(task_id);

    // Flush cache to make data visible to AICPU.
    dcci(slot, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dsb((mem_dsb_t)0);
}

#endif  // PLATFORM_AICORE_PMU_COLLECTOR_AICORE_H_
