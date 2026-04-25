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
 * @file pmu_collector_aicpu.cpp
 * @brief AICPU-side PMU init/finalize + record commit from AICore slot (a5)
 *
 * AICPU programs PMU event selectors and starts PMU_CTRL_0/1 once at init,
 * publishes (pmu_buffer_addr, pmu_reg_base) into each Handshake so AICore
 * can read PMU MMIO itself, and commits slots AICore wrote into
 * PmuBuffer::dual_issue_slots on each task FIN. At finalize it restores
 * CTRL to the pre-run state.
 *
 * On DAV_3510 the halResMap mapping is a single 3 MB page per AICore that
 * covers CTRL offsets (0x0, 0x5108) and PMU offsets (0x2400-0x2524,
 * 0x4200-0x42AC), so the same per-core reg base is used for init/finalize
 * (here) and for AICore-side MMIO reads.
 */

#include "aicpu/pmu_collector_aicpu.h"

#include <cstring>

#include "aicpu/platform_regs.h"
#include "common/platform_config.h"
#include "common/unified_log.h"

static uint64_t g_platform_pmu_base = 0;
static bool g_enable_pmu = false;

// Saved CTRL register state per core, indexed by logical core_id.
// Populated by pmu_aicpu_init(), consumed by pmu_aicpu_finalize().
static uint32_t g_pmu_saved_ctrl0[PLATFORM_MAX_CORES];
static uint32_t g_pmu_saved_ctrl1[PLATFORM_MAX_CORES];

// Per-core cached PmuBuffer pointer.
static PmuBuffer *s_pmu_buffers[PLATFORM_MAX_CORES];
static PmuSetupHeader *s_pmu_header = nullptr;

// Per-core resolved register base address, keyed by logical core_id.
// Populated by pmu_aicpu_init() from get_platform_regs() — the same halResMap
// mapping used for CTRL MMIO also covers PMU MMIO on DAV_3510.
static uint64_t s_pmu_reg_addrs[PLATFORM_MAX_CORES] = {0};

extern "C" void set_platform_pmu_base(uint64_t pmu_data_base) { g_platform_pmu_base = pmu_data_base; }

extern "C" uint64_t get_platform_pmu_base() { return g_platform_pmu_base; }

extern "C" void set_pmu_enabled(bool enable) { g_enable_pmu = enable; }

extern "C" bool is_pmu_enabled() { return g_enable_pmu; }

// ---------------------------------------------------------------------------
// Low-level MMIO helpers (internal use only)
// ---------------------------------------------------------------------------

static void pmu_program_events(uint64_t reg_base, const PmuEventConfig &events) {
    for (int i = 0; i < PMU_COUNTER_COUNT_A5; i++) {
        write_reg(reg_base, reg_index(RegId::PMU_CNT0_IDX, i), events.event_ids[i]);
    }
}

static void pmu_start(uint64_t reg_base, uint32_t &saved_ctrl0, uint32_t &saved_ctrl1) {
    // Clear counters by reading them once
    for (int i = 0; i < PMU_COUNTER_COUNT_A5; i++) {
        (void)read_reg(reg_base, reg_index(RegId::PMU_CNT0, i));
    }
    (void)read_reg(reg_base, RegId::PMU_CNT_TOTAL0);
    (void)read_reg(reg_base, RegId::PMU_CNT_TOTAL1);

    // Full cycle counting range: start at 0, stop at 0xFFFFFFFF
    write_reg(reg_base, RegId::PMU_START_CYC0, 0x0);
    write_reg(reg_base, RegId::PMU_START_CYC1, 0x0);
    write_reg(reg_base, RegId::PMU_STOP_CYC0, 0xFFFFFFFF);
    write_reg(reg_base, RegId::PMU_STOP_CYC1, 0xFFFFFFFF);

    // Save and set CTRL_0 / CTRL_1 (a5 has dual control registers)
    saved_ctrl0 = static_cast<uint32_t>(read_reg(reg_base, RegId::PMU_CTRL_0));
    saved_ctrl1 = static_cast<uint32_t>(read_reg(reg_base, RegId::PMU_CTRL_1));
    write_reg(reg_base, RegId::PMU_CTRL_0, REG_MMIO_PMU_CTRL_0_ENABLE_VAL);
    write_reg(reg_base, RegId::PMU_CTRL_1, REG_MMIO_PMU_CTRL_1_ENABLE_VAL);
}

static void pmu_stop(uint64_t reg_base, uint32_t saved_ctrl0, uint32_t saved_ctrl1) {
    write_reg(reg_base, RegId::PMU_CTRL_0, saved_ctrl0);
    write_reg(reg_base, RegId::PMU_CTRL_1, saved_ctrl1);
}

// ---------------------------------------------------------------------------
// High-level interface
// ---------------------------------------------------------------------------

void pmu_aicpu_init(Handshake *handshakes, const uint32_t *physical_core_ids, int num_cores) {
    void *pmu_base = reinterpret_cast<void *>(get_platform_pmu_base());
    if (pmu_base == nullptr) {
        LOG_ERROR("pmu_aicpu_init: pmu_data_base is NULL");
        return;
    }
    if (handshakes == nullptr) {
        LOG_ERROR("pmu_aicpu_init: handshakes is NULL");
        return;
    }
    s_pmu_header = reinterpret_cast<PmuSetupHeader *>(pmu_base);
    uint32_t pmu_event_type = s_pmu_header->event_type;

    // Resolve per-core register base from physical_core_ids. On DAV_3510 the
    // halResMap(RES_AICORE) mapping covers CTRL + PMU in a single per-core
    // page, so PMU helpers reuse get_platform_regs().
    uint64_t *regs_array = reinterpret_cast<uint64_t *>(get_platform_regs());
    for (int i = 0; i < num_cores; i++) {
        if (i >= PLATFORM_MAX_CORES) {
            LOG_ERROR("pmu_aicpu_init: num_cores %d exceeds PLATFORM_MAX_CORES %d", num_cores, PLATFORM_MAX_CORES);
            break;
        }
        s_pmu_reg_addrs[i] = regs_array ? regs_array[physical_core_ids[i]] : 0;
        s_pmu_buffers[i] = reinterpret_cast<PmuBuffer *>(s_pmu_header->buffer_ptrs[i]);
        g_pmu_saved_ctrl0[i] = 0;
        g_pmu_saved_ctrl1[i] = 0;
    }

    // Program event selectors and start PMU counters on all cores with a valid
    // PMU reg base.
    const PmuEventConfig *evt = pmu_resolve_event_config_a5(static_cast<PmuEventType>(pmu_event_type));
    if (evt == nullptr) {
        evt = &PMU_EVENTS_A5_PIPE_UTIL;
    }
    for (int i = 0; i < num_cores; i++) {
        uint64_t reg_addr = s_pmu_reg_addrs[i];
        if (reg_addr == 0) {
            LOG_WARN("pmu_aicpu_init: core %d has no PMU reg_addr, skipping MMIO programming", i);
            continue;
        }
        pmu_program_events(reg_addr, *evt);
        pmu_start(reg_addr, g_pmu_saved_ctrl0[i], g_pmu_saved_ctrl1[i]);
    }

    // Publish per-core (pmu_buffer_addr, pmu_reg_base) into the matching
    // Handshake so AICore can read PMU MMIO and write the dual-issue slot.
    // Must happen before the caller sets aicpu_regs_ready=1 — AICore spins
    // on that and reads these fields under the same release/acquire pair.
    for (int i = 0; i < num_cores; i++) {
        handshakes[i].pmu_buffer_addr = reinterpret_cast<uint64_t>(s_pmu_buffers[i]);
        handshakes[i].pmu_reg_base = s_pmu_reg_addrs[i];
    }

    LOG_INFO("PMU initialized: %d cores, event_type=%u", num_cores, pmu_event_type);
}

void pmu_aicpu_complete_record(
    int core_id, int thread_idx, uint32_t reg_task_id, uint64_t task_id, uint32_t func_id, CoreType core_type
) {
    if (s_pmu_header == nullptr || core_id < 0 || core_id >= PLATFORM_MAX_CORES) {
        return;
    }
    PmuBuffer *buf = s_pmu_buffers[core_id];
    if (buf == nullptr) {
        return;
    }
    PmuBufferState *state = get_pmu_buffer_state(s_pmu_header, core_id);

    // Stamp thread ownership on every commit. a5 binds each core to a fixed
    // AICPU scheduler thread at init time, so this value is stable — host
    // reads it at collect time to emit the CSV thread_id column.
    state->owning_thread_id = static_cast<uint32_t>(thread_idx);

    // Account for every commit attempt so host can detect silent slot loss.
    state->total_record_count += 1;

    uint32_t idx = buf->count;
    if (idx >= static_cast<uint32_t>(PLATFORM_PMU_RECORDS_PER_BUFFER)) {
        // Buffer full — drop the record. Host surfaces the total at finalize.
        state->dropped_record_count += 1;
        return;
    }

    // Fetch AICore's writes into the dual-issue slot (index = reg_task_id & 1).
    // Match the slot on the 32-bit register token AICore wrote, not the logical
    // task_id (which may be a 64-bit PTO2 (ring<<32|local) value).
    PmuRecord *slot = &buf->dual_issue_slots[reg_task_id & 1u];
    cache_invalidate_range(slot, sizeof(PmuRecord));

    if (static_cast<uint32_t>(slot->task_id) != reg_task_id) {
        // AICore hasn't published this slot yet. Should not happen because
        // AICore writes task_id last + dcci flush before COND FIN, but bail
        // out defensively rather than committing stale counter values.
        // Counted in total_record_count above so host sees the gap.
        return;
    }

    PmuRecord *rec = &buf->records[idx];
    rec->task_id = task_id;
    rec->func_id = func_id;
    rec->core_type = core_type;
    rec->pmu_total_cycles = slot->pmu_total_cycles;
    for (int i = 0; i < PMU_COUNTER_COUNT_A5; i++) {
        rec->pmu_counters[i] = slot->pmu_counters[i];
    }
    buf->count = idx + 1;
}

void pmu_aicpu_finalize(const int *cur_thread_cores, int core_num) {
    if (s_pmu_header == nullptr) {
        return;
    }
    for (int i = 0; i < core_num; i++) {
        int core_id = cur_thread_cores[i];
        if (core_id < 0 || core_id >= PLATFORM_MAX_CORES) {
            LOG_ERROR("pmu_aicpu_finalize: invalid core_id %d (max %d)", core_id, PLATFORM_MAX_CORES);
            continue;
        }
        uint64_t reg_addr = s_pmu_reg_addrs[core_id];
        if (reg_addr != 0) {
            pmu_stop(reg_addr, g_pmu_saved_ctrl0[core_id], g_pmu_saved_ctrl1[core_id]);
        }
    }
}
