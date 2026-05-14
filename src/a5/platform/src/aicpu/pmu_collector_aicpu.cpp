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
 * @brief AICPU-side PMU init/finalize + record commit from AICore staging-ring slot (a5)
 *
 * Buffer switching mirrors a2a3 pmu_collector_aicpu.cpp:
 *   - SPSC free_queue: Host pushes free PmuBuffers, AICPU pops when switching.
 *   - Per-thread ready_queue: AICPU enqueues full buffers for host collection.
 *   - On free_queue empty or ready_queue full: overwrite current buffer (data lost,
 *     same policy as a2a3 — avoids blocking the AICPU dispatch loop).
 *
 * a5-specific: AICore reads PMU MMIO itself (via ld_dev) and writes the
 * snapshot into a per-core stable PmuAicoreRing
 * (state.aicore_ring_ptr); AICPU validates the slot and commits into
 * records[] on COND FIN. The ring address is set once by the host and
 * never reassigned, so the AICore writer is decoupled from the AICPU
 * PmuBuffer rotation.
 */

#include "aicpu/pmu_collector_aicpu.h"

#include <cstring>

#include "aicpu/platform_regs.h"
#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "common/unified_log.h"

static uint64_t g_platform_pmu_base = 0;
static bool g_enable_pmu = false;
// Device-resident `pmu_reg_addrs` (uint64_t[num_aicore]). The
// host pre-allocates the buffer in init_pmu() and stashes its
// device pointer here via set_platform_pmu_reg_addrs(); pmu_aicpu_init
// fills the entries from regs[physical_core_ids[i]] after handshake.
static uint64_t g_platform_pmu_reg_addrs = 0;

// Saved CTRL register state per core, indexed by logical core_id.
// Populated by pmu_aicpu_init(), consumed by pmu_aicpu_finalize().
static uint32_t g_pmu_saved_ctrl0[PLATFORM_MAX_CORES];
static uint32_t g_pmu_saved_ctrl1[PLATFORM_MAX_CORES];

// Per-core cached PmuBufferState pointer and PMU header pointer.
static PmuBufferState *s_pmu_buffer_states[PLATFORM_MAX_CORES];
static PmuDataHeader *s_pmu_header = nullptr;

// Per-core cached PmuAicoreRing pointer (the stable AICore staging ring).
// Populated by pmu_aicpu_init() from state->aicore_ring_ptr (host-published
// at SHM init); read by complete_record. Never reassigned during the run.
// AICPU treats these as plain pointers (no __gm__ on AICPU).
static PmuAicoreRing *s_pmu_aicore_rings[PLATFORM_MAX_CORES];

// Per-core resolved PMU MMIO base address, keyed by logical core_id.
// Populated by pmu_aicpu_init(); 0 means "no PMU for this core" (sim).
static uint64_t s_pmu_reg_addrs[PLATFORM_MAX_CORES] = {0};

extern "C" void set_platform_pmu_base(uint64_t pmu_data_base) { g_platform_pmu_base = pmu_data_base; }

extern "C" uint64_t get_platform_pmu_base() { return g_platform_pmu_base; }

extern "C" void set_pmu_enabled(bool enable) { g_enable_pmu = enable; }

extern "C" bool is_pmu_enabled() { return g_enable_pmu; }

extern "C" void set_platform_pmu_reg_addrs(uint64_t table) { g_platform_pmu_reg_addrs = table; }
extern "C" uint64_t get_platform_pmu_reg_addrs() { return g_platform_pmu_reg_addrs; }

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
// Internal: enqueue full buffer to per-thread ready_queue
// ---------------------------------------------------------------------------

static int enqueue_pmu_ready_buffer(int thread_idx, uint32_t core_index, uint64_t buffer_ptr, uint32_t buffer_seq) {
    uint32_t capacity = PLATFORM_PMU_READYQUEUE_SIZE;
    uint32_t current_tail = s_pmu_header->queue_tails[thread_idx];
    uint32_t current_head = s_pmu_header->queue_heads[thread_idx];

    uint32_t next_tail = (current_tail + 1) % capacity;
    if (next_tail == current_head) {
        return -1;  // Queue full
    }

    s_pmu_header->queues[thread_idx][current_tail].core_index = core_index;
    s_pmu_header->queues[thread_idx][current_tail].buffer_ptr = buffer_ptr;
    s_pmu_header->queues[thread_idx][current_tail].buffer_seq = buffer_seq;
    s_pmu_header->queue_tails[thread_idx] = next_tail;
    return 0;
}

// ---------------------------------------------------------------------------
// Internal: switch the current buffer for one core (called from
// complete_record when records[count] hits PLATFORM_PMU_RECORDS_PER_BUFFER)
// ---------------------------------------------------------------------------

static void pmu_switch_buffer(int core_id, int thread_idx) {
    PmuBufferState *state = s_pmu_buffer_states[core_id];
    if (state == nullptr) {
        return;
    }

    PmuBuffer *full_buf = reinterpret_cast<PmuBuffer *>(state->current_buf_ptr);
    if (full_buf == nullptr) {
        return;
    }

    // Check free_queue before committing the full buffer
    rmb();
    uint32_t head = state->free_queue.head;
    uint32_t tail = state->free_queue.tail;

    if (head == tail) {
        // No replacement buffer available — overwrite current buffer to keep AICPU alive
        LOG_WARN("Thread %d: Core %d no free PMU buffer, overwriting current buffer (data lost)", thread_idx, core_id);
        state->dropped_record_count += full_buf->count;
        full_buf->count = 0;
        wmb();
        return;
    }

    // Enqueue full buffer to ready_queue
    uint32_t seq = state->current_buf_seq;
    int rc = enqueue_pmu_ready_buffer(thread_idx, static_cast<uint32_t>(core_id), state->current_buf_ptr, seq);
    if (rc != 0) {
        LOG_ERROR(
            "Thread %d: Core %d failed to enqueue PMU buffer (ready_queue full), data lost!", thread_idx, core_id
        );
        state->dropped_record_count += full_buf->count;
        full_buf->count = 0;
        wmb();
        return;
    }

    // Pop next buffer from free_queue
    uint64_t new_buf_ptr = state->free_queue.buffer_ptrs[head % PLATFORM_PMU_SLOT_COUNT];
    rmb();
    state->free_queue.head = head + 1;
    state->current_buf_ptr = new_buf_ptr;
    state->current_buf_seq = seq + 1;
    wmb();

    PmuBuffer *new_buf = reinterpret_cast<PmuBuffer *>(new_buf_ptr);
    new_buf->count = 0;
    wmb();

    LOG_INFO_V0("Thread %d: Core %d switched to new PMU buffer (addr=0x%lx)", thread_idx, core_id, new_buf_ptr);
}

// ---------------------------------------------------------------------------
// High-level interface
// ---------------------------------------------------------------------------

void pmu_aicpu_init(const uint32_t *physical_core_ids, int num_cores) {
    void *pmu_base = reinterpret_cast<void *>(get_platform_pmu_base());
    if (pmu_base == nullptr) {
        LOG_ERROR("pmu_aicpu_init: pmu_data_base is NULL");
        return;
    }
    if (physical_core_ids == nullptr) {
        LOG_ERROR("pmu_aicpu_init: physical_core_ids is NULL");
        return;
    }
    s_pmu_header = reinterpret_cast<PmuDataHeader *>(pmu_base);
    // Read event_type from SHM header (written by host at init)
    uint32_t pmu_event_type = s_pmu_header->event_type;

    // Resolve per-core PMU MMIO base from physical_core_ids. 0 means "no PMU
    // for this core" (sim or misconfigured) — subsequent record/stop become no-ops.
    uint64_t *regs_array = reinterpret_cast<uint64_t *>(get_platform_regs());
    uint64_t *aicore_pmu_reg_table = reinterpret_cast<uint64_t *>(g_platform_pmu_reg_addrs);
    for (int i = 0; i < num_cores; i++) {
        if (i >= PLATFORM_MAX_CORES) {
            LOG_ERROR("pmu_aicpu_init: num_cores %d exceeds PLATFORM_MAX_CORES %d", num_cores, PLATFORM_MAX_CORES);
            break;
        }
        uint64_t reg_addr = regs_array ? regs_array[physical_core_ids[i]] : 0;
        s_pmu_reg_addrs[i] = reg_addr;
        g_pmu_saved_ctrl0[i] = 0;
        g_pmu_saved_ctrl1[i] = 0;
        // Publish the resolved PMU reg base into the AICore-visible table so
        // the AICore-side `get_aicore_pmu_reg_base()` lazy getter resolves
        // the correct value. The table lives in device memory and is also
        // visible to AICore via KernelArgs::pmu_reg_addrs.
        if (aicore_pmu_reg_table != nullptr) {
            aicore_pmu_reg_table[i] = reg_addr;
        }
    }
    wmb();

    // Program event selectors and start PMU counters
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

    // Pop initial PmuBuffer from each core's free_queue and cache the stable
    // AICore staging ring pointer.
    for (int i = 0; i < num_cores; i++) {
        PmuBufferState *state = get_pmu_buffer_state(pmu_base, i);
        s_pmu_buffer_states[i] = state;
        s_pmu_aicore_rings[i] = reinterpret_cast<PmuAicoreRing *>(state->aicore_ring_ptr);

        rmb();
        uint32_t head = state->free_queue.head;
        uint32_t tail = state->free_queue.tail;

        if (head != tail) {
            uint64_t buf_ptr = state->free_queue.buffer_ptrs[head % PLATFORM_PMU_SLOT_COUNT];
            rmb();
            state->free_queue.head = head + 1;
            state->current_buf_ptr = buf_ptr;
            state->current_buf_seq = 0;
            wmb();

            PmuBuffer *buf = reinterpret_cast<PmuBuffer *>(buf_ptr);
            buf->count = 0;

            LOG_DEBUG("Core %d: popped initial PMU buffer (addr=0x%lx)", i, buf_ptr);
        } else {
            LOG_ERROR("Core %d: PMU free_queue is empty during init!", i);
            state->current_buf_ptr = 0;
        }
    }

    LOG_INFO_V0("PMU initialized: %d cores, event_type=%u", num_cores, pmu_event_type);
}

void pmu_aicpu_complete_record(
    int core_id, int thread_idx, uint32_t reg_task_id, uint64_t task_id, uint32_t func_id, CoreType core_type
) {
    if (s_pmu_header == nullptr || core_id < 0 || core_id >= PLATFORM_MAX_CORES) {
        return;
    }

    PmuBufferState *state = s_pmu_buffer_states[core_id];
    if (state == nullptr) {
        return;
    }

    // Account the task *before* any drop path so total reflects every task the
    // AICPU tried to record. collected + dropped + mismatch == total invariant on host.
    state->total_record_count += 1;

    // Read AICore-published PmuRecord from the stable per-core staging ring.
    PmuAicoreRing *ring = s_pmu_aicore_rings[core_id];
    if (ring == nullptr) {
        state->dropped_record_count += 1;
        wmb();
        return;
    }
    PmuRecord *slot = &ring->dual_issue_slots[reg_task_id % PLATFORM_PMU_AICORE_RING_SIZE];
    cache_invalidate_range(slot, sizeof(PmuRecord));

    if (static_cast<uint32_t>(slot->task_id) != reg_task_id) {
        // AICore hasn't published this slot yet — hard invariant violation,
        // separate from capacity drops.
        state->mismatch_record_count += 1;
        wmb();
        return;
    }

    rmb();
    uint64_t cur_ptr = state->current_buf_ptr;
    if (cur_ptr == 0) {
        state->dropped_record_count += 1;
        wmb();
        return;
    }
    PmuBuffer *buf = reinterpret_cast<PmuBuffer *>(cur_ptr);

    // Switch buffer if full (internal — ring address is unchanged)
    if (buf->count >= static_cast<uint32_t>(PLATFORM_PMU_RECORDS_PER_BUFFER)) {
        pmu_switch_buffer(core_id, thread_idx);
        rmb();
        cur_ptr = state->current_buf_ptr;
        if (cur_ptr == 0) {
            state->dropped_record_count += 1;
            wmb();
            return;
        }
        buf = reinterpret_cast<PmuBuffer *>(cur_ptr);
    }

    uint32_t idx = buf->count;

    PmuRecord *rec = &buf->records[idx];
    rec->task_id = task_id;
    rec->func_id = func_id;
    rec->core_type = core_type;
    rec->pmu_total_cycles = slot->pmu_total_cycles;
    for (int i = 0; i < PMU_COUNTER_COUNT_A5; i++) {
        rec->pmu_counters[i] = slot->pmu_counters[i];
    }
    buf->count = idx + 1;
    wmb();
}

void pmu_aicpu_flush_buffers(int thread_idx, const int *cur_thread_cores, int core_num) {
    if (s_pmu_header == nullptr) {
        LOG_ERROR("pmu_aicpu_flush_buffers: PMU not initialized (s_pmu_header=NULL), thread %d", thread_idx);
        return;
    }

    for (int i = 0; i < core_num; i++) {
        int core_id = cur_thread_cores[i];
        if (core_id < 0 || core_id >= PLATFORM_MAX_CORES) {
            LOG_ERROR(
                "pmu_aicpu_flush_buffers: thread %d got invalid core_id %d (max %d)", thread_idx, core_id,
                PLATFORM_MAX_CORES
            );
            continue;
        }

        PmuBufferState *state = s_pmu_buffer_states[core_id];
        if (state == nullptr) {
            LOG_WARN(
                "pmu_aicpu_flush_buffers: thread %d core %d has no PmuBufferState (skipped during init?)", thread_idx,
                core_id
            );
            continue;
        }

        rmb();
        uint64_t buf_ptr = state->current_buf_ptr;
        if (buf_ptr == 0) {
            // No active buffer — either never allocated or already flushed. Not an error.
            continue;
        }

        PmuBuffer *buf = reinterpret_cast<PmuBuffer *>(buf_ptr);
        if (buf->count == 0) {
            // Active buffer but empty — nothing to flush.
            continue;
        }

        uint32_t seq = state->current_buf_seq;
        int rc = enqueue_pmu_ready_buffer(thread_idx, static_cast<uint32_t>(core_id), buf_ptr, seq);
        if (rc == 0) {
            LOG_INFO_V0("Thread %d: Core %d flushed PMU buffer with %u records", thread_idx, core_id, buf->count);
            state->current_buf_ptr = 0;
            wmb();
        } else {
            // ready_queue full at end-of-run: account the loss and clear the
            // buffer so host reconcile sees a clean state (current_buf_ptr=0)
            // and dropped == flush failures rather than silent leftover.
            LOG_ERROR(
                "Thread %d: Core %d failed to flush PMU buffer (ready_queue full), %u records lost!", thread_idx,
                core_id, buf->count
            );
            state->dropped_record_count += buf->count;
            buf->count = 0;
            state->current_buf_ptr = 0;
            wmb();
        }
    }
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
