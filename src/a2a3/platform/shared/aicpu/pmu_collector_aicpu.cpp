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
 * @brief AICPU-side AICore PMU counter collection implementation
 *
 * Uses read_reg/write_reg from platform_regs for MMIO register access,
 * consistent with the rest of the platform layer.
 *
 * Buffer switching:
 *   - SPSC free_queue: Host pushes free PmuBuffers, AICPU pops when switching.
 *   - Per-thread ready_queue: AICPU enqueues full buffers for host collection.
 *   - Full buffers are published before AICPU tries to recover a replacement.
 *     If recovery is delayed, later records are counted as dropped until host
 *     replenishes free_queue.
 */

#include "aicpu/pmu_collector_aicpu.h"

#include <cstring>

#include "aicpu/platform_regs.h"
#include "aicpu/profiler_device_engine.h"
#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "common/unified_log.h"

static uint64_t g_platform_pmu_base = 0;
static bool g_enable_pmu = false;

// Saved CTRL register state per core, indexed by logical core_id.
// Populated by pmu_aicpu_init(), consumed by pmu_aicpu_finalize().
static uint32_t g_pmu_saved_ctrl0[PLATFORM_MAX_CORES];

// Per-core cached PmuBufferState pointer and current buffer pointer.
static PmuBufferState *s_pmu_buffer_states[PLATFORM_MAX_CORES];
static PmuDataHeader *s_pmu_header = nullptr;

// Per-core resolved PMU MMIO base address, keyed by logical core_id.
// Populated by pmu_aicpu_init(); 0 means "no PMU for this core" (sim).
static uint64_t s_pmu_reg_addrs[PLATFORM_MAX_CORES] = {0};

static constexpr uint64_t kPmuQueueBackpressureWaitCycles = PLATFORM_DFX_BACKPRESSURE_TIMEOUT_CYCLES;

extern "C" void set_platform_pmu_base(uint64_t pmu_data_base) { g_platform_pmu_base = pmu_data_base; }

extern "C" uint64_t get_platform_pmu_base() { return g_platform_pmu_base; }

extern "C" void set_pmu_enabled(bool enable) { g_enable_pmu = enable; }

extern "C" bool is_pmu_enabled() { return g_enable_pmu; }

// ---------------------------------------------------------------------------
// Low-level MMIO helpers (internal use only)
// ---------------------------------------------------------------------------

static void pmu_program_events(uint64_t reg_base, const PmuEventConfig &events) {
    for (int i = 0; i < PMU_COUNTER_COUNT_A2A3; i++) {
        write_reg(reg_base, reg_index(RegId::PMU_CNT0_IDX, i), events.event_ids[i]);
    }
}

static uint32_t pmu_start(uint64_t reg_base) {
    // Clear counters by reading them once
    for (int i = 0; i < PMU_COUNTER_COUNT_A2A3; i++) {
        (void)read_reg(reg_base, reg_index(RegId::PMU_CNT0, i));
    }
    (void)read_reg(reg_base, RegId::PMU_CNT_TOTAL0);
    (void)read_reg(reg_base, RegId::PMU_CNT_TOTAL1);

    // Set full cycle counting range: start at 0, stop at 0xFFFFFFFF
    write_reg(reg_base, RegId::PMU_START_CYC0, 0x0);
    write_reg(reg_base, RegId::PMU_START_CYC1, 0x0);
    write_reg(reg_base, RegId::PMU_STOP_CYC0, 0xFFFFFFFF);
    write_reg(reg_base, RegId::PMU_STOP_CYC1, 0xFFFFFFFF);

    // Save and set CTRL_0
    uint32_t saved_ctrl0 = static_cast<uint32_t>(read_reg(reg_base, RegId::PMU_CTRL_0));
    write_reg(reg_base, RegId::PMU_CTRL_0, REG_MMIO_PMU_CTRL_0_ENABLE_VAL);
    return saved_ctrl0;
}

static void pmu_stop(uint64_t reg_base, uint32_t saved_ctrl0) { write_reg(reg_base, RegId::PMU_CTRL_0, saved_ctrl0); }

static void pmu_read_counters(uint64_t reg_base, PmuRecord *out) {
    for (int i = 0; i < PMU_COUNTER_COUNT_A2A3; i++) {
        out->pmu_counters[i] = static_cast<uint32_t>(read_reg(reg_base, reg_index(RegId::PMU_CNT0, i)));
    }
    uint64_t lo = read_reg(reg_base, RegId::PMU_CNT_TOTAL0);
    uint64_t hi = read_reg(reg_base, RegId::PMU_CNT_TOTAL1);
    out->pmu_total_cycles = lo | (hi << 32);
}

// ---------------------------------------------------------------------------
// Internal: enqueue full buffer to per-thread ready_queue
// ---------------------------------------------------------------------------

struct PmuDeviceModule {
    struct Context {
        PmuDataHeader *header;
        int thread_idx;
        int core_id;
    };

    using DataHeader = PmuDataHeader;
    using State = PmuBufferState;
    using FreeQueue = PmuFreeQueue;
    using Buffer = PmuBuffer;

    static constexpr uint32_t kReadyQueueSize = PLATFORM_PMU_READYQUEUE_SIZE;
    static constexpr uint32_t kSlotCount = PLATFORM_PMU_SLOT_COUNT;
    static constexpr uint64_t kBackpressureWaitCycles = kPmuQueueBackpressureWaitCycles;

    static DataHeader *header(Context ctx) { return ctx.header; }
    static int ready_thread(Context ctx) { return ctx.thread_idx; }
    static FreeQueue *free_queue(State *state) { return &state->free_queue; }

    static uint64_t current_ptr(State *state) { return state->current_buf_ptr; }
    static void set_current_ptr(State *state, uint64_t ptr) { state->current_buf_ptr = ptr; }
    static uint32_t current_seq(State *state) { return state->current_buf_seq; }
    static void set_current_seq(State *state, uint32_t seq) { state->current_buf_seq = seq; }

    static uint32_t count(Buffer *buffer) { return buffer->count; }
    static void set_count(Buffer *buffer, uint32_t count) { buffer->count = count; }

    static void write_ready_entry(Context ctx, uint32_t tail, uint64_t buffer_ptr, uint32_t buffer_seq) {
        ctx.header->queues[ctx.thread_idx][tail].core_index = static_cast<uint32_t>(ctx.core_id);
        ctx.header->queues[ctx.thread_idx][tail].buffer_ptr = buffer_ptr;
        ctx.header->queues[ctx.thread_idx][tail].buffer_seq = buffer_seq;
    }

    static void account_dropped(Context, State *state, uint32_t count) { state->dropped_record_count += count; }
    static void on_pop_success(Context, State *, Buffer *) {}
    static void on_current_cleared(Context, State *) {}
    static void on_no_replacement(Context, State *) {}
    static void on_null_free_slot(Context, State *) {}
    static void on_enqueue_failed(Context ctx, State *, Buffer *) {
        LOG_ERROR(
            "Thread %d: Core %d failed to enqueue PMU buffer (ready_queue full), data lost!", ctx.thread_idx,
            ctx.core_id
        );
    }
    static void on_switch_complete(Context ctx, State *, Buffer *buffer) {
        if (buffer != nullptr) {
            LOG_DEBUG(
                "Thread %d: Core %d switched to new PMU buffer (addr=0x%lx)", ctx.thread_idx, ctx.core_id,
                reinterpret_cast<uint64_t>(buffer)
            );
        }
    }
};

using PmuEngine = profiling_device::DeviceProfilerEngine<PmuDeviceModule>;

static PmuDeviceModule::Context pmu_context(int core_id, int thread_idx) {
    return PmuDeviceModule::Context{s_pmu_header, thread_idx, core_id};
}

static int enqueue_pmu_ready_buffer(int thread_idx, uint32_t core_index, uint64_t buffer_ptr, uint32_t buffer_seq) {
    return PmuEngine::enqueue_ready(pmu_context(static_cast<int>(core_index), thread_idx), buffer_ptr, buffer_seq);
}

static PmuBuffer *try_pop_pmu_buffer(int core_id, PmuBufferState *state, uint32_t next_seq) {
    return PmuEngine::pop_free(pmu_context(core_id, /*thread_idx=*/0), state, next_seq);
}

// ---------------------------------------------------------------------------
// Internal: switch the current buffer for one core
// ---------------------------------------------------------------------------

static void pmu_switch_buffer(int core_id, int thread_idx) {
    PmuBufferState *state = s_pmu_buffer_states[core_id];
    if (state == nullptr) {
        return;
    }

    PmuEngine::switch_buffer(pmu_context(core_id, thread_idx), state);
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

    s_pmu_header = get_pmu_header(pmu_base);

    // Read event_type from SHM header (written by host at init)
    uint32_t pmu_event_type = s_pmu_header->event_type;

    // Resolve per-core PMU MMIO base from physical_core_ids. 0 means "no PMU
    // for this core" (sim or misconfigured) — subsequent record/stop become no-ops.
    uint64_t *pmu_regs_array = reinterpret_cast<uint64_t *>(get_platform_pmu_reg_addrs());
    for (int i = 0; i < num_cores; i++) {
        if (i >= PLATFORM_MAX_CORES) {
            LOG_ERROR("pmu_aicpu_init: num_cores %d exceeds PLATFORM_MAX_CORES %d", num_cores, PLATFORM_MAX_CORES);
            break;
        }
        s_pmu_reg_addrs[i] = pmu_regs_array ? pmu_regs_array[physical_core_ids[i]] : 0;
    }

    // Program event selectors and start PMU counters on all cores
    const PmuEventConfig *evt = pmu_resolve_event_config_a2a3(static_cast<PmuEventType>(pmu_event_type));
    if (evt == nullptr) {
        evt = &PMU_EVENTS_A2A3_PIPE_UTIL;
    }
    for (int i = 0; i < num_cores; i++) {
        uint64_t reg_addr = s_pmu_reg_addrs[i];
        if (reg_addr == 0) {
            LOG_WARN("pmu_aicpu_init: core %d has no PMU reg_addr, skipping (sim or misconfigured)", i);
            continue;
        }
        pmu_program_events(reg_addr, *evt);
        g_pmu_saved_ctrl0[i] = pmu_start(reg_addr);
    }

    // Pop initial PmuBuffer from each core's free_queue
    for (int i = 0; i < num_cores; i++) {
        PmuBufferState *state = get_pmu_buffer_state(pmu_base, i);
        s_pmu_buffer_states[i] = state;

        rmb();
        uint32_t head = state->free_queue.head;
        uint32_t tail = state->free_queue.tail;

        if (head != tail) {
            (void)try_pop_pmu_buffer(i, state, 0);
        } else {
            LOG_ERROR("Core %d: PMU free_queue is empty during init!", i);
            state->current_buf_ptr = 0;
        }
    }

    wmb();
}

void pmu_aicpu_record_task(int core_id, int thread_idx, uint64_t task_id, uint32_t func_id, CoreType core_type) {
    if (s_pmu_header == nullptr || core_id < 0 || core_id >= PLATFORM_MAX_CORES) {
        return;
    }
    uint64_t reg_addr = s_pmu_reg_addrs[core_id];
    if (reg_addr == 0) {
        return;
    }

    PmuBufferState *state = s_pmu_buffer_states[core_id];
    if (state == nullptr) {
        return;
    }

    // Account the task *before* any drop path so total reflects every task the
    // AICPU tried to record. total == collected + dropped invariant on host.
    state->total_record_count += 1;

    rmb();
    uint64_t cur_ptr = state->current_buf_ptr;
    PmuBuffer *pmu_buf = nullptr;
    if (cur_ptr == 0) {
        pmu_buf = try_pop_pmu_buffer(core_id, state, state->current_buf_seq);
        if (pmu_buf == nullptr) {
            state->dropped_record_count += 1;
            wmb();
            return;
        }
        cur_ptr = state->current_buf_ptr;
    } else {
        pmu_buf = reinterpret_cast<PmuBuffer *>(cur_ptr);
    }

    // Switch buffer if full
    if (pmu_buf->count >= static_cast<uint32_t>(PLATFORM_PMU_RECORDS_PER_BUFFER)) {
        pmu_switch_buffer(core_id, thread_idx);
        rmb();
        cur_ptr = state->current_buf_ptr;
        if (cur_ptr == 0) {
            pmu_buf = try_pop_pmu_buffer(core_id, state, state->current_buf_seq);
            if (pmu_buf == nullptr) {
                state->dropped_record_count += 1;
                wmb();
                return;
            }
            cur_ptr = state->current_buf_ptr;
        } else {
            pmu_buf = reinterpret_cast<PmuBuffer *>(cur_ptr);
        }
    }

    uint32_t idx = pmu_buf->count;
    PmuRecord *rec = &pmu_buf->records[idx];
    rec->task_id = task_id;
    rec->func_id = func_id;
    rec->core_type = core_type;
    pmu_read_counters(reg_addr, rec);
    pmu_buf->count = idx + 1;
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
            state->current_buf_ptr = 0;
            wmb();
        } else {
            // ready_queue full at end-of-run: account the loss and clear the
            // buffer so host reconcile sees a clean state (current_buf_ptr=0)
            // and dropped == flush failures rather than silent wip-mismatch.
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
            pmu_stop(reg_addr, g_pmu_saved_ctrl0[core_id]);
        }
    }
}
