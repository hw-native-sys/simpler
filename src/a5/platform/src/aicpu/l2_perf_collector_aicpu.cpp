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
 * @file l2_perf_collector_aicpu.cpp
 * @brief AICPU performance data collection implementation (memcpy-based)
 *
 * Host pre-allocates one L2PerfBuffer per core and one PhaseBuffer per thread
 * on the device. AICPU writes records directly into them via cached pointers.
 * When a buffer fills up, subsequent records are silently dropped — there is
 * no buffer switching or flushing.
 */

#include "aicpu/l2_perf_collector_aicpu.h"

#include <cinttypes>
#include <cstring>

#include "aicpu/platform_regs.h"
#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "common/unified_log.h"

// Cached pointers for hot-path access (set during init)
static L2PerfSetupHeader *s_setup_header = nullptr;

// Per-thread PhaseBuffer cache
static PhaseBuffer *s_current_phase_buf[PLATFORM_MAX_AICPU_THREADS] = {};

static int s_orch_thread_idx = -1;

// L2 perf platform state. Published by the host (via dlsym'd setters on sim)
// or by the AICPU kernel entry (onboard) before perf init runs, so downstream
// perf code can discover enablement + device-base without reading the generic
// Runtime struct.
static uint64_t g_platform_l2_perf_base = 0;
static bool g_enable_l2_swimlane = false;

extern "C" void set_platform_l2_perf_base(uint64_t l2_perf_data_base) { g_platform_l2_perf_base = l2_perf_data_base; }
extern "C" uint64_t get_platform_l2_perf_base() { return g_platform_l2_perf_base; }
extern "C" void set_l2_swimlane_enabled(bool enable) { g_enable_l2_swimlane = enable; }
extern "C" bool is_l2_swimlane_enabled() { return g_enable_l2_swimlane; }

void l2_perf_aicpu_init_profiling(Runtime *runtime) {
    void *l2_perf_base = reinterpret_cast<void *>(g_platform_l2_perf_base);
    if (l2_perf_base == nullptr) {
        LOG_ERROR("l2_perf_data_base is NULL, cannot initialize profiling");
        return;
    }

    s_setup_header = get_perf_setup_header(l2_perf_base);

    int32_t task_count = runtime->get_task_count();
    s_setup_header->total_tasks = static_cast<uint32_t>(task_count);

    LOG_INFO("Initializing performance profiling for %d cores (memcpy-based)", runtime->worker_count);

    // Initialize each core's L2PerfBuffer and publish the pointer to the handshake
    for (int i = 0; i < runtime->worker_count; i++) {
        Handshake *h = &runtime->workers[i];
        uint64_t buf_ptr = s_setup_header->core_buffer_ptrs[i];

        if (buf_ptr == 0) {
            LOG_ERROR("Core %d: core_buffer_ptrs[%d] is NULL during init!", i, i);
            h->l2_perf_records_addr = 0;
            continue;
        }

        L2PerfBuffer *buf = reinterpret_cast<L2PerfBuffer *>(buf_ptr);
        buf->count = 0;
        h->l2_perf_records_addr = buf_ptr;

        LOG_DEBUG("Core %d: L2PerfBuffer at 0x%lx", i, buf_ptr);
    }

    wmb();

    LOG_INFO("Performance profiling initialized for %d cores", runtime->worker_count);
}

int l2_perf_aicpu_complete_record(
    L2PerfBuffer *l2_perf_buf, uint32_t expected_reg_task_id, uint64_t task_id, uint32_t func_id, CoreType core_type,
    uint64_t dispatch_time, uint64_t finish_time, const uint64_t *fanout, int32_t fanout_count
) {
    rmb();
    uint32_t count = l2_perf_buf->count;
    // Buffer-full check lives here (AICore does not branch on capacity); return -1
    // silently drops the record, caller ignores the failure.
    if (count >= PLATFORM_PROF_BUFFER_SIZE) return -1;

    // Read from WIP staging slot (AICore writes here, parity = reg_task_id & 1)
    L2PerfRecord *wip = &l2_perf_buf->wip[expected_reg_task_id & 1u];
    // One PoC cache line: matches AICore l2_perf_aicore_record_task() dcci(..., SINGLE_CACHE_LINE, ...)
    // and aicpu/cache_ops.cpp step size; wip timing fields live in the first line.
    cache_invalidate_range(wip, 64);
    if (static_cast<uint32_t>(wip->task_id) != expected_reg_task_id) return -1;

    // Copy AICore timing to committed record slot
    L2PerfRecord *record = &l2_perf_buf->records[count];
    record->start_time = wip->start_time;
    record->end_time = wip->end_time;

    // Fill AICPU-owned fields
    record->task_id = task_id;
    record->func_id = func_id;
    record->core_type = core_type;
    record->dispatch_time = dispatch_time;
    record->finish_time = finish_time;

    if (fanout != nullptr && fanout_count > 0) {
        int32_t n = (fanout_count > RUNTIME_MAX_FANOUT) ? RUNTIME_MAX_FANOUT : fanout_count;
        for (int32_t i = 0; i < n; i++) {
            record->fanout[i] = fanout[i];
        }
        record->fanout_count = n;
    } else {
        record->fanout_count = 0;
    }

    l2_perf_buf->count = count + 1;
    wmb();
    return 0;
}

void l2_perf_aicpu_update_total_tasks(uint32_t total_tasks) {
    void *l2_perf_base = reinterpret_cast<void *>(g_platform_l2_perf_base);
    if (l2_perf_base == nullptr) {
        return;
    }

    L2PerfSetupHeader *header = get_perf_setup_header(l2_perf_base);
    header->total_tasks = total_tasks;
    wmb();
}

void l2_perf_aicpu_init_phase_profiling(int num_sched_threads) {
    void *l2_perf_base = reinterpret_cast<void *>(g_platform_l2_perf_base);
    if (l2_perf_base == nullptr) {
        LOG_ERROR("l2_perf_data_base is NULL, cannot initialize phase profiling");
        return;
    }

    s_setup_header = get_perf_setup_header(l2_perf_base);

    AicpuPhaseHeader *phase_header = &s_setup_header->phase_header;
    phase_header->magic = AICPU_PHASE_MAGIC;
    phase_header->num_sched_threads = num_sched_threads;
    phase_header->records_per_thread = PLATFORM_PHASE_RECORDS_PER_THREAD;
    phase_header->num_cores = 0;

    memset(phase_header->core_to_thread, -1, sizeof(phase_header->core_to_thread));
    memset(&phase_header->orch_summary, 0, sizeof(AicpuOrchSummary));

    // Cache per-thread PhaseBuffer pointers. Include all threads: scheduler +
    // orchestrator (orchestrator may become scheduler).
    int total_threads = num_sched_threads + 1;
    if (total_threads > PLATFORM_MAX_AICPU_THREADS) {
        total_threads = PLATFORM_MAX_AICPU_THREADS;
    }
    for (int t = 0; t < total_threads; t++) {
        uint64_t buf_ptr = s_setup_header->phase_buffer_ptrs[t];
        if (buf_ptr == 0) {
            LOG_ERROR("Thread %d: phase_buffer_ptrs[%d] is NULL during init!", t, t);
            s_current_phase_buf[t] = nullptr;
            continue;
        }

        PhaseBuffer *buf = reinterpret_cast<PhaseBuffer *>(buf_ptr);
        buf->count = 0;
        s_current_phase_buf[t] = buf;

        LOG_DEBUG("Thread %d: PhaseBuffer at 0x%lx", t, buf_ptr);
    }

    // Clear remaining slots
    for (int t = total_threads; t < PLATFORM_MAX_AICPU_THREADS; t++) {
        s_current_phase_buf[t] = nullptr;
    }

    wmb();

    LOG_INFO(
        "Phase profiling initialized: %d scheduler + 1 orch thread, %d records/thread", num_sched_threads,
        PLATFORM_PHASE_RECORDS_PER_THREAD
    );
}

void l2_perf_aicpu_record_phase(
    int thread_idx, AicpuPhaseId phase_id, uint64_t start_time, uint64_t end_time, uint32_t loop_iter,
    uint64_t tasks_processed
) {
    if (s_setup_header == nullptr) {
        return;
    }

    PhaseBuffer *buf = s_current_phase_buf[thread_idx];
    if (buf == nullptr) return;

    uint32_t idx = buf->count;
    if (idx >= PLATFORM_PHASE_RECORDS_PER_THREAD) {
        // Buffer full; silently drop.
        return;
    }

    AicpuPhaseRecord *record = &buf->records[idx];
    record->start_time = start_time;
    record->end_time = end_time;
    record->loop_iter = loop_iter;
    record->phase_id = phase_id;
    record->task_id = tasks_processed;

    buf->count = idx + 1;
}

void l2_perf_aicpu_write_orch_summary(const AicpuOrchSummary *src) {
    if (s_setup_header == nullptr) {
        return;
    }

    AicpuOrchSummary *dst = &s_setup_header->phase_header.orch_summary;

    memcpy(dst, src, sizeof(AicpuOrchSummary));
    dst->magic = AICPU_PHASE_MAGIC;
    dst->padding = 0;

    wmb();

    LOG_INFO(
        "Orchestrator summary written: %" PRId64 " tasks, %.3fus", static_cast<int64_t>(src->submit_count),
        cycles_to_us(src->end_time - src->start_time)
    );
}

void l2_perf_aicpu_set_orch_thread_idx(int thread_idx) { s_orch_thread_idx = thread_idx; }

void l2_perf_aicpu_record_orch_phase(
    AicpuPhaseId phase_id, uint64_t start_time, uint64_t end_time, uint32_t submit_idx, uint64_t task_id
) {
    if (s_orch_thread_idx < 0 || s_setup_header == nullptr) return;
    l2_perf_aicpu_record_phase(s_orch_thread_idx, phase_id, start_time, end_time, submit_idx, task_id);
}

void l2_perf_aicpu_init_core_assignments(int total_cores) {
    if (s_setup_header == nullptr) {
        return;
    }
    AicpuPhaseHeader *phase_header = &s_setup_header->phase_header;
    memset(phase_header->core_to_thread, -1, sizeof(phase_header->core_to_thread));
    phase_header->num_cores = static_cast<uint32_t>(total_cores);
    wmb();
    LOG_INFO("Core-to-thread mapping init: %d cores", total_cores);
}

void l2_perf_aicpu_write_core_assignments_for_thread(int thread_idx, const int *core_ids, int core_num) {
    if (s_setup_header == nullptr) {
        return;
    }
    AicpuPhaseHeader *phase_header = &s_setup_header->phase_header;
    for (int i = 0; i < core_num; i++) {
        int core_id = core_ids[i];
        if (core_id >= 0 && core_id < PLATFORM_MAX_CORES) {
            phase_header->core_to_thread[core_id] = static_cast<int8_t>(thread_idx);
        }
    }
    wmb();
}
