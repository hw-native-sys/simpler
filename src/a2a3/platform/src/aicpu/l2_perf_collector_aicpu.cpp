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
 * @brief AICPU performance data collection implementation (SPSC free queue)
 *
 * Uses per-core L2PerfBufferState with SPSC free queues for O(1) buffer switching.
 * Host memory manager dynamically allocates replacement buffers and pushes
 * them into the free_queue. Device pops from free_queue when switching.
 */

#include "aicpu/l2_perf_collector_aicpu.h"

#include <cinttypes>
#include <cstring>

#include "aicpu/platform_regs.h"
#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "common/unified_log.h"

// Cached pointers for hot-path access (set during init)
static AicpuPhaseHeader *s_phase_header = nullptr;
static L2PerfDataHeader *s_l2_perf_header = nullptr;

// Per-core L2PerfBufferState cache
static L2PerfBufferState *s_perf_buffer_states[PLATFORM_MAX_CORES] = {};

// Per-core L2PerfAicoreBufferState cache (lives in the same shared region;
// host writes initial pool + the rotation channel that AICore polls).
//
// All AICore-side bookkeeping (rotation channel, free queue,
// total_record_count, current_buf_seq) is owned by this shared struct — see
// l2_perf_profiling.h. We deliberately do not keep AICPU-process-local
// mirror counters because the struct's volatile fields are the single
// source of truth across init/complete/rotate/flush. The high-water-mark
// formula `total_record_count - current_buf_seq * BUFFER_SIZE` correctly
// handles the failed-rotation case (free_queue empty or ready_queue full)
// since current_buf_seq only bumps on a successful rotation.
static L2PerfAicoreBufferState *s_aicore_buffer_states[PLATFORM_MAX_CORES] = {};

// Per-core cached current-records-buffer pointer. Written by AICPU when
// rotating buffers from inside `complete_record`. AICore writes to its own
// per-core L2PerfAicoreBuffer (host-allocated, AICPU rotates) and AICPU
// never reads from it on the hot path.
static L2PerfBuffer *s_perf_records_buffers[PLATFORM_MAX_CORES] = {};

// Per-thread PhaseBufferState cache
static PhaseBufferState *s_phase_buffer_states[PLATFORM_MAX_AICPU_THREADS] = {};
static PhaseBuffer *s_current_phase_buf[PLATFORM_MAX_AICPU_THREADS] = {};

static int s_orch_thread_idx = -1;

// L2 perf platform state. Published by the host (via dlsym'd setters on sim)
// or by the AICPU kernel entry (onboard) before perf init runs, so downstream
// perf code can discover enablement + device-base without reading the generic
// Runtime struct. Two channels (mirrors PMU):
//   - g_enable_l2_swimlane (bool) — set at kernel entry from the bitmask bit
//   - g_l2_perf_level (L2PerfLevel) — promoted in
//     l2_perf_aicpu_init from the shared-memory header so
//     `>= AICPU_TIMING / SCHED_PHASES / ORCH_PHASES` gates have the granular
//     value (exposed via get_l2_perf_level()).
static uint64_t g_platform_l2_perf_base = 0;
static bool g_enable_l2_swimlane = false;
static L2PerfLevel g_l2_perf_level = L2PerfLevel::DISABLED;

// AICore rotation-table device pointer (= KernelArgs::aicore_ring_addr).
// Published by the host (sim: dlsym'd setter; onboard: from k_args via the
// kernel entry); AICPU init walks it to fill per-core &rotation addresses.
static uint64_t g_platform_aicore_rotation_table = 0;

extern "C" void set_platform_l2_perf_base(uint64_t l2_perf_data_base) { g_platform_l2_perf_base = l2_perf_data_base; }
extern "C" uint64_t get_platform_l2_perf_base() { return g_platform_l2_perf_base; }
extern "C" void set_l2_swimlane_enabled(bool enable) { g_enable_l2_swimlane = enable; }
extern "C" bool is_l2_swimlane_enabled() { return g_enable_l2_swimlane; }
extern "C" void set_platform_aicore_rotation_table(uint64_t table_addr) {
    g_platform_aicore_rotation_table = table_addr;
}
extern "C" uint64_t get_platform_aicore_rotation_table() { return g_platform_aicore_rotation_table; }
L2PerfLevel get_l2_perf_level() { return g_l2_perf_level; }

/**
 * Enqueue ready buffer to per-thread queue
 *
 * @param header L2PerfDataHeader pointer
 * @param thread_idx Thread index
 * @param core_index Core index (or thread_idx for phase entries)
 * @param buffer_ptr Device pointer to the full buffer
 * @param buffer_seq Sequence number for ordering
 * @param is_phase 0 = L2PerfRecord, 1 = Phase
 * @return 0 on success, -1 if queue full
 */
static int enqueue_ready_buffer(
    L2PerfDataHeader *header, int thread_idx, uint32_t core_index, uint64_t buffer_ptr, uint32_t buffer_seq,
    uint32_t is_phase
) {
    uint32_t capacity = PLATFORM_PROF_READYQUEUE_SIZE;
    uint32_t current_tail = header->queue_tails[thread_idx];
    uint32_t current_head = header->queue_heads[thread_idx];

    // Check if queue is full
    uint32_t next_tail = (current_tail + 1) % capacity;
    if (next_tail == current_head) {
        return -1;
    }

    header->queues[thread_idx][current_tail].core_index = core_index;
    header->queues[thread_idx][current_tail].is_phase = is_phase;
    header->queues[thread_idx][current_tail].buffer_ptr = buffer_ptr;
    header->queues[thread_idx][current_tail].buffer_seq = buffer_seq;
    header->queue_tails[thread_idx] = next_tail;

    return 0;
}

void l2_perf_aicpu_init(int worker_count) {
    void *l2_perf_base = reinterpret_cast<void *>(g_platform_l2_perf_base);
    if (l2_perf_base == nullptr) {
        LOG_ERROR("l2_perf_data_base is NULL, cannot initialize profiling");
        return;
    }

    s_l2_perf_header = get_l2_perf_header(l2_perf_base);

    // Read the granular perf_level from the shared-memory header (host wrote
    // it in L2PerfCollector::initialize). The kernel-entry setter only seeded
    // the binary g_enable_l2_swimlane via the bitmask bit.
    g_l2_perf_level = static_cast<L2PerfLevel>(s_l2_perf_header->l2_perf_level);

    LOG_INFO_V0(
        "Initializing performance profiling for %d cores (free queue), l2_perf_level=%u", worker_count,
        static_cast<uint32_t>(g_l2_perf_level)
    );

    // Populate the per-core AicoreRotation device-address table. AICore reads
    // `aicore_ring_addr[block_idx]` from KernelArgs to find its rotation
    // channel; the table itself is host-allocated, but the entries are
    // device-internal addresses (`&ac_state->rotation`) that the host would
    // otherwise have to translate from host-mapped to device-mapped. AICPU
    // already runs on the device, so it can write the addresses directly
    // without any translation — that keeps the host side decoupled from the
    // AICore shared-memory layout.
    uint64_t *rotation_table = reinterpret_cast<uint64_t *>(g_platform_aicore_rotation_table);

    // Pop first buffer from free_queue for each core
    for (int i = 0; i < worker_count; i++) {
        L2PerfBufferState *state = get_perf_buffer_state(l2_perf_base, i);
        L2PerfAicoreBufferState *ac_state = get_aicore_buffer_state(l2_perf_base, worker_count, i);

        s_perf_buffer_states[i] = state;
        s_aicore_buffer_states[i] = ac_state;

        if (rotation_table != nullptr) {
            rotation_table[i] = reinterpret_cast<uint64_t>(&ac_state->rotation);
        }

        // Pop first buffer from free_queue
        rmb();
        uint32_t head = state->free_queue.head;
        uint32_t tail = state->free_queue.tail;

        if (head != tail) {
            uint64_t buf_ptr = state->free_queue.buffer_ptrs[head % PLATFORM_PROF_SLOT_COUNT];
            rmb();
            state->free_queue.head = head + 1;
            state->current_buf_ptr = buf_ptr;
            state->current_buf_seq = 0;
            wmb();

            L2PerfBuffer *buf = reinterpret_cast<L2PerfBuffer *>(buf_ptr);
            buf->count = 0;
            s_perf_records_buffers[i] = buf;

            LOG_DEBUG("Core %d: popped initial buffer (addr=0x%lx)", i, buf_ptr);
        } else {
            LOG_ERROR("Core %d: free_queue is empty during init!", i);
            state->current_buf_ptr = 0;
            s_perf_records_buffers[i] = nullptr;
        }

        // Prime the AICore rotation channel with the initial buffer.
        // Generation starts at 1 so AICore's cached_generation==0 triggers
        // a refresh on its first task.
        rmb();
        uint32_t ac_head = ac_state->free_queue.head;
        uint32_t ac_tail = ac_state->free_queue.tail;
        if (ac_head != ac_tail) {
            uint64_t ac_buf_ptr = ac_state->free_queue.buffer_ptrs[ac_head % PLATFORM_PROF_SLOT_COUNT];
            rmb();
            ac_state->free_queue.head = ac_head + 1;
            ac_state->current_buf_seq = 0;
            ac_state->rotation.current_buf_ptr = ac_buf_ptr;
            ac_state->rotation.generation = 1;
            wmb();
            L2PerfAicoreBuffer *ac_buf = reinterpret_cast<L2PerfAicoreBuffer *>(ac_buf_ptr);
            ac_buf->count = 0;
            LOG_DEBUG("Core %d: primed AICore rotation with buf=0x%lx, gen=1", i, ac_buf_ptr);
        } else {
            LOG_ERROR("Core %d: AICore free_queue is empty during init!", i);
            ac_state->rotation.current_buf_ptr = 0;
            ac_state->rotation.generation = 1;
            wmb();
        }
    }

    wmb();

    LOG_INFO_V0("Performance profiling initialized for %d cores (with AICore rotation)", worker_count);
}

/**
 * Internal records-buffer rotation. Called from `l2_perf_aicpu_complete_record`
 * after a record is committed and the buffer hits capacity. Only swaps an
 * AICPU-private records pointer — AICore reads from a stable ring and is
 * unaffected by this call.
 */
static void switch_records_buffer(int core_id, int thread_idx) {
    L2PerfBufferState *state = s_perf_buffer_states[core_id];
    if (state == nullptr) {
        return;
    }

    L2PerfBuffer *full_buf = s_perf_records_buffers[core_id];
    if (full_buf == nullptr) {
        return;
    }

    LOG_INFO_V0("Thread %d: Core %d buffer is full (count=%u)", thread_idx, core_id, full_buf->count);

    // Check free_queue before committing the full buffer
    rmb();
    uint32_t head = state->free_queue.head;
    uint32_t tail = state->free_queue.tail;

    if (head == tail) {
        // No replacement buffer available — overwrite current buffer to keep AICore alive
        LOG_WARN("Thread %d: Core %d no free buffer, overwriting current buffer (data lost)", thread_idx, core_id);
        state->dropped_record_count = state->dropped_record_count + full_buf->count;
        full_buf->count = 0;
        wmb();
        return;
    }

    // Enqueue full buffer to ReadyQueue
    uint32_t seq = state->current_buf_seq;
    int rc = enqueue_ready_buffer(s_l2_perf_header, thread_idx, core_id, state->current_buf_ptr, seq, 0);
    if (rc != 0) {
        LOG_ERROR("Thread %d: Core %d failed to enqueue buffer (queue full), data lost!", thread_idx, core_id);
        // Revert: discard data and keep writing
        state->dropped_record_count = state->dropped_record_count + full_buf->count;
        full_buf->count = 0;
        wmb();
        return;
    }

    // Pop next buffer from free_queue
    uint64_t new_buf_ptr = state->free_queue.buffer_ptrs[head % PLATFORM_PROF_SLOT_COUNT];
    rmb();
    state->free_queue.head = head + 1;
    state->current_buf_ptr = new_buf_ptr;
    state->current_buf_seq = seq + 1;
    wmb();

    L2PerfBuffer *new_buf = reinterpret_cast<L2PerfBuffer *>(new_buf_ptr);
    new_buf->count = 0;
    s_perf_records_buffers[core_id] = new_buf;

    LOG_INFO_V0("Thread %d: Core %d switched to new buffer (addr=0x%lx)", thread_idx, core_id, new_buf_ptr);
}

// Try to rotate the AICore buffer for `core_id`. Called from the completion
// path after a successful L2PerfRecord commit so the just-FIN'd task's
// AICore record is guaranteed to be in the old buffer before we enqueue it.
// On success bumps `ac_state->current_buf_seq`; on failure (empty free queue
// or full ready queue) the old buffer is abandoned in place, AICore overflows
// it from now on, and the drop count grows.
static void aicore_rotate(int core_id, int thread_idx) {
    L2PerfAicoreBufferState *ac_state = s_aicore_buffer_states[core_id];
    if (ac_state == nullptr) {
        return;
    }

    uint64_t old_buf_ptr = ac_state->rotation.current_buf_ptr;
    uint32_t seq = ac_state->current_buf_seq;

    rmb();
    uint32_t head = ac_state->free_queue.head;
    uint32_t tail = ac_state->free_queue.tail;
    if (head == tail) {
        // No replacement available — AICore continues to write into the
        // old buffer; its slot counter will hit BUFFER_SIZE and drop the
        // overflow. Account the drop on AICPU side too. Note that we do
        // NOT bump current_buf_seq here — flush will still see the buffer
        // as the "current full" one and stamp it correctly.
        ac_state->dropped_record_count =
            ac_state->dropped_record_count + static_cast<uint32_t>(PLATFORM_AICORE_BUFFER_SIZE);
        LOG_WARN(
            "Thread %d: Core %d AICore free_queue empty at rotation; up to %d subsequent records will drop", thread_idx,
            core_id, PLATFORM_AICORE_BUFFER_SIZE
        );
        return;
    }

    // Enqueue the just-filled AICore buffer with count = BUFFER_SIZE.
    if (old_buf_ptr != 0) {
        L2PerfAicoreBuffer *old_buf = reinterpret_cast<L2PerfAicoreBuffer *>(old_buf_ptr);
        old_buf->count = static_cast<uint32_t>(PLATFORM_AICORE_BUFFER_SIZE);
        wmb();
        int rc = enqueue_ready_buffer(s_l2_perf_header, thread_idx, core_id, old_buf_ptr, seq, /*is_phase=*/2);
        if (rc != 0) {
            LOG_ERROR(
                "Thread %d: Core %d failed to enqueue AICore buffer (queue full), %d records lost", thread_idx, core_id,
                PLATFORM_AICORE_BUFFER_SIZE
            );
            ac_state->dropped_record_count =
                ac_state->dropped_record_count + static_cast<uint32_t>(PLATFORM_AICORE_BUFFER_SIZE);
            return;
        }
    }

    // Pop next buffer from free_queue and publish via the rotation channel.
    uint64_t new_buf_ptr = ac_state->free_queue.buffer_ptrs[head % PLATFORM_PROF_SLOT_COUNT];
    rmb();
    ac_state->free_queue.head = head + 1;
    ac_state->current_buf_seq = seq + 1;
    L2PerfAicoreBuffer *new_buf = reinterpret_cast<L2PerfAicoreBuffer *>(new_buf_ptr);
    new_buf->count = 0;

    wmb();  // ensure new_buf->count=0 visible before AICore sees new ptr
    ac_state->rotation.current_buf_ptr = new_buf_ptr;
    ac_state->rotation.generation = ac_state->rotation.generation + 1;
    wmb();
}

// Public no-op shim kept so callers compile during the cross-runtime
// transition; the rotation has been moved into l2_perf_aicpu_complete_record
// where it is race-free vs in-flight AICore record writes.
void l2_perf_aicpu_maybe_rotate_aicore(int /*core_id*/, int /*thread_idx*/) {}

int l2_perf_aicpu_complete_record(
    int core_id, int thread_idx, uint32_t expected_reg_task_id, uint64_t task_id, uint32_t func_id, CoreType core_type,
    uint64_t dispatch_time, uint64_t finish_time
) {
    if (core_id < 0 || core_id >= PLATFORM_MAX_CORES) {
        return -1;
    }
    L2PerfBufferState *state = s_perf_buffer_states[core_id];
    if (state == nullptr) {
        return -1;
    }

    // Account every commit attempt up front so host can detect silent loss as
    // `device_total - (collected + dropped)`.
    state->total_record_count += 1;

    L2PerfBuffer *l2_perf_buf = s_perf_records_buffers[core_id];
    if (l2_perf_buf == nullptr) {
        // No active records buffer (init ran out of free buffers); count as drop
        // so host reconciliation stays consistent.
        state->dropped_record_count += 1;
        return -1;
    }
    uint32_t count = l2_perf_buf->count;
    if (count >= PLATFORM_PROF_BUFFER_SIZE) {
        // Defensive: should not happen because we rotate at end of every commit.
        state->dropped_record_count += 1;
        return -1;
    }

    // AICore-as-producer: AICore writes start/end/task_id directly into its
    // own per-core L2PerfAicoreBuffer (indexed by reg_task_id % SIZE). AICPU
    // writes only AICPU-owned fields here; start/end stay zero on-device and
    // are patched by the host when the buffer is consumed. Join key is
    // `reg_task_id` (monotonic per core), stored alongside the PTO2-encoded
    // `task_id` so the host can match without a hashmap lookup. This
    // eliminates the per-task rmb() + staging cache-line read the previous
    // design required.
    L2PerfRecord *record = &l2_perf_buf->records[count];
    record->start_time = 0;
    record->end_time = 0;
    record->duration = 0;
    record->task_id = task_id;
    record->reg_task_id = expected_reg_task_id;
    record->func_id = func_id;
    record->core_type = core_type;

    // AICPU_TIMING and above: dispatch/finish timing.
    if (g_l2_perf_level >= L2PerfLevel::AICPU_TIMING) {
        record->dispatch_time = dispatch_time;
        record->finish_time = finish_time;
    } else {
        record->dispatch_time = 0;
        record->finish_time = 0;
    }

    uint32_t new_count = count + 1;
    l2_perf_buf->count = new_count;
    wmb();

    // Rotate AICpu's L2PerfBuffer after the write so the just-committed
    // record is preserved.
    if (new_count >= PLATFORM_PROF_BUFFER_SIZE) {
        switch_records_buffer(core_id, thread_idx);
    }

    // AICore-side rotation: piggy-back on this completion. The current task's
    // AICore record has already been written (AICore writes it before the FIN
    // signal that brought us here), so at every PLATFORM_AICORE_BUFFER_SIZE-th
    // completion we hand the just-filled AICore buffer to the ready queue and
    // pop a fresh one. Doing this in the completion path (not the dispatch
    // path) closes the dispatch-boundary race: at a dispatch boundary AICore
    // may still be running task K-1 with its record write yet to happen, so a
    // dispatch-time rotation could leak K-1's record into the wrong buffer.
    //
    // Residual race at the next dispatch: under dual-issue, AICPU may have
    // already dispatched task K+1 when it processes the FIN of K=BUFFER_SIZE-1.
    // AICPU's rotation (~1 us — a few wmbs + an enqueue + a pop) needs to
    // complete before AICore's `record_task` for K+1 reads the published
    // rotation channel; otherwise AICore's local `slot_within_buf` hits the
    // `slot >= BUFFER_SIZE` guard and silently drops K+1's record. For typical
    // kernels (>10 us) the rotation completes long before AICore's record
    // write, so this never fires; for adversarial sub-microsecond kernels it
    // can. The host parser surfaces the loss via the `unmatched > 0` warning
    // in join_aicore_records().
    //
    // total_record_count is uint32_t — wraps after ~4 G completions per core.
    // At realistic dispatch rates this is multi-week continuous-run territory;
    // we accept the limitation rather than widening the on-device counter.
    L2PerfAicoreBufferState *ac_state = s_aicore_buffer_states[core_id];
    if (ac_state != nullptr) {
        uint32_t completed = ac_state->total_record_count + 1;
        ac_state->total_record_count = completed;
        if ((completed & (PLATFORM_AICORE_BUFFER_SIZE - 1)) == 0) {
            aicore_rotate(core_id, thread_idx);
        }
    }

    return 0;
}

void l2_perf_aicpu_flush_buffers(int thread_idx, const int *cur_thread_cores, int core_num) {
    if (!g_enable_l2_swimlane) {
        return;
    }

    void *l2_perf_base = reinterpret_cast<void *>(g_platform_l2_perf_base);
    if (l2_perf_base == nullptr) {
        return;
    }

    rmb();

    LOG_INFO_V0("Thread %d: Flushing performance buffers for %d cores", thread_idx, core_num);

    int flushed_count = 0;

    for (int i = 0; i < core_num; i++) {
        int core_id = cur_thread_cores[i];
        L2PerfBufferState *state = s_perf_buffer_states[core_id];
        if (state == nullptr) continue;

        rmb();
        uint64_t buf_ptr = state->current_buf_ptr;
        if (buf_ptr == 0) {
            // No active buffer
        } else {
            L2PerfBuffer *buf = reinterpret_cast<L2PerfBuffer *>(buf_ptr);
            if (buf->count > 0) {
                uint32_t seq = state->current_buf_seq;
                int rc = enqueue_ready_buffer(s_l2_perf_header, thread_idx, core_id, buf_ptr, seq, 0);
                if (rc == 0) {
                    LOG_INFO_V0("Thread %d: Core %d flushed buffer with %u records", thread_idx, core_id, buf->count);
                    flushed_count++;
                    state->current_buf_ptr = 0;
                    s_perf_records_buffers[core_id] = nullptr;
                    wmb();
                } else {
                    // ready_queue full at end-of-run: account the loss and clear the
                    // buffer so host reconcile sees a clean state (current_buf_ptr=0)
                    // and dropped == flush failures rather than ring/task_id mismatch.
                    LOG_ERROR(
                        "Thread %d: Core %d failed to enqueue buffer (queue full), %u records lost!", thread_idx,
                        core_id, buf->count
                    );
                    state->dropped_record_count = state->dropped_record_count + buf->count;
                    buf->count = 0;
                    state->current_buf_ptr = 0;
                    s_perf_records_buffers[core_id] = nullptr;
                    wmb();
                }
            }
        }

        // Also flush the current AICore buffer to the ready queue so the host
        // sees this session's final batch of AICore timestamps.
        //
        // High-water mark uses the rotation accounting (total_record_count -
        // current_buf_seq * BUFFER_SIZE) so the failed-rotation case is
        // correctly handled: when an earlier rotation couldn't find a free
        // buffer the current buffer holds BUFFER_SIZE valid records plus an
        // unknown number of dropped overflow attempts; the formula clamps
        // to BUFFER_SIZE in that case rather than stamping a stale partial
        // count.
        L2PerfAicoreBufferState *ac_state = s_aicore_buffer_states[core_id];
        if (ac_state == nullptr) continue;

        rmb();
        uint64_t ac_buf_ptr = ac_state->rotation.current_buf_ptr;
        if (ac_buf_ptr == 0) continue;

        uint32_t live = ac_state->total_record_count -
                        ac_state->current_buf_seq * static_cast<uint32_t>(PLATFORM_AICORE_BUFFER_SIZE);
        if (live == 0) {
            // Last rotation matched the last completion exactly — buffer is
            // freshly popped and empty. Skip.
            continue;
        }
        uint32_t ac_mark = (live > static_cast<uint32_t>(PLATFORM_AICORE_BUFFER_SIZE)) ?
                               static_cast<uint32_t>(PLATFORM_AICORE_BUFFER_SIZE) :
                               live;
        L2PerfAicoreBuffer *ac_buf = reinterpret_cast<L2PerfAicoreBuffer *>(ac_buf_ptr);
        ac_buf->count = ac_mark;
        wmb();

        uint32_t ac_seq = ac_state->current_buf_seq;
        int rc = enqueue_ready_buffer(s_l2_perf_header, thread_idx, core_id, ac_buf_ptr, ac_seq, /*is_phase=*/2);
        if (rc == 0) {
            LOG_INFO_V0(
                "Thread %d: Core %d flushed AICore buffer (seq=%u, count=%u)", thread_idx, core_id, ac_seq, ac_mark
            );
            ac_state->rotation.current_buf_ptr = 0;
            wmb();
        } else {
            LOG_ERROR("Thread %d: Core %d failed to enqueue AICore buffer at flush (queue full)", thread_idx, core_id);
            ac_state->dropped_record_count = ac_state->dropped_record_count + ac_mark;
            ac_state->rotation.current_buf_ptr = 0;
            wmb();
        }
    }

    wmb();

    LOG_INFO_V0("Thread %d: Performance buffer flush complete, %d buffers flushed", thread_idx, flushed_count);
}

void l2_perf_aicpu_init_phase(int worker_count, int num_sched_threads) {
    void *l2_perf_base = reinterpret_cast<void *>(g_platform_l2_perf_base);
    if (l2_perf_base == nullptr) {
        LOG_ERROR("l2_perf_data_base is NULL, cannot initialize phase profiling");
        return;
    }

    s_phase_header = get_phase_header(l2_perf_base, worker_count);
    s_l2_perf_header = get_l2_perf_header(l2_perf_base);

    s_phase_header->magic = AICPU_PHASE_MAGIC;
    s_phase_header->num_sched_threads = num_sched_threads;
    s_phase_header->records_per_thread = PLATFORM_PHASE_RECORDS_PER_THREAD;
    s_phase_header->num_cores = 0;

    memset(s_phase_header->core_to_thread, -1, sizeof(s_phase_header->core_to_thread));

    // Cache per-thread record pointers and clear buffers
    // Include all threads: scheduler + orchestrator (orchestrators may become schedulers)
    int total_threads = num_sched_threads + 1;
    if (total_threads > PLATFORM_MAX_AICPU_THREADS) {
        total_threads = PLATFORM_MAX_AICPU_THREADS;
    }
    for (int t = 0; t < total_threads; t++) {
        PhaseBufferState *state = get_phase_buffer_state(l2_perf_base, worker_count, t);

        s_phase_buffer_states[t] = state;

        // Pop first buffer from free_queue
        rmb();
        uint32_t head = state->free_queue.head;
        uint32_t tail = state->free_queue.tail;

        if (head != tail) {
            uint64_t buf_ptr = state->free_queue.buffer_ptrs[head % PLATFORM_PROF_SLOT_COUNT];
            rmb();
            state->free_queue.head = head + 1;
            state->current_buf_ptr = buf_ptr;
            state->current_buf_seq = 0;
            wmb();

            PhaseBuffer *buf = reinterpret_cast<PhaseBuffer *>(buf_ptr);
            buf->count = 0;
            s_current_phase_buf[t] = buf;

            LOG_DEBUG("Thread %d: popped initial phase buffer (addr=0x%lx)", t, buf_ptr);
        } else {
            LOG_ERROR("Thread %d: phase free_queue is empty during init!", t);
            state->current_buf_ptr = 0;
            s_current_phase_buf[t] = nullptr;
        }
    }

    // Clear remaining slots
    for (int t = total_threads; t < PLATFORM_MAX_AICPU_THREADS; t++) {
        s_phase_buffer_states[t] = nullptr;
        s_current_phase_buf[t] = nullptr;
    }

    wmb();

    LOG_INFO_V0(
        "Phase profiling initialized: %d scheduler + 1 orch thread, %d records/thread", num_sched_threads,
        PLATFORM_PHASE_RECORDS_PER_THREAD
    );
}

/**
 * Switch phase buffer when current buffer is full (free queue version)
 *
 * Enqueues the full buffer to ReadyQueue and pops the next buffer from free_queue.
 * If no free buffer is available, sets s_current_phase_buf to nullptr so subsequent
 * records are dropped (preserving already-enqueued data).
 */
static void switch_phase_buffer(int thread_idx) {
    PhaseBufferState *state = s_phase_buffer_states[thread_idx];
    if (state == nullptr) return;

    PhaseBuffer *full_buf = s_current_phase_buf[thread_idx];
    if (full_buf == nullptr) return;

    LOG_INFO_V0("Thread %d: phase buffer is full (count=%u)", thread_idx, full_buf->count);

    // Enqueue to ReadyQueue
    uint32_t seq = state->current_buf_seq;
    int rc = enqueue_ready_buffer(s_l2_perf_header, thread_idx, thread_idx, state->current_buf_ptr, seq, 1);
    if (rc != 0) {
        LOG_ERROR(
            "Thread %d: failed to enqueue phase buffer (queue full), %u records lost!", thread_idx, full_buf->count
        );
        state->dropped_record_count = state->dropped_record_count + full_buf->count;
        full_buf->count = 0;
        s_current_phase_buf[thread_idx] = nullptr;
        state->current_buf_ptr = 0;
        wmb();
        return;
    }

    // Pop next buffer from free_queue
    rmb();
    uint32_t head = state->free_queue.head;
    uint32_t tail = state->free_queue.tail;

    if (head != tail) {
        uint64_t new_buf_ptr = state->free_queue.buffer_ptrs[head % PLATFORM_PROF_SLOT_COUNT];
        rmb();
        state->free_queue.head = head + 1;
        state->current_buf_ptr = new_buf_ptr;
        state->current_buf_seq = seq + 1;
        wmb();

        PhaseBuffer *new_buf = reinterpret_cast<PhaseBuffer *>(new_buf_ptr);
        new_buf->count = 0;
        s_current_phase_buf[thread_idx] = new_buf;

        LOG_INFO_V0("Thread %d: switched to new phase buffer", thread_idx);
    } else {
        // No free buffer available, drop subsequent records
        LOG_WARN("Thread %d: no free phase buffer available, dropping records until Host catches up", thread_idx);
        s_current_phase_buf[thread_idx] = nullptr;
        state->current_buf_ptr = 0;
        wmb();
    }
}

void l2_perf_aicpu_record_phase(
    int thread_idx, AicpuPhaseId phase_id, uint64_t start_time, uint64_t end_time, uint32_t loop_iter,
    uint64_t tasks_processed, uint32_t extra1, uint32_t extra2
) {
    if (s_phase_header == nullptr) {
        return;
    }

    PhaseBufferState *state = s_phase_buffer_states[thread_idx];
    if (state == nullptr) {
        return;
    }

    // Account every commit attempt up front so host can detect silent loss
    // as `device_total - (collected + dropped)` (mirrors PERF accounting).
    state->total_record_count += 1;

    PhaseBuffer *buf = s_current_phase_buf[thread_idx];

    // Try to recover from nullptr (no buffer was available on previous switch)
    if (buf == nullptr) {
        rmb();
        uint32_t head = state->free_queue.head;
        uint32_t tail = state->free_queue.tail;

        if (head != tail) {
            uint64_t buf_ptr = state->free_queue.buffer_ptrs[head % PLATFORM_PROF_SLOT_COUNT];
            rmb();
            state->free_queue.head = head + 1;
            state->current_buf_ptr = buf_ptr;
            state->current_buf_seq = state->current_buf_seq + 1;
            wmb();

            buf = reinterpret_cast<PhaseBuffer *>(buf_ptr);
            buf->count = 0;
            s_current_phase_buf[thread_idx] = buf;

            LOG_INFO_V0("Thread %d: recovered phase buffer", thread_idx);
        }
        if (buf == nullptr) {
            state->dropped_record_count += 1;
            return;
        }
    }

    uint32_t idx = buf->count;

    if (idx >= PLATFORM_PHASE_RECORDS_PER_THREAD) {
        // Buffer full, switch to next buffer
        switch_phase_buffer(thread_idx);
        buf = s_current_phase_buf[thread_idx];
        if (buf == nullptr) {
            state->dropped_record_count += 1;
            return;
        }
        idx = buf->count;
        if (idx >= PLATFORM_PHASE_RECORDS_PER_THREAD) {
            state->dropped_record_count += 1;
            return;
        }
    }

    AicpuPhaseRecord *record = &buf->records[idx];
    record->start_time = start_time;
    record->end_time = end_time;
    record->loop_iter = loop_iter;
    record->phase_id = phase_id;
    record->task_id = tasks_processed;
    record->extra1 = extra1;
    record->extra2 = extra2;

    buf->count = idx + 1;
}

void l2_perf_aicpu_set_orch_thread_idx(int thread_idx) { s_orch_thread_idx = thread_idx; }

void l2_perf_aicpu_record_orch_phase(
    AicpuPhaseId phase_id, uint64_t start_time, uint64_t end_time, uint32_t submit_idx, uint64_t task_id
) {
    if (s_orch_thread_idx < 0 || s_phase_header == nullptr) return;
    l2_perf_aicpu_record_phase(s_orch_thread_idx, phase_id, start_time, end_time, submit_idx, task_id);
}

void l2_perf_aicpu_flush_phase_buffers(int thread_idx) {
    if (s_phase_header == nullptr || s_l2_perf_header == nullptr) {
        return;
    }

    PhaseBufferState *state = s_phase_buffer_states[thread_idx];
    if (state == nullptr) return;

    rmb();
    uint64_t buf_ptr = state->current_buf_ptr;
    if (buf_ptr == 0) {
        // No active buffer
        return;
    }

    PhaseBuffer *buf = reinterpret_cast<PhaseBuffer *>(buf_ptr);
    if (buf->count == 0) {
        return;
    }

    uint32_t seq = state->current_buf_seq;
    int rc = enqueue_ready_buffer(s_l2_perf_header, thread_idx, thread_idx, buf_ptr, seq, 1);
    if (rc == 0) {
        LOG_INFO_V0("Thread %d: flushed phase buffer with %u records", thread_idx, buf->count);
    } else {
        LOG_ERROR("Thread %d: failed to enqueue phase buffer (queue full), %u records lost!", thread_idx, buf->count);
        state->dropped_record_count = state->dropped_record_count + buf->count;
        buf->count = 0;
    }
    state->current_buf_ptr = 0;
    s_current_phase_buf[thread_idx] = nullptr;
    wmb();
}

void l2_perf_aicpu_init_core_assignments(int total_cores) {
    if (s_phase_header == nullptr) {
        return;
    }
    memset(s_phase_header->core_to_thread, -1, sizeof(s_phase_header->core_to_thread));
    s_phase_header->num_cores = static_cast<uint32_t>(total_cores);
    wmb();
    LOG_INFO_V0("Core-to-thread mapping init: %d cores", total_cores);
}

void l2_perf_aicpu_write_core_assignments_for_thread(int thread_idx, const int *core_ids, int core_num) {
    if (s_phase_header == nullptr) {
        return;
    }
    for (int i = 0; i < core_num; i++) {
        int core_id = core_ids[i];
        if (core_id >= 0 && core_id < PLATFORM_MAX_CORES) {
            s_phase_header->core_to_thread[core_id] = static_cast<int8_t>(thread_idx);
        }
    }
    wmb();
}
