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
 *
 * AICore writes timing into a per-core stable L2PerfAicoreRing
 * (state.aicore_ring_ptr); AICPU reads the slot in complete_record and
 * commits into records[]. The ring address is set once by the host at init
 * and never reassigned, so AICore's write address is decoupled from the
 * AICPU rotating L2PerfBuffer.
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

// Per-core stable AICore staging ring pointer cache. Populated by
// l2_perf_aicpu_init() from state->aicore_ring_ptr (host-published at SHM
// init time); read by complete_record. Never reassigned during the run, so
// no synchronization is needed for concurrent readers. AICPU sees these
// addresses as plain pointers (no __gm__ on AICPU).
static L2PerfAicoreRing *s_perf_aicore_rings[PLATFORM_MAX_CORES] = {};

// Per-thread PhaseBufferState cache
static PhaseBufferState *s_phase_buffer_states[PLATFORM_MAX_AICPU_THREADS] = {};
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

    LOG_INFO_V0("Initializing performance profiling for %d cores (memcpy-based)", worker_count);

    // Pop first buffer from free_queue for each core, and cache the stable
    // AICore staging ring pointer so complete_record can read it without
    // touching SHM.
    for (int i = 0; i < worker_count; i++) {
        L2PerfBufferState *state = get_perf_buffer_state(l2_perf_base, i);

        s_perf_buffer_states[i] = state;
        s_perf_aicore_rings[i] = reinterpret_cast<L2PerfAicoreRing *>(state->aicore_ring_ptr);

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

            LOG_DEBUG("Core %d: popped initial buffer (addr=0x%lx)", i, buf_ptr);
        } else {
            LOG_ERROR("Core %d: free_queue is empty during init!", i);
            state->current_buf_ptr = 0;
        }
    }

    wmb();

    LOG_INFO_V0("Performance profiling initialized for %d cores", worker_count);
}

/**
 * Switch performance buffer when the current buffer is full.
 *
 * Internal-only: complete_record calls this when records[count] is at
 * capacity. Enqueues the full buffer to the per-thread ready_queue and
 * pops a fresh one from the free_queue. Failure paths bump
 * dropped_record_count so reconcile sees a consistent device state.
 *
 * The AICore staging ring (state->aicore_ring_ptr / s_perf_aicore_rings[])
 * is **never** touched here: AICore's write address is the same for the
 * entire run.
 */
static void switch_buffer(int core_id, int thread_idx) {
    L2PerfBufferState *state = s_perf_buffer_states[core_id];
    if (state == nullptr) {
        return;
    }

    L2PerfBuffer *full_buf = reinterpret_cast<L2PerfBuffer *>(state->current_buf_ptr);
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
        state->dropped_record_count += full_buf->count;
        full_buf->count = 0;
        wmb();
        return;
    }

    // Enqueue full buffer to ReadyQueue
    uint32_t seq = state->current_buf_seq;
    int rc = enqueue_ready_buffer(s_l2_perf_header, thread_idx, core_id, state->current_buf_ptr, seq, 0);
    if (rc != 0) {
        LOG_ERROR("Thread %d: Core %d failed to enqueue buffer (queue full), data lost!", thread_idx, core_id);
        state->dropped_record_count += full_buf->count;
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
    wmb();

    LOG_INFO_V0("Thread %d: Core %d switched to new buffer (addr=0x%lx)", thread_idx, core_id, new_buf_ptr);
}

int l2_perf_aicpu_complete_record(
    int core_id, uint32_t expected_reg_task_id, uint64_t task_id, uint32_t func_id, CoreType core_type,
    uint64_t dispatch_time, uint64_t finish_time, const uint64_t *fanout, int32_t fanout_count
) {
    if (core_id < 0 || core_id >= PLATFORM_MAX_CORES) return -1;

    L2PerfBufferState *state = s_perf_buffer_states[core_id];
    if (state == nullptr) return -1;

    // Account every commit attempt before any drop path so reconcile equation
    // (collected + dropped + mismatch == total) holds.
    state->total_record_count += 1;

    // Read AICore-published timing from the per-core stable staging ring.
    L2PerfAicoreRing *ring = s_perf_aicore_rings[core_id];
    if (ring == nullptr) {
        state->dropped_record_count += 1;
        wmb();
        return -1;
    }
    L2PerfRecord *slot = &ring->dual_issue_slots[expected_reg_task_id % PLATFORM_L2_AICORE_RING_SIZE];
    cache_invalidate_range(slot, 64);
    if (static_cast<uint32_t>(slot->task_id) != expected_reg_task_id) {
        // AICore hasn't published this slot yet — count separately from
        // capacity drops (mismatch is a hard invariant violation).
        state->mismatch_record_count += 1;
        wmb();
        return -1;
    }

    rmb();
    uint64_t cur_ptr = state->current_buf_ptr;
    if (cur_ptr == 0) {
        // Buffer was flushed/cleared — late FIN after flush. Count as dropped
        // so the buffer doesn't get re-populated post-stop().
        state->dropped_record_count += 1;
        wmb();
        return -1;
    }

    L2PerfBuffer *l2_perf_buf = reinterpret_cast<L2PerfBuffer *>(cur_ptr);
    uint32_t count = l2_perf_buf->count;
    if (count >= PLATFORM_PROF_BUFFER_SIZE) {
        // Flip to a fresh buffer when full. Caller doesn't need a thread_idx
        // routed to switch_buffer, so derive it from the AicpuPhaseHeader's
        // core→thread map (filled at init by the runtime). Until phase
        // metadata is populated, fall back to thread 0 — the per-thread
        // ready_queue is only used for collector throughput, so a temporary
        // single-thread serialization is harmless.
        int thread_idx = 0;
        if (s_phase_header != nullptr && core_id >= 0 && core_id < PLATFORM_MAX_CORES) {
            int8_t mapped = s_phase_header->core_to_thread[core_id];
            if (mapped >= 0 && mapped < PLATFORM_MAX_AICPU_THREADS) {
                thread_idx = static_cast<int>(static_cast<unsigned char>(mapped));
            }
        }
        switch_buffer(core_id, thread_idx);
        rmb();
        cur_ptr = state->current_buf_ptr;
        if (cur_ptr == 0) {
            state->dropped_record_count += 1;
            wmb();
            return -1;
        }
        l2_perf_buf = reinterpret_cast<L2PerfBuffer *>(cur_ptr);
        count = l2_perf_buf->count;
        if (count >= PLATFORM_PROF_BUFFER_SIZE) {
            state->dropped_record_count += 1;
            wmb();
            return -1;
        }
    }

    // Copy AICore timing to committed record slot
    L2PerfRecord *record = &l2_perf_buf->records[count];
    record->start_time = slot->start_time;
    record->end_time = slot->end_time;

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

void l2_perf_aicpu_flush_buffers(int thread_idx, const int *cur_thread_cores, int core_num) {
    if (!is_l2_swimlane_enabled()) {
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
            continue;
        }

        L2PerfBuffer *buf = reinterpret_cast<L2PerfBuffer *>(buf_ptr);
        if (buf->count == 0) {
            continue;
        }

        uint32_t seq = state->current_buf_seq;
        int rc = enqueue_ready_buffer(s_l2_perf_header, thread_idx, core_id, buf_ptr, seq, 0);
        if (rc == 0) {
            LOG_INFO_V0("Thread %d: Core %d flushed buffer with %u records", thread_idx, core_id, buf->count);
            flushed_count++;
            state->current_buf_ptr = 0;
            wmb();
        } else {
            // ready_queue full at end-of-run: account the loss and clear the
            // buffer so host reconcile sees a clean state (current_buf_ptr=0)
            // and dropped == flush failures rather than silent leftover.
            LOG_ERROR(
                "Thread %d: Core %d failed to enqueue buffer (queue full), %u records lost!", thread_idx, core_id,
                buf->count
            );
            state->dropped_record_count += buf->count;
            buf->count = 0;
            state->current_buf_ptr = 0;
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
    memset(&s_phase_header->orch_summary, 0, sizeof(AicpuOrchSummary));

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
        LOG_ERROR("Thread %d: failed to enqueue phase buffer (queue full), discarding data", thread_idx);
        // Treat the entire un-enqueued buffer as dropped to keep the
        // reconcile equation balanced. record_phase already counted these
        // records in total_record_count when they were committed.
        state->dropped_record_count += full_buf->count;
        full_buf->count = 0;
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
    uint64_t tasks_processed
) {
    if (s_phase_header == nullptr) {
        return;
    }

    PhaseBufferState *state = s_phase_buffer_states[thread_idx];
    if (state == nullptr) return;

    // Account every commit attempt before any drop path so reconcile
    // (collected + dropped + mismatch == total) holds. PHASE has no
    // ring/AICore staging path so mismatch stays 0 here.
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
            wmb();
            return;  // Still no buffer available
        }
    }

    uint32_t idx = buf->count;

    if (idx >= PLATFORM_PHASE_RECORDS_PER_THREAD) {
        // Buffer full, switch to next buffer
        switch_phase_buffer(thread_idx);
        buf = s_current_phase_buf[thread_idx];
        if (buf == nullptr) {
            state->dropped_record_count += 1;
            wmb();
            return;  // No buffer available
        }
        idx = buf->count;
        if (idx >= PLATFORM_PHASE_RECORDS_PER_THREAD) {
            state->dropped_record_count += 1;
            wmb();
            return;  // Switch failed; drop this record
        }
    }

    AicpuPhaseRecord *record = &buf->records[idx];
    record->start_time = start_time;
    record->end_time = end_time;
    record->loop_iter = loop_iter;
    record->phase_id = phase_id;
    record->task_id = tasks_processed;

    buf->count = idx + 1;
    wmb();
}

void l2_perf_aicpu_write_orch_summary(const AicpuOrchSummary *src) {
    if (s_phase_header == nullptr) {
        return;
    }

    AicpuOrchSummary *dst = &s_phase_header->orch_summary;

    memcpy(dst, src, sizeof(AicpuOrchSummary));
    dst->magic = AICPU_PHASE_MAGIC;
    dst->padding = 0;

    wmb();

    LOG_INFO_V0(
        "Orchestrator summary written: %" PRId64 " tasks, %.3fus", static_cast<int64_t>(src->submit_count),
        cycles_to_us(src->end_time - src->start_time)
    );
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
        state->dropped_record_count += buf->count;
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
