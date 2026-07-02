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

// Platform-layer scope_stats collector (streaming).
//
// Owns the scope depth/site stack and exposes pure-value APIs for runtime to
// report the task/heap ring start/end and tensormap usage at scope boundaries.
// One ScopeStatsRecord is appended per boundary (begin and end) into a pooled
// ScopeStatsBuffer; full buffers stream to the host via the per-thread
// ready_queue (SPSC, mirrors dep_gen). No runtime-specific types cross the
// boundary.

#include "aicpu/scope_stats_collector_aicpu.h"
#include <cstring>

#include "aicpu/device_time.h"
#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "common/scope_stats.h"
#include "common/unified_log.h"

// ---------------------------------------------------------------------------
// Collector state
// ---------------------------------------------------------------------------

int32_t scope_stats_depth = -1;
static bool scope_stats_enabled = false;

// Streaming transport state — single dep_gen-style instance (the orchestrator).
static ScopeStatsDataHeader *s_scope_stats_header = nullptr;
static ScopeStatsBufferState *s_scope_stats_state = nullptr;
static int s_orch_thread_idx = -1;  // set via scope_stats_aicpu_set_orch_thread_idx

// Cumulative heap-ring wrap counts per ring, indexed by SCOPE_STATS_HEAP_SIDE_*.
// Fed by the runtime via scope_stats_note_heap_wrap at each wrap; used to unroll
// the boundary-sampled wrapping offsets into monotonic values (see
// unroll_heap_offset). Reset in set_platform_scope_stats_base.
static uint64_t s_heap_wraps[PTO2_SCOPE_STATS_MAX_RING_DEPTH][2] = {};

static constexpr uint64_t kScopeStatsQueueBackpressureWaitCycles = PLATFORM_PROF_SYS_CNT_FREQ / 50000;  // 20 us

namespace {

const char *s_pending_site_file = nullptr;
int32_t s_pending_site_line = 0;

const char *s_scope_site_file[PTO2_SCOPE_STATS_MAX_SCOPE_DEPTH] = {};
int32_t s_scope_site_line[PTO2_SCOPE_STATS_MAX_SCOPE_DEPTH] = {};
// Ring id of each open scope, recorded at scope_stats_begin. Lets a heap-wrap
// report (which carries no ring id) be attributed to the current scope's ring.
int s_scope_ring[PTO2_SCOPE_STATS_MAX_SCOPE_DEPTH] = {};

inline const char *basename_of(const char *path) {
    if (!path) return "(unknown)";
    const char *base = path;
    for (const char *p = path; *p; ++p) {
        if (*p == '/' || *p == '\\') base = p + 1;
    }
    return base;
}

inline void copy_basename(char (&dst)[32], const char *src) {
    // site_file is a compile-time-constant string pointer; the same scope source
    // line (e.g. a begin_scope inside an unroll loop) reuses the same pointer on
    // every iteration. Memoize the basename scan keyed by that pointer so a
    // repeated site becomes a single 32B copy instead of a full-path rescan.
    static const char *s_cached_src = nullptr;
    static char s_cached[32];
    if (src != nullptr && src == s_cached_src) {
        memcpy(dst, s_cached, sizeof(dst));
        return;
    }
    const char *base = basename_of(src);
    size_t i = 0;
    for (; i + 1 < sizeof(dst) && base[i]; i++)
        dst[i] = base[i];
    dst[i] = '\0';
    if (src != nullptr) {
        s_cached_src = src;
        memcpy(s_cached, dst, sizeof(dst));
    }
}

// Enqueue a full buffer onto the orchestrator thread's ready_queue. Returns 0
// on success, -1 if the queue is full or the orch thread index is unset.
bool wait_for_ready_queue_space(ScopeStatsDataHeader *header, int thread_idx, uint32_t *tail_out, uint32_t *head_out) {
    if (header == nullptr || thread_idx < 0 || thread_idx >= PLATFORM_MAX_AICPU_THREADS) {
        return false;
    }
    const uint32_t capacity = PLATFORM_SCOPE_STATS_READYQUEUE_SIZE;
    const uint64_t start = get_sys_cnt_aicpu();

    do {
        uint32_t current_tail = header->queue_tails[thread_idx];
        uint32_t current_head = header->queue_heads[thread_idx];
        uint32_t next_tail = (current_tail + 1) % capacity;
        if (next_tail != current_head) {
            *tail_out = current_tail;
            *head_out = current_head;
            return true;
        }
        if (get_sys_cnt_aicpu() - start >= kScopeStatsQueueBackpressureWaitCycles) {
            break;
        }
    } while (true);
    return false;
}

bool wait_for_free_queue_entry(ScopeStatsFreeQueue *free_queue, uint32_t *head_out, uint32_t *tail_out) {
    if (free_queue == nullptr) {
        return false;
    }
    const uint64_t start = get_sys_cnt_aicpu();

    do {
        uint32_t head = free_queue->head;
        uint32_t tail = free_queue->tail;
        if (head != tail) {
            *head_out = head;
            *tail_out = tail;
            rmb();  // acquire: order the tail read above before the caller's buffer_ptrs read
            return true;
        }
        if (get_sys_cnt_aicpu() - start >= kScopeStatsQueueBackpressureWaitCycles) {
            break;
        }
    } while (true);
    return false;
}

int enqueue_ready_buffer(uint64_t buffer_ptr, uint32_t buffer_seq) {
    int q = s_orch_thread_idx;
    uint32_t capacity = PLATFORM_SCOPE_STATS_READYQUEUE_SIZE;
    uint32_t current_tail = 0;
    uint32_t current_head = 0;
    if (!wait_for_ready_queue_space(s_scope_stats_header, q, &current_tail, &current_head)) {
        return -1;
    }

    uint32_t next_tail = (current_tail + 1) % capacity;
    s_scope_stats_header->queues[q][current_tail].instance_index = 0;
    s_scope_stats_header->queues[q][current_tail].buffer_ptr = buffer_ptr;
    s_scope_stats_header->queues[q][current_tail].buffer_seq = buffer_seq;
    // Single hand-off barrier: drain every record written into the buffer plus
    // the ready-entry fields above before the host can observe the advanced
    // tail. Replaces the former per-record wmb() in scope_stats_end().
    wmb();
    s_scope_stats_header->queue_tails[q] = next_tail;
    return 0;
}

// Pop a free buffer into current_buf_ptr. Returns true if one was available.
bool pop_free_buffer() {
    if (s_scope_stats_state == nullptr) return false;
    uint32_t head = 0;
    uint32_t tail = 0;
    if (!wait_for_free_queue_entry(&s_scope_stats_state->free_queue, &head, &tail)) {
        return false;
    }

    uint64_t buf_ptr = s_scope_stats_state->free_queue.buffer_ptrs[head % PLATFORM_SCOPE_STATS_SLOT_COUNT];
    s_scope_stats_state->free_queue.head = head + 1;
    if (buf_ptr == 0) {
        return false;
    }
    s_scope_stats_state->current_buf_ptr = buf_ptr;
    reinterpret_cast<ScopeStatsBuffer *>(buf_ptr)->count = 0;
    wmb();
    return true;
}

// Commit the full current buffer to the ready_queue before popping a
// replacement. If no replacement is available, later records drop until host
// replenishes free_queue.
void switch_buffer() {
    if (s_scope_stats_state == nullptr) {
        return;
    }
    ScopeStatsBuffer *full_buf = reinterpret_cast<ScopeStatsBuffer *>(s_scope_stats_state->current_buf_ptr);
    if (full_buf == nullptr) {
        return;
    }

    uint32_t seq = s_scope_stats_state->current_buf_seq;
    int rc = enqueue_ready_buffer(s_scope_stats_state->current_buf_ptr, seq);
    if (rc != 0) {
        s_scope_stats_state->dropped_record_count += full_buf->count;
        full_buf->count = 0;
        wmb();
        return;
    }

    s_scope_stats_state->current_buf_ptr = 0;
    s_scope_stats_state->current_buf_seq = seq + 1;
    wmb();
    (void)pop_free_buffer();
}

// Unroll a wrapping heap byte offset into a monotonic value using the
// accumulated wrap count for this ring/side and the registered heap capacity.
// With monotonic heap_start/heap_end, every host-side delta (real_occupancy,
// scope_alloc, high-water) is an exact subtraction regardless of wrap count.
inline uint64_t unroll_heap_offset(uint64_t offset, int ring_id, int side) {
    if (s_scope_stats_header == nullptr) return offset;
    if (ring_id < 0 || ring_id >= PTO2_SCOPE_STATS_MAX_RING_DEPTH) return offset;
    return offset + s_heap_wraps[ring_id][side] * s_scope_stats_header->heap_cap[ring_id];
}

// Append one record carrying the task/heap ring start/end and tensormap usage,
// tagged with the current depth/site and the boundary phase. Called once per
// scope boundary.
void append_record_snapshot(
    int ring_id, int16_t phase, int32_t task_start, int32_t task_end, uint64_t heap_start, uint64_t heap_end,
    int32_t dep_pool_start, int32_t dep_pool_end, int32_t tensormap_used
) {
    if (s_scope_stats_state == nullptr) return;
    // Single volatile read of current_buf_ptr (re-read only after a pop/switch).
    uint64_t cur = s_scope_stats_state->current_buf_ptr;
    if (cur == 0) {
        pop_free_buffer();
        cur = s_scope_stats_state->current_buf_ptr;
    }
    ScopeStatsBuffer *buf = reinterpret_cast<ScopeStatsBuffer *>(cur);
    // Single volatile read of buf->count (reused as the write index below).
    uint32_t idx = (buf != nullptr) ? buf->count : 0;
    if (buf != nullptr && idx >= static_cast<uint32_t>(PLATFORM_SCOPE_STATS_RECORDS_PER_BUFFER)) {
        switch_buffer();
        cur = s_scope_stats_state->current_buf_ptr;
        buf = reinterpret_cast<ScopeStatsBuffer *>(cur);
        idx = (buf != nullptr) ? buf->count : 0;
    }
    s_scope_stats_state->total_record_count += 1;
    if (buf == nullptr) {
        s_scope_stats_state->dropped_record_count += 1;
        return;
    }
    ScopeStatsRecord &rec = buf->records[idx];
    copy_basename(rec.site_file_basename, s_scope_site_file[scope_stats_depth]);
    rec.site_line = s_scope_site_line[scope_stats_depth];
    rec.depth = static_cast<int16_t>(scope_stats_depth);
    rec.ring_id = static_cast<int16_t>(ring_id);
    rec.phase = phase;
    rec._pad = 0;
    rec.task_start = task_start;
    rec.task_end = task_end;
    rec.dep_pool_start = dep_pool_start;
    rec.dep_pool_end = dep_pool_end;
    rec.tensormap_used = tensormap_used;
    rec.heap_start = unroll_heap_offset(heap_start, ring_id, SCOPE_STATS_HEAP_SIDE_RECLAIM);
    rec.heap_end = unroll_heap_offset(heap_end, ring_id, SCOPE_STATS_HEAP_SIDE_ALLOC);
    buf->count = idx + 1;
}

}  // namespace

// ---------------------------------------------------------------------------
// Setter symbols — always exported, unconditionally compiled
// ---------------------------------------------------------------------------

extern "C" void set_scope_stats_enabled(bool enable) { scope_stats_enabled = enable; }

extern "C" bool is_scope_stats_enabled() { return scope_stats_enabled; }

extern "C" void set_platform_scope_stats_base(uint64_t scope_stats_data_base) {
    void *base = reinterpret_cast<void *>(scope_stats_data_base);
    if (base != nullptr) {
        s_scope_stats_header = get_scope_stats_header(base);
        s_scope_stats_state = get_scope_stats_buffer_state(base, /*instance_index=*/0);
    } else {
        s_scope_stats_header = nullptr;
        s_scope_stats_state = nullptr;
    }
    // Reset collector-local statics so a prior run that crashed mid-scope (or
    // reused the same AICPU .so process) can't leak stale depth/peak data into
    // the new run's records. The shared header/free_queue is host-owned — the
    // host already zeroed it at init, so this no longer touches shared memory.
    scope_stats_depth = -1;
    s_pending_site_file = nullptr;
    s_pending_site_line = 0;
    memset(s_scope_site_file, 0, sizeof(s_scope_site_file));
    memset(s_scope_site_line, 0, sizeof(s_scope_site_line));
    memset(s_scope_ring, 0, sizeof(s_scope_ring));
    memset(s_heap_wraps, 0, sizeof(s_heap_wraps));
}

void scope_stats_aicpu_set_orch_thread_idx(int thread_idx) { s_orch_thread_idx = thread_idx; }

void scope_stats_aicpu_flush_buffers() {
    if (s_scope_stats_state == nullptr) {
        return;
    }
    rmb();
    uint64_t buf_ptr = s_scope_stats_state->current_buf_ptr;
    if (buf_ptr == 0) {
        return;  // Idempotent: nothing in flight.
    }
    ScopeStatsBuffer *buf = reinterpret_cast<ScopeStatsBuffer *>(buf_ptr);
    if (buf->count == 0) {
        return;
    }

    uint32_t seq = s_scope_stats_state->current_buf_seq;
    int rc = enqueue_ready_buffer(buf_ptr, seq);
    if (rc == 0) {
        LOG_INFO_V0("scope_stats: flushed buffer with %u records", buf->count);
    } else {
        LOG_ERROR("scope_stats: flush failed (ready_queue full), %u records dropped", buf->count);
        s_scope_stats_state->dropped_record_count += buf->count;
        buf->count = 0;
    }
    s_scope_stats_state->current_buf_ptr = 0;
    wmb();
}

// ---------------------------------------------------------------------------
// Scope lifecycle probes
// ---------------------------------------------------------------------------

extern "C" void scope_stats_set_pending_site(const char *file, int line) {
    s_pending_site_file = file;
    s_pending_site_line = line;
}

// Push depth/site and emit the begin-boundary record (ring start/end + usage).
extern "C" void scope_stats_begin(
    int ring_id, int32_t task_start, int32_t task_end, uint64_t heap_start, uint64_t heap_end, int32_t dep_pool_start,
    int32_t dep_pool_end, int32_t tensormap_used
) {
    if (!scope_stats_enabled) return;
    if (scope_stats_depth + 1 >= PTO2_SCOPE_STATS_MAX_SCOPE_DEPTH) return;
    int32_t d = ++scope_stats_depth;
    s_scope_site_file[d] = s_pending_site_file;
    s_scope_site_line[d] = s_pending_site_line;
    s_scope_ring[d] = ring_id;
    s_pending_site_file = nullptr;
    s_pending_site_line = 0;
    append_record_snapshot(
        ring_id, SCOPE_STATS_PHASE_BEGIN, task_start, task_end, heap_start, heap_end, dep_pool_start, dep_pool_end,
        tensormap_used
    );
}

// Emit the end-boundary record, then tear down depth/site.
extern "C" void scope_stats_end(
    int ring_id, int32_t task_start, int32_t task_end, uint64_t heap_start, uint64_t heap_end, int32_t dep_pool_start,
    int32_t dep_pool_end, int32_t tensormap_used
) {
    if (!scope_stats_enabled) return;
    if (scope_stats_depth < 0) return;
    int32_t d = scope_stats_depth;
    append_record_snapshot(
        ring_id, SCOPE_STATS_PHASE_END, task_start, task_end, heap_start, heap_end, dep_pool_start, dep_pool_end,
        tensormap_used
    );
    s_scope_site_file[d] = nullptr;
    s_scope_site_line[d] = 0;
    --scope_stats_depth;
}

extern "C" void scope_stats_on_fatal() {
    if (!scope_stats_enabled) return;
    if (s_scope_stats_header == nullptr) return;
    s_scope_stats_header->fatal_latched = 1;
    wmb();
    // Deliver the partial buffer to the host before the abort unwinds.
    scope_stats_aicpu_flush_buffers();
}

// ---------------------------------------------------------------------------
// Capacity registration — called by runtime at init
// ---------------------------------------------------------------------------

extern "C" void
scope_stats_set_ring_capacity(int ring_id, int32_t window_cap, uint64_t heap_cap, int32_t dep_pool_cap) {
    if (!s_scope_stats_header) return;
    if (ring_id < 0 || ring_id >= PTO2_SCOPE_STATS_MAX_RING_DEPTH) return;
    s_scope_stats_header->task_window_cap[ring_id] = window_cap;
    s_scope_stats_header->heap_cap[ring_id] = heap_cap;
    s_scope_stats_header->dep_pool_cap[ring_id] = dep_pool_cap;
}

extern "C" void scope_stats_set_tensormap_capacity(int32_t cap) {
    if (!s_scope_stats_header) return;
    s_scope_stats_header->tensormap_cap = cap;
}

extern "C" void scope_stats_note_heap_wrap(int side) {
    if (!scope_stats_enabled) return;
    if (scope_stats_depth < 0) return;  // wrap outside any scope — unattributable
    if (side != SCOPE_STATS_HEAP_SIDE_ALLOC && side != SCOPE_STATS_HEAP_SIDE_RECLAIM) return;
    int ring_id = s_scope_ring[scope_stats_depth];
    if (ring_id < 0 || ring_id >= PTO2_SCOPE_STATS_MAX_RING_DEPTH) return;
    s_heap_wraps[ring_id][side] += 1;
}
