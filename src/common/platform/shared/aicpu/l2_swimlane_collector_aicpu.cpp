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
 * @file l2_swimlane_collector_aicpu.cpp
 * @brief AICPU performance data collection implementation (SPSC free queue)
 *
 * Uses per-core L2SwimlaneAicpuTaskPool with SPSC free queues for O(1) buffer switching.
 * Host memory manager dynamically allocates replacement buffers and pushes
 * them into the free_queue. Device pops from free_queue when switching.
 */

#include "aicpu/l2_swimlane_collector_aicpu.h"

#include <cinttypes>
#include <cstring>

#include "aicpu/platform_regs.h"
#include "aicpu/profiler_device_engine.h"
#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "common/unified_log.h"

// Cached pointers for hot-path access (set during init). Phase metadata
// (num_sched_phase_threads, num_orch_phase_threads, num_phase_cores,
// core_to_thread[]) lives inside L2SwimlaneDataHeader after the phase-header
// merge; we keep a separate bool so phase-gated paths can check init-ran
// without re-reading the device-shared header.
static L2SwimlaneDataHeader *s_l2_swimlane_header = nullptr;
static bool s_phase_initialized = false;

// Per-core L2SwimlaneAicpuTaskPool cache
static L2SwimlaneAicpuTaskPool *s_aicpu_task_pools[PLATFORM_MAX_CORES] = {};

// Per-core L2SwimlaneAicoreTaskPool cache (lives in the same shared region;
// host writes initial pool + the rotation channel that AICore polls).
//
// All AICore-side bookkeeping (rotation channel, free queue,
// total_record_count, current_buf_seq) is owned by this shared struct — see
// l2_swimlane_profiling.h. We deliberately do not keep AICPU-process-local
// mirror counters because the struct's volatile fields are the single
// source of truth across init/complete/rotate/flush. The high-water-mark
// formula `total_record_count - current_buf_seq * BUFFER_SIZE` correctly
// handles the failed-rotation case (free_queue empty or ready_queue full)
// since current_buf_seq only bumps on a successful rotation.
static L2SwimlaneAicoreTaskPool *s_aicore_task_pools[PLATFORM_MAX_CORES] = {};

// Per-core AICPU-side dispatch count. Incremented on every
// `l2_swimlane_aicpu_on_aicore_dispatch` call (= once per AICore dispatch).
// When the pre-bump value is a non-zero multiple of PLATFORM_AICORE_BUFFER_SIZE,
// AICPU rotates the AICore buffer before the upcoming write_reg(DATA_MAIN_BASE).
// Single-writer per cell (the scheduler thread that owns the core).
static uint32_t s_aicore_dispatched_count[PLATFORM_MAX_CORES] = {};

// Per-core deferred AICore old-buffer enqueue (ACK-gate for the FIN-before-record
// window). The tensormap_and_ringbuffer AICore executor writes FIN before the
// swimlane record, so releasing a just-filled buffer on the FIN that gated the
// boundary dispatch could publish a buffer whose tail record has not drained
// yet. Instead the rotation stashes the old buffer here and it is released only
// once AICore ACKs the first task of the NEW buffer (reg_task_id == gate): by
// AICore's single-threaded program order that ACK proves the old buffer's last
// record dcci+dsb has landed. host_build_graph writes the record before FIN, so
// it has no such window — it never wires the ACK hook and its stashed buffer is
// released by the next-rotation / run-end backstop instead. Single-writer per
// cell (the scheduler thread that owns the core).
struct AicorePendingEnqueue {
    uint64_t buf_ptr = 0;  // 0 == nothing pending
    uint32_t buf_seq = 0;
    uint32_t gate_reg_task_id = 0;
};
static AicorePendingEnqueue s_aicore_pending_enqueue[PLATFORM_MAX_CORES] = {};

// Per-core cached current-records-buffer pointer. Written by AICPU when
// rotating buffers from inside `complete_record`. AICore writes to its own
// per-core L2SwimlaneAicoreTaskBuffer (host-allocated, AICPU rotates) and AICPU
// never reads from it on the hot path.
static L2SwimlaneAicpuTaskBuffer *s_current_aicpu_task_buffers[PLATFORM_MAX_CORES] = {};

// Per-thread sched-phase pool/buffer caches (per-scheduler-thread)
static L2SwimlaneAicpuSchedPhasePool *s_sched_phase_pools[PLATFORM_MAX_AICPU_THREADS] = {};
static L2SwimlaneAicpuSchedPhaseBuffer *s_current_sched_phase_buffers[PLATFORM_MAX_AICPU_THREADS] = {};

// Per-thread orch-phase pool/buffer caches (one orch thread).
static L2SwimlaneAicpuOrchPhasePool *s_orch_phase_pools[PLATFORM_MAX_AICPU_THREADS] = {};
static L2SwimlaneAicpuOrchPhaseBuffer *s_current_orch_phase_buffers[PLATFORM_MAX_AICPU_THREADS] = {};

static int s_orch_thread_idx = -1;

// L2 swimlane platform state. Published by the host (via dlsym'd setters on sim)
// or by the AICPU kernel entry (onboard) before perf init runs, so downstream
// perf code can discover enablement + device-base without reading the generic
// Runtime struct. Two channels (mirrors PMU):
//   - g_enable_l2_swimlane (bool) — set at kernel entry from the bitmask bit
//   - g_l2_swimlane_level (L2SwimlaneLevel) — promoted in
//     l2_swimlane_aicpu_init from the shared-memory header so
//     `>= AICPU_TIMING / SCHED_PHASES / ORCH_PHASES` gates have the granular
//     value (exposed via get_l2_swimlane_level()).
static uint64_t g_platform_l2_swimlane_base = 0;
static bool g_enable_l2_swimlane = false;
static L2SwimlaneLevel g_l2_swimlane_level = L2SwimlaneLevel::DISABLED;

// AICore rotation-table device pointer (= KernelArgs::l2_swimlane_aicore_rotation_table).
// Published by the host (sim: dlsym'd setter; onboard: from k_args via the
// kernel entry); AICPU init walks it to fill per-core &rotation addresses.
static uint64_t g_platform_l2_swimlane_aicore_rotation_table = 0;

extern "C" void set_platform_l2_swimlane_base(uint64_t l2_swimlane_data_base) {
    g_platform_l2_swimlane_base = l2_swimlane_data_base;
}
extern "C" uint64_t get_platform_l2_swimlane_base() { return g_platform_l2_swimlane_base; }
extern "C" void set_l2_swimlane_enabled(bool enable) { g_enable_l2_swimlane = enable; }
extern "C" bool is_l2_swimlane_enabled() { return g_enable_l2_swimlane; }
extern "C" void set_platform_l2_swimlane_aicore_rotation_table(uint64_t table_addr) {
    g_platform_l2_swimlane_aicore_rotation_table = table_addr;
}
extern "C" uint64_t get_platform_l2_swimlane_aicore_rotation_table() {
    return g_platform_l2_swimlane_aicore_rotation_table;
}
L2SwimlaneLevel get_l2_swimlane_level() { return g_l2_swimlane_level; }

static constexpr uint64_t kL2SwimlaneQueueBackpressureWaitCycles = PLATFORM_DFX_BACKPRESSURE_TIMEOUT_CYCLES;

template <typename BufferT>
struct L2SwimlaneDeviceModule {
    struct Context {
        L2SwimlaneDataHeader *header;
        int thread_idx;
        uint32_t core_index;
        L2SwimlaneBufferKind kind;
        BufferT **current_buf;
    };

    using DataHeader = L2SwimlaneDataHeader;
    using State = L2SwimlaneAicpuTaskPool;
    using FreeQueue = L2SwimlaneFreeQueue;
    using Buffer = BufferT;

    static constexpr uint32_t kReadyQueueSize = PLATFORM_PROF_READYQUEUE_SIZE;
    static constexpr uint32_t kSlotCount = PLATFORM_PROF_SLOT_COUNT;
    static constexpr uint64_t kBackpressureWaitCycles = kL2SwimlaneQueueBackpressureWaitCycles;

    static DataHeader *header(Context ctx) { return ctx.header; }
    static int ready_thread(Context ctx) { return ctx.thread_idx; }
    static FreeQueue *free_queue(State *state) { return &state->free_queue; }

    static uint64_t current_ptr(State *state) { return state->head.current_buf_ptr; }
    static void set_current_ptr(State *state, uint64_t ptr) { state->head.current_buf_ptr = ptr; }
    static uint32_t current_seq(State *state) { return state->head.current_buf_seq; }
    static void set_current_seq(State *state, uint32_t seq) { state->head.current_buf_seq = seq; }

    static uint32_t count(Buffer *buffer) { return buffer->count; }
    static void set_count(Buffer *buffer, uint32_t count) { buffer->count = count; }

    static void write_ready_entry(Context ctx, uint32_t tail, uint64_t buffer_ptr, uint32_t buffer_seq) {
        ctx.header->queues[ctx.thread_idx][tail].core_index = ctx.core_index;
        ctx.header->queues[ctx.thread_idx][tail].kind = ctx.kind;
        ctx.header->queues[ctx.thread_idx][tail].buffer_ptr = buffer_ptr;
        ctx.header->queues[ctx.thread_idx][tail].buffer_seq = buffer_seq;
    }

    static void account_dropped(Context, State *state, uint32_t count) {
        state->head.dropped_record_count = state->head.dropped_record_count + count;
    }
    static const char *phase_label(L2SwimlaneBufferKind kind) {
        return kind == L2SwimlaneBufferKind::AicpuOrchPhase ? "orch" : "sched";
    }
    static bool is_phase(Context ctx) {
        return ctx.kind == L2SwimlaneBufferKind::AicpuSchedPhase || ctx.kind == L2SwimlaneBufferKind::AicpuOrchPhase;
    }
    static void on_pop_success(Context ctx, State *, Buffer *buffer) {
        if (ctx.current_buf != nullptr) {
            *ctx.current_buf = buffer;
        }
    }
    static void on_current_cleared(Context ctx, State *) {
        if (ctx.current_buf != nullptr) {
            *ctx.current_buf = nullptr;
        }
    }
    static void on_no_replacement(Context ctx, State *) {
        if (is_phase(ctx)) {
            LOG_WARN(
                "Thread %d: no free %s phase buffer available, dropping records until Host catches up", ctx.thread_idx,
                phase_label(ctx.kind)
            );
        }
    }
    static void on_null_free_slot(Context, State *) {}
    static void on_enqueue_failed(Context ctx, State *, Buffer *) {
        if (is_phase(ctx)) {
            LOG_ERROR(
                "Thread %d: failed to enqueue %s phase buffer for pool %u (queue full), data lost!", ctx.thread_idx,
                phase_label(ctx.kind), ctx.core_index
            );
        } else {
            LOG_ERROR(
                "Thread %d: Core %u failed to enqueue buffer (queue full), data lost!", ctx.thread_idx, ctx.core_index
            );
        }
    }
    static void on_switch_complete(Context ctx, State *, Buffer *buffer) {
        if (buffer == nullptr) {
            return;
        }
        if (is_phase(ctx)) {
            LOG_INFO_V0(
                "Thread %d: switched to new %s phase buffer for pool %u (addr=0x%lx)", ctx.thread_idx,
                phase_label(ctx.kind), ctx.core_index, reinterpret_cast<uint64_t>(buffer)
            );
        } else {
            LOG_INFO_V0(
                "Thread %d: Core %u switched to new buffer (addr=0x%lx)", ctx.thread_idx, ctx.core_index,
                reinterpret_cast<uint64_t>(buffer)
            );
        }
    }
};

using L2SwimlaneTaskDeviceModule = L2SwimlaneDeviceModule<L2SwimlaneAicpuTaskBuffer>;
using L2SwimlaneTaskEngine = profiling_device::DeviceProfilerEngine<L2SwimlaneTaskDeviceModule>;

template <typename Buffer>
static typename L2SwimlaneDeviceModule<Buffer>::Context
l2_buffer_context(int thread_idx, uint32_t core_index, L2SwimlaneBufferKind kind, Buffer **current_buf) {
    return typename L2SwimlaneDeviceModule<Buffer>::Context{
        s_l2_swimlane_header, thread_idx, core_index, kind, current_buf
    };
}

static L2SwimlaneTaskDeviceModule::Context l2_task_context(
    int thread_idx, uint32_t core_index, L2SwimlaneBufferKind kind, L2SwimlaneAicpuTaskBuffer **current_buf
) {
    return l2_buffer_context(thread_idx, core_index, kind, current_buf);
}

static bool wait_for_free_queue_entry(L2SwimlaneFreeQueue *free_queue, uint32_t *head_out, uint32_t *tail_out) {
    return L2SwimlaneTaskEngine::wait_for_free_queue_entry(s_l2_swimlane_header, free_queue, head_out, tail_out);
}

/**
 * Enqueue ready buffer to per-thread queue
 *
 * @param header L2SwimlaneDataHeader pointer
 * @param thread_idx AICPU thread index (selects the per-thread ready queue)
 * @param core_index Core index for task entries, or pool ordinal for phase entries
 * @param buffer_ptr Device pointer to the full buffer
 * @param buffer_seq Sequence number for ordering
 * @param kind Buffer kind discriminator (see L2SwimlaneBufferKind)
 * @return 0 on success, -1 if queue full
 */
static int enqueue_ready_buffer(
    L2SwimlaneDataHeader *header, int thread_idx, uint32_t core_index, uint64_t buffer_ptr, uint32_t buffer_seq,
    L2SwimlaneBufferKind kind
) {
    auto ctx = L2SwimlaneTaskDeviceModule::Context{header, thread_idx, core_index, kind, nullptr};
    return L2SwimlaneTaskEngine::enqueue_ready(ctx, buffer_ptr, buffer_seq);
}

static L2SwimlaneAicpuTaskBuffer *
try_pop_records_buffer(int core_id, L2SwimlaneAicpuTaskPool *state, uint32_t next_seq) {
    return L2SwimlaneTaskEngine::pop_free(
        l2_task_context(/*thread_idx=*/0, static_cast<uint32_t>(core_id), L2SwimlaneBufferKind::AicpuTask,
                        &s_current_aicpu_task_buffers[core_id]),
        state, next_seq
    );
}

void l2_swimlane_aicpu_init(int worker_count) {
    // Reset cross-launch state up front. AICPU statics persist across launches
    // on the same loaded .so; without this reset, an enabled→disabled launch
    // sequence would leave s_phase_initialized=true from the prior run, and
    // any subsequent record_sched_phase / record_orch_phase call would
    // dereference the prior launch's (now-freed) s_sched_phase_pools /
    // s_orch_phase_pools pointers. Same shape as the [[block_local]] reset
    // in onboard/aicore/kernel.cpp for the AICore-side rotation slot
    // (fixed in #936).
    s_phase_initialized = false;

    // Reset AICore dispatch-count bookkeeping for the same reason: the next
    // launch must start counting from 0 so the rotation boundary check
    // (count % BUFFER_SIZE == 0) lands on the right dispatches. Stale values
    // from a prior launch would skip the first rotation (count already past a
    // boundary) or trigger one prematurely. The deferred-enqueue stash must be
    // cleared too: a prior launch that ended with a buffer still stashed (gate
    // ACK never observed, or run-end flush hit a full queue) would otherwise
    // leave a dangling buf_ptr that a matching reg_task_id in this launch could
    // flush — freeing a buffer that no longer belongs to us.
    for (int i = 0; i < PLATFORM_MAX_CORES; i++) {
        s_aicore_dispatched_count[i] = 0;
        s_aicore_pending_enqueue[i] = AicorePendingEnqueue{};
    }

    void *l2_swimlane_base = reinterpret_cast<void *>(g_platform_l2_swimlane_base);
    if (l2_swimlane_base == nullptr) {
        LOG_ERROR("l2_swimlane_data_base is NULL, cannot initialize profiling");
        return;
    }

    s_l2_swimlane_header = get_l2_swimlane_header(l2_swimlane_base);

    // Read the granular perf_level from the shared-memory header (host wrote
    // it in L2SwimlaneCollector::initialize). The kernel-entry setter only seeded
    // the binary g_enable_l2_swimlane via the bitmask bit.
    g_l2_swimlane_level = static_cast<L2SwimlaneLevel>(s_l2_swimlane_header->l2_swimlane_level);

    LOG_INFO_V0(
        "Initializing performance profiling for %d cores (free queue), l2_swimlane_level=%u", worker_count,
        static_cast<uint32_t>(g_l2_swimlane_level)
    );

    // Populate the per-core AICore head device-address table. AICore reads
    // `l2_swimlane_aicore_rotation_table[block_idx]` from KernelArgs to find
    // its `L2SwimlaneActiveHead` cache line; the table itself is
    // host-allocated, but the entries are device-internal addresses
    // (`&ac_state->head`) that the host would otherwise have to translate
    // from host-mapped to device-mapped. AICPU already runs on the device,
    // so it can write the addresses directly without any translation — that
    // keeps the host side decoupled from the AICore shared-memory layout.
    uint64_t *head_table = reinterpret_cast<uint64_t *>(g_platform_l2_swimlane_aicore_rotation_table);

    // Pop first buffer from free_queue for each core
    for (int i = 0; i < worker_count; i++) {
        L2SwimlaneAicpuTaskPool *state = get_perf_buffer_state(l2_swimlane_base, i);
        L2SwimlaneAicoreTaskPool *ac_state = get_aicore_buffer_state(l2_swimlane_base, worker_count, i);

        s_aicpu_task_pools[i] = state;
        s_aicore_task_pools[i] = ac_state;

        if (head_table != nullptr) {
            head_table[i] = reinterpret_cast<uint64_t>(&ac_state->head);
        }

        // Pop first buffer from free_queue
        rmb();
        uint32_t head = state->free_queue.head;
        uint32_t tail = state->free_queue.tail;

        if (head != tail) {
            uint64_t buf_ptr = state->free_queue.buffer_ptrs[head % PLATFORM_PROF_SLOT_COUNT];
            rmb();
            state->free_queue.head = head + 1;
            state->head.current_buf_ptr = buf_ptr;
            state->head.current_buf_seq = 0;
            wmb();

            L2SwimlaneAicpuTaskBuffer *buf = reinterpret_cast<L2SwimlaneAicpuTaskBuffer *>(buf_ptr);
            buf->count = 0;
            s_current_aicpu_task_buffers[i] = buf;

            LOG_DEBUG("Core %d: popped initial buffer (addr=0x%lx)", i, buf_ptr);
        } else {
            LOG_ERROR("Core %d: free_queue is empty during init!", i);
            state->head.current_buf_ptr = 0;
            s_current_aicpu_task_buffers[i] = nullptr;
        }

        // Prime the AICore head channel with the initial buffer. Seq starts
        // at 0; AICore's local `cached_buf_seq` defaults to UINT32_MAX so the
        // first record_task call observes a mismatch and loads the buffer.
        rmb();
        uint32_t ac_head = ac_state->free_queue.head;
        uint32_t ac_tail = ac_state->free_queue.tail;
        if (ac_head != ac_tail) {
            uint64_t ac_buf_ptr = ac_state->free_queue.buffer_ptrs[ac_head % PLATFORM_PROF_SLOT_COUNT];
            rmb();
            ac_state->free_queue.head = ac_head + 1;
            // Same publish pattern as aicore_rotate: ptr first, then a fence,
            // then seq. AICore lazy-resolves the head on its first task, so
            // strict ordering here matters only if AICore is ever changed to
            // start polling before the first dispatch — keeping the patterns
            // aligned future-proofs that.
            ac_state->head.current_buf_ptr = ac_buf_ptr;
            wmb();
            ac_state->head.current_buf_seq = 0;
            wmb();
            L2SwimlaneAicoreTaskBuffer *ac_buf = reinterpret_cast<L2SwimlaneAicoreTaskBuffer *>(ac_buf_ptr);
            ac_buf->count = 0;
            LOG_DEBUG("Core %d: primed AICore head with buf=0x%lx, seq=0", i, ac_buf_ptr);
        } else {
            LOG_ERROR("Core %d: AICore free_queue is empty during init!", i);
            ac_state->head.current_buf_ptr = 0;
            ac_state->head.current_buf_seq = 0;
            wmb();
        }
    }

    wmb();

    LOG_INFO_V0("Performance profiling initialized for %d cores (with AICore rotation)", worker_count);
}

/**
 * Internal records-buffer rotation. Called from `l2_swimlane_aicpu_complete_task`
 * after a record is committed and the buffer hits capacity. Only swaps an
 * AICPU-private records pointer — AICore reads from a stable ring and is
 * unaffected by this call.
 */
static void switch_task_buffer(int core_id, int thread_idx) {
    L2SwimlaneAicpuTaskPool *state = s_aicpu_task_pools[core_id];
    if (state == nullptr) {
        return;
    }

    auto *full_buf = s_current_aicpu_task_buffers[core_id];
    if (full_buf == nullptr) {
        return;
    }
    LOG_INFO_V0("Thread %d: Core %d buffer is full (count=%u)", thread_idx, core_id, full_buf->count);
    L2SwimlaneTaskEngine::switch_buffer(
        l2_task_context(
            thread_idx, static_cast<uint32_t>(core_id), L2SwimlaneBufferKind::AicpuTask,
            &s_current_aicpu_task_buffers[core_id]
        ),
        state
    );
}

// Release a deferred AICore buffer (stashed by aicore_rotate) to the ready
// queue. Shared by the ACK hook (prompt release), the next-rotation backstop,
// and run-end flush. No-op when nothing is pending. On ready-queue-full the
// buffer stays stashed for a later retry; the same no-double-count rationale as
// the rotation queue-full branch applies (reconcile reports silent_loss).
static void publish_aicore_pending_buffer(int core_id, int thread_idx) {
    AicorePendingEnqueue &pe = s_aicore_pending_enqueue[core_id];
    if (pe.buf_ptr == 0) {
        return;
    }
    L2SwimlaneAicoreTaskBuffer *old_buf = reinterpret_cast<L2SwimlaneAicoreTaskBuffer *>(pe.buf_ptr);
    old_buf->count = static_cast<uint32_t>(PLATFORM_AICORE_BUFFER_SIZE);
    wmb();
    int rc = enqueue_ready_buffer(
        s_l2_swimlane_header, thread_idx, core_id, pe.buf_ptr, pe.buf_seq, L2SwimlaneBufferKind::AicoreTask
    );
    if (rc != 0) {
        LOG_ERROR(
            "Thread %d: Core %d failed to enqueue deferred AICore buffer (queue full); will retry at flush", thread_idx,
            core_id
        );
        return;
    }
    pe.buf_ptr = 0;
}

// Rotate the AICore buffer for `core_id` at a PLATFORM_AICORE_BUFFER_SIZE
// dispatch boundary: pop a fresh buffer from free_queue, publish it via the head
// channel, and DEFER the just-filled buffer's enqueue via
// s_aicore_pending_enqueue (see that declaration for the FIN-before-record
// rationale). `new_buf_first_reg_task_id` is the reg_task_id of the boundary
// dispatch — the first task written into the new buffer — used as the ACK gate
// that later releases the stashed buffer. On success bumps
// `ac_state->head.current_buf_seq`; on empty free queue the old buffer is kept
// active in place (AICore overflows it, drop count grows).
static void aicore_rotate(int core_id, int thread_idx, uint32_t new_buf_first_reg_task_id) {
    L2SwimlaneAicoreTaskPool *ac_state = s_aicore_task_pools[core_id];
    if (ac_state == nullptr) {
        return;
    }

    uint64_t old_buf_ptr = ac_state->head.current_buf_ptr;
    uint32_t seq = ac_state->head.current_buf_seq;

    uint32_t head = 0;
    uint32_t tail = 0;
    if (!wait_for_free_queue_entry(&ac_state->free_queue, &head, &tail)) {
        // No replacement available — AICore continues to write into the old
        // buffer; its slot counter will hit BUFFER_SIZE and the slot guard
        // silently drops further records. We deliberately do NOT bump
        // dropped_record_count here: AICPU has no precise view of how many
        // tasks will actually fall in this gap before the run ends. The
        // pre-emptive BUFFER_SIZE bump that used to live here over-counted
        // when the run ended early — the old buffer's already-written
        // records still flushed (counted toward `collected`), and the
        // pre-emptive bump on top of that broke the
        // `collected + dropped == total` reconcile invariant. The drop is
        // visible at reconcile time as silent loss
        // (`total - collected - dropped > 0`) and the WARN below records
        // the failure mode.
        LOG_WARN(
            "Thread %d: Core %d AICore free_queue empty at rotation; AICore slot guard will drop overflow records",
            thread_idx, core_id
        );
        return;
    }

    uint64_t new_buf_ptr = ac_state->free_queue.buffer_ptrs[head % PLATFORM_PROF_SLOT_COUNT];
    rmb();
    if (new_buf_ptr == 0) {
        LOG_WARN(
            "Thread %d: Core %d AICore free_queue returned a null buffer at rotation; keeping old buffer active",
            thread_idx, core_id
        );
        return;
    }

    // Defer the just-filled buffer's enqueue behind the ACK gate instead of
    // handing it to the host now (see s_aicore_pending_enqueue). First drain any
    // buffer still stashed from a prior rotation whose gate ACK was never
    // observed: its gate task completed a full BUFFER_SIZE batch ago, so it is
    // long safe to release. If that drain fails (ready queue full) the older
    // buffer is overwritten below and shows up as silent_loss at reconcile —
    // the same lossy regime the pre-defer code hit on a queue-full enqueue.
    if (old_buf_ptr != 0) {
        publish_aicore_pending_buffer(core_id, thread_idx);
        AicorePendingEnqueue &pe = s_aicore_pending_enqueue[core_id];
        pe.buf_ptr = old_buf_ptr;
        pe.buf_seq = seq;
        pe.gate_reg_task_id = new_buf_first_reg_task_id;
    }

    // Pop next buffer from free_queue and publish via the head channel.
    // Publish order matters: AICore observes head.current_buf_seq change to
    // detect rotation, then reads head.current_buf_ptr. Write ptr first so
    // AICore can never see a new seq with a stale ptr. new_buf->count=0 must
    // also be visible before AICore's slot writes begin.
    ac_state->free_queue.head = head + 1;
    L2SwimlaneAicoreTaskBuffer *new_buf = reinterpret_cast<L2SwimlaneAicoreTaskBuffer *>(new_buf_ptr);
    new_buf->count = 0;

    wmb();
    ac_state->head.current_buf_ptr = new_buf_ptr;
    wmb();
    ac_state->head.current_buf_seq = seq + 1;
    wmb();
}

// Pre-dispatch hook. Called from the dispatch path (scheduler_dispatch in
// tensormap_and_ringbuffer; aicpu_executor in host_build_graph) immediately
// before `write_reg(DATA_MAIN_BASE)` for each AICore task. Maintains the
// per-core dispatch count and rotates the AICore buffer when the count is
// about to cross a PLATFORM_AICORE_BUFFER_SIZE boundary.
//
// Race safety: rotation runs before the dispatch register write and publishes
// the new buffer to AICore, but does NOT hand the old buffer to the host here.
// The old buffer is stashed and released only once AICore ACKs the first task
// of the new buffer (l2_swimlane_aicpu_on_aicore_ack) — the completion-before-
// dispatch invariant proves prior tasks FIN'd, but tensormap_and_ringbuffer
// writes FIN before the swimlane record, so FIN alone does not prove the tail
// record has drained. `reg_task_id` is that boundary dispatch's token, stored
// as the ACK gate.
//
// total_record_count accounting also lives here: one AICore record == one
// dispatch, so the dispatch count IS the AICore-side total. Bumping here
// (instead of inside complete_task) means level=1 (AICORE_TIMING-only) gets
// accurate reconcile counts even when complete_task is bypassed.
void l2_swimlane_aicpu_on_aicore_dispatch(int core_id, int thread_idx, uint32_t reg_task_id) {
    if (!g_enable_l2_swimlane) {
        return;
    }
    if (core_id < 0 || core_id >= PLATFORM_MAX_CORES) {
        return;
    }
    L2SwimlaneAicoreTaskPool *ac_state = s_aicore_task_pools[core_id];
    if (ac_state == nullptr) {
        return;
    }
    uint32_t prev = s_aicore_dispatched_count[core_id];
    // Rotate exactly on the first dispatch of each non-initial BUFFER_SIZE
    // batch (prev = BUFFER_SIZE, 2*BUFFER_SIZE, ...). PLATFORM_AICORE_BUFFER_SIZE
    // is asserted power-of-two so the mod lowers to a bitwise AND.
    if (prev > 0 && (prev & (PLATFORM_AICORE_BUFFER_SIZE - 1)) == 0) {
        aicore_rotate(core_id, thread_idx, reg_task_id);
    }
    s_aicore_dispatched_count[core_id] = prev + 1;
    ac_state->head.total_record_count += 1;
}

void l2_swimlane_aicpu_on_aicore_ack(int core_id, int thread_idx, uint32_t reg_task_id) {
    if (core_id < 0 || core_id >= PLATFORM_MAX_CORES) {
        return;
    }
    AicorePendingEnqueue &pe = s_aicore_pending_enqueue[core_id];
    // Fast path: nothing stashed, or this is not the gating ACK/FIN. gate_reg_task_id
    // is the boundary dispatch's token; the first COND event carrying it means
    // AICore reached at least ACK of the new buffer's first task, hence passed
    // record_task + dsb of the old buffer's last record. reg_task_id is unique
    // per BUFFER_SIZE window, so this cannot fire early.
    if (pe.buf_ptr == 0 || pe.gate_reg_task_id != reg_task_id) {
        return;
    }
    publish_aicore_pending_buffer(core_id, thread_idx);
}

int l2_swimlane_aicpu_complete_task(
    int core_id, int thread_idx, uint32_t reg_task_id, uint64_t dispatch_time, uint64_t finish_time
) {
    if (core_id < 0 || core_id >= PLATFORM_MAX_CORES) {
        return -1;
    }
    L2SwimlaneAicpuTaskPool *state = s_aicpu_task_pools[core_id];
    if (state == nullptr) {
        return -1;
    }

    // Account every commit attempt up front so host can detect silent loss as
    // `device_total - (collected + dropped)`.
    state->head.total_record_count += 1;

    L2SwimlaneAicpuTaskBuffer *l2_swimlane_buf = s_current_aicpu_task_buffers[core_id];
    if (l2_swimlane_buf == nullptr) {
        l2_swimlane_buf = try_pop_records_buffer(core_id, state, state->head.current_buf_seq);
        if (l2_swimlane_buf == nullptr) {
            // No active records buffer (init ran out of free buffers or host has
            // not refilled after the last published full buffer); count as drop
            // so host reconciliation stays consistent.
            state->head.dropped_record_count += 1;
            return -1;
        }
    }
    uint32_t count = l2_swimlane_buf->count;
    if (count >= PLATFORM_PROF_BUFFER_SIZE) {
        // Defensive: should not happen because we rotate at end of every commit.
        state->head.dropped_record_count += 1;
        return -1;
    }

    // AICPU-only timing — three fields, two cache half-lines. Identity
    // (task_token_raw, core_type) lives in the AICore record; the host
    // joins by reg_task_id. See L2SwimlaneAicpuTaskRecord header comment.
    L2SwimlaneAicpuTaskRecord *record = &l2_swimlane_buf->records[count];
    record->reg_task_id = reg_task_id;
    record->dispatch_time = dispatch_time;
    record->finish_time = finish_time;

    uint32_t new_count = count + 1;
    l2_swimlane_buf->count = new_count;

    // Rotate AICpu's L2SwimlaneAicpuTaskBuffer after the write so the just-committed
    // record is preserved.
    if (new_count >= PLATFORM_PROF_BUFFER_SIZE) {
        switch_task_buffer(core_id, thread_idx);
    }

    // AICore-pool stats (total_record_count) are bumped on the dispatch side,
    // not here. See l2_swimlane_aicpu_on_aicore_dispatch — counting per
    // dispatch keeps reconcile counts accurate even at level=1 where this
    // function never runs.
    return 0;
}

void l2_swimlane_aicpu_flush(int thread_idx, const int *cur_thread_cores, int core_num) {
    if (!g_enable_l2_swimlane) {
        return;
    }

    void *l2_swimlane_base = reinterpret_cast<void *>(g_platform_l2_swimlane_base);
    if (l2_swimlane_base == nullptr) {
        return;
    }

    rmb();

    LOG_INFO_V0("Thread %d: Flushing performance buffers for %d cores", thread_idx, core_num);

    int flushed_count = 0;

    for (int i = 0; i < core_num; i++) {
        int core_id = cur_thread_cores[i];
        L2SwimlaneAicpuTaskPool *state = s_aicpu_task_pools[core_id];
        if (state == nullptr) continue;

        rmb();
        uint64_t buf_ptr = state->head.current_buf_ptr;
        if (buf_ptr == 0) {
            // No active buffer
        } else {
            L2SwimlaneAicpuTaskBuffer *buf = reinterpret_cast<L2SwimlaneAicpuTaskBuffer *>(buf_ptr);
            if (buf->count > 0) {
                uint32_t seq = state->head.current_buf_seq;
                int rc = enqueue_ready_buffer(
                    s_l2_swimlane_header, thread_idx, core_id, buf_ptr, seq, L2SwimlaneBufferKind::AicpuTask
                );
                if (rc == 0) {
                    LOG_INFO_V0("Thread %d: Core %d flushed buffer with %u records", thread_idx, core_id, buf->count);
                    flushed_count++;
                    state->head.current_buf_ptr = 0;
                    s_current_aicpu_task_buffers[core_id] = nullptr;
                    wmb();
                } else {
                    // ready_queue full at end-of-run: account the loss and clear the
                    // buffer so host reconcile sees a clean state (current_buf_ptr=0)
                    // and dropped == flush failures rather than ring/task_id mismatch.
                    LOG_ERROR(
                        "Thread %d: Core %d failed to enqueue buffer (queue full), %u records lost!", thread_idx,
                        core_id, buf->count
                    );
                    state->head.dropped_record_count = state->head.dropped_record_count + buf->count;
                    buf->count = 0;
                    state->head.current_buf_ptr = 0;
                    s_current_aicpu_task_buffers[core_id] = nullptr;
                    wmb();
                }
            }
        }

        // Also flush the current AICore buffer to the ready queue so the host
        // sees this session's final batch of AICore timestamps.
        //
        // High-water mark uses the rotation accounting (total_record_count -
        // current_buf_seq * BUFFER_SIZE). total_record_count is bumped per
        // dispatch in l2_swimlane_aicpu_on_aicore_dispatch and is therefore
        // accurate at all levels — including level=1 where complete_task is
        // bypassed. The formula clamps to BUFFER_SIZE if an earlier rotation
        // failed (no free buffer), so we never stamp a partial count when
        // the buffer is actually full.
        L2SwimlaneAicoreTaskPool *ac_state = s_aicore_task_pools[core_id];
        if (ac_state == nullptr) continue;

        // Drain any ACK-gated buffer still stashed from a rotation whose gate
        // ACK was never observed before teardown. By run end every AICore task
        // has completed, so the buffer's records are fully drained. Enqueue it
        // before the current buffer so ready-queue order follows buffer_seq.
        publish_aicore_pending_buffer(core_id, thread_idx);

        rmb();
        uint64_t ac_buf_ptr = ac_state->head.current_buf_ptr;
        if (ac_buf_ptr == 0) continue;

        // At AICPU_TIMING+, `total_record_count` is bumped on every complete
        // and gives an accurate live count for the current buffer. At
        // AICORE_TIMING (level=1) complete_task is skipped, so that counter
        // stays 0 and the formula bails even when AICore has filled records.
        // Fall back to the buffer's full capacity in that case; the host-side
        // copy_aicore_buffer skips trailing slots whose start_time is still 0,
        // so over-stating count costs only a scan pass — never spurious records.
        uint32_t ac_mark;
        if (g_l2_swimlane_level >= L2SwimlaneLevel::AICPU_TIMING) {
            uint32_t live = ac_state->head.total_record_count -
                            ac_state->head.current_buf_seq * static_cast<uint32_t>(PLATFORM_AICORE_BUFFER_SIZE);
            if (live == 0) {
                continue;
            }
            ac_mark = (live > static_cast<uint32_t>(PLATFORM_AICORE_BUFFER_SIZE)) ?
                          static_cast<uint32_t>(PLATFORM_AICORE_BUFFER_SIZE) :
                          live;
        } else {
            ac_mark = static_cast<uint32_t>(PLATFORM_AICORE_BUFFER_SIZE);
        }
        L2SwimlaneAicoreTaskBuffer *ac_buf = reinterpret_cast<L2SwimlaneAicoreTaskBuffer *>(ac_buf_ptr);
        ac_buf->count = ac_mark;
        wmb();

        uint32_t ac_seq = ac_state->head.current_buf_seq;
        int rc = enqueue_ready_buffer(
            s_l2_swimlane_header, thread_idx, core_id, ac_buf_ptr, ac_seq, L2SwimlaneBufferKind::AicoreTask
        );
        if (rc == 0) {
            LOG_INFO_V0(
                "Thread %d: Core %d flushed AICore buffer (seq=%u, count=%u)", thread_idx, core_id, ac_seq, ac_mark
            );
            ac_state->head.current_buf_ptr = 0;
            wmb();
        } else {
            LOG_ERROR("Thread %d: Core %d failed to enqueue AICore buffer at flush (queue full)", thread_idx, core_id);
            ac_state->head.dropped_record_count = ac_state->head.dropped_record_count + ac_mark;
            ac_state->head.current_buf_ptr = 0;
            wmb();
        }
    }

    wmb();

    LOG_INFO_V0("Thread %d: Performance buffer flush complete, %d buffers flushed", thread_idx, flushed_count);
}

// Pop the first buffer from a pool's free_queue and cache it as the current
// active buffer. Uses the same engine path as subsequent phase rotations.
template <typename Buffer>
static Buffer *prime_phase_pool(
    L2SwimlaneAicpuTaskPool *state, int thread_idx, L2SwimlaneBufferKind kind, Buffer **current_buf_out,
    const char *kind_label
) {
    auto ctx = l2_buffer_context(thread_idx, 0, kind, current_buf_out);
    Buffer *buf = profiling_device::DeviceProfilerEngine<L2SwimlaneDeviceModule<Buffer>>::try_pop_free(
        ctx, state, /*next_seq=*/0
    );
    if (buf == nullptr) {
        LOG_ERROR("Thread %d: %s phase free_queue is empty during init!", thread_idx, kind_label);
        if (current_buf_out != nullptr) {
            *current_buf_out = nullptr;
        }
        state->head.current_buf_ptr = 0;
        return nullptr;
    }

    LOG_DEBUG(
        "Thread %d: popped initial %s phase buffer (addr=0x%lx)", thread_idx, kind_label,
        reinterpret_cast<uint64_t>(buf)
    );
    return buf;
}

void l2_swimlane_aicpu_init_phase(int worker_count, int num_sched_phase_threads, int num_orch_phase_threads) {
    void *l2_swimlane_base = reinterpret_cast<void *>(g_platform_l2_swimlane_base);
    if (l2_swimlane_base == nullptr) {
        LOG_ERROR("l2_swimlane_data_base is NULL, cannot initialize phase profiling");
        return;
    }

    s_l2_swimlane_header = get_l2_swimlane_header(l2_swimlane_base);

    s_l2_swimlane_header->num_sched_phase_threads = static_cast<uint32_t>(num_sched_phase_threads);
    s_l2_swimlane_header->num_orch_phase_threads = static_cast<uint32_t>(num_orch_phase_threads);
    s_l2_swimlane_header->num_phase_cores = 0;
    memset(s_l2_swimlane_header->core_to_thread, -1, sizeof(s_l2_swimlane_header->core_to_thread));
    s_phase_initialized = true;

    int sched_n = num_sched_phase_threads;
    if (sched_n > PLATFORM_MAX_AICPU_THREADS) sched_n = PLATFORM_MAX_AICPU_THREADS;
    int orch_n = num_orch_phase_threads;
    if (orch_n > PLATFORM_MAX_AICPU_THREADS) orch_n = PLATFORM_MAX_AICPU_THREADS;

    for (int t = 0; t < sched_n; t++) {
        auto *state = get_sched_phase_buffer_state(l2_swimlane_base, worker_count, t);
        s_sched_phase_pools[t] = state;
        s_current_sched_phase_buffers[t] = prime_phase_pool<L2SwimlaneAicpuSchedPhaseBuffer>(
            state, t, L2SwimlaneBufferKind::AicpuSchedPhase, &s_current_sched_phase_buffers[t], "sched"
        );
    }
    for (int t = sched_n; t < PLATFORM_MAX_AICPU_THREADS; t++) {
        s_sched_phase_pools[t] = nullptr;
        s_current_sched_phase_buffers[t] = nullptr;
    }

    for (int t = 0; t < orch_n; t++) {
        auto *state = get_orch_phase_buffer_state(l2_swimlane_base, worker_count, t);
        s_orch_phase_pools[t] = state;
        s_current_orch_phase_buffers[t] = prime_phase_pool<L2SwimlaneAicpuOrchPhaseBuffer>(
            state, t, L2SwimlaneBufferKind::AicpuOrchPhase, &s_current_orch_phase_buffers[t], "orch"
        );
    }
    for (int t = orch_n; t < PLATFORM_MAX_AICPU_THREADS; t++) {
        s_orch_phase_pools[t] = nullptr;
        s_current_orch_phase_buffers[t] = nullptr;
    }

    wmb();

    LOG_INFO_V0(
        "Phase profiling initialized: %d sched threads, %d orch threads, %d records/thread", num_sched_phase_threads,
        num_orch_phase_threads, PLATFORM_PHASE_RECORDS_PER_THREAD
    );
}

// Generic phase-buffer switch. Enqueue the full buffer to its thread's
// ready queue under `kind`, then pop a fresh buffer from free_queue. Sets
// `*current_buf_out` to nullptr if no free buffer is available — subsequent
// records on that thread will drop until the host catches up.
// `thread_idx` is the AICPU thread doing the enqueue (always the caller); it
// selects that thread's own SPSC ready queue, which it must own exclusively.
// `pool_idx` is the pool ordinal the host uses to file records and recycle the
// buffer to that pool (the same ordinal indexes the output lane). For sched
// pools the two coincide (thread t → queue t, pool t); for the single orch
// instance they differ (orchestrator's thread, but pool ordinal 0).
template <typename Buffer>
static void switch_phase_buffer(
    int thread_idx, uint32_t pool_idx, L2SwimlaneAicpuTaskPool *state, Buffer **current_buf_out,
    L2SwimlaneBufferKind kind, const char *kind_label
) {
    Buffer *full_buf = *current_buf_out;
    if (state == nullptr || full_buf == nullptr) return;

    LOG_INFO_V0("Thread %d: %s phase buffer is full (count=%u)", thread_idx, kind_label, full_buf->count);

    auto ctx = l2_buffer_context(thread_idx, pool_idx, kind, current_buf_out);
    profiling_device::DeviceProfilerEngine<L2SwimlaneDeviceModule<Buffer>>::switch_buffer(ctx, state);
}

// Acquire a writable slot in the per-thread phase buffer. Handles the
// nullptr-recover path (a prior switch couldn't pop a free buffer) and the
// buffer-full → switch path. Returns nullptr if the record must be dropped;
// callers should bump `dropped_record_count` and return when nullptr.
template <typename Buffer, typename Record>
static Record *acquire_phase_slot(
    int thread_idx, uint32_t pool_idx, L2SwimlaneAicpuTaskPool *state, Buffer **current_buf_out,
    L2SwimlaneBufferKind kind, const char *kind_label
) {
    Buffer *buf = *current_buf_out;
    if (buf == nullptr) {
        auto ctx = l2_buffer_context(thread_idx, pool_idx, kind, current_buf_out);
        buf = profiling_device::DeviceProfilerEngine<L2SwimlaneDeviceModule<Buffer>>::pop_free(
            ctx, state, state->head.current_buf_seq
        );
        if (buf != nullptr) {
            LOG_INFO_V0("Thread %d: recovered %s phase buffer", thread_idx, kind_label);
        }
        if (buf == nullptr) return nullptr;
    }

    uint32_t idx = buf->count;
    if (idx >= PLATFORM_PHASE_RECORDS_PER_THREAD) {
        switch_phase_buffer(thread_idx, pool_idx, state, current_buf_out, kind, kind_label);
        buf = *current_buf_out;
        if (buf == nullptr) return nullptr;
        idx = buf->count;
        if (idx >= PLATFORM_PHASE_RECORDS_PER_THREAD) return nullptr;
    }
    Record *record = &buf->records[idx];
    buf->count = idx + 1;
    return record;
}

static inline L2SwimlaneAicpuSchedPhaseRecord *acquire_sched_phase_record(int thread_idx) {
    if (!s_phase_initialized) return nullptr;
    auto *state = s_sched_phase_pools[thread_idx];
    if (state == nullptr) return nullptr;

    state->head.total_record_count += 1;

    auto *record = acquire_phase_slot<L2SwimlaneAicpuSchedPhaseBuffer, L2SwimlaneAicpuSchedPhaseRecord>(
        /*thread_idx=*/thread_idx, /*pool_idx=*/static_cast<uint32_t>(thread_idx), state,
        &s_current_sched_phase_buffers[thread_idx], L2SwimlaneBufferKind::AicpuSchedPhase, "sched"
    );
    if (record == nullptr) {
        state->head.dropped_record_count += 1;
        return nullptr;
    }
    return record;
}

static inline void fill_sched_phase_record(
    L2SwimlaneAicpuSchedPhaseRecord *record, L2SwimlaneSchedPhaseKind kind, uint64_t start_time, uint64_t end_time,
    uint32_t loop_iter, uint32_t tasks_processed, const int16_t *shared_at_start, const int16_t *shared_at_end
) {
    record->start_time = start_time;
    record->end_time = end_time;
    record->loop_iter = loop_iter;
    record->kind = kind;
    record->tasks_processed = tasks_processed;
    auto copy_snapshot = [](int16_t dst[L2SWIMLANE_NUM_QUEUE_SHAPES], const int16_t *src) {
        if (src == nullptr) {
            for (int i = 0; i < L2SWIMLANE_NUM_QUEUE_SHAPES; i++)
                dst[i] = 0;
        } else {
            for (int i = 0; i < L2SWIMLANE_NUM_QUEUE_SHAPES; i++)
                dst[i] = src[i];
        }
    };
    copy_snapshot(record->shared_depth_at_start, shared_at_start);
    copy_snapshot(record->shared_depth_at_end, shared_at_end);
}

void l2_swimlane_aicpu_record_sched_phase(
    int thread_idx, L2SwimlaneSchedPhaseKind kind, uint64_t start_time, uint64_t end_time, uint32_t loop_iter,
    uint32_t tasks_processed, uint32_t pop_hit, uint32_t pop_miss, const int16_t *shared_at_start,
    const int16_t *shared_at_end
) {
    auto *record = acquire_sched_phase_record(thread_idx);
    if (record == nullptr) return;
    fill_sched_phase_record(
        record, kind, start_time, end_time, loop_iter, tasks_processed, shared_at_start, shared_at_end
    );
    record->phase_data.dispatch.pop_hit = pop_hit;
    record->phase_data.dispatch.pop_miss = pop_miss;
}

static void record_aicpu_worker_task(
    int thread_idx, L2SwimlaneSchedPhaseKind kind, uint64_t complete_time, uint32_t loop_iter, uint64_t task_id
) {
    auto *record = acquire_sched_phase_record(thread_idx);
    if (record == nullptr) return;
    fill_sched_phase_record(
        record, kind, complete_time, complete_time, loop_iter, /*tasks_processed=*/1, /*shared_at_start=*/nullptr,
        /*shared_at_end=*/nullptr
    );
    record->phase_data.dummy_task.local_id = static_cast<uint32_t>(task_id);
    record->phase_data.dummy_task.ring_id = static_cast<uint32_t>(task_id >> 32);
}

void l2_swimlane_aicpu_record_dummy_task(int thread_idx, uint64_t complete_time, uint32_t loop_iter, uint64_t task_id) {
    record_aicpu_worker_task(thread_idx, L2SwimlaneSchedPhaseKind::DummyTask, complete_time, loop_iter, task_id);
}

void l2_swimlane_aicpu_record_predicated_skip(
    int thread_idx, uint64_t complete_time, uint32_t loop_iter, uint64_t task_id
) {
    record_aicpu_worker_task(thread_idx, L2SwimlaneSchedPhaseKind::PredicatedSkip, complete_time, loop_iter, task_id);
}

void l2_swimlane_aicpu_record_graph_prepare(
    int thread_idx, uint64_t start_time, uint64_t end_time, uint32_t loop_iter, uint64_t task_id,
    uint32_t nodes_materialized
) {
    auto *record = acquire_sched_phase_record(thread_idx);
    if (record == nullptr) return;
    fill_sched_phase_record(
        record, L2SwimlaneSchedPhaseKind::GraphPrepare, start_time, end_time, loop_iter, nodes_materialized,
        /*shared_at_start=*/nullptr, /*shared_at_end=*/nullptr
    );
    record->phase_data.graph_task.local_id = static_cast<uint32_t>(task_id);
    record->phase_data.graph_task.ring_id = static_cast<uint32_t>(task_id >> 32);
}

void l2_swimlane_aicpu_set_orch_thread_idx(int thread_idx) { s_orch_thread_idx = thread_idx; }

void l2_swimlane_aicpu_record_orch_phase(
    uint64_t start_time, uint64_t end_time, uint64_t task_id, uint32_t submit_idx
) {
    if (s_orch_thread_idx < 0 || !s_phase_initialized) return;
    // Single orch instance (dep_gen / scope_stats style): all orch records
    // funnel into pool ordinal 0, regardless of which AICPU thread the
    // orchestrator runs on. s_orch_thread_idx is the orchestrator's AICPU
    // thread index — used only to pick its own ready queue (SPSC owner); the
    // entry is tagged with pool ordinal 0 so the host files it into orch lane 0.
    auto *state = s_orch_phase_pools[0];
    if (state == nullptr) return;

    state->head.total_record_count += 1;

    auto *record = acquire_phase_slot<L2SwimlaneAicpuOrchPhaseBuffer, L2SwimlaneAicpuOrchPhaseRecord>(
        /*thread_idx=*/s_orch_thread_idx, /*pool_idx=*/0, state, &s_current_orch_phase_buffers[0],
        L2SwimlaneBufferKind::AicpuOrchPhase, "orch"
    );
    if (record == nullptr) {
        state->head.dropped_record_count += 1;
        return;
    }
    record->start_time = start_time;
    record->end_time = end_time;
    record->task_id = task_id;
    record->submit_idx = submit_idx;
}

// Final-drain flush of one phase pool's active buffer. `thread_idx` / `pool_idx`
// as in switch_phase_buffer.
static void flush_phase_pool(
    int thread_idx, uint32_t pool_idx, L2SwimlaneAicpuTaskPool *state, L2SwimlaneBufferKind kind, const char *kind_label
) {
    if (state == nullptr) return;
    rmb();
    uint64_t buf_ptr = state->head.current_buf_ptr;
    if (buf_ptr == 0) return;
    // `count` sits AFTER the records[] array in TypedBuffer, so its byte offset
    // is N * sizeof(Record) — different for sched (40B) vs orch (32B) records.
    // Read/write it through the matching buffer type; a single fixed cast reads
    // past the orch buffer, sees 0, and silently skips the orch flush.
    volatile uint32_t *count_ptr = (kind == L2SwimlaneBufferKind::AicpuOrchPhase) ?
                                       &reinterpret_cast<L2SwimlaneAicpuOrchPhaseBuffer *>(buf_ptr)->count :
                                       &reinterpret_cast<L2SwimlaneAicpuSchedPhaseBuffer *>(buf_ptr)->count;
    if (*count_ptr == 0) return;
    uint32_t seq = state->head.current_buf_seq;
    int rc = enqueue_ready_buffer(s_l2_swimlane_header, thread_idx, pool_idx, buf_ptr, seq, kind);
    if (rc == 0) {
        LOG_INFO_V0("Thread %d: flushed %s phase buffer with %u records", thread_idx, kind_label, *count_ptr);
    } else {
        LOG_ERROR(
            "Thread %d: failed to enqueue %s phase buffer (queue full), %u records lost!", thread_idx, kind_label,
            *count_ptr
        );
        state->head.dropped_record_count += *count_ptr;
        *count_ptr = 0;
    }
    state->head.current_buf_ptr = 0;
    wmb();
}

// Final-drain flush of the scheduler-phase pool owned by this scheduler thread.
void l2_swimlane_aicpu_flush_sched_phase_buffer(int thread_idx) {
    if (!s_phase_initialized || s_l2_swimlane_header == nullptr) return;
    flush_phase_pool(
        thread_idx, static_cast<uint32_t>(thread_idx), s_sched_phase_pools[thread_idx],
        L2SwimlaneBufferKind::AicpuSchedPhase, "sched"
    );
    s_current_sched_phase_buffers[thread_idx] = nullptr;
}

// Final-drain flush of the single orchestrator's orch-phase pool (ordinal 0).
// Called once by the orchestrator thread at orchestration end; see
// record_orch_phase for the pool-0 / own-ready-queue tagging.
void l2_swimlane_aicpu_flush_orch_phase_buffer(int thread_idx) {
    if (!s_phase_initialized || s_l2_swimlane_header == nullptr) return;
    flush_phase_pool(thread_idx, /*pool_idx=*/0, s_orch_phase_pools[0], L2SwimlaneBufferKind::AicpuOrchPhase, "orch");
    s_current_orch_phase_buffers[0] = nullptr;
}

void l2_swimlane_aicpu_init_core_assignments(int total_cores) {
    if (!s_phase_initialized) {
        return;
    }
    memset(s_l2_swimlane_header->core_to_thread, -1, sizeof(s_l2_swimlane_header->core_to_thread));
    s_l2_swimlane_header->num_phase_cores = static_cast<uint32_t>(total_cores);
    wmb();
    LOG_INFO_V0("Core-to-thread mapping init: %d cores", total_cores);
}

void l2_swimlane_aicpu_write_core_assignments_for_thread(int thread_idx, const int *core_ids, int core_num) {
    if (!s_phase_initialized) {
        return;
    }
    for (int i = 0; i < core_num; i++) {
        int core_id = core_ids[i];
        if (core_id >= 0 && core_id < PLATFORM_MAX_CORES) {
            s_l2_swimlane_header->core_to_thread[core_id] = static_cast<int8_t>(thread_idx);
        }
    }
    wmb();
}
