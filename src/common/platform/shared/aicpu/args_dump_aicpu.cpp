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
 * @file args_dump_aicpu.cpp
 * @brief AICPU args dump collection implementation
 *
 * - Per-thread DumpBufferState with SPSC free queues
 * - Per-thread ready queue for handing off full metadata buffers
 * - Per-thread circular arena for tensor payload data
 */

#include "aicpu/args_dump_aicpu.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "aicpu/device_run_result_base_aicpu.h"
#include "aicpu/diagnostic_run_terminal_aicpu.h"
#include "aicpu/profiler_device_engine.h"
#include "callable_limits.h"
#include "common/diagnostic_run_terminal.h"
#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "common/unified_log.h"

static_assert(CHIP_MAX_FUNC_ID <= UINT16_MAX, "function IDs must not collide with the args-dump unknown sentinel");

// Cached pointers for hot-path access (set during init)
static uint64_t g_platform_dump_base = 0;
static DumpDataHeader *s_dump_header = nullptr;
static DumpBufferState *s_dump_states[PLATFORM_MAX_AICPU_THREADS] = {};
static DumpMetaBuffer *s_current_dump_buf[PLATFORM_MAX_AICPU_THREADS] = {};

static DumpArgReservation *s_open_records[PLATFORM_MAX_AICPU_THREADS] = {};
static bool s_unsettled_record[PLATFORM_MAX_AICPU_THREADS] = {};

static bool s_logged_ready_queue_full[PLATFORM_MAX_AICPU_THREADS] = {};
static bool s_logged_no_free_meta_buffer[PLATFORM_MAX_AICPU_THREADS] = {};
static bool s_logged_dump_args_layout_mismatch = false;
static uint32_t s_records_written[PLATFORM_MAX_AICPU_THREADS] = {};
static uint32_t s_buffers_switched[PLATFORM_MAX_AICPU_THREADS] = {};
static uint32_t s_buffers_flushed[PLATFORM_MAX_AICPU_THREADS] = {};

// This run's accounting, per producer thread. Held here rather than in the
// shared DumpBufferState because the two counters that do live there —
// published_payload_count and completed_payload_count — are lifetime cursors
// that drive arena reuse and must never be reset; a per-run figure derived by
// differencing them would be a second meaning for the same field. Reset with
// the counters above at each run's device init.
static uint32_t s_records_dropped[PLATFORM_MAX_AICPU_THREADS] = {};
// The share of the above that never reached a buffer. A record refused before
// the write is one this producer took responsibility for and never wrote; one
// lost after the write is already counted among the records it wrote. `total`
// is written + refused for exactly that reason — adding every drop instead
// would count the second kind twice and make a lossy run fail conservation.
static uint32_t s_records_refused[PLATFORM_MAX_AICPU_THREADS] = {};
static uint32_t s_records_published[PLATFORM_MAX_AICPU_THREADS] = {};
static uint32_t s_buffers_published[PLATFORM_MAX_AICPU_THREADS] = {};
static uint32_t s_payloads_published[PLATFORM_MAX_AICPU_THREADS] = {};
static uint64_t s_payload_bytes_published[PLATFORM_MAX_AICPU_THREADS] = {};

// Where this run's producer accounting is published, and which bank of it is
// this run's. Zero means the host mapped none, which is the default path.
/**
 * This run's terminal bank — the address of its first producer entry — pushed
 * at kernel entry from the launching run's KernelArgs. 0 means the host
 * resolved none, which is every run with this collector's retention off, and
 * is what keeps the device off an address nobody mapped.
 */
static uint64_t s_run_terminal_bank = 0;

static inline void saturating_add_u32(uint32_t &counter, uint32_t by) {
    counter = diagnostic_terminal_add_u32(counter, by);
}

static inline void account_dropped_records(int thread_idx, DumpBufferState *state, uint32_t dropped_records) {
    if (dropped_records == 0) {
        return;
    }
    if (thread_idx >= 0 && thread_idx < PLATFORM_MAX_AICPU_THREADS) {
        saturating_add_u32(s_records_dropped[thread_idx], dropped_records);
    }
    if (state == nullptr) {
        return;
    }
    state->dropped_record_count = diagnostic_terminal_add_u32(state->dropped_record_count, dropped_records);
    state->run_dropped_record_count = diagnostic_terminal_add_u32(state->run_dropped_record_count, dropped_records);
}

// A record refused before anything was written for it: counted among this
// run's drops, and separately as one the producer never wrote.
static inline void account_refused_record(int thread_idx, DumpBufferState *state) {
    if (thread_idx >= 0 && thread_idx < PLATFORM_MAX_AICPU_THREADS) {
        saturating_add_u32(s_records_refused[thread_idx], 1);
    }
    account_dropped_records(thread_idx, state, 1);
}

static constexpr uint64_t kDumpQueueBackpressureWaitCycles = PLATFORM_DFX_BACKPRESSURE_TIMEOUT_CYCLES;

static bool g_enable_dump_args = false;
// Dump level latched from this invocation in dump_args_init(). The selective
// (PARTIAL) and metadata-first (HYBRID) modes are derived from it rather
// than tracked as separate flags — mirrors g_chip_swimlane_level.
static DumpArgsLevel g_dump_args_level = DumpArgsLevel::OFF;

extern "C" void set_platform_dump_base(uint64_t dump_data_base) { g_platform_dump_base = dump_data_base; }

void set_platform_dump_args_run_terminal(uint64_t bank) { s_run_terminal_bank = bank; }

void dump_args_close_run(int thread_idx) {
    if (thread_idx < 0 || thread_idx >= PLATFORM_MAX_AICPU_THREADS) {
        return;
    }
    if (dump_arg_has_open_record(thread_idx) || s_unsettled_record[thread_idx]) return;
    uint32_t live_at_close = 0;
    const DumpMetaBuffer *held = s_current_dump_buf[thread_idx];
    if (held != nullptr) {
        live_at_close = held->count;
    }

    DiagnosticRunTerminalCounts counts{};
    // Every record this producer took responsibility for, in the unit
    // `published_records` and `live_at_close` are in: the ones it wrote plus
    // the ones it refused before writing. A record written and then lost with
    // its buffer is already among the written ones, so adding every drop here
    // would count it twice.
    counts.total = diagnostic_terminal_add_u32(s_records_written[thread_idx], s_records_refused[thread_idx]);
    counts.dropped = s_records_dropped[thread_idx];
    counts.published_records = s_records_published[thread_idx];
    counts.published_buffers = s_buffers_published[thread_idx];
    counts.live_at_close = live_at_close;
    counts.published_payloads = s_payloads_published[thread_idx];
    counts.published_payload_bytes = s_payload_bytes_published[thread_idx];
    // No device-side fatal latch exists for this collector; a counter that
    // reached its sentinel is reported as saturation by the figures themselves.

    simpler::dfx::terminal::publish(
        s_run_terminal_bank, PLATFORM_MAX_AICPU_THREADS, thread_idx, get_platform_run_result_epoch(), counts
    );
}

extern "C" uint64_t get_platform_dump_base() { return g_platform_dump_base; }

// Both tables are open-addressed and mark a free slot with TaskId::invalid(), the
// reserved sentinel no mint produces — so probing tells an unused slot from one
// holding a real task apart by identity, with no value set aside for the purpose.
struct DumpTaskMaskEntry {
    TaskId task_id;
    ArgsDumpArgMask mask;
    ArgsDumpArgMask flags;
};
struct DumpTaskScalarDtypeEntry {
    TaskId task_id;
    uint32_t scalar_count;
    uint8_t scalar_dtypes[32];
};
static constexpr uint32_t DUMP_TASK_MASK_TABLE_CAPACITY = 32768;
static DumpTaskMaskEntry *g_dump_mask_table = nullptr;
static DumpTaskScalarDtypeEntry *g_dump_scalar_dtype_table = nullptr;
static bool ensure_dump_args_mask_table() {
    if (g_dump_mask_table != nullptr) {
        return true;
    }
    g_dump_mask_table =
        static_cast<DumpTaskMaskEntry *>(malloc(sizeof(DumpTaskMaskEntry) * DUMP_TASK_MASK_TABLE_CAPACITY));
    if (g_dump_mask_table == nullptr) {
        LOG_ERROR("Failed to allocate args dump selective mask table");
        return false;
    }
    for (uint32_t i = 0; i < DUMP_TASK_MASK_TABLE_CAPACITY; i++) {
        g_dump_mask_table[i].task_id = TaskId::invalid();
        g_dump_mask_table[i].mask = ARGS_DUMP_ARG_MASK_NONE;
        g_dump_mask_table[i].flags = ARGS_DUMP_ARG_MASK_NONE;
    }
    return true;
}

static bool ensure_dump_args_scalar_dtype_table() {
    if (g_dump_scalar_dtype_table != nullptr) {
        return true;
    }
    g_dump_scalar_dtype_table = static_cast<DumpTaskScalarDtypeEntry *>(
        malloc(sizeof(DumpTaskScalarDtypeEntry) * DUMP_TASK_MASK_TABLE_CAPACITY)
    );
    if (g_dump_scalar_dtype_table == nullptr) {
        LOG_ERROR("Failed to allocate args dump scalar dtype table");
        return false;
    }
    for (uint32_t i = 0; i < DUMP_TASK_MASK_TABLE_CAPACITY; i++) {
        g_dump_scalar_dtype_table[i].task_id = TaskId::invalid();
        g_dump_scalar_dtype_table[i].scalar_count = 0;
        memset(g_dump_scalar_dtype_table[i].scalar_dtypes, 0, sizeof(g_dump_scalar_dtype_table[i].scalar_dtypes));
    }
    return true;
}

static void clear_dump_args_tables() {
    if (g_dump_mask_table != nullptr) {
        for (uint32_t i = 0; i < DUMP_TASK_MASK_TABLE_CAPACITY; i++) {
            g_dump_mask_table[i].task_id = TaskId::invalid();
            g_dump_mask_table[i].mask = ARGS_DUMP_ARG_MASK_NONE;
            g_dump_mask_table[i].flags = ARGS_DUMP_ARG_MASK_NONE;
        }
    }
    if (g_dump_scalar_dtype_table != nullptr) {
        for (uint32_t i = 0; i < DUMP_TASK_MASK_TABLE_CAPACITY; i++) {
            g_dump_scalar_dtype_table[i].task_id = TaskId::invalid();
            g_dump_scalar_dtype_table[i].scalar_count = 0;
            memset(g_dump_scalar_dtype_table[i].scalar_dtypes, 0, sizeof(g_dump_scalar_dtype_table[i].scalar_dtypes));
        }
    }
}

// The key is opaque here: this module makes no claim about what any of its bits
// mean, only that the whole of it identifies a task. Folding the two halves is what
// gets both of them into the index, so a layout that varies its low bits and one
// that varies its high bits both spread. Any handle maps to a slot — the pool's
// size is independent of anything the runtime sizes.
static uint32_t resolve_dump_args_task_slot(TaskId task_id) {
    uint64_t h = TaskId::to_uint64(task_id);
    h ^= h >> 32;
    return static_cast<uint32_t>(h) & (DUMP_TASK_MASK_TABLE_CAPACITY - 1);
}

extern "C" void set_dump_args_task_scalar_dtypes(TaskId task_id, uint32_t scalar_count, const uint8_t *scalar_dtypes) {
    if (scalar_count == 0 || scalar_dtypes == nullptr) {
        return;
    }
    if (scalar_count > 32) {
        LOG_ERROR("args dump scalar dtype count exceeds max (%u > 32)", scalar_count);
        return;
    }
    if (!ensure_dump_args_scalar_dtype_table()) {
        return;
    }
    uint32_t idx = resolve_dump_args_task_slot(task_id);
    for (uint32_t probe = 0; probe < DUMP_TASK_MASK_TABLE_CAPACITY; probe++) {
        DumpTaskScalarDtypeEntry &entry =
            g_dump_scalar_dtype_table[(idx + probe) & (DUMP_TASK_MASK_TABLE_CAPACITY - 1)];
        if (entry.task_id == TaskId::invalid() || entry.task_id == task_id) {
            entry.task_id = task_id;
            entry.scalar_count = scalar_count;
            memcpy(entry.scalar_dtypes, scalar_dtypes, scalar_count * sizeof(uint8_t));
            return;
        }
    }
    LOG_ERROR("args dump scalar dtype table is full");
}

extern "C" bool get_dump_args_task_scalar_dtypes(TaskId task_id, uint32_t *scalar_count, uint8_t *scalar_dtypes) {
    if (g_dump_scalar_dtype_table == nullptr || scalar_count == nullptr || scalar_dtypes == nullptr) {
        return false;
    }
    uint32_t idx = resolve_dump_args_task_slot(task_id);
    for (uint32_t probe = 0; probe < DUMP_TASK_MASK_TABLE_CAPACITY; probe++) {
        const DumpTaskScalarDtypeEntry &entry =
            g_dump_scalar_dtype_table[(idx + probe) & (DUMP_TASK_MASK_TABLE_CAPACITY - 1)];
        if (entry.task_id == task_id) {
            *scalar_count = entry.scalar_count;
            memcpy(scalar_dtypes, entry.scalar_dtypes, entry.scalar_count * sizeof(uint8_t));
            return true;
        }
        if (entry.task_id == TaskId::invalid()) {
            return false;
        }
    }
    return false;
}

extern "C" void set_dump_args_enabled(bool enable) {
    g_enable_dump_args = enable;
    g_dump_args_level = DumpArgsLevel::OFF;
    if (enable) {
        if (!ensure_dump_args_mask_table() || !ensure_dump_args_scalar_dtype_table()) {
            g_enable_dump_args = false;
            free(g_dump_mask_table);
            g_dump_mask_table = nullptr;
            free(g_dump_scalar_dtype_table);
            g_dump_scalar_dtype_table = nullptr;
            return;
        }
    } else {
        free(g_dump_mask_table);
        g_dump_mask_table = nullptr;
        free(g_dump_scalar_dtype_table);
        g_dump_scalar_dtype_table = nullptr;
    }
    clear_dump_args_tables();
}

extern "C" bool is_dump_args_enabled() { return g_enable_dump_args; }

extern "C" bool is_dump_args_selective_mode() { return g_dump_args_level == DumpArgsLevel::PARTIAL; }

extern "C" bool should_load_dump_args_task_masks() {
    return g_dump_args_level == DumpArgsLevel::PARTIAL || g_dump_args_level == DumpArgsLevel::HYBRID;
}

extern "C" void set_dump_args_task_mask(TaskId task_id, ArgsDumpArgMask mask, ArgsDumpArgMask flags) {
    if (mask == ARGS_DUMP_ARG_MASK_NONE) {
        return;
    }
    if (!ensure_dump_args_mask_table()) {
        return;
    }
    uint32_t idx = resolve_dump_args_task_slot(task_id);
    for (uint32_t probe = 0; probe < DUMP_TASK_MASK_TABLE_CAPACITY; probe++) {
        DumpTaskMaskEntry &entry = g_dump_mask_table[(idx + probe) & (DUMP_TASK_MASK_TABLE_CAPACITY - 1)];
        if (entry.task_id == TaskId::invalid() || entry.task_id == task_id) {
            entry.task_id = task_id;
            entry.mask = mask;
            entry.flags = flags;
            return;
        }
    }
    LOG_ERROR("args dump selective mask table is full");
}

extern "C" void get_dump_args_task_masks(TaskId task_id, ArgsDumpArgMask *mask, ArgsDumpArgMask *flags) {
    if (mask != nullptr) {
        *mask = ARGS_DUMP_ARG_MASK_NONE;
    }
    if (flags != nullptr) {
        *flags = ARGS_DUMP_ARG_MASK_NONE;
    }
    if (g_dump_mask_table == nullptr) {
        return;
    }
    uint32_t idx = resolve_dump_args_task_slot(task_id);
    for (uint32_t probe = 0; probe < DUMP_TASK_MASK_TABLE_CAPACITY; probe++) {
        const DumpTaskMaskEntry &entry = g_dump_mask_table[(idx + probe) & (DUMP_TASK_MASK_TABLE_CAPACITY - 1)];
        if (entry.task_id == task_id) {
            if (mask != nullptr) {
                *mask = entry.mask;
            }
            if (flags != nullptr) {
                *flags = entry.flags;
            }
            return;
        }
        if (entry.task_id == TaskId::invalid()) {
            return;
        }
    }
}

bool get_dump_arg_role_from_direction(ArgDirection dir, ArgsDumpRole *role) {
    switch (dir) {
    case ArgDirection::IN:
        *role = ArgsDumpRole::INPUT;
        return true;
    case ArgDirection::OUT:
        *role = ArgsDumpRole::OUTPUT;
        return true;
    case ArgDirection::INOUT:
        *role = ArgsDumpRole::INOUT;
        return true;
    case ArgDirection::SCALAR:
        return false;
    }
    return false;
}

int32_t count_callable_tensor_args(const CoreCallable &callable) {
    int32_t tensor_count = 0;
    for (int32_t i = 0; i < callable.sig_count(); i++) {
        if (callable.sig(i) != ArgDirection::SCALAR) {
            tensor_count++;
        }
    }
    return tensor_count;
}

bool should_dump_arg_at_stage(ArgsDumpRole role, ArgsDumpStage stage) {
    switch (role) {
    case ArgsDumpRole::INPUT:
        return stage == ArgsDumpStage::BEFORE_DISPATCH;
    case ArgsDumpRole::OUTPUT:
        return stage == ArgsDumpStage::AFTER_COMPLETION;
    case ArgsDumpRole::INOUT:
        return true;
    }
    return false;
}

bool should_dump_task(ArgsDumpArgMask arg_mask) { return dump_args_task_selected(g_dump_args_level, arg_mask); }

bool should_dump_arg(ArgsDumpArgMask arg_mask, int32_t arg_index) {
    if (!is_dump_args_selective_mode()) {
        return true;
    }
    if (arg_index < 0 || arg_index >= static_cast<int32_t>(ARGS_DUMP_ARG_MASK_BITS)) return false;
    return (arg_mask & (ArgsDumpArgMask{1} << arg_index)) != 0;
}

bool has_dump_arg_flag(ArgsDumpArgMask arg_mask, int32_t arg_index) {
    if (arg_index < 0 || arg_index >= static_cast<int32_t>(ARGS_DUMP_ARG_MASK_BITS)) return false;
    return (arg_mask & (ArgsDumpArgMask{1} << arg_index)) != 0;
}

bool try_log_dump_args_layout_mismatch() {
    if (s_logged_dump_args_layout_mismatch) {
        return false;
    }
    s_logged_dump_args_layout_mismatch = true;
    return true;
}

/**
 * Enqueue a full dump metadata buffer to the thread's ready queue.
 */
struct DumpDeviceModule {
    struct Context {
        DumpDataHeader *header;
        int thread_idx;
    };

    using DataHeader = DumpDataHeader;
    using State = DumpBufferState;
    using FreeQueue = DumpFreeQueue;
    using Buffer = DumpMetaBuffer;

    static constexpr uint32_t kReadyQueueSize = PLATFORM_DUMP_READYQUEUE_SIZE;
    static constexpr uint32_t kSlotCount = PLATFORM_DUMP_SLOT_COUNT;
    static constexpr uint64_t kBackpressureWaitCycles = kDumpQueueBackpressureWaitCycles;

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
        DumpBufferState *state = s_dump_states[ctx.thread_idx];
        auto *buffer = reinterpret_cast<DumpMetaBuffer *>(buffer_ptr);
        uint64_t payload_count = 0;
        uint64_t payload_bytes = 0;
        uint32_t record_count = buffer->count;
        if (record_count > PLATFORM_DUMP_RECORDS_PER_BUFFER) {
            record_count = PLATFORM_DUMP_RECORDS_PER_BUFFER;
        }
        for (uint32_t i = 0; i < record_count; i++) {
            const ArgsDumpRecord &record = buffer->records[i];
            if (record.kind == static_cast<uint8_t>(ArgsDumpKind::TENSOR) && record.payload_size > 0) {
                payload_count++;
                payload_bytes += record.payload_size;
            }
        }
        state->published_payload_count += payload_count;
        // This run's share of the same commit. The cursor above is a lifetime
        // one the arena's reuse depends on; these are the figures that balance
        // against the host's receipts for this identity alone.
        saturating_add_u32(s_records_published[ctx.thread_idx], record_count);
        saturating_add_u32(s_buffers_published[ctx.thread_idx], 1);
        saturating_add_u32(s_payloads_published[ctx.thread_idx], static_cast<uint32_t>(payload_count));
        s_payload_bytes_published[ctx.thread_idx] =
            diagnostic_terminal_add_u64(s_payload_bytes_published[ctx.thread_idx], payload_bytes);
        wmb();
        ctx.header->queues[ctx.thread_idx][tail].thread_index = static_cast<uint32_t>(ctx.thread_idx);
        ctx.header->queues[ctx.thread_idx][tail].buffer_ptr = buffer_ptr;
        ctx.header->queues[ctx.thread_idx][tail].buffer_seq = buffer_seq;
    }

    static void account_dropped(Context ctx, State *state, uint32_t count) {
        account_dropped_records(ctx.thread_idx, state, count);
    }
    static void on_pop_success(Context ctx, State *state, Buffer *buffer) {
        buffer->run_epoch = get_platform_run_result_epoch();
        buffer->local_seq = state->current_buf_seq;
        s_current_dump_buf[ctx.thread_idx] = buffer;
    }
    static void on_current_cleared(Context ctx, State *) { s_current_dump_buf[ctx.thread_idx] = nullptr; }
    static void on_null_free_slot(Context, State *) {}

    static void on_enqueue_failed(Context ctx, State *, Buffer *) {
        if (!s_logged_ready_queue_full[ctx.thread_idx]) {
            s_logged_ready_queue_full[ctx.thread_idx] = true;
            LOG_WARN(
                "Args dump ready queue full on thread %d after bounded wait, "
                "dropping current metadata buffer. Increase PLATFORM_DUMP_READYQUEUE_SIZE.",
                ctx.thread_idx
            );
        }
    }

    static void on_no_replacement(Context ctx, State *) {
        if (!s_logged_no_free_meta_buffer[ctx.thread_idx]) {
            s_logged_no_free_meta_buffer[ctx.thread_idx] = true;
            LOG_WARN(
                "Args dump published a full metadata buffer on thread %d but no replacement was available; "
                "records will drop until recovery. Increase PLATFORM_DUMP_BUFFERS_PER_THREAD.",
                ctx.thread_idx
            );
        }
    }

    static void on_switch_complete(Context ctx, State *, Buffer *) { s_buffers_switched[ctx.thread_idx]++; }
};

using DumpEngine = profiling_device::DeviceProfilerEngine<DumpDeviceModule>;

static DumpDeviceModule::Context dump_context(int thread_idx) {
    return DumpDeviceModule::Context{s_dump_header, thread_idx};
}

static int enqueue_dump_ready_buffer(int thread_idx, uint64_t buffer_ptr, uint32_t buffer_seq) {
    return DumpEngine::enqueue_ready(dump_context(thread_idx), buffer_ptr, buffer_seq);
}

static DumpMetaBuffer *try_pop_dump_meta_buffer(int thread_idx, DumpBufferState *state, uint32_t next_seq) {
    if (thread_idx < 0 || thread_idx >= PLATFORM_MAX_AICPU_THREADS || state == nullptr) {
        return nullptr;
    }
    return DumpEngine::pop_free(dump_context(thread_idx), state, next_seq);
}

/**
 * Switch metadata buffer: enqueue the full buffer first, then pop a new one.
 * If no replacement is available, later records drop until host replenishes
 * free_queue.
 */
static int switch_dump_meta_buffer(int thread_idx) {
    if (thread_idx < 0 || thread_idx >= PLATFORM_MAX_AICPU_THREADS) {
        return -1;
    }
    DumpBufferState *state = s_dump_states[thread_idx];
    DumpMetaBuffer *cur = s_current_dump_buf[thread_idx];
    if (state == nullptr || cur == nullptr) {
        return -1;
    }
    DumpEngine::switch_buffer(dump_context(thread_idx), state);
    return 0;
}

static bool ensure_dump_arena_capacity(int thread_idx, DumpBufferState *state, uint64_t copy_bytes) {
    if (copy_bytes == 0) {
        return true;
    }
    if (state == nullptr || copy_bytes > state->arena_size) {
        return false;
    }

    const uint64_t physical_offset = state->arena_write_offset % state->arena_size;
    const bool arena_full = state->arena_write_offset != 0 && physical_offset == 0;
    const bool crosses_arena_end = copy_bytes > state->arena_size - physical_offset;
    if (!arena_full && !crosses_arena_end) {
        return true;
    }

    DumpMetaBuffer *buf = s_current_dump_buf[thread_idx];
    if (buf != nullptr && buf->count > 0) {
        uint32_t seq = state->current_buf_seq;
        if (switch_dump_meta_buffer(thread_idx) != 0 || state->current_buf_seq == seq) {
            return false;
        }
    }

    // Per-lane arena barrier. `target_payload_count` is snapshotted after the
    // publish above, so it names exactly the payloads this thread has handed to
    // the host; the arena bytes about to be overwritten are safe once the host
    // has written all of them to args.bin and acked with an equal
    // `completed_payload_count`. Both counters are per-thread and the arena is
    // this thread's alone, so no peer lane is involved. The budget is the
    // host-crash backstop only.
    const uint64_t target_payload_count = state->published_payload_count;
    const uint64_t wait_start = get_sys_cnt_aicpu();
    while (state->completed_payload_count < target_payload_count) {
        if (get_sys_cnt_aicpu() - wait_start >= kDumpQueueBackpressureWaitCycles) {
            return false;
        }
        SPIN_WAIT_HINT();
    }
    rmb();
    if (physical_offset != 0) {
        state->arena_write_offset += state->arena_size - physical_offset;
    }
    return true;
}

struct CircularArenaWriter {
    char *arena;
    uint64_t arena_size;
    uint64_t base_offset;
    uint64_t bytes_written;

    void write(const void *src, uint64_t size) {
        if (size == 0) {
            return;
        }
        uint64_t pos = (base_offset + bytes_written) % arena_size;
        if (pos + size <= arena_size) {
            memcpy(arena + pos, src, size);
        } else {
            uint64_t first = arena_size - pos;
            memcpy(arena + pos, src, first);
            memcpy(arena, reinterpret_cast<const char *>(src) + first, size - first);
        }
        bytes_written += size;
    }
};

// PyTorch-style contiguous: strides[d] == prod(shapes[d+1..]) for all d.
static inline bool dump_arg_is_contiguous(const ArgsDumpInfo &info) {
    if (info.ndims == 0) return true;
    uint64_t expected = 1;
    for (int32_t d = static_cast<int32_t>(info.ndims) - 1; d >= 0 && d < PLATFORM_DUMP_MAX_DIMS; --d) {
        if (info.strides[d] != expected) return false;
        expected *= info.shapes[d];
    }
    return true;
}

void dump_args_init(int num_dump_threads, DumpArgsLevel level) {
    for (int t = 0; t < PLATFORM_MAX_AICPU_THREADS; ++t) {
        if (dump_arg_has_open_record(t)) return;
    }
    void *dump_base = reinterpret_cast<void *>(get_platform_dump_base());
    if (dump_base == nullptr) {
        LOG_ERROR("platform dump base is NULL, cannot initialize args dump");
        return;
    }
    if (num_dump_threads <= 0 || num_dump_threads > PLATFORM_MAX_AICPU_THREADS) {
        LOG_ERROR("invalid args dump thread count %d (expected 1..%d)", num_dump_threads, PLATFORM_MAX_AICPU_THREADS);
        return;
    }

    s_dump_header = get_dump_header(dump_base);

    // The descriptor selection is stable for this invocation before dispatch.
    // PARTIAL → selective (only Arg::dump()-marked args); FULL → every task;
    // HYBRID → every task's metadata; payload is copied only for
    // arguments selected through the same per-task mask as PARTIAL.
    g_dump_args_level = level;

    LOG_INFO("Initializing args dump for %d threads", num_dump_threads);

    // Pop initial metadata buffer from free_queue for each thread
    for (int t = 0; t < num_dump_threads; t++) {
        s_unsettled_record[t] = false;
        DumpBufferState *state = get_dump_buffer_state(dump_base, t);
        s_dump_states[t] = state;
        // Transport cursors span invocations; ordinary host credits start at
        // this boundary, while retained collectors acknowledge lifetime totals.
        state->run_published_payload_base = state->published_payload_count;
        state->run_dropped_record_count = 0;
        wmb();

        // Keep the buffer this thread already holds, or pop the first one.
        //
        // A buffer is only released by a successful enqueue, so a pointer still
        // set here means the previous run could not hand that buffer over — it
        // had nothing to publish, or the ready queue was full. Popping a
        // replacement would strand it: nothing returns it to the free queue,
        // because AICPU is the queue's consumer and never its producer. Reusing
        // it in place is the return, and re-stamping is what makes that safe —
        // the buffer still carries the previous run's identity, and its count
        // must start this run at zero.
        rmb();
        uint64_t retained = state->current_buf_ptr;
        uint32_t head = state->free_queue.head;
        uint32_t tail = state->free_queue.tail;
        if (retained != 0) {
            DumpMetaBuffer *buf = reinterpret_cast<DumpMetaBuffer *>(retained);
            buf->count = 0;
            buf->run_epoch = get_platform_run_result_epoch();
            buf->local_seq = 0;
            wmb();
            state->current_buf_seq = 0;
            s_current_dump_buf[t] = buf;
            LOG_DEBUG("Thread %d: reusing retained dump buffer (addr=0x%lx)", t, retained);
        } else if (head != tail) {
            // The engine's pop stamps identity through on_pop_success.
            (void)try_pop_dump_meta_buffer(t, state, 0);
            uint64_t buf_ptr = state->current_buf_ptr;
            LOG_DEBUG("Thread %d: popped initial dump buffer (addr=0x%lx)", t, buf_ptr);
        } else {
            LOG_ERROR("Thread %d: dump free_queue is empty during init!", t);
            s_current_dump_buf[t] = nullptr;
            state->current_buf_ptr = 0;
        }
        wmb();
        state->run_epoch = get_platform_run_result_epoch();
        wmb();
    }

    memset(s_logged_ready_queue_full, 0, sizeof(s_logged_ready_queue_full));
    memset(s_logged_no_free_meta_buffer, 0, sizeof(s_logged_no_free_meta_buffer));
    memset(s_records_written, 0, sizeof(s_records_written));
    memset(s_buffers_switched, 0, sizeof(s_buffers_switched));
    memset(s_buffers_flushed, 0, sizeof(s_buffers_flushed));
    memset(s_records_dropped, 0, sizeof(s_records_dropped));
    memset(s_records_refused, 0, sizeof(s_records_refused));
    memset(s_records_published, 0, sizeof(s_records_published));
    memset(s_buffers_published, 0, sizeof(s_buffers_published));
    memset(s_payloads_published, 0, sizeof(s_payloads_published));
    memset(s_payload_bytes_published, 0, sizeof(s_payload_bytes_published));
    for (int t = 0; t < num_dump_threads; t++) {
        // Exactly the lanes this run brought up, because exactly those close.
        // A claim on a lane the run never starts is never settled, and a
        // participant that never settles is the run reporting its own
        // accounting unknown — so claiming the whole dimensioned array would
        // make every run with fewer lanes than the platform maximum unknown.
        simpler::dfx::terminal::begin_run(
            s_run_terminal_bank, PLATFORM_MAX_AICPU_THREADS, t, get_platform_run_result_epoch()
        );
    }
}

bool dump_arg_has_open_record(int thread_idx) {
    return thread_idx >= 0 && thread_idx < PLATFORM_MAX_AICPU_THREADS && s_open_records[thread_idx] != nullptr;
}

static bool owns_reservation(const DumpArgReservation *reservation) {
    return reservation != nullptr && reservation->open() && reservation->lane < PLATFORM_MAX_AICPU_THREADS &&
           s_open_records[reservation->lane] == reservation;
}

DumpArgFlushResult dump_args_try_flush(int thread_idx) {
    if (s_dump_header == nullptr || thread_idx < 0 || thread_idx >= PLATFORM_MAX_AICPU_THREADS ||
        s_dump_states[thread_idx] == nullptr)
        return DumpArgFlushResult::Invalid;
    if (dump_arg_has_open_record(thread_idx)) return DumpArgFlushResult::WouldBlock;
    auto *state = s_dump_states[thread_idx];
    auto *buffer = s_current_dump_buf[thread_idx];
    if (buffer == nullptr || buffer->count == 0) return DumpArgFlushResult::Complete;
    const uint32_t tail = s_dump_header->queue_tails[thread_idx];
    const uint32_t next = (tail + 1) % PLATFORM_DUMP_READYQUEUE_SIZE;
    if (next == s_dump_header->queue_heads[thread_idx]) return DumpArgFlushResult::WouldBlock;
    rmb();
    DumpDeviceModule::write_ready_entry(
        dump_context(thread_idx), tail, reinterpret_cast<uint64_t>(buffer), state->current_buf_seq
    );
    wmb();
    s_dump_header->queue_tails[thread_idx] = next;
    s_current_dump_buf[thread_idx] = nullptr;
    state->current_buf_ptr = 0;
    ++state->current_buf_seq;
    ++s_buffers_switched[thread_idx];
    wmb();
    return DumpArgFlushResult::Complete;
}

static bool try_arena_capacity(int lane, DumpBufferState *state, uint64_t bytes, uint64_t *offset) {
    *offset = state->arena_write_offset;
    if (bytes == 0) return true;
    const uint64_t physical = state->arena_write_offset % state->arena_size;
    if (bytes <= state->arena_size - physical && (physical != 0 || state->arena_write_offset == 0)) return true;
    if (dump_args_try_flush(lane) != DumpArgFlushResult::Complete) return false;
    if (state->completed_payload_count < state->published_payload_count) return false;
    rmb();
    if (physical != 0) {
        const uint64_t padding = state->arena_size - physical;
        if (state->arena_write_offset > UINT64_MAX - padding) return false;
        *offset += padding;
    }
    return true;
}

static DumpArgBeginResult
begin_dump_record(int thread_idx, const ArgsDumpInfo &info, DumpArgReservation *reservation, bool blocking) {
    if (s_dump_header == nullptr || thread_idx < 0 || thread_idx >= PLATFORM_MAX_AICPU_THREADS ||
        reservation == nullptr)
        return DumpArgBeginResult::Refused;
    if (dump_arg_has_open_record(thread_idx) || reservation->open()) return DumpArgBeginResult::WouldBlock;
    auto *state = s_dump_states[thread_idx];
    if (state == nullptr || g_dump_args_level == DumpArgsLevel::OFF) return DumpArgBeginResult::Refused;
    auto refuse = [&] {
        account_refused_record(thread_idx, state);
        return DumpArgBeginResult::Refused;
    };
    if (info.ndims > PLATFORM_DUMP_MAX_DIMS || info.func_count < 0 || info.func_count > ARGS_DUMP_MAX_FUNC_IDS)
        return refuse();
    const bool scalar = info.kind == static_cast<uint8_t>(ArgsDumpKind::SCALAR);
    const uint64_t element_bytes = get_element_size(static_cast<DataType>(info.dtype));
    uint64_t elements = 1;
    for (uint32_t dimension = 0; dimension < info.ndims; ++dimension) {
        if (info.shapes[dimension] != 0 && elements > UINT64_MAX / info.shapes[dimension]) return refuse();
        elements *= info.shapes[dimension];
    }
    if (element_bytes == 0 || elements > UINT64_MAX / element_bytes) return refuse();
    uint64_t bytes = elements * element_bytes;
    bool truncated = false;
    if (scalar || (g_dump_args_level == DumpArgsLevel::HYBRID && info.capture_payload == 0)) bytes = 0;
    else if (bytes > state->arena_size) {
        bytes = state->arena_size / 2;
        truncated = true;
    }
    if (bytes != 0 && (info.buffer_addr == 0 || state->arena_base == 0 || state->arena_size == 0)) return refuse();
    if (bytes != 0) {
        uint64_t last_element = info.start_offset;
        for (uint32_t dimension = 0; dimension < info.ndims; ++dimension) {
            const uint64_t extent = static_cast<uint64_t>(info.shapes[dimension] - 1) * info.strides[dimension];
            if (last_element > UINT64_MAX - extent) return refuse();
            last_element += extent;
        }
        if (last_element > (UINT64_MAX - element_bytes) / element_bytes ||
            info.buffer_addr > UINT64_MAX - (last_element + 1) * element_bytes)
            return refuse();
    }
    const uint64_t rollback = state->arena_write_offset;
    uint64_t offset = rollback;
    auto *buffer = s_current_dump_buf[thread_idx];
    if (blocking) {
        if (buffer == nullptr) buffer = try_pop_dump_meta_buffer(thread_idx, state, state->current_buf_seq);
        if (buffer == nullptr) return refuse();
        if (buffer->count >= PLATFORM_DUMP_RECORDS_PER_BUFFER) {
            if (switch_dump_meta_buffer(thread_idx) != 0) return refuse();
        }
        if (!ensure_dump_arena_capacity(thread_idx, state, bytes)) return refuse();
        offset = state->arena_write_offset;
    } else {
        if (buffer != nullptr && buffer->count >= PLATFORM_DUMP_RECORDS_PER_BUFFER &&
            dump_args_try_flush(thread_idx) != DumpArgFlushResult::Complete)
            return DumpArgBeginResult::WouldBlock;
        if (!try_arena_capacity(thread_idx, state, bytes, &offset)) return DumpArgBeginResult::WouldBlock;
    }
    buffer = s_current_dump_buf[thread_idx];
    if (buffer == nullptr) {
        buffer = blocking ? try_pop_dump_meta_buffer(thread_idx, state, state->current_buf_seq) :
                            DumpEngine::try_pop_free(dump_context(thread_idx), state, state->current_buf_seq);
    }
    if (buffer == nullptr) return blocking ? refuse() : DumpArgBeginResult::WouldBlock;
    if (offset > UINT64_MAX - bytes) return refuse();
    *reservation = {};
    reservation->info = info;
    reservation->lane = thread_idx;
    reservation->rollback_offset = rollback;
    reservation->arena_offset = offset;
    reservation->copy_bytes = bytes;
    reservation->element_bytes = element_bytes;
    reservation->contiguous = dump_arg_is_contiguous(info);
    reservation->truncated = truncated;
    state->arena_write_offset = offset + bytes;
    s_open_records[thread_idx] = reservation;
    return DumpArgBeginResult::Ready;
}

DumpArgBeginResult dump_arg_try_begin(int thread_idx, const ArgsDumpInfo &info, DumpArgReservation *reservation) {
    return begin_dump_record(thread_idx, info, reservation, false);
}

DumpArgCopyResult dump_arg_copy_chunk(DumpArgReservation *reservation, uint64_t max_bytes) {
    if (!owns_reservation(reservation) || max_bytes == 0) return DumpArgCopyResult::Invalid;
    auto &record = *reservation;
    auto *state = s_dump_states[record.lane];
    uint64_t remaining = std::min(max_bytes, record.copy_bytes - record.copied);
    CircularArenaWriter writer{
        reinterpret_cast<char *>(state->arena_base), state->arena_size, record.arena_offset, record.copied
    };
    while (remaining != 0) {
        uint64_t element = record.copied / record.element_bytes;
        const uint64_t byte_in_element = record.copied % record.element_bytes;
        uint64_t source_element = record.info.start_offset;
        uint64_t count = remaining;
        if (record.contiguous) source_element += element;
        else {
            for (int32_t dimension = record.info.ndims - 1; dimension >= 0; --dimension) {
                const uint64_t coordinate = element % record.info.shapes[dimension];
                element /= record.info.shapes[dimension];
                source_element += coordinate * record.info.strides[dimension];
            }
            count = std::min(remaining, record.element_bytes - byte_in_element);
        }
        const auto *source = reinterpret_cast<const unsigned char *>(record.info.buffer_addr) +
                             source_element * record.element_bytes + byte_in_element;
        cache_invalidate_range(source, count);
        rmb();
        writer.write(source, count);
        record.copied += count;
        remaining -= count;
    }
    return record.copied == record.copy_bytes ? DumpArgCopyResult::Complete : DumpArgCopyResult::InProgress;
}

bool dump_arg_commit(DumpArgReservation *reservation) {
    if (!owns_reservation(reservation) || reservation->copied != reservation->copy_bytes) return false;
    const auto &info = reservation->info;
    const int lane = reservation->lane;
    auto *buffer = s_current_dump_buf[lane];
    if (buffer == nullptr || buffer->count >= PLATFORM_DUMP_RECORDS_PER_BUFFER) return false;
    auto *record = &buffer->records[buffer->count];
    record->task_id = info.task_id;
    record->arg_index = info.arg_index;
    record->is_contiguous = reservation->contiguous ? 1 : 0;
    record->role = static_cast<uint8_t>(info.role);
    record->stage = static_cast<uint8_t>(info.stage);
    record->ndims = info.ndims;
    record->dtype = info.dtype;
    record->truncated = reservation->truncated ? 1 : 0;
    record->payload_offset = reservation->arena_offset;
    record->payload_size = reservation->copy_bytes;
    record->scalar_value = info.kind == static_cast<uint8_t>(ArgsDumpKind::SCALAR) ? info.scalar_value : 0;
    record->kind = info.kind;
    record->flags = info.flags;
    record->func_count = static_cast<uint8_t>(info.func_count);
    for (int32_t i = 0; i < info.func_count; ++i)
        record->func_ids[i] = static_cast<uint16_t>(info.func_ids[i]);
    record->start_offset = info.start_offset;
    for (uint32_t i = 0; i < info.ndims; ++i) {
        record->shapes[i] = info.shapes[i];
        record->strides[i] = info.strides[i];
    }
    wmb();
    ++buffer->count;
    wmb();
    saturating_add_u32(s_records_written[lane], 1);
    s_open_records[lane] = nullptr;
    reservation->lane = -1;
    return true;
}

bool dump_arg_abandon(DumpArgReservation *reservation) {
    if (!owns_reservation(reservation)) return false;
    const int lane = reservation->lane;
    auto *state = s_dump_states[lane];
    state->arena_write_offset = reservation->rollback_offset;
    account_refused_record(lane, state);
    s_unsettled_record[lane] = true;
    s_open_records[lane] = nullptr;
    reservation->lane = -1;
    return true;
}

int dump_arg_record(int thread_idx, const ArgsDumpInfo &info) {
    DumpArgReservation reservation;
    if (begin_dump_record(thread_idx, info, &reservation, true) != DumpArgBeginResult::Ready) return -1;
    if (dump_arg_copy_chunk(&reservation, UINT64_MAX) == DumpArgCopyResult::Complete && dump_arg_commit(&reservation))
        return 0;
    (void)dump_arg_abandon(&reservation);
    return -1;
}

void dump_args_flush(int thread_idx) {
    if (dump_arg_has_open_record(thread_idx)) return;
    if (s_dump_header == nullptr) {
        return;
    }
    if (thread_idx < 0 || thread_idx >= PLATFORM_MAX_AICPU_THREADS) {
        return;
    }

    DumpMetaBuffer *buf = s_current_dump_buf[thread_idx];
    DumpBufferState *state = s_dump_states[thread_idx];
    if (buf != nullptr && buf->count > 0 && state != nullptr) {
        uint64_t buf_addr = reinterpret_cast<uint64_t>(buf);
        uint32_t seq = state->current_buf_seq;
        if (enqueue_dump_ready_buffer(thread_idx, buf_addr, seq) == 0) {
            s_current_dump_buf[thread_idx] = nullptr;
            state->current_buf_ptr = 0;
            wmb();
        } else {
            // Bounded to one per thread per run (unlike the hot-path
            // dump_arg_record site), so no spam guard is needed here.
            LOG_ERROR(
                "Thread %d: failed to flush args-dump buffer (ready_queue full), %u records lost!", thread_idx,
                buf->count
            );
            // ready_queue full at end-of-run: account the loss, but keep the
            // buffer. Clearing the pointer would strand it — the host never saw
            // it, so nothing returns it to the free queue. Retaining it lets the
            // next run's init reuse it in place, which is the only return
            // available to AICPU.
            account_dropped_records(thread_idx, state, buf->count);
            buf->count = 0;
            wmb();
        }
    }

    s_buffers_flushed[thread_idx]++;
    uint32_t dropped = s_dump_states[thread_idx] ? s_dump_states[thread_idx]->run_dropped_record_count : 0;
    LOG_INFO(
        "Thread %d: dump_args_flush (records=%u, buf_switches=%u, flushes=%u, dropped=%u)", thread_idx,
        s_records_written[thread_idx], s_buffers_switched[thread_idx], s_buffers_flushed[thread_idx], dropped
    );
}
