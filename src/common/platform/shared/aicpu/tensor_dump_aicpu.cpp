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
 * @file tensor_dump_aicpu.cpp
 * @brief AICPU args dump collection implementation
 *
 * - Per-thread DumpBufferState with SPSC free queues
 * - Per-thread ready queue for handing off full metadata buffers
 * - Per-thread circular arena for tensor payload data
 */

#include "aicpu/tensor_dump_aicpu.h"

#include <cstdlib>
#include <cstring>

#include "aicpu/device_time.h"
#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "common/unified_log.h"

// Cached pointers for hot-path access (set during init)
static uint64_t g_platform_dump_base = 0;
static DumpDataHeader *s_dump_header = nullptr;
static DumpBufferState *s_dump_states[PLATFORM_MAX_AICPU_THREADS] = {};
static DumpMetaBuffer *s_current_dump_buf[PLATFORM_MAX_AICPU_THREADS] = {};

static bool s_logged_ready_queue_full[PLATFORM_MAX_AICPU_THREADS] = {};
static bool s_logged_no_free_meta_buffer[PLATFORM_MAX_AICPU_THREADS] = {};
static bool s_logged_dump_args_layout_mismatch = false;
static uint32_t s_records_written[PLATFORM_MAX_AICPU_THREADS] = {};
static uint32_t s_buffers_switched[PLATFORM_MAX_AICPU_THREADS] = {};
static uint32_t s_buffers_flushed[PLATFORM_MAX_AICPU_THREADS] = {};

static inline void account_dropped_records(DumpBufferState *state, uint32_t dropped_records) {
    if (state == nullptr || dropped_records == 0) {
        return;
    }
    uint32_t prev = state->dropped_record_count;
    uint32_t next = prev + dropped_records;
    state->dropped_record_count = (next < prev) ? UINT32_MAX : next;
}

static constexpr uint64_t kDumpQueueBackpressureWaitCycles = PLATFORM_PROF_SYS_CNT_FREQ / 50000;  // 20 us

static bool g_enable_dump_args = false;
// Dump level latched from the header in dump_args_init(). The selective
// (PARTIAL) and json-only (FULL_JSON_ONLY) modes are derived from it rather
// than tracked as separate flags — mirrors g_l2_swimlane_level.
static DumpTensorLevel g_dump_args_level = DumpTensorLevel::OFF;

extern "C" void set_platform_dump_base(uint64_t dump_data_base) { g_platform_dump_base = dump_data_base; }

extern "C" uint64_t get_platform_dump_base() { return g_platform_dump_base; }

struct DumpTaskMaskEntry {
    uint64_t task_id;
    TensorDumpArgMask mask;
    TensorDumpArgMask flags;
};
struct DumpTaskScalarDtypeEntry {
    uint64_t task_id;
    uint32_t scalar_count;
    uint8_t scalar_dtypes[32];
};
static constexpr uint64_t DUMP_TASK_MASK_EMPTY_TASK_ID = UINT64_MAX;
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
        g_dump_mask_table[i].task_id = DUMP_TASK_MASK_EMPTY_TASK_ID;
        g_dump_mask_table[i].mask = TENSOR_DUMP_ARG_MASK_NONE;
        g_dump_mask_table[i].flags = TENSOR_DUMP_ARG_MASK_NONE;
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
        g_dump_scalar_dtype_table[i].task_id = DUMP_TASK_MASK_EMPTY_TASK_ID;
        g_dump_scalar_dtype_table[i].scalar_count = 0;
        memset(g_dump_scalar_dtype_table[i].scalar_dtypes, 0, sizeof(g_dump_scalar_dtype_table[i].scalar_dtypes));
    }
    return true;
}

static void clear_dump_args_tables() {
    if (g_dump_mask_table != nullptr) {
        for (uint32_t i = 0; i < DUMP_TASK_MASK_TABLE_CAPACITY; i++) {
            g_dump_mask_table[i].task_id = DUMP_TASK_MASK_EMPTY_TASK_ID;
            g_dump_mask_table[i].mask = TENSOR_DUMP_ARG_MASK_NONE;
            g_dump_mask_table[i].flags = TENSOR_DUMP_ARG_MASK_NONE;
        }
    }
    if (g_dump_scalar_dtype_table != nullptr) {
        for (uint32_t i = 0; i < DUMP_TASK_MASK_TABLE_CAPACITY; i++) {
            g_dump_scalar_dtype_table[i].task_id = DUMP_TASK_MASK_EMPTY_TASK_ID;
            g_dump_scalar_dtype_table[i].scalar_count = 0;
            memset(g_dump_scalar_dtype_table[i].scalar_dtypes, 0, sizeof(g_dump_scalar_dtype_table[i].scalar_dtypes));
        }
    }
}

static bool resolve_dump_args_task_slot(uint64_t task_id, uint32_t *idx_out) {
    uint32_t ring_id = static_cast<uint32_t>(task_id >> 32);
    if (ring_id >= TENSOR_DUMP_MASK_POOL_MAX_RINGS) {
        return false;
    }
    uint32_t slot = static_cast<uint32_t>(task_id) & TENSOR_DUMP_MASK_POOL_DEFAULT_SLOT_MASK;
    *idx_out = (ring_id * TENSOR_DUMP_MASK_POOL_MAX_SLOTS + slot) & (DUMP_TASK_MASK_TABLE_CAPACITY - 1);
    return true;
}

extern "C" void
set_dump_args_task_scalar_dtypes(uint64_t task_id, uint32_t scalar_count, const uint8_t *scalar_dtypes) {
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
    uint32_t idx = 0;
    if (!resolve_dump_args_task_slot(task_id, &idx)) {
        return;
    }
    for (uint32_t probe = 0; probe < DUMP_TASK_MASK_TABLE_CAPACITY; probe++) {
        DumpTaskScalarDtypeEntry &entry =
            g_dump_scalar_dtype_table[(idx + probe) & (DUMP_TASK_MASK_TABLE_CAPACITY - 1)];
        if (entry.task_id == DUMP_TASK_MASK_EMPTY_TASK_ID || entry.task_id == task_id) {
            entry.task_id = task_id;
            entry.scalar_count = scalar_count;
            memcpy(entry.scalar_dtypes, scalar_dtypes, scalar_count * sizeof(uint8_t));
            return;
        }
    }
    LOG_ERROR("args dump scalar dtype table is full");
}

extern "C" bool get_dump_args_task_scalar_dtypes(uint64_t task_id, uint32_t *scalar_count, uint8_t *scalar_dtypes) {
    if (g_dump_scalar_dtype_table == nullptr || scalar_count == nullptr || scalar_dtypes == nullptr) {
        return false;
    }
    uint32_t idx = 0;
    if (!resolve_dump_args_task_slot(task_id, &idx)) {
        return false;
    }
    for (uint32_t probe = 0; probe < DUMP_TASK_MASK_TABLE_CAPACITY; probe++) {
        const DumpTaskScalarDtypeEntry &entry =
            g_dump_scalar_dtype_table[(idx + probe) & (DUMP_TASK_MASK_TABLE_CAPACITY - 1)];
        if (entry.task_id == task_id) {
            *scalar_count = entry.scalar_count;
            memcpy(scalar_dtypes, entry.scalar_dtypes, entry.scalar_count * sizeof(uint8_t));
            return true;
        }
        if (entry.task_id == DUMP_TASK_MASK_EMPTY_TASK_ID) {
            return false;
        }
    }
    return false;
}

extern "C" void set_dump_args_enabled(bool enable) {
    g_enable_dump_args = enable;
    g_dump_args_level = DumpTensorLevel::OFF;
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

extern "C" bool is_dump_args_selective_mode() { return g_dump_args_level == DumpTensorLevel::PARTIAL; }

extern "C" void set_dump_args_task_mask(uint64_t task_id, TensorDumpArgMask mask, TensorDumpArgMask flags) {
    if (mask == TENSOR_DUMP_ARG_MASK_NONE) {
        return;
    }
    if (!ensure_dump_args_mask_table()) {
        return;
    }
    uint32_t ring_id = static_cast<uint32_t>(task_id >> 32);
    if (ring_id >= TENSOR_DUMP_MASK_POOL_MAX_RINGS) {
        return;
    }
    uint32_t slot = static_cast<uint32_t>(task_id) & TENSOR_DUMP_MASK_POOL_DEFAULT_SLOT_MASK;
    uint32_t idx = (ring_id * TENSOR_DUMP_MASK_POOL_MAX_SLOTS + slot) & (DUMP_TASK_MASK_TABLE_CAPACITY - 1);
    for (uint32_t probe = 0; probe < DUMP_TASK_MASK_TABLE_CAPACITY; probe++) {
        DumpTaskMaskEntry &entry = g_dump_mask_table[(idx + probe) & (DUMP_TASK_MASK_TABLE_CAPACITY - 1)];
        if (entry.task_id == DUMP_TASK_MASK_EMPTY_TASK_ID || entry.task_id == task_id) {
            entry.task_id = task_id;
            entry.mask = mask;
            entry.flags = flags;
            return;
        }
    }
    LOG_ERROR("args dump selective mask table is full");
}

extern "C" void get_dump_args_task_masks(uint64_t task_id, TensorDumpArgMask *mask, TensorDumpArgMask *flags) {
    if (mask != nullptr) {
        *mask = TENSOR_DUMP_ARG_MASK_NONE;
    }
    if (flags != nullptr) {
        *flags = TENSOR_DUMP_ARG_MASK_NONE;
    }
    if (g_dump_mask_table == nullptr) {
        return;
    }
    uint32_t ring_id = static_cast<uint32_t>(task_id >> 32);
    if (ring_id >= TENSOR_DUMP_MASK_POOL_MAX_RINGS) {
        return;
    }
    uint32_t slot = static_cast<uint32_t>(task_id) & TENSOR_DUMP_MASK_POOL_DEFAULT_SLOT_MASK;
    uint32_t idx = (ring_id * TENSOR_DUMP_MASK_POOL_MAX_SLOTS + slot) & (DUMP_TASK_MASK_TABLE_CAPACITY - 1);
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
        if (entry.task_id == DUMP_TASK_MASK_EMPTY_TASK_ID) {
            return;
        }
    }
}

bool get_dump_arg_role_from_direction(ArgDirection dir, TensorDumpRole *role) {
    switch (dir) {
    case ArgDirection::IN:
        *role = TensorDumpRole::INPUT;
        return true;
    case ArgDirection::OUT:
        *role = TensorDumpRole::OUTPUT;
        return true;
    case ArgDirection::INOUT:
        *role = TensorDumpRole::INOUT;
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

bool should_dump_arg_at_stage(TensorDumpRole role, TensorDumpStage stage) {
    switch (role) {
    case TensorDumpRole::INPUT:
        return stage == TensorDumpStage::BEFORE_DISPATCH;
    case TensorDumpRole::OUTPUT:
        return stage == TensorDumpStage::AFTER_COMPLETION;
    case TensorDumpRole::INOUT:
        return true;
    }
    return false;
}

bool should_dump_task(TensorDumpArgMask arg_mask) {
    if (!is_dump_args_selective_mode()) {
        return true;
    }
    return arg_mask != TENSOR_DUMP_ARG_MASK_NONE;
}

bool should_dump_arg(TensorDumpArgMask arg_mask, int32_t arg_index) {
    if (!is_dump_args_selective_mode()) {
        return true;
    }
    if (arg_index < 0 || arg_index >= static_cast<int32_t>(TENSOR_DUMP_ARG_MASK_BITS)) return false;
    return (arg_mask & (TensorDumpArgMask{1} << arg_index)) != 0;
}

bool has_dump_arg_flag(TensorDumpArgMask arg_mask, int32_t arg_index) {
    if (arg_index < 0 || arg_index >= static_cast<int32_t>(TENSOR_DUMP_ARG_MASK_BITS)) return false;
    return (arg_mask & (TensorDumpArgMask{1} << arg_index)) != 0;
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
static bool wait_for_ready_queue_space(DumpDataHeader *header, int thread_idx, uint32_t *tail_out, uint32_t *head_out) {
    if (header == nullptr || thread_idx < 0 || thread_idx >= PLATFORM_MAX_AICPU_THREADS) {
        return false;
    }
    const uint32_t capacity = PLATFORM_DUMP_READYQUEUE_SIZE;
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
        if (get_sys_cnt_aicpu() - start >= kDumpQueueBackpressureWaitCycles) {
            break;
        }
    } while (true);
    return false;
}

static bool wait_for_free_queue_entry(DumpFreeQueue *free_queue, uint32_t *head_out, uint32_t *tail_out) {
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
        if (get_sys_cnt_aicpu() - start >= kDumpQueueBackpressureWaitCycles) {
            break;
        }
    } while (true);
    return false;
}

static int enqueue_dump_ready_buffer(int thread_idx, uint64_t buffer_ptr, uint32_t buffer_seq) {
    uint32_t capacity = PLATFORM_DUMP_READYQUEUE_SIZE;
    uint32_t current_tail = 0;
    uint32_t current_head = 0;
    if (!wait_for_ready_queue_space(s_dump_header, thread_idx, &current_tail, &current_head)) {
        return -1;
    }

    uint32_t next_tail = (current_tail + 1) % capacity;
    s_dump_header->queues[thread_idx][current_tail].thread_index = static_cast<uint32_t>(thread_idx);
    s_dump_header->queues[thread_idx][current_tail].buffer_ptr = buffer_ptr;
    s_dump_header->queues[thread_idx][current_tail].buffer_seq = buffer_seq;
    wmb();  // publish: entry fields visible before the tail advance
    s_dump_header->queue_tails[thread_idx] = next_tail;

    return 0;
}

static DumpMetaBuffer *try_pop_dump_meta_buffer(int thread_idx, DumpBufferState *state, uint32_t next_seq) {
    if (thread_idx < 0 || thread_idx >= PLATFORM_MAX_AICPU_THREADS || state == nullptr) {
        return nullptr;
    }
    uint32_t head = 0;
    uint32_t tail = 0;
    if (!wait_for_free_queue_entry(&state->free_queue, &head, &tail)) {
        return nullptr;
    }

    uint64_t new_ptr = state->free_queue.buffer_ptrs[head % PLATFORM_DUMP_SLOT_COUNT];
    state->free_queue.head = head + 1;
    if (new_ptr == 0) {
        return nullptr;
    }

    DumpMetaBuffer *new_buf = reinterpret_cast<DumpMetaBuffer *>(new_ptr);
    new_buf->count = 0;
    s_current_dump_buf[thread_idx] = new_buf;
    state->current_buf_ptr = new_ptr;
    state->current_buf_seq = next_seq;
    wmb();
    return new_buf;
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

    uint64_t buf_addr = reinterpret_cast<uint64_t>(cur);
    uint32_t seq = state->current_buf_seq;
    int rc = enqueue_dump_ready_buffer(thread_idx, buf_addr, seq);
    if (rc != 0) {
        account_dropped_records(state, cur->count);
        cur->count = 0;
        wmb();
        if (!s_logged_ready_queue_full[thread_idx]) {
            s_logged_ready_queue_full[thread_idx] = true;
            LOG_WARN(
                "Args dump ready queue full on thread %d after bounded wait, "
                "dropping current metadata buffer. Increase PLATFORM_DUMP_READYQUEUE_SIZE.",
                thread_idx
            );
        }
        return 0;
    }

    uint32_t next_seq = seq + 1;
    s_current_dump_buf[thread_idx] = nullptr;
    state->current_buf_ptr = 0;
    state->current_buf_seq = next_seq;
    wmb();

    if (try_pop_dump_meta_buffer(thread_idx, state, next_seq) == nullptr && !s_logged_no_free_meta_buffer[thread_idx]) {
        s_logged_no_free_meta_buffer[thread_idx] = true;
        LOG_WARN(
            "Args dump published a full metadata buffer on thread %d but no replacement was available; "
            "records will drop until recovery. Increase PLATFORM_DUMP_BUFFERS_PER_THREAD.",
            thread_idx
        );
    }

    s_buffers_switched[thread_idx]++;

    return 0;
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

static inline uint64_t get_dump_arg_num_elements(const TensorDumpInfo &info) {
    uint64_t elements = 1;
    for (uint32_t d = 0; d < info.ndims && d < PLATFORM_DUMP_MAX_DIMS; d++) {
        elements *= info.shapes[d];
    }
    return elements;
}

// PyTorch-style contiguous: strides[d] == prod(shapes[d+1..]) for all d.
static inline bool dump_arg_is_contiguous(const TensorDumpInfo &info) {
    if (info.ndims == 0) return true;
    uint64_t expected = 1;
    for (int32_t d = static_cast<int32_t>(info.ndims) - 1; d >= 0 && d < PLATFORM_DUMP_MAX_DIMS; --d) {
        if (info.strides[d] != expected) return false;
        expected *= info.shapes[d];
    }
    return true;
}

static inline void write_dump_arg_contiguous_prefix(
    CircularArenaWriter *writer, const TensorDumpInfo &info, uint64_t elem_sz, uint64_t copy_bytes
) {
    const char *src = reinterpret_cast<const char *>(info.buffer_addr) + info.start_offset * elem_sz;
    writer->write(src, copy_bytes);
}

// Recursive per-dim gather: at dim d we step shapes[d] times with element step
// strides[d]. Inner-most dim writes `shapes[d]` contiguous elements (only valid
// when strides[innermost] == 1 — non-unit innermost stride degrades to per-element).
static void gather_dump_arg_dim(
    CircularArenaWriter *writer, const TensorDumpInfo &info, uint64_t elem_sz, uint32_t dim,
    uint64_t base_element_index, uint64_t *remaining_bytes
) {
    if (*remaining_bytes == 0 || dim >= PLATFORM_DUMP_MAX_DIMS) {
        return;
    }
    if (dim + 1 >= info.ndims) {
        // Innermost dim: if strides[dim] == 1 the row is contiguous in memory and
        // a single memcpy works. Otherwise emit element-by-element.
        const uint32_t s = info.strides[dim];
        if (s == 1) {
            const char *src = reinterpret_cast<const char *>(info.buffer_addr) + base_element_index * elem_sz;
            uint64_t row_bytes =
                static_cast<uint64_t>(info.shapes[dim]) * elem_sz;  // promote first to avoid uint32 overflow
            uint64_t bytes_to_copy = (row_bytes < *remaining_bytes) ? row_bytes : *remaining_bytes;
            writer->write(src, bytes_to_copy);
            *remaining_bytes -= bytes_to_copy;
        } else {
            for (uint32_t i = 0; i < info.shapes[dim] && *remaining_bytes >= elem_sz; i++) {
                uint64_t pos = base_element_index + static_cast<uint64_t>(i) * static_cast<uint64_t>(s);
                const char *src = reinterpret_cast<const char *>(info.buffer_addr) + pos * elem_sz;
                writer->write(src, elem_sz);
                *remaining_bytes -= elem_sz;
            }
        }
        return;
    }

    const int32_t s = info.strides[dim];
    for (uint32_t i = 0; i < info.shapes[dim] && *remaining_bytes > 0; i++) {
        uint64_t next_base = base_element_index + static_cast<uint64_t>(i) * static_cast<uint64_t>(s);
        gather_dump_arg_dim(writer, info, elem_sz, dim + 1, next_base, remaining_bytes);
    }
}

static inline void write_dump_arg_logical_prefix(
    CircularArenaWriter *writer, const TensorDumpInfo &info, uint64_t elem_sz, uint64_t copy_bytes
) {
    if (copy_bytes == 0) {
        return;
    }
    if (dump_arg_is_contiguous(info)) {
        write_dump_arg_contiguous_prefix(writer, info, elem_sz, copy_bytes);
        return;
    }
    // Walk the view origin first; per-dim recursion adds (i * strides[d]) for each axis.
    uint64_t remaining_bytes = copy_bytes;
    gather_dump_arg_dim(writer, info, elem_sz, 0, info.start_offset, &remaining_bytes);
}

void dump_args_init(int num_dump_threads) {
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

    // Latch dump level from the host-written header before any task is dumped.
    // PARTIAL → selective (only Arg::dump()-marked args); FULL → every task;
    // FULL_JSON_ONLY → every task, metadata only (no payload copied into arena).
    g_dump_args_level = static_cast<DumpTensorLevel>(s_dump_header->dump_tensor_level);

    LOG_INFO_V0("Initializing args dump for %d threads", num_dump_threads);

    // Pop initial metadata buffer from free_queue for each thread
    for (int t = 0; t < num_dump_threads; t++) {
        DumpBufferState *state = get_dump_buffer_state(dump_base, t);
        s_dump_states[t] = state;

        rmb();
        uint32_t head = state->free_queue.head;
        uint32_t tail = state->free_queue.tail;
        if (head != tail) {
            (void)try_pop_dump_meta_buffer(t, state, 0);
            uint64_t buf_ptr = state->current_buf_ptr;
            LOG_DEBUG("Thread %d: popped initial dump buffer (addr=0x%lx)", t, buf_ptr);
        } else {
            LOG_ERROR("Thread %d: dump free_queue is empty during init!", t);
            s_current_dump_buf[t] = nullptr;
            state->current_buf_ptr = 0;
        }
    }

    memset(s_logged_ready_queue_full, 0, sizeof(s_logged_ready_queue_full));
    memset(s_logged_no_free_meta_buffer, 0, sizeof(s_logged_no_free_meta_buffer));
    memset(s_records_written, 0, sizeof(s_records_written));
    memset(s_buffers_switched, 0, sizeof(s_buffers_switched));
    memset(s_buffers_flushed, 0, sizeof(s_buffers_flushed));
}

int dump_arg_record(int thread_idx, const TensorDumpInfo &info) {
    if (s_dump_header == nullptr) {
        return -1;
    }
    if (thread_idx < 0 || thread_idx >= PLATFORM_MAX_AICPU_THREADS) {
        return -1;
    }

    DumpBufferState *state = s_dump_states[thread_idx];
    DumpMetaBuffer *buf = s_current_dump_buf[thread_idx];
    if (buf == nullptr) {
        buf = try_pop_dump_meta_buffer(thread_idx, state, state != nullptr ? state->current_buf_seq : 0);
        if (buf == nullptr) {
            account_dropped_records(state, 1);
            return -1;
        }
    }

    // Switch metadata buffer if full
    if (buf->count >= PLATFORM_DUMP_RECORDS_PER_BUFFER) {
        if (switch_dump_meta_buffer(thread_idx) != 0) {
            return -1;  // No free buffer
        }
        buf = s_current_dump_buf[thread_idx];
        if (buf == nullptr) {
            buf = try_pop_dump_meta_buffer(thread_idx, state, state != nullptr ? state->current_buf_seq : 0);
            if (buf == nullptr) {
                account_dropped_records(state, 1);
                return -1;
            }
        }
    }

    // Reserve space in arena
    // Compute actual tensor data size from shape (not buffer.size which may include padding)
    TensorDumpKind kind = static_cast<TensorDumpKind>(info.kind);
    uint64_t actual_elements = get_dump_arg_num_elements(info);
    uint64_t elem_sz = get_element_size(static_cast<DataType>(info.dtype));
    uint64_t bytes = actual_elements * elem_sz;
    uint64_t copy_bytes = bytes;
    bool truncated = false;
    bool is_contiguous = dump_arg_is_contiguous(info);
    bool is_scalar = kind == TensorDumpKind::SCALAR;

    if (is_scalar || g_dump_args_level == DumpTensorLevel::FULL_JSON_ONLY) {
        // JSON-only level captures full metadata but no payload, so the
        // record carries shape/dtype/strides with payload_size == 0.
        copy_bytes = 0;
    } else if (bytes > state->arena_size) {
        // Tensor larger than entire arena — copy a partial sample
        copy_bytes = state->arena_size / 2;
        truncated = true;
    }

    uint64_t offset = state->arena_write_offset;
    state->arena_write_offset = offset + copy_bytes;

    // Copy tensor data into arena (circular wraparound)
    char *arena = reinterpret_cast<char *>(state->arena_base);
    uint64_t arena_sz = state->arena_size;
    CircularArenaWriter writer = {arena, arena_sz, offset, 0};
    if (!is_scalar && copy_bytes > 0) {
        write_dump_arg_logical_prefix(&writer, info, elem_sz, copy_bytes);
    }
    wmb();

    // Append metadata record
    uint32_t idx = buf->count;
    TensorDumpRecord *rec = &buf->records[idx];
    rec->task_id = info.task_id;
    rec->arg_index = info.arg_index;
    rec->is_contiguous = is_contiguous ? 1 : 0;
    rec->role = static_cast<uint8_t>(info.role);
    rec->stage = static_cast<uint8_t>(info.stage);
    rec->ndims = info.ndims;
    rec->dtype = info.dtype;
    rec->truncated = truncated ? 1 : 0;
    rec->payload_offset = offset;
    rec->payload_size = copy_bytes;
    rec->scalar_value = is_scalar ? info.scalar_value : 0;
    rec->kind = info.kind;
    rec->flags = info.flags;
    // kernel_id is bounded by RUNTIME_MAX_FUNC_ID (1024), far below the 0xFFFF
    // "unknown" sentinel, so this narrowing is lossless and never collides.
    // func_ids carries the task's active-subtask set (its mix membership).
    rec->func_count = static_cast<uint8_t>(info.func_count);
    for (int32_t i = 0; i < info.func_count && i < TENSOR_DUMP_MAX_FUNC_IDS; i++) {
        rec->func_ids[i] = static_cast<uint16_t>(info.func_ids[i]);  // -1 -> 0xFFFF (unknown)
    }
    rec->start_offset = info.start_offset;
    for (int d = 0; d < info.ndims && d < PLATFORM_DUMP_MAX_DIMS; d++) {
        rec->shapes[d] = info.shapes[d];
        rec->strides[d] = info.strides[d];
    }
    buf->count = idx + 1;
    wmb();

    s_records_written[thread_idx]++;

    return 0;
}

void dump_args_flush(int thread_idx) {
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
            // ready_queue full at end-of-run: account the loss and clear the
            // buffer so host reconcile sees a clean state (current_buf_ptr=0)
            // and dropped == flush failures rather than silent wip-mismatch.
            // Bounded to one per thread per run (unlike the hot-path
            // dump_arg_record site), so no spam guard is needed here.
            LOG_ERROR(
                "Thread %d: failed to flush args-dump buffer (ready_queue full), %u records lost!", thread_idx,
                buf->count
            );
            account_dropped_records(state, buf->count);
            buf->count = 0;
            s_current_dump_buf[thread_idx] = nullptr;
            state->current_buf_ptr = 0;
            wmb();
        }
    }

    s_buffers_flushed[thread_idx]++;
    uint32_t dropped = s_dump_states[thread_idx] ? s_dump_states[thread_idx]->dropped_record_count : 0;
    LOG_INFO_V0(
        "Thread %d: dump_args_flush (records=%u, buf_switches=%u, flushes=%u, dropped=%u)", thread_idx,
        s_records_written[thread_idx], s_buffers_switched[thread_idx], s_buffers_flushed[thread_idx], dropped
    );
}
