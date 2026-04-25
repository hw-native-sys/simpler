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
 * @brief AICPU tensor dump collection implementation
 *
 * - Per-thread DumpBufferState with SPSC free queues
 * - Per-thread ready queue for handing off full metadata buffers
 * - Per-thread circular arena for tensor payload data
 */

#include "aicpu/tensor_dump_aicpu.h"

#include <cstring>

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
static bool s_logged_dump_layout_mismatch = false;
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

extern "C" void set_platform_dump_base(uint64_t dump_data_base) { g_platform_dump_base = dump_data_base; }

extern "C" uint64_t get_platform_dump_base() { return g_platform_dump_base; }

static bool g_enable_dump_tensor = false;

extern "C" void set_dump_tensor_enabled(bool enable) { g_enable_dump_tensor = enable; }

extern "C" bool is_dump_tensor_enabled() { return g_enable_dump_tensor; }

bool get_tensor_dump_role_from_direction(ArgDirection dir, TensorDumpRole *role) {
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

bool should_dump_tensor_at_stage(TensorDumpRole role, TensorDumpStage stage) {
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

bool try_log_tensor_dump_layout_mismatch() {
    if (s_logged_dump_layout_mismatch) {
        return false;
    }
    s_logged_dump_layout_mismatch = true;
    return true;
}

/**
 * Enqueue a full dump metadata buffer to the thread's ready queue.
 */
static int enqueue_dump_ready_buffer(int thread_idx, uint64_t buffer_ptr, uint32_t buffer_seq) {
    uint32_t capacity = PLATFORM_DUMP_READYQUEUE_SIZE;
    uint32_t current_tail = s_dump_header->queue_tails[thread_idx];
    uint32_t current_head = s_dump_header->queue_heads[thread_idx];

    uint32_t next_tail = (current_tail + 1) % capacity;
    if (next_tail == current_head) {
        return -1;  // Queue full
    }

    s_dump_header->queues[thread_idx][current_tail].thread_index = static_cast<uint32_t>(thread_idx);
    s_dump_header->queues[thread_idx][current_tail].buffer_ptr = buffer_ptr;
    s_dump_header->queues[thread_idx][current_tail].buffer_seq = buffer_seq;
    wmb();
    s_dump_header->queue_tails[thread_idx] = next_tail;
    wmb();

    return 0;
}

/**
 * Maximum spin-wait iterations when free_queue or ready_queue is exhausted.
 * Gives host mgmt_loop time to replenish before falling back to buffer overwrite.
 */
static constexpr uint32_t DUMP_SPIN_WAIT_LIMIT = 1000000;

/**
 * Switch metadata buffer: enqueue the full buffer, pop a new one.
 * Spin-waits briefly for host to replenish before falling back to overwrite.
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

    // Spin-wait for a free buffer, giving host mgmt_loop time to replenish
    rmb();
    uint32_t head = state->free_queue.head;
    uint32_t tail = state->free_queue.tail;
    if (head == tail) {
        for (uint32_t spin = 0; spin < DUMP_SPIN_WAIT_LIMIT; spin++) {
            rmb();
            head = state->free_queue.head;
            tail = state->free_queue.tail;
            if (head != tail) {
                break;
            }
        }
    }
    if (head == tail) {
        // Still empty after spin — overwrite current buffer
        account_dropped_records(state, cur->count);
        cur->count = 0;
        wmb();
        if (!s_logged_no_free_meta_buffer[thread_idx]) {
            s_logged_no_free_meta_buffer[thread_idx] = true;
            LOG_WARN(
                "Tensor dump ran out of free metadata buffers on thread %d after spin-wait, "
                "overwriting current buffer. Increase PLATFORM_DUMP_BUFFERS_PER_THREAD.",
                thread_idx
            );
        }
        return 0;
    }

    // Enqueue the full buffer (spin-wait if ready queue is full)
    uint64_t buf_addr = reinterpret_cast<uint64_t>(cur);
    uint32_t seq = state->current_buf_seq;
    int rc = enqueue_dump_ready_buffer(thread_idx, buf_addr, seq);
    if (rc != 0) {
        for (uint32_t spin = 0; spin < DUMP_SPIN_WAIT_LIMIT; spin++) {
            rmb();
            rc = enqueue_dump_ready_buffer(thread_idx, buf_addr, seq);
            if (rc == 0) {
                break;
            }
        }
    }
    if (rc != 0) {
        // Still full after spin — overwrite current buffer
        account_dropped_records(state, cur->count);
        cur->count = 0;
        wmb();
        if (!s_logged_ready_queue_full[thread_idx]) {
            s_logged_ready_queue_full[thread_idx] = true;
            LOG_WARN(
                "Tensor dump ready queue full on thread %d after spin-wait, "
                "overwriting current buffer. Increase PLATFORM_DUMP_READYQUEUE_SIZE.",
                thread_idx
            );
        }
        return 0;
    }

    // Pop next buffer from free_queue
    uint64_t new_ptr = state->free_queue.buffer_ptrs[head % PLATFORM_DUMP_SLOT_COUNT];
    rmb();
    state->free_queue.head = head + 1;

    DumpMetaBuffer *new_buf = reinterpret_cast<DumpMetaBuffer *>(new_ptr);
    new_buf->count = 0;
    s_current_dump_buf[thread_idx] = new_buf;
    state->current_buf_ptr = new_ptr;
    state->current_buf_seq = seq + 1;
    wmb();

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

static inline uint64_t get_tensor_dump_num_elements(const TensorDumpInfo &info) {
    uint64_t elements = 1;
    for (uint32_t d = 0; d < info.ndims && d < PLATFORM_DUMP_MAX_DIMS; d++) {
        elements *= info.shapes[d];
    }
    return elements;
}

static inline bool tensor_dump_is_contiguous(const TensorDumpInfo &info) {
    if (info.ndims == 0) {
        return true;
    }
    for (uint32_t d = 1; d < info.ndims && d < PLATFORM_DUMP_MAX_DIMS; d++) {
        if (info.shapes[d] != info.raw_shapes[d]) {
            return false;
        }
    }
    return true;
}

static inline uint64_t tensor_dump_start_offset_elements(const TensorDumpInfo &info) {
    uint64_t result = 0;
    uint64_t stride = 1;
    for (int d = static_cast<int>(info.ndims) - 1; d >= 0; d--) {
        result += static_cast<uint64_t>(info.offsets[d]) * stride;
        stride *= info.raw_shapes[d];
    }
    return result;
}

static inline void write_tensor_dump_contiguous_prefix(
    CircularArenaWriter *writer, const TensorDumpInfo &info, uint64_t elem_sz, uint64_t copy_bytes
) {
    uint64_t start_offset = tensor_dump_start_offset_elements(info);
    const char *src = reinterpret_cast<const char *>(info.buffer_addr) + start_offset * elem_sz;
    writer->write(src, copy_bytes);
}

static void gather_tensor_dump_dim(
    CircularArenaWriter *writer, const TensorDumpInfo &info, uint64_t elem_sz, uint32_t dim,
    uint64_t base_element_index, uint64_t *remaining_bytes
) {
    if (*remaining_bytes == 0 || dim >= PLATFORM_DUMP_MAX_DIMS) {
        return;
    }
    if (dim + 1 >= info.ndims) {
        uint64_t row_start = base_element_index + info.offsets[dim];
        const char *src = reinterpret_cast<const char *>(info.buffer_addr) + row_start * elem_sz;
        uint64_t row_bytes = static_cast<uint64_t>(info.shapes[dim]) * elem_sz;
        uint64_t bytes_to_copy = (row_bytes < *remaining_bytes) ? row_bytes : *remaining_bytes;
        writer->write(src, bytes_to_copy);
        *remaining_bytes -= bytes_to_copy;
        return;
    }

    uint64_t inner_stride = 1;
    for (uint32_t d = dim + 1; d < info.ndims && d < PLATFORM_DUMP_MAX_DIMS; d++) {
        inner_stride *= info.raw_shapes[d];
    }
    for (uint32_t i = 0; i < info.shapes[dim] && *remaining_bytes > 0; i++) {
        uint64_t next_base = base_element_index + (static_cast<uint64_t>(info.offsets[dim]) + i) * inner_stride;
        gather_tensor_dump_dim(writer, info, elem_sz, dim + 1, next_base, remaining_bytes);
    }
}

static inline void write_tensor_dump_logical_prefix(
    CircularArenaWriter *writer, const TensorDumpInfo &info, uint64_t elem_sz, uint64_t copy_bytes
) {
    if (copy_bytes == 0) {
        return;
    }
    if (tensor_dump_is_contiguous(info)) {
        write_tensor_dump_contiguous_prefix(writer, info, elem_sz, copy_bytes);
        return;
    }

    uint64_t remaining_bytes = copy_bytes;
    gather_tensor_dump_dim(writer, info, elem_sz, 0, 0, &remaining_bytes);
}

void dump_tensor_init(int num_dump_threads) {
    void *dump_base = reinterpret_cast<void *>(get_platform_dump_base());
    if (dump_base == nullptr) {
        LOG_ERROR("platform dump base is NULL, cannot initialize tensor dump");
        return;
    }

    s_dump_header = get_dump_header(dump_base);

    LOG_INFO("Initializing tensor dump for %d threads", num_dump_threads);

    // Pop initial metadata buffer from free_queue for each thread
    for (int t = 0; t < num_dump_threads; t++) {
        DumpBufferState *state = get_dump_buffer_state(dump_base, t);
        s_dump_states[t] = state;

        rmb();
        uint32_t head = state->free_queue.head;
        uint32_t tail = state->free_queue.tail;
        if (head != tail) {
            uint64_t buf_ptr = state->free_queue.buffer_ptrs[head % PLATFORM_DUMP_SLOT_COUNT];
            rmb();
            state->free_queue.head = head + 1;
            wmb();

            DumpMetaBuffer *buf = reinterpret_cast<DumpMetaBuffer *>(buf_ptr);
            buf->count = 0;
            s_current_dump_buf[t] = buf;
            state->current_buf_ptr = buf_ptr;
            state->current_buf_seq = 0;
            wmb();
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

int dump_tensor_record(int thread_idx, const TensorDumpInfo &info) {
    if (s_dump_header == nullptr) {
        return -1;
    }
    if (thread_idx < 0 || thread_idx >= PLATFORM_MAX_AICPU_THREADS) {
        return -1;
    }

    DumpBufferState *state = s_dump_states[thread_idx];
    DumpMetaBuffer *buf = s_current_dump_buf[thread_idx];
    if (buf == nullptr) {
        return -1;
    }

    // Switch metadata buffer if full
    if (buf->count >= PLATFORM_DUMP_RECORDS_PER_BUFFER) {
        if (switch_dump_meta_buffer(thread_idx) != 0) {
            return -1;  // No free buffer
        }
        buf = s_current_dump_buf[thread_idx];
        if (buf == nullptr) {
            return -1;
        }
    }

    // Reserve space in arena
    // Compute actual tensor data size from shape (not buffer.size which may include padding)
    uint64_t actual_elements = get_tensor_dump_num_elements(info);
    uint64_t elem_sz = get_element_size(static_cast<DataType>(info.dtype));
    uint64_t bytes = actual_elements * elem_sz;
    uint64_t copy_bytes = bytes;
    bool truncated = false;
    bool is_contiguous = tensor_dump_is_contiguous(info);

    if (bytes > state->arena_size) {
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
    write_tensor_dump_logical_prefix(&writer, info, elem_sz, copy_bytes);
    wmb();

    // Append metadata record
    uint32_t idx = buf->count;
    TensorDumpRecord *rec = &buf->records[idx];
    rec->task_id = info.task_id;
    rec->subtask_id = info.subtask_id;
    rec->func_id = info.func_id;
    rec->arg_index = info.arg_index;
    rec->is_contiguous = is_contiguous ? 1 : 0;
    rec->role = static_cast<uint8_t>(info.role);
    rec->stage = static_cast<uint8_t>(info.stage);
    rec->ndims = info.ndims;
    rec->dtype = info.dtype;
    rec->truncated = truncated ? 1 : 0;
    rec->payload_offset = offset;
    rec->payload_size = copy_bytes;
    for (int d = 0; d < info.ndims && d < PLATFORM_DUMP_MAX_DIMS; d++) {
        rec->raw_shapes[d] = info.raw_shapes[d];
        rec->shapes[d] = info.shapes[d];
        rec->offsets[d] = info.offsets[d];
    }
    buf->count = idx + 1;
    wmb();

    s_records_written[thread_idx]++;

    return 0;
}

void dump_tensor_flush(int thread_idx) {
    if (s_dump_header == nullptr) {
        return;
    }
    if (thread_idx < 0 || thread_idx >= PLATFORM_MAX_AICPU_THREADS) {
        return;
    }

    DumpMetaBuffer *buf = s_current_dump_buf[thread_idx];
    if (buf != nullptr && buf->count > 0) {
        uint64_t buf_addr = reinterpret_cast<uint64_t>(buf);
        uint32_t seq = s_dump_states[thread_idx]->current_buf_seq;
        if (enqueue_dump_ready_buffer(thread_idx, buf_addr, seq) != 0) {
            account_dropped_records(s_dump_states[thread_idx], buf->count);
            buf->count = 0;
            wmb();
            if (!s_logged_ready_queue_full[thread_idx]) {
                s_logged_ready_queue_full[thread_idx] = true;
                LOG_WARN(
                    "Tensor dump ready queue is full on thread %d, so the current metadata buffer will be "
                    "overwritten. Increase PLATFORM_DUMP_READYQUEUE_SIZE.",
                    thread_idx
                );
            }
        }
        s_current_dump_buf[thread_idx] = nullptr;
        s_dump_states[thread_idx]->current_buf_ptr = 0;
    }

    s_buffers_flushed[thread_idx]++;
    uint32_t dropped = s_dump_states[thread_idx] ? s_dump_states[thread_idx]->dropped_record_count : 0;
    LOG_INFO(
        "Thread %d: dump_tensor_flush (records=%u, buf_switches=%u, flushes=%u, dropped=%u)", thread_idx,
        s_records_written[thread_idx], s_buffers_switched[thread_idx], s_buffers_flushed[thread_idx], dropped
    );
}
