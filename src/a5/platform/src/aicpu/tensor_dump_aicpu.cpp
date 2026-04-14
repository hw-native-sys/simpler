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
 * @brief AICPU tensor dump collection implementation (memcpy-based)
 *
 * Simplified version of A2A3's tensor_dump_aicpu.cpp:
 * - No SPSC free queues or ready queues
 * - Per-thread DumpBuffer with count-first layout (like PerfBuffer)
 * - Per-thread circular arena for tensor payload data
 * - Silently drops records when DumpBuffer is full
 * - Host copies everything back after stream sync
 */

#include "aicpu/tensor_dump_aicpu.h"

#include <cstring>

#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "common/unified_log.h"

// =============================================================================
// Static State
// =============================================================================

static uint64_t g_platform_dump_base = 0;
static bool g_enable_dump_tensor = false;

static DumpSetupHeader *s_setup_header = nullptr;
static DumpBuffer *s_dump_buffers[PLATFORM_MAX_AICPU_THREADS] = {};
static DumpArenaHeader *s_arena_headers[PLATFORM_MAX_AICPU_THREADS] = {};
static char *s_arena_data[PLATFORM_MAX_AICPU_THREADS] = {};

static bool s_logged_dump_layout_mismatch = false;
static uint32_t s_records_written[PLATFORM_MAX_AICPU_THREADS] = {};

// =============================================================================
// Extern "C" API
// =============================================================================

extern "C" void set_platform_dump_base(uint64_t dump_data_base) { g_platform_dump_base = dump_data_base; }

extern "C" uint64_t get_platform_dump_base() { return g_platform_dump_base; }

extern "C" void set_enable_dump_tensor(bool enable) { g_enable_dump_tensor = enable; }

extern "C" bool get_enable_dump_tensor() { return g_enable_dump_tensor; }

// =============================================================================
// Helper Functions (same as A2A3)
// =============================================================================

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

// =============================================================================
// Circular Arena Writer (same as A2A3)
// =============================================================================

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

// =============================================================================
// Public API Implementation
// =============================================================================

void dump_tensor_init(int num_dump_threads) {
    void *dump_base = reinterpret_cast<void *>(get_platform_dump_base());
    if (dump_base == nullptr) {
        LOG_ERROR("platform dump base is NULL, cannot initialize tensor dump");
        return;
    }

    s_setup_header = get_dump_setup_header(dump_base);

    LOG_INFO("Initializing tensor dump for %d threads (memcpy-based)", num_dump_threads);

    for (int t = 0; t < num_dump_threads && t < PLATFORM_MAX_AICPU_THREADS; t++) {
        uint64_t buf_ptr = s_setup_header->dump_buffer_ptrs[t];
        uint64_t arena_hdr_ptr = s_setup_header->arena_header_ptrs[t];
        uint64_t arena_data_ptr = s_setup_header->arena_data_ptrs[t];

        if (buf_ptr == 0) {
            LOG_ERROR("Thread %d: dump_buffer_ptrs[%d] is NULL during init!", t, t);
            s_dump_buffers[t] = nullptr;
            continue;
        }

        DumpBuffer *buf = reinterpret_cast<DumpBuffer *>(buf_ptr);
        buf->count = 0;
        buf->dropped_count = 0;
        s_dump_buffers[t] = buf;

        s_arena_headers[t] = reinterpret_cast<DumpArenaHeader *>(arena_hdr_ptr);
        s_arena_data[t] = reinterpret_cast<char *>(arena_data_ptr);

        if (s_arena_headers[t] != nullptr) {
            s_arena_headers[t]->write_offset = 0;
        }

        LOG_DEBUG(
            "Thread %d: DumpBuffer at 0x%lx, arena at 0x%lx (size=%lu)", t, buf_ptr, arena_data_ptr,
            s_setup_header->arena_sizes[t]
        );
    }

    memset(s_records_written, 0, sizeof(s_records_written));
    s_logged_dump_layout_mismatch = false;

    wmb();
    LOG_INFO("Tensor dump initialized for %d threads", num_dump_threads);
}

int dump_tensor_record(int thread_idx, const TensorDumpInfo &info) {
    if (s_setup_header == nullptr) {
        return -1;
    }

    DumpBuffer *buf = s_dump_buffers[thread_idx];
    if (buf == nullptr) {
        return -1;
    }

    TensorDumpRecord *records = get_dump_buffer_records(buf);

    // Check capacity — drop if full
    uint32_t count = buf->count;
    if (count >= buf->capacity) {
        uint32_t prev = buf->dropped_count;
        uint32_t next = prev + 1;
        buf->dropped_count = (next < prev) ? UINT32_MAX : next;
        return 0;
    }

    // Compute tensor data size
    uint64_t actual_elements = get_tensor_dump_num_elements(info);
    uint64_t elem_sz = get_element_size(static_cast<DataType>(info.dtype));
    uint64_t bytes = actual_elements * elem_sz;
    uint64_t copy_bytes = bytes;
    bool truncated = false;
    bool is_contiguous = tensor_dump_is_contiguous(info);

    DumpArenaHeader *arena_hdr = s_arena_headers[thread_idx];
    char *arena = s_arena_data[thread_idx];

    if (arena_hdr != nullptr && arena != nullptr) {
        uint64_t arena_sz = arena_hdr->arena_size;
        if (bytes > arena_sz) {
            copy_bytes = arena_sz / 2;
            truncated = true;
        }

        uint64_t offset = arena_hdr->write_offset;
        arena_hdr->write_offset = offset + copy_bytes;

        CircularArenaWriter writer = {arena, arena_sz, offset, 0};
        write_tensor_dump_logical_prefix(&writer, info, elem_sz, copy_bytes);
        wmb();

        // Fill metadata record
        TensorDumpRecord *rec = &records[count];
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
    } else {
        // No arena — record metadata only, no payload
        TensorDumpRecord *rec = &records[count];
        rec->task_id = info.task_id;
        rec->subtask_id = info.subtask_id;
        rec->func_id = info.func_id;
        rec->arg_index = info.arg_index;
        rec->is_contiguous = is_contiguous ? 1 : 0;
        rec->role = static_cast<uint8_t>(info.role);
        rec->stage = static_cast<uint8_t>(info.stage);
        rec->ndims = info.ndims;
        rec->dtype = info.dtype;
        rec->truncated = 1;
        rec->payload_offset = 0;
        rec->payload_size = 0;
        for (int d = 0; d < info.ndims && d < PLATFORM_DUMP_MAX_DIMS; d++) {
            rec->raw_shapes[d] = info.raw_shapes[d];
            rec->shapes[d] = info.shapes[d];
            rec->offsets[d] = info.offsets[d];
        }
    }

    buf->count = count + 1;
    wmb();

    if (thread_idx >= 0 && thread_idx < PLATFORM_MAX_AICPU_THREADS) {
        s_records_written[thread_idx]++;
    }

    return 0;
}

void dump_tensor_flush(int thread_idx) {
    // In the memcpy design, flush is a no-op for data — host reads after sync.
    // Log per-thread statistics for diagnostics.
    if (thread_idx >= 0 && thread_idx < PLATFORM_MAX_AICPU_THREADS) {
        DumpBuffer *buf = s_dump_buffers[thread_idx];
        uint32_t dropped = (buf != nullptr) ? buf->dropped_count : 0;
        LOG_INFO(
            "Thread %d: dump_tensor_flush (records=%u, dropped=%u)", thread_idx, s_records_written[thread_idx], dropped
        );
    }
}
