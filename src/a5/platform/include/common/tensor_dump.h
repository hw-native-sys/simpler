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
 * @file tensor_dump.h
 * @brief Tensor dump data structures for device-to-host tensor collection (memcpy-based)
 *
 * A5 simplified design: pre-allocated buffers + direct write + memcpy collect-after-sync.
 * Mirrors L2PerfCollector pattern — no shared memory, no background threads,
 * no SPSC queues.
 *
 * Memory layout (allocated only when enable_dump_tensor=true):
 *
 *   DumpSetupHeader (single, published via kernel_args.dump_data_base)
 *   ├── dump_buffer_ptrs[]   → per-thread DumpBuffer (count + records[])
 *   ├── arena_header_ptrs[]  → per-thread DumpArenaHeader (write_offset)
 *   └── arena_data_ptrs[]    → per-thread arena data region
 *
 * After stream sync, host copies everything back via rtMemcpy / memcpy.
 */

#ifndef SRC_A5_PLATFORM_INCLUDE_COMMON_TENSOR_DUMP_H_
#define SRC_A5_PLATFORM_INCLUDE_COMMON_TENSOR_DUMP_H_

#include <cstddef>
#include <cstdint>

#include "common/platform_config.h"

// =============================================================================
// Constants
// =============================================================================

constexpr uint32_t TENSOR_DUMP_MAGIC = 0x44554D50;  // "DUMP"

// =============================================================================
// TensorDumpRole - Formal kernel signature direction
// =============================================================================

enum class TensorDumpRole : uint8_t {
    INPUT = 0,
    OUTPUT = 1,
    INOUT = 2,
};

// =============================================================================
// TensorDumpStage - When the tensor was captured
// =============================================================================

enum class TensorDumpStage : uint8_t {
    BEFORE_DISPATCH = 0,
    AFTER_COMPLETION = 1,
};

// =============================================================================
// TensorDumpRecord - Single Tensor Dump Entry (128B = 2 cache lines)
// =============================================================================

/**
 * Per-tensor metadata + payload reference.
 * Identical layout to A2A3 for binary compatibility.
 *
 * Cache line 1 (64B): identifiers, payload location, compact scalar metadata
 * Cache line 2 (64B): logical/source layout arrays
 */
struct alignas(64) TensorDumpRecord {
    // === Cache line 1 (64B) ===
    uint64_t task_id;         // PTO2 encoding or plain task index
    uint8_t subtask_id;       // PTO2SubtaskSlot raw value (AIC=0, AIV0=1, AIV1=2)
    uint8_t role;             // TensorDumpRole (formal callable signature)
    uint8_t stage;            // TensorDumpStage (before/after execution)
    uint8_t ndims;            // Number of dimensions
    uint32_t func_id;         // Kernel function identifier
    uint32_t arg_index;       // Position in PTO2TaskPayload::tensors[]
    uint8_t dtype;            // DataType raw enum value
    uint8_t truncated;        // 1 if payload was truncated (tensor > arena capacity)
    uint8_t is_contiguous;    // 1 when source view is already contiguous
    uint8_t pad0_align;       // Explicit alignment before 64-bit payload offsets
    uint64_t payload_offset;  // Monotonic byte offset into thread arena
    uint64_t payload_size;    // Bytes actually copied (may be < full tensor bytes)
    uint8_t pad0[24];         // Preserve 64B cache-line layout

    // === Cache line 2 (64B) ===
    uint32_t shapes[PLATFORM_DUMP_MAX_DIMS];      // Current view shape
    uint32_t offsets[PLATFORM_DUMP_MAX_DIMS];     // Multi-dimensional offsets
    uint32_t raw_shapes[PLATFORM_DUMP_MAX_DIMS];  // Underlying source layout shape
    uint8_t pad1[4];                              // Pad to 128 bytes
} __attribute__((aligned(64)));

static_assert(sizeof(TensorDumpRecord) == 128, "TensorDumpRecord must be 128 bytes (2 cache lines)");

// =============================================================================
// DumpBuffer - Per-Thread Record Buffer (count-first, like L2PerfBuffer)
// =============================================================================

/**
 * Per-thread dump record buffer. AICPU writes records sequentially;
 * when count reaches capacity, further records are silently dropped.
 * Host copies the buffer back after stream sync.
 */
struct DumpBuffer {
    volatile uint32_t count;          // Records written by AICPU (at offset 0)
    uint32_t capacity;                // Max records (set by host during init)
    volatile uint32_t dropped_count;  // Records dropped (buffer or arena full)
    uint32_t pad[13];                 // Pad header to 64B cache line
    // TensorDumpRecord records[] follows (flexible array member)
    // Access via: reinterpret_cast<TensorDumpRecord *>(this + 1)
} __attribute__((aligned(64)));

static_assert(sizeof(DumpBuffer) == 64, "DumpBuffer header must be 64 bytes");

// =============================================================================
// DumpArenaHeader - Per-Thread Arena Metadata
// =============================================================================

/**
 * Per-thread arena metadata. Separate from the arena data region
 * so host can read just the header to determine how much data was written.
 */
struct DumpArenaHeader {
    volatile uint64_t write_offset;  // Monotonic write cursor (AICPU increments)
    uint64_t arena_size;             // Total arena bytes (set by host)
    uint32_t pad[12];                // Pad to 64B
} __attribute__((aligned(64)));

static_assert(sizeof(DumpArenaHeader) == 64, "DumpArenaHeader must be 64 bytes");

// =============================================================================
// DumpSetupHeader - Host-Initialized, AICPU Reads
// =============================================================================

/**
 * Setup header published via kernel_args.dump_data_base.
 * Host initializes all fields and copies to device before execution.
 * AICPU reads pointers during dump_tensor_init().
 */
struct DumpSetupHeader {
    uint32_t num_dump_threads;
    uint32_t records_per_buffer;
    uint32_t magic;
    uint32_t pad0;
    // Per-thread device pointers
    uint64_t dump_buffer_ptrs[PLATFORM_MAX_AICPU_THREADS];   // -> DumpBuffer
    uint64_t arena_header_ptrs[PLATFORM_MAX_AICPU_THREADS];  // -> DumpArenaHeader
    uint64_t arena_data_ptrs[PLATFORM_MAX_AICPU_THREADS];    // -> arena data region
    uint64_t arena_sizes[PLATFORM_MAX_AICPU_THREADS];
} __attribute__((aligned(64)));

// =============================================================================
// TensorDumpInfo - Lightweight Info Struct (passed from runtime to platform API)
// =============================================================================

/**
 * Caller fills this struct from runtime-specific tensor types.
 * Platform layer is agnostic to runtime-specific types (Tensor, PTO2TaskPayload, etc.).
 * Identical to A2A3 TensorDumpInfo for API compatibility.
 */
struct TensorDumpInfo {
    uint64_t task_id;
    uint8_t subtask_id;
    TensorDumpRole role;
    TensorDumpStage stage;
    uint8_t dtype;
    uint8_t ndims;
    uint32_t func_id;
    uint32_t arg_index;
    uint64_t buffer_addr;
    uint32_t shapes[PLATFORM_DUMP_MAX_DIMS];
    uint32_t offsets[PLATFORM_DUMP_MAX_DIMS];
    uint32_t raw_shapes[PLATFORM_DUMP_MAX_DIMS];
};

// =============================================================================
// Helper Functions - Memory Layout
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Calculate DumpBuffer allocation size (header + records array).
 *
 * @param capacity Number of TensorDumpRecord entries
 * @return Total bytes to allocate for one DumpBuffer
 */
inline size_t calc_dump_buffer_size(int capacity) {
    return sizeof(DumpBuffer) + static_cast<size_t>(capacity) * sizeof(TensorDumpRecord);
}

/**
 * Calculate per-thread arena size from configuration constants.
 *
 * @return Arena size in bytes per thread
 */
inline uint64_t calc_dump_arena_size() {
    return static_cast<uint64_t>(PLATFORM_DUMP_BUFFERS_PER_THREAD) * PLATFORM_DUMP_RECORDS_PER_BUFFER *
           PLATFORM_DUMP_AVG_TENSOR_BYTES;
}

/**
 * Get DumpSetupHeader pointer from dump base address.
 *
 * @param base_ptr Dump shared memory base address (kernel_args.dump_data_base)
 * @return DumpSetupHeader pointer
 */
inline DumpSetupHeader *get_dump_setup_header(void *base_ptr) { return reinterpret_cast<DumpSetupHeader *>(base_ptr); }

/**
 * Get pointer to the records array of a DumpBuffer.
 *
 * @param buf DumpBuffer pointer
 * @return Pointer to the first TensorDumpRecord
 */
inline TensorDumpRecord *get_dump_buffer_records(DumpBuffer *buf) {
    return reinterpret_cast<TensorDumpRecord *>(buf + 1);
}

#ifdef __cplusplus
}
#endif

#endif  // SRC_A5_PLATFORM_INCLUDE_COMMON_TENSOR_DUMP_H_
