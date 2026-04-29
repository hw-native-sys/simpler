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
 * @brief Tensor dump data structures for device-to-host tensor collection
 *
 * Independent shared memory region for capturing per-task tensor I/O.
 * Fully decoupled from profiling — uses its own ready queues, buffer states,
 * and memory manager thread.
 *
 * Memory layout (allocated only when enable_dump_tensor=true):
 * ┌─────────────────────────────────────────────────────────────┐
 * │ DumpDataHeader (fixed header)                               │
 * │  - Per-thread ready queues (circular FIFOs)                 │
 * │  - Metadata (num_dump_threads, config)                      │
 * ├─────────────────────────────────────────────────────────────┤
 * │ DumpBufferState[0] (Thread 0)                               │
 * │  - free_queue: SPSC queue of DumpMetaBuffer addresses       │
 * │  - current_buf_ptr, arena_base, arena_write_offset          │
 * ├─────────────────────────────────────────────────────────────┤
 * │ DumpBufferState[1] (Thread 1)                               │
 * ├─────────────────────────────────────────────────────────────┤
 * │ ...                                                         │
 * └─────────────────────────────────────────────────────────────┘
 *
 * Per-thread payload arenas are separate allocations.
 * DumpMetaBuffers are allocated by the host and pushed
 * into per-thread free_queues.
 *
 * A5 specifics: memory access uses host-shadow copies + explicit memcpy
 * instead of halHostRegister/SVM. Data structures and interaction logic
 * are identical to A2A3.
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
// DumpMetaBuffer - Fixed-Size Record Buffer
// =============================================================================

/**
 * Fixed-size dump record buffer.
 * Capacity: PLATFORM_DUMP_RECORDS_PER_BUFFER
 * Allocated by host, pushed into per-thread free_queue.
 */
struct DumpMetaBuffer {
    TensorDumpRecord records[PLATFORM_DUMP_RECORDS_PER_BUFFER];
    volatile uint32_t count;  // Current record count
} __attribute__((aligned(64)));

// =============================================================================
// DumpFreeQueue - SPSC Lock-Free Queue for Free Buffers
// =============================================================================

/**
 * Single Producer Single Consumer (SPSC) lock-free queue.
 * Same layout and semantics as PerfFreeQueue, separate type for decoupling.
 *
 * Producer: Host (DumpMemoryManager thread) pushes recycled/new buffers
 * Consumer: Device (AICPU thread) pops buffers when switching
 */
struct DumpFreeQueue {
    volatile uint64_t buffer_ptrs[PLATFORM_DUMP_SLOT_COUNT];
    volatile uint32_t head;  // Consumer read position (Device increments)
    volatile uint32_t tail;  // Producer write position (Host increments)
    uint32_t pad[22];        // Pad to 128 bytes (40 + 88 = 128)
} __attribute__((aligned(64)));

static_assert(sizeof(DumpFreeQueue) == 128, "DumpFreeQueue must be 128 bytes");

// =============================================================================
// DumpBufferState - Per-Thread Buffer State
// =============================================================================

/**
 * Per-thread buffer management state.
 *
 * Writers:
 * - free_queue.tail: Host writes (pushes new buffers)
 * - free_queue.head: Device writes (pops buffers)
 * - current_buf_ptr: Device writes (after pop), Host reads (for flush/collect)
 * - current_buf_seq: Device writes (monotonic counter)
 * - arena_write_offset: Device writes (monotonic), Host reads (for overwrite detection)
 * - dropped_record_count: Device writes (records lost before host export)
 */
struct DumpBufferState {
    DumpFreeQueue free_queue;                // SPSC queue of free DumpMetaBuffer addresses
    volatile uint64_t current_buf_ptr;       // Current active DumpMetaBuffer (0 = none)
    volatile uint32_t current_buf_seq;       // Sequence number for ordering
    uint32_t pad0;                           // Alignment
    volatile uint64_t arena_base;            // Device pointer to this thread's arena
    volatile uint64_t arena_size;            // Arena size in bytes
    volatile uint64_t arena_write_offset;    // Monotonic write cursor (host computes % arena_size)
    volatile uint32_t dropped_record_count;  // Records dropped before host export
    uint8_t pad1[84];                        // Pad to 256 bytes (172 + 84 = 256)
} __attribute__((aligned(64)));

static_assert(sizeof(DumpBufferState) == 256, "DumpBufferState must be 256 bytes");

// =============================================================================
// DumpReadyQueueEntry - Ready Queue Entry
// =============================================================================

/**
 * When a DumpMetaBuffer is full, AICPU adds this entry to the thread's ready queue.
 * Host memory manager retrieves entries and processes them.
 */
struct DumpReadyQueueEntry {
    uint32_t thread_index;  // Thread index (0 ~ num_dump_threads-1)
    uint32_t pad0;
    uint64_t buffer_ptr;  // Device pointer to the full DumpMetaBuffer
    uint32_t buffer_seq;  // Sequence number for ordering
    uint32_t pad1;
} __attribute__((aligned(32)));

// =============================================================================
// DumpDataHeader - Fixed Header
// =============================================================================

/**
 * Dump data fixed header, located at the start of dump shared memory.
 *
 * Contains:
 * 1. Per-thread ready queues (circular FIFOs) — one per AICPU thread
 * 2. Metadata (thread count, config)
 *
 * Ready queue design mirrors PerfDataHeader but is independent:
 * - Per-thread queues avoid lock contention
 * - Producer: AICPU thread (adds full DumpMetaBuffers)
 * - Consumer: Host DumpMemoryManager thread
 * - Queue empty: head == tail
 * - Queue full: (tail + 1) % capacity == head
 */
struct DumpDataHeader {
    // Per-thread ready queues
    DumpReadyQueueEntry queues[PLATFORM_MAX_AICPU_THREADS][PLATFORM_DUMP_READYQUEUE_SIZE];
    volatile uint32_t queue_heads[PLATFORM_MAX_AICPU_THREADS];  // Host reads (consumer)
    volatile uint32_t queue_tails[PLATFORM_MAX_AICPU_THREADS];  // AICPU writes (producer)

    // Metadata (Host initializes, Device reads)
    uint32_t num_dump_threads;
    uint32_t records_per_buffer;
    uint64_t arena_size_per_thread;
    uint32_t magic;
    uint32_t pad;
} __attribute__((aligned(64)));

// =============================================================================
// TensorDumpInfo - Lightweight Info Struct (passed from runtime to platform API)
// =============================================================================

/**
 * Caller fills this struct from runtime-specific tensor types.
 * Platform layer is agnostic to runtime-specific types (Tensor, PTO2TaskPayload, etc.).
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
 * Calculate total memory size for dump header + buffer states.
 *
 * @param num_dump_threads Number of AICPU scheduling threads
 * @return Total bytes for DumpDataHeader + DumpBufferState array
 */
inline size_t calc_dump_data_size(int num_dump_threads) {
    return sizeof(DumpDataHeader) + num_dump_threads * sizeof(DumpBufferState);
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
 * Get DumpDataHeader pointer.
 *
 * @param base_ptr Dump shared memory base address
 * @return DumpDataHeader pointer
 */
inline DumpDataHeader *get_dump_header(void *base_ptr) { return reinterpret_cast<DumpDataHeader *>(base_ptr); }

/**
 * Get DumpBufferState array start address (after DumpDataHeader).
 *
 * @param base_ptr Dump shared memory base address
 * @return DumpBufferState array pointer
 */
inline DumpBufferState *get_dump_buffer_states(void *base_ptr) {
    return reinterpret_cast<DumpBufferState *>(reinterpret_cast<char *>(base_ptr) + sizeof(DumpDataHeader));
}

/**
 * Get DumpBufferState for specified thread.
 *
 * @param base_ptr Dump shared memory base address
 * @param thread_idx Thread index (0 ~ num_dump_threads-1)
 * @return DumpBufferState pointer
 */
inline DumpBufferState *get_dump_buffer_state(void *base_ptr, int thread_idx) {
    return &get_dump_buffer_states(base_ptr)[thread_idx];
}

#ifdef __cplusplus
}
#endif

#endif  // SRC_A5_PLATFORM_INCLUDE_COMMON_TENSOR_DUMP_H_
