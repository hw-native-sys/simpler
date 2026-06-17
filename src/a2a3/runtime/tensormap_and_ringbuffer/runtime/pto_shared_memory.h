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

#pragma once

#include "utils/device_arena.h"
#include "pto_runtime2_types.h"

struct PTO2SharedMemoryHandle;

struct alignas(64) PTO2RingFlowControl
{
    // === Cache Line 0: Written by Orchestrator, Read by Scheduler ===
    alignas(64) std::atomic<int32_t> current_task_index;  // Task ring head (next to allocate)

    // === Cache Line 1: Written by Scheduler, Read by Orchestrator (for back-pressure) ===
    alignas(64) std::atomic<int32_t> last_task_alive;  // Task ring tail (oldest active task)

    void init()
    {
        current_task_index.store(0, std::memory_order_relaxed);
        last_task_alive.store(0, std::memory_order_relaxed);
    }

    bool validate(PTO2SharedMemoryHandle *handle, int32_t ring_id) const;
};

static_assert(sizeof(PTO2RingFlowControl) == 128, "PTO2RingFlowControl must be exactly 2 cache lines (128B)");

struct alignas(64) PTO2SharedMemoryRingHeader
{
    PTO2RingFlowControl fc;

    // Layout metadata (set once at init)
    uint64_t task_window_size;
    int32_t task_window_mask;
    uint64_t heap_size;
    uint64_t task_descriptors_offset;  // Offset from SM base, in bytes

    // Per-ring data pointers (host-side, set by setup_pointers)
    PTO2TaskDescriptor *task_descriptors;
    PTO2TaskPayload *task_payloads;
    PTO2TaskSlotState *slot_states;

    PTO2TaskDescriptor &get_task_by_slot(int32_t slot)
    {
        return task_descriptors[slot];
    }

    PTO2TaskDescriptor &get_task_by_task_id(int32_t local_id)
    {
        return task_descriptors[local_id & task_window_mask];
    }

    PTO2TaskPayload &get_payload_by_slot(int32_t slot)
    {
        return task_payloads[slot];
    }

    PTO2TaskPayload &get_payload_by_task_id(int32_t local_id)
    {
        return task_payloads[local_id & task_window_mask];
    }

    PTO2TaskSlotState &get_slot_state_by_slot(int32_t slot)
    {
        return slot_states[slot];
    }

    PTO2TaskSlotState &get_slot_state_by_task_id(int32_t local_id)
    {
        return slot_states[local_id & task_window_mask];
    }
};

struct alignas(PTO2_ALIGN_SIZE) PTO2SharedMemoryHeader
{
    // === PER-RING FLOW CONTROL + LAYOUT INFO (set once at init) ===
    PTO2SharedMemoryRingHeader rings[PTO2_MAX_RING_DEPTH];

    // === GLOBAL FIELDS ===
    std::atomic<int32_t> orchestrator_done;  // Flag: orchestration complete

    // Total shared memory size (for validation)
    uint64_t total_size;

    // Graph output for copy-back (set by orchestrator when using packed buffer)
    // Host finalize copies from this address instead of dev_ptr when non-zero
    std::atomic<uint64_t> graph_output_ptr;   // Address where final output was written (packed buffer)
    std::atomic<uint64_t> graph_output_size;  // Size in bytes

    // === ERROR REPORTING ===

    // Orchestrator fatal error code (Orchestrator → Scheduler, AICPU → Host)
    // Non-zero signals fatal error. Written by orchestrator, read by scheduler and host.
    std::atomic<int32_t> orch_error_code;

    // Scheduler error state (Scheduler → Host, independent of orchestrator)
    // Written by scheduler threads on timeout; read by orchestrator and host.
    std::atomic<uint32_t> sched_error_bitmap;  // Bit X set = thread X had error
    std::atomic<int32_t> sched_error_code;     // Last scheduler error code (last-writer-wins)
    std::atomic<int32_t> sched_error_thread;   // Thread index of last error writer
};

static_assert((sizeof(PTO2SharedMemoryHeader) % PTO2_ALIGN_SIZE == 0) && (sizeof(PTO2SharedMemoryHeader) < 4096), "PTO2SharedMemoryHeader should be reasonably sized");

struct PTO2SharedMemoryHandle
{
    void *sm_base;     // Base address of shared memory
    uint64_t sm_size;  // Total size of shared memory

    PTO2SharedMemoryHeader *header;

    // Ownership flag
    bool is_owner;  // True if this handle allocated the memory

    // === Static helpers ===

    static uint64_t calculate_size(uint64_t task_window_size)
    {
        uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH];
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) task_window_sizes[r] = task_window_size;
        return calculate_size_per_ring(task_window_sizes);
    }
    static uint64_t calculate_size_per_ring(const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH])
    {
        uint64_t size = 0;

        // Header (aligned to cache line)
        size += PTO2_ALIGN_UP(sizeof(PTO2SharedMemoryHeader), PTO2_ALIGN_SIZE);

        // Per-ring task descriptors and payloads
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++)
        {
            size += PTO2_ALIGN_UP(task_window_sizes[r] * sizeof(PTO2TaskDescriptor), PTO2_ALIGN_SIZE);
            size += PTO2_ALIGN_UP(task_window_sizes[r] * sizeof(PTO2TaskPayload), PTO2_ALIGN_SIZE);
            size += PTO2_ALIGN_UP(task_window_sizes[r] * sizeof(PTO2TaskSlotState), PTO2_ALIGN_SIZE);
        }

        return size;
    }

    static PTO2SharedMemoryHandle *create_and_init_default(DeviceArena &arena)
    {
        const uint64_t buffer_size = calculate_size(PTO2_TASK_WINDOW_SIZE);
        const size_t off_handle = arena.reserve(sizeof(PTO2SharedMemoryHandle), alignof(PTO2SharedMemoryHandle));
        const size_t off_buffer = arena.reserve(static_cast<size_t>(buffer_size), PTO2_ALIGN_SIZE);
        if (arena.commit() == nullptr) return nullptr;

        auto *handle = static_cast<PTO2SharedMemoryHandle *>(arena.region_ptr(off_handle));
        memset(handle, 0, sizeof(*handle));
        void *buffer = arena.region_ptr(off_buffer);
        memset(buffer, 0, static_cast<size_t>(buffer_size));
        if (!handle->init(buffer, buffer_size, PTO2_TASK_WINDOW_SIZE, PTO2_HEAP_SIZE)) return nullptr;
        return handle;
    }

    // === Instance methods ===

    bool init(void *sm_base_arg, uint64_t sm_size_arg, uint64_t task_window_size, uint64_t heap_size)
    {
        if (!sm_base_arg || sm_size_arg == 0) return false;
        if (sm_size_arg < calculate_size(task_window_size)) return false;

        sm_base = sm_base_arg;
        sm_size = sm_size_arg;
        is_owner = false;
        setup_pointers(task_window_size);
        init_header(task_window_size, heap_size);
        return true;
    }

    void destroy()
    {
        // Arena-owned wrappers (is_owner == false) are reclaimed by arena.release();
        // calling destroy on them is a no-op so existing callers stay safe.
        if (is_owner && sm_base)
        {
            free(sm_base);
            free(this);
        }
    }
    void print_layout()
    {
        if (!header) return;

        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++)
        {}
    }
    bool validate()
    {
        if (!sm_base) return false;
        if (!header) return false;

        PTO2SharedMemoryHeader *h = header;

        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++)
            if (!h->rings[r].fc.validate(this, r)) return false;

        return true;
    }

private:
    void init_header(uint64_t task_window_size, uint64_t heap_size)
    {
        uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH];
        uint64_t heap_sizes[PTO2_MAX_RING_DEPTH];
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++)
        {
            task_window_sizes[r] = task_window_size;
            heap_sizes[r] = heap_size;
        }
        init_header_per_ring(task_window_sizes, heap_sizes);
    }
    void init_header_per_ring(const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH], const uint64_t heap_sizes[PTO2_MAX_RING_DEPTH])
    {
        // Per-ring flow control (start at 0)
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) header->rings[r].fc.init();

        header->orchestrator_done.store(0, std::memory_order_relaxed);

        // Per-ring layout info
        uint64_t offset = PTO2_ALIGN_UP(sizeof(PTO2SharedMemoryHeader), PTO2_ALIGN_SIZE);
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++)
        {
            header->rings[r].task_window_size = task_window_sizes[r];
            header->rings[r].task_window_mask = static_cast<int32_t>(task_window_sizes[r] - 1);
            header->rings[r].heap_size = heap_sizes[r];
            header->rings[r].task_descriptors_offset = offset;
            offset += PTO2_ALIGN_UP(task_window_sizes[r] * sizeof(PTO2TaskDescriptor), PTO2_ALIGN_SIZE);
            offset += PTO2_ALIGN_UP(task_window_sizes[r] * sizeof(PTO2TaskPayload), PTO2_ALIGN_SIZE);
            offset += PTO2_ALIGN_UP(task_window_sizes[r] * sizeof(PTO2TaskSlotState), PTO2_ALIGN_SIZE);
        }

        header->total_size = sm_size;
        header->graph_output_ptr.store(0, std::memory_order_relaxed);
        header->graph_output_size.store(0, std::memory_order_relaxed);

        // Error reporting
        header->orch_error_code.store(PTO2_ERROR_NONE, std::memory_order_relaxed);
        header->sched_error_bitmap.store(0, std::memory_order_relaxed);
        header->sched_error_code.store(PTO2_ERROR_NONE, std::memory_order_relaxed);
        header->sched_error_thread.store(-1, std::memory_order_relaxed);

        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++)
        {
            auto &ring = header->rings[r];
            for (uint64_t i = 0; i < task_window_sizes[r]; i++)
            {
                ring.slot_states[i].bind_ring(static_cast<uint8_t>(r));
                ring.slot_states[i].reset_for_reuse();
                ring.slot_states[i].active_mask = ActiveMask{};
            }
        }
    }
    void setup_pointers(uint64_t task_window_size)
    {
        uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH];
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) task_window_sizes[r] = task_window_size;
        setup_pointers_per_ring(task_window_sizes);
    }
    void setup_pointers_per_ring(const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH])
    {
        char *ptr = (char *)sm_base;

        // Header
        header = (PTO2SharedMemoryHeader *)ptr;
        ptr += PTO2_ALIGN_UP(sizeof(PTO2SharedMemoryHeader), PTO2_ALIGN_SIZE);

        // Per-ring task descriptors, payloads, and slot states
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++)
        {
            auto &ring = header->rings[r];
            ring.task_descriptors = (PTO2TaskDescriptor *)ptr;
            ptr += PTO2_ALIGN_UP(task_window_sizes[r] * sizeof(PTO2TaskDescriptor), PTO2_ALIGN_SIZE);

            ring.task_payloads = (PTO2TaskPayload *)ptr;
            ptr += PTO2_ALIGN_UP(task_window_sizes[r] * sizeof(PTO2TaskPayload), PTO2_ALIGN_SIZE);

            ring.slot_states = (PTO2TaskSlotState *)ptr;
            ptr += PTO2_ALIGN_UP(task_window_sizes[r] * sizeof(PTO2TaskSlotState), PTO2_ALIGN_SIZE);
        }
    }
};

namespace pto2_sm_layout {

inline std::atomic<int32_t> *orch_error_code_addr(void *sm_dev_base) noexcept
{
    return reinterpret_cast<std::atomic<int32_t> *>(static_cast<char *>(sm_dev_base) + offsetof(PTO2SharedMemoryHeader, orch_error_code));
}

inline PTO2SharedMemoryRingHeader *ring_header_addr(void *sm_dev_base, int ring_id) noexcept
{
    return reinterpret_cast<PTO2SharedMemoryRingHeader *>(static_cast<char *>(sm_dev_base) + offsetof(PTO2SharedMemoryHeader, rings) + static_cast<size_t>(ring_id) * sizeof(PTO2SharedMemoryRingHeader));
}

inline std::atomic<int32_t> *ring_current_task_index_addr(void *sm_dev_base, int ring_id) noexcept
{
    return reinterpret_cast<std::atomic<int32_t> *>(reinterpret_cast<char *>(ring_header_addr(sm_dev_base, ring_id)) + offsetof(PTO2SharedMemoryRingHeader, fc) + offsetof(PTO2RingFlowControl, current_task_index));
}

inline std::atomic<int32_t> *ring_last_task_alive_addr(void *sm_dev_base, int ring_id) noexcept
{
    return reinterpret_cast<std::atomic<int32_t> *>(reinterpret_cast<char *>(ring_header_addr(sm_dev_base, ring_id)) + offsetof(PTO2SharedMemoryRingHeader, fc) + offsetof(PTO2RingFlowControl, last_task_alive));
}

inline PTO2TaskDescriptor *ring_task_descriptors_addr(void *sm_dev_base, const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH], int ring_id) noexcept
{
    assert(ring_id >= 0 && ring_id < PTO2_MAX_RING_DEPTH && "pto2_sm_layout: ring_id out of range");
    char *p = static_cast<char *>(sm_dev_base);
    p += PTO2_ALIGN_UP(sizeof(PTO2SharedMemoryHeader), PTO2_ALIGN_SIZE);
    for (int r = 0; r < ring_id; r++)
    {
        p += PTO2_ALIGN_UP(task_window_sizes[r] * sizeof(PTO2TaskDescriptor), PTO2_ALIGN_SIZE);
        p += PTO2_ALIGN_UP(task_window_sizes[r] * sizeof(PTO2TaskPayload), PTO2_ALIGN_SIZE);
        p += PTO2_ALIGN_UP(task_window_sizes[r] * sizeof(PTO2TaskSlotState), PTO2_ALIGN_SIZE);
    }
    return reinterpret_cast<PTO2TaskDescriptor *>(p);
}

}  // namespace pto2_sm_layout
