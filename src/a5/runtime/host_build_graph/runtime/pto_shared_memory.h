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
 * PTO Runtime2 - Shared Memory Layout
 *
 * Defines the shared memory structure for Orchestrator-Scheduler communication.
 *
 * Memory Layout (single ring):
 *   +---------------------------+
 *   | SharedMemoryHeader        |  (flow control + sync)
 *   +---------------------------+
 *   | TaskDescriptor[]          |
 *   | TaskPayload[]             |
 *   | TaskSlotState[]           |
 *   +---------------------------+
 *
 * Design principles:
 * - Only data needed for Orchestrator<->Scheduler communication is here
 * - TensorMap, scope_stack, ready_queues, dep_pool are in private memory
 * - Flow control via atomic counters/flags (no locks needed for single-word R/W)
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#pragma once

#include <stddef.h>

#include <cstring>

#include "utils/device_arena.h"
#include "pto_runtime2_types.h"

// =============================================================================
// Shared Memory Header
// =============================================================================

struct PTO2SharedMemoryHandle;

/**
 * Per-ring flow control state in shared memory.
 * Written/read by Orchestrator and Scheduler for synchronization.
 */
struct alignas(64) PTO2RingFlowControl {
    // Written by Orchestrator, read by Scheduler. There is no reverse channel:
    // the ring is whole-graph-resident, so the scheduler never reclaims task
    // slots and has nothing to publish back.
    alignas(64) std::atomic<int32_t> current_task_index;  // Task ring head (next to allocate)

    // Per-boot SM reset. PTO2TaskAllocator::init() seeds its private
    // local_task_id_ to 0 *without* dereferencing current_task_index — it
    // relies on this reset running on every AICPU boot so 0 stays in sync. If
    // you ever change the initial fc value or the boot ordering, update
    // PTO2TaskAllocator::init (pto_ring_buffer.h) in the same change, or
    // submit IDs will be off by the divergence.
    void init() { current_task_index.store(0, std::memory_order_relaxed); }

    bool validate(PTO2SharedMemoryHandle *handle, int32_t ring_id) const;
};

static_assert(sizeof(PTO2RingFlowControl) == 64, "PTO2RingFlowControl must be exactly one cache line (64B)");

/**
 * Per-ring shared memory header section.
 *
 * Groups flow-control, layout info, and per-ring data pointers for a single ring.
 * Pointers are host-side only (set by setup_pointers, invalid on device).
 */
struct alignas(64) PTO2SharedMemoryRingHeader {
    PTO2RingFlowControl fc;

    // Highest task_id such that every task with id in [0, completed_watermark]
    // has its completion_flags byte set. Advanced over the full contiguous
    // completed prefix at task-completion time (on_mixed_task_complete). The host
    // consumer-wait gates on it: a producer slot P's consumers have all retired
    // once completed_watermark >= P.last_consumer_local_id. On its own cache line
    // (concurrent CAS-advance by completing threads).
    alignas(64) std::atomic<int32_t> completed_watermark;

    // Layout metadata (set once at init)
    alignas(64) uint64_t task_window_size;
    int32_t task_window_mask;
    uint64_t heap_size;
    uint64_t task_descriptors_offset;  // Offset from SM base, in bytes

    // Per-ring data pointers (host-side, set by setup_pointers)
    PTO2TaskDescriptor *task_descriptors;
    PTO2TaskPayload *task_payloads;
    PTO2TaskSlotState *slot_states;

    // Polling-completion state (device-addressed array, one byte per slot).
    // 0 = pending, 1 = task fully COMPLETED. Writer = the task's completer at
    // on_mixed_task_complete; reader = consumer fanin polling (is_completion_flag_set).
    // Cleared per-slot in orch::prepare_task as each slot is claimed. Indexed by
    // local_id & task_window_mask.
    std::atomic<uint8_t> *completion_flags;

    bool is_completion_flag_set(int32_t local_id, std::memory_order order = std::memory_order_acquire) const {
        return completion_flags[local_id & task_window_mask].load(order) != 0;
    }

    void set_completion_flag(int32_t local_id, std::memory_order order = std::memory_order_release) const {
        completion_flags[local_id & task_window_mask].store(1, order);
    }

    // set completion flag first before updating the watermark (logic requirement)
    void update_completed_watermark() {
        int32_t curr_watermark = completed_watermark.load(std::memory_order_acquire);
        const int32_t submitted = fc.current_task_index.load(std::memory_order_acquire);

        int32_t next = curr_watermark;
        while (true) {
            while (next + 1 < submitted && is_completion_flag_set(next + 1)) {
                ++next;
            }
            if (next == curr_watermark) {
                return;
            }

            if (completed_watermark.compare_exchange_strong(
                    curr_watermark, next, std::memory_order_acq_rel, std::memory_order_acquire
                )) {
                curr_watermark = next;
            } else {
                // The acquire release semantics of the successful CAS guarantee that in the case of failure this thread
                // also synchronises with the thread reporting the completion through the intermediary thread(s).
                next = std::max(next, curr_watermark);
            }
        }
    }

    int32_t get_slot_by_task_id(int32_t local_task_id) { return local_task_id & task_window_mask; }

    PTO2TaskDescriptor &get_task_by_slot(int32_t slot) { return task_descriptors[slot]; }

    PTO2TaskDescriptor &get_task_by_task_id(int32_t local_id) {
        return task_descriptors[get_slot_by_task_id(local_id)];
    }

    PTO2TaskPayload &get_payload_by_slot(int32_t slot) { return *slot_states[slot].payload; }

    PTO2TaskPayload &get_payload_by_task_id(int32_t local_id) {
        return get_payload_by_slot(get_slot_by_task_id(local_id));
    }

    PTO2TaskSlotState &get_slot_state_by_slot(int32_t slot) { return slot_states[slot]; }

    PTO2TaskSlotState &get_slot_state_by_task_id(int32_t local_id) {
        return slot_states[get_slot_by_task_id(local_id)];
    }
};

static_assert(sizeof(PTO2SharedMemoryRingHeader) == 192, "PTO2SharedMemoryRingHeader layout drift");
static_assert(
    offsetof(PTO2SharedMemoryRingHeader, task_descriptors_offset) == 152,
    "PTO2SharedMemoryRingHeader task_descriptors_offset layout drift"
);

/**
 * Shared memory header structure
 *
 * Contains per-ring flow control and global layout information.
 */
struct alignas(PTO2_ALIGN_SIZE) PTO2SharedMemoryHeader {
    // === RING FLOW CONTROL + LAYOUT INFO (single ring, set once at init) ===
    PTO2SharedMemoryRingHeader ring;

    // === GLOBAL FIELDS ===
    std::atomic<int32_t> orchestrator_done;  // Flag: orchestration complete

    // Total shared memory size (for validation)
    uint64_t total_size;

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

static_assert(sizeof(PTO2SharedMemoryHeader) == 256, "PTO2SharedMemoryHeader layout drift");
static_assert(offsetof(PTO2SharedMemoryHeader, total_size) == 200, "PTO2SharedMemoryHeader total_size layout drift");
static_assert(
    offsetof(PTO2SharedMemoryHeader, orch_error_code) == 208, "PTO2SharedMemoryHeader orch_error_code layout drift"
);

// =============================================================================
// Shared Memory Handle
// =============================================================================

/**
 * Handle for shared memory lifecycle management (create/destroy).
 * Runtime components (orchestrator, scheduler) use PTO2SharedMemoryHeader* directly.
 */
struct PTO2SharedMemoryHandle {
    void *sm_base;     // Base address of shared memory
    uint64_t sm_size;  // Total size of shared memory

    PTO2SharedMemoryHeader *header;

    // Ownership flag
    bool is_owner;  // True if this handle allocated the memory

    // === Static helpers ===

    static uint64_t calculate_size(uint64_t task_window_size);
    static uint64_t calculate_size_per_ring(const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH]);

    // UT convenience: reserve wrapper + sm_base on `arena`, commit, and init
    // using default PTO2_TASK_WINDOW_SIZE / PTO2_HEAP_SIZE. Only valid when the
    // arena is otherwise empty (the call performs the single commit). All
    // memory is owned by the arena — caller must not call destroy().
    static PTO2SharedMemoryHandle *create_and_init_default(DeviceArena &arena);

    // === Instance methods ===

    // In-place init for caller-provided wrapper storage (e.g. a region carved
    // out of a DeviceArena). Sets is_owner = false, calls setup_pointers and
    // init_header. Returns false when `sm_size` is too small for the requested
    // `task_window_size`.
    bool init(void *sm_base, uint64_t sm_size, uint64_t task_window_size, uint64_t heap_size);
    bool init_per_ring(
        void *sm_base, uint64_t sm_size, const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH],
        const uint64_t heap_sizes[PTO2_MAX_RING_DEPTH]
    );

    // Attach to an ALREADY-populated shared memory region: point the handle and
    // every ring header's data pointers (descriptors / payloads / slot_states)
    // at `sm_base`, but do NOT reset the flow-control counters / slot states.
    // Used by host_build_graph host-orch, where the host orchestrator populated
    // the SM and H2D'd it; the device must re-point at its own SM base without
    // wiping the contents (unlike init_per_ring, which also resets the header).
    //
    // `live_slots` is the pitch the uploaded arrays were laid out with — the
    // number of slots the host actually submitted, not the ring capacity. It must
    // match what the host used or every segment past the descriptors resolves to
    // the wrong address, so both sides derive it from the same submitted count.
    // The capacity and mask in the header are unchanged, and `local_id & mask`
    // yields `local_id`, which is below `live_slots` for every ring task.
    bool attach_populated(
        void *sm_base, uint64_t sm_size, const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH], uint64_t live_slots,
        uint64_t payload_bytes
    );

    void destroy();
    void print_layout();
    bool validate();

private:
    void init_header(uint64_t task_window_size, uint64_t heap_size);
    void init_header_per_ring(
        const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH], const uint64_t heap_sizes[PTO2_MAX_RING_DEPTH]
    );
    void setup_pointers(uint64_t task_window_size);
    // `pitch` is the slot count the arrays are dimensioned for. init_per_ring
    // passes the ring capacity (the mirror the orchestrator writes into);
    // attach_populated passes the submitted count (the compacted image that
    // shipped).
    void setup_pointers_per_ring(
        const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH], uint64_t pitch, uint64_t payload_bytes
    );
};

// =============================================================================
// SM Device Layout Helpers
// =============================================================================
//
// When the host pre-builds a runtime-arena image, it needs the device-side
// addresses of several SM sub-fields (ring flow-control counters,
// task_descriptors arrays, orch_error_code) so it can wire them into the
// orchestrator / scheduler init_data path without dereferencing the SM —
// the SM lives in device memory and cannot be touched from host.
//
// These helpers compute those addresses by offset arithmetic on the SM
// device base. Pure pointer math, no loads/stores; safe to call from host.
// The same arithmetic happens on AICPU too (via PTO2SharedMemoryHandle's
// own setup_pointers), so values are guaranteed consistent across sides.
namespace pto2_sm_layout {

inline std::atomic<int32_t> *orch_error_code_addr(void *sm_dev_base) noexcept {
    return reinterpret_cast<std::atomic<int32_t> *>(
        static_cast<char *>(sm_dev_base) + offsetof(PTO2SharedMemoryHeader, orch_error_code)
    );
}

inline PTO2SharedMemoryRingHeader *ring_header_addr(void *sm_dev_base) noexcept {
    return reinterpret_cast<PTO2SharedMemoryRingHeader *>(
        static_cast<char *>(sm_dev_base) + offsetof(PTO2SharedMemoryHeader, ring)
    );
}

inline std::atomic<int32_t> *ring_current_task_index_addr(void *sm_dev_base) noexcept {
    return reinterpret_cast<std::atomic<int32_t> *>(
        reinterpret_cast<char *>(ring_header_addr(sm_dev_base)) + offsetof(PTO2SharedMemoryRingHeader, fc) +
        offsetof(PTO2RingFlowControl, current_task_index)
    );
}

// Byte offsets (from the SM base) of the ring's three segments. The layout is:
// header, then descriptors -> payloads -> slot_states, every segment
// PTO2_ALIGN_UP-padded.
//
// Two parameters, and the host mirror and the shipped image differ in both.
//
// The *pitch* is how many slots the arrays are dimensioned for: the mirror uses the
// ring capacity, the image uses the submitted count, which is what makes the four
// live prefixes contiguous and the upload one copy.
//
// The *payload stride* is how far apart consecutive payloads sit. The mirror uses
// sizeof(PTO2TaskPayload), whose tensor array is dimensioned for the widest task the
// API allows. The image uses the widest task in this bind — the array is the last
// field, so anything past that task's entries is read by nobody.
struct PTO2RingSegmentOffsets {
    uint64_t descriptors;
    uint64_t payloads;
    uint64_t slot_states;
    uint64_t completion_flags;  // polling-completion byte array (1 byte/slot)
    uint64_t end;               // offset just past completion_flags (total SM size)
};

// Single source of truth for the SM segment layout. Returns offsets (not
// pointers), so it serves BOTH the host-side pointer setup (`setup_pointers`,
// which adds `sm_base`) and the device-address helpers below (which add
// `sm_dev_base`). Adding or reordering a segment is a one-line edit here; every
// consumer follows automatically, so the layout walk can never silently
// disagree across call sites.
inline PTO2RingSegmentOffsets
ring_segment_offsets_with_payload_bytes(uint64_t task_window_size, uint64_t payload_bytes) noexcept {
    uint64_t off = PTO2_ALIGN_UP(sizeof(PTO2SharedMemoryHeader), PTO2_ALIGN_SIZE);
    PTO2RingSegmentOffsets o{};
    o.descriptors = off;
    off += PTO2_ALIGN_UP(task_window_size * sizeof(PTO2TaskDescriptor), PTO2_ALIGN_SIZE);
    o.payloads = off;
    off += PTO2_ALIGN_UP(payload_bytes, PTO2_ALIGN_SIZE);
    o.slot_states = off;
    off += PTO2_ALIGN_UP(task_window_size * sizeof(PTO2TaskSlotState), PTO2_ALIGN_SIZE);
    o.completion_flags = off;
    off += PTO2_ALIGN_UP(task_window_size * sizeof(std::atomic<uint8_t>), PTO2_ALIGN_SIZE);
    o.end = off;
    return o;
}

inline PTO2RingSegmentOffsets ring_segment_offsets(uint64_t task_window_size) noexcept {
    return ring_segment_offsets_with_payload_bytes(task_window_size, task_window_size * sizeof(PTO2TaskPayload));
}

// The pitch the shipped image uses for a given submitted task count. A bind that
// submits nothing still ships its header and still attaches, and a zero-length
// array has no layout, so the pitch never drops below one slot.
inline uint64_t live_slot_pitch(uint64_t submitted_tasks) noexcept {
    return submitted_tasks == 0 ? 1 : submitted_tasks;
}

// The stride a shipped payload array uses for a given widest task. The tensor array
// is the payload's last field, so a task reads nothing past its own entries and the
// stride need only cover them — but every payload in the image shares one stride, so
// it is the widest task that sets it.
//
// Rounded up to PTO2_ALIGN_SIZE because PTO2TaskPayload's own alignment is 64 and the
// image places consecutive payloads at multiples of the stride.
// Restack the live prefix of every ring segment from the ring-pitched mirror the
// orchestrator wrote into an image pitched to `submitted_tasks`, where the four
// prefixes are contiguous and can travel as one copy.
//
// `out_base` must be PTO2_ALIGN_SIZE-aligned and hold
// `ring_segment_offsets_with_payload_bytes(live_slot_pitch(submitted_tasks),
// payload_bytes).end` bytes. Returns that byte count.
//
// Two things the restack has to fix up, both because the image is not the mirror:
//
//   - the ring header's data pointers name the mirror's arrays, so they leave as
//     null rather than carrying host addresses into device memory (the device
//     resolves them in attach_populated);
//   - a slot state names its payload and descriptor by a delta from its own
//     address, and the restack changed those distances, so each binding is
//     re-taken against the image.
inline uint64_t compact_live_image(
    const char *mirror_base, uint64_t task_window_size, uint64_t submitted_tasks, uint64_t payload_bytes, char *out_base
) noexcept {
    // The mirror is pitched to the capacity and to the type, so a larger live count
    // or a wider stride reads past the segment it is copying from and ships a corrupt
    // image. attach_populated tests the same two bounds on the device side.
    always_assert(submitted_tasks <= task_window_size);
    always_assert(payload_bytes <= task_window_size * sizeof(PTO2TaskPayload));
    const PTO2RingSegmentOffsets from = ring_segment_offsets(task_window_size);
    const PTO2RingSegmentOffsets to =
        ring_segment_offsets_with_payload_bytes(live_slot_pitch(submitted_tasks), payload_bytes);

    // The header and the descriptors offset are pitch-independent, so the header
    // lands where it already was.
    std::memcpy(out_base, mirror_base, to.descriptors);
    auto &out_ring = reinterpret_cast<PTO2SharedMemoryHeader *>(out_base)->ring;
    out_ring.task_descriptors = nullptr;
    out_ring.task_payloads = nullptr;
    out_ring.slot_states = nullptr;
    out_ring.completion_flags = nullptr;

    const uint64_t nt = submitted_tasks;
    std::memcpy(out_base + to.descriptors, mirror_base + from.descriptors, nt * sizeof(PTO2TaskDescriptor));
    // Per payload, because the source and destination strides differ: the mirror is
    // pitched to the type, the image to the widest task in this bind.
    std::memcpy(out_base + to.payloads, mirror_base + from.payloads, payload_bytes);
    std::memcpy(out_base + to.slot_states, mirror_base + from.slot_states, nt * sizeof(PTO2TaskSlotState));
    std::memcpy(out_base + to.completion_flags, mirror_base + from.completion_flags, nt * sizeof(std::atomic<uint8_t>));

    auto *out_slots = reinterpret_cast<PTO2TaskSlotState *>(out_base + to.slot_states);
    auto *out_descriptors = reinterpret_cast<PTO2TaskDescriptor *>(out_base + to.descriptors);
    const auto *source_slots = reinterpret_cast<const PTO2TaskSlotState *>(mirror_base + from.slot_states);
    const char *source_payload_base = mirror_base + from.payloads;
    for (uint64_t i = 0; i < nt; ++i) {
        const char *source_payload = reinterpret_cast<const char *>(source_slots[i].payload.get());
        always_assert(source_payload >= source_payload_base);
        const uint64_t payload_offset = static_cast<uint64_t>(source_payload - source_payload_base);
        always_assert(payload_offset + sizeof(PTO2TaskPayload) <= payload_bytes);
        auto *out_payload = reinterpret_cast<PTO2TaskPayload *>(out_base + to.payloads + payload_offset);
        out_slots[i].bind_buffers(out_payload, &out_descriptors[i]);
    }
    return to.end;
}

}  // namespace pto2_sm_layout
