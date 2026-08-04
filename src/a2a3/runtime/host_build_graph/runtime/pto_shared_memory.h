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
#include <array>

#include "common/platform_config.h"
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
    // === Cache Line 0: Written by Orchestrator, Read by Scheduler ===
    alignas(64) std::atomic<int32_t> current_task_index;  // Task ring head (next to allocate)

    // === Cache Line 1: Written by Scheduler, Read by Orchestrator (for back-pressure) ===
    alignas(64) std::atomic<int32_t> last_task_alive;  // Task ring tail (oldest active task)

    // Per-boot SM reset. PTO2TaskAllocator::init() seeds its private
    // local_task_id_ from initial_local_task_id (default 0 in production)
    // *without* dereferencing current_task_index — it relies on this reset
    // running on every AICPU boot so 0 stays in sync. If you ever change
    // the initial fc value or the boot ordering, update the default in
    // PTO2TaskAllocator::init (pto_ring_buffer.h) in the same change, or
    // submit IDs will be off by the divergence.
    void init() {
        current_task_index.store(0, std::memory_order_relaxed);
        last_task_alive.store(0, std::memory_order_relaxed);
    }

    bool validate(PTO2SharedMemoryHandle *handle, int32_t ring_id) const;
};

static_assert(sizeof(PTO2RingFlowControl) == 128, "PTO2RingFlowControl must be exactly 2 cache lines (128B)");

/**
 * Per-ring shared memory header section.
 *
 * Groups flow-control, layout info, and per-ring data pointers for a single ring.
 * Pointers are host-side only (set by setup_pointers, invalid on device).
 */
struct alignas(64) PTO2SharedMemoryRingHeader {
    PTO2RingFlowControl fc;

    // Lowest task_id not yet guaranteed complete: every task with id in
    // [0, completed_watermark) has its completion_flags entry set. Every completer
    // (device on_mixed_task_complete, or the host orchestrator for an inline-completed
    // hidden-alloc task) calls update_completed_watermark exactly once, right after its
    // own flag is actually set — never before. When on_mixed_task_complete's
    // try_set_completion_flag fails, the call moves to retry_set_completion_flags, which
    // makes it once the retried try_set_completion_flag succeeds instead (docs/
    // RUNTIME_LOGIC.md §8.4); only the one landing exactly at the current watermark walks
    // it forward over the contiguous completed prefix. The host consumer-wait gates on it: a producer
    // slot P's consumers have all retired once completed_watermark >
    // P.last_consumer_local_id. On its own cache line (concurrent CAS-advance by
    // completing threads).
    alignas(64) std::atomic<int32_t> completed_watermark;

    struct alignas(64) AlignedInt32 {
        int32_t val_{0u};
        int8_t pad_[64 - sizeof(int32_t)];
    };

    // Embedded by value, not behind a unique_ptr: this header is placed by
    // memset over raw arena/shared memory (PTO2SharedMemoryHandle::init_header,
    // create_and_init_default), never via a constructor call, so a smart-pointer
    // member's default initializer would never run and would dereference null.
    alignas(64) mutable std::array<AlignedInt32, PLATFORM_MAX_AICPU_THREADS + 1> cached_completed_watermark;

    // Layout metadata (set once at init)
    alignas(64) uint64_t task_window_size;
    int32_t task_window_mask;
    static constexpr int32_t shuffle_lower_bits{4};  // At least 0 and at most log_2(task_window_size)
    int32_t shuffle_higher_bits;                     // log_2(task_window_size) - shuffle_lower_bits
    uint64_t heap_size;
    uint64_t task_descriptors_offset;  // Offset from SM base, in bytes

    // Per-ring data pointers (host-side, set by setup_pointers)
    PTO2TaskDescriptor *task_descriptors;
    PTO2TaskPayload *task_payloads;
    PTO2TaskSlotState *slot_states;

    // Polling-completion state (device-addressed array, one int32 per slot,
    // task_window_size slots total). -1 = pending, local_id = task fully
    // COMPLETED. -1 (not 0) is the pending sentinel because local_id 0 is a
    // valid completion stamp (the first task ever submitted) and must stay
    // distinguishable from "not yet completed". Writer = the task's completer
    // at on_mixed_task_complete; reader = consumer fanin polling
    // (is_completion_flag_set). Set to -1 host-side at init. Indexed by
    // flag_index(local_id), a bit-reindexing of local_id & task_window_mask
    // (same task_window_size-entry period, different physical slot — see
    // flag_index), so the entry for local_id is reused by local_id +
    // task_window_size once the run submits more tasks than the array has
    // slots for — see set_completion_flag / try_set_completion_flag for the
    // ordering this reuse requires.
    std::atomic<int32_t> *completion_flags;

    // To have subsequent tasks on different cachelines one needs shuffle_higher_bits >= 4 (because 64 / sizeof(int32_t)
    // = 2^4) and shuffle_lower_bits >= 1. Every 2^shuffle_lower_bits will typically be on the same cacheline which is
    // good for update_completion_watermark
    constexpr int32_t flag_index(const int32_t local_id) const {
        int32_t low_bits = local_id & ((static_cast<int32_t>(1) << shuffle_lower_bits) - static_cast<int32_t>(1));
        low_bits <<= shuffle_higher_bits;

        int32_t high_bits = local_id & task_window_mask;
        high_bits >>= shuffle_lower_bits;

        return low_bits | high_bits;
    }

    // Once local_id falls behind completed_watermark it is reported set for
    // the rest of the run, even after a later lap overwrites its slot: the
    // watermark fallback is itself monotonic and outlives the raw slot value.
    bool is_completion_flag_set(
        const int32_t thread_idx, const int32_t local_id, std::memory_order order = std::memory_order_acquire
    ) const {
        int32_t &cached_cw = cached_completed_watermark[thread_idx].val_;
        return (local_id < cached_cw) || (completion_flags[flag_index(local_id)].load(order) == local_id) ||
               (local_id < (cached_cw = completed_watermark.load(std::memory_order_acquire)));
    }

    // local_id and local_id - task_window_size share a slot (flag_index wraps
    // modulo task_window_size). earlier_task is one past that slot's previous
    // occupant, so the wait below blocks the store until completed_watermark
    // certifies the previous occupant complete: task t must complete before
    // task t + task_window_size may publish its own entry into the shared
    // slot. This bounds how far a slot can be reused ahead of retirement, not
    // how many tasks a run may submit in total.
    // Must be followed by a call to update_completed_watermark with same thread_idx and local_id (logical requirement)
    // Only the host orchestrator's inline hidden-alloc completion path (submit_task,
    // §7.2) still calls this blocking form; the device scheduler uses
    // try_set_completion_flag below instead.
    void set_completion_flag(
        const int32_t thread_idx, const int32_t local_id, std::memory_order order = std::memory_order_release
    ) const {
        const int32_t earlier_task = local_id - task_window_mask;
        int32_t &cached_cw = cached_completed_watermark[thread_idx].val_;
        while ((cached_cw < earlier_task) &&
               ((cached_cw = completed_watermark.load(std::memory_order_acquire)) < earlier_task)) {
            SPIN_WAIT_HINT();
        }
        completion_flags[flag_index(local_id)].store(local_id, order);
    }

    // Non-blocking counterpart to set_completion_flag: same slot-reuse condition
    // (the previous occupant of local_id's aliased slot must be certified by
    // completed_watermark first), but returns false instead of spinning when it
    // is not yet certified, leaving completion_flags untouched. The device
    // scheduler defers a false return to failed_heap_of_set_completion_flag and
    // retries later rather than blocking the dispatch loop — see
    // docs/RUNTIME_LOGIC.md "Flag-slot reuse" (§8.2) for the relaxed
    // correctness argument this relies on. Must be followed by a call to
    // update_completed_watermark with the same thread_idx and local_id only when
    // this returns true — on false, the caller must defer that call to whichever
    // retry finally succeeds (never call it against a local_id whose flag isn't
    // set yet, §8.4).
    bool try_set_completion_flag(
        const int32_t thread_idx, const int32_t local_id, std::memory_order order = std::memory_order_release
    ) {
        const int32_t earlier_task = local_id - task_window_mask;
        int32_t &cached_cw = cached_completed_watermark[thread_idx].val_;
        if (cached_cw < earlier_task) {
            cached_cw = completed_watermark.load(std::memory_order_acquire);
            if (cached_cw < earlier_task) {
                return false;
            }
        }
        completion_flags[flag_index(local_id)].store(local_id, order);
        return true;
    }

    // Must be called for `local_id` only after set_completion_flag(thread_idx, local_id):
    // the walk below never checks local_id's own completion_flags entry, it trusts the
    // caller already set it, so calling out of order could advance the watermark past
    // local_id before that write is actually visible.
    void update_completed_watermark(const int32_t thread_idx, const int32_t local_id) {
        // Thread fence required such that setting the flag of task local_id is written to GM before completed_watermark
        // is read
        std::atomic_thread_fence(std::memory_order_seq_cst);

        int32_t curr_watermark = completed_watermark.load(std::memory_order_acquire);
        // No-op unless this call sits exactly at the frontier: every completer calls this
        // eagerly, but only the one whose own local_id equals the current watermark performs
        // the advance walk below. A completer that finishes out of order (local_id ahead of
        // the watermark) defers to whichever thread later completes the frontier task.
        if (curr_watermark != local_id) {
            cached_completed_watermark[thread_idx].val_ = curr_watermark;
            return;
        }

        const int32_t submitted = fc.current_task_index.load(std::memory_order_acquire);

        int32_t next = local_id + 1;
        while (true) {
            while (next < submitted && is_completion_flag_set(thread_idx, next)) {
                ++next;
            }

            if (not completed_watermark.compare_exchange_strong(
                    curr_watermark, next, std::memory_order_acq_rel, std::memory_order_acquire
                )) {
                // In case of failure, we hand off the update responsibility to the thread who succeeded
                cached_completed_watermark[thread_idx].val_ = curr_watermark;
                break;
            }
            // Thread fence required such that completed_watermark is written to GM before the next flag is read
            std::atomic_thread_fence(std::memory_order_seq_cst);
            if (not(next < submitted && is_completion_flag_set(thread_idx, next))) {
                break;
            }
            curr_watermark = next++;
        }
    }

    int32_t get_slot_by_task_id(int32_t local_task_id) { return local_task_id & task_window_mask; }

    PTO2TaskDescriptor &get_task_by_slot(int32_t slot) { return task_descriptors[slot]; }

    PTO2TaskDescriptor &get_task_by_task_id(int32_t local_id) {
        return task_descriptors[get_slot_by_task_id(local_id)];
    }

    PTO2TaskPayload &get_payload_by_slot(int32_t slot) { return task_payloads[slot]; }

    PTO2TaskPayload &get_payload_by_task_id(int32_t local_id) { return task_payloads[get_slot_by_task_id(local_id)]; }

    PTO2TaskSlotState &get_slot_state_by_slot(int32_t slot) { return slot_states[slot]; }

    PTO2TaskSlotState &get_slot_state_by_task_id(int32_t local_id) {
        return slot_states[get_slot_by_task_id(local_id)];
    }
};

static_assert(sizeof(PTO2SharedMemoryRingHeader) == 576, "PTO2SharedMemoryRingHeader layout drift");
static_assert(
    offsetof(PTO2SharedMemoryRingHeader, task_descriptors_offset) == 536,
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

static_assert(sizeof(PTO2SharedMemoryHeader) == 640, "PTO2SharedMemoryHeader layout drift");
static_assert(offsetof(PTO2SharedMemoryHeader, total_size) == 584, "PTO2SharedMemoryHeader total_size layout drift");
static_assert(
    offsetof(PTO2SharedMemoryHeader, orch_error_code) == 592, "PTO2SharedMemoryHeader orch_error_code layout drift"
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
    bool attach_populated(void *sm_base, uint64_t sm_size, const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH]);

    void destroy();
    void print_layout();
    bool validate();

private:
    bool init_header(uint64_t task_window_size, uint64_t heap_size);
    bool init_header_per_ring(
        const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH], const uint64_t heap_sizes[PTO2_MAX_RING_DEPTH]
    );
    void setup_pointers(uint64_t task_window_size);
    void setup_pointers_per_ring(const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH]);
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

inline std::atomic<int32_t> *ring_last_task_alive_addr(void *sm_dev_base) noexcept {
    return reinterpret_cast<std::atomic<int32_t> *>(
        reinterpret_cast<char *>(ring_header_addr(sm_dev_base)) + offsetof(PTO2SharedMemoryRingHeader, fc) +
        offsetof(PTO2RingFlowControl, last_task_alive)
    );
}

// Byte offsets (from the SM base) of the ring's three segments. The layout is:
// header, then descriptors -> payloads -> slot_states, every segment
// PTO2_ALIGN_UP-padded.
struct PTO2RingSegmentOffsets {
    uint64_t descriptors;
    uint64_t payloads;
    uint64_t slot_states;
    uint64_t completion_flags;  // polling-completion flag array (1 int32/slot)
    uint64_t end;               // offset just past completion_flags (total SM size)
};

// Single source of truth for the SM segment layout. Returns offsets (not
// pointers), so it serves BOTH the host-side pointer setup (`setup_pointers`,
// which adds `sm_base`) and the device-address helpers below (which add
// `sm_dev_base`). Adding or reordering a segment is a one-line edit here; every
// consumer follows automatically, so the layout walk can never silently
// disagree across call sites.
inline PTO2RingSegmentOffsets ring_segment_offsets(uint64_t task_window_size) noexcept {
    uint64_t off = PTO2_ALIGN_UP(sizeof(PTO2SharedMemoryHeader), PTO2_ALIGN_SIZE);
    PTO2RingSegmentOffsets o{};
    o.descriptors = off;
    off += PTO2_ALIGN_UP(task_window_size * sizeof(PTO2TaskDescriptor), PTO2_ALIGN_SIZE);
    o.payloads = off;
    off += PTO2_ALIGN_UP(task_window_size * sizeof(PTO2TaskPayload), PTO2_ALIGN_SIZE);
    o.slot_states = off;
    off += PTO2_ALIGN_UP(task_window_size * sizeof(PTO2TaskSlotState), PTO2_ALIGN_SIZE);
    o.completion_flags = off;
    off += PTO2_ALIGN_UP(task_window_size * sizeof(std::atomic<int32_t>), PTO2_ALIGN_SIZE);
    o.end = off;
    return o;
}

// Device address of the task_descriptors array.
inline PTO2TaskDescriptor *ring_task_descriptors_addr(void *sm_dev_base, uint64_t task_window_size) noexcept {
    return reinterpret_cast<PTO2TaskDescriptor *>(
        static_cast<char *>(sm_dev_base) + ring_segment_offsets(task_window_size).descriptors
    );
}

// Device address of the slot_states array (used by the allocator's deadlock
// detector to inspect the head task's state/fanout).
inline PTO2TaskSlotState *ring_slot_states_addr(void *sm_dev_base, uint64_t task_window_size) noexcept {
    return reinterpret_cast<PTO2TaskSlotState *>(
        static_cast<char *>(sm_dev_base) + ring_segment_offsets(task_window_size).slot_states
    );
}

// Device address of the polling completion_flags array.
inline std::atomic<int32_t> *ring_completion_flags_addr(void *sm_dev_base, uint64_t task_window_size) noexcept {
    return reinterpret_cast<std::atomic<int32_t> *>(
        static_cast<char *>(sm_dev_base) + ring_segment_offsets(task_window_size).completion_flags
    );
}

}  // namespace pto2_sm_layout
