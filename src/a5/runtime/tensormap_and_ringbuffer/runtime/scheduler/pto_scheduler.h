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
 * PTO Runtime2 - Scheduler Interface
 *
 * The Scheduler is responsible for:
 * 1. Maintaining per-resource-shape ready queues
 * 2. Tracking task state (PENDING -> COMPLETED -> CONSUMED)
 * 3. Managing fanin/fanout refcounts for dependency resolution
 * 4. Advancing last_task_alive for heap reclamation
 * 5. Two-stage mixed-task completion (subtask done bits → mixed-task complete)
 *
 * The Scheduler runs on Device AI_CPU and processes:
 * - Task state transitions based on fanin_refcount
 * - Buffer lifecycle based on fanout_refcount
 * - Ring pointer advancement for flow control
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#pragma once

#include <atomic>
#include <cstring>

#include "common/core_type.h"
#include "device_arena.h"
#include "pto_async_wait.h"
#include "pto_constants.h"
#include "pto_ring_buffer.h"
#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"
#include "tensor_arg.h"  // TensorArgType, CORE_MAX_TENSOR_ARGS

// Forward decl: scheduler holds a back-pointer to the orchestrator so the
// wiring thread can reach `orch->tensor_map` / `orch->rings[].fanin_pool`
// without taking a full include cycle.
struct PTO2OrchestratorState;

// Wiring-thread lap counters. Single instance (wiring is single-thread)
// embedded in PTO2SchedulerState (see member `wiring_perf` below) so the
// hot path reaches them via `sched->wiring_perf.*` — no globals. Same
// pattern as SchedL2PerfCounters but kept here (not in scheduler_types.h)
// to avoid pulling platform-specific scheduler config into the orchestrator
// include path.
#if PTO2_WIRING_PROFILING
struct alignas(64) WiringPerfCounters {
    // === wire_task laps (per TaskSubmit event) ===
    uint64_t blob_cycle{0};       // tags/refs expansion + builder re-attach
    uint64_t lookup_cycle{0};     // compute_task_fanin
    uint64_t insert_cycle{0};     // register_task_outputs
    uint64_t fanout_cycle{0};     // lock_fanout + prepend + fanout_count++
    uint64_t ready_cycle{0};      // fanin_count + fanin_refcount + push_ready_routed
    uint64_t task_count{0};       // number of wire_task calls
    // === drain_wiring_queue laps ===
    uint64_t sync_cycle{0};       // tensor_map.sync_tensormap (once per drain)
    uint64_t drain_pop_cycle{0};  // pop + backoff + capacity gate
    uint64_t scope_end_cycle{0};  // on_scope_end branch
    // === wiring_thread_run outer-loop laps ===
    uint64_t advance_ring_cycle{0};  // advance_ring_pointers across all rings

    void reset() { *this = WiringPerfCounters{}; }
};
#endif

// Lifecycle profiling: wire_task / release_fanin_and_check_ready stamp
// fanin_zero_cycles + enter_global_queue_cycles via get_sys_cnt_aicpu().
#include "aicpu/device_time.h"

#if PTO2_SCHED_PROFILING
#define PTO2_SCHED_CYCLE_START() uint64_t _st0 = get_sys_cnt_aicpu(), _st1
#define PTO2_SCHED_CYCLE_LAP(acc)   \
    do {                            \
        _st1 = get_sys_cnt_aicpu(); \
        acc += (_st1 - _st0);       \
        _st0 = _st1;                \
    } while (0)
#endif

// =============================================================================
// Ready Queue (Lock-free bounded MPMC — Vyukov design)
// =============================================================================

/**
 * Per-slot entry: sequence counter for ABA safety + task payload
 */
struct PTO2ReadyQueueSlot {
    std::atomic<int64_t> sequence;
    PTO2TaskSlotState *slot_state;
};

/**
 * Thread-local ready buffer for local-first dispatch optimization.
 *
 * Two buffers per scheduling thread, one per CoreType (AIC=0, AIV=1).
 * Initialized once before the scheduling loop; must be empty at
 * the start of each iteration (verified by always_assert).
 *
 * Phase 1 fills per-CoreType buffers via on_task_complete().
 * The dispatch stage drains them local-first via get_ready_tasks_batch,
 * with any remaining tasks pushed to the global ready queue.
 */
// Number of CoreType values eligible for local dispatch (AIC=0, AIV=1)
static constexpr int PTO2_LOCAL_DISPATCH_TYPE_NUM = 2;

struct PTO2LocalReadyBuffer {
    PTO2TaskSlotState **slot_states = nullptr;
    int count = 0;
    int capacity = 0;

    void reset(PTO2TaskSlotState **buf, int cap) {
        slot_states = buf;
        count = 0;
        capacity = cap;
    }

    bool try_push(PTO2TaskSlotState *s) {
        if (slot_states && count < capacity) {
            slot_states[count++] = s;
            return true;
        }
        return false;
    }

    PTO2TaskSlotState *pop() { return (count > 0) ? slot_states[--count] : nullptr; }
};

/**
 * Lock-free bounded MPMC queue (Dmitry Vyukov design)
 *
 * Key properties:
 * - enqueue_pos and dequeue_pos on separate cache lines (no false sharing)
 * - Per-slot sequence counter prevents ABA problem
 * - Empty queue pop returns immediately (single atomic load, no lock)
 * - CAS contention is split: producers only touch enqueue_pos,
 *   consumers only touch dequeue_pos
 */
struct alignas(64) PTO2ReadyQueue {
    PTO2ReadyQueueSlot *slots;
    uint64_t capacity;
    uint64_t mask;        // capacity - 1
    char _pad0[64 - 24];  // Pad to own cache line

    std::atomic<uint64_t> enqueue_pos;
    char _pad1[64 - sizeof(std::atomic<uint64_t>)];  // Own cache line

    std::atomic<uint64_t> dequeue_pos;
    char _pad2[64 - sizeof(std::atomic<uint64_t>)];  // Own cache line

    uint64_t size() {
        uint64_t e = enqueue_pos.load(std::memory_order_relaxed);
        uint64_t d = dequeue_pos.load(std::memory_order_relaxed);
        return (e >= d) ? (e - d) : 0;
    }

    bool push(PTO2TaskSlotState *slot_state) {
        uint64_t pos;
        PTO2ReadyQueueSlot *slot;
        while (true) {
            pos = enqueue_pos.load(std::memory_order_relaxed);
            slot = &slots[pos & mask];
            int64_t seq = slot->sequence.load(std::memory_order_acquire);
            int64_t diff = seq - static_cast<int64_t>(pos);
            if (diff == 0) {
                if (enqueue_pos.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed
                    )) {
                    break;
                }
            } else if (diff < 0) {
                return false;  // Queue full
            }
        }

        slot->slot_state = slot_state;
        slot->sequence.store(static_cast<int64_t>(pos + 1), std::memory_order_release);
        return true;
    }

    // Batch push: reserve count slots with a single CAS after confirming
    // every target slot is available under the usual Vyukov sequence check.
    void push_batch(PTO2TaskSlotState **items, int count) {
        if (count == 0) return;

        uint64_t pos;
        while (true) {
            pos = enqueue_pos.load(std::memory_order_relaxed);
            bool ready = true;
            for (int i = 0; i < count; i++) {
                PTO2ReadyQueueSlot *slot = &slots[(pos + i) & mask];
                int64_t seq = slot->sequence.load(std::memory_order_acquire);
                int64_t diff = seq - static_cast<int64_t>(pos + i);
                if (diff != 0) {
                    ready = false;
                    break;
                }
            }
            if (!ready) {
                continue;
            }
            if (enqueue_pos.compare_exchange_weak(
                    pos, pos + count, std::memory_order_relaxed, std::memory_order_relaxed
                )) {
                break;
            }
        }

        for (int i = 0; i < count; i++) {
            PTO2ReadyQueueSlot *slot = &slots[(pos + i) & mask];
            slot->slot_state = items[i];
            slot->sequence.store(static_cast<int64_t>(pos + i + 1), std::memory_order_release);
        }
    }

#if PTO2_ORCH_PROFILING || PTO2_SCHED_PROFILING
    bool push(PTO2TaskSlotState *slot_state, uint64_t &atomic_count, uint64_t &wait_cycle) {
        uint64_t pos;
        PTO2ReadyQueueSlot *slot;
        uint64_t t0 = get_sys_cnt_aicpu();
        bool contended = false;
        uint32_t atomic_ops = 0;
        while (true) {
            pos = enqueue_pos.load(std::memory_order_relaxed);
            slot = &slots[pos & mask];
            int64_t seq = slot->sequence.load(std::memory_order_acquire);
            int64_t diff = seq - static_cast<int64_t>(pos);
            atomic_ops += 2;  // enqueue_pos.load + sequence.load
            if (diff == 0) {
                if (enqueue_pos.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed
                    )) {
                    atomic_ops++;  // successful CAS
                    break;
                }
                contended = true;
                atomic_ops++;  // failed CAS
            } else if (diff < 0) {
                return false;  // Queue full
            } else {
                contended = true;  // diff > 0: slot not yet released, spin
            }
        }
        atomic_ops++;  // final sequence.store
        atomic_count += atomic_ops;
        if (contended) {
            wait_cycle += (get_sys_cnt_aicpu() - t0);
        }

        slot->slot_state = slot_state;
        slot->sequence.store(static_cast<int64_t>(pos + 1), std::memory_order_release);
        return true;
    }
#endif

    PTO2TaskSlotState *pop() {
        // Fast-path: skip slot load when queue is clearly empty
        uint64_t d = dequeue_pos.load(std::memory_order_relaxed);
        uint64_t e = enqueue_pos.load(std::memory_order_relaxed);
        if (d >= e) {
            return nullptr;
        }

        uint64_t pos;
        PTO2ReadyQueueSlot *slot;
        while (true) {
            pos = dequeue_pos.load(std::memory_order_relaxed);
            slot = &slots[pos & mask];
            int64_t seq = slot->sequence.load(std::memory_order_acquire);
            int64_t diff = seq - static_cast<int64_t>(pos + 1);
            if (diff == 0) {
                if (dequeue_pos.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed
                    ))
                    break;
            } else if (diff < 0) {
                return nullptr;  // Queue empty
            }
        }

        PTO2TaskSlotState *result = slot->slot_state;
        slot->sequence.store(static_cast<int64_t>(pos + mask + 1), std::memory_order_release);
        return result;
    }

#if PTO2_SCHED_PROFILING
    PTO2TaskSlotState *pop(uint64_t &atomic_count, uint64_t &wait_cycle) {
        // Fast-path: skip slot load when queue is clearly empty
        uint64_t d = dequeue_pos.load(std::memory_order_relaxed);
        uint64_t e = enqueue_pos.load(std::memory_order_relaxed);
        atomic_count += 2;  // dequeue_pos.load + enqueue_pos.load
        if (d >= e) {
            return nullptr;
        }

        uint64_t pos;
        PTO2ReadyQueueSlot *slot;
        uint64_t t0 = get_sys_cnt_aicpu();
        bool contended = false;
        uint32_t atomic_ops = 0;
        while (true) {
            pos = dequeue_pos.load(std::memory_order_relaxed);
            slot = &slots[pos & mask];
            int64_t seq = slot->sequence.load(std::memory_order_acquire);
            int64_t diff = seq - static_cast<int64_t>(pos + 1);
            atomic_ops += 2;  // dequeue_pos.load + sequence.load
            if (diff == 0) {
                if (dequeue_pos.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed
                    )) {
                    atomic_ops++;  // successful CAS
                    break;
                }
                contended = true;
                atomic_ops++;  // failed CAS
            } else if (diff < 0) {
                atomic_count += atomic_ops;
                return nullptr;  // Queue empty
            } else {
                contended = true;
            }
        }
        atomic_ops++;  // final sequence.store
        atomic_count += atomic_ops;
        if (contended) {
            wait_cycle += (get_sys_cnt_aicpu() - t0);
        }

        PTO2TaskSlotState *result = slot->slot_state;
        slot->sequence.store(static_cast<int64_t>(pos + mask + 1), std::memory_order_release);
        return result;
    }
#endif

    // Batch pop: reserve a contiguous run of ready slots with a single CAS.
    // Returns actual number of items popped (may be less than max_count).
    int pop_batch(PTO2TaskSlotState **out, int max_count) {
        uint64_t pos;
        int count;
        while (true) {
            pos = dequeue_pos.load(std::memory_order_relaxed);
            count = 0;
            while (count < max_count) {
                PTO2ReadyQueueSlot *slot = &slots[(pos + count) & mask];
                int64_t seq = slot->sequence.load(std::memory_order_acquire);
                int64_t diff = seq - static_cast<int64_t>(pos + count + 1);
                if (diff == 0) {
                    count++;
                    continue;
                }
                if (diff < 0) {
                    break;
                }
                count = -1;
                break;
            }
            if (count == 0) return 0;
            if (count < 0) continue;
            if (dequeue_pos.compare_exchange_weak(
                    pos, pos + count, std::memory_order_relaxed, std::memory_order_relaxed
                )) {
                break;
            }
        }

        for (int i = 0; i < count; i++) {
            PTO2ReadyQueueSlot *slot = &slots[(pos + i) & mask];
            out[i] = slot->slot_state;
            slot->sequence.store(static_cast<int64_t>(pos + i + mask + 1), std::memory_order_release);
        }
        return count;
    }

#if PTO2_SCHED_PROFILING
    int pop_batch(PTO2TaskSlotState **out, int max_count, uint64_t &atomic_count, uint64_t &wait_cycle) {
        uint64_t pos;
        int count;
        uint64_t t0 = get_sys_cnt_aicpu();
        bool contended = false;
        uint32_t atomic_ops = 0;
        while (true) {
            pos = dequeue_pos.load(std::memory_order_relaxed);
            atomic_ops++;  // dequeue_pos.load
            count = 0;
            while (count < max_count) {
                PTO2ReadyQueueSlot *slot = &slots[(pos + count) & mask];
                int64_t seq = slot->sequence.load(std::memory_order_acquire);
                int64_t diff = seq - static_cast<int64_t>(pos + count + 1);
                atomic_ops++;  // sequence.load
                if (diff == 0) {
                    count++;
                    continue;
                }
                if (diff < 0) {
                    break;
                }
                contended = true;
                count = -1;
                break;
            }
            if (count == 0) {
                atomic_count += atomic_ops;
                return 0;
            }
            if (count < 0) {
                continue;
            }
            if (dequeue_pos.compare_exchange_weak(
                    pos, pos + count, std::memory_order_relaxed, std::memory_order_relaxed
                )) {
                atomic_ops++;  // successful CAS
                break;
            }
            contended = true;
            atomic_ops++;  // failed CAS
        }

        for (int i = 0; i < count; i++) {
            PTO2ReadyQueueSlot *slot = &slots[(pos + i) & mask];
            out[i] = slot->slot_state;
            slot->sequence.store(static_cast<int64_t>(pos + i + mask + 1), std::memory_order_release);
            atomic_ops++;  // sequence.store
        }
        atomic_count += atomic_ops;
        if (contended) {
            wait_cycle += (get_sys_cnt_aicpu() - t0);
        }
        return count;
    }
#endif
};

// Cold-path ready queue operations (defined in pto_scheduler.cpp). Declared
// as non-member so PTO2ReadyQueue stays a POD-like struct with cache-line
// alignment. Storage is owned by the caller-supplied arena.
//   reserve_layout: declare the slots[] region on the arena (must precede commit)
//   init_from_layout: bind slots pointer from arena.region_ptr(off) and
//                     initialize sequence counters
//   destroy: forget the slots pointer (arena owns the buffer)
size_t ready_queue_reserve_layout(DeviceArena &arena, uint64_t capacity);
// Writes everything *except* the arena-internal `slots` pointer field
// (sequences/positions on the slot array, capacity, mask). Uses
// arena.region_ptr(slots_off) only to address the slot array for writes;
// does NOT store the pointer in `queue->slots`. Call
// `ready_queue_wire_arena_pointers` afterwards to set the field itself.
bool ready_queue_init_data_from_layout(PTO2ReadyQueue *queue, DeviceArena &arena, size_t slots_off, uint64_t capacity);
// Stores queue->slots = arena.region_ptr(slots_off). Idempotent.
void ready_queue_wire_arena_pointers(PTO2ReadyQueue *queue, DeviceArena &arena, size_t slots_off);
void ready_queue_destroy(PTO2ReadyQueue *queue);

// =============================================================================
// Wiring queue event (tagged union)
// =============================================================================
//
// The wiring SPSC queue is a tagged-event channel from the orchestrator to
// scheduler thread 0. Two event kinds today:
//
//   TaskSubmit: payload is a PTO2TaskSlotState* (the slot to wire).
//   ScopeEnd:   payload is {scope_begin, scope_count} indexing into
//               orch->scope_tasks[]; the wiring thread iterates this range
//               and calls release_producer on each entry.
//
// Keeping scope_end on the same FIFO as task submits is essential: by the
// time wiring sees ScopeEnd, every TaskSubmit pushed before it has already
// finalized its producer fanout_count++, so release_producer cannot
// prematurely transition a producer to CONSUMED. See
// `.claude/plans/wiring-wiring-tensormap-submit-task-sub-zazzy-moore.md`.

enum class PTO2WiringEventKind : uint32_t {
    TaskSubmit = 0,
    ScopeEnd = 1,
};

// 40-byte tagged event. Payload semantics by kind:
//   TaskSubmit: slot points to a single PTO2TaskSlotState; the trailing
//               tags[] + in_manual_scope hold wire_task's compute_task_fanin /
//               register_task_outputs inputs. slot_count unused.
//   ScopeEnd:   slot_array points at orch->scope_tasks[begin]; slot_count is
//               the number of slot pointers in this scope range. Since orch
//               never rewinds scope_tasks (it is sized to hold a whole run),
//               the pointer stays valid until completed_ is observed.
//               tags[] / in_manual_scope are unused (kept for layout
//               uniformity so push/pop_batch never have to branch on kind).
struct PTO2WiringEvent {
    PTO2WiringEventKind kind;  // 4B  @0
    int32_t slot_count;        // 4B  @4 — ScopeEnd only
    union {
        PTO2TaskSlotState *slot;          // TaskSubmit       @8
        PTO2TaskSlotState **slot_array;   // ScopeEnd         @8
    };  // 8B
    // TaskSubmit-only payload. tags[] mirrors args.tag_data() (uint8_t after
    // the TensorArgType width refactor — the whole 16-slot array fits in 16B,
    // see for_task_submit). Read in wire_task to drive compute_task_fanin.
    TensorArgType tags[CORE_MAX_TENSOR_ARGS];  // 16B @16
    uint8_t in_manual_scope;                   // 1B  @32
    uint8_t _pad[7];                           // pad to 8B alignment (sizeof == 40B)

    static PTO2WiringEvent for_task_submit(
        PTO2TaskSlotState *slot, const TensorArgType *tags_src, int32_t /*tensor_count*/, bool in_manual_scope
    ) {
        PTO2WiringEvent ev;
        ev.kind = PTO2WiringEventKind::TaskSubmit;
        ev.slot_count = 0;
        ev.slot = slot;
        // args.tag_data() always points at a fixed-size CORE_MAX_TENSOR_ARGS
        // backing array (16B with uint8_t). Unconditional 16B memcpy is one
        // sub-cache-line transfer regardless of actual tensor_count — extra
        // bytes past tensor_count are uninitialized garbage in the source but
        // wire_task never reads them.
        memcpy(ev.tags, tags_src, sizeof(ev.tags));
        ev.in_manual_scope = in_manual_scope ? 1u : 0u;
        return ev;
    }

    static PTO2WiringEvent for_scope_end(PTO2TaskSlotState **slot_array, int32_t slot_count) {
        PTO2WiringEvent ev;
        ev.kind = PTO2WiringEventKind::ScopeEnd;
        ev.slot_count = slot_count;
        ev.slot_array = slot_array;
        return ev;
    }
};

static_assert(sizeof(PTO2WiringEvent) == 40, "PTO2WiringEvent must be exactly 40B");
static_assert(offsetof(PTO2WiringEvent, tags) == 16, "tags must start at offset 16");
static_assert(offsetof(PTO2WiringEvent, in_manual_scope) == 32, "in_manual_scope must be at offset 32");

// =============================================================================
// SPSC Queue (Single-Producer Single-Consumer, wait-free)
// =============================================================================
//
// Bounded ring buffer optimized for the wiring queue use case:
//   - Producer: orchestrator thread (push)
//   - Consumer: scheduler thread 0 (pop_batch)
//
// Design based on Rigtorp's cached-index technique: each side caches
// the other's index locally, avoiding cross-core cache line bouncing
// on the hot path. Only when the local cache says "full" or "empty"
// does the thread issue an acquire load on the remote index.
//
// Memory layout: 4 cache-line-aligned fields ensure zero false sharing.

struct alignas(64) PTO2SpscQueue {
    // --- Producer cache lines (orchestrator thread) ---
    alignas(64) std::atomic<uint64_t> head_{0};
    alignas(64) uint64_t tail_cached_{0};

    // --- Consumer cache lines (scheduler thread 0) ---
    alignas(64) std::atomic<uint64_t> tail_{0};
    alignas(64) uint64_t head_cached_{0};

    // --- Shared (immutable after init) ---
    PTO2WiringEvent *buffer_{nullptr};
    uint64_t mask_{0};

    // Reserve the backing buffer region on the supplied arena. Returns the
    // region offset, to be passed to init_from_layout() after the arena is
    // committed. Cache-line aligned: the buffer is shared between the
    // orchestrator (push) and scheduler thread 0 (pop_batch), so its base
    // must not false-share with neighboring regions.
    static size_t reserve_layout(DeviceArena &arena, uint64_t capacity) {
        return arena.reserve(capacity * sizeof(PTO2WiringEvent), PTO2_ALIGN_SIZE);
    }

    // Writes everything except the arena-internal `buffer_` pointer field
    // (zeros the event array, mask/head/tail). The host pre-builds the image
    // without storing a host address in buffer_; the AICPU wires buffer_ at
    // boot via wire_arena_pointers().
    bool init_data_from_layout(DeviceArena &arena, size_t buffer_off, uint64_t capacity) {
        if (capacity == 0 || (capacity & (capacity - 1)) != 0) return false;
        auto *buf = static_cast<PTO2WiringEvent *>(arena.region_ptr(buffer_off));
        // calloc'd-equivalent: zero so spurious early pops observe a benign
        // (TaskSubmit, nullptr) event.
        for (uint64_t i = 0; i < capacity; i++) {
            buf[i] = PTO2WiringEvent{};
        }
        mask_ = capacity - 1;
        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
        tail_cached_ = 0;
        head_cached_ = 0;
        return true;
    }

    // Wire the arena-internal pointer. Called by both host (with host arena)
    // and AICPU (with device arena attached to the prebuilt image).
    void wire_arena_pointers(DeviceArena &arena, size_t buffer_off) {
        buffer_ = static_cast<PTO2WiringEvent *>(arena.region_ptr(buffer_off));
    }

    // Arena owns the buffer; here we only forget our pointer.
    void destroy() { buffer_ = nullptr; }

    // Push one event (producer only). Returns false if queue is full.
    // Full condition: next_h - tail > mask_ (i.e. > capacity-1), so the
    // effective usable capacity is capacity-1 (one slot is wasted as a
    // sentinel to distinguish full from empty). uint64_t wrapping is safe
    // since head and tail are monotonically increasing and subtraction
    // wraps correctly.
    bool push(const PTO2WiringEvent &ev) {
        uint64_t h = head_.load(std::memory_order_relaxed);
        uint64_t next_h = h + 1;
        if (next_h - tail_cached_ > mask_) {
            tail_cached_ = tail_.load(std::memory_order_acquire);
            if (next_h - tail_cached_ > mask_) {
                return false;
            }
        }
        buffer_[h & mask_] = ev;
        head_.store(next_h, std::memory_order_release);
        return true;
    }

    // Pop up to max_count events (consumer only). Returns actual count.
    int pop_batch(PTO2WiringEvent *out, int max_count) {
        uint64_t t = tail_.load(std::memory_order_relaxed);
        uint64_t avail = head_cached_ - t;
        if (avail == 0) {
            head_cached_ = head_.load(std::memory_order_acquire);
            avail = head_cached_ - t;
            if (avail == 0) return 0;
        }
        int count = (avail < static_cast<uint64_t>(max_count)) ? static_cast<int>(avail) : max_count;
        for (int i = 0; i < count; i++) {
            out[i] = buffer_[(t + i) & mask_];
        }
        tail_.store(t + count, std::memory_order_release);
        return count;
    }

    // Approximate size (used for backoff decisions, not exact).
    uint64_t size() const {
        uint64_t h = head_.load(std::memory_order_acquire);
        uint64_t t = tail_.load(std::memory_order_acquire);
        return h - t;
    }
};

static_assert(sizeof(PTO2SpscQueue) == 256, "PTO2SpscQueue must be exactly 4 cache lines (256B)");
// =============================================================================

/**
 * Statistics returned by mixed-task completion processing
 */
struct CompletionStats {
    int32_t fanout_edges;       // Number of fanout edges traversed (notify consumers)
    int32_t tasks_enqueued;     // Number of consumers that became READY
    int32_t fanin_edges;        // Number of fanin edges traversed (release producers)
    bool mixed_task_completed;  // True only when this callback completed a mixed task
};

/**
 * Layout descriptor produced by PTO2SchedulerState::reserve_layout(). Holds
 * the arena offsets of every sub-region the scheduler needs plus the
 * capacities used at layout time (init_from_layout reuses them).
 */
struct PTO2SchedulerLayout {
    size_t off_ready_queue_slots[PTO2_NUM_RESOURCE_SHAPES];
    size_t off_dummy_ready_queue_slots;
    size_t off_dep_pool_entries[PTO2_MAX_RING_DEPTH];
    size_t off_wiring_spsc_buffer;
    uint64_t ready_queue_capacity;
    uint64_t spsc_capacity;
    int32_t dep_pool_capacity;
};

/**
 * Scheduler state structure
 *
 * Contains dynamic state updated during task execution.
 * Separated from shared memory for cache efficiency.
 * Hot-path methods are defined inline (implicitly inline as member functions).
 */
struct PTO2SchedulerState {
    // Shared memory access
    PTO2SharedMemoryHeader *sm_header;

    // Back-pointer to the orchestrator. Wiring reaches `orchestrator->tensor_map`
    // (sole-writer after step 4) and `orchestrator->rings[r].fanin_pool` (spill
    // pool for fanin overflow) through this. Set in pto_runtime2_init alongside
    // the existing `orch->scheduler = …` assignment.
    PTO2OrchestratorState *orchestrator{nullptr};

    // Per-ring state
    struct alignas(64) RingSchedState {
        // --- Cache Line 0: ring pointer (read-only) + hot path (read-write) ---
        PTO2SharedMemoryRingHeader *ring;
        int32_t last_task_alive;
        // Shadow of the most recently published SM `last_task_alive`. Used to
        // throttle SM writes to every K=16 local advances — see sync_to_sm
        // below. Single-writer (wiring thread); no lock needed after step 1/2
        // moved wiring + scope_end onto a dedicated thread.
        int32_t last_published_to_sm{0};

        // --- Cache Line 1+: Thread 0 only (wiring dep_pool) ---
        alignas(64) PTO2DepListPool dep_pool;

        // Initialize arena-internal data + arena-external pointers; does NOT
        // store dep_pool.base (that lives in the runtime arena and is wired
        // by SchedulerState::wire_arena_pointers). The `ring` field stores
        // the device address of the SM ring header — computed via offset
        // arithmetic, no SM dereference.
        bool init_data_from_layout(void *sm_dev_base, int32_t ring_id);
        void destroy();

        // directa-pubonly: K=16 batched publish.
        //
        // Each advance still pushes the local `last_task_alive` counter
        // forward, but the SM cache line is only stored when local has
        // advanced PUBLISH_INTERVAL_K positions past the most recently
        // published value. This cuts cache-invalidate traffic on orch's
        // SM read path by ~K× without changing total work.
        //
        // Safety: for paged_attention_unroll Case1, task_window_size
        // (16384) >> task_count (1024), so orch backpressure on slot
        // reuse never fires. PTO2_TENSORMAP_POOL_SIZE (65536) >> 1280
        // also rules out tensormap cleanup pressure. Workloads that
        // approach those bounds need an explicit fallback (publish even
        // with delta < K when stale long enough). Not present here.
        void sync_to_sm() {
            constexpr int32_t PUBLISH_INTERVAL_K = 16;
            if (last_task_alive - last_published_to_sm < PUBLISH_INTERVAL_K) return;
            ring->fc.last_task_alive.store(last_task_alive, std::memory_order_release);
            last_published_to_sm = last_task_alive;
        }

        void advance_ring_pointers() {
            int32_t current_task_index = ring->fc.current_task_index.load(std::memory_order_acquire);
            int32_t old_last_task_alive = last_task_alive;

            while (last_task_alive < current_task_index) {
                PTO2TaskSlotState &slot_state = ring->get_slot_state_by_task_id(last_task_alive);
                if (slot_state.task_state.load(std::memory_order_acquire) != PTO2_TASK_CONSUMED) {
                    break;
                }
                last_task_alive++;
            }

            // Eager reset: prepare reclaimed slots for reuse while still hot in cache.
            // Safe because last_task_alive has advanced past these slots but
            // sync_to_sm has not yet published — the orchestrator cannot reuse
            // them until the release store below.
            // Skips payload, task, ring_id — immutable after RingSchedState::init().
            for (int32_t id = old_last_task_alive; id < last_task_alive; id++) {
                ring->get_slot_state_by_task_id(id).reset_for_reuse();
            }

            sync_to_sm();
        }
    } ring_sched_states[PTO2_MAX_RING_DEPTH];

    // Ready queues remain global (scheduling is ring-agnostic)
    PTO2ReadyQueue ready_queues[PTO2_NUM_RESOURCE_SHAPES];

    // Dependency-only tasks (active_mask is empty, shape == DUMMY). Drained by
    // the dispatch loop and completed inline -- never goes to AICore.
    PTO2ReadyQueue dummy_ready_queue;

    // Wiring subsystem — groups all wiring-related state for cache-line isolation.
    //
    // Three cache-line regions by writer:
    //   1. batch_*  / backoff — thread 0 exclusive (local batch buffer)
    //   2. queue    — SPSC: orchestrator push, thread 0 pop
    //   3. orch_needs_drain — orchestrator write, thread 0 read
    struct alignas(64) WiringState {
        // Batch buffer holds at most this many wiring events between pops.
        // 16 keeps drain amortization stable; events are 40B (step-4
        // tags-on-event layout with uint8_t TensorArgType) so the batch
        // region is 12 + 4 pad + 16 * 40 = 656B, padded up to 704B (11
        // cache lines) via alignas on `queue`. See size asserts below.
        static constexpr uint64_t BATCH_SIZE = 16;
        static constexpr int BACKOFF_LIMIT = 32;

        // --- Thread 0 exclusive: local batch buffer + backoff ---
        int batch_count = 0;
        int batch_index = 0;
        int backoff_counter = 0;
        PTO2WiringEvent batch[BATCH_SIZE];

        // --- SPSC queue: orchestrator (push) ↔ thread 0 (pop) ---
        alignas(64) PTO2SpscQueue queue;

        // --- Orchestrator write, thread 0 read ---
        // `orch_needs_drain` is poked by orch (rare) to bypass the wiring
        // backoff. `idle` is poked by wiring every drain iteration: true once
        // batch + queue are both empty, false the moment events are popped.
        // wait_for_tensor_ready() spins on (queue.size() == 0 && idle == true)
        // to know it's safe to read tensor_map without racing wiring writes.
        alignas(64) std::atomic<bool> orch_needs_drain{false};
        std::atomic<bool> idle{true};
    } wiring;

    static_assert(
        offsetof(WiringState, queue) == 704,
        "WiringState: batch region must be exactly 11 cache lines before queue"
    );
    static_assert(sizeof(WiringState) == 1024, "WiringState must be exactly 16 cache lines (1024B)");

    alignas(64) AsyncWaitList async_wait_list;

    // Statistics (cold path, isolated from hot-path fields)
#if PTO2_SCHED_PROFILING
    alignas(64) std::atomic<int64_t> tasks_completed;
    std::atomic<int64_t> tasks_consumed;
#endif

#if PTO2_WIRING_PROFILING
    // Wiring-thread profiling counters (single thread → no per-thread array).
    // Read at end-of-run by wiring_thread_run / aicpu_executor for the lap
    // breakdown log lines; reset by `wiring_perf.reset()`.
    alignas(64) WiringPerfCounters wiring_perf;
#endif
    // =========================================================================
    // Inline hot-path methods
    // =========================================================================

    /**
     * Drain wiring queue: pop submitted tasks and wire their fanout edges.
     * Called by scheduler thread 0 each loop iteration. Sets fanin_count,
     * acquires fanout_lock per producer, allocates dep_pool entries, and
     * pushes ready tasks to the appropriate ready queue.
     *
     * Body is in scheduler_wire.cpp (needs full PTO2OrchestratorState
     * definition for tensor_map / fanin_pool access).
     *
     * @return Number of events processed this call.
     */
    int drain_wiring_queue(bool force_drain = false);

    // Route a ready slot to the right global queue. Dummy tasks (empty
    // active_mask) live in dummy_ready_queue; everything else goes to the
    // per-shape ready_queues[]. Used by paths that do not have a thread-local
    // ready buffer (e.g. wiring). See push_ready_routed_local for the
    // dispatch-time fast path.
    void push_ready_routed(PTO2TaskSlotState *slot_state) {
        PTO2ResourceShape shape = slot_state->active_mask.to_shape();
        if (shape == PTO2ResourceShape::DUMMY) {
            dummy_ready_queue.push(slot_state);
        } else {
            ready_queues[static_cast<int32_t>(shape)].push(slot_state);
        }
    }

    /**
     * Wire a single TaskSubmit event:
     *   - sync_tensormap against current SM last_task_alive
     *   - run compute_task_fanin to extend the explicit_dep prefix orch left
     *     in payload.fanin_inline_slot_states[]
     *   - register_task_outputs into the tensor_map
     *   - lock_fanout / prepend dep / fanout_count++ on each producer
     *   - set fanin_count, release wiring sentinel, push to ready queue if
     *     immediately ready
     *
     * Body is in scheduler_wire.cpp (needs full PTO2OrchestratorState).
     */
    void wire_task(RingSchedState &rss, const PTO2WiringEvent &ev);

    void check_and_handle_consumed(PTO2TaskSlotState &slot_state) {
        if (slot_state.fanout_refcount.load(std::memory_order_acquire) != slot_state.fanout_count) return;

        PTO2TaskState expected = PTO2_TASK_COMPLETED;
        if (!slot_state.task_state.compare_exchange_strong(
                expected, PTO2_TASK_CONSUMED, std::memory_order_acq_rel, std::memory_order_acquire
            )) {
            return;
        }

#if PTO2_SCHED_PROFILING
        tasks_consumed.fetch_add(1, std::memory_order_relaxed);
#endif

        // Advance is owned by the wiring thread (single writer); sched only
        // publishes the CONSUMED transition via the CAS above. The wiring
        // thread's main loop polls task_state and walks last_task_alive
        // forward on its next iter.
    }

#if PTO2_ORCH_PROFILING || PTO2_SCHED_PROFILING
    void check_and_handle_consumed(PTO2TaskSlotState &slot_state, uint64_t &atomic_count) {
        int32_t fc = slot_state.fanout_count;
        int32_t rc = slot_state.fanout_refcount.load(std::memory_order_acquire);

        atomic_count += 2;  // fanout_count.load + fanout_refcount.load

        if (rc != fc) return;

        PTO2TaskState expected = PTO2_TASK_COMPLETED;
        if (!slot_state.task_state.compare_exchange_strong(
                expected, PTO2_TASK_CONSUMED, std::memory_order_acq_rel, std::memory_order_acquire
            )) {
            atomic_count += 1;  // failed CAS
            return;
        }

        atomic_count += 1;  // successful CAS

#if PTO2_SCHED_PROFILING
        tasks_consumed.fetch_add(1, std::memory_order_relaxed);
#endif

        // Advance is owned by the wiring thread (single writer); see the
        // non-profiling variant above.
    }
#endif

    void release_producer(PTO2TaskSlotState &slot_state) {
        slot_state.fanout_refcount.fetch_add(1, std::memory_order_acq_rel);
        check_and_handle_consumed(slot_state);
    }

#if PTO2_ORCH_PROFILING || PTO2_SCHED_PROFILING
    void release_producer(PTO2TaskSlotState &slot_state, uint64_t &atomic_count) {
        slot_state.fanout_refcount.fetch_add(1, std::memory_order_acq_rel);
        atomic_count += 1;  // fanout_refcount.fetch_add
        check_and_handle_consumed(slot_state, atomic_count);
    }
#endif

    bool release_fanin_and_check_ready(PTO2TaskSlotState &slot_state, PTO2LocalReadyBuffer *local_bufs = nullptr) {
        // Atomically increment fanin_refcount and check if all producers are done
        // ACQ_REL on fanin_refcount already synchronizes with the orchestrator's
        // init release, making fanin_count visible — plain load suffices.
        int32_t new_refcount = slot_state.fanin_refcount.fetch_add(1, std::memory_order_acq_rel) + 1;

        if (new_refcount == slot_state.fanin_count) {
            // Last producer brought fanin to 0 → task is logically ready.
#if PTO2_SCHED_PROFILING
            uint64_t now = get_sys_cnt_aicpu();
            slot_state.fanin_zero_cycles = now;
#endif
            // Local-first: try per-CoreType thread-local buffer before global queue
            // Route by active_mask: AIC-containing tasks → buf[0], AIV-only → buf[1]
            // DUMMY shape is out of range for local_bufs (sized PTO2_NUM_RESOURCE_SHAPES);
            // dummy slots bypass the local fast path and go straight to dummy_ready_queue.
            PTO2ResourceShape shape = slot_state.active_mask.to_shape();
            if (shape == PTO2ResourceShape::DUMMY) {
                dummy_ready_queue.push(&slot_state);
#if PTO2_SCHED_PROFILING
                slot_state.enter_global_queue_cycles = now;
#endif
            } else if (!local_bufs || !local_bufs[static_cast<int32_t>(shape)].try_push(&slot_state)) {
                ready_queues[static_cast<int32_t>(shape)].push(&slot_state);
#if PTO2_SCHED_PROFILING
                slot_state.enter_global_queue_cycles = now;
#endif
            }
            // else: pushed to sched-local buf; enter_global_queue_cycles stays 0
            // (set later by flush_local_bufs if it spills to global, or remains 0
            // if same sched consumes it directly).
            return true;
        }
        return false;
    }

#if PTO2_ORCH_PROFILING || PTO2_SCHED_PROFILING
    bool release_fanin_and_check_ready(
        PTO2TaskSlotState &slot_state, uint64_t &atomic_count, uint64_t &push_wait,
        PTO2LocalReadyBuffer *local_bufs = nullptr
    ) {
        int32_t new_refcount = slot_state.fanin_refcount.fetch_add(1, std::memory_order_acq_rel) + 1;
        atomic_count += 1;  // fanin_refcount.fetch_add

        if (new_refcount == slot_state.fanin_count) {
#if PTO2_SCHED_PROFILING
            uint64_t now = get_sys_cnt_aicpu();
            slot_state.fanin_zero_cycles = now;
#endif
            // Local-first: try per-CoreType thread-local buffer before global queue.
            // Dummy slots bypass local_bufs (out-of-range for PTO2_NUM_RESOURCE_SHAPES)
            // and go straight to dummy_ready_queue; use the profiling-aware push so
            // atomic_count / push_wait stay consistent with the non-dummy path.
            PTO2ResourceShape shape = slot_state.active_mask.to_shape();
            if (shape == PTO2ResourceShape::DUMMY) {
                dummy_ready_queue.push(&slot_state, atomic_count, push_wait);
#if PTO2_SCHED_PROFILING
                slot_state.enter_global_queue_cycles = now;
#endif
            } else if (!local_bufs || !local_bufs[static_cast<int32_t>(shape)].try_push(&slot_state)) {
                ready_queues[static_cast<int32_t>(shape)].push(&slot_state, atomic_count, push_wait);
#if PTO2_SCHED_PROFILING
                slot_state.enter_global_queue_cycles = now;
#endif
            }
            return true;
        }
        return false;
    }
#endif

    int get_ready_tasks_batch(
        PTO2ResourceShape shape, PTO2LocalReadyBuffer &local_buf, PTO2TaskSlotState **out, int max_count
    ) {
        int count = 0;
        while (count < max_count && local_buf.count > 0) {
            out[count++] = local_buf.slot_states[--local_buf.count];
        }
        int remaining = max_count - count;
        if (remaining > 0) {
            count += ready_queues[static_cast<int32_t>(shape)].pop_batch(out + count, remaining);
        }
        return count;
    }

#if PTO2_SCHED_PROFILING
    int get_ready_tasks_batch(
        PTO2ResourceShape shape, PTO2LocalReadyBuffer &local_buf, PTO2TaskSlotState **out, int max_count,
        uint64_t &atomic_count, uint64_t &wait_cycle, uint64_t &local_dispatch_count
    ) {
        int count = 0;
        while (count < max_count && local_buf.count > 0) {
            local_dispatch_count++;
            out[count++] = local_buf.slot_states[--local_buf.count];
        }
        int remaining = max_count - count;
        if (remaining > 0) {
            count +=
                ready_queues[static_cast<int32_t>(shape)].pop_batch(out + count, remaining, atomic_count, wait_cycle);
        }
        return count;
    }
#endif

    void on_scope_end(PTO2TaskSlotState **task_slot_states, int32_t count) {
#if PTO2_ORCH_PROFILING
        extern uint64_t g_orch_scope_end_atomic_count;
        if (count > 0) __builtin_prefetch(task_slot_states[0], 1, 0);
        for (int32_t i = 0; i < count; i++) {
            if (i + 1 < count) __builtin_prefetch(task_slot_states[i + 1], 1, 0);
            release_producer(*task_slot_states[i], g_orch_scope_end_atomic_count);
        }
#else
        if (count > 0) __builtin_prefetch(task_slot_states[0], 1, 0);
        for (int32_t i = 0; i < count; i++) {
            if (i + 1 < count) __builtin_prefetch(task_slot_states[i + 1], 1, 0);
            release_producer(*task_slot_states[i]);
        }
#endif
    }

    /**
     * Subtask completion: atomic counter model.
     * Called when a single subtask (AIC, AIV0, or AIV1) finishes on any block.
     * Atomically increments completed_subtasks and checks whether all subtasks
     * across all blocks are done.
     *
     * @return true if this was the last subtask, completing the entire task.
     */
    bool on_subtask_complete(PTO2TaskSlotState &slot_state) {
        int16_t prev = slot_state.completed_subtasks.fetch_add(1, std::memory_order_acq_rel);
        return (prev + 1) == slot_state.total_required_subtasks;
    }

    /**
     * Two-stage completion: second stage.
     * Called exactly once when all subtasks of a mixed task are done
     * (i.e., on_subtask_complete returned true).
     * Handles fanout notification, fanin release, and self-consumption check.
     */
#if PTO2_SCHED_PROFILING
    CompletionStats
#else
    void
#endif
    on_mixed_task_complete(
        PTO2TaskSlotState &slot_state,
#if PTO2_SCHED_PROFILING
        int thread_idx,
#endif

        PTO2LocalReadyBuffer *local_bufs = nullptr
    ) {
#if PTO2_SCHED_PROFILING
        CompletionStats stats = {0, 0, 0, true};
#endif
#if PTO2_SCHED_PROFILING
        extern uint64_t g_sched_lock_cycle[], g_sched_fanout_cycle[];
        extern uint64_t g_sched_lock_atomic_count[], g_sched_lock_wait_cycle[];
        extern uint64_t g_sched_fanout_atomic_count[], g_sched_push_wait_cycle[];
        uint64_t lock_atomics = 0, lock_wait = 0;
        PTO2_SCHED_CYCLE_START();
#endif

#if PTO2_SCHED_PROFILING
        slot_state.lock_fanout(lock_atomics, lock_wait);
#else
        slot_state.lock_fanout();
#endif
        slot_state.task_state.store(PTO2_TASK_COMPLETED, std::memory_order_release);
        PTO2DepListEntry *current = slot_state.fanout_head;  // Protected by fanout_lock
        slot_state.unlock_fanout();

#if PTO2_SCHED_PROFILING
        lock_atomics += 2;  // state.store + unlock.store
        g_sched_lock_atomic_count[thread_idx] += lock_atomics;
        g_sched_lock_wait_cycle[thread_idx] += lock_wait;
        PTO2_SCHED_CYCLE_LAP(g_sched_lock_cycle[thread_idx]);
#endif

        // Fanout: notify consumers
#if PTO2_SCHED_PROFILING
        uint64_t fanout_atomics = 0, push_wait = 0;
#endif
        while (current != nullptr) {
            PTO2TaskSlotState &consumer_slot = *current->slot_state;
#if PTO2_SCHED_PROFILING
            stats.fanout_edges++;
            if (release_fanin_and_check_ready(consumer_slot, fanout_atomics, push_wait, local_bufs)) {
                stats.tasks_enqueued++;
            }
#else
            release_fanin_and_check_ready(consumer_slot, local_bufs);
#endif
            current = current->next;
        }

#if PTO2_SCHED_PROFILING
        g_sched_fanout_atomic_count[thread_idx] += fanout_atomics;
        g_sched_push_wait_cycle[thread_idx] += push_wait;
        PTO2_SCHED_CYCLE_LAP(g_sched_fanout_cycle[thread_idx]);
        return stats;
#endif
    }

    /**
     * Cold path: release producers (fanin traversal) + check self for CONSUMED.
     * Returns fanin edge count for profiling.
     */

#if PTO2_SCHED_PROFILING
    int32_t on_task_release(PTO2TaskSlotState &slot_state, int32_t thread_idx) {
        PTO2_SCHED_CYCLE_START();
        extern uint64_t g_sched_fanin_cycle[], g_sched_fanin_atomic_count[];
        extern uint64_t g_sched_self_atomic_count[];
        extern uint64_t g_sched_self_consumed_cycle[];
        extern uint64_t g_sched_complete_count[];
        uint64_t fanin_atomics = 0;
#else
    int32_t on_task_release(PTO2TaskSlotState &slot_state) {
#endif
        PTO2TaskPayload *payload = slot_state.payload;
        for_each_fanin_slot_state(*payload, [&](PTO2TaskSlotState *producer_slot_state) {
#if PTO2_SCHED_PROFILING
            release_producer(*producer_slot_state, fanin_atomics);
#else
            release_producer(*producer_slot_state);
#endif
        });
#if PTO2_SCHED_PROFILING
        g_sched_fanin_atomic_count[thread_idx] += fanin_atomics;
        PTO2_SCHED_CYCLE_LAP(g_sched_fanin_cycle[thread_idx]);
#endif

        // Self consumed check
#if PTO2_SCHED_PROFILING
        uint64_t self_atomics = 0;
        check_and_handle_consumed(slot_state, self_atomics);
        g_sched_self_atomic_count[thread_idx] += self_atomics;
        PTO2_SCHED_CYCLE_LAP(g_sched_self_consumed_cycle[thread_idx]);
        g_sched_complete_count[thread_idx]++;
#else
        check_and_handle_consumed(slot_state);
#endif
        return payload->fanin_actual_count;
    }

    // === Cold-path API (defined in pto_scheduler.cpp) ===

    // Phase 1: declare every sub-region (ready_queue slots, dummy queue slots,
    // per-ring dep_pool entries, wiring SPSC buffer) on the supplied arena.
    // Capacities are baked into the returned layout; init_data_from_layout uses
    // the same values.
    static PTO2SchedulerLayout reserve_layout(DeviceArena &arena, int32_t dep_pool_capacity = PTO2_DEP_LIST_POOL_SIZE);

    // Phase 3a: write everything *except* arena-internal pointer fields.
    // `sm_dev_base` is the device address of the SM (only stored, never
    // dereferenced here). Safe to call on a host arena that holds the
    // prebuilt image buffer. (The orchestrator counterpart takes
    // task_window_size for ring task_descriptors address arithmetic; the
    // scheduler only needs the SM header / ring header base addresses,
    // both window-size-independent.)
    bool init_data_from_layout(const PTO2SchedulerLayout &layout, DeviceArena &arena, void *sm_dev_base);

    // Phase 3b: write the arena-internal pointer fields
    // (ready_queues[].slots, dummy_ready_queue.slots, dep_pool.base for each
    // ring, wiring.queue.buffer_). Called on both host and device sides.
    void wire_arena_pointers(const PTO2SchedulerLayout &layout, DeviceArena &arena);

    // Forget per-region pointers; arena owns the backing memory.
    void destroy();
    void print_stats();
    void print_queues();
};

// Scheduler cold-path API is declared as PTO2SchedulerState member functions.
// See init()/destroy()/print_stats()/print_queues() below the struct definition.

template <bool Profiling>
inline AsyncPollResult AsyncWaitList::poll_and_complete(
    volatile AICoreCompletionMailbox *aicore_mailbox, PTO2SchedulerState *sched, PTO2LocalReadyBuffer *local_bufs,
    PTO2TaskSlotState **deferred_release_slot_states, int32_t &deferred_release_count, int32_t deferred_release_capacity
#if PTO2_SCHED_PROFILING
    ,
    int thread_idx
#endif
) {
    AsyncPollResult result;
    if (!try_lock()) return result;

    int32_t drain_err = PTO2_ERROR_NONE;
    drain_aicore_completion_mailbox_locked(aicore_mailbox, drain_err);
    if (drain_err != PTO2_ERROR_NONE) {
        result.error_code = drain_err;
        unlock();
        return result;
    }

    for (int32_t i = count - 1; i >= 0; --i) {
        AsyncWaitEntry &entry = entries[i];
        for (int32_t c = 0; c < entry.condition_count; c++) {
            CompletionCondition &cond = entry.conditions[c];
            if (cond.satisfied) continue;
            CompletionPollResult poll = cond.test();
            if (poll.state == CompletionPollState::FAILED) {
                result.error_code = poll.error_code;
                result.failed_slot_state = entry.slot_state;
                unlock();
                return result;
            }
            if (poll.state == CompletionPollState::READY) {
                cond.satisfied = true;
                entry.waiting_completion_count--;
            }
        }

        if (entry.normal_done && entry.waiting_completion_count <= 0) {
            if (deferred_release_count >= deferred_release_capacity) {
                result.error_code = PTO2_ERROR_ASYNC_WAIT_OVERFLOW;
                result.failed_slot_state = entry.slot_state;
                unlock();
                return result;
            }
#if PTO2_SCHED_PROFILING
            sched->on_mixed_task_complete(*entry.slot_state, thread_idx, local_bufs);
#else
            sched->on_mixed_task_complete(*entry.slot_state, local_bufs);
#endif
            deferred_release_slot_states[deferred_release_count++] = entry.slot_state;
            result.completed++;

            int32_t last = count - 1;
            if (i != last) entries[i] = entries[last];
            count = last;
        }
    }

    unlock();
    return result;
}

// =============================================================================
// Scheduler Profiling Data
// =============================================================================

#if PTO2_SCHED_PROFILING
struct PTO2SchedProfilingData {
    // Sub-phase cycle breakdown within on_mixed_task_complete
    uint64_t lock_cycle;           // lock_fanout + state store + unlock
    uint64_t fanout_cycle;         // fanout traversal
    uint64_t fanin_cycle;          // fanin traversal
    uint64_t self_consumed_cycle;  // self check_and_handle_consumed

    // Wait times
    uint64_t lock_wait_cycle;  // spin-wait in fanout_lock
    uint64_t push_wait_cycle;  // CAS contention in push()
    uint64_t pop_wait_cycle;   // CAS contention in pop()

    // Atomic counts per sub-phase
    uint64_t lock_atomic_count;
    uint64_t fanout_atomic_count;
    uint64_t fanin_atomic_count;
    uint64_t self_atomic_count;
    uint64_t pop_atomic_count;

    int64_t complete_count;
};

/**
 * Get and reset scheduler profiling data for a specific thread.
 * Returns accumulated profiling data and resets counters.
 */
PTO2SchedProfilingData scheduler_get_profiling(int thread_idx);
#endif
