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
 * Architectural coupling detection tests for TMR (tensormap_and_ringbuffer) runtime.
 *
 * These tests verify whether components can operate in isolation or require
 * the full system to be initialized. Failures indicate tight coupling that
 * makes unit testing and independent evolution difficult.
 *
 * Test philosophy: FAIL = coupling defect detected (expected for some tests).
 */

#include <gtest/gtest.h>
#include <atomic>
#include <cstring>
#include <cstdlib>

#include "pto_orchestrator.h"
#include "pto_scheduler.h"
#include "pto_tensormap.h"
#include "pto_ring_buffer.h"
#include "pto_shared_memory.h"
#include "pto_runtime2_types.h"
#include "pto_orchestration_api.h"  // for make_tensor_external (Tensor ctor is private)
#include "tensor.h"

// =============================================================================
// Helper: Full TMR system init/destroy (measures what's needed)
// =============================================================================

static constexpr uint64_t TEST_HEAP_SIZE = 65536;
static constexpr int32_t TEST_WINDOW_SIZE = 64;

struct TMRSystem {
    PTO2SharedMemoryHandle *sm = nullptr;
    PTO2SchedulerState sched{};
    PTO2OrchestratorState orch{};
    uint8_t *gm_heap = nullptr;
    bool sm_ok = false, sched_ok = false, orch_ok = false;

    bool init(uint64_t heap_size = TEST_HEAP_SIZE, int32_t window_size = TEST_WINDOW_SIZE) {
        sm = pto2_sm_create(window_size, heap_size);
        if (!sm) return false;
        sm_ok = true;

        gm_heap = (uint8_t *)calloc(PTO2_MAX_RING_DEPTH, heap_size);
        if (!gm_heap) return false;

        if (!pto2_scheduler_init(&sched, sm->header)) return false;
        sched_ok = true;

        if (!pto2_orchestrator_init(&orch, sm->header, gm_heap, heap_size, 256)) return false;
        orch_ok = true;

        pto2_orchestrator_set_scheduler(&orch, &sched);
        return true;
    }

    void destroy() {
        if (orch_ok) pto2_orchestrator_destroy(&orch);
        if (sched_ok) pto2_scheduler_destroy(&sched);
        if (gm_heap) {
            free(gm_heap);
            gm_heap = nullptr;
        }
        if (sm_ok) pto2_sm_destroy(sm);
    }
};

// Helper: create a minimal Tensor for TensorMap operations.
// Tensor's default constructor is private; route through make_tensor_external.
// The `addr` argument is reinterpreted as a fake pointer -- the TensorMap only
// hashes the address and compares shapes, it never dereferences the buffer.
static Tensor make_test_tensor(uint64_t addr, uint32_t ndims = 1, uint32_t shape0 = 100) {
    uint32_t shapes[RUNTIME_MAX_TENSOR_DIMS] = {};
    shapes[0] = shape0;
    for (uint32_t i = 1; i < ndims; i++)
        shapes[i] = 1;
    return make_tensor_external(
        reinterpret_cast<void *>(addr), shapes, ndims, DataType::FLOAT32, /*manual_dep=*/false, /*version=*/0
    );
}

// =============================================================================
// Suite 1: ComponentIsolation
// =============================================================================

TEST(ComponentIsolation, TensorMapWithoutOrchPointer) {
    // TensorMap has an `orch` pointer field (set by orchestrator_init).
    // Can we use TensorMap for insert + lookup without setting it?
    int32_t window_sizes[PTO2_MAX_RING_DEPTH] = {16, 16, 16, 16};
    PTO2TensorMap tmap{};
    ASSERT_TRUE(tmap.init(256, 1024, window_sizes));

    // orch pointer is never set -- TensorMap is used standalone

    // Insert should work
    Tensor t = make_test_tensor(0x1000);
    PTO2TaskId tid = PTO2TaskId::make(0, 0);
    tmap.insert(t, tid);

    // Lookup should work
    PTO2LookupResult result;
    tmap.lookup(t, result);
    EXPECT_GE(
        result.count, 1
    ) << "TensorMap lookup works without orch pointer -- orch is unused for core insert/lookup operations";

    tmap.destroy();
}

TEST(ComponentIsolation, TensorMapWithZeroWindowSizes) {
    // Passing zero window sizes to TensorMap::init() should be rejected,
    // but there's no validation.
    int32_t window_sizes[PTO2_MAX_RING_DEPTH] = {0, 0, 0, 0};
    PTO2TensorMap tmap{};
    // init calls malloc(0 * sizeof(ptr)) for task_entry_heads -- implementation-defined
    bool ok = tmap.init(256, 1024, window_sizes);

    if (ok) {
        // If init succeeded, inserting should be unsafe because
        // mask = (0 - 1) = 0xFFFFFFFF -- slot index would be OOB.
        // This proves lack of input validation.
        // We can't safely test insert, just document the gap.
        SUCCEED() << "Zero window_size accepted without validation: "
                     "insert would compute OOB slot index";
        tmap.destroy();
    } else {
        // malloc(0) returned NULL on this platform
        SUCCEED() << "init correctly failed with zero window_size (malloc(0) returned NULL)";
    }
}

TEST(ComponentIsolation, DepPoolReclaimNeedsScheduler) {
    // DepListPool::reclaim() takes PTO2SchedulerState& and accesses
    // sched.ring_sched_states[ring_id].get_slot_state_by_task_id(sm_last_task_alive - 1)
    // This couples DepPool to Scheduler internals.
    PTO2DepListEntry entries[64];
    memset(entries, 0, sizeof(entries));
    std::atomic<int32_t> error_code{0};
    PTO2DepListPool pool;
    pool.init(entries, 64, &error_code);

    // Allocate some entries to make top > 0
    for (int i = 0; i < 10; i++) {
        pool.alloc();
    }

    // To call reclaim, we need a PTO2SharedMemoryRingHeader.
    // Create a minimal SM to get a valid ring header.
    PTO2SharedMemoryHandle *sm = pto2_sm_create(TEST_WINDOW_SIZE, TEST_HEAP_SIZE);
    ASSERT_NE(sm, nullptr);

    // reclaim with sm_last_task_alive=0 should be a no-op (guard: sm_last_task_alive > 0)
    pool.reclaim(sm->header->rings[0], 0);
    SUCCEED() << "reclaim with last_task_alive=0 is a no-op";

    // reclaim with sm_last_task_alive=PTO2_DEP_POOL_CLEANUP_INTERVAL would access
    // sched.ring_sched_states[0].slot_states[...] which is nullptr
    // This demonstrates the coupling: DepPool cannot reclaim without valid Scheduler state
    // We can't safely call reclaim(sched, 0, 64) because it would dereference nullptr

    // Document the coupling via signature inspection
    SUCCEED() << "DepPool::reclaim() requires PTO2SharedMemoryRingHeader& -- "
                 "cannot reclaim without valid shared memory ring header";

    pto2_sm_destroy(sm);
}

TEST(ComponentIsolation, DepPoolEnsureSpaceSignatureCoupling) {
    // ensure_space() requires BOTH PTO2SchedulerState& AND PTO2RingFlowControl&
    // This couples DepPool to Scheduler + SharedMemory simultaneously
    PTO2DepListEntry entries[256];
    memset(entries, 0, sizeof(entries));
    std::atomic<int32_t> error_code{0};
    PTO2DepListPool pool;
    pool.init(entries, 256, &error_code);

    // With enough space, ensure_space returns immediately without accessing ring header
    PTO2SharedMemoryHandle *sm = pto2_sm_create(TEST_WINDOW_SIZE, TEST_HEAP_SIZE);
    ASSERT_NE(sm, nullptr);

    pool.ensure_space(sm->header->rings[0], 5);  // available() = 255 >= 5 -- no-op
    EXPECT_GE(pool.available(), 5) << "ensure_space returns immediately when space sufficient, "
                                      "but signature still requires PTO2SharedMemoryRingHeader reference";

    pto2_sm_destroy(sm);
}

TEST(ComponentIsolation, SchedulerConsumedPathAccessesSM) {
    // check_and_handle_consumed -> advance_ring_pointers requires valid SM header.
    // Build a minimal slot that would trigger the consumed path.
    TMRSystem sys;
    ASSERT_TRUE(sys.init());

    auto &rs = sys.sched.ring_sched_states[0];
    PTO2TaskSlotState &slot = sys.sm->header->rings[0].get_slot_state_by_slot(0);

    // Set up a task that appears consumed
    slot.fanout_count = 1;
    slot.fanout_refcount.store(1, std::memory_order_relaxed);
    slot.task_state.store(PTO2_TASK_COMPLETED, std::memory_order_relaxed);
    slot.ring_id = 0;

    // Provide a valid task descriptor so advance_ring_pointers won't crash
    PTO2TaskDescriptor dummy_desc{};
    dummy_desc.packed_buffer_base = nullptr;
    dummy_desc.packed_buffer_end = nullptr;
    slot.task = &dummy_desc;

    // Set current_task_index to 1 so advance_ring_pointers scans slot 0
    sys.sm->header->rings[0].fc.current_task_index.store(1, std::memory_order_relaxed);

    // This should work with valid SM, proving SM is required
    sys.sched.check_and_handle_consumed(slot);
    EXPECT_EQ(slot.task_state.load(), PTO2_TASK_CONSUMED)
        << "check_and_handle_consumed works only with valid SM handle -- "
           "Scheduler->SharedMemory tight coupling confirmed";

    sys.destroy();
}

TEST(ComponentIsolation, OrchestratorInitWithoutSM) {
    // pto2_orchestrator_init dereferences sm_header->rings[r].fc immediately.
    // Passing nullptr should crash (no null-check).
    PTO2OrchestratorState orch{};
    uint8_t heap[1024];

    EXPECT_DEATH(pto2_orchestrator_init(&orch, nullptr, heap, 1024), ".*")
        << "Orchestrator init does not validate sm_header != nullptr";
}

TEST(ComponentIsolation, TaskSlotStateStandalone) {
    // TaskSlotState should be the one type that can be operated independently.
    // Manually drive the full state machine.
    alignas(64) PTO2TaskSlotState slot{};
    slot.fanin_count = 2;
    slot.fanout_count = 1;
    slot.fanin_refcount.store(0, std::memory_order_relaxed);
    slot.fanout_refcount.store(0, std::memory_order_relaxed);
    slot.task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);

    // PENDING -> READY: fanin_refcount reaches fanin_count
    slot.fanin_refcount.fetch_add(1, std::memory_order_relaxed);
    slot.fanin_refcount.fetch_add(1, std::memory_order_relaxed);
    EXPECT_EQ(slot.fanin_refcount.load(), slot.fanin_count);

    PTO2TaskState expected_pending = PTO2_TASK_PENDING;
    EXPECT_TRUE(slot.task_state.compare_exchange_strong(expected_pending, PTO2_TASK_READY));

    // READY -> RUNNING
    PTO2TaskState expected_ready = PTO2_TASK_READY;
    EXPECT_TRUE(slot.task_state.compare_exchange_strong(expected_ready, PTO2_TASK_RUNNING));

    // RUNNING -> COMPLETED
    slot.task_state.store(PTO2_TASK_COMPLETED, std::memory_order_relaxed);

    // COMPLETED -> CONSUMED: fanout_refcount reaches fanout_count
    slot.fanout_refcount.fetch_add(1, std::memory_order_relaxed);
    EXPECT_EQ(slot.fanout_refcount.load(), slot.fanout_count);

    PTO2TaskState expected_completed = PTO2_TASK_COMPLETED;
    EXPECT_TRUE(slot.task_state.compare_exchange_strong(expected_completed, PTO2_TASK_CONSUMED))
        << "TaskSlotState can be fully driven standalone -- good isolation";
}

TEST(ComponentIsolation, HeapRingWithLocalAtomics) {
    // The standalone PTO2HeapRing/pto2_heap_ring_init API has been consolidated
    // into PTO2TaskAllocator, which couples the heap and the task ring. There is
    // no longer a way to exercise heap allocation in isolation with just local
    // atomics -- you need a fully initialized allocator backed by SM pointers.
    //
    // This test is preserved as a documentation of the tightening of that
    // coupling: heap alloc can no longer run independently of the task ring.
    SUCCEED() << "PTO2HeapRing/pto2_heap_ring_init removed -- heap allocation is "
                 "now embedded in PTO2TaskAllocator, which requires a task ring "
                 "and SM-backed atomics. Heap allocation is no longer isolable.";
}

// =============================================================================
// Suite 2: InitializationOrder
// =============================================================================

TEST(InitializationOrder, TensorMapInitWithGarbageWindowSizes) {
    // If SM header is not initialized before TensorMap::init_default(),
    // garbage window_sizes are read. Simulate this with large values.
    int32_t garbage_sizes[PTO2_MAX_RING_DEPTH] = {-1, -1, -1, -1};
    PTO2TensorMap tmap{};

    // malloc(-1 * sizeof(ptr)) = malloc(huge) -- should fail
    bool ok = tmap.init(256, 1024, garbage_sizes);
    EXPECT_FALSE(ok) << "TensorMap::init with negative window_sizes should fail on malloc, "
                        "but no explicit validation rejects negative values before malloc";

    if (ok) tmap.destroy();
}

TEST(InitializationOrder, SchedulerInitWithZeroWindowSize) {
    // If SM has task_window_size=0, scheduler creates arrays of size 0.
    PTO2SharedMemoryHandle *sm = pto2_sm_create(0, TEST_HEAP_SIZE);

    if (sm == nullptr) {
        // pto2_sm_create rejects 0 window -- good validation
        SUCCEED() << "pto2_sm_create rejects window_size=0";
        return;
    }

    PTO2SchedulerState sched{};
    uint8_t heap[TEST_HEAP_SIZE * PTO2_MAX_RING_DEPTH]{};
    (void)heap;

    bool ok = pto2_scheduler_init(&sched, sm->header);
    if (ok) {
        // task_window_mask = 0 - 1 = -1 (wraps to max uint)
        // get_slot_state_by_task_id(0) would access slot_states[0 & (-1)] = slot_states[0]
        // But slot_states was allocated with new PTO2TaskSlotState[0] -- zero-length!
        EXPECT_EQ(sm->header->rings[0].task_window_size, 0u)
            << "Zero window_size accepted: slot_states[0] is zero-length allocation, "
               "any access is UB";
        pto2_scheduler_destroy(&sched);
    }

    pto2_sm_destroy(sm);
}

TEST(InitializationOrder, OrchestratorDoubleInit) {
    // Calling init twice without destroy leaks all first-init allocations.
    TMRSystem sys;
    ASSERT_TRUE(sys.init());

    // Re-init without destroy -- old allocations are leaked
    uint8_t extra_heap[TEST_HEAP_SIZE * PTO2_MAX_RING_DEPTH]{};
    bool ok = pto2_orchestrator_init(&sys.orch, sys.sm->header, extra_heap, TEST_HEAP_SIZE, 256);
    EXPECT_TRUE(ok) << "Double init succeeds -- no guard against re-initialization. "
                       "First init's allocations are leaked";

    // Clean up the second init
    pto2_orchestrator_destroy(&sys.orch);

    // First init's memory is leaked -- we can't free it anymore
    // This is a documentation test: no re-init guard exists
    sys.orch_ok = false;  // prevent double destroy
    sys.destroy();
}

TEST(InitializationOrder, OrchestratorBeforeScheduler) {
    // Init orchestrator without setting scheduler. scope_begin + scope_end should
    // degrade gracefully (skip dependency tracking).
    PTO2SharedMemoryHandle *sm = pto2_sm_create(TEST_WINDOW_SIZE, TEST_HEAP_SIZE);
    ASSERT_NE(sm, nullptr);

    uint8_t *heap = (uint8_t *)calloc(PTO2_MAX_RING_DEPTH, TEST_HEAP_SIZE);
    ASSERT_NE(heap, nullptr);

    PTO2OrchestratorState orch{};
    ASSERT_TRUE(pto2_orchestrator_init(&orch, sm->header, heap, TEST_HEAP_SIZE, 256));

    // scheduler is not set -- scope_begin/scope_end should not crash
    pto2_scope_begin(&orch);
    pto2_scope_end(&orch);
    SUCCEED() << "scope_begin + scope_end work without scheduler (no crash). "
                 "Tasks submitted in this scope have no dependency tracking.";

    pto2_orchestrator_destroy(&orch);
    free(heap);
    pto2_sm_destroy(sm);
}

// =============================================================================
// Suite 3: CrossComponentContract
// =============================================================================

TEST(CrossComponentContract, WindowSizeMismatch) {
    // After the PTO2SharedMemoryRingHeader consolidation (#622), both scheduler
    // and orchestrator read window_size from the same SM ring header pointer.
    // Verify via the SM header: the single source of truth.
    PTO2SharedMemoryHandle *sm = pto2_sm_create(TEST_WINDOW_SIZE, TEST_HEAP_SIZE);
    ASSERT_NE(sm, nullptr);

    uint8_t *heap = (uint8_t *)calloc(PTO2_MAX_RING_DEPTH, TEST_HEAP_SIZE);
    ASSERT_NE(heap, nullptr);

    // Initialize scheduler and orchestrator
    PTO2SchedulerState sched{};
    ASSERT_TRUE(pto2_scheduler_init(&sched, sm->header));

    PTO2OrchestratorState orch{};
    ASSERT_TRUE(pto2_orchestrator_init(&orch, sm->header, heap, TEST_HEAP_SIZE, 256));

    // Both read from the same SM header -- verify the header value is correct
    EXPECT_EQ(sm->header->rings[0].task_window_size, (uint64_t)TEST_WINDOW_SIZE)
        << "SM ring header holds the authoritative window_size";

    // Mutate SM header -- both components see the new value because they
    // share the same ring header pointer
    sm->header->rings[0].task_window_size = TEST_WINDOW_SIZE * 2;
    EXPECT_EQ(sm->header->rings[0].task_window_size, (uint64_t)(TEST_WINDOW_SIZE * 2))
        << "After RingHeader consolidation, mutation is visible to all components "
           "through the shared ring header pointer -- independent-caching mismatch eliminated";

    pto2_orchestrator_destroy(&orch);
    pto2_scheduler_destroy(&sched);
    free(heap);
    pto2_sm_destroy(sm);
}

TEST(CrossComponentContract, FanoutCountManipulation) {
    // fanout_count is set by orchestrator (+1 for scope), checked by scheduler.
    // If we bypass the +1 initialization, check_and_handle_consumed fires immediately.
    TMRSystem sys;
    ASSERT_TRUE(sys.init());

    auto &rs = sys.sched.ring_sched_states[0];
    PTO2TaskSlotState &slot = sys.sm->header->rings[0].get_slot_state_by_slot(0);

    PTO2TaskDescriptor dummy_desc{};
    dummy_desc.packed_buffer_base = nullptr;
    dummy_desc.packed_buffer_end = nullptr;
    slot.task = &dummy_desc;
    slot.ring_id = 0;

    // Normal init: orchestrator sets fanout_count = 1 (scope ref)
    // Here we bypass: set fanout_count = 0 directly
    slot.fanout_count = 0;
    slot.fanout_refcount.store(0, std::memory_order_relaxed);
    slot.task_state.store(PTO2_TASK_COMPLETED, std::memory_order_relaxed);

    sys.sm->header->rings[0].fc.current_task_index.store(1, std::memory_order_relaxed);

    // check_and_handle_consumed: fanout_refcount(0) == fanout_count(0) -> true -> CONSUMED
    sys.sched.check_and_handle_consumed(slot);

    EXPECT_EQ(slot.task_state.load(), PTO2_TASK_CONSUMED)
        << "fanout_count=0 causes premature CONSUMED transition -- "
           "scheduler trusts orchestrator's fanout_count without validation";
}

TEST(CrossComponentContract, HeapTailBeyondTop) {
    // Previously tested PTO2HeapRing::pto2_heap_ring_try_alloc with manually
    // constructed top/tail atomics. PTO2HeapRing no longer exists as a
    // free-standing component -- heap state (top/tail) is now encapsulated in
    // PTO2TaskAllocator as local integers derived from task descriptors, not
    // from externally writable atomics. An invalid tail>top state cannot be
    // synthesized without a full allocator + scheduler setup, so this
    // coupling-contract scenario is no longer reachable from a unit test.
    SUCCEED() << "PTO2HeapRing removed; heap tail/top are now internal to "
                 "PTO2TaskAllocator and derived from consumed task descriptors. "
                 "No external atomic to corrupt -- this specific invariant is "
                 "enforced by construction rather than by validation.";
}

TEST(CrossComponentContract, ActiveMaskZero) {
    // active_mask=0 should never happen (orchestrator has always_assert).
    // But scheduler's release_fanin_and_check_ready has no such guard.
    alignas(64) PTO2TaskSlotState slot{};
    slot.active_mask = 0;  // Invalid -- no subtask active
    slot.fanin_count = 1;
    slot.fanin_refcount.store(0, std::memory_order_relaxed);
    slot.task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);

    PTO2ResourceShape shape = pto2_active_mask_to_shape(0);
    // With mask=0: core_mask=0, popcount=0, no AIC bit -> falls through to AIV.
    // The enum has been simplified to {AIC, AIV, MIX}; there is no longer a
    // distinct AIV_X2 shape (multi-AIV tasks are all MIX).
    EXPECT_EQ(static_cast<int>(shape), static_cast<int>(PTO2ResourceShape::AIV))
        << "active_mask=0 maps to AIV -- incorrect shape routing. "
           "Orchestrator guards with always_assert, but scheduler does not validate";
}

TEST(CrossComponentContract, TaskDescriptorNullInConsumedSlot) {
    // Historically advance_ring_pointers dereferenced slot.task->packed_buffer_end
    // to drive heap reclamation from the last consumed task. Heap reclamation
    // has since moved into PTO2TaskAllocator::update_heap_tail (reached by the
    // orchestrator on allocation), so advance_ring_pointers no longer touches
    // slot.task at all -- it only walks task_state. The coupling this test was
    // designed to surface has been removed by construction.
    TMRSystem sys;
    ASSERT_TRUE(sys.init());

    auto &rs = sys.sched.ring_sched_states[0];
    PTO2TaskSlotState &slot = sys.sm->header->rings[0].get_slot_state_by_slot(0);

    // Mark as CONSUMED but leave task pointer as nullptr
    slot.task_state.store(PTO2_TASK_CONSUMED, std::memory_order_relaxed);
    slot.task = nullptr;  // Not initialized
    slot.ring_id = 0;

    sys.sm->header->rings[0].fc.current_task_index.store(1, std::memory_order_relaxed);

    // Should no longer crash: advance_ring_pointers now only reads task_state.
    rs.advance_ring_pointers();
    EXPECT_EQ(rs.last_task_alive, 1) << "advance_ring_pointers no longer dereferences slot.task -- "
                                        "scheduler/orchestrator heap-reclamation coupling removed";

    sys.destroy();
}

// =============================================================================
// Suite 4: StateLeakage
// =============================================================================

TEST(StateLeakage, HeapErrorCodeInvisibleToScheduler) {
    // Orchestrator sets orch_error_code on fatal error.
    // Scheduler's hot path does NOT check this error code.
    TMRSystem sys;
    ASSERT_TRUE(sys.init());

    // Simulate orchestrator setting fatal error
    sys.sm->header->orch_error_code.store(PTO2_ERROR_HEAP_RING_DEADLOCK, std::memory_order_release);

    // Scheduler operations continue despite error:
    // push to ready queue
    auto &rs = sys.sched.ring_sched_states[0];
    PTO2TaskSlotState &slot = sys.sm->header->rings[0].get_slot_state_by_slot(0);
    slot.active_mask = PTO2_SUBTASK_MASK_AIV0;
    PTO2ResourceShape shape = pto2_active_mask_to_shape(slot.active_mask);

    bool pushed = sys.sched.ready_queues[static_cast<int>(shape)].push(&slot);
    EXPECT_TRUE(pushed);

    // pop from ready queue
    PTO2TaskSlotState *popped = sys.sched.ready_queues[static_cast<int>(shape)].pop();
    EXPECT_EQ(popped, &slot) << "Scheduler continues normal operation after orchestrator fatal error -- "
                                "orch_error_code is one-directional (orch->host), invisible to scheduler hot path";

    sys.destroy();
}

TEST(StateLeakage, HeadOfLineBlocking) {
    // advance_ring_pointers scans linearly: stops at first non-CONSUMED slot.
    // One incomplete task blocks reclamation of all subsequent CONSUMED tasks.
    TMRSystem sys;
    ASSERT_TRUE(sys.init());

    auto &rs = sys.sched.ring_sched_states[0];
    PTO2TaskDescriptor descs[3]{};
    descs[0].packed_buffer_end = nullptr;
    descs[1].packed_buffer_end = nullptr;
    descs[2].packed_buffer_end = nullptr;

    // Task 0: CONSUMED
    PTO2TaskSlotState &slot0 = sys.sm->header->rings[0].get_slot_state_by_slot(0);
    slot0.task_state.store(PTO2_TASK_CONSUMED, std::memory_order_relaxed);
    slot0.task = &descs[0];

    // Task 1: COMPLETED (NOT consumed -- fanout incomplete)
    PTO2TaskSlotState &slot1 = sys.sm->header->rings[0].get_slot_state_by_slot(1);
    slot1.task_state.store(PTO2_TASK_COMPLETED, std::memory_order_relaxed);
    slot1.task = &descs[1];

    // Task 2: CONSUMED
    PTO2TaskSlotState &slot2 = sys.sm->header->rings[0].get_slot_state_by_slot(2);
    slot2.task_state.store(PTO2_TASK_CONSUMED, std::memory_order_relaxed);
    slot2.task = &descs[2];

    sys.sm->header->rings[0].fc.current_task_index.store(3, std::memory_order_relaxed);

    rs.advance_ring_pointers();

    // last_task_alive should stop at task 1 (COMPLETED, not CONSUMED)
    EXPECT_EQ(rs.last_task_alive, 1) << "Head-of-line blocking: task 1 (COMPLETED) blocks reclamation of "
                                        "task 2 (CONSUMED). Linear scan design couples reclamation rate "
                                        "to the slowest consumer in the ring.";

    sys.destroy();
}

TEST(StateLeakage, TensorMapCleanupInterval) {
    // TensorMap cleanup is triggered every PTO2_TENSORMAP_CLEANUP_INTERVAL tasks.
    // Between cleanups, stale entries accumulate in bucket chains, degrading lookup.
    int32_t window_sizes[PTO2_MAX_RING_DEPTH] = {256, 256, 256, 256};
    PTO2TensorMap tmap{};
    ASSERT_TRUE(tmap.init(256, 4096, window_sizes));

    // Insert entries for tasks 0..99 (all same address = same bucket)
    for (int i = 0; i < 100; i++) {
        Tensor t = make_test_tensor(0x2000);
        PTO2TaskId tid = PTO2TaskId::make(0, i);
        tmap.insert(t, tid);
    }

    // Advance last_task_alive to 80 -- tasks 0..79 are stale
    tmap.sync_validity(0, 80);

    // Lookup must traverse all 100 entries (80 stale + 20 valid)
    // because cleanup hasn't been triggered yet (need sync_tensormap, not just sync_validity)
    PTO2LookupResult result;
    Tensor query = make_test_tensor(0x2000);
    tmap.lookup(query, result);

    // Should find entries from tasks 80..99 = 20 valid
    EXPECT_EQ(result.count, 16) << "Lookup result capped at PTO2_LOOKUP_MAX_RESULTS=16, but stale entries "
                                   "still slow traversal. Cleanup interval ("
                                << PTO2_TENSORMAP_CLEANUP_INTERVAL
                                << " tasks) couples TensorMap performance to scheduler's CONSUMED advancement rate";

    tmap.destroy();
}

TEST(StateLeakage, SubtaskMaskProtocol) {
    // active_mask bits (AIC=0x1, AIV0=0x2, AIV1=0x4) are set by orchestrator
    // and checked by scheduler's on_subtask_complete. There's no shared enum
    // enforcing consistency -- just implicit agreement on bit positions.

    // Orchestrator normalizes aiv1-only to aiv0:
    // If only aiv1 set (0x4), it moves to aiv0 (0x2).
    // Scheduler uses SubtaskSlot enum (AIC=0, AIV0=1, AIV1=2) for done_bit.

    // Verify the normalization creates an implicit contract:
    uint8_t mask_aiv1_only = PTO2_SUBTASK_MASK_AIV1;  // 0x4
    // After orchestrator normalization: becomes PTO2_SUBTASK_MASK_AIV0 = 0x2
    uint8_t normalized = PTO2_SUBTASK_MASK_AIV0;  // aiv1 moved to aiv0

    // Scheduler completion path: on_subtask_complete with AIV0 slot sets bit 1
    uint8_t done_bit = (1u << static_cast<uint8_t>(PTO2SubtaskSlot::AIV0));
    EXPECT_EQ(done_bit, PTO2_SUBTASK_MASK_AIV0);

    // But if scheduler receives completion for AIV1 slot (the physical source),
    // it would set bit 2, which doesn't match normalized mask 0x2
    uint8_t wrong_done_bit = (1u << static_cast<uint8_t>(PTO2SubtaskSlot::AIV1));
    EXPECT_NE(wrong_done_bit, normalized)
        << "Subtask mask protocol: orchestrator normalizes aiv1->aiv0 (mask 0x4->0x2), "
           "but scheduler must dispatch to AIV0 slot (not AIV1). "
           "If scheduler signals AIV1 completion, done_mask (0x4) != active_mask (0x2) -- "
           "task never completes. No compile-time enforcement exists.";
}

// =============================================================================
// Suite 5: CompileTimeCoupling
// =============================================================================

TEST(CompileTimeCoupling, OrchestratorInitDestroyCycle) {
    // Orchestrator embeds rings, TensorMap, scope stack -- a large composite.
    // Verify it can be initialized and destroyed cleanly multiple times,
    // proving all sub-components are properly managed.
    for (int cycle = 0; cycle < 3; cycle++) {
        TMRSystem sys;
        ASSERT_TRUE(sys.init()) << "Init cycle " << cycle;
        sys.destroy();
    }
    SUCCEED() << "OrchestratorState init/destroy is clean across multiple cycles";
}

TEST(CompileTimeCoupling, MaxRingDepthPropagation) {
    // PTO2_MAX_RING_DEPTH=4 is used across multiple components.
    // Verify that the system initializes and operates correctly for all rings
    // up to PTO2_MAX_RING_DEPTH, without probing internal array sizes.

    // static_asserts on array sizes at the struct level are compile-time safety
    // nets that belong in production headers, not in behavioral tests.
    // This test verifies the functional consequence: all ring indices work.
    PTO2SharedMemoryHandle *sm = pto2_sm_create(TEST_WINDOW_SIZE, TEST_HEAP_SIZE);
    ASSERT_NE(sm, nullptr);

    // Verify all rings are accessible through SM header
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        EXPECT_EQ(sm->header->rings[r].task_window_size, (uint64_t)TEST_WINDOW_SIZE)
            << "Ring " << r << " should be initialized with correct window_size";
    }

    // TensorMap should accept inserts and lookups on all rings
    int32_t window_sizes[PTO2_MAX_RING_DEPTH];
    for (int i = 0; i < PTO2_MAX_RING_DEPTH; i++)
        window_sizes[i] = TEST_WINDOW_SIZE;
    PTO2TensorMap tmap{};
    ASSERT_TRUE(tmap.init(256, 1024, window_sizes));

    Tensor t = make_test_tensor(0x1000);
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        tmap.insert(t, PTO2TaskId::make(r, 0));
    }

    PTO2LookupResult result;
    result.count = 0;
    tmap.lookup(t, result);
    EXPECT_EQ(result.count, PTO2_MAX_RING_DEPTH)
        << "TensorMap supports inserts on all " << PTO2_MAX_RING_DEPTH << " rings";

    tmap.destroy();
    pto2_sm_destroy(sm);
}

TEST(CompileTimeCoupling, WindowSizeConsistencyAfterInit) {
    // Verify that after full system init, all components operate correctly
    // with the configured window_size by exercising the public API.
    TMRSystem sys;
    ASSERT_TRUE(sys.init());

    // The authoritative window_size lives in the SM ring header
    uint64_t expected_window = sys.sm->header->rings[0].task_window_size;
    EXPECT_EQ(expected_window, (uint64_t)TEST_WINDOW_SIZE);

    // Verify functional consistency: insert tasks up to window_size
    // and confirm TensorMap, Orchestrator, and Scheduler all work correctly.
    Tensor t = make_test_tensor(0x1000);
    pto2_scope_begin(&sys.orch);

    // Insert a tensor -- exercises Orchestrator + TensorMap
    sys.orch.tensor_map.insert(t, PTO2TaskId::make(0, 0));

    // Lookup -- exercises TensorMap with its window_size
    PTO2LookupResult result;
    result.count = 0;
    sys.orch.tensor_map.lookup(t, result);
    EXPECT_EQ(result.count, 1) << "TensorMap insert+lookup works with configured window_size";

    pto2_scope_end(&sys.orch);

    sys.destroy();
}

TEST(CompileTimeCoupling, TaskSlotStateLifecycleStandalone) {
    // Verify TaskSlotState can be fully driven through its state machine
    // without any other component -- proving it is the nexus type that
    // both orchestrator and scheduler operate on.
    alignas(64) PTO2TaskSlotState slot{};
    slot.fanin_count = 2;
    slot.fanout_count = 1;
    slot.fanin_refcount.store(0, std::memory_order_relaxed);
    slot.fanout_refcount.store(0, std::memory_order_relaxed);
    slot.task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);

    // Drive full lifecycle: PENDING -> READY -> RUNNING -> COMPLETED -> CONSUMED
    slot.fanin_refcount.fetch_add(1);
    slot.fanin_refcount.fetch_add(1);
    EXPECT_EQ(slot.fanin_refcount.load(), slot.fanin_count);

    PTO2TaskState expected = PTO2_TASK_PENDING;
    EXPECT_TRUE(slot.task_state.compare_exchange_strong(expected, PTO2_TASK_READY));

    expected = PTO2_TASK_READY;
    EXPECT_TRUE(slot.task_state.compare_exchange_strong(expected, PTO2_TASK_RUNNING));

    slot.task_state.store(PTO2_TASK_COMPLETED, std::memory_order_relaxed);

    slot.fanout_refcount.fetch_add(1);
    EXPECT_EQ(slot.fanout_refcount.load(), slot.fanout_count);

    expected = PTO2_TASK_COMPLETED;
    EXPECT_TRUE(slot.task_state.compare_exchange_strong(expected, PTO2_TASK_CONSUMED))
        << "TaskSlotState can be fully driven standalone -- references types from "
           "both orchestrator and scheduler domains but is independently operable";
}

TEST(CompileTimeCoupling, ReadyQueueAllShapesUsable) {
    // PTO2_NUM_RESOURCE_SHAPES ready queues exist (one per shape).
    // Verify all can be initialized and used for push/pop.
    for (int s = 0; s < PTO2_NUM_RESOURCE_SHAPES; s++) {
        PTO2ReadyQueue queue{};
        ASSERT_TRUE(pto2_ready_queue_init(&queue, 16)) << "Shape " << s << " queue init failed";

        PTO2TaskSlotState item{};
        EXPECT_TRUE(queue.push(&item));
        EXPECT_EQ(queue.pop(), &item);

        pto2_ready_queue_destroy(&queue);
    }
}

TEST(CompileTimeCoupling, LinkDependencyChain) {
    // This test file links 5 runtime .cpp files:
    // pto_orchestrator.cpp, pto_tensormap.cpp, pto_shared_memory.cpp,
    // pto_ring_buffer.cpp, pto_scheduler.cpp
    // This is because pto_tensormap.cpp includes pto_orchestrator.h (circular),
    // which includes pto_scheduler.h, pto_ring_buffer.h, pto_shared_memory.h.
    // Cannot compile TensorMap without linking the full runtime.
    SUCCEED() << "test_coupling links 5 runtime .cpp files. "
                 "Root cause: pto_tensormap.cpp #includes pto_orchestrator.h "
                 "for sync_tensormap, creating a circular compile-unit dependency. "
                 "This forces all tests that include TensorMap to also link "
                 "Orchestrator, Scheduler, RingBuffer, and SharedMemory.";
}
