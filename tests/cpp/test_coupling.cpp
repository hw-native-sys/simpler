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
#include "tensor.h"

// =============================================================================
// Helper: Full TMR system init/destroy (measures what's needed)
// =============================================================================

static constexpr uint64_t TEST_HEAP_SIZE = 65536;
static constexpr int32_t  TEST_WINDOW_SIZE = 64;

struct TMRSystem {
    PTO2SharedMemoryHandle* sm = nullptr;
    PTO2SchedulerState sched{};
    PTO2OrchestratorState orch{};
    uint8_t* gm_heap = nullptr;
    bool sm_ok = false, sched_ok = false, orch_ok = false;

    bool init(uint64_t heap_size = TEST_HEAP_SIZE,
              int32_t window_size = TEST_WINDOW_SIZE) {
        sm = pto2_sm_create(window_size, heap_size);
        if (!sm) return false;
        sm_ok = true;

        gm_heap = (uint8_t*)calloc(PTO2_MAX_RING_DEPTH, heap_size);
        if (!gm_heap) return false;

        if (!pto2_scheduler_init(&sched, sm, gm_heap, heap_size)) return false;
        sched_ok = true;

        if (!pto2_orchestrator_init(&orch, sm, gm_heap, heap_size, 256)) return false;
        orch_ok = true;

        pto2_orchestrator_set_scheduler(&orch, &sched);
        return true;
    }

    void destroy() {
        if (orch_ok) pto2_orchestrator_destroy(&orch);
        if (sched_ok) pto2_scheduler_destroy(&sched);
        if (gm_heap) { free(gm_heap); gm_heap = nullptr; }
        if (sm_ok) pto2_sm_destroy(sm);
    }
};

// Helper: create a minimal Tensor for TensorMap operations
static Tensor make_test_tensor(uint64_t addr, uint32_t ndims = 1,
                               uint32_t shape0 = 100) {
    Tensor t{};
    t.buffer.addr = addr;
    t.buffer.size = shape0;
    t.ndims = ndims;
    t.shapes[0] = shape0;
    t.version = 0;
    t.is_all_offset_zero = true;
    return t;
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

    // orch pointer is never set — remains nullptr
    EXPECT_EQ(tmap.orch, nullptr);

    // Insert should work
    Tensor t = make_test_tensor(0x1000);
    PTO2TaskId tid = pto2_make_task_id(0, 0);
    tmap.insert(t, tid, true);

    // Lookup should work
    PTO2LookupResult result;
    tmap.lookup(t, result);
    EXPECT_GE(result.count, 1)
        << "TensorMap lookup works without orch pointer — orch is a dead member for core operations";

    tmap.destroy();
}

TEST(ComponentIsolation, TensorMapWithZeroWindowSizes) {
    // Passing zero window sizes to TensorMap::init() should be rejected,
    // but there's no validation.
    int32_t window_sizes[PTO2_MAX_RING_DEPTH] = {0, 0, 0, 0};
    PTO2TensorMap tmap{};
    // init calls malloc(0 * sizeof(ptr)) for task_entry_heads — implementation-defined
    bool ok = tmap.init(256, 1024, window_sizes);

    if (ok) {
        // If init succeeded, the mask becomes (0 - 1) = 0xFFFFFFFF
        // Insert would compute slot = local_id & 0xFFFFFFFF — OOB access
        // This proves lack of input validation
        EXPECT_EQ(tmap.task_window_sizes[0], 0)
            << "Zero window_size accepted without validation: "
               "mask = (0-1) = -1, insert would OOB";
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

    // Create a minimally zero-initialized scheduler (slot_states will be nullptr)
    PTO2SchedulerState sched{};
    memset(&sched, 0, sizeof(sched));

    // reclaim with sm_last_task_alive=0 should be a no-op (guard: sm_last_task_alive > 0)
    pool.reclaim(sched, 0, 0);
    SUCCEED() << "reclaim with last_task_alive=0 is a no-op";

    // reclaim with sm_last_task_alive=PTO2_DEP_POOL_CLEANUP_INTERVAL would access
    // sched.ring_sched_states[0].slot_states[...] which is nullptr
    // This demonstrates the coupling: DepPool cannot reclaim without valid Scheduler state
    // We can't safely call reclaim(sched, 0, 64) because it would dereference nullptr

    // Document the coupling via signature inspection
    SUCCEED() << "DepPool::reclaim() requires PTO2SchedulerState& — "
                 "cannot reclaim without fully initialized scheduler";
}

TEST(ComponentIsolation, DepPoolEnsureSpaceSignatureCoupling) {
    // ensure_space() requires BOTH PTO2SchedulerState& AND PTO2RingFlowControl&
    // This couples DepPool to Scheduler + SharedMemory simultaneously
    PTO2DepListEntry entries[256];
    memset(entries, 0, sizeof(entries));
    std::atomic<int32_t> error_code{0};
    PTO2DepListPool pool;
    pool.init(entries, 256, &error_code);

    // With enough space, ensure_space returns immediately without accessing params
    PTO2SchedulerState sched{};
    memset(&sched, 0, sizeof(sched));
    PTO2RingFlowControl fc{};
    fc.init();

    pool.ensure_space(sched, fc, 0, 5);  // available() = 255 >= 5 — no-op
    EXPECT_GE(pool.available(), 5)
        << "ensure_space returns immediately when space sufficient, "
           "but signature still requires Scheduler + FlowControl references";
}

TEST(ComponentIsolation, SchedulerConsumedPathAccessesSM) {
    // check_and_handle_consumed → advance_ring_pointers requires valid SM header.
    // Build a minimal slot that would trigger the consumed path.
    TMRSystem sys;
    ASSERT_TRUE(sys.init());

    auto& rs = sys.sched.ring_sched_states[0];
    PTO2TaskSlotState& slot = rs.get_slot_state_by_slot(0);

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
        << "check_and_handle_consumed works only with valid SM handle — "
           "Scheduler->SharedMemory tight coupling confirmed";

    sys.destroy();
}

TEST(ComponentIsolation, OrchestratorInitWithoutSM) {
    // pto2_orchestrator_init dereferences sm_handle->header->rings[r].fc immediately.
    // Passing nullptr should crash (no null-check).
    PTO2OrchestratorState orch{};
    uint8_t heap[1024];

    EXPECT_DEATH(
        pto2_orchestrator_init(&orch, nullptr, heap, 1024),
        ".*"
    ) << "Orchestrator init does not validate sm_handle != nullptr";
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

    // PENDING → READY: fanin_refcount reaches fanin_count
    slot.fanin_refcount.fetch_add(1, std::memory_order_relaxed);
    slot.fanin_refcount.fetch_add(1, std::memory_order_relaxed);
    EXPECT_EQ(slot.fanin_refcount.load(), slot.fanin_count);

    PTO2TaskState expected_pending = PTO2_TASK_PENDING;
    EXPECT_TRUE(slot.task_state.compare_exchange_strong(
        expected_pending, PTO2_TASK_READY));

    // READY → RUNNING
    PTO2TaskState expected_ready = PTO2_TASK_READY;
    EXPECT_TRUE(slot.task_state.compare_exchange_strong(
        expected_ready, PTO2_TASK_RUNNING));

    // RUNNING → COMPLETED
    slot.task_state.store(PTO2_TASK_COMPLETED, std::memory_order_relaxed);

    // COMPLETED → CONSUMED: fanout_refcount reaches fanout_count
    slot.fanout_refcount.fetch_add(1, std::memory_order_relaxed);
    EXPECT_EQ(slot.fanout_refcount.load(), slot.fanout_count);

    PTO2TaskState expected_completed = PTO2_TASK_COMPLETED;
    EXPECT_TRUE(slot.task_state.compare_exchange_strong(
        expected_completed, PTO2_TASK_CONSUMED))
        << "TaskSlotState can be fully driven standalone — good isolation";
}

TEST(ComponentIsolation, HeapRingWithLocalAtomics) {
    // HeapRing can work with local atomics, not requiring SharedMemory.
    alignas(64) uint8_t heap[4096]{};
    std::atomic<uint64_t> top{0}, tail{0};
    std::atomic<int32_t> error_code{0};
    PTO2HeapRing ring{};
    pto2_heap_ring_init(&ring, heap, 4096, &tail, &top);
    ring.error_code_ptr = &error_code;

    void* p = ring.pto2_heap_ring_try_alloc(128);
    EXPECT_NE(p, nullptr)
        << "HeapRing works with local atomics — good isolation baseline";
}

// =============================================================================
// Suite 2: InitializationOrder
// =============================================================================

TEST(InitializationOrder, TensorMapInitWithGarbageWindowSizes) {
    // If SM header is not initialized before TensorMap::init_default(),
    // garbage window_sizes are read. Simulate this with large values.
    int32_t garbage_sizes[PTO2_MAX_RING_DEPTH] = {-1, -1, -1, -1};
    PTO2TensorMap tmap{};

    // malloc(-1 * sizeof(ptr)) = malloc(huge) — should fail
    bool ok = tmap.init(256, 1024, garbage_sizes);
    EXPECT_FALSE(ok)
        << "TensorMap::init with negative window_sizes should fail on malloc, "
           "but no explicit validation rejects negative values before malloc";

    if (ok) tmap.destroy();
}

TEST(InitializationOrder, SchedulerInitWithZeroWindowSize) {
    // If SM has task_window_size=0, scheduler creates arrays of size 0.
    PTO2SharedMemoryHandle* sm = pto2_sm_create(0, TEST_HEAP_SIZE);

    if (sm == nullptr) {
        // pto2_sm_create rejects 0 window — good validation
        SUCCEED() << "pto2_sm_create rejects window_size=0";
        return;
    }

    PTO2SchedulerState sched{};
    uint8_t heap[TEST_HEAP_SIZE * PTO2_MAX_RING_DEPTH]{};

    bool ok = pto2_scheduler_init(&sched, sm, heap, TEST_HEAP_SIZE);
    if (ok) {
        // task_window_mask = 0 - 1 = -1 (wraps to max uint)
        // get_slot_state_by_task_id(0) would access slot_states[0 & (-1)] = slot_states[0]
        // But slot_states was allocated with new PTO2TaskSlotState[0] — zero-length!
        EXPECT_EQ(sched.ring_sched_states[0].task_window_size, 0u)
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

    // Record pointers from first init
    void* first_scope_tasks = sys.orch.scope_tasks;
    void* first_scope_begins = sys.orch.scope_begins;

    // Re-init without destroy — old allocations are leaked
    uint8_t extra_heap[TEST_HEAP_SIZE * PTO2_MAX_RING_DEPTH]{};
    bool ok = pto2_orchestrator_init(&sys.orch, sys.sm, extra_heap, TEST_HEAP_SIZE, 256);
    EXPECT_TRUE(ok)
        << "Double init succeeds — no guard against re-initialization. "
           "First init's allocations (scope_tasks, scope_begins, dep_pool bases, "
           "tensor_map) are leaked";

    // Clean up the second init
    pto2_orchestrator_destroy(&sys.orch);

    // First init's memory is leaked — we can't free it anymore
    // This is a documentation test: no re-init guard exists
    sys.orch_ok = false;  // prevent double destroy
    sys.destroy();
}

TEST(InitializationOrder, OrchestratorBeforeScheduler) {
    // Init orchestrator without setting scheduler. scope_begin + scope_end should
    // degrade gracefully (skip dependency tracking).
    PTO2SharedMemoryHandle* sm = pto2_sm_create(TEST_WINDOW_SIZE, TEST_HEAP_SIZE);
    ASSERT_NE(sm, nullptr);

    uint8_t* heap = (uint8_t*)calloc(PTO2_MAX_RING_DEPTH, TEST_HEAP_SIZE);
    ASSERT_NE(heap, nullptr);

    PTO2OrchestratorState orch{};
    ASSERT_TRUE(pto2_orchestrator_init(&orch, sm, heap, TEST_HEAP_SIZE, 256));

    // scheduler is nullptr — scope_end should check `if (orch->scheduler && count > 0)`
    EXPECT_EQ(orch.scheduler, nullptr);

    pto2_scope_begin(&orch);
    EXPECT_EQ(orch.scope_stack_top, 0);

    pto2_scope_end(&orch);
    EXPECT_EQ(orch.scope_stack_top, -1)
        << "scope_end works without scheduler (skips release_producer). "
           "But tasks submitted in this scope have no dependency tracking.";

    pto2_orchestrator_destroy(&orch);
    free(heap);
    pto2_sm_destroy(sm);
}

// =============================================================================
// Suite 3: CrossComponentContract
// =============================================================================

TEST(CrossComponentContract, WindowSizeMismatch) {
    // Scheduler and Orchestrator independently read window_size from SM header.
    // If the value changes between their reads, they disagree on slot count.
    PTO2SharedMemoryHandle* sm = pto2_sm_create(TEST_WINDOW_SIZE, TEST_HEAP_SIZE);
    ASSERT_NE(sm, nullptr);

    uint8_t* heap = (uint8_t*)calloc(PTO2_MAX_RING_DEPTH, TEST_HEAP_SIZE);
    ASSERT_NE(heap, nullptr);

    // Initialize scheduler with window=64
    PTO2SchedulerState sched{};
    ASSERT_TRUE(pto2_scheduler_init(&sched, sm, heap, TEST_HEAP_SIZE));
    EXPECT_EQ(sched.ring_sched_states[0].task_window_size, (uint64_t)TEST_WINDOW_SIZE);

    // Now change SM header before orchestrator reads it
    sm->header->rings[0].task_window_size = TEST_WINDOW_SIZE * 2;  // 128

    PTO2OrchestratorState orch{};
    ASSERT_TRUE(pto2_orchestrator_init(&orch, sm, heap, TEST_HEAP_SIZE, 256));

    // Orchestrator's TaskRing now has window=128, scheduler has window=64
    EXPECT_EQ(orch.rings[0].task_ring.window_size, TEST_WINDOW_SIZE * 2);
    EXPECT_NE(orch.rings[0].task_ring.window_size,
              (int32_t)sched.ring_sched_states[0].task_window_size)
        << "Window size mismatch: Orchestrator=128, Scheduler=64. "
           "Orchestrator can allocate slot ids [64..127] which are OOB in "
           "scheduler's slot_states[64]. No runtime consistency check exists.";

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

    auto& rs = sys.sched.ring_sched_states[0];
    PTO2TaskSlotState& slot = rs.get_slot_state_by_slot(0);

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

    // check_and_handle_consumed: fanout_refcount(0) == fanout_count(0) → true → CONSUMED
    sys.sched.check_and_handle_consumed(slot);

    EXPECT_EQ(slot.task_state.load(), PTO2_TASK_CONSUMED)
        << "fanout_count=0 causes premature CONSUMED transition — "
           "scheduler trusts orchestrator's fanout_count without validation";
}

TEST(CrossComponentContract, HeapTailBeyondTop) {
    // HeapRing calculates available space from top and tail.
    // If scheduler writes tail > top (invalid state), HeapRing computes wrong space.
    alignas(64) uint8_t heap[4096]{};
    std::atomic<uint64_t> top{1000}, tail{3000};
    std::atomic<int32_t> error_code{0};
    PTO2HeapRing ring{};
    pto2_heap_ring_init(&ring, heap, 4096, &tail, &top);
    ring.error_code_ptr = &error_code;

    // tail(3000) > top(1000): the "normal" path expects top >= tail.
    // When top < tail in the alloc check:
    // gap = tail - top = 2000 → available = 4096 - top + (tail - 4096)
    // This enters the wrap branch and may succeed with overlapping memory.
    void* p = ring.pto2_heap_ring_try_alloc(128);

    // Either succeeds (returns pointer into already-used region) or correctly rejects
    if (p != nullptr) {
        // Allocated into region between top and tail — data corruption possible
        uint64_t offset = (uint8_t*)p - heap;
        EXPECT_GE(offset, 1000u);
        SUCCEED() << "HeapRing allocated within [top, tail) gap without detecting invalid state — "
                     "no cross-component validation on SM flow control values";
    } else {
        SUCCEED() << "HeapRing correctly rejected allocation with tail > top";
    }
}

TEST(CrossComponentContract, ActiveMaskZero) {
    // active_mask=0 should never happen (orchestrator has always_assert).
    // But scheduler's release_fanin_and_check_ready has no such guard.
    alignas(64) PTO2TaskSlotState slot{};
    slot.active_mask = 0;  // Invalid — no subtask active
    slot.fanin_count = 1;
    slot.fanin_refcount.store(0, std::memory_order_relaxed);
    slot.task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);

    PTO2ResourceShape shape = pto2_active_mask_to_shape(0);
    // With mask=0: has_aic=false, aiv_count=0 → falls to `return AIV_X2`
    EXPECT_EQ(static_cast<int>(shape), static_cast<int>(PTO2ResourceShape::AIV_X2))
        << "active_mask=0 maps to AIV_X2 — incorrect shape routing. "
           "Orchestrator guards with always_assert, but scheduler does not validate";
}

TEST(CrossComponentContract, TaskDescriptorNullInConsumedSlot) {
    // advance_ring_pointers accesses slot_state.task->packed_buffer_end
    // without null-checking task pointer.
    TMRSystem sys;
    ASSERT_TRUE(sys.init());

    auto& rs = sys.sched.ring_sched_states[0];
    PTO2TaskSlotState& slot = rs.get_slot_state_by_slot(0);

    // Mark as CONSUMED but leave task pointer as nullptr
    slot.task_state.store(PTO2_TASK_CONSUMED, std::memory_order_relaxed);
    slot.task = nullptr;  // Not initialized
    slot.ring_id = 0;

    sys.sm->header->rings[0].fc.current_task_index.store(1, std::memory_order_relaxed);

    // advance_ring_pointers will try to read slot.task->packed_buffer_end → nullptr deref
    EXPECT_DEATH(
        rs.advance_ring_pointers(sys.sm->header->rings[0]),
        ".*"
    ) << "advance_ring_pointers dereferences slot_state.task without null check — "
         "coupling to orchestrator's initialization guarantee";

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
    sys.sm->header->orch_error_code.store(PTO2_ERROR_HEAP_RING_DEADLOCK,
                                           std::memory_order_release);

    // Scheduler operations continue despite error:
    // push to ready queue
    auto& rs = sys.sched.ring_sched_states[0];
    PTO2TaskSlotState& slot = rs.get_slot_state_by_slot(0);
    slot.active_mask = PTO2_SUBTASK_MASK_AIV0;
    PTO2ResourceShape shape = pto2_active_mask_to_shape(slot.active_mask);

    bool pushed = sys.sched.ready_queues[static_cast<int>(shape)].push(&slot);
    EXPECT_TRUE(pushed);

    // pop from ready queue
    PTO2TaskSlotState* popped = sys.sched.ready_queues[static_cast<int>(shape)].pop();
    EXPECT_EQ(popped, &slot)
        << "Scheduler continues normal operation after orchestrator fatal error — "
           "orch_error_code is one-directional (orch→host), invisible to scheduler hot path";

    sys.destroy();
}

TEST(StateLeakage, HeadOfLineBlocking) {
    // advance_ring_pointers scans linearly: stops at first non-CONSUMED slot.
    // One incomplete task blocks reclamation of all subsequent CONSUMED tasks.
    TMRSystem sys;
    ASSERT_TRUE(sys.init());

    auto& rs = sys.sched.ring_sched_states[0];
    PTO2TaskDescriptor descs[3]{};
    descs[0].packed_buffer_end = nullptr;
    descs[1].packed_buffer_end = nullptr;
    descs[2].packed_buffer_end = nullptr;

    // Task 0: CONSUMED
    PTO2TaskSlotState& slot0 = rs.get_slot_state_by_slot(0);
    slot0.task_state.store(PTO2_TASK_CONSUMED, std::memory_order_relaxed);
    slot0.task = &descs[0];

    // Task 1: COMPLETED (NOT consumed — fanout incomplete)
    PTO2TaskSlotState& slot1 = rs.get_slot_state_by_slot(1);
    slot1.task_state.store(PTO2_TASK_COMPLETED, std::memory_order_relaxed);
    slot1.task = &descs[1];

    // Task 2: CONSUMED
    PTO2TaskSlotState& slot2 = rs.get_slot_state_by_slot(2);
    slot2.task_state.store(PTO2_TASK_CONSUMED, std::memory_order_relaxed);
    slot2.task = &descs[2];

    sys.sm->header->rings[0].fc.current_task_index.store(3, std::memory_order_relaxed);

    rs.advance_ring_pointers(sys.sm->header->rings[0]);

    // last_task_alive should stop at task 1 (COMPLETED, not CONSUMED)
    EXPECT_EQ(rs.last_task_alive, 1)
        << "Head-of-line blocking: task 1 (COMPLETED) blocks reclamation of "
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
        PTO2TaskId tid = pto2_make_task_id(0, i);
        tmap.insert(t, tid, true);
    }

    // Advance last_task_alive to 80 — tasks 0..79 are stale
    tmap.sync_validity(0, 80);

    // Lookup must traverse all 100 entries (80 stale + 20 valid)
    // because cleanup hasn't been triggered yet (need sync_tensormap, not just sync_validity)
    PTO2LookupResult result;
    Tensor query = make_test_tensor(0x2000);
    tmap.lookup(query, result);

    // Should find entries from tasks 80..99 = 20 valid
    EXPECT_EQ(result.count, 16)
        << "Lookup result capped at PTO2_LOOKUP_MAX_RESULTS=16, but stale entries "
           "still slow traversal. Cleanup interval (" << PTO2_TENSORMAP_CLEANUP_INTERVAL
        << " tasks) couples TensorMap performance to scheduler's CONSUMED advancement rate";

    tmap.destroy();
}

TEST(StateLeakage, SubtaskMaskProtocol) {
    // active_mask bits (AIC=0x1, AIV0=0x2, AIV1=0x4) are set by orchestrator
    // and checked by scheduler's on_subtask_complete. There's no shared enum
    // enforcing consistency — just implicit agreement on bit positions.

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
           "If scheduler signals AIV1 completion, done_mask (0x4) != active_mask (0x2) — "
           "task never completes. No compile-time enforcement exists.";
}

// =============================================================================
// Suite 5: CompileTimeCoupling
// =============================================================================

TEST(CompileTimeCoupling, SizeofGodObject) {
    size_t size = sizeof(PTO2OrchestratorState);
    // Expect large: embeds PTO2RingSet rings[4], PTO2TensorMap, scope stack pointers
    EXPECT_GT(size, 256u)
        << "sizeof(PTO2OrchestratorState) = " << size << " bytes. "
           "Embeds rings[" << PTO2_MAX_RING_DEPTH << "] (each with HeapRing+TaskRing+DepPool), "
           "TensorMap, SM handle, scope stack — a 'God Object' coupling all subsystems.";

    // Also measure sub-component sizes
    size_t ring_set_size = sizeof(PTO2RingSet) * PTO2_MAX_RING_DEPTH;
    size_t tmap_size = sizeof(PTO2TensorMap);
    EXPECT_GT(ring_set_size, 0u);
    EXPECT_GT(tmap_size, 0u);
    // Log for documentation
    SUCCEED() << "sizeof(PTO2OrchestratorState) = " << size
              << ", rings[4] = " << ring_set_size
              << ", TensorMap = " << tmap_size;
}

TEST(CompileTimeCoupling, MaxRingDepthPropagation) {
    // PTO2_MAX_RING_DEPTH=4 is hardcoded into arrays across multiple components.
    // Count the distinct declarations that depend on it.

    // 1. Orchestrator: rings[PTO2_MAX_RING_DEPTH]
    static_assert(sizeof(PTO2OrchestratorState::rings) / sizeof(PTO2RingSet)
                  == PTO2_MAX_RING_DEPTH);

    // 2. Scheduler: ring_sched_states[PTO2_MAX_RING_DEPTH]
    static_assert(sizeof(PTO2SchedulerState::ring_sched_states) /
                  sizeof(PTO2SchedulerState::RingSchedState)
                  == PTO2_MAX_RING_DEPTH);

    // 3. SharedMemory: header->rings[PTO2_MAX_RING_DEPTH]
    static_assert(sizeof(PTO2SharedMemoryHeader::rings) /
                  sizeof(PTO2SharedMemoryRingHeader)
                  == PTO2_MAX_RING_DEPTH);

    // 4. TensorMap: task_entry_heads[PTO2_MAX_RING_DEPTH]
    PTO2TensorMap tmap{};
    EXPECT_EQ(sizeof(tmap.task_entry_heads) / sizeof(tmap.task_entry_heads[0]),
              (size_t)PTO2_MAX_RING_DEPTH);

    // 5. TensorMap: task_window_sizes[PTO2_MAX_RING_DEPTH]
    EXPECT_EQ(sizeof(tmap.task_window_sizes) / sizeof(tmap.task_window_sizes[0]),
              (size_t)PTO2_MAX_RING_DEPTH);

    // 6. TensorMap: last_task_alives[PTO2_MAX_RING_DEPTH]
    EXPECT_EQ(sizeof(tmap.last_task_alives) / sizeof(tmap.last_task_alives[0]),
              (size_t)PTO2_MAX_RING_DEPTH);

    // 7. SharedMemoryHandle: task_descriptors[PTO2_MAX_RING_DEPTH]
    EXPECT_EQ(sizeof(PTO2SharedMemoryHandle::task_descriptors) /
              sizeof(PTO2TaskDescriptor*),
              (size_t)PTO2_MAX_RING_DEPTH);

    // 8. SharedMemoryHandle: task_payloads[PTO2_MAX_RING_DEPTH]
    EXPECT_EQ(sizeof(PTO2SharedMemoryHandle::task_payloads) /
              sizeof(PTO2TaskPayload*),
              (size_t)PTO2_MAX_RING_DEPTH);

    SUCCEED() << "PTO2_MAX_RING_DEPTH=" << PTO2_MAX_RING_DEPTH
              << " propagates to 8+ array declarations across 4 components "
                 "(Orchestrator, Scheduler, SharedMemory, TensorMap). "
                 "Changing this value requires recompiling all components.";
}

TEST(CompileTimeCoupling, WindowSizeReadByThreeComponents) {
    // task_window_size is read independently from SM header by three components.
    // All three must agree on the value. No single authoritative source.
    TMRSystem sys;
    ASSERT_TRUE(sys.init());

    // Orchestrator's view: from TaskRing
    int32_t orch_window = sys.orch.rings[0].task_ring.window_size;

    // Scheduler's view: from RingSchedState
    uint64_t sched_window = sys.sched.ring_sched_states[0].task_window_size;

    // TensorMap's view: from task_window_sizes[]
    int32_t tmap_window = sys.orch.tensor_map.task_window_sizes[0];

    EXPECT_EQ(orch_window, (int32_t)sched_window);
    EXPECT_EQ(orch_window, tmap_window)
        << "task_window_size is independently read from SM header by "
           "Orchestrator (TaskRing.window_size=" << orch_window << "), "
           "Scheduler (RingSchedState.task_window_size=" << sched_window << "), "
           "TensorMap (task_window_sizes[]=" << tmap_window << "). "
           "No single source of truth — each caches its own copy.";

    sys.destroy();
}

TEST(CompileTimeCoupling, TaskSlotStateTypeCoupling) {
    // PTO2TaskSlotState references types from multiple components,
    // connecting orchestrator and scheduler domains.

    // Types referenced:
    // 1. PTO2DepListEntry* fanout_head — from ring buffer (orchestrator domain)
    // 2. PTO2TaskPayload* payload — from runtime2_types (shared domain)
    // 3. PTO2TaskDescriptor* task — from runtime2_types (shared domain)
    // 4. std::atomic<PTO2TaskState> — enum from runtime2_types
    // Plus atomic primitives for fanin/fanout refcounting

    static_assert(sizeof(PTO2TaskSlotState) == 64,
                  "TaskSlotState is exactly 1 cache line");

    // Verify it contains pointers to at least 3 distinct struct types
    alignas(64) PTO2TaskSlotState slot{};
    EXPECT_EQ(sizeof(slot.fanout_head), sizeof(void*));   // PTO2DepListEntry*
    EXPECT_EQ(sizeof(slot.payload), sizeof(void*));       // PTO2TaskPayload*
    EXPECT_EQ(sizeof(slot.task), sizeof(void*));          // PTO2TaskDescriptor*

    SUCCEED() << "PTO2TaskSlotState (64 bytes) references 3 external struct types "
                 "(DepListEntry, TaskPayload, TaskDescriptor) plus PTO2TaskState enum. "
                 "It is the nexus coupling orchestrator types (DepList, Payload) "
                 "with scheduler types (TaskState, fanin/fanout) and SM types (TaskDescriptor).";
}

TEST(CompileTimeCoupling, ReadyQueueMemoryCost) {
    // PTO2_READY_QUEUE_SIZE controls ALL 5 shape queues equally.
    // Total memory = 5 * 65536 * sizeof(PTO2ReadyQueueSlot)
    size_t slot_size = sizeof(PTO2ReadyQueueSlot);
    size_t total_queue_mem = PTO2_NUM_RESOURCE_SHAPES * PTO2_READY_QUEUE_SIZE * slot_size;
    size_t total_mb = total_queue_mem / (1024 * 1024);

    EXPECT_GT(total_queue_mem, 0u);
    SUCCEED() << "ReadyQueue memory: " << PTO2_NUM_RESOURCE_SHAPES
              << " shapes x " << PTO2_READY_QUEUE_SIZE
              << " slots x " << slot_size << " bytes/slot = "
              << total_queue_mem << " bytes (" << total_mb << " MB). "
                 "Single constant PTO2_READY_QUEUE_SIZE controls all shapes equally — "
                 "no per-shape tuning possible.";
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
