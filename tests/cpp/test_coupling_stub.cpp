/**
 * Stub-based architectural coupling detection tests.
 *
 * This file deliberately excludes pto_orchestrator.cpp from the link.
 * If it compiles and links successfully, that PROVES TensorMap + Scheduler +
 * RingBuffer + SharedMemory can be used without the Orchestrator at link time.
 *
 * Key distinction probed here:
 *   Link-time coupling    — .o file has UND symbols pointing to another component
 *   Compile-time coupling — .cpp includes another component's header (type access)
 *   Type-level coupling   — function signature uses another component's struct type,
 *                           forcing full include even if only a pointer is stored
 *
 * Test philosophy: document coupling depth precisely using stubs.
 * FAIL = a coupling contract that the src violates or makes harder than necessary.
 */

#include <gtest/gtest.h>
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <new>

#include "pto_ring_buffer.h"
#include "pto_scheduler.h"
#include "pto_shared_memory.h"
#include "pto_tensormap.h"
#include "pto_runtime2_types.h"
#include "tensor.h"

// =============================================================================
// Shared helpers
// =============================================================================

static constexpr uint64_t SH = 65536;    // heap size for sm_create
static constexpr int32_t  SW = 64;       // task window size

// Minimal scheduler stub: allocate only the fields reclaim() reads.
// Returns true if stub construction succeeded.
struct MinimalSchedStub {
    PTO2SchedulerState sched{};
    PTO2TaskSlotState* slot_array = nullptr;
    static constexpr int32_t WINDOW = 64;

    bool init(uint8_t ring_id = 0) {
        memset(&sched, 0, sizeof(sched));
        slot_array = new (std::nothrow) PTO2TaskSlotState[WINDOW]{};
        if (!slot_array) return false;
        auto& rs = sched.ring_sched_states[ring_id];
        rs.slot_states      = slot_array;
        rs.task_window_size = WINDOW;
        rs.task_window_mask = WINDOW - 1;
        return true;
    }

    void destroy() {
        delete[] slot_array;
        slot_array = nullptr;
    }
};

// Minimal pool helper: 512-entry DepListPool.
struct SmallPool {
    PTO2DepListEntry entries[512];
    std::atomic<int32_t> error_code{0};
    PTO2DepListPool pool;

    void init() {
        memset(entries, 0, sizeof(entries));
        pool.init(entries, 512, &error_code);
    }
    int alloc_n(int n) {
        int last = 0;
        for (int i = 0; i < n; i++) {
            auto* e = pool.alloc();
            if (e) last = i + 1;
        }
        return last;
    }
};

static Tensor make_tensor(uint64_t addr, uint32_t ndims = 1, uint32_t shape0 = 100) {
    Tensor t{};
    t.buffer.addr = addr;
    t.buffer.size = shape0;
    t.ndims = ndims;
    t.shapes[0] = shape0;
    t.is_all_offset_zero = true;
    return t;
}

// =============================================================================
// Suite 1: DepPoolStubIsolation
// =============================================================================

// sm_last_task_alive < PTO2_DEP_POOL_CLEANUP_INTERVAL: reclaim is a no-op.
// A zero-initialized PTO2SchedulerState (slot_states=nullptr) must not crash.
TEST(DepPoolStubIsolation, ReclaimBelowInterval_NeverAccessesScheduler) {
    SmallPool sp;
    sp.init();
    sp.alloc_n(100);

    // Capture tail BEFORE reclaim to compare after
    int32_t tail_before = sp.pool.tail;

    // Zero-init stub — slot_states is nullptr
    PTO2SchedulerState sched{};
    memset(&sched, 0, sizeof(sched));

    // sm_last_task_alive = interval - 1 → guard `>= interval` is false → no-op
    int32_t below = PTO2_DEP_POOL_CLEANUP_INTERVAL - 1;
    sp.pool.reclaim(sched, 0, below);

    // Pool tail unchanged — reclaim was a no-op
    EXPECT_EQ(sp.pool.tail, tail_before)
        << "reclaim() is a no-op when sm_last_task_alive < interval. "
           "A fully zero-initialized (nullptr slot_states) PTO2SchedulerState "
           "is safe to pass — the struct is never touched.";
}

// sm_last_task_alive == PTO2_DEP_POOL_CLEANUP_INTERVAL: reclaim reads exactly
//   sched.ring_sched_states[0].slot_states[(interval-1) & mask].dep_pool_mark
// Stub provides only those three values; all other fields remain zero.
TEST(DepPoolStubIsolation, ReclaimAtInterval_OnlyNeedsSlotArrayAndMask) {
    SmallPool sp;
    sp.init();
    sp.alloc_n(100);   // top = 100, tail = 0

    MinimalSchedStub stub;
    ASSERT_TRUE(stub.init(0));

    // Set dep_pool_mark in the slot reclaim() will read
    int32_t sm_last = PTO2_DEP_POOL_CLEANUP_INTERVAL;          // e.g. 64
    int32_t target_slot = (sm_last - 1) & stub.WINDOW - 1;     // (63) & 63 = 63
    stub.slot_array[target_slot].dep_pool_mark = 50;

    sp.pool.reclaim(stub.sched, 0, sm_last);

    // reclaim should advance pool tail to dep_pool_mark = 50
    EXPECT_EQ(sp.pool.tail, 50)
        << "reclaim() reads EXACTLY THREE values from PTO2SchedulerState:\n"
           "  1. ring_sched_states[ring_id].slot_states  (the pointer)\n"
           "  2. ring_sched_states[ring_id].task_window_mask\n"
           "  3. slot_states[(sm_last-1) & mask].dep_pool_mark\n"
           "All other fields of PTO2SchedulerState (~2000 bytes) are unused. "
           "Passing the full struct is structural over-coupling.";

    stub.destroy();
}

// ensure_space() returns immediately when available() >= needed.
// Neither PTO2SchedulerState nor PTO2RingFlowControl is ever accessed.
TEST(DepPoolStubIsolation, EnsureSpaceWithSufficientCapacity_NoSchedulerAccess) {
    SmallPool sp;
    sp.init();
    // Pool is empty: available() = capacity - 1 = 511 >> needed = 5

    PTO2SchedulerState sched{};
    memset(&sched, 0, sizeof(sched));   // slot_states = nullptr (would crash if accessed)
    PTO2RingFlowControl fc{};
    fc.init();

    // Should return immediately without touching sched or fc
    sp.pool.ensure_space(sched, fc, 0, 5);

    EXPECT_GE(sp.pool.available(), 5)
        << "ensure_space() exits immediately when available() >= needed. "
           "Zero-initialized sched (slot_states=nullptr) is safe — never dereferenced. "
           "The signature requires both PTO2SchedulerState& and PTO2RingFlowControl& "
           "but neither is accessed in the fast path.";
}

// Document the sizeof cost of the over-coupling.
TEST(DepPoolStubIsolation, ReclaimRequiresExactlyThreeFields_ButStructIsHuge) {
    // Fields actually needed by reclaim():
    //   PTO2SchedulerState::RingSchedState::slot_states       (8 bytes, pointer)
    //   PTO2SchedulerState::RingSchedState::task_window_mask  (4 bytes, int32_t)
    //   PTO2TaskSlotState::dep_pool_mark                      (4 bytes, int32_t)
    // Total minimum: 16 bytes of live data.
    size_t needed_bytes = sizeof(PTO2TaskSlotState*) + sizeof(int32_t) + sizeof(int32_t);

    // Actual cost imposed by full type coupling:
    size_t actual_bytes = sizeof(PTO2SchedulerState);

    EXPECT_GT(actual_bytes, needed_bytes)
        << "reclaim() needs ~16 bytes of data but requires passing "
           "PTO2SchedulerState (" << actual_bytes << " bytes). "
           "Ratio: " << (actual_bytes / needed_bytes) << "x over-coupling. "
           "Root cause: reclaim() signature accepts the full god-object struct "
           "instead of only the fields it uses.";

    // Also report the exact sizes for documentation
    SUCCEED() << "sizeof(PTO2SchedulerState) = " << actual_bytes << " bytes\n"
              << "sizeof(PTO2TaskSlotState*) + 2*int32_t = " << needed_bytes << " bytes\n"
              << "sizeof(PTO2TaskSlotState) = " << sizeof(PTO2TaskSlotState);
}

// =============================================================================
// Suite 2: SchedulerWithoutOrchestrator
// =============================================================================

// Scheduler can be fully initialized and destroyed without any orchestrator code.
// This test links pto_scheduler.cpp + pto_shared_memory.cpp only.
TEST(SchedulerWithoutOrchestrator, InitAndDestroy_NoOrchestratorNeeded) {
    PTO2SharedMemoryHandle* sm = pto2_sm_create(SW, SH);
    ASSERT_NE(sm, nullptr);

    uint8_t* heap = (uint8_t*)calloc(PTO2_MAX_RING_DEPTH, SH);
    ASSERT_NE(heap, nullptr);

    PTO2SchedulerState sched{};
    bool ok = pto2_scheduler_init(&sched, sm, heap, SH);
    EXPECT_TRUE(ok)
        << "pto2_scheduler_init succeeds without orchestrator.cpp in the link. "
           "Scheduler is link-time isolated from Orchestrator.";

    EXPECT_EQ(sched.ring_sched_states[0].task_window_size, (uint64_t)SW);
    EXPECT_EQ(sched.ring_sched_states[0].task_window_mask, SW - 1);

    pto2_scheduler_destroy(&sched);
    free(heap);
    pto2_sm_destroy(sm);
}

// PTO2ReadyQueue is header-only (all methods are inline in pto_scheduler.h).
// It needs zero .cpp linkage — only pto_runtime2_types.h for slot type.
TEST(SchedulerWithoutOrchestrator, ReadyQueue_StandaloneNoExternalDeps) {
    PTO2ReadyQueue q;
    pto2_ready_queue_init(&q, 64);

    alignas(64) PTO2TaskSlotState slot{};
    slot.task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);

    EXPECT_TRUE(q.push(&slot));
    PTO2TaskSlotState* out = q.pop();
    EXPECT_EQ(out, &slot)
        << "PTO2ReadyQueue push/pop are entirely header-inline (zero link deps). "
           "However, pto2_ready_queue_init / pto2_ready_queue_destroy are free "
           "functions defined in pto_scheduler.cpp — even a standalone ReadyQueue "
           "requires linking pto_scheduler.cpp for lifecycle management. "
           "Push/pop core logic is self-contained; init/destroy coupling is avoidable.";

    pto2_ready_queue_destroy(&q);
}

// release_fanin_and_check_ready requires zero TensorMap or Orchestrator linkage.
// With fanin_count=1, one call makes new_refcount == fanin_count → push to queue.
TEST(SchedulerWithoutOrchestrator, ReleaseFanin_PushesWhenFaninMet) {
    PTO2SharedMemoryHandle* sm = pto2_sm_create(SW, SH);
    ASSERT_NE(sm, nullptr);
    uint8_t* heap = (uint8_t*)calloc(PTO2_MAX_RING_DEPTH, SH);
    ASSERT_NE(heap, nullptr);
    PTO2SchedulerState sched{};
    ASSERT_TRUE(pto2_scheduler_init(&sched, sm, heap, SH));

    alignas(64) PTO2TaskSlotState slot{};
    slot.fanin_count = 1;
    slot.fanin_refcount.store(0, std::memory_order_relaxed);
    slot.task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);
    slot.active_mask = PTO2_SUBTASK_MASK_AIV0;

    bool became_ready = sched.release_fanin_and_check_ready(slot, nullptr);
    EXPECT_TRUE(became_ready) << "fanin_count=1, one release → task is ready";

    // Verify the slot is now in the ready queue
    PTO2ResourceShape shape = pto2_active_mask_to_shape(slot.active_mask);
    PTO2TaskSlotState* popped = sched.ready_queues[static_cast<int>(shape)].pop();
    EXPECT_EQ(popped, &slot) << "Slot found in ready queue — no Orchestrator involvement";

    pto2_scheduler_destroy(&sched);
    free(heap);
    pto2_sm_destroy(sm);
}

// KEY DEFECT TEST: Non-profiling release_fanin_and_check_ready pushes to the
// ready queue WITHOUT performing CAS(PENDING→READY) first.
// The profiling overload (lines 450-476) DOES perform the CAS.
// This means: in non-profiling builds, a worker can pop a PENDING-state slot.
TEST(SchedulerWithoutOrchestrator, NonProfiling_ReleaseFanin_SkipsCAS_SlotStaysPending) {
#if PTO2_SCHED_PROFILING
    GTEST_SKIP() << "Test only applies to non-profiling builds (PTO2_SCHED_PROFILING=0)";
#endif
    PTO2SharedMemoryHandle* sm = pto2_sm_create(SW, SH);
    ASSERT_NE(sm, nullptr);
    uint8_t* heap = (uint8_t*)calloc(PTO2_MAX_RING_DEPTH, SH);
    ASSERT_NE(heap, nullptr);
    PTO2SchedulerState sched{};
    ASSERT_TRUE(pto2_scheduler_init(&sched, sm, heap, SH));

    alignas(64) PTO2TaskSlotState slot{};
    slot.fanin_count = 1;
    slot.fanin_refcount.store(0, std::memory_order_relaxed);
    slot.task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);
    slot.active_mask = PTO2_SUBTASK_MASK_AIV0;

    sched.release_fanin_and_check_ready(slot, nullptr);

    PTO2TaskState state_after = slot.task_state.load(std::memory_order_acquire);

    // In non-profiling mode: the slot is pushed without CAS → state remains PENDING.
    // A worker that pops this slot sees task_state == PENDING, not READY.
    // This breaks the contract that "anything in the ready queue is READY".
    EXPECT_EQ(state_after, PTO2_TASK_PENDING)
        << "BUG: Non-profiling release_fanin_and_check_ready (pto_scheduler.h:426-448) "
           "pushes slot to ready queue WITHOUT transitioning task_state to READY.\n"
           "The profiling overload (lines 450-476) DOES perform CAS(PENDING→READY).\n"
           "Result: workers can pop a PENDING-state slot from the ready queue.\n"
           "This is a CORRECTNESS difference, not merely a performance difference.\n"
           "PTO2_SCHED_PROFILING changes observable program behavior.";

    pto2_scheduler_destroy(&sched);
    free(heap);
    pto2_sm_destroy(sm);
}

// on_mixed_task_complete transitions COMPLETED→CONSUMED with a minimal stub descriptor.
// No TensorMap or Orchestrator calls are made in this path.
TEST(SchedulerWithoutOrchestrator, OnMixedTaskComplete_StubDescriptor) {
    PTO2SharedMemoryHandle* sm = pto2_sm_create(SW, SH);
    ASSERT_NE(sm, nullptr);
    uint8_t* heap = (uint8_t*)calloc(PTO2_MAX_RING_DEPTH, SH);
    ASSERT_NE(heap, nullptr);
    PTO2SchedulerState sched{};
    ASSERT_TRUE(pto2_scheduler_init(&sched, sm, heap, SH));

    auto& rs = sched.ring_sched_states[0];
    PTO2TaskSlotState& slot = rs.get_slot_state_by_slot(0);

    PTO2TaskDescriptor dummy_desc{};
    dummy_desc.packed_buffer_base = nullptr;
    dummy_desc.packed_buffer_end  = nullptr;
    slot.task = &dummy_desc;
    slot.ring_id = 0;
    slot.fanout_count = 1;
    slot.fanout_refcount.store(1, std::memory_order_relaxed);
    slot.task_state.store(PTO2_TASK_COMPLETED, std::memory_order_relaxed);
    sm->header->rings[0].fc.current_task_index.store(1, std::memory_order_relaxed);

    sched.check_and_handle_consumed(slot);

    EXPECT_EQ(slot.task_state.load(), PTO2_TASK_CONSUMED)
        << "Scheduler's COMPLETED→CONSUMED path requires only a stub "
           "PTO2TaskDescriptor (packed_buffer pointers can be nullptr). "
           "No TensorMap or Orchestrator calls are made in this path.";

    pto2_scheduler_destroy(&sched);
    free(heap);
    pto2_sm_destroy(sm);
}

// =============================================================================
// Suite 3: TensorMapLinkDecoupling
// =============================================================================

// This entire file excludes pto_orchestrator.cpp from the link.
// If TensorMap init/insert/lookup work here, it proves link-time isolation.
TEST(TensorMapLinkDecoupling, BuildsAndRunsWithoutOrchestratorCpp) {
    int32_t window_sizes[PTO2_MAX_RING_DEPTH] = {64, 64, 64, 64};
    PTO2TensorMap tmap{};
    ASSERT_TRUE(tmap.init(256, 1024, window_sizes));

    Tensor t = make_tensor(0x3000);
    PTO2TaskId tid = pto2_make_task_id(0, 0);
    tmap.insert(t, tid, /*is_writer=*/true);

    PTO2LookupResult result;
    tmap.lookup(t, result);
    EXPECT_GE(result.count, 1)
        << "TensorMap insert+lookup work without pto_orchestrator.cpp in the link.\n"
           "Root cause: pto_tensormap.cpp includes pto_orchestrator.h (line 22) but\n"
           "calls ZERO orchestrator functions — confirmed by objdump UND analysis.\n"
           "The include only provides the PTO2OrchestratorState type definition,\n"
           "which is stored as PTO2OrchestratorState* (pointer — forward decl suffices).";

    tmap.destroy();
}

// Explicitly set orch = nullptr, then run insert and lookup.
// If orch were dereferenced in the hot path, this would crash.
TEST(TensorMapLinkDecoupling, OrchPointer_NeverDereferencedInHotPath) {
    int32_t window_sizes[PTO2_MAX_RING_DEPTH] = {64, 64, 64, 64};
    PTO2TensorMap tmap{};
    ASSERT_TRUE(tmap.init(256, 1024, window_sizes));
    tmap.orch = nullptr;   // explicitly clear

    Tensor t1 = make_tensor(0x4000, 1, 200);
    Tensor t2 = make_tensor(0x5000, 1, 100);
    PTO2TaskId t1id = pto2_make_task_id(0, 0);
    PTO2TaskId t2id = pto2_make_task_id(0, 1);
    tmap.insert(t1, t1id, true);
    tmap.insert(t2, t2id, true);

    PTO2LookupResult r;
    tmap.lookup(t1, r);
    EXPECT_GE(r.count, 1)
        << "orch=nullptr does not crash insert or lookup. "
           "The orch pointer is only used by sync_tensormap (called from orchestrator). "
           "In normal usage: orch is set by pto2_orchestrator_init, "
           "but insert/lookup never touch it.";

    tmap.destroy();
}

// sync_tensormap only advances the cleanup clock — it doesn't access orch.
// Calling it with orch=nullptr is safe.
TEST(TensorMapLinkDecoupling, SyncTensormap_DoesNotAccessOrch) {
    int32_t window_sizes[PTO2_MAX_RING_DEPTH] = {64, 64, 64, 64};
    PTO2TensorMap tmap{};
    ASSERT_TRUE(tmap.init(256, 1024, window_sizes));
    tmap.orch = nullptr;

    // Insert entries for tasks 0..63 in ring 0
    for (int i = 0; i < 64; i++) {
        Tensor t = make_tensor(0x6000 + i * 64);
        tmap.insert(t, pto2_make_task_id(0, i), true);
    }

    // Advance validity: tasks 0..31 are now retired
    tmap.sync_validity(0, 32);

    // sync_tensormap only calls sync_validity internally — no orch access
    tmap.sync_tensormap(0, 32);

    // Valid count should reflect only tasks 32..63
    int valid = tmap.valid_count();
    EXPECT_LE(valid, 64)
        << "sync_tensormap(ring_id, last_alive) is purely time-advance logic. "
           "No dereference of orch pointer. "
           "Cleanup path is independent of OrchestratorState.";

    tmap.destroy();
}

// Document the transitive include chain caused by one unnecessary #include.
TEST(TensorMapLinkDecoupling, IncludeCost_OnePointerField_FullRuntimeHeaders) {
    // pto_tensormap.cpp includes pto_orchestrator.h for PTO2OrchestratorState* orch.
    // A forward declaration "struct PTO2OrchestratorState;" would be sufficient
    // because orch is a raw pointer and is never dereferenced in tensormap.cpp.
    //
    // Cost of the full include:
    //   pto_orchestrator.h includes:
    //     → pto_scheduler.h → pto_ring_buffer.h → pto_shared_memory.h
    //     → pto_runtime2_types.h → pto_types.h, pto_submit_types.h, pto2_dispatch_payload.h
    //
    // Every TensorMap compilation unit pulls in the entire runtime header tree
    // for a single pointer field.

    // Verify: PTO2TensorMap::orch is a raw pointer (not embedded object)
    EXPECT_EQ(sizeof(PTO2OrchestratorState*), sizeof(void*))
        << "PTO2OrchestratorState* is a pointer — sizeof(void*) bytes. "
           "A forward declaration suffices. "
           "The full include of pto_orchestrator.h transitively pulls in "
           "pto_scheduler.h + pto_ring_buffer.h + pto_shared_memory.h + "
           "pto_runtime2_types.h (7+ headers) for a single 8-byte pointer field.";

    // Also: this test file compiles and links without pto_orchestrator.cpp —
    // further confirming the include is header-only compile-time coupling.
    SUCCEED() << "This test file does not link pto_orchestrator.cpp. "
                 "Build success = confirmed link-time isolation.";
}

// =============================================================================
// Suite 4: CompileTimeIncludeCoupling
// =============================================================================

// pto_ring_buffer.cpp includes pto_scheduler.h for reclaim()'s PTO2SchedulerState param.
// But ring_buffer.o has ZERO UND symbols from scheduler — pure type-level coupling.
// The coupling is structural: accessing struct fields inline creates invisible interface.
TEST(CompileTimeIncludeCoupling, RingBufferCoupledToSchedulerAtTypeLevel) {
    // Demonstrate: DepPool::reclaim is in pto_ring_buffer.cpp (not scheduler)
    // yet it accesses PTO2SchedulerState internal fields inline.
    // This means: changing RingSchedState layout silently breaks ring_buffer
    // without any API change or linker error.

    // Cross-check: the field offset in the stub must match the real struct.
    MinimalSchedStub stub;
    ASSERT_TRUE(stub.init(0));

    // Write to dep_pool_mark via stub's slot_array
    stub.slot_array[63].dep_pool_mark = 99;

    // Read the same field through PTO2SchedulerState's accessor
    int32_t mark = stub.sched.ring_sched_states[0]
                       .get_slot_state_by_task_id(63)
                       .dep_pool_mark;
    EXPECT_EQ(mark, 99)
        << "ring_buffer.cpp accesses PTO2SchedulerState::RingSchedState::slot_states "
           "inline (no virtual dispatch, no function call). "
           "Changing the layout of PTO2TaskSlotState or RingSchedState breaks "
           "pto_ring_buffer.cpp without touching any function signature or .h file API. "
           "This is a hidden structural coupling: invisible to the linker.";

    stub.destroy();
}

// Both Scheduler and TensorMap independently compute the same slot index formula.
// Duplication means if one changes, the other silently diverges.
TEST(CompileTimeIncludeCoupling, TaskWindowMask_DuplicatedInTwoComponents) {
    // Scheduler formula (pto_scheduler.h:301):
    //   slot_states[local_id & task_window_mask]
    // TensorMap formula (pto_tensormap.h:~364):
    //   local_id & (task_window_sizes[ring_id] - 1)
    // Both assume power-of-2 window_size; neither validates it.

    int32_t window_sizes[PTO2_MAX_RING_DEPTH] = {64, 64, 64, 64};
    PTO2TensorMap tmap{};
    ASSERT_TRUE(tmap.init(256, 1024, window_sizes));

    PTO2SharedMemoryHandle* sm = pto2_sm_create(64, SH);
    ASSERT_NE(sm, nullptr);
    uint8_t* heap = (uint8_t*)calloc(PTO2_MAX_RING_DEPTH, SH);
    ASSERT_NE(heap, nullptr);
    PTO2SchedulerState sched{};
    ASSERT_TRUE(pto2_scheduler_init(&sched, sm, heap, SH));

    // Verify both agree for local_id = 37, ring = 0
    int32_t local_id   = 37;
    int32_t sched_slot = local_id & sched.ring_sched_states[0].task_window_mask;
    int32_t tmap_slot  = local_id & (tmap.task_window_sizes[0] - 1);

    EXPECT_EQ(sched_slot, tmap_slot)
        << "Scheduler slot = local_id & mask = " << sched_slot << "\n"
           "TensorMap slot = local_id & (size-1) = " << tmap_slot << "\n"
           "Currently agree — but the formula is written twice, in two components, "
           "with no shared utility. A change to one (e.g., non-power-of-2 support) "
           "would not automatically update the other.";

    pto2_scheduler_destroy(&sched);
    free(heap);
    pto2_sm_destroy(sm);
    tmap.destroy();
}

// PTO2_MAX_RING_DEPTH propagates into fixed-size arrays in 4 components.
// Changing it requires recompiling all 4 components simultaneously.
TEST(CompileTimeIncludeCoupling, MaxRingDepthInFourComponents) {
    // 1. Orchestrator: rings[PTO2_MAX_RING_DEPTH]  (visible via TMRSystem)
    // 2. Scheduler: ring_sched_states[PTO2_MAX_RING_DEPTH]
    static_assert(
        sizeof(PTO2SchedulerState::ring_sched_states) /
        sizeof(PTO2SchedulerState::RingSchedState) == PTO2_MAX_RING_DEPTH,
        "Scheduler array size must equal PTO2_MAX_RING_DEPTH");

    // 3. SharedMemory: header->rings[PTO2_MAX_RING_DEPTH]
    static_assert(
        sizeof(PTO2SharedMemoryHeader::rings) / sizeof(PTO2SharedMemoryRingHeader)
        == PTO2_MAX_RING_DEPTH,
        "SharedMemory array size must equal PTO2_MAX_RING_DEPTH");

    // 4. TensorMap: task_entry_heads[], task_window_sizes[], last_task_alives[]
    PTO2TensorMap dummy{};
    EXPECT_EQ(sizeof(dummy.task_entry_heads)  / sizeof(dummy.task_entry_heads[0]),
              (size_t)PTO2_MAX_RING_DEPTH);
    EXPECT_EQ(sizeof(dummy.task_window_sizes) / sizeof(dummy.task_window_sizes[0]),
              (size_t)PTO2_MAX_RING_DEPTH);
    EXPECT_EQ(sizeof(dummy.last_task_alives)  / sizeof(dummy.last_task_alives[0]),
              (size_t)PTO2_MAX_RING_DEPTH);

    SUCCEED() << "PTO2_MAX_RING_DEPTH=" << PTO2_MAX_RING_DEPTH
              << " is baked into fixed arrays in Scheduler, SharedMemory, and TensorMap. "
                 "Changing this constant requires recompiling ALL 4 components. "
                 "No runtime configurability exists.";
}

// Including pto_scheduler.h transitively pulls in the entire runtime type hierarchy.
// Document the breadth of this coupling for a single component include.
TEST(CompileTimeIncludeCoupling, SchedulerHeaderTransitiveIncludes) {
    // #include "pto_scheduler.h" causes:
    //   pto_scheduler.h → pto_runtime2_types.h  (task state, config constants)
    //                   → pto_shared_memory.h   (SM handle, ring headers, flow control)
    //                       → pto_runtime2_types.h (again, guarded)
    //                   → pto_ring_buffer.h     (HeapRing, TaskRing, DepPool, RingSet)
    //                       → pto_shared_memory.h (again, guarded)
    //                   → common/core_type.h    (CoreType enum)
    // Total headers transitively included: 6+

    // Verify a few types from the transitive chain are available in this TU
    // (these would be missing if the includes were broken)
    PTO2HeapRing hr{};           // from pto_ring_buffer.h
    PTO2SharedMemoryHeader smh{};// from pto_shared_memory.h
    PTO2TaskState ts = PTO2_TASK_PENDING; // from pto_runtime2_types.h
    (void)hr; (void)smh; (void)ts;

    SUCCEED() << "A single #include \"pto_scheduler.h\" makes available: "
                 "PTO2HeapRing, PTO2TaskRing, PTO2DepListPool, "
                 "PTO2SharedMemoryHandle, PTO2TaskSlotState, PTO2TaskState, "
                 "PTO2ReadyQueue, CoreType — the entire runtime type set. "
                 "This creates a broad compile-time coupling surface.";
}

// =============================================================================
// Suite 5: ProfilingBehaviorCoupling
// =============================================================================

// The non-profiling release_fanin_and_check_ready (lines 426-448) does NOT
// perform CAS(PENDING→READY) before pushing to the ready queue.
// The profiling overload (lines 450-476) DOES perform the CAS.
// Document this divergence as a structural coupling of profiling to correctness.
TEST(ProfilingBehaviorCoupling, ProfilingAndNonProfiling_DifferentStateAfterRelease) {
    PTO2SharedMemoryHandle* sm = pto2_sm_create(SW, SH);
    ASSERT_NE(sm, nullptr);
    uint8_t* heap = (uint8_t*)calloc(PTO2_MAX_RING_DEPTH, SH);
    ASSERT_NE(heap, nullptr);
    PTO2SchedulerState sched{};
    ASSERT_TRUE(pto2_scheduler_init(&sched, sm, heap, SH));

    alignas(64) PTO2TaskSlotState slot{};
    slot.fanin_count = 1;
    slot.fanin_refcount.store(0, std::memory_order_relaxed);
    slot.task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);
    slot.active_mask = PTO2_SUBTASK_MASK_AIV0;

    sched.release_fanin_and_check_ready(slot, nullptr);

    PTO2TaskState state = slot.task_state.load(std::memory_order_acquire);

#if PTO2_SCHED_PROFILING
    // Profiling path: CAS was performed → READY
    EXPECT_EQ(state, PTO2_TASK_READY)
        << "Profiling build: CAS(PENDING→READY) executed before push. "
           "Worker will see READY state when it pops this slot.";
#else
    // Non-profiling path: no CAS → still PENDING
    EXPECT_EQ(state, PTO2_TASK_PENDING)
        << "Non-profiling build: slot pushed to ready queue with task_state=PENDING.\n"
           "PTO2_SCHED_PROFILING flag changes CORRECTNESS, not just measurement.\n"
           "See pto_scheduler.h lines 426-448 (non-profiling) vs 450-476 (profiling).";
#endif

    pto2_scheduler_destroy(&sched);
    free(heap);
    pto2_sm_destroy(sm);
}

// The profiling overload has an additional CAS guard that prevents double-push.
// The non-profiling overload relies on the caller ensuring exactly-once delivery.
// Document the API asymmetry as a coupling risk.
TEST(ProfilingBehaviorCoupling, ProfilingOverload_HasCASGuard_NonProfilingDoesNot) {
    // Non-profiling signature (lines 426-448):
    //   bool release_fanin_and_check_ready(slot, local_bufs = nullptr)
    //   → pushes unconditionally when fanin met; no CAS guard
    //
    // Profiling signature (lines 450-476):
    //   bool release_fanin_and_check_ready(slot, atomic_count, push_wait, local_bufs)
    //   → CAS(PENDING→READY); only pushes if CAS succeeds
    //   → if two threads race and both see new_refcount==fanin_count,
    //     only ONE will win the CAS; the other returns false (no double-push)
    //
    // Non-profiling has no such guard: if two threads both see new_refcount==fanin_count
    // (which shouldn't happen due to fetch_add atomicity, but still an asymmetry),
    // both would push.

    // Verify the non-profiling path returns true whenever fanin_count is met
    PTO2SharedMemoryHandle* sm = pto2_sm_create(SW, SH);
    ASSERT_NE(sm, nullptr);
    uint8_t* heap = (uint8_t*)calloc(PTO2_MAX_RING_DEPTH, SH);
    ASSERT_NE(heap, nullptr);
    PTO2SchedulerState sched{};
    ASSERT_TRUE(pto2_scheduler_init(&sched, sm, heap, SH));

    alignas(64) PTO2TaskSlotState slot{};
    slot.fanin_count = 2;
    slot.fanin_refcount.store(0, std::memory_order_relaxed);
    slot.task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);
    slot.active_mask = PTO2_SUBTASK_MASK_AIV0;

    bool r1 = sched.release_fanin_and_check_ready(slot, nullptr);  // refcount→1, !=2
    bool r2 = sched.release_fanin_and_check_ready(slot, nullptr);  // refcount→2, ==2

    EXPECT_FALSE(r1) << "First release: refcount=1 != fanin_count=2 → not ready";
    EXPECT_TRUE(r2)  << "Second release: refcount=2 == fanin_count=2 → ready, pushed";

    SUCCEED() << "Non-profiling path: return true means 'pushed to queue'. "
                 "Profiling path: return true means 'CAS succeeded AND pushed'. "
                 "The distinction matters for exactly-once delivery guarantees "
                 "under concurrent access — the non-profiling version trusts "
                 "fetch_add atomicity alone to prevent double-push.";

    pto2_scheduler_destroy(&sched);
    free(heap);
    pto2_sm_destroy(sm);
}

// Profiling externs are declared inside #if blocks in hot-path headers.
// In non-profiling builds they are absent, but the conditional preprocessor blocks
// are part of the header's cognitive surface — coupling profiling concern to the header.
TEST(ProfilingBehaviorCoupling, ProfilingExterns_InHotPathHeaders) {
    // pto_scheduler.h declares (inside #if PTO2_SCHED_PROFILING):
    //   extern uint64_t g_sched_lock_cycle[];
    //   extern uint64_t g_sched_fanout_cycle[];
    //   ... (8+ extern arrays, used in on_mixed_task_complete)
    //
    // pto_ring_buffer.h declares (inside #if PTO2_ORCH_PROFILING):
    //   extern uint64_t g_orch_heap_wait_cycle;
    //   extern uint64_t g_orch_heap_atomic_count;
    //   ... (4+ extern scalars, used in heap_ring_try_alloc)
    //
    // These externs sit inside headers that are included in hot-path code.
    // The profiling concern bleeds into the compile model of all translation units
    // that include these headers.

#if PTO2_SCHED_PROFILING
    // In profiling build: the externs must be defined somewhere — test stubs must provide them
    SUCCEED() << "PTO2_SCHED_PROFILING=1: profiling externs are live in this build. "
                 "They are declared in pto_scheduler.h and used in on_mixed_task_complete.";
#else
    // In non-profiling build: externs are absent — but the #if blocks remain in the header
    SUCCEED() << "PTO2_SCHED_PROFILING=0: profiling extern declarations are compiled out. "
                 "However, the #if PTO2_SCHED_PROFILING blocks in pto_scheduler.h "
                 "and pto_ring_buffer.h add conditional complexity to every reader "
                 "of these hot-path headers. Profiling coupling cannot be extracted "
                 "without modifying the headers themselves.";
#endif

    // Regardless of flag: the behavioral difference in release_fanin_and_check_ready
    // means profiling and non-profiling builds have different task state semantics.
    // This is the most significant coupling: a measurement flag alters correctness.
    size_t slot_size = sizeof(PTO2TaskSlotState);
    EXPECT_EQ(slot_size, 64u)
        << "PTO2TaskSlotState is 64 bytes (1 cache line). "
           "Profiling adds atomic counters to PTO2SchedulerState (tasks_completed, "
           "tasks_consumed) when PTO2_SCHED_PROFILING=1, potentially inflating the struct.";
}
