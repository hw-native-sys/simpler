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

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <new>
#include <utility>

#include "call_config.h"
#include "ring.h"
#include "orchestrator.h"
#include "scope.h"
#include "tensormap.h"
#include "types.h"
#include "task_args.h"

// Producers under `key` whose written view overlaps `view`, in registration order.
static std::vector<TaskSlot> map_hits(const TensorMap &tm, RunId run_id, TensorKey key, const TensorFootprint &view) {
    std::vector<TaskSlot> out;
    tm.lookup_overlapping(run_id, key, view, out);
    return out;
}

// The single producer covering the whole backing under `key`, or INVALID_SLOT when there is
// none. Asserts there is at most one: a caller states the key holds one live writer.
static TaskSlot map_producer(const TensorMap &tm, RunId run_id, TensorKey key) {
    std::vector<TaskSlot> out = map_hits(tm, run_id, key, TensorFootprint{});
    EXPECT_LE(out.size(), 1u);
    return out.empty() ? INVALID_SLOT : out[0];
}

// The footprint of byte `row` of a `rows`-byte backing — what `row_view_args` names.
static TensorFootprint row_fp(uint64_t row, uint64_t rows) {
    TensorFootprint f{};
    f.byte_offset = row;
    f.backing_nbytes = rows;
    f.ndims = 1;
    f.shapes[0] = 1;
    f.strides[0] = 1;
    f.dtype = DataType::UINT8;
    return f;
}

// ---------------------------------------------------------------------------
// Fixture: wires the Orchestrator components together (no Scheduler thread)
// ---------------------------------------------------------------------------

struct OrchestratorFixture : public ::testing::Test {
    TensorMap tm;
    Ring allocator;
    Scope scope;
    NextLevelReadyQueues rq_next_level;
    ReadyQueue rq_sub;
    Orchestrator orch;
    CallConfig cfg;
    RunId run_id{INVALID_RUN_ID};

    // Tests in this file only submit NEXT_LEVEL tasks, so `rq` is a
    // convenience alias for the next-level queue. Kept public so existing
    // `rq.try_pop(...)` / `EXPECT_TRUE(rq.try_pop(...))` lines continue to
    // work without rewriting every assertion.
    struct WorkerQueueView {
        NextLevelReadyQueues *queues;
        bool try_pop(TaskSlot &out) { return queues->try_pop_single(0, out); }
        bool try_pop(RunId run_id, TaskSlot &out) { return queues->try_pop_single(0, run_id, out); }
    } rq{&rq_next_level};

    void SetUp() override {
        allocator.init(/*heap_bytes=*/1ULL << 20);
        rq_next_level.reset({0, 1, 3});
        orch.init(&tm, &allocator, &scope, &rq_sub, &rq_next_level);
        run_id = orch.begin_run();
    }

    void TearDown() override { allocator.shutdown(); }

    // Per-slot accessor -- slot state lives inside the Ring now.
    TaskSlotState &S(TaskSlot id) { return *allocator.slot_state(id); }

    static CallableIdentity C(uint8_t seed) {
        CallableIdentity c;
        c.digest.fill(seed);
        return c;
    }

    // A canonical identity keyed by `buffer_id` (fixed owner nonce/path), so two args with the same
    // buffer_id land on one dependency key — the successor of the former buffer-address key. Whether
    // they carry an edge is then their byte ranges' business.
    static CanonicalIdentity identity_for(uint64_t buffer_id) {
        CanonicalIdentity id{};
        id.buffer_id = buffer_id;
        return id;
    }

    // The local dependency key the orchestrator derives for a ref over `buffer_id`.
    static TensorKey ref_key(uint64_t buffer_id) {
        return TensorKey::local_host(static_cast<uint64_t>(CanonicalIdentityHash{}(identity_for(buffer_id))));
    }

    // Helper: a TaskArgs whose only ref is a local POSIX_SHM backing keyed by `buffer_id`.
    static TaskArgs single_tensor_args(uint64_t buffer_id, TensorArgType tag) {
        TaskArgs a;
        Tensor r{};
        r.buffer.backend_kind = static_cast<uint8_t>(BackendKind::POSIX_SHM);
        r.buffer.access = static_cast<uint8_t>(AccessMode::READWRITE);  // the tags below include writes
        r.buffer.nbytes = 1;  // has a local backing (not a placeholder) -> tracked by infer_deps
        r.buffer.identity = identity_for(buffer_id);
        r.ndims = 1;
        r.shapes[0] = 1;
        r.strides[0] = 1;
        r.dtype = DataType::UINT8;
        a.add_tensor(r, tag);
        return a;
    }

    // Helper: a TaskArgs holding row `row` of a `rows`-row backing keyed by `buffer_id` — one
    // UINT8 element per row, so row r is exactly byte r. The rank-major slice `x[r]` that
    // distributed codegen hands each worker.
    static TaskArgs row_view_args(uint64_t buffer_id, uint64_t row, uint64_t rows, TensorArgType tag) {
        TaskArgs a;
        Tensor r{};
        r.buffer.backend_kind = static_cast<uint8_t>(BackendKind::POSIX_SHM);
        r.buffer.access = static_cast<uint8_t>(AccessMode::READWRITE);
        r.buffer.nbytes = rows;
        r.buffer.identity = identity_for(buffer_id);
        r.byte_offset = row;
        r.ndims = 1;
        r.shapes[0] = 1;
        r.strides[0] = 1;
        r.dtype = DataType::UINT8;
        a.add_tensor(r, tag);
        return a;
    }

    // Helper: a TaskArgs holding rows [r0, r0+rows) x cols [c0, c0+cols) of a `R`x`C` UINT8 matrix
    // keyed by `buffer_id`. Two of these can be disjoint while their bounding boxes interleave,
    // which is what makes the per-dimension stage of the overlap cascade load-bearing.
    static TaskArgs tile_view_args(
        uint64_t buffer_id, uint32_t r0, uint32_t rows, uint32_t c0, uint32_t cols, uint32_t R, uint32_t C,
        TensorArgType tag
    ) {
        TaskArgs a;
        Tensor r{};
        r.buffer.backend_kind = static_cast<uint8_t>(BackendKind::POSIX_SHM);
        r.buffer.access = static_cast<uint8_t>(AccessMode::READWRITE);
        r.buffer.nbytes = static_cast<uint64_t>(R) * C;
        r.buffer.identity = identity_for(buffer_id);
        r.byte_offset = static_cast<uint64_t>(r0) * C + c0;
        r.ndims = 2;
        r.shapes[0] = rows;
        r.shapes[1] = cols;
        r.strides[0] = C;
        r.strides[1] = 1;
        r.dtype = DataType::UINT8;
        a.add_tensor(r, tag);
        return a;
    }

    // Helper: a TaskArgs whose only ref is a REMOTE_SIDECAR placeholder (no local backing; the remote
    // descriptor rides in the paired RemoteTaskArgsSidecar).
    static TaskArgs remote_placeholder_args(TensorArgType tag) {
        TaskArgs a;
        Tensor r{};
        r.buffer.backend_kind = static_cast<uint8_t>(BackendKind::REMOTE_SIDECAR);
        r.buffer.access = static_cast<uint8_t>(AccessMode::READWRITE);
        r.ndims = 1;
        r.shapes[0] = 1;
        r.strides[0] = 1;
        r.dtype = DataType::UINT8;
        a.add_tensor(r, tag);
        return a;
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(ReadyQueueTest, UnscopedAccessPreservesRunInsertionOrder) {
    ReadyQueue queue;
    queue.push(/*run_id=*/7, /*slot=*/70);
    queue.push(/*run_id=*/8, /*slot=*/80);
    queue.push(/*run_id=*/7, /*slot=*/71);

    TaskSlot slot = INVALID_SLOT;
    ASSERT_TRUE(queue.try_front(slot));
    EXPECT_EQ(slot, 70);
    ASSERT_TRUE(queue.try_pop(slot));
    EXPECT_EQ(slot, 70);
    ASSERT_TRUE(queue.try_pop(slot));
    EXPECT_EQ(slot, 71);
    ASSERT_TRUE(queue.try_pop(slot));
    EXPECT_EQ(slot, 80);
    EXPECT_TRUE(queue.empty());
}

TEST(ReadyQueueTest, EmptyPartitionDoesNotReserveAFutureRunPosition) {
    ReadyQueue queue;
    queue.push(7, 70);
    TaskSlot slot = INVALID_SLOT;
    ASSERT_TRUE(queue.try_pop(slot));
    EXPECT_EQ(slot, 70);
    queue.push(8, 80);
    queue.push(7, 71);

    // Unscoped access follows nonempty partition insertion order. Scoped
    // access selects the active run even when its partition was re-created.
    ASSERT_TRUE(queue.try_front(slot));
    EXPECT_EQ(slot, 80);
    ASSERT_TRUE(queue.try_pop(7, slot));
    EXPECT_EQ(slot, 71);
    ASSERT_TRUE(queue.try_pop(slot));
    EXPECT_EQ(slot, 80);
    EXPECT_TRUE(queue.empty());
}

TEST_F(OrchestratorFixture, IndependentTaskIsImmediatelyReady) {
    auto a = single_tensor_args(0xCAFE, TensorArgType::OUTPUT);
    auto res = orch.submit_next_level(C(42), a, cfg, 0);
    EXPECT_NE(res.task_slot, INVALID_SLOT);

    TaskSlot slot;
    EXPECT_TRUE(rq.try_pop(slot));
    EXPECT_EQ(slot, res.task_slot);
    EXPECT_EQ(S(slot).state.load(), TaskState::READY);
}

TEST_F(OrchestratorFixture, NextLevelTargetsMustBeValidAndDistinct) {
    auto args = single_tensor_args(0xCAFE, TensorArgType::OUTPUT);
    EXPECT_THROW((void)orch.submit_next_level(C(42), args, cfg, -1), std::invalid_argument);
    EXPECT_THROW((void)orch.submit_next_level_group(C(42), {args, args}, cfg, {0}), std::invalid_argument);
    EXPECT_THROW((void)orch.submit_next_level_group(C(42), {args, args}, cfg, {0, 0}), std::invalid_argument);
}

TEST_F(OrchestratorFixture, DependentTaskIsPending) {
    // Task A produces an OUTPUT at key 0xBEEF
    auto args_a = single_tensor_args(0xBEEF, TensorArgType::OUTPUT);
    auto a = orch.submit_next_level(C(42), args_a, cfg, 0);
    TaskSlot a_slot;
    rq.try_pop(a_slot);

    // Task B reads INPUT at the same key -- depends on A
    auto args_b = single_tensor_args(0xBEEF, TensorArgType::INPUT);
    auto b = orch.submit_next_level(C(42), args_b, cfg, 0);
    EXPECT_EQ(S(b.task_slot).state.load(), TaskState::PENDING);
    EXPECT_EQ(S(b.task_slot).fanin_count, 1);

    TaskSlot extra;
    EXPECT_FALSE(rq.try_pop(extra));  // B should NOT be in ready queue
}

TEST_F(OrchestratorFixture, GroupDuplicateInputsKeepOneProducer) {
    auto producer_args = single_tensor_args(0xBEEF, TensorArgType::OUTPUT);
    auto producer = orch.submit_next_level(C(42), producer_args, cfg, 0);
    TaskSlot ready;
    ASSERT_TRUE(rq.try_pop(ready));
    ASSERT_EQ(ready, producer.task_slot);

    TaskArgs input0 = single_tensor_args(0xBEEF, TensorArgType::INPUT);
    TaskArgs input1 = single_tensor_args(0xBEEF, TensorArgType::INPUT);
    auto consumer = orch.submit_next_level_group(C(43), {input0, input1}, cfg, {0, 1});

    EXPECT_EQ(S(consumer.task_slot).state.load(), TaskState::PENDING);
    EXPECT_EQ(S(consumer.task_slot).fanin_count, 1);
    ASSERT_EQ(S(consumer.task_slot).fanin_producers.size(), 1u);
    EXPECT_EQ(S(consumer.task_slot).fanin_producers[0], producer.task_slot);
}

TEST_F(OrchestratorFixture, SubmitAfterFailedProducerPoisonsConsumer) {
    orch.scope_begin();

    auto args_a = single_tensor_args(0xD00D, TensorArgType::OUTPUT);
    auto a = orch.submit_next_level(C(42), args_a, cfg, 0);
    TaskSlot ready;
    ASSERT_TRUE(rq.try_pop(ready));
    ASSERT_EQ(ready, a.task_slot);

    TaskSlotState &producer = S(a.task_slot);
    producer.failure_message = "producer failed";
    producer.state.store(TaskState::FAILED, std::memory_order_release);
    producer.fanout_released.store(1, std::memory_order_release);

    auto args_b = single_tensor_args(0xD00D, TensorArgType::INPUT);
    auto b = orch.submit_next_level(C(43), args_b, cfg, 0);

    EXPECT_EQ(S(b.task_slot).state.load(), TaskState::FAILED);
    EXPECT_EQ(S(b.task_slot).failure_message, "producer failed");
    EXPECT_EQ(S(b.task_slot).fanin_count, 0);
    ASSERT_EQ(S(b.task_slot).fanin_producers.size(), 1u);
    EXPECT_EQ(S(b.task_slot).fanin_producers[0], a.task_slot);

    TaskSlot extra;
    EXPECT_FALSE(rq.try_pop(extra));

    orch.scope_end();
    EXPECT_EQ(S(a.task_slot).state.load(), TaskState::CONSUMED);
    EXPECT_EQ(S(b.task_slot).state.load(), TaskState::CONSUMED);
}

// A poisoned slot never reaches an endpoint, so nothing ever calls
// mark_task_accepted for it. Counting it as a pending accept would leave the
// count permanently above zero and push the acceptance fence out to run
// completion — the run still finishes, because acceptance_ready() also accepts
// a terminal phase, so the symptom is a lost overlap rather than a hang. The
// live sibling here is what makes that observable: its own accept is the only
// one owing, so the fence must open when it lands, while the run is still live.
TEST_F(OrchestratorFixture, PoisonedConsumerIsNotCountedAsAPendingAccept) {
    orch.scope_begin();

    auto args_a = single_tensor_args(0xD00D, TensorArgType::OUTPUT);
    auto a = orch.submit_next_level(C(42), args_a, cfg, 0);
    TaskSlot ready;
    ASSERT_TRUE(rq.try_pop(ready));
    ASSERT_EQ(ready, a.task_slot);

    // Submitted before the poison: once a run records an error, submit_next_level
    // rethrows it rather than admitting anything more, so the live sibling has to
    // exist first. It owes the only accept the run can still receive.
    auto live = orch.submit_next_level(C(44), single_tensor_args(0xFEED, TensorArgType::OUTPUT), cfg, 0);
    TaskSlot live_ready;
    ASSERT_TRUE(rq.try_pop(live_ready));
    ASSERT_EQ(live_ready, live.task_slot);

    TaskSlotState &producer = S(a.task_slot);
    producer.failure_message = "producer failed";
    producer.state.store(TaskState::FAILED, std::memory_order_release);
    producer.fanout_released.store(1, std::memory_order_release);

    auto args_b = single_tensor_args(0xD00D, TensorArgType::INPUT);
    auto poisoned = orch.submit_next_level(C(43), args_b, cfg, 0);
    ASSERT_EQ(S(poisoned.task_slot).state.load(), TaskState::FAILED);

    orch.close_run_submission(run_id);
    EXPECT_FALSE(orch.run_accepted(run_id)) << "the live task has not been accepted yet";

    orch.mark_task_accepted(a.task_slot);
    orch.mark_task_accepted(live.task_slot);
    EXPECT_TRUE(orch.run_accepted(run_id)) << "the poisoned slot is still owing an accept it can never receive";

    orch.scope_end();
}

TEST_F(OrchestratorFixture, TensorMapTracksProducer) {
    auto args_a = single_tensor_args(0x1234, TensorArgType::OUTPUT);
    auto a = orch.submit_next_level(C(42), args_a, cfg, 0);
    TaskSlot drain_slot;
    rq.try_pop(drain_slot);

    EXPECT_EQ(map_producer(tm, run_id, ref_key(0x1234)), a.task_slot);
}

TEST_F(OrchestratorFixture, OnConsumedCleansUpTensorMap) {
    auto args_a = single_tensor_args(0x42, TensorArgType::OUTPUT);
    auto a = orch.submit_next_level(C(42), args_a, cfg, 0);
    TaskSlot slot;
    rq.try_pop(slot);

    EXPECT_EQ(map_producer(tm, run_id, ref_key(0x42)), slot);

    S(slot).state.store(TaskState::COMPLETED, std::memory_order_relaxed);
    orch.on_consumed(slot);

    EXPECT_EQ(map_producer(tm, run_id, ref_key(0x42)), INVALID_SLOT);
    EXPECT_EQ(S(slot).state.load(), TaskState::CONSUMED);
}

TEST_F(OrchestratorFixture, ConsumingASupersededProducerKeepsTheNewerMapping) {
    // Two OUTPUT writers of one key carry no edge between them (insert-only,
    // see OutputAndOutputExistingAreInsertOnly), so the first can reach
    // CONSUMED while the second is still running. Its cleanup must leave the
    // second's mapping intact -- otherwise a later INPUT reader looks up an
    // empty slot, infers no dependency, and can be dispatched before the
    // second writer has written the buffer.
    auto args_a = single_tensor_args(0x5150, TensorArgType::OUTPUT);
    auto a = orch.submit_next_level(C(42), args_a, cfg, 0);
    TaskSlot drain;
    ASSERT_TRUE(rq.try_pop(drain));

    auto args_b = single_tensor_args(0x5150, TensorArgType::OUTPUT);
    auto b = orch.submit_next_level(C(43), args_b, cfg, 0);
    ASSERT_TRUE(rq.try_pop(drain));
    ASSERT_EQ(map_producer(tm, run_id, ref_key(0x5150)), b.task_slot);

    S(a.task_slot).state.store(TaskState::COMPLETED, std::memory_order_relaxed);
    ASSERT_TRUE(orch.on_consumed(a.task_slot));

    EXPECT_EQ(map_producer(tm, run_id, ref_key(0x5150)), b.task_slot);

    auto args_c = single_tensor_args(0x5150, TensorArgType::INPUT);
    auto c = orch.submit_next_level(C(44), args_c, cfg, 0);
    EXPECT_EQ(S(c.task_slot).state.load(), TaskState::PENDING);
    EXPECT_EQ(S(c.task_slot).fanin_count, 1);
    ASSERT_EQ(S(c.task_slot).fanin_producers.size(), 1u);
    EXPECT_EQ(S(c.task_slot).fanin_producers[0], b.task_slot);
}

TEST_F(OrchestratorFixture, DisjointViewsOfOneHostBackingDoNotSerialize) {
    // Distributed codegen gives each rank an INOUT view of one rank-major host tensor. The two
    // views share a key -- a host key carries no worker id to separate them -- so without the
    // byte-range check rank 1's lookup resolves rank 0's slot and the two dispatches serialize.
    // Each rank's kernel is meanwhile spinning on the other's cross-rank notify, so the edge
    // deadlocks the run rather than merely slowing it (pypto#2448).
    auto rank0 = row_view_args(0xBEEF, /*row=*/0, /*rows=*/2, TensorArgType::INOUT);
    auto a = orch.submit_next_level(C(42), rank0, cfg, 0);
    TaskSlot ready;
    ASSERT_TRUE(rq.try_pop(ready));
    ASSERT_EQ(ready, a.task_slot);

    auto rank1 = row_view_args(0xBEEF, /*row=*/1, /*rows=*/2, TensorArgType::INOUT);
    auto b = orch.submit_next_level(C(42), rank1, cfg, 0);

    EXPECT_EQ(S(b.task_slot).state.load(), TaskState::READY);
    EXPECT_EQ(S(b.task_slot).fanin_count, 0);
    EXPECT_TRUE(S(b.task_slot).fanin_producers.empty());

    // Both writers stay live under the one key, each owning its own row.
    EXPECT_EQ(map_hits(tm, run_id, ref_key(0xBEEF), row_fp(0, 2)), (std::vector<TaskSlot>{a.task_slot}));
    EXPECT_EQ(map_hits(tm, run_id, ref_key(0xBEEF), row_fp(1, 2)), (std::vector<TaskSlot>{b.task_slot}));
}

TEST_F(OrchestratorFixture, DisjointColumnBlocksOfOneHostBackingDoNotSerialize) {
    // The same INOUT dispatch as above, sliced on a non-leading axis. `x[:, 0:4]` and
    // `x[:, 8:12]` of a 16x16 matrix put every row of one between two rows of the other, so
    // their bounding ranges interleave -- [0, 244) and [8, 252) -- and the O(1) stage alone
    // would wire the same false edge. Only the per-dimension stage separates them, and it has
    // to do so on the footprint infer_deps builds from the Tensor, not a hand-made one.
    auto west = tile_view_args(0xBEEF, 0, 16, 0, 4, 16, 16, TensorArgType::INOUT);
    auto a = orch.submit_next_level(C(42), west, cfg, 0);
    TaskSlot ready;
    ASSERT_TRUE(rq.try_pop(ready));
    ASSERT_EQ(ready, a.task_slot);

    auto east = tile_view_args(0xBEEF, 0, 16, 8, 4, 16, 16, TensorArgType::INOUT);
    auto b = orch.submit_next_level(C(42), east, cfg, 0);

    EXPECT_EQ(S(b.task_slot).state.load(), TaskState::READY);
    EXPECT_EQ(S(b.task_slot).fanin_count, 0);
    EXPECT_TRUE(S(b.task_slot).fanin_producers.empty());
}

TEST_F(OrchestratorFixture, DisjointTilesOfOneHostBackingDoNotSerialize) {
    // A 2x2 grid of 8x8 tiles, every one INOUT: four dispatches, no edge between any pair.
    struct {
        uint32_t r0, c0;
    } quads[] = {{0, 0}, {0, 8}, {8, 0}, {8, 8}};
    for (const auto &q : quads) {
        auto args = tile_view_args(0xBEEF, q.r0, 8, q.c0, 8, 16, 16, TensorArgType::INOUT);
        auto res = orch.submit_next_level(C(42), args, cfg, 0);
        EXPECT_EQ(S(res.task_slot).state.load(), TaskState::READY);
        EXPECT_EQ(S(res.task_slot).fanin_count, 0) << "tile (" << q.r0 << ", " << q.c0 << ") took an edge";
        TaskSlot ready;
        ASSERT_TRUE(rq.try_pop(ready));
        ASSERT_EQ(ready, res.task_slot);
    }
    // All four remain live under the one key, each owning its quadrant.
    EXPECT_EQ(tm.size(), 4);
}

TEST_F(OrchestratorFixture, IntersectingTilesOfOneHostBackingStillSerialize) {
    // The converse of the two above: tiles that do share elements keep their edge, so the
    // refinement is not simply switching dependency inference off for 2-D views.
    auto first = tile_view_args(0xBEEF, 0, 8, 0, 8, 16, 16, TensorArgType::INOUT);
    auto a = orch.submit_next_level(C(42), first, cfg, 0);
    TaskSlot ready;
    ASSERT_TRUE(rq.try_pop(ready));

    auto straddling = tile_view_args(0xBEEF, 4, 8, 4, 8, 16, 16, TensorArgType::INOUT);
    auto b = orch.submit_next_level(C(43), straddling, cfg, 0);

    EXPECT_EQ(S(b.task_slot).state.load(), TaskState::PENDING);
    ASSERT_EQ(S(b.task_slot).fanin_producers.size(), 1u);
    EXPECT_EQ(S(b.task_slot).fanin_producers[0], a.task_slot);
}

TEST_F(OrchestratorFixture, OverlappingViewsOfOneHostBackingStillSerialize) {
    // The converse of the disjoint case: same key, ranges that share a byte, so the edge is real
    // and must survive the refinement.
    auto writer = row_view_args(0xBEEF, /*row=*/1, /*rows=*/4, TensorArgType::INOUT);
    auto a = orch.submit_next_level(C(42), writer, cfg, 0);
    TaskSlot ready;
    ASSERT_TRUE(rq.try_pop(ready));

    auto reader = row_view_args(0xBEEF, /*row=*/1, /*rows=*/4, TensorArgType::INPUT);
    auto b = orch.submit_next_level(C(43), reader, cfg, 0);

    EXPECT_EQ(S(b.task_slot).state.load(), TaskState::PENDING);
    ASSERT_EQ(S(b.task_slot).fanin_producers.size(), 1u);
    EXPECT_EQ(S(b.task_slot).fanin_producers[0], a.task_slot);
}

TEST_F(OrchestratorFixture, AReaderOfTheWholeBackingDependsOnEveryDisjointWriter) {
    // A whole-backing view overlaps both rows, so it takes both edges -- the case that makes
    // dropping a superseded-but-not-covered producer unsound.
    auto rank0 = row_view_args(0xBEEF, /*row=*/0, /*rows=*/2, TensorArgType::OUTPUT_EXISTING);
    auto a = orch.submit_next_level(C(42), rank0, cfg, 0);
    TaskSlot ready;
    ASSERT_TRUE(rq.try_pop(ready));

    auto rank1 = row_view_args(0xBEEF, /*row=*/1, /*rows=*/2, TensorArgType::OUTPUT_EXISTING);
    auto b = orch.submit_next_level(C(42), rank1, cfg, 0);
    ASSERT_TRUE(rq.try_pop(ready));

    TaskArgs full;
    Tensor r{};
    r.buffer.backend_kind = static_cast<uint8_t>(BackendKind::POSIX_SHM);
    r.buffer.access = static_cast<uint8_t>(AccessMode::READWRITE);
    r.buffer.nbytes = 2;
    r.buffer.identity = identity_for(0xBEEF);
    r.ndims = 1;
    r.shapes[0] = 2;
    r.strides[0] = 1;
    r.dtype = DataType::UINT8;
    full.add_tensor(r, TensorArgType::INPUT);

    auto c = orch.submit_next_level(C(43), full, cfg, 0);
    EXPECT_EQ(S(c.task_slot).state.load(), TaskState::PENDING);
    EXPECT_EQ(S(c.task_slot).fanin_count, 2);
    ASSERT_EQ(S(c.task_slot).fanin_producers.size(), 2u);
    EXPECT_EQ(S(c.task_slot).fanin_producers[0], a.task_slot);
    EXPECT_EQ(S(c.task_slot).fanin_producers[1], b.task_slot);
}

TEST_F(OrchestratorFixture, ConsumingOneDisjointWriterLeavesTheOther) {
    auto rank0 = row_view_args(0xBEEF, /*row=*/0, /*rows=*/2, TensorArgType::INOUT);
    auto a = orch.submit_next_level(C(42), rank0, cfg, 0);
    TaskSlot ready;
    ASSERT_TRUE(rq.try_pop(ready));

    auto rank1 = row_view_args(0xBEEF, /*row=*/1, /*rows=*/2, TensorArgType::INOUT);
    auto b = orch.submit_next_level(C(42), rank1, cfg, 0);
    ASSERT_TRUE(rq.try_pop(ready));

    S(a.task_slot).state.store(TaskState::COMPLETED, std::memory_order_relaxed);
    ASSERT_TRUE(orch.on_consumed(a.task_slot));

    // Cleanup is keyed by (key, owner), so it must not take the row it does not own with it.
    EXPECT_TRUE(map_hits(tm, run_id, ref_key(0xBEEF), row_fp(0, 2)).empty());
    EXPECT_EQ(map_hits(tm, run_id, ref_key(0xBEEF), row_fp(1, 2)), (std::vector<TaskSlot>{b.task_slot}));
}

TEST_F(OrchestratorFixture, ScopeRegistersAndReleasesRef) {
    orch.scope_begin();
    auto args_a = single_tensor_args(0x77, TensorArgType::OUTPUT);
    auto a = orch.submit_next_level(C(42), args_a, cfg, 0);
    TaskSlot slot;
    rq.try_pop(slot);

    {
        std::lock_guard<std::mutex> lk(S(slot).fanout_mu);
        EXPECT_EQ(S(slot).fanout_total, 1);
    }

    // Simulate the completion path that would run if this test drove the
    // full scheduler: state -> COMPLETED + the self try_consume that
    // on_task_complete would normally fire (bumps fanout_released by 1).
    // Without this simulated self-release, the `>= total + 1` threshold in
    // release_ref / try_consume cannot be met from scope_end alone.
    S(slot).state.store(TaskState::COMPLETED, std::memory_order_relaxed);
    S(slot).fanout_released.fetch_add(1, std::memory_order_relaxed);
    orch.scope_end();

    EXPECT_EQ(S(slot).state.load(), TaskState::CONSUMED);
}

TEST_F(OrchestratorFixture, NoDepTagSkipsDependencyTracking) {
    // OUTPUT-tagged input registers a producer
    auto args_a = single_tensor_args(0xAAAA, TensorArgType::OUTPUT);
    auto a = orch.submit_next_level(C(42), args_a, cfg, 0);
    TaskSlot drain_slot;
    rq.try_pop(drain_slot);

    // Second task references same key but tagged NO_DEP -- should be independent
    auto args_b = single_tensor_args(0xAAAA, TensorArgType::NO_DEP);
    auto b = orch.submit_next_level(C(42), args_b, cfg, 0);
    EXPECT_EQ(S(b.task_slot).state.load(), TaskState::READY);
    EXPECT_EQ(S(b.task_slot).fanin_count, 0);
}

TEST_F(OrchestratorFixture, GroupTaskStoresArgsListPerMember) {
    TaskArgs a0 = single_tensor_args(0xA0, TensorArgType::OUTPUT);
    TaskArgs a1 = single_tensor_args(0xA1, TensorArgType::OUTPUT);
    auto res = orch.submit_next_level_group(C(42), {a0, a1}, cfg, {0, 1});

    EXPECT_NE(res.task_slot, INVALID_SLOT);
    EXPECT_TRUE(S(res.task_slot).is_group());
    EXPECT_EQ(S(res.task_slot).group_size(), 2);
    EXPECT_EQ(S(res.task_slot).task_args_list.size(), 2u);

    // args_view(i) yields each member's distinct tensor list.
    EXPECT_EQ(S(res.task_slot).args(0).tensor(0).buffer.identity.buffer_id, 0xA0u);
    EXPECT_EQ(S(res.task_slot).args(1).tensor(0).buffer.identity.buffer_id, 0xA1u);

    // Both keys registered as producers for the group slot.
    EXPECT_EQ(map_producer(tm, run_id, ref_key(0xA0)), res.task_slot);
    EXPECT_EQ(map_producer(tm, run_id, ref_key(0xA1)), res.task_slot);
}

TEST_F(OrchestratorFixture, SingleTaskStoresTaskArgsDirectly) {
    TaskArgs a0 = single_tensor_args(0xC0, TensorArgType::OUTPUT);
    auto res = orch.submit_next_level(C(42), a0, cfg, 0);
    ASSERT_NE(res.task_slot, INVALID_SLOT);
    EXPECT_FALSE(S(res.task_slot).is_group());
    EXPECT_EQ(S(res.task_slot).group_size(), 1);
    EXPECT_EQ(S(res.task_slot).task_args.tensor_count(), 1);
    EXPECT_EQ(S(res.task_slot).args(0).tensor(0).buffer.identity.buffer_id, 0xC0u);
}

TEST_F(OrchestratorFixture, RemoteOutputSidecarRegistersRemoteKey) {
    TaskArgs args = remote_placeholder_args(TensorArgType::OUTPUT);

    RemoteTaskArgsSidecar sidecar;
    sidecar.tensors.resize(1);
    sidecar.tensors[0].present = true;
    sidecar.tensors[0].desc.address_space = RemoteAddressSpace::REMOTE_DEVICE;
    sidecar.tensors[0].desc.owner_worker_id = 3;
    sidecar.tensors[0].desc.buffer_id = 9;
    sidecar.tensors[0].desc.generation = 2;
    sidecar.tensors[0].desc.offset = 64;
    sidecar.tensors[0].desc.nbytes = 1024;

    auto res = orch.submit_next_level(C(42), args, cfg, 3, {3}, sidecar);
    ASSERT_NE(res.task_slot, INVALID_SLOT);
    // No local backing: the arg is a REMOTE_SIDECAR placeholder, routed by the sidecar's remote key.
    EXPECT_EQ(
        S(res.task_slot).task_args.tensor(0).buffer.backend_kind, static_cast<uint8_t>(BackendKind::REMOTE_SIDECAR)
    );

    TensorKey key = TensorKey::remote_buffer(TensorAddressKind::REMOTE_BUFFER, 3, 9, 2, 64);
    EXPECT_EQ(map_producer(tm, run_id, key), res.task_slot);
}

TEST_F(OrchestratorFixture, RemoteBarePayloadFailsBeforeSlotCommit) {
    TaskArgs args = single_tensor_args(0x1234, TensorArgType::INPUT);

    RemoteTaskArgsSidecar sidecar;
    sidecar.tensors.resize(1);

    EXPECT_THROW({ (void)orch.submit_next_level(C(42), args, cfg, 3, {3}, sidecar); }, std::invalid_argument);
}

TEST_F(OrchestratorFixture, RemoteSidecarRejectsNonOwnerEligibleEndpointWithoutImport) {
    TaskArgs args = remote_placeholder_args(TensorArgType::INPUT);

    RemoteTaskArgsSidecar sidecar;
    sidecar.tensors.resize(1);
    sidecar.tensors[0].present = true;
    sidecar.tensors[0].desc.address_space = RemoteAddressSpace::REMOTE_DEVICE;
    sidecar.tensors[0].desc.owner_worker_id = 3;
    sidecar.tensors[0].desc.buffer_id = 9;
    sidecar.tensors[0].desc.generation = 2;
    sidecar.tensors[0].desc.nbytes = 1;

    EXPECT_THROW({ (void)orch.submit_next_level(C(42), args, cfg, 3, {3, 4}, sidecar); }, std::invalid_argument);
}

TEST_F(OrchestratorFixture, RemoteInputSidecarUsesRemoteTensorMapKey) {
    TaskArgs output_args = remote_placeholder_args(TensorArgType::OUTPUT);

    RemoteTaskArgsSidecar output_sidecar;
    output_sidecar.tensors.resize(1);
    output_sidecar.tensors[0].present = true;
    output_sidecar.tensors[0].desc.address_space = RemoteAddressSpace::REMOTE_DEVICE;
    output_sidecar.tensors[0].desc.owner_worker_id = 3;
    output_sidecar.tensors[0].desc.buffer_id = 9;
    output_sidecar.tensors[0].desc.generation = 2;
    output_sidecar.tensors[0].desc.offset = 0;
    output_sidecar.tensors[0].desc.nbytes = 1;
    auto producer = orch.submit_next_level(C(42), output_args, cfg, 3, {3}, output_sidecar);
    TaskSlot ready;
    ASSERT_TRUE(rq_next_level.try_pop_single(3, ready));
    ASSERT_EQ(ready, producer.task_slot);

    TaskArgs input_args = remote_placeholder_args(TensorArgType::INPUT);
    auto consumer = orch.submit_next_level(C(43), input_args, cfg, 3, {3}, output_sidecar);

    EXPECT_EQ(S(consumer.task_slot).state.load(), TaskState::PENDING);
    EXPECT_EQ(S(consumer.task_slot).fanin_count, 1);
    ASSERT_EQ(S(consumer.task_slot).fanin_producers.size(), 1u);
    EXPECT_EQ(S(consumer.task_slot).fanin_producers[0], producer.task_slot);
}

TEST_F(OrchestratorFixture, InoutWiresCreatorAsFanin) {
    // INOUT is the only tag that pulls in the prior writer as a fanin
    // producer -- matching L2's orchestrator.cpp Step B where only
    // INPUT / INOUT do tensor_map.lookup. Users who want a WaW dep on
    // the alloc-slot (so its HeapRing slab stays live while they write)
    // must tag the buffer INOUT.
    auto creator_args = single_tensor_args(0xFEED, TensorArgType::OUTPUT);
    auto creator = orch.submit_next_level(C(42), creator_args, cfg, 0);
    TaskSlot drain;
    rq.try_pop(drain);
    // Mark the creator COMPLETED so the new submit mimics the alloc-slot
    // path (COMPLETED producer with non-zero fanout).
    S(creator.task_slot).state.store(TaskState::COMPLETED, std::memory_order_relaxed);

    auto writer_args = single_tensor_args(0xFEED, TensorArgType::INOUT);
    auto writer = orch.submit_next_level(C(42), writer_args, cfg, 0);
    TaskSlot writer_slot;
    rq.try_pop(writer_slot);

    // TensorMap now points at the new writer.
    EXPECT_EQ(map_producer(tm, run_id, ref_key(0xFEED)), writer.task_slot);
    // Writer has the creator recorded as a fanin producer (via INOUT
    // lookup) but no *live* fanin since the creator is already COMPLETED.
    EXPECT_EQ(S(writer.task_slot).fanin_count, 0);
    ASSERT_EQ(S(writer.task_slot).fanin_producers.size(), 1u);
    EXPECT_EQ(S(writer.task_slot).fanin_producers[0], creator.task_slot);
    // Creator's fanout_total bumped so it waits for writer before CONSUMED.
    {
        std::lock_guard<std::mutex> lk(S(creator.task_slot).fanout_mu);
        EXPECT_EQ(S(creator.task_slot).fanout_total, 1);
        ASSERT_EQ(S(creator.task_slot).fanout_consumers.size(), 1u);
        EXPECT_EQ(S(creator.task_slot).fanout_consumers[0], writer.task_slot);
    }
}

TEST_F(OrchestratorFixture, OutputAndOutputExistingAreInsertOnly) {
    // Contrast with INOUT: plain OUTPUT and OUTPUT_EXISTING are pure
    // overwrites -- insert into TensorMap, no lookup, so no fanin wire
    // on the prior writer. Matches L2 semantics for both tags. Users
    // who need creator lifetime must tag the buffer INOUT.
    struct Case {
        uint64_t key;
        TensorArgType tag;
    };
    for (Case c : {Case{0xABCD, TensorArgType::OUTPUT}, Case{0xBEEF, TensorArgType::OUTPUT_EXISTING}}) {
        auto prior_args = single_tensor_args(c.key, TensorArgType::OUTPUT);
        auto prior = orch.submit_next_level(C(42), prior_args, cfg, 0);
        TaskSlot drain;
        rq.try_pop(drain);
        S(prior.task_slot).state.store(TaskState::COMPLETED, std::memory_order_relaxed);

        auto writer_args = single_tensor_args(c.key, c.tag);
        auto writer = orch.submit_next_level(C(42), writer_args, cfg, 0);

        EXPECT_EQ(map_producer(tm, run_id, ref_key(c.key)), writer.task_slot);
        EXPECT_EQ(S(writer.task_slot).fanin_count, 0);
        EXPECT_TRUE(S(writer.task_slot).fanin_producers.empty()) << "tag=" << static_cast<int>(c.tag);
        {
            std::lock_guard<std::mutex> lk(S(prior.task_slot).fanout_mu);
            EXPECT_EQ(S(prior.task_slot).fanout_total, 0) << "tag=" << static_cast<int>(c.tag);
        }
    }
}

TEST_F(OrchestratorFixture, EmptyRunCompletesWhenSubmissionCloses) {
    EXPECT_FALSE(orch.run_done(run_id));
    orch.close_run_submission(run_id);
    EXPECT_TRUE(orch.run_done(run_id));
    EXPECT_FALSE(orch.run_failed(run_id));
    EXPECT_NO_THROW(orch.wait_run(run_id));
    orch.release_run(run_id);
}

TEST_F(OrchestratorFixture, RunAcceptanceWaitsForEveryDispatchedGroupMember) {
    std::vector<TaskArgs> args{
        single_tensor_args(0x8010, TensorArgType::OUTPUT),
        single_tensor_args(0x8020, TensorArgType::OUTPUT),
    };
    auto result = orch.submit_next_level_group(C(80), args, cfg, {0, 1});

    orch.close_run_submission(run_id);
    EXPECT_FALSE(orch.run_accepted(run_id));

    orch.mark_task_accepted(result.task_slot);
    EXPECT_FALSE(orch.run_accepted(run_id));

    orch.mark_task_accepted(result.task_slot);
    EXPECT_TRUE(orch.run_accepted(run_id));
    EXPECT_NO_THROW(orch.wait_run_accepted(run_id));
}

TEST_F(OrchestratorFixture, TerminalFailureUnblocksRunAcceptance) {
    auto result = orch.submit_next_level(C(80), single_tensor_args(0x8030, TensorArgType::OUTPUT), cfg, 0);
    orch.fail_run_submission(run_id, std::make_exception_ptr(std::runtime_error("graph build failed")));
    EXPECT_EQ(S(result.task_slot).state.load(std::memory_order_acquire), TaskState::CONSUMED);
    EXPECT_TRUE(orch.run_accepted(run_id));
    EXPECT_NO_THROW(orch.wait_run_accepted(run_id));
}

TEST_F(OrchestratorFixture, TimedWaitCanRetryAfterTimeout) {
    EXPECT_FALSE(orch.wait_run_for(run_id, 0.0));
    EXPECT_FALSE(orch.run_done(run_id));

    orch.close_run_submission(run_id);
    EXPECT_TRUE(orch.wait_run_for(run_id, 0.0));
    orch.release_run(run_id);
}

TEST_F(OrchestratorFixture, OneTaskRunCompletesAfterConsumption) {
    auto result = orch.submit_next_level(C(80), single_tensor_args(0x8000, TensorArgType::OUTPUT), cfg, 0);
    EXPECT_EQ(S(result.task_slot).run_id, run_id);

    orch.close_run_submission(run_id);
    EXPECT_FALSE(orch.run_done(run_id));

    S(result.task_slot).state.store(TaskState::COMPLETED, std::memory_order_release);
    EXPECT_TRUE(orch.on_consumed(result.task_slot));
    EXPECT_TRUE(orch.run_done(run_id));
    EXPECT_NO_THROW(orch.wait_run(run_id));
    orch.release_run(run_id);
}

// mark_task_accepted runs on a WorkerThread, where an escaping exception ends
// the process. Neither an unknown slot nor an over-count may throw out of it; a
// count mismatch becomes the run's error, the way decrement_run_tasks does it.
TEST_F(OrchestratorFixture, AcceptanceFenceNeverThrowsOutOfAWorkerThread) {
    auto result = orch.submit_next_level(C(90), single_tensor_args(0x9000, TensorArgType::OUTPUT), cfg, 0);
    orch.close_run_submission(run_id);

    EXPECT_NO_THROW(orch.mark_task_accepted(TaskSlot{-1}));
    EXPECT_NO_THROW(orch.mark_task_accepted(result.task_slot));
    EXPECT_TRUE(orch.run_accepted(run_id));
    EXPECT_FALSE(orch.run_failed(run_id));

    // One accept past the submitted count: reported, not thrown.
    EXPECT_NO_THROW(orch.mark_task_accepted(result.task_slot));
    EXPECT_TRUE(orch.run_failed(run_id));

    S(result.task_slot).state.store(TaskState::COMPLETED, std::memory_order_release);
    EXPECT_TRUE(orch.on_consumed(result.task_slot));
    orch.release_run(run_id);
}

// A waiter that races the run's release must return, not throw: a released run
// is past acceptance by definition.
TEST_F(OrchestratorFixture, AcceptanceWaitOnAReleasedRunReturns) {
    auto result = orch.submit_next_level(C(92), single_tensor_args(0x9200, TensorArgType::OUTPUT), cfg, 0);
    orch.close_run_submission(run_id);
    orch.mark_task_accepted(result.task_slot);
    S(result.task_slot).state.store(TaskState::COMPLETED, std::memory_order_release);
    EXPECT_TRUE(orch.on_consumed(result.task_slot));
    RunId released = run_id;
    orch.release_run(released);

    EXPECT_NO_THROW(orch.wait_run_accepted(released));
    run_id = orch.begin_run();
}

TEST_F(OrchestratorFixture, SequentialRunsHaveDistinctIdsAndErrorsDoNotLeak) {
    auto failed_task = orch.submit_next_level(C(81), single_tensor_args(0x8100, TensorArgType::OUTPUT), cfg, 0);
    orch.report_task_error(failed_task.task_slot, "run one failed");
    S(failed_task.task_slot).failure_message = "run one failed";
    S(failed_task.task_slot).state.store(TaskState::FAILED, std::memory_order_release);
    EXPECT_TRUE(orch.on_consumed(failed_task.task_slot));
    orch.close_run_submission(run_id);
    EXPECT_TRUE(orch.run_failed(run_id));
    EXPECT_THROW(orch.wait_run(run_id), std::runtime_error);
    orch.release_run(run_id);

    RunId second = orch.begin_run();
    EXPECT_NE(second, run_id);
    orch.close_run_submission(second);
    EXPECT_FALSE(orch.run_failed(second));
    EXPECT_NO_THROW(orch.wait_run(second));
    orch.release_run(second);
}

TEST_F(OrchestratorFixture, FailedSubmissionCompletesWithoutEnteringDeviceWork) {
    orch.fail_run_submission(run_id, std::make_exception_ptr(std::runtime_error("graph build failed")));
    EXPECT_TRUE(orch.run_done(run_id));
    EXPECT_TRUE(orch.run_failed(run_id));
    EXPECT_THROW(orch.wait_run(run_id), std::runtime_error);
    orch.release_run(run_id);
}

TEST_F(OrchestratorFixture, ReleasingRunDoesNotResetSlotsOwnedByAnotherRegisteredRun) {
    auto first_task = orch.submit_next_level(C(82), single_tensor_args(0x8200, TensorArgType::OUTPUT), cfg, 0);
    S(first_task.task_slot).state.store(TaskState::COMPLETED, std::memory_order_release);
    EXPECT_TRUE(orch.on_consumed(first_task.task_slot));
    orch.close_run_submission(run_id);
    orch.wait_run(run_id);

    RunId second = orch.begin_run();
    EXPECT_EQ(allocator.next_task_id(), 1);
    orch.release_run(run_id);
    EXPECT_EQ(allocator.next_task_id(), 1);

    orch.close_run_submission(second);
    orch.wait_run(second);
    orch.release_run(second);
    EXPECT_EQ(allocator.next_task_id(), 0);
}

// report_task_error runs on a worker thread, where an escaping exception would
// terminate the process, so a slot whose run is already gone is a no-op.
TEST_F(OrchestratorFixture, ReportingErrorForAReleasedRunIsIgnored) {
    orch.close_run_submission(run_id);
    orch.wait_run(run_id);
    RunId released = run_id;
    orch.release_run(released);

    run_id = orch.begin_run();
    auto live = orch.submit_next_level(C(83), single_tensor_args(0x8300, TensorArgType::OUTPUT), cfg, 0);
    S(live.task_slot).run_id = released;

    EXPECT_NO_THROW(orch.report_task_error(live.task_slot, "late endpoint failure"));
    EXPECT_FALSE(orch.run_failed(run_id));
}

// on_consumed runs on the scheduler thread and reaches the same accounting.
TEST_F(OrchestratorFixture, ConsumingASlotOwnedByAReleasedRunIsIgnored) {
    auto task = orch.submit_next_level(C(84), single_tensor_args(0x8400, TensorArgType::OUTPUT), cfg, 0);
    S(task.task_slot).state.store(TaskState::COMPLETED, std::memory_order_release);
    EXPECT_TRUE(orch.on_consumed(task.task_slot));
    orch.close_run_submission(run_id);
    orch.wait_run(run_id);
    RunId released = run_id;
    orch.release_run(released);

    run_id = orch.begin_run();
    auto live = orch.submit_next_level(C(85), single_tensor_args(0x8500, TensorArgType::OUTPUT), cfg, 0);
    orch.close_run_submission(run_id);

    S(live.task_slot).run_id = released;
    S(live.task_slot).state.store(TaskState::COMPLETED, std::memory_order_release);
    EXPECT_NO_THROW((void)orch.on_consumed(live.task_slot));

    // The decrement landed on no run at all, so the live run still owes a task.
    EXPECT_FALSE(orch.run_done(run_id));
}

// Failing a run is the recovery path: refusing a run whose submission already
// closed would leave the caller's fence wait blocked forever.
TEST_F(OrchestratorFixture, FailingAnAlreadyClosedRunStillResolvesTheFence) {
    auto task = orch.submit_next_level(C(86), single_tensor_args(0x8600, TensorArgType::OUTPUT), cfg, 0);
    orch.close_run_submission(run_id);
    EXPECT_FALSE(orch.run_done(run_id));

    EXPECT_NO_THROW(
        orch.fail_run_submission(run_id, std::make_exception_ptr(std::runtime_error("late submission failure")))
    );

    S(task.task_slot).state.store(TaskState::COMPLETED, std::memory_order_release);
    EXPECT_TRUE(orch.on_consumed(task.task_slot));
    EXPECT_TRUE(orch.run_done(run_id));
    EXPECT_TRUE(orch.run_failed(run_id));
    EXPECT_THROW(orch.wait_run(run_id), std::runtime_error);
    orch.release_run(run_id);
}

TEST_F(OrchestratorFixture, FailedSubmissionCarriesItsMessageToTheFence) {
    orch.fail_run_submission(run_id, std::make_exception_ptr(std::runtime_error("orchestration: ValueError: bad arg")));
    EXPECT_TRUE(orch.run_failed(run_id));
    try {
        orch.wait_run(run_id);
        FAIL() << "wait_run must rethrow the submission error";
    } catch (const std::runtime_error &e) {
        EXPECT_STREQ(e.what(), "orchestration: ValueError: bad arg");
    }
    orch.release_run(run_id);
}

// Cancelling a run must not release a producer reference that a RUNNING
// consumer still holds. Releasing it early lets the producer reach CONSUMED —
// and its HeapRing output be reclaimed — while the device is still reading it,
// and the consumer's real completion then releases the same reference twice.
TEST_F(OrchestratorFixture, CancelKeepsProducerRefsHeldByARunningConsumer) {
    orch.scope_begin();
    auto producer = orch.submit_next_level(C(70), single_tensor_args(0x7000, TensorArgType::OUTPUT), cfg, 0);
    auto consumer = orch.submit_next_level(C(71), single_tensor_args(0x7000, TensorArgType::INPUT), cfg, 0);
    ASSERT_EQ(S(consumer.task_slot).fanin_producers.size(), 1u);
    ASSERT_EQ(S(consumer.task_slot).fanin_producers[0], producer.task_slot);

    // The consumer is on the device when the graph callback throws.
    S(consumer.task_slot).state.store(TaskState::RUNNING, std::memory_order_release);
    const int32_t released_before = S(producer.task_slot).fanout_released.load(std::memory_order_acquire);

    orch.fail_run_submission(run_id, std::make_exception_ptr(std::runtime_error("orchestration: boom")));

    EXPECT_EQ(S(consumer.task_slot).state.load(std::memory_order_acquire), TaskState::RUNNING)
        << "a RUNNING slot is owned by the device and must not be cancelled";
    // Exactly one release: the producer's own terminal self-release, because
    // the cancel path failed it. The reference the RUNNING consumer holds is
    // NOT among them — releasing that one too would make this 2, and would let
    // the producer be reclaimed under a consumer still reading it.
    EXPECT_EQ(S(producer.task_slot).fanout_released.load(std::memory_order_acquire), released_before + 1)
        << "cancellation released a producer reference still held by a RUNNING consumer";

    // The run stays live on purpose: only the consumer's real completion can
    // release the last reference, and that path belongs to the scheduler.
    S(consumer.task_slot).state.store(TaskState::COMPLETED, std::memory_order_release);
    (void)orch.on_consumed(consumer.task_slot);
    orch.scope_end();
}

// The converse: a consumer the cancel path *did* fail must release the producer
// it will never read, or an unstarted consumer pins its producer forever.
TEST_F(OrchestratorFixture, CancelReleasesProducerRefsHeldByAnUnstartedConsumer) {
    orch.scope_begin();
    auto producer = orch.submit_next_level(C(72), single_tensor_args(0x7200, TensorArgType::OUTPUT), cfg, 0);
    auto consumer = orch.submit_next_level(C(73), single_tensor_args(0x7200, TensorArgType::INPUT), cfg, 0);
    ASSERT_EQ(S(consumer.task_slot).state.load(std::memory_order_acquire), TaskState::PENDING);
    const int32_t released_before = S(producer.task_slot).fanout_released.load(std::memory_order_acquire);

    orch.fail_run_submission(run_id, std::make_exception_ptr(std::runtime_error("orchestration: boom")));

    EXPECT_GT(S(producer.task_slot).fanout_released.load(std::memory_order_acquire), released_before)
        << "a cancelled consumer must release the producer it will never read";

    orch.scope_end();
}

// retire erases a run's ready-queue partitions. A late enqueue — a dependency
// releasing its consumer, or the scheduler requeueing a task whose SUB worker
// was busy — must not rebuild one: nothing drains a partition whose run will
// never hold the FIFO head again.
// A failure claim won while a slot was BUILDING leaves the propagation to the
// submitting thread. If that thread throws before it gets there, the slot is
// FAILED with its references still held — and skipping it for being FAILED is
// what would leave the run's fence unreachable forever.
TEST_F(OrchestratorFixture, ACancelledRunTakesOverAPropagationItsSubmitAbandoned) {
    auto task = orch.submit_next_level(C(76), single_tensor_args(0x7600, TensorArgType::OUTPUT), cfg, 0);
    TaskSlot slot = task.task_slot;
    TaskSlot ready;
    ASSERT_TRUE(rq.try_pop(run_id, ready));

    // The state an abandoned submit leaves behind.
    S(slot).state.store(TaskState::BUILDING, std::memory_order_release);
    ASSERT_TRUE(claim_task_failure(S(slot), "producer failed").has_value());
    ASSERT_TRUE(S(slot).failure_propagation_pending.load(std::memory_order_acquire));

    orch.fail_run_submission(run_id, std::make_exception_ptr(std::runtime_error("callback threw")));

    EXPECT_EQ(S(slot).state.load(std::memory_order_acquire), TaskState::CONSUMED)
        << "an abandoned propagation stranded the slot";
    EXPECT_FALSE(S(slot).failure_propagation_pending.load(std::memory_order_acquire));
    EXPECT_TRUE(orch.run_done(run_id)) << "the run fence never became reachable";
    orch.release_run(run_id);
}

TEST_F(OrchestratorFixture, CancellationRetriesBeforeAnyPreparedFailureReleasesCommit) {
    auto producer = orch.submit_next_level(C(77), single_tensor_args(0x7700, TensorArgType::OUTPUT), cfg, 0);
    TaskSlot ready;
    ASSERT_TRUE(rq.try_pop(run_id, ready));
    auto consumer = orch.submit_next_level(C(78), single_tensor_args(0x7700, TensorArgType::INPUT), cfg, 0);
    ASSERT_EQ(S(consumer.task_slot).state.load(std::memory_order_acquire), TaskState::PENDING);

    int32_t snapshots = 0;
    orch.set_test_hook([&](OrchestratorTestPoint point) {
        if (point == OrchestratorTestPoint::FAILURE_FANIN_SNAPSHOT && ++snapshots == 2) throw std::bad_alloc();
    });

    EXPECT_THROW(
        orch.fail_run_submission(run_id, std::make_exception_ptr(std::runtime_error("graph construction failed"))),
        std::bad_alloc
    );

    EXPECT_TRUE(S(producer.task_slot).failure_propagation_pending.load(std::memory_order_acquire));
    EXPECT_TRUE(S(consumer.task_slot).failure_propagation_pending.load(std::memory_order_acquire));
    EXPECT_EQ(S(producer.task_slot).fanout_released.load(std::memory_order_acquire), 0);
    EXPECT_EQ(S(consumer.task_slot).fanout_released.load(std::memory_order_acquire), 0)
        << "one prepared slot released before every cancellation snapshot succeeded";

    orch.set_test_hook({});
    EXPECT_NO_THROW(orch.fail_run_submission(run_id));
    EXPECT_TRUE(orch.run_done(run_id));
    EXPECT_EQ(S(producer.task_slot).state.load(std::memory_order_acquire), TaskState::CONSUMED);
    EXPECT_EQ(S(consumer.task_slot).state.load(std::memory_order_acquire), TaskState::CONSUMED);
    orch.release_run(run_id);
}

TEST_F(OrchestratorFixture, FailedReverseEdgePublicationRollsBackTheProducerReference) {
    auto producer = orch.submit_next_level(C(79), single_tensor_args(0x7900, TensorArgType::OUTPUT), cfg, 0);
    TaskSlot ready;
    ASSERT_TRUE(rq.try_pop(run_id, ready));

    bool injected = false;
    orch.set_test_hook([&](OrchestratorTestPoint point) {
        if (point == OrchestratorTestPoint::PRODUCER_FORWARD_EDGE_PUBLISHED && !injected) {
            injected = true;
            throw std::bad_alloc();
        }
    });
    EXPECT_THROW(
        (void)orch.submit_next_level(C(80), single_tensor_args(0x7900, TensorArgType::INPUT), cfg, 0), std::bad_alloc
    );
    ASSERT_TRUE(injected);

    {
        std::lock_guard<std::mutex> lk(S(producer.task_slot).fanout_mu);
        EXPECT_TRUE(S(producer.task_slot).fanout_consumers.empty());
        EXPECT_EQ(S(producer.task_slot).fanout_total, 0);
    }

    orch.set_test_hook({});
    EXPECT_NO_THROW(
        orch.fail_run_submission(run_id, std::make_exception_ptr(std::runtime_error("edge publication failed")))
    );
    EXPECT_TRUE(orch.run_done(run_id));
    EXPECT_EQ(S(producer.task_slot).state.load(std::memory_order_acquire), TaskState::CONSUMED);
    orch.release_run(run_id);
}

// Every route back into a ready queue goes through enqueue_ready — including
// the scheduler's requeue of a task whose target worker was busy, which is the
// one that can fire after its run is already terminal.
TEST_F(OrchestratorFixture, ATerminalRunsQueuePartitionIsNotRebuilt) {
    auto task = orch.submit_next_level(C(74), single_tensor_args(0x7400, TensorArgType::OUTPUT), cfg, 0);
    auto sub = orch.submit_sub(C(75), single_tensor_args(0x7500, TensorArgType::OUTPUT));
    TaskSlot ready;
    ASSERT_TRUE(rq.try_pop(run_id, ready));
    ASSERT_EQ(ready, task.task_slot);
    ASSERT_TRUE(rq_sub.try_pop(run_id, ready));
    ASSERT_EQ(ready, sub.task_slot);

    for (TaskSlot slot : {task.task_slot, sub.task_slot}) {
        S(slot).state.store(TaskState::COMPLETED, std::memory_order_release);
        ASSERT_TRUE(orch.on_consumed(slot));
    }
    orch.close_run_submission(run_id);
    ASSERT_TRUE(orch.run_done(run_id));

    // The slots are long gone, but a stale enqueue can still arrive here.
    orch.enqueue_ready(task.task_slot);
    orch.enqueue_ready(sub.task_slot);

    TaskSlot leaked;
    EXPECT_FALSE(rq.try_pop(run_id, leaked)) << "a terminal run's partition was rebuilt by a late enqueue";
    EXPECT_FALSE(rq_next_level.try_front_group(run_id, leaked));
    EXPECT_FALSE(rq_sub.try_pop(run_id, leaked)) << "a busy-worker requeue bypassed the terminal-run guard";
    EXPECT_TRUE(rq_sub.empty());
    orch.release_run(run_id);
}

// A run whose lease has been released is not dispatchable, even while its id is
// still the FIFO head the scheduler last observed.
TEST_F(OrchestratorFixture, ReleasedLeaseMakesTheRunUndispatchable) {
    EXPECT_EQ(orch.dispatchable_run_id(), run_id);
    EXPECT_TRUE(orch.can_dispatch_run(run_id));

    orch.close_run_submission(run_id);
    EXPECT_TRUE(orch.run_done(run_id));

    EXPECT_EQ(orch.dispatchable_run_id(), INVALID_RUN_ID)
        << "a terminal run must not be offered to the scheduler as dispatchable";
    EXPECT_FALSE(orch.can_dispatch_run(run_id));
    orch.release_run(run_id);
}

TEST_F(OrchestratorFixture, FifoHeadCanExecuteWhileGraphConstructionIsOpen) {
    EXPECT_THROW(orch.begin_run(), std::logic_error);
    EXPECT_TRUE(orch.can_dispatch_run(run_id));
    auto task = orch.submit_next_level(C(89), single_tensor_args(0x8900, TensorArgType::OUTPUT), cfg, 0);

    TaskSlot ready;
    ASSERT_TRUE(rq.try_pop(run_id, ready));
    EXPECT_EQ(ready, task.task_slot);
    S(task.task_slot).state.store(TaskState::COMPLETED, std::memory_order_release);
    ASSERT_TRUE(orch.on_consumed(task.task_slot));
    EXPECT_FALSE(orch.run_done(run_id));
    EXPECT_THROW(orch.begin_run(), std::logic_error);

    orch.close_run_submission(run_id);
    EXPECT_TRUE(orch.run_done(run_id));
    orch.release_run(run_id);
}

TEST_F(OrchestratorFixture, BuildingSuccessorActivatesAfterPriorRunIsTerminal) {
    auto first = orch.submit_next_level(C(90), single_tensor_args(0x9000, TensorArgType::OUTPUT), cfg, 0);
    orch.close_run_submission(run_id);

    TaskSlot ready;
    ASSERT_TRUE(rq.try_pop(run_id, ready));
    RunId second = orch.begin_run();
    auto interactive = orch.submit_next_level(C(91), single_tensor_args(0x9100, TensorArgType::OUTPUT), cfg, 0);
    EXPECT_FALSE(orch.can_dispatch_run(second));

    S(first.task_slot).state.store(TaskState::COMPLETED, std::memory_order_release);
    ASSERT_TRUE(orch.on_consumed(first.task_slot));
    EXPECT_TRUE(orch.can_dispatch_run(second));
    ASSERT_TRUE(rq.try_pop(second, ready));
    EXPECT_EQ(ready, interactive.task_slot);

    S(interactive.task_slot).state.store(TaskState::COMPLETED, std::memory_order_release);
    ASSERT_TRUE(orch.on_consumed(interactive.task_slot));
    EXPECT_FALSE(orch.run_done(second));
    orch.close_run_submission(second);
    EXPECT_TRUE(orch.run_done(second));

    orch.release_run(run_id);
    orch.release_run(second);
}

TEST_F(OrchestratorFixture, PreparedRunWaitsForActiveRunAndThirdAdmissionBlocks) {
    auto first = orch.submit_next_level(C(90), single_tensor_args(0x9000, TensorArgType::OUTPUT), cfg, 0);
    orch.close_run_submission(run_id);

    TaskSlot active_slot;
    ASSERT_TRUE(rq.try_pop(run_id, active_slot));
    ASSERT_EQ(active_slot, first.task_slot);

    RunId second = orch.begin_run();
    auto prepared = orch.submit_next_level(C(91), single_tensor_args(0x9100, TensorArgType::OUTPUT), cfg, 0);
    orch.close_run_submission(second);

    EXPECT_FALSE(orch.can_dispatch_run(second));

    auto third = std::async(std::launch::async, [this] {
        return orch.begin_run();
    });
    EXPECT_EQ(third.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout)
        << "third admission must block before graph construction";

    S(first.task_slot).state.store(TaskState::COMPLETED, std::memory_order_release);
    ASSERT_TRUE(orch.on_consumed(first.task_slot));

    ASSERT_EQ(third.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    RunId third_id = third.get();
    ASSERT_TRUE(rq.try_pop(second, active_slot));
    EXPECT_EQ(active_slot, prepared.task_slot);

    S(prepared.task_slot).state.store(TaskState::COMPLETED, std::memory_order_release);
    ASSERT_TRUE(orch.on_consumed(prepared.task_slot));
    orch.close_run_submission(third_id);

    EXPECT_NO_THROW(orch.wait_run(run_id));
    EXPECT_NO_THROW(orch.wait_run(second));
    EXPECT_NO_THROW(orch.wait_run(third_id));
    orch.release_run(run_id);
    orch.release_run(second);
    orch.release_run(third_id);
}

// At depth one the pipeline slot a retiring run gives back is the only one
// there is, so the waiter in begin_run has exactly one wakeup to catch. It is
// caught because the release happens under the mutex whose predicate reads it:
// releasing outside it can land between a waiter finding "full" and its
// registering on the condition variable, and the notify then reaches nobody.
//
// The loop is what exercises the window — a single pass would rarely place the
// release there — so a stranded slot shows up as this test not finishing.
TEST(DepthOneAdmission, ARetiringRunAlwaysWakesTheOneWaiterItUnblocks) {
    for (int round = 0; round < 200; ++round) {
        TensorMap tm;
        Ring allocator;
        Scope scope;
        NextLevelReadyQueues rq_next_level;
        ReadyQueue rq_sub;
        Orchestrator orch;
        allocator.init(/*heap_bytes=*/1ULL << 20);
        rq_next_level.reset({0});
        orch.init(&tm, &allocator, &scope, &rq_sub, &rq_next_level);
        orch.configure_pipeline_depth(1);

        RunId first = orch.begin_run();
        CallConfig cfg;
        TaskArgs args;
        Tensor t{};
        t.buffer.backend_kind = static_cast<uint8_t>(BackendKind::POSIX_SHM);
        t.buffer.access = static_cast<uint8_t>(AccessMode::READWRITE);
        t.buffer.nbytes = 1;
        t.buffer.identity.buffer_id = 0xD000 + static_cast<uint64_t>(round);
        t.ndims = 1;
        t.shapes[0] = 1;
        t.strides[0] = 1;
        t.dtype = DataType::UINT8;
        args.add_tensor(t, TensorArgType::OUTPUT);
        CallableIdentity callable;
        callable.digest.fill(77);
        auto task = orch.submit_next_level(callable, args, cfg, 0);
        orch.close_run_submission(first);
        TaskSlot ready = INVALID_SLOT;
        ASSERT_TRUE(rq_next_level.try_pop_single(0, first, ready));

        // The successor races the retirement for the only slot in the pool.
        auto successor = std::async(std::launch::async, [&orch] {
            return orch.begin_run();
        });
        allocator.slot_state(task.task_slot)->state.store(TaskState::COMPLETED, std::memory_order_release);
        ASSERT_TRUE(orch.on_consumed(task.task_slot));

        ASSERT_EQ(successor.wait_for(std::chrono::seconds(5)), std::future_status::ready)
            << "round " << round << ": the freed pipeline slot never woke its waiter";
        RunId second = successor.get();
        orch.close_run_submission(second);
        orch.release_run(first);
        orch.release_run(second);
        allocator.shutdown();
    }
}

TEST(DepthOneAdmission, FailedBeginRollsBackEveryPublishedOwnerAndKeepsRunIdsMonotonic) {
    for (OrchestratorTestPoint failure_point :
         {OrchestratorTestPoint::BEGIN_RUN_MAP_PUBLISHED, OrchestratorTestPoint::BEGIN_RUN_FIFO_PUBLISHED}) {
        TensorMap tm;
        Ring allocator;
        Scope scope;
        NextLevelReadyQueues rq_next_level;
        ReadyQueue rq_sub;
        Orchestrator orch;
        allocator.init(/*heap_bytes=*/1ULL << 20);
        rq_next_level.reset({0});
        orch.init(&tm, &allocator, &scope, &rq_sub, &rq_next_level);
        orch.configure_pipeline_depth(1);

        bool injected = false;
        orch.set_test_hook([&](OrchestratorTestPoint point) {
            if (point == failure_point && !std::exchange(injected, true)) {
                throw std::runtime_error("injected begin_run publication failure");
            }
        });
        EXPECT_THROW(orch.begin_run(), std::runtime_error);
        ASSERT_TRUE(injected);

        orch.set_test_hook({});
        RunId recovered = orch.begin_run();
        EXPECT_EQ(recovered, 2U);
        EXPECT_EQ(orch.active_run_id(), recovered);
        orch.close_run_submission(recovered);
        EXPECT_TRUE(orch.run_done(recovered));
        orch.release_run(recovered);
        allocator.shutdown();
    }
}

// A submit that throws after scope registration but before the matching charge
// leaves only an extra release, never a reference nothing can release. Python
// closes the scope before cancelling, which is the order used here.
TEST(SubmitFailure, ASlotWhoseSubmitThrewIsFullyReclaimedByCancellation) {
    TensorMap tm;
    Ring allocator;
    Scope scope;
    NextLevelReadyQueues rq_next_level;
    ReadyQueue rq_sub;
    Orchestrator orch;
    allocator.init(/*heap_bytes=*/1ULL << 20);
    rq_next_level.reset({0});
    orch.init(&tm, &allocator, &scope, &rq_sub, &rq_next_level);

    RunId run = orch.begin_run();
    orch.scope_begin();
    bool injected = false;
    orch.set_test_hook([&](OrchestratorTestPoint point) {
        if (point == OrchestratorTestPoint::SCOPE_REGISTERED && !injected) {
            injected = true;
            throw std::bad_alloc();
        }
    });

    CallConfig cfg;
    TaskArgs args;
    Tensor t{};
    t.buffer.backend_kind = static_cast<uint8_t>(BackendKind::POSIX_SHM);
    t.buffer.access = static_cast<uint8_t>(AccessMode::READWRITE);
    t.buffer.nbytes = 1;
    t.buffer.identity.buffer_id = 0xE100;
    t.ndims = 1;
    t.shapes[0] = 1;
    t.strides[0] = 1;
    t.dtype = DataType::UINT8;
    args.add_tensor(t, TensorArgType::OUTPUT);
    CallableIdentity callable;
    callable.digest.fill(88);
    EXPECT_THROW((void)orch.submit_next_level(callable, args, cfg, 0), std::bad_alloc);
    ASSERT_TRUE(injected);

    orch.set_test_hook({});
    orch.scope_end();
    orch.fail_run_submission(run, std::make_exception_ptr(std::runtime_error("graph construction failed")));

    EXPECT_EQ(allocator.slot_state(0)->state.load(std::memory_order_acquire), TaskState::CONSUMED)
        << "a slot charged a scope reference it never registered stalls the run fence";
    EXPECT_TRUE(orch.run_done(run));
    EXPECT_THROW(orch.wait_run(run), std::runtime_error);
    orch.release_run(run);
    allocator.shutdown();
}

TEST_F(OrchestratorFixture, AllocRegistrationFailureReleasesTheUnownedRingSlot) {
    bool injected = false;
    orch.set_test_hook([&](OrchestratorTestPoint point) {
        if (point == OrchestratorTestPoint::ALLOC_RUN_SLOT_REGISTERING && !injected) {
            injected = true;
            throw std::bad_alloc();
        }
    });

    CanonicalIdentity id{};
    id.buffer_id = 1;
    EXPECT_THROW((void)orch.alloc(std::vector<uint32_t>{16}, DataType::UINT8, id), std::bad_alloc);
    ASSERT_TRUE(injected);
    ASSERT_NE(allocator.slot_state(0), nullptr);
    EXPECT_EQ(allocator.slot_state(0)->state.load(std::memory_order_acquire), TaskState::CONSUMED);
    EXPECT_EQ(allocator.active_count(), 0) << "a slot the run never owned was left live in the Ring";

    orch.set_test_hook({});
    EXPECT_NO_THROW(
        orch.fail_run_submission(run_id, std::make_exception_ptr(std::runtime_error("alloc registration failed")))
    );
    EXPECT_TRUE(orch.run_done(run_id));
    EXPECT_THROW(orch.wait_run(run_id), std::runtime_error);
    orch.release_run(run_id);
}

TEST_F(OrchestratorFixture, AllocFailureAfterScopeRegistrationRemainsCancellationClaimable) {
    orch.scope_begin();
    bool injected = false;
    orch.set_test_hook([&](OrchestratorTestPoint point) {
        if (point == OrchestratorTestPoint::SCOPE_REGISTERED && !injected) {
            injected = true;
            throw std::bad_alloc();
        }
    });

    CanonicalIdentity id{};
    id.buffer_id = 1;
    EXPECT_THROW((void)orch.alloc(std::vector<uint32_t>{16}, DataType::UINT8, id), std::bad_alloc);
    ASSERT_TRUE(injected);
    ASSERT_NE(allocator.slot_state(0), nullptr);
    EXPECT_EQ(allocator.slot_state(0)->state.load(std::memory_order_acquire), TaskState::BUILDING);
    EXPECT_EQ(allocator.slot_state(0)->fanout_total, 0)
        << "alloc charged a scope reference before scope registration completed";

    orch.set_test_hook({});
    orch.scope_end();
    EXPECT_NO_THROW(
        orch.fail_run_submission(run_id, std::make_exception_ptr(std::runtime_error("alloc scope failed")))
    );
    EXPECT_EQ(allocator.slot_state(0)->state.load(std::memory_order_acquire), TaskState::CONSUMED);
    EXPECT_EQ(allocator.active_count(), 0);
    EXPECT_TRUE(orch.run_done(run_id));
    EXPECT_THROW(orch.wait_run(run_id), std::runtime_error);
    orch.release_run(run_id);
}

TEST_F(OrchestratorFixture, AllocOutputPublicationFailureLeavesAReclaimableJournal) {
    orch.scope_begin();
    bool injected = false;
    orch.set_test_hook([&](OrchestratorTestPoint point) {
        if (point == OrchestratorTestPoint::ALLOC_OUTPUT_KEY_PREPARED && !injected) {
            injected = true;
            throw std::bad_alloc();
        }
    });

    CanonicalIdentity id{};
    id.buffer_id = 1;
    EXPECT_THROW((void)orch.alloc(std::vector<uint32_t>{16}, DataType::UINT8, id), std::bad_alloc);
    ASSERT_TRUE(injected);
    ASSERT_NE(allocator.slot_state(0), nullptr);
    EXPECT_EQ(allocator.slot_state(0)->state.load(std::memory_order_acquire), TaskState::BUILDING);
    EXPECT_EQ(allocator.slot_state(0)->output_keys.size(), 1u);
    EXPECT_EQ(tm.size(), 0) << "alloc published an output mapping before its cleanup journal";

    orch.set_test_hook({});
    orch.scope_end();
    EXPECT_NO_THROW(
        orch.fail_run_submission(run_id, std::make_exception_ptr(std::runtime_error("alloc output failed")))
    );
    EXPECT_EQ(allocator.slot_state(0)->state.load(std::memory_order_acquire), TaskState::CONSUMED);
    EXPECT_EQ(allocator.active_count(), 0);
    EXPECT_EQ(tm.size(), 0);
    EXPECT_TRUE(orch.run_done(run_id));
    EXPECT_THROW(orch.wait_run(run_id), std::runtime_error);
    orch.release_run(run_id);
}

TEST_F(OrchestratorFixture, SubmitRegistrationFailureReleasesTheUnownedHeapRingSlot) {
    bool injected = false;
    orch.set_test_hook([&](OrchestratorTestPoint point) {
        if (point == OrchestratorTestPoint::SUBMIT_RUN_SLOT_REGISTERING && !injected) {
            injected = true;
            throw std::bad_alloc();
        }
    });

    TaskArgs args;
    Tensor output{};
    output.buffer.backend_kind = static_cast<uint8_t>(BackendKind::POSIX_SHM);
    output.buffer.access = static_cast<uint8_t>(AccessMode::READWRITE);
    output.buffer.nbytes = 1;
    output.buffer.identity.buffer_id = 0;
    output.ndims = 1;
    output.shapes[0] = 16;
    output.strides[0] = 1;
    output.dtype = DataType::UINT8;
    args.add_tensor(output, TensorArgType::OUTPUT);
    EXPECT_THROW((void)orch.submit_next_level(C(90), args, cfg, 0), std::bad_alloc);
    ASSERT_TRUE(injected);
    ASSERT_NE(allocator.slot_state(0), nullptr);
    EXPECT_EQ(allocator.slot_state(0)->state.load(std::memory_order_acquire), TaskState::CONSUMED);
    EXPECT_EQ(allocator.active_count(), 0) << "a task slot absent from its run kept the HeapRing slab live";
    EXPECT_EQ(tm.size(), 0);

    orch.set_test_hook({});
    EXPECT_NO_THROW(
        orch.fail_run_submission(run_id, std::make_exception_ptr(std::runtime_error("submit registration failed")))
    );
    EXPECT_TRUE(orch.run_done(run_id));
    EXPECT_THROW(orch.wait_run(run_id), std::runtime_error);
    orch.release_run(run_id);
}

TEST_F(OrchestratorFixture, SubmitOutputJournalFailurePreservesThePreviousOwnerAndReclaims) {
    constexpr uint64_t first_buffer_id = 0xE300;
    constexpr uint64_t previous_buffer_id = 0xE301;
    // A wire ChipTensor is keyed by its identity's canonical hash, not by a VA.
    TensorKey first_key = ref_key(first_buffer_id);
    TensorKey previous_key = ref_key(previous_buffer_id);

    auto previous =
        orch.submit_next_level(C(91), single_tensor_args(previous_buffer_id, TensorArgType::OUTPUT_EXISTING), cfg, 0);
    TaskSlot ready = INVALID_SLOT;
    ASSERT_TRUE(rq.try_pop(run_id, ready));
    ASSERT_EQ(ready, previous.task_slot);
    TaskState expected = TaskState::READY;
    ASSERT_TRUE(S(previous.task_slot)
                    .state.compare_exchange_strong(
                        expected, TaskState::RUNNING, std::memory_order_acq_rel, std::memory_order_acquire
                    ));

    int32_t prepared_keys = 0;
    orch.set_test_hook([&](OrchestratorTestPoint point) {
        if (point == OrchestratorTestPoint::SUBMIT_OUTPUT_KEY_PREPARED && ++prepared_keys == 2) {
            throw std::bad_alloc();
        }
    });

    TaskArgs replacement;
    Tensor first{};
    first.buffer.backend_kind = static_cast<uint8_t>(BackendKind::POSIX_SHM);
    first.buffer.access = static_cast<uint8_t>(AccessMode::READWRITE);
    first.buffer.nbytes = 1;
    first.buffer.identity = identity_for(first_buffer_id);
    first.ndims = 1;
    first.shapes[0] = 1;
    first.strides[0] = 1;
    first.dtype = DataType::UINT8;
    replacement.add_tensor(first, TensorArgType::OUTPUT_EXISTING);
    Tensor second = first;
    second.buffer.identity = identity_for(previous_buffer_id);
    replacement.add_tensor(second, TensorArgType::OUTPUT_EXISTING);

    EXPECT_THROW((void)orch.submit_next_level(C(92), replacement, cfg, 0), std::bad_alloc);
    ASSERT_EQ(prepared_keys, 2);
    TaskSlot failed_slot = allocator.next_task_id() - 1;
    ASSERT_NE(failed_slot, previous.task_slot);
    EXPECT_EQ(S(failed_slot).state.load(std::memory_order_acquire), TaskState::BUILDING);
    ASSERT_EQ(S(failed_slot).output_keys.size(), 2u);
    EXPECT_EQ(map_producer(tm, run_id, first_key), failed_slot)
        << "the first fully-published output is needed to exercise cancellation cleanup";
    EXPECT_EQ(map_producer(tm, run_id, previous_key), previous.task_slot)
        << "a failed second insert displaced the prior owner before publication";

    orch.set_test_hook({});
    EXPECT_NO_THROW(
        orch.fail_run_submission(run_id, std::make_exception_ptr(std::runtime_error("output publication failed")))
    );
    EXPECT_EQ(S(failed_slot).state.load(std::memory_order_acquire), TaskState::CONSUMED);
    EXPECT_EQ(map_producer(tm, run_id, first_key), INVALID_SLOT);
    EXPECT_EQ(map_producer(tm, run_id, previous_key), previous.task_slot)
        << "failed-slot cleanup erased a TensorMap entry still owned by the running producer";
    EXPECT_FALSE(orch.run_done(run_id));

    S(previous.task_slot).state.store(TaskState::COMPLETED, std::memory_order_release);
    EXPECT_TRUE(orch.on_consumed(previous.task_slot));
    EXPECT_TRUE(orch.run_done(run_id));
    EXPECT_EQ(map_producer(tm, run_id, previous_key), INVALID_SLOT);
    EXPECT_EQ(allocator.active_count(), 0);
    EXPECT_THROW(orch.wait_run(run_id), std::runtime_error);
    orch.release_run(run_id);
}

// Group bookkeeping is prepared by the real submit path while its slot is
// still BUILDING. Fault exactly between the two vector allocations: publishing
// RUNNING first would make this slot neither dispatchable nor cancellable, but
// BUILDING leaves it owned by the submission-failure cancellation path.
TEST_F(OrchestratorFixture, AGroupBookkeepingAllocationFailureLeavesAReclaimableSlot) {
    bool injected = false;
    orch.set_test_hook([&](OrchestratorTestPoint point) {
        if (point == OrchestratorTestPoint::GROUP_MEMBER_STATES_PREPARED && !injected) {
            injected = true;
            throw std::bad_alloc();
        }
    });

    TaskArgs first = single_tensor_args(0xE200, TensorArgType::OUTPUT);
    TaskArgs second = single_tensor_args(0xE201, TensorArgType::OUTPUT);
    EXPECT_THROW((void)orch.submit_sub_group(C(89), {first, second}), std::bad_alloc);
    ASSERT_TRUE(injected);
    ASSERT_NE(allocator.slot_state(0), nullptr);
    EXPECT_EQ(allocator.slot_state(0)->state.load(std::memory_order_acquire), TaskState::BUILDING)
        << "group preparation published a non-cancellable state before both allocations succeeded";

    orch.set_test_hook({});
    EXPECT_NO_THROW(
        orch.fail_run_submission(run_id, std::make_exception_ptr(std::runtime_error("group preparation failed")))
    );
    EXPECT_EQ(allocator.slot_state(0)->state.load(std::memory_order_acquire), TaskState::CONSUMED);
    EXPECT_TRUE(orch.run_done(run_id));
    EXPECT_THROW(orch.wait_run(run_id), std::runtime_error);
    orch.release_run(run_id);
}

TEST_F(OrchestratorFixture, FailedPreparedConstructionReturnsAdmissionWithoutDispatch) {
    auto active = orch.submit_next_level(C(92), single_tensor_args(0x9200, TensorArgType::OUTPUT), cfg, 0);
    orch.close_run_submission(run_id);
    TaskSlot active_slot;
    ASSERT_TRUE(rq.try_pop(run_id, active_slot));

    RunId failed = orch.begin_run();
    auto cancelled = orch.submit_next_level(C(93), single_tensor_args(0x9300, TensorArgType::OUTPUT), cfg, 0);
    orch.fail_run_submission(failed, std::make_exception_ptr(std::runtime_error("graph build failed")));

    EXPECT_TRUE(orch.run_done(failed));
    EXPECT_EQ(S(cancelled.task_slot).state.load(std::memory_order_acquire), TaskState::CONSUMED);
    TaskSlot unexpected;
    EXPECT_FALSE(rq.try_pop(failed, unexpected));

    auto replacement = std::async(std::launch::async, [this] {
        return orch.begin_run();
    });
    ASSERT_EQ(replacement.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    RunId replacement_id = replacement.get();

    S(active.task_slot).state.store(TaskState::COMPLETED, std::memory_order_release);
    ASSERT_TRUE(orch.on_consumed(active.task_slot));
    orch.close_run_submission(replacement_id);

    EXPECT_THROW(orch.wait_run(failed), std::runtime_error);
    EXPECT_NO_THROW(orch.wait_run(replacement_id));
    orch.release_run(run_id);
    orch.release_run(failed);
    orch.release_run(replacement_id);
}
