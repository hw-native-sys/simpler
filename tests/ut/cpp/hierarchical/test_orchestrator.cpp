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

#include "call_config.h"
#include "ring.h"
#include "orchestrator.h"
#include "scope.h"
#include "tensormap.h"
#include "types.h"
#include "task_args.h"

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

    // Helper: build a TaskArgs whose only tensor has the given (data, tag).
    static TaskArgs single_tensor_args(uint64_t data_ptr, TensorArgType tag) {
        TaskArgs a;
        Tensor t{};
        t.buffer.addr = data_ptr;
        t.ndims = 1;
        t.shapes[0] = 1;
        t.dtype = DataType::UINT8;
        a.add_tensor(t, tag);
        return a;
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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

TEST_F(OrchestratorFixture, TensorMapTracksProducer) {
    auto args_a = single_tensor_args(0x1234, TensorArgType::OUTPUT);
    auto a = orch.submit_next_level(C(42), args_a, cfg, 0);
    TaskSlot drain_slot;
    rq.try_pop(drain_slot);

    EXPECT_EQ(tm.lookup(run_id, TensorKey{0x1234, -1}), a.task_slot);
}

TEST_F(OrchestratorFixture, OnConsumedCleansUpTensorMap) {
    auto args_a = single_tensor_args(0x42, TensorArgType::OUTPUT);
    auto a = orch.submit_next_level(C(42), args_a, cfg, 0);
    TaskSlot slot;
    rq.try_pop(slot);

    EXPECT_EQ(tm.lookup(run_id, TensorKey{0x42, -1}), slot);

    S(slot).state.store(TaskState::COMPLETED, std::memory_order_relaxed);
    orch.on_consumed(slot);

    EXPECT_EQ(tm.lookup(run_id, TensorKey{0x42, -1}), INVALID_SLOT);
    EXPECT_EQ(S(slot).state.load(), TaskState::CONSUMED);
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
    EXPECT_EQ(S(res.task_slot).args_view(0).tensors(0).buffer.addr, 0xA0u);
    EXPECT_EQ(S(res.task_slot).args_view(1).tensors(0).buffer.addr, 0xA1u);

    // Both keys registered as producers for the group slot.
    EXPECT_EQ(tm.lookup(run_id, TensorKey{0xA0, -1}), res.task_slot);
    EXPECT_EQ(tm.lookup(run_id, TensorKey{0xA1, -1}), res.task_slot);
}

TEST_F(OrchestratorFixture, SingleTaskStoresTaskArgsDirectly) {
    TaskArgs a0 = single_tensor_args(0xC0, TensorArgType::OUTPUT);
    auto res = orch.submit_next_level(C(42), a0, cfg, 0);
    ASSERT_NE(res.task_slot, INVALID_SLOT);
    EXPECT_FALSE(S(res.task_slot).is_group());
    EXPECT_EQ(S(res.task_slot).group_size(), 1);
    EXPECT_EQ(S(res.task_slot).task_args.tensor_count(), 1);
    EXPECT_EQ(S(res.task_slot).args_view(0).tensors(0).buffer.addr, 0xC0u);
}

TEST_F(OrchestratorFixture, OutputAutoAllocsFromHeapRing) {
    // An OUTPUT tensor submitted with `data == 0` is auto-allocated from
    // the HeapRing: the slot's task_args tensor ends up with a non-zero
    // data pointer that falls inside the allocator's mmap'd region, and
    // the TensorMap routes that pointer to the slot.
    TaskArgs args;
    Tensor t{};
    t.buffer.addr = 0;
    t.ndims = 1;
    t.shapes[0] = 1024;  // 1024 * 1 byte = 1024, one aligned slab
    t.dtype = DataType::UINT8;
    args.add_tensor(t, TensorArgType::OUTPUT);

    auto res = orch.submit_next_level(C(42), args, cfg, 0);
    ASSERT_NE(res.task_slot, INVALID_SLOT);

    uint64_t data = S(res.task_slot).task_args.tensor(0).buffer.addr;
    ASSERT_NE(data, 0u);
    uintptr_t base = reinterpret_cast<uintptr_t>(allocator.heap_base(0));
    EXPECT_GE(data, base);
    EXPECT_LT(data, base + allocator.heap_size(0));
    EXPECT_EQ(data % HEAP_ALIGN, 0u);

    EXPECT_EQ(tm.lookup(run_id, TensorKey{data, -1}), res.task_slot);
}

TEST_F(OrchestratorFixture, RemoteOutputSidecarSkipsLocalAutoAllocAndRegistersRemoteKey) {
    TaskArgs args;
    Tensor t{};
    t.buffer.addr = 0;
    t.ndims = 1;
    t.shapes[0] = 1024;
    t.dtype = DataType::UINT8;
    args.add_tensor(t, TensorArgType::OUTPUT);

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
    EXPECT_EQ(S(res.task_slot).task_args.tensor(0).buffer.addr, 0u);

    TensorKey key = TensorKey::remote_buffer(TensorAddressKind::REMOTE_BUFFER, 3, 9, 2, 64);
    EXPECT_EQ(tm.lookup(run_id, key), res.task_slot);
}

TEST_F(OrchestratorFixture, RemoteBarePayloadFailsBeforeSlotCommit) {
    TaskArgs args = single_tensor_args(0x1234, TensorArgType::INPUT);

    RemoteTaskArgsSidecar sidecar;
    sidecar.tensors.resize(1);

    EXPECT_THROW({ (void)orch.submit_next_level(C(42), args, cfg, 3, {3}, sidecar); }, std::invalid_argument);
}

TEST_F(OrchestratorFixture, RemoteSidecarRejectsNonOwnerEligibleEndpointWithoutImport) {
    TaskArgs args;
    Tensor t{};
    t.buffer.addr = 0;
    t.ndims = 1;
    t.shapes[0] = 1;
    t.dtype = DataType::UINT8;
    args.add_tensor(t, TensorArgType::INPUT);

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
    TaskArgs output_args;
    Tensor out{};
    out.buffer.addr = 0;
    out.ndims = 1;
    out.shapes[0] = 1;
    out.dtype = DataType::UINT8;
    output_args.add_tensor(out, TensorArgType::OUTPUT);

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

    TaskArgs input_args;
    Tensor in = out;
    input_args.add_tensor(in, TensorArgType::INPUT);
    auto consumer = orch.submit_next_level(C(43), input_args, cfg, 3, {3}, output_sidecar);

    EXPECT_EQ(S(consumer.task_slot).state.load(), TaskState::PENDING);
    EXPECT_EQ(S(consumer.task_slot).fanin_count, 1);
    ASSERT_EQ(S(consumer.task_slot).fanin_producers.size(), 1u);
    EXPECT_EQ(S(consumer.task_slot).fanin_producers[0], producer.task_slot);
}

TEST_F(OrchestratorFixture, InoutWiresCreatorAsFanin) {
    // INOUT is the only tag that pulls in the prior writer as a fanin
    // producer -- matching L2's pto_orchestrator.cpp Step B where only
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
    EXPECT_EQ(tm.lookup(run_id, TensorKey{0xFEED, -1}), writer.task_slot);
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

        EXPECT_EQ(tm.lookup(run_id, TensorKey{c.key, -1}), writer.task_slot);
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
