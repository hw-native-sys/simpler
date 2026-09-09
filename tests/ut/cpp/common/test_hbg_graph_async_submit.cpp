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
#include <condition_variable>
#include <functional>
#include <cstdint>
#include <mutex>
#include <set>
#include <thread>

#include "host_build_graph/graph_recorder_pool.h"
#include "orchestration_api.h"

namespace {

// The handshakes below wait for an event the peer thread is guaranteed to
// publish, so this bound only decides how long a genuine failure takes to
// report. All of this repo's self-hosted CPU runners share one machine, so it is
// budgeted for a badly loaded host rather than for the expected microseconds.
constexpr std::chrono::seconds kHandshakeTimeout{5};

RuntimeContext *g_bound_runtime = nullptr;

extern "C" RuntimeContext *framework_current_runtime(void) { return g_bound_runtime; }
extern "C" void framework_bind_runtime(RuntimeContext *rt) { g_bound_runtime = rt; }

// One recording's formal parameters and the tensor storage its TensorRefs point at.
// The two travel together because gen_scalar_params_from_args resolves against storage
// this object owns, the way an entry's GraphBoundary does.
struct FakeBoundary {
    std::array<simpler::hbg::Tensor, GRAPH_MAX_TENSOR_ARGS> tensors{};
    GraphTaskArgs params;
};

struct FakeRuntime {
    const RuntimeOps *ops;
    ScopeMode pending_scope_mode{ScopeMode::AUTO};
    std::mutex mutex;
    std::condition_variable cv;
    bool record_entered{false};
    bool later_submit_entered{false};
    bool overlapped{false};
    bool record_every_begin{false};
    bool gate_four_prepares{false};
    int four_prepare_overlaps{0};
    int prepare_overlaps{0};
    int begin_calls{0};
    int prepare_calls{0};
    int end_calls{0};
    // A recorded body reports a fatal on the recorder thread while the bind thread
    // reads it, which is what the runtime's own fatal_code is atomic for.
    std::atomic<bool> fatal{false};
    // graph_end's view of the fatal at the moment it ran, so a test can tell "end
    // was reached after the body latched" from "end was reached at all".
    bool end_saw_fatal{false};
    std::atomic<int> abort_calls{0};
    const void *abort_handle{nullptr};
    // Written from a recorder worker (the recorded body calls rt_graph_commit) and from
    // the main thread, so it cannot be a plain int.
    std::atomic<int> commit_calls{0};
    int scope_begin_calls{0};
    int scope_end_calls{0};
    std::thread::id record_thread;
    std::thread::id prepare_thread;
    std::thread::id submit_thread;

    // What graph_prepare actually received, so the parameter list the body resolves
    // against can be compared against the boundary the caller passed.
    bool prepare_saw_args{false};
    const void *prepare_handle{nullptr};
    int32_t recorded_tensor_count{-1};
    int32_t recorded_scalar_count{-1};
    uint64_t recorded_tensor_addr{0};
    uint64_t recorded_tensor_size{0};
    uint32_t recorded_ndims{0};
    uint64_t recorded_scalar{0};
    TensorArgType recorded_tag{};
    const void *recorded_args_object{nullptr};
    const void *recorded_tensor_storage{nullptr};

    // Stands in for the in-flight entry's own copy of the formal parameters: the real
    // graph_begin builds one per entry and hands it out through GraphScopeResult, and the
    // recorded body reads that rather than the caller's arguments. One per recording, so
    // a queued body still reads what its own begin built after a later begin has run.
    // The storage is owned by the test, not by this struct, which is what keeps
    // FakeRuntime standard-layout so the offsetof guards below stay well-defined.
    FakeBoundary *boundaries{nullptr};
    int32_t boundary_capacity{0};
};

static_assert(offsetof(FakeRuntime, ops) == 0);
static_assert(offsetof(FakeRuntime, pending_scope_mode) == offsetof(RuntimeContext, pending_scope_mode));

FakeRuntime *as_fake(RuntimeContext *rt) { return reinterpret_cast<FakeRuntime *>(rt); }

bool fake_is_fatal(RuntimeContext *rt) { return as_fake(rt)->fatal.load(std::memory_order_acquire); }

GraphScopeResult fake_graph_begin(RuntimeContext *rt, uint64_t, const GraphTaskArgs &args) {
    FakeRuntime &fake = *as_fake(rt);
    std::lock_guard<std::mutex> lock(fake.mutex);
    fake.begin_calls++;
    GraphScopeResult result;
    result.execute_block = false;
    result.recording = fake.record_every_begin || fake.begin_calls == 1;
    if (result.recording) {
        // Each recording resolves against its own list, so a body queued by an earlier
        // begin is unaffected by a later one -- as with an entry's own GraphBoundary.
        // A non-void return rules out ASSERT_, and EXPECT_ would record the failure and
        // then index past the array anyway, so a test owning too few boundaries reports
        // no recording instead.
        if (fake.begin_calls > fake.boundary_capacity) {
            ADD_FAILURE() << "test must own one boundary per recording";
            result.recording = false;
            return result;
        }
        FakeBoundary &boundary = fake.boundaries[fake.begin_calls - 1];
        // Deep-copy the parameters the way graph_begin does: tensors first, so the
        // TensorRefs that follow point at storage this object owns, then resolved values.
        boundary.params.reset();
        for (int32_t i = 0; i < args.tensor_count(); ++i) {
            boundary.tensors[static_cast<size_t>(i)] = args.tensor(i).ref();
        }
        for (int32_t i = 0; i < args.tensor_count(); ++i) {
            boundary.params.add_input(boundary.tensors[static_cast<size_t>(i)]);
        }
        boundary.params.gen_scalar_params_from_args(args);
        // The handle the recording thread must hand back to graph_prepare, and the
        // parameters the recorded body reads.
        result.recording_handle = &fake;
        result.params = &boundary.params;
    }
    if (!result.recording) {
        fake.later_submit_entered = true;
        fake.cv.notify_all();
    }
    return result;
}

bool fake_graph_prepare(RuntimeContext *rt, void *recording_handle, const GraphTaskArgs &args) {
    FakeRuntime &fake = *as_fake(rt);
    std::unique_lock<std::mutex> lock(fake.mutex);
    fake.prepare_calls++;
    fake.prepare_handle = recording_handle;
    fake.prepare_thread = std::this_thread::get_id();
    fake.prepare_saw_args = true;
    fake.recorded_tensor_count = args.tensor_count();
    fake.recorded_scalar_count = args.scalar_count();
    fake.recorded_args_object = &args;
    if (args.tensor_count() > 0) {
        const simpler::hbg::Tensor &tensor = args.tensor(0).ref();
        fake.recorded_tensor_addr = tensor.buffer.addr;
        fake.recorded_tensor_size = tensor.buffer.size;
        fake.recorded_ndims = tensor.ndims;
        fake.recorded_tag = args.tag(0);
        fake.recorded_tensor_storage = &tensor;
    }
    if (args.scalar_count() > 0) {
        fake.recorded_scalar = args.scalar(0).to<uint64_t>();
    }
    if (fake.gate_four_prepares) {
        fake.cv.notify_all();
        const bool all_prepared = fake.cv.wait_for(lock, kHandshakeTimeout, [&] {
            return fake.prepare_calls == 4;
        });
        if (all_prepared) fake.four_prepare_overlaps++;
        return true;
    }
    // The hash-keyed outer shell is enough for the caller to continue. Hold
    // prepare until the later same-hash submission proves that start() did not
    // retain the old worker-ready handshake.
    const bool saw_later_submit = fake.cv.wait_for(lock, kHandshakeTimeout, [&] {
        return fake.later_submit_entered;
    });
    if (saw_later_submit) fake.prepare_overlaps++;
    return true;
}

void fake_graph_abort(RuntimeContext *rt, void *recording_handle) {
    FakeRuntime &fake = *as_fake(rt);
    fake.abort_handle = recording_handle;
    fake.abort_calls.fetch_add(1, std::memory_order_acq_rel);
}

bool fake_graph_end(RuntimeContext *rt) {
    FakeRuntime &fake = *as_fake(rt);
    std::lock_guard<std::mutex> lock(fake.mutex);
    fake.end_calls++;
    fake.submit_thread = std::this_thread::get_id();
    // Mirrors the real graph_end's contract: on a fatal it retires the entry itself
    // and reports that no Definition was published.
    fake.end_saw_fatal = fake.fatal.load(std::memory_order_acquire);
    return !fake.end_saw_fatal;
}

void fake_graph_commit(RuntimeContext *rt) { as_fake(rt)->commit_calls++; }

void fake_scope_begin(RuntimeContext *rt) { as_fake(rt)->scope_begin_calls++; }

void fake_scope_end(RuntimeContext *rt) { as_fake(rt)->scope_end_calls++; }

// The pool is the runtime's now, so rt_graph_submit reaches it through these two rather
// than through a static in the orchestration header. Wiring them to the same process-wide
// pool the runtime uses is what keeps this test exercising the asynchronous path -- with
// them null, rt_graph_submit correctly falls back to recording inline and the test would
// be asserting the fallback instead.
// This test's own pool, not the runtime's: the point is the pool's behaviour, and a
// file-local instance keeps the test from linking the runtime translation unit that owns
// the process-wide one.
GraphAsyncRecordingState &test_pool() {
    static GraphAsyncRecordingState pool;
    return pool;
}

bool fake_graph_record_start(RuntimeContext *, const GraphTaskArgs &args, void *job) {
    auto *record = static_cast<std::function<void(const GraphTaskArgs &)> *>(job);
    return test_pool().start(args, std::move(*record));
}

void fake_graph_record_wait(RuntimeContext *) { test_pool().wait(); }

const RuntimeOps kFakeOps = {
    .scope_begin = fake_scope_begin,
    .scope_end = fake_scope_end,
    .is_fatal = fake_is_fatal,
    .graph_begin = fake_graph_begin,
    .graph_prepare = fake_graph_prepare,
    .graph_abort = fake_graph_abort,
    .graph_end = fake_graph_end,
    .graph_commit = fake_graph_commit,
    .graph_record_start = fake_graph_record_start,
    .graph_record_wait = fake_graph_record_wait,
};

}  // namespace

TEST(HbgGraphAsyncSubmit, PrewarmedRecorderPoolGrowsPastThePrewarmedCount) {
    // One more than the prewarmed count, so the pool must create a worker for
    // the last job rather than queue it behind a busy one.
    constexpr int kGraphCount = 9;
    GraphAsyncRecordingState pool;
    ASSERT_TRUE(pool.prewarm());
    std::mutex gate_mutex;
    std::condition_variable gate_cv;
    int entered = 0;
    bool release = false;
    std::set<std::thread::id> worker_ids;
    GraphTaskArgs empty_args;

    for (int i = 0; i < kGraphCount; ++i) {
        ASSERT_TRUE(pool.start(empty_args, [&](const GraphTaskArgs &) {
            std::unique_lock<std::mutex> lock(gate_mutex);
            worker_ids.insert(std::this_thread::get_id());
            entered++;
            gate_cv.notify_all();
            gate_cv.wait(lock, [&]() {
                return release;
            });
        }));
    }

    bool all_entered = false;
    {
        std::unique_lock<std::mutex> lock(gate_mutex);
        all_entered = gate_cv.wait_for(lock, kHandshakeTimeout, [&]() {
            return entered == kGraphCount;
        });
        release = true;
    }
    gate_cv.notify_all();
    pool.wait();

    EXPECT_TRUE(all_entered) << "every Graph recording must start before any one of them finishes";
    EXPECT_EQ(worker_ids.size(), static_cast<size_t>(kGraphCount))
        << "a concurrent miss past the prewarmed count must grow the pool instead of queueing";
}

TEST(HbgGraphAsyncSubmit, FourDistinctGraphMissesDoNotInsertAnIntermediateCommit) {
    std::array<FakeBoundary, 4> boundaries;
    FakeRuntime fake{};
    fake.ops = &kFakeOps;
    fake.boundaries = boundaries.data();
    fake.boundary_capacity = static_cast<int32_t>(boundaries.size());
    fake.record_every_begin = true;
    fake.gate_four_prepares = true;
    framework_bind_runtime(reinterpret_cast<RuntimeContext *>(&fake));

    uint32_t storage[4]{};
    uint32_t shape[] = {4};
    simpler::hbg::Tensor boundary = simpler::hbg::make_tensor_external(storage, shape, 1);
    GraphTaskArgs args;
    args.add_input(boundary);

    int body_calls = 0;
    for (uint64_t graph_key = 0x1800; graph_key < 0x1804; ++graph_key) {
        ScopeGuard scope;
        const GraphSubmitResult result = rt_submit_graph_impl(graph_key, args, [&](const GraphTaskArgs &) {
            std::lock_guard<std::mutex> lock(fake.mutex);
            body_calls++;
        });
        EXPECT_TRUE(result.recording);
    }
    rt_graph_commit();
    framework_bind_runtime(nullptr);

    EXPECT_EQ(fake.begin_calls, 4);
    EXPECT_EQ(fake.prepare_calls, 4);
    EXPECT_EQ(fake.four_prepare_overlaps, 4) << "all four recorders must enter before commit waits for any Definition";
    EXPECT_EQ(fake.end_calls, 4);
    EXPECT_EQ(fake.commit_calls, 1) << "only the explicit final commit may wait for the recorder pool";
    EXPECT_EQ(body_calls, 4);
}

TEST(HbgGraphAsyncSubmit, WorkerRecordsWhileMainSubmitsLaterGraphs) {
    std::array<FakeBoundary, 4> boundaries;
    FakeRuntime fake{};
    fake.ops = &kFakeOps;
    fake.boundaries = boundaries.data();
    fake.boundary_capacity = static_cast<int32_t>(boundaries.size());
    framework_bind_runtime(reinterpret_cast<RuntimeContext *>(&fake));

    uint32_t storage[4]{};
    uint32_t shape[] = {4};
    simpler::hbg::Tensor boundary = simpler::hbg::make_tensor_external(storage, shape, 1);
    GraphTaskArgs args;
    args.add_input(boundary);

    int body_calls = 0;
    const std::thread::id caller = std::this_thread::get_id();
    GraphSubmitResult first;
    {
        ScopeGuard scope;
        first = rt_submit_graph_impl(0x1715, args, [&](const GraphTaskArgs &) {
            // An explicit commit reached from a Graph body must not make the
            // recorder wait for its own in-flight job.
            rt_graph_commit();
            std::unique_lock<std::mutex> lock(fake.mutex);
            body_calls++;
            fake.record_entered = true;
            fake.record_thread = std::this_thread::get_id();
            fake.cv.notify_all();
            const bool saw_later_submit = fake.cv.wait_for(lock, kHandshakeTimeout, [&] {
                return fake.later_submit_entered;
            });
            fake.overlapped |= saw_later_submit;
        });
    }
    GraphSubmitResult second;
    {
        ScopeGuard scope;
        second = rt_submit_graph_impl(0x1715, args, [&](const GraphTaskArgs &) {
            ADD_FAILURE() << "a later submission for an in-flight Graph must not execute the body";
        });
    }
    rt_graph_commit();

    {
        std::lock_guard<std::mutex> lock(fake.mutex);
        fake.begin_calls = 0;
        fake.record_entered = false;
        fake.later_submit_entered = false;
        fake.overlapped = false;
    }
    {
        ScopeGuard scope;
        first = rt_submit_graph_impl(0x1715, args, [&](const GraphTaskArgs &) {
            rt_graph_commit();
            std::unique_lock<std::mutex> lock(fake.mutex);
            body_calls++;
            fake.record_entered = true;
            fake.record_thread = std::this_thread::get_id();
            fake.cv.notify_all();
            const bool saw_later_submit = fake.cv.wait_for(lock, kHandshakeTimeout, [&] {
                return fake.later_submit_entered;
            });
            fake.overlapped |= saw_later_submit;
        });
    }
    {
        ScopeGuard scope;
        second = rt_submit_graph_impl(0x1715, args, [&](const GraphTaskArgs &) {
            ADD_FAILURE() << "a later submission for an in-flight Graph must not execute the body";
        });
    }
    rt_graph_commit();

    framework_bind_runtime(nullptr);
    EXPECT_TRUE(first.recording);
    EXPECT_FALSE(second.recording);
    EXPECT_EQ(fake.begin_calls, 2);
    EXPECT_EQ(body_calls, 2);
    EXPECT_EQ(fake.prepare_calls, 2);
    EXPECT_EQ(fake.prepare_overlaps, 2) << "main-thread Graph submission must not wait for worker graph_prepare";
    EXPECT_EQ(fake.prepare_handle, &fake) << "the recording thread must bind through the handle graph_begin returned";
    EXPECT_EQ(fake.end_calls, 2);
    EXPECT_EQ(fake.commit_calls, 4);
    EXPECT_EQ(fake.scope_begin_calls, 4);
    EXPECT_EQ(fake.scope_end_calls, 4);
    EXPECT_TRUE(fake.overlapped);
    EXPECT_NE(fake.record_thread, caller);
    EXPECT_EQ(fake.prepare_thread, fake.record_thread);
    EXPECT_EQ(fake.submit_thread, fake.record_thread);
}

// The recorded body runs on the worker after the caller's frame is free to end,
// so it must read a boundary the worker owns. GraphOwnedArgs is that copy; this
// pins both halves of its contract — the values survive, and the storage they
// live in is not the caller's.
TEST(HbgGraphAsyncSubmit, RecordingReadsAnOwnedCopyOfTheBoundary) {
    std::array<FakeBoundary, 4> boundaries;
    FakeRuntime fake{};
    fake.ops = &kFakeOps;
    fake.boundaries = boundaries.data();
    fake.boundary_capacity = static_cast<int32_t>(boundaries.size());
    framework_bind_runtime(reinterpret_cast<RuntimeContext *>(&fake));

    uint32_t storage[8]{};
    uint32_t shape[] = {8};
    simpler::hbg::Tensor boundary = simpler::hbg::make_tensor_external(storage, shape, 1);
    constexpr uint64_t kScalar = 0x5eed1715ULL;
    GraphTaskArgs args;
    args.add_input(boundary);
    args.add_scalar(kScalar);

    const void *caller_args_object = &args;
    const void *caller_tensor_storage = &args.tensor(0).ref();

    // This case is about the copy, not the overlap: release the gate that
    // fake_graph_prepare waits on so it returns immediately instead of sitting
    // out the handshake timeout no later submission is going to satisfy.
    fake.later_submit_entered = true;

    {
        ScopeGuard scope;
        (void)rt_submit_graph_impl(0x1719, args, [](const GraphTaskArgs &) {});
    }
    rt_graph_commit();
    framework_bind_runtime(nullptr);

    ASSERT_TRUE(fake.prepare_saw_args);
    EXPECT_EQ(fake.recorded_tensor_count, args.tensor_count());
    EXPECT_EQ(fake.recorded_scalar_count, args.scalar_count());
    EXPECT_EQ(fake.recorded_tensor_addr, boundary.buffer.addr);
    EXPECT_EQ(fake.recorded_tensor_size, boundary.buffer.size);
    EXPECT_EQ(fake.recorded_ndims, boundary.ndims);
    EXPECT_EQ(fake.recorded_tag, TensorArgType::INPUT);
    EXPECT_EQ(fake.recorded_scalar, kScalar);

    EXPECT_NE(fake.recorded_args_object, caller_args_object) << "the worker must not read the caller's GraphTaskArgs";
    EXPECT_NE(fake.recorded_tensor_storage, caller_tensor_storage)
        << "the worker must not read simpler::hbg::Tensor storage the caller owns";
}

// A fatal reported inside a recorded body has to reach graph_end. graph_end is the
// only thing that transitions the in-flight entry out of RECORDING, and
// graph_commit's drain blocks on recording_cv until every entry has — with no
// timeout, on the bind thread. rt_graph_end used to short-circuit on is_fatal() and
// report success, which skipped the retire and hung that drain.
//
// The same case pins the other half: the wrapper must NOT follow a declined end with
// an abort. graph_end retires the entry it bound on every path that has one, so a
// caller-side abort is always a second one — and graph_commit frees a drained entry
// after releasing recording_mutex, so the second abort has nothing left to
// synchronize against and writes to freed memory.
//
// Neither half is reachable from a test that calls OrchestratorState::graph_end()
// directly: both live in the wrapper, which is why this case is here and not with the
// orchestrator's own graph tests.
TEST(HbgGraphAsyncSubmit, AFatalInsideARecordedBodyReachesGraphEndAndAbortsNothing) {
    std::array<FakeBoundary, 4> boundaries;
    FakeRuntime fake{};
    fake.ops = &kFakeOps;
    fake.boundaries = boundaries.data();
    fake.boundary_capacity = static_cast<int32_t>(boundaries.size());
    framework_bind_runtime(reinterpret_cast<RuntimeContext *>(&fake));

    uint32_t storage[4]{};
    uint32_t shape[] = {4};
    simpler::hbg::Tensor boundary = simpler::hbg::make_tensor_external(storage, shape, 1);
    GraphTaskArgs args;
    args.add_input(boundary);

    // No later submission to overlap with, so release the handshake gate rather than
    // letting fake_graph_prepare sit out a timeout nothing will satisfy.
    fake.later_submit_entered = true;

    int body_calls = 0;
    {
        ScopeGuard scope;
        (void)rt_submit_graph_impl(0x1722, args, [&](const GraphTaskArgs &) {
            body_calls++;
            // Stands in for the body's own rt_report_fatal: from the wrapper's side a
            // fatal is just is_fatal() turning true partway through the pass.
            fake.fatal.store(true, std::memory_order_release);
        });
    }
    // Drains the recorder pool through the ops table, as orchestration completion does.
    rt_graph_commit();
    framework_bind_runtime(nullptr);

    EXPECT_EQ(body_calls, 1);
    EXPECT_EQ(fake.end_calls, 1) << "a fatal must not stop the recording pass from reaching graph_end";
    EXPECT_TRUE(fake.end_saw_fatal) << "graph_end has to observe the fatal — it is the retire point";
    EXPECT_EQ(fake.abort_calls.load(std::memory_order_acquire), 0)
        << "graph_end retires its own entry, so a caller-side abort would be a second one racing graph_commit's free";
}
