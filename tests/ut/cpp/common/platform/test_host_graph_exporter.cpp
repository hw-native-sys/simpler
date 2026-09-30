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
// Who owns a host-built dependency graph between the thread that built it and
// the file it becomes.
//
// The exporter is driven through its real `seal` / `flush_retained_runs` /
// `finish_retained_runs`, with its real writer thread, budget, error record and
// publication. Only the hand-off is supplied by the case: `seal` takes it as a
// function pointer precisely so a test can stand in for the runtime's
// thread-local capture without linking a runtime.
//
// The races are driven, not hoped for. A pause inside the hand-off holds a lease
// open at a known point, which is what makes "flush must wait for an
// unqueued write" and "finish must not return under an open lease"
// deterministic rather than timing-dependent.

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <new>
#include <string>
#include <thread>

#include "host/host_graph_exporter.h"

namespace fs = std::filesystem;
namespace hg = simpler::dfx::host_graph;

// --- allocation-failure injection -------------------------------------------
//
// `publish` stages its temporary and only then allocates — the path strings,
// the stream's construction and the body's serialization all do — so an
// allocation that fails is the only way into its `catch`, and the only way to
// cover what that exit leaves on disk.
//
// Only the throwing `operator new` is replaced, so every deallocation still
// runs the default, which releases what `std::malloc` returned. The aligned
// and nothrow forms are left alone: they pair with deletes of their own, and
// passing them through costs the sweep nothing but a few injection points.
static std::atomic<long> g_alloc_countdown{-1};  // negative: pass everything through
static std::atomic<long> g_alloc_count{0};

void *operator new(std::size_t size) {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
    // Disarmed as it fires, so one arming is one failure: the recovery path
    // below allocates too, and a latched countdown would fail that instead.
    if (g_alloc_countdown.load(std::memory_order_relaxed) == 0) {
        g_alloc_countdown.store(-1, std::memory_order_relaxed);
        throw std::bad_alloc();
    }
    const long remaining = g_alloc_countdown.load(std::memory_order_relaxed);
    if (remaining > 0) g_alloc_countdown.store(remaining - 1, std::memory_order_relaxed);
    void *p = std::malloc(size == 0 ? 1 : size);
    if (p == nullptr) throw std::bad_alloc();
    return p;
}

namespace {

/** Fail the `nth` allocation from now, once. */
void arm_alloc_failure(long nth) { g_alloc_countdown.store(nth, std::memory_order_relaxed); }
void disarm_alloc_failure() { g_alloc_countdown.store(-1, std::memory_order_relaxed); }

/** A unique directory tree per case, removed with it. */
class OutputRoot {
public:
    explicit OutputRoot(const char *name) {
        root_ = fs::temp_directory_path() / ("host_graph_" + std::string(name) + "_" + std::to_string(::getpid()));
        fs::remove_all(root_);
        fs::create_directories(root_);
    }
    ~OutputRoot() {
        std::error_code ec;
        fs::remove_all(root_, ec);
    }
    std::string prefix(const char *run) const { return (root_ / run).string(); }
    fs::path artifact(const char *run) const { return root_ / run / "deps.json"; }
    fs::path temporary(const char *run) const { return root_ / run / "deps.json.tmp"; }

private:
    fs::path root_;
};

/**
 * Wait for a state to hold, bounded.
 *
 * Every observation in this file goes through here: a sleep proves nothing
 * about another thread's progress, and a case that waited on one would pass or
 * hang by scheduling luck.
 */
template <typename Pred>
bool wait_for_state(Pred pred, std::chrono::milliseconds budget = std::chrono::milliseconds(10000)) {
    const auto deadline = std::chrono::steady_clock::now() + budget;
    while (!pred()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

/** Fill a graph with `tasks` tasks, one arg and one edge each. */
void build(hg::HostGraphExport *out, uint32_t tasks) {
    out->task_arg_offsets.push_back(0);
    for (uint32_t i = 0; i < tasks; i++) {
        hg::TaskEntry t;
        t.task_id = 0x1000 + i;
        out->tasks.push_back(t);
        hg::TaskArgEntry a{};
        a.idx = 0;
        a.arg_type = hg::kArgInout;
        a.has_tensor_info = true;
        a.tensor_id = 0x77;
        a.ndims = 1;
        a.shape[0] = 8;
        a.strides[0] = 1;
        out->task_args.push_back(a);
        out->task_arg_offsets.push_back(static_cast<uint32_t>(out->task_args.size()));
        if (i > 0) {
            hg::EdgeEntry e{};
            e.pred = 0x1000 + i - 1;
            e.succ = 0x1000 + i;
            e.consumer_arg_idx = -1;
            e.kind = hg::EdgeKind::Explicit;
            out->edges.push_back(e);
        }
    }
    // The tensor is the storage the args above name, so it exists exactly when
    // one of them does: a capture of no tasks holds no tensor either.
    if (tasks == 0) return;
    hg::TensorEntry tensor{};
    tensor.tensor_id = 0x77;
    tensor.buffer_addr = 0xdead0000;
    tensor.buffer_numel = 8;
    out->tensors.push_back(tensor);
}

// The hand-off the exporter is given. Static because it must be a plain
// function pointer, which is what keeps the platform layer free of runtime
// symbols — the same shape the production `take` has.
std::atomic<uint32_t> g_take_tasks{4};
std::atomic<int> g_take_outcome{static_cast<int>(hg::TakeOutcome::Complete)};
std::atomic<bool> g_take_pause{false};
std::atomic<bool> g_take_entered{false};

int take_stub(hg::HostGraphExport *out) {
    g_take_entered.store(true);
    while (g_take_pause.load())
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const int outcome = g_take_outcome.load();
    if (outcome != static_cast<int>(hg::TakeOutcome::Complete)) return outcome;
    build(out, g_take_tasks.load());
    return outcome;
}

void reset_take(uint32_t tasks = 4) {
    g_take_tasks.store(tasks);
    g_take_outcome.store(static_cast<int>(hg::TakeOutcome::Complete));
    g_take_pause.store(false);
    g_take_entered.store(false);
}

class HostGraphExporterTest : public ::testing::Test {
protected:
    void SetUp() override { reset_take(); }
    void TearDown() override {
        g_take_pause.store(false);
        exporter_.pause_publication_for_test(false);
        exporter_.finish_retained_runs();
    }

    void configure(bool retained, size_t budget = simpler::dfx::runs::kDefaultBudgetBytes) {
        exporter_.configure_retained_runs(retained, budget);
    }

    hg::HostGraphExporter exporter_;
};

/** A graph handed over with room for it lands on disk after the flush. */
TEST_F(HostGraphExporterTest, ARetainedGraphIsPublishedInTheBackground) {
    configure(true);
    OutputRoot root("retained");

    exporter_.pause_publication_for_test(true);
    ASSERT_TRUE(exporter_.seal(11, root.prefix("run-a"), &take_stub));
    // Publication stays paused until the baseline includes the held slot's charge.
    const size_t charged_while_held = exporter_.stats_for_test().charged_bytes;
    exporter_.pause_publication_for_test(false);

    std::string error;
    EXPECT_TRUE(exporter_.flush_retained_runs(-1, &error)) << error;
    EXPECT_TRUE(fs::exists(root.artifact("run-a")));
    EXPECT_FALSE(fs::exists(root.temporary("run-a"))) << "a successful publication left its temporary behind";

    const auto stats = exporter_.stats_for_test();
    EXPECT_EQ(stats.published, 1u);
    EXPECT_EQ(stats.inline_writes, 0u);
    EXPECT_EQ(stats.open_slots, 0u) << "the slot is returned once the artifact exists";
    EXPECT_LT(stats.charged_bytes, charged_while_held) << "a published slot kept its charge";
}

/** Retention off leaves every entry point inert. */
TEST_F(HostGraphExporterTest, RetentionOffPublishesNothingAndReportsNothing) {
    configure(false);
    OutputRoot root("off");

    EXPECT_TRUE(exporter_.seal(21, root.prefix("run-a"), &take_stub));
    EXPECT_FALSE(fs::exists(root.artifact("run-a")));

    std::string error;
    EXPECT_TRUE(exporter_.flush_retained_runs(0, &error)) << error;
    EXPECT_EQ(exporter_.stats_for_test().published, 0u);
}

/**
 * A third graph with both slots held is written by the sealing thread.
 *
 * Not refused: the graph is complete and in hand, and failing a run over a
 * diagnostic buys nothing. What it costs is the write, on this thread.
 */
TEST_F(HostGraphExporterTest, AThirdGraphIsWrittenInlineRatherThanRefused) {
    configure(true);
    OutputRoot root("inline");

    // Held publications keep both slots occupied, so the third seal finds none.
    exporter_.pause_publication_for_test(true);
    ASSERT_TRUE(exporter_.seal(31, root.prefix("run-a"), &take_stub));
    ASSERT_TRUE(exporter_.seal(32, root.prefix("run-b"), &take_stub));
    EXPECT_EQ(exporter_.stats_for_test().open_slots, 2u) << "two unpublished graphs is the retention bound";

    // On its own thread only because the hold reaches its publication too; the
    // route it took is the point, and it registered before any I/O — so the
    // wait is on the in-flight count, which the declaration moves, not on the
    // completed one, which the hold is stopping.
    std::thread sealing([&] {
        EXPECT_TRUE(exporter_.seal(33, root.prefix("run-c"), &take_stub));
    });
    const bool wrote_inline = wait_for_state([this] {
        return exporter_.stats_for_test().inline_in_flight == 1;
    });
    size_t charged_while_held = 0;
    if (wrote_inline) {
        const auto held = exporter_.stats_for_test();
        EXPECT_EQ(held.open_slots, 2u) << "an inline write takes no slot and is not charged";
        charged_while_held = held.charged_bytes;
    }

    // Released and joined before any assertion that could leave the function:
    // an abandoned joinable thread terminates the process instead of failing
    // the case.
    exporter_.pause_publication_for_test(false);
    sealing.join();
    ASSERT_TRUE(wrote_inline) << "the third seal never declared an inline write";
    std::string error;
    EXPECT_TRUE(exporter_.flush_retained_runs(-1, &error)) << error;
    for (const char *run : {"run-a", "run-b", "run-c"}) {
        EXPECT_TRUE(fs::exists(root.artifact(run))) << run;
    }
    const auto stats = exporter_.stats_for_test();
    EXPECT_EQ(stats.published, 3u);
    EXPECT_EQ(stats.inline_writes, 1u) << "only the third had no slot";
    EXPECT_EQ(stats.open_slots, 0u);
    EXPECT_LT(stats.charged_bytes, charged_while_held) << "a published slot kept its charge";
}

/** A budget that cannot hold the graph takes the same inline route. */
TEST_F(HostGraphExporterTest, ABudgetRefusalWritesInlineAndSettlesItsCharges) {
    // The fixed overhead plus the minimum working set, against a payload of
    // roughly twice what is then left. The margin is the point: the entry sizes
    // and the vectors' growth policy both move the payload, so a budget the
    // graph only just overruns would decide this case by ABI rather than by the
    // refusal it is about.
    configure(true, simpler::dfx::runs::kMinWorkingSetBytes + simpler::dfx::runs::kWriterScratchBytes * 2);
    OutputRoot root("budget");
    reset_take(/*tasks=*/120000);

    ASSERT_TRUE(exporter_.seal(41, root.prefix("run-a"), &take_stub));
    // The refusal is what the name claims settles, so it is asserted: a charge
    // that took and was not credited would leave this above the graph's own
    // payload, which the budget is sized below.
    const size_t charged_after_refusal = exporter_.stats_for_test().charged_bytes;

    std::string error;
    EXPECT_TRUE(exporter_.flush_retained_runs(-1, &error)) << error;
    EXPECT_TRUE(fs::exists(root.artifact("run-a")));
    const auto stats = exporter_.stats_for_test();
    EXPECT_EQ(stats.inline_writes, 1u);
    EXPECT_EQ(stats.open_slots, 0u);
    EXPECT_LT(charged_after_refusal, simpler::dfx::runs::kMinWorkingSetBytes) << "a refused graph left a charge behind";
    EXPECT_EQ(stats.charged_bytes, charged_after_refusal) << "publishing inline moved the charge";
}

/** Nothing captured on this thread is a reported failure, not a silent skip. */
TEST_F(HostGraphExporterTest, AnUncapturedGraphFailsTheFlush) {
    configure(true);
    OutputRoot root("uncaptured");
    g_take_outcome.store(static_cast<int>(hg::TakeOutcome::NotCaptured));

    EXPECT_FALSE(exporter_.seal(51, root.prefix("run-a"), &take_stub));
    EXPECT_FALSE(fs::exists(root.artifact("run-a")));

    std::string error;
    EXPECT_FALSE(exporter_.flush_retained_runs(0, &error));
    EXPECT_FALSE(error.empty());
    EXPECT_EQ(exporter_.stats_for_test().not_captured, 1u);
}

/** A task left open means missing edges, and the format cannot say so. */
TEST_F(HostGraphExporterTest, AnIncompleteCaptureFailsTheFlush) {
    configure(true);
    OutputRoot root("incomplete");
    g_take_outcome.store(static_cast<int>(hg::TakeOutcome::Incomplete));

    EXPECT_FALSE(exporter_.seal(61, root.prefix("run-a"), &take_stub));
    EXPECT_FALSE(fs::exists(root.artifact("run-a")));

    std::string error;
    EXPECT_FALSE(exporter_.flush_retained_runs(0, &error));
}

/** An orchestration that submitted nothing is a real answer, and publishes. */
TEST_F(HostGraphExporterTest, AnEmptyGraphPublishesAnEmptyGraph) {
    configure(true);
    OutputRoot root("empty");
    reset_take(/*tasks=*/0);

    ASSERT_TRUE(exporter_.seal(71, root.prefix("run-a"), &take_stub));
    std::string error;
    EXPECT_TRUE(exporter_.flush_retained_runs(-1, &error)) << error;

    std::ifstream in(root.artifact("run-a"));
    ASSERT_TRUE(in.good());
    std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    EXPECT_EQ(body, "{\"runtime\":\"host_build_graph\",\"tasks\":[],\"tensors\":[],\"edges\":[]}\n");
}

/** An occupied destination keeps the file that is there. */
TEST_F(HostGraphExporterTest, AnOccupiedDestinationFailsWithoutClobbering) {
    configure(true);
    OutputRoot root("occupied");
    fs::create_directories(root.artifact("run-a").parent_path());
    {
        std::ofstream pre(root.artifact("run-a"));
        pre << "not this run's graph";
    }

    ASSERT_TRUE(exporter_.seal(81, root.prefix("run-a"), &take_stub));
    std::string error;
    EXPECT_FALSE(exporter_.flush_retained_runs(-1, &error));

    std::ifstream in(root.artifact("run-a"));
    std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    EXPECT_EQ(body, "not this run's graph") << "link must never replace an existing name";
    EXPECT_FALSE(fs::exists(root.temporary("run-a"))) << "the failed attempt cleaned up its own temporary";
}

/** A temporary this exporter did not create is named, not deleted. */
TEST_F(HostGraphExporterTest, AForeignTemporaryIsRefusedAndKept) {
    configure(true);
    OutputRoot root("foreign-tmp");
    fs::create_directories(root.temporary("run-a").parent_path());
    {
        std::ofstream pre(root.temporary("run-a"));
        pre << "someone else's staging";
    }

    ASSERT_TRUE(exporter_.seal(91, root.prefix("run-a"), &take_stub));
    std::string error;
    EXPECT_FALSE(exporter_.flush_retained_runs(-1, &error));
    EXPECT_FALSE(fs::exists(root.artifact("run-a")));

    std::ifstream in(root.temporary("run-a"));
    std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    EXPECT_EQ(body, "someone else's staging");
}

/** A destination with no room for the file name is refused by name. */
TEST_F(HostGraphExporterTest, AnOverlongDestinationIsRefused) {
    configure(true);
    EXPECT_FALSE(exporter_.seal(101, std::string(hg::kMaxOutputDirBytes, 'd'), &take_stub));

    std::string error;
    EXPECT_FALSE(exporter_.flush_retained_runs(0, &error));
    EXPECT_EQ(exporter_.stats_for_test().failed, 1u);
}

/**
 * A flush waits for an unqueued write it cannot see in the queue.
 *
 * The hole this closes: an inline write happens outside the queue, so a flush
 * that only waited on the queue would return success before the write finished
 * or its error existed.
 */
TEST_F(HostGraphExporterTest, AFlushWaitsForAnInlineWriteAlreadyUnderWay) {
    // The inline write is the *only* outstanding work, which is what makes the
    // timed flush below evidence. A budget refusal is the route that leaves
    // nothing queued: no graph reaches the queue, so no writer is ever started,
    // and the flush can be held back by the inline registration alone. Reaching
    // the same route by filling both slots would leave a busy writer and a
    // non-empty queue, either of which fails the flush on its own — so that
    // shape would still pass with the inline count dropped from the idle test.
    configure(true, simpler::dfx::runs::kMinWorkingSetBytes + simpler::dfx::runs::kWriterScratchBytes * 2);
    OutputRoot root("flush-inline");
    reset_take(/*tasks=*/120000);

    // Held inside `publish`, after the write has registered.
    exporter_.pause_publication_for_test(true);
    std::atomic<bool> sealed{false};
    std::thread sealing([&] {
        (void)exporter_.seal(113, root.prefix("run-c"), &take_stub);
        sealed.store(true);
    });
    const bool registered = wait_for_state([&] {
        return exporter_.stats_for_test().inline_in_flight == 1 || sealed.load();
    });
    EXPECT_TRUE(registered) << "the seal never declared an inline write";
    EXPECT_FALSE(sealed.load()) << "the held publication should still have the seal inside it";
    EXPECT_EQ(exporter_.stats_for_test().open_slots, 0u) << "nothing is queued, so the flush has only the write";

    // The flush cannot complete while that write is outstanding, even though
    // the queue-and-writer pair alone shows nothing.
    std::string error;
    EXPECT_FALSE(exporter_.flush_retained_runs(50, &error)) << "a flush must not pass an unqueued write";

    exporter_.pause_publication_for_test(false);
    sealing.join();
    EXPECT_TRUE(exporter_.flush_retained_runs(-1, &error)) << error;
    EXPECT_TRUE(fs::exists(root.artifact("run-c")));
}

/**
 * A terminal close excludes future operations, not just present ones.
 *
 * `finish` closes admission and observes the outstanding leases in one critical
 * section, so a seal cannot register after the observation and then be joined
 * out from under.
 */
TEST_F(HostGraphExporterTest, FinishWaitsForAnOpenLeaseAndRefusesTheNextSeal) {
    configure(true);
    OutputRoot root("finish-race");

    g_take_pause.store(true);
    std::thread sealing([&] {
        (void)exporter_.seal(121, root.prefix("run-a"), &take_stub);
    });
    while (!g_take_entered.load())
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

    std::atomic<bool> finished{false};
    std::thread closing([&] {
        exporter_.finish_retained_runs();
        finished.store(true);
    });
    // The transition, not a sleep: once admission is closed the close is inside
    // its first phase, and the lease still open is what it must wait for.
    EXPECT_TRUE(wait_for_state([this] {
        return exporter_.stats_for_test().admission_closed;
    })) << "the close never closed admission";
    EXPECT_FALSE(finished.load()) << "finish must not return while a lease is open";

    g_take_pause.store(false);
    sealing.join();
    closing.join();
    EXPECT_TRUE(finished.load());
    EXPECT_TRUE(fs::exists(root.artifact("run-a"))) << "the lease that was open still published";

    // Admission is closed: a later seal takes no lease, writes nothing, and
    // reads no torn-down state.
    EXPECT_FALSE(exporter_.seal(122, root.prefix("run-b"), &take_stub));
    EXPECT_FALSE(fs::exists(root.artifact("run-b")));
    EXPECT_EQ(exporter_.stats_for_test().lease_refused, 1u);
}

/**
 * A close keeps an already-running consumer until the seal it already admitted
 * has enqueued, and refuses anything arriving after it.
 *
 * The interleaving `FinishWaitsForAnOpenLease...` above cannot reach: there the
 * first seal *is* the paused one, so the writer is started after the close and
 * finds the graph waiting for it. Here a graph is published first, which leaves
 * a writer idle on an empty queue — and that is the writer a close must not let
 * leave, because the seal already admitted has not reached its enqueue yet.
 *
 * Everything the closing thread touches is heap-owned and captured by value, so
 * a close that completes after this case returns writes to live memory. On a
 * build that stops the consumer early the close never returns at all, so the
 * case reports and detaches rather than hanging the suite.
 */
TEST_F(HostGraphExporterTest, ACloseKeepsItsConsumerForTheSealItAdmittedAndRefusesLaterOnes) {
    struct CloseState {
        std::atomic<bool> finished{false};
    };
    OutputRoot root("close-existing-writer");
    auto local = std::make_shared<hg::HostGraphExporter>();
    auto state = std::make_shared<CloseState>();
    local->configure_retained_runs(true, simpler::dfx::runs::kDefaultBudgetBytes);

    // One published graph, so a writer exists and is idle with an empty queue.
    ASSERT_TRUE(local->seal(151, root.prefix("run-a"), &take_stub));
    std::string error;
    ASSERT_TRUE(local->flush_retained_runs(-1, &error)) << error;
    ASSERT_TRUE(fs::exists(root.artifact("run-a")));

    // A second seal is admitted — its lease is taken — and held inside the
    // hand-off, before it can enqueue anything.
    g_take_entered.store(false);
    g_take_pause.store(true);
    std::thread sealing([local, prefix = root.prefix("run-b")] {
        EXPECT_TRUE(local->seal(152, prefix, &take_stub));
    });
    const bool admitted = wait_for_state([] {
        return g_take_entered.load();
    });
    EXPECT_TRUE(admitted) << "the second seal never reached its hand-off";

    std::thread closing([local, state] {
        local->finish_retained_runs();
        state->finished.store(true);
    });

    // The observed transition, not a sleep: the close has closed admission and
    // is therefore inside its first phase.
    const bool closing_observed = admitted && wait_for_state([&local] {
                                      return local->stats_for_test().admission_closed;
                                  });
    if (closing_observed) {
        // Admission is closed while the lease it must wait for is still open,
        // so the close cannot have returned.
        EXPECT_FALSE(state->finished.load()) << "a close must not return while an admitted seal is in flight";
        // A seal arriving now is refused by the lease, which is taken before
        // the hand-off — so it cannot block on the held one above.
        EXPECT_FALSE(local->seal(153, root.prefix("run-c"), &take_stub));
        EXPECT_FALSE(fs::exists(root.artifact("run-c")));
    } else {
        ADD_FAILURE() << "the close never closed admission";
    }

    // Released: the graph is enqueued after the close began, and the consumer
    // that was already running is what has to publish it.
    g_take_pause.store(false);
    sealing.join();

    if (!wait_for_state([&state] {
            return state->finished.load();
        })) {
        ADD_FAILURE() << "the close never returned: the consumer left before an admitted seal enqueued";
        // Detached, and both objects it uses are owned by its own captures, so
        // a late completion touches nothing this frame owns.
        closing.detach();
        return;
    }
    closing.join();
    EXPECT_TRUE(fs::exists(root.artifact("run-b"))) << "a graph accepted before the close is still published";
    const auto stats = local->stats_for_test();
    EXPECT_EQ(stats.published, 2u) << "neither accepted graph was dropped";
    EXPECT_EQ(stats.lease_refused, 1u) << "the one arriving after the close was refused";
    EXPECT_EQ(stats.open_slots, 0u);
}

/** A destructor that never saw `finish` still joins its writer. */
TEST_F(HostGraphExporterTest, TheDestructorPublishesAndJoinsWithoutFinish) {
    OutputRoot root("destructor");
    {
        hg::HostGraphExporter local;
        local.configure_retained_runs(true, simpler::dfx::runs::kDefaultBudgetBytes);
        ASSERT_TRUE(local.seal(131, root.prefix("run-a"), &take_stub));
        // No finish() — the init-failure path destroys a context without one.
    }
    EXPECT_TRUE(fs::exists(root.artifact("run-a")));
}

/** A flush whose budget expires claims nothing. */
TEST_F(HostGraphExporterTest, AFlushTimeoutClaimsNothing) {
    configure(true);
    OutputRoot root("timeout");

    // Queued and held inside its publication: declared work that cannot finish
    // inside the budget, which is the only thing a timeout may be about.
    exporter_.pause_publication_for_test(true);
    ASSERT_TRUE(exporter_.seal(141, root.prefix("run-a"), &take_stub));

    std::string error;
    EXPECT_FALSE(exporter_.flush_retained_runs(10, &error));
    EXPECT_NE(error.find("did not finish"), std::string::npos) << error;
    EXPECT_FALSE(fs::exists(root.artifact("run-a"))) << "a timeout claims nothing about the artifact";

    exporter_.pause_publication_for_test(false);
    EXPECT_TRUE(exporter_.flush_retained_runs(-1, &error)) << error;
    EXPECT_TRUE(fs::exists(root.artifact("run-a")));
}

/**
 * A publication that throws leaves no temporary of its own behind.
 *
 * The temporary is staged with `O_EXCL`, so one that outlives its publication
 * does not merely litter — it makes every later publication to that directory
 * fail to reserve it, and the destination is lost for good. Every non-throwing
 * exit unlinked its own temporary before; the throwing ones did not.
 *
 * The property is read off the destination rather than off the temporary: after
 * a failed attempt and a clean one, `deps.json` has to be there. A leak makes
 * the clean attempt fail on `O_EXCL` and the file never appears, which is the
 * failure this case reports.
 *
 * The injection point is swept over a calibrated publication instead of being
 * named, because which allocation `publish` reaches first is an implementation
 * detail of the standard library. The window is opened only once the graph is
 * queued and the writer is held at the top of `publish`, so the injection
 * covers the publication and not `seal`. That boundary is about this file's
 * `take_stub`, which builds a graph and so allocates, where the production
 * hand-off only moves five vectors: `seal` is `noexcept`, so a failure inside
 * the stub's build would terminate the process rather than exercise anything.
 */
TEST_F(HostGraphExporterTest, AThrowingPublicationLeavesNoTemporaryBehind) {
    configure(true);
    OutputRoot root("throwing");
    uint64_t epoch = 200;

    // Calibration over the region the injection covers: the publication past
    // the hold. Bounded because the sweep pays for each point.
    exporter_.pause_publication_for_test(true);
    ASSERT_TRUE(exporter_.seal(epoch++, root.prefix("calibrate"), &take_stub));
    g_alloc_count.store(0, std::memory_order_relaxed);
    exporter_.pause_publication_for_test(false);
    std::string error;
    ASSERT_TRUE(exporter_.flush_retained_runs(-1, &error)) << error;
    const long cycle = g_alloc_count.load(std::memory_order_relaxed);
    ASSERT_GT(cycle, 0) << "a publication that allocates nothing cannot throw";

    for (long n = 1; n <= std::min<long>(cycle, 200); n++) {
        const std::string run = "run-" + std::to_string(n);

        // Queued and held at the top of `publish`, before any of its own
        // allocations, so arming here puts the failure inside the publication.
        exporter_.pause_publication_for_test(true);
        ASSERT_TRUE(exporter_.seal(epoch++, root.prefix(run.c_str()), &take_stub)) << "n=" << n;
        arm_alloc_failure(n);
        exporter_.pause_publication_for_test(false);
        try {
            std::string injected_error;
            (void)exporter_.flush_retained_runs(-1, &injected_error);
        } catch (const std::bad_alloc &) {
            // The injection landed on this thread's own flush bookkeeping
            // rather than in the writer's publication. The destination check
            // below is the property either way.
        }
        disarm_alloc_failure();

        // Whatever that publication did, the destination must still be
        // reservable: a leaked temporary makes this second attempt fail to
        // stage, and the artifact never appears.
        ASSERT_TRUE(exporter_.seal(epoch++, root.prefix(run.c_str()), &take_stub)) << "n=" << n;
        std::string clean_error;
        (void)exporter_.flush_retained_runs(-1, &clean_error);
        EXPECT_TRUE(fs::exists(root.artifact(run.c_str())))
            << "allocation " << n << " of the publication left its temporary behind, so the destination is "
            << "unreservable";
    }
}

}  // namespace
