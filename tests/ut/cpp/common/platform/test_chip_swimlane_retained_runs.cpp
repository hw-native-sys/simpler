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
 * Retained runs: a run's host-side receipt, sealing and file
 * write continue while the next run executes.
 *
 * These drive the real thing — the collector's mgmt and poll threads, the
 * device-side producer, the per-queue transport cut and the writer
 * that publishes — rather than a model of it. Host and device share process
 * memory here, so nothing below is evidence about device cache visibility.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "aicpu/chip_swimlane_collector_aicpu.h"
#include "aicpu/device_run_result_base_aicpu.h"
#include "common/chip_swimlane_profiling.h"
#include "host/chip_swimlane_collector.h"
#include "host/run_boundary.h"

namespace fs = std::filesystem;

namespace {

void *retained_alloc(size_t size) { return std::calloc(1, size); }

// An allocator that runs out after a fixed number of successful calls, so a
// collector's `init()` fails part-way and its rollback guard runs.
int g_alloc_budget = -1;

void *retained_alloc_limited(size_t size) {
    if (g_alloc_budget == 0) return nullptr;
    if (g_alloc_budget > 0) g_alloc_budget--;
    return std::calloc(1, size);
}

int retained_free(void *ptr) {
    std::free(ptr);
    return 0;
}

// A free that reports failure while still reclaiming the memory, so the case
// leaks nothing and the collector sees only the status a real failure carries.
int retained_free_failing(void *ptr) {
    std::free(ptr);
    return -1;
}

// A host-mapping registration that refuses, which is what drives `init()` into
// the cleanup path for a device pointer it never registered with the manager.
int retained_register_failing(void *dev_ptr, size_t size, int device_id, void **host_ptr_out) {
    (void)dev_ptr;
    (void)size;
    (void)device_id;
    if (host_ptr_out != nullptr) *host_ptr_out = nullptr;
    return -1;
}

std::thread retained_thread_factory(std::function<void()> fn) { return std::thread(std::move(fn)); }

/** A private output root per case, removed at teardown. */
class ArtifactRoot {
public:
    explicit ArtifactRoot(const char *name) {
        path_ = fs::temp_directory_path() /
                ("simpler-dfx-retained-" + std::string(name) + "-" + std::to_string(::getpid()));
        std::error_code ec;
        fs::remove_all(path_, ec);
        fs::create_directories(path_, ec);
    }
    ~ArtifactRoot() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    std::string str() const { return path_.string(); }
    fs::path path() const { return path_; }

private:
    fs::path path_;
};

/** Files a collector published, across every reserved artifact directory. */
std::vector<fs::path> published_files(const fs::path &root) {
    std::vector<fs::path> found;
    std::error_code ec;
    for (const auto &dir : fs::directory_iterator(root, ec)) {
        if (!dir.is_directory()) continue;
        for (const auto &entry : fs::directory_iterator(dir.path(), ec)) {
            const std::string name = entry.path().filename().string();
            if (name.rfind("records_e", 0) == 0 && entry.path().extension() == ".json") found.push_back(entry.path());
        }
    }
    return found;
}

std::string read_file(const fs::path &p) {
    std::ifstream in(p);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

/**
 * Ordered teardown for a collector a case started for itself.
 *
 * A failed `ASSERT_*` returns from the case body, so this belongs on a
 * destructor rather than at the end of the function: stop admitting, join the
 * writer, then join the readers and free. Every step is idempotent, so a case
 * that tears its collector down explicitly — with a free callback of its own,
 * say — leaves this with nothing to do.
 */
class RetainedRunsTeardown {
public:
    RetainedRunsTeardown(ChipSwimlaneCollector &collector, ChipSwimlaneFreeCallback free_cb) :
        collector_(collector),
        free_cb_(std::move(free_cb)) {}
    RetainedRunsTeardown(const RetainedRunsTeardown &) = delete;
    RetainedRunsTeardown &operator=(const RetainedRunsTeardown &) = delete;
    ~RetainedRunsTeardown() {
        collector_.finish_retained_runs();
        collector_.stop();
        collector_.finalize(nullptr, free_cb_);
    }

private:
    ChipSwimlaneCollector &collector_;
    ChipSwimlaneFreeCallback free_cb_;
};

/**
 * One collector with its threads running and a retained run.
 *
 * `aicpu_thread_num` is what gives the run more than one device ready queue,
 * which the cross-queue case needs: a queue is owned by exactly one drain
 * thread and its target may only ever be discharged by its own traffic.
 */
struct RetainedRunsFixture {
    ChipSwimlaneCollector collector;
    ArtifactRoot dir;
    int num_aicore;

    RetainedRunsFixture(const char *name, int cores, int threads, size_t budget_bytes = 0) :
        dir(name),
        num_aicore(cores) {
        EXPECT_EQ(
            collector.initialize(
                cores, threads, /*device_id=*/0, ChipSwimlaneLevel::TASK_TIMING, retained_alloc, nullptr, retained_free
            ),
            0
        );
        collector.configure_retained_runs(
            /*retain_across_runs=*/true, budget_bytes != 0 ? budget_bytes : simpler::dfx::runs::kDefaultBudgetBytes
        );
        collector.start(retained_thread_factory);
    }

    ~RetainedRunsFixture() {
        collector.finish_retained_runs();
        collector.stop();
        collector.finalize(nullptr, retained_free);
    }

    void *shm() { return collector.get_chip_swimlane_setup_device_ptr(); }

    /** Open a run: arm its bank, admit the epoch, bring the device side up. */
    void begin(uint64_t epoch) {
        ASSERT_NE(shm(), nullptr);
        EXPECT_TRUE(collector.run_begin(epoch, dir.str(), ChipSwimlaneLevel::TASK_TIMING));
        set_platform_run_result(/*region_base=*/0, epoch);
        set_chip_swimlane_enabled(true);
        set_platform_chip_swimlane_base(reinterpret_cast<uint64_t>(shm()));
        set_platform_chip_swimlane_aicore_rotation_table(0);
        set_platform_chip_swimlane_run_terminal_bank(
            reinterpret_cast<uint64_t>(collector.arm_run_terminal_bank(/*bank_index=*/0, epoch))
        );
        chip_swimlane_aicpu_init(num_aicore);
    }

    /** Dispatch AICore records on `core_id`, writing each slot the host keeps. */
    void dispatch(int core_id, int records) {
        auto *ac_state = get_aicore_buffer_state(shm(), core_id);
        for (int i = 0; i < records; i++) {
            chip_swimlane_aicpu_on_aicore_dispatch(core_id, /*thread_idx=*/0, static_cast<uint32_t>(i + 1));
            auto *buf = reinterpret_cast<ChipSwimlaneAicoreTaskBuffer *>(ac_state->head.current_buf_ptr);
            if (buf == nullptr) continue;
            const uint32_t slot = ac_state->head.live_record_count > 0 ? ac_state->head.live_record_count - 1 : 0;
            if (slot >= static_cast<uint32_t>(PLATFORM_AICORE_BUFFER_SIZE)) continue;
            buf->records[slot].start_time = 1000 + static_cast<uint64_t>(i);
            buf->records[slot].end_time = 2000 + static_cast<uint64_t>(i);
            buf->records[slot].reg_task_id = static_cast<uint32_t>(i + 1);
        }
    }

    /** Close the device side of a run, then hand the run to the writer. */
    void close(uint64_t epoch, const int *cores, int core_num, int thread_idx = 0) {
        chip_swimlane_aicpu_flush(thread_idx, cores, core_num);
        collector.run_close(epoch, /*bank_index=*/0, /*device_execution_complete=*/true);
    }

    /**
     * Close a run the way both runner bases do: whatever this run produced on
     * the host reaches the collector through the shared boundary helper, which
     * is what orders it against the epoch's metadata snapshot.
     */
    template <typename PublishHostState>
    void close_through_boundary(uint64_t epoch, const int *cores, int core_num, PublishHostState &&publish) {
        chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, core_num);
        simpler::dfx::runs::close_run_boundary(
            collector, epoch, /*bank_index=*/0, /*device_execution_complete=*/true,
            std::forward<PublishHostState>(publish)
        );
    }

    bool wait_for_files(size_t count, int timeout_ms = 8000) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            if (published_files(dir.path()).size() >= count) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return published_files(dir.path()).size() >= count;
    }
};

/** How many rows in this artifact carry `epoch` as their trailing column. */
size_t rows_for_epoch(const std::string &body, uint64_t epoch) {
    const std::string column = ", " + std::to_string(epoch) + "]";
    size_t rows = 0;
    for (size_t at = body.find(column); at != std::string::npos; at = body.find(column, at + 1)) {
        rows++;
    }
    return rows;
}

/**
 * `count` host submit records, the shape a run hands the collector at its
 * boundary. The count is what tells two runs' publications apart in the
 * artifact: the writer reports it as `recorded_records`, in a metadata block
 * every level emits, where the rows themselves are rendered only from
 * SCHED_PHASES up.
 */
std::vector<HostPhaseRecord> host_submit_rows(size_t count) {
    std::vector<HostPhaseRecord> rows(count);
    for (size_t i = 0; i < count; i++) {
        rows[i].start_ns = 100 + i;
        rows[i].end_ns = 200 + i;
        rows[i].payload = 4000 + i;
        rows[i].kind = 0;
        rows[i].index = static_cast<uint32_t>(i + 1);
        rows[i].thread_id = 7;
        rows[i]._pad = 0;
    }
    return rows;
}

/** The writer's rendering of how many host submit records an epoch carries. */
std::string recorded_records_field(size_t count) { return "\"recorded_records\": " + std::to_string(count); }

/**
 * A collector with a retained run but no reader threads started.
 *
 * This is the shape every unprovable close has: with no collector shard
 * polling, no reference-release acknowledgement can ever land, so the collector
 * reaches its fatal and quarantine paths through production code rather than
 * through an injected failure.
 */
struct ReaderlessCollector {
    ChipSwimlaneCollector collector;
    ArtifactRoot dir;

    explicit ReaderlessCollector(const char *name) :
        dir(name) {
        EXPECT_EQ(
            collector.initialize(
                /*cores=*/1, /*threads=*/1, /*device_id=*/0, ChipSwimlaneLevel::TASK_TIMING, retained_alloc, nullptr,
                retained_free
            ),
            0
        );
        collector.configure_retained_runs(/*retain_across_runs=*/true, simpler::dfx::runs::kDefaultBudgetBytes);
    }

    /**
     * Wait until a seal has marked storage as held pending the reader join.
     *
     * The seal happens on the writer's own thread, and a `finish` or a `flush`
     * on a collector that is already fatal returns on that fatal rather than
     * waiting for it — which is the right contract, because nothing the writer
     * could still do would change a fatal answer. So a case that is about the
     * verdict has to await the verdict itself.
     *
     * The deferral flag and not the verdict count: `finish_retained_run`
     * records the row first and raises the flag after retiring the cut, so a
     * wait on the count could observe the flag still unset. This is the last
     * thing the quarantine branch publishes before the fatal, so everything
     * that branch does is visible once it is true.
     *
     * Bounded, and an expired deadline fails the caller: the writer is alive
     * here and `run_close` has already bumped its progress counter, so a
     * verdict that never arrives is a defect and not a slow machine.
     */
    bool wait_for_deferred_release(int timeout_ms = 8000) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            if (collector.retained_run_stats_for_test().release_deferred) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return collector.retained_run_stats_for_test().release_deferred;
    }
};

}  // namespace

// Three runs in a row, with nobody calling flush. The writer is what
// frees a slot, so if publication were only a caller's job the third
// `run_begin` would block for ever against a two-slot cap.
TEST(ChipSwimlaneRetainedRunsTest, ThreeRunsProgressWithoutAnyFlush) {
    RetainedRunsFixture fx("progress", /*cores=*/1, /*threads=*/1);
    const int cores[] = {0};
    for (uint64_t epoch = 9001; epoch <= 9003; epoch++) {
        fx.begin(epoch);
        fx.dispatch(/*core_id=*/0, 4);
        fx.close(epoch, cores, 1);
    }
    EXPECT_TRUE(fx.wait_for_files(3)) << "the writer did not publish on its own";

    std::string error;
    EXPECT_TRUE(fx.collector.flush_retained_runs(4000, &error)) << error;
    const auto files = published_files(fx.dir.path());
    ASSERT_EQ(files.size(), 3u);
    for (const auto &f : files) {
        const std::string body = read_file(f);
        EXPECT_NE(body.find("\"collection\""), std::string::npos) << f.string();
        EXPECT_NE(body.find("\"session_id\""), std::string::npos);
    }
}

// A predecessor's records survive a successor's admission. This is the whole
// point of the per-epoch store: the legacy `begin_run` wipes one shared set,
// which would take N's buffers away while they are still arriving.
TEST(ChipSwimlaneRetainedRunsTest, PredecessorRecordsSurviveSuccessorAdmission) {
    RetainedRunsFixture fx("overlap", /*cores=*/1, /*threads=*/1);
    const int cores[] = {0};
    constexpr uint64_t kFirst = 9101;
    constexpr uint64_t kSecond = 9102;

    fx.begin(kFirst);
    fx.dispatch(/*core_id=*/0, 6);
    chip_swimlane_aicpu_flush(0, cores, 1);
    // The successor is admitted before the predecessor is sealed.
    fx.collector.run_close(kFirst, 0, true);
    fx.begin(kSecond);
    fx.dispatch(/*core_id=*/0, 2);
    fx.close(kSecond, cores, 1);

    ASSERT_TRUE(fx.wait_for_files(2));
    bool saw_first = false;
    for (const auto &f : published_files(fx.dir.path())) {
        if (f.filename().string() == "records_e" + std::to_string(kFirst) + ".json") {
            saw_first = true;
            const std::string body = read_file(f);
            // Six AICore records, each a row in the aicore_tasks stream, and
            // the predecessor's own epoch on every one of them.
            const std::string epoch_column = ", " + std::to_string(kFirst) + "]";
            size_t rows = 0;
            for (size_t at = body.find(epoch_column); at != std::string::npos; at = body.find(epoch_column, at + 1)) {
                rows++;
            }
            EXPECT_EQ(rows, 6u) << "the predecessor's records did not reach its artifact";
        }
    }
    EXPECT_TRUE(saw_first);
}

// The refusal side of the same invariant: a run the collector will not admit
// must leave the predecessor's store untouched. `run_begin` returning false is
// an admission result, so the runner fails the run on it rather than falling
// back to `begin_run` — whose reset is what would take the predecessor's
// records away. Nothing here calls `begin_run`, which is the point: the
// predecessor's record count is what proves no reset happened.
TEST(ChipSwimlaneRetainedRunsTest, ARefusedAdmissionKeepsThePredecessorsRecords) {
    RetainedRunsFixture fx("refusal", /*cores=*/1, /*threads=*/1);
    constexpr uint64_t kFirst = 9121;
    constexpr uint64_t kSecond = 9122;

    fx.begin(kFirst);
    // Handed straight to the collector: no producer is running on this shard,
    // so this thread is its only writer, and the count is then exact rather
    // than a wait.
    ChipSwimlaneAicoreTaskBuffer held{};
    held.run_epoch = kFirst;
    held.count = 3;
    for (uint32_t i = 0; i < held.count; i++) {
        held.records[i].start_time = 4000 + i;
        held.records[i].end_time = 5000 + i;
    }
    ReadyBufferInfo info{};
    info.type = ProfBufferType::AICORE_TASK;
    info.index = 0;
    info.dev_buffer_ptr = &held;
    info.host_buffer_ptr = &held;
    fx.collector.on_buffer_collected(info, /*collector_shard=*/0);
    ASSERT_EQ(fx.collector.collected_aicore_records_for_test()[0].size(), 3u);

    // Joining the reader shards is what makes the next admission unprovable:
    // no shard is left to acknowledge the run table, so the collector refuses
    // the successor and records its own fatal — a refusal reached through
    // production code rather than an injected failure.
    fx.collector.stop();
    EXPECT_FALSE(fx.collector.run_begin(kSecond, fx.dir.str(), ChipSwimlaneLevel::TASK_TIMING));
    EXPECT_TRUE(fx.collector.retained_run_stats_for_test().fatal);

    EXPECT_EQ(fx.collector.collected_aicore_records_for_test()[0].size(), 3u)
        << "the refused admission dropped the predecessor's records";
    EXPECT_TRUE(published_files(fx.dir.path()).empty()) << "a run that was never admitted published an artifact";
}

// A run the collectors admitted and nothing submitted for gives its slot back.
// Left in place it is invisible to both the writer and a flush — it has no
// target — so it would hold retained capacity for the collector's whole life.
// Two rollbacks through the same single free slot are what prove the capacity
// actually comes back, and a predecessor open across both must come through
// untouched. The slot count is asserted before the second admission, because a
// capacity wait is what an unreleased slot would produce.
TEST(ChipSwimlaneRetainedRunsTest, UnlaunchedRunsGiveTheirSlotsBackAndKeepAPredecessor) {
    RetainedRunsFixture fx("abandon", /*cores=*/1, /*threads=*/1);
    const int cores[] = {0};
    constexpr uint64_t kKept = 9601;
    constexpr uint64_t kFirstRolled = 9602;
    constexpr uint64_t kSecondRolled = 9603;

    // A predecessor that stays open across every rollback below, with records
    // handed straight to the collector: no producer runs on this shard, so
    // this thread is its only writer and the count is exact.
    fx.begin(kKept);
    ChipSwimlaneAicoreTaskBuffer held{};
    held.run_epoch = kKept;
    held.count = 4;
    for (uint32_t i = 0; i < held.count; i++) {
        held.records[i].start_time = 7000 + i;
        held.records[i].end_time = 8000 + i;
    }
    ReadyBufferInfo info{};
    info.type = ProfBufferType::AICORE_TASK;
    info.index = 0;
    info.dev_buffer_ptr = &held;
    info.host_buffer_ptr = &held;
    fx.collector.on_buffer_collected(info, /*collector_shard=*/0);
    ASSERT_EQ(fx.collector.collected_aicore_records_for_test()[0].size(), 4u);

    // Admitted, then withdrawn without a launch — the shape a transaction that
    // ends at `NotStarted` leaves behind.
    ASSERT_TRUE(fx.collector.run_begin(kFirstRolled, fx.dir.str(), ChipSwimlaneLevel::TASK_TIMING));
    ASSERT_EQ(fx.collector.retained_run_stats_for_test().open_slots, 2u);
    ASSERT_TRUE(fx.collector.abandon_run(kFirstRolled)) << "the withdrawal could not prove the references released";
    ASSERT_EQ(fx.collector.retained_run_stats_for_test().open_slots, 1u)
        << "the unlaunched run kept its slot, so retained capacity is gone for good";

    // The same free slot again: only a rollback that really released it can
    // admit this one without waiting for capacity.
    ASSERT_TRUE(fx.collector.run_begin(kSecondRolled, fx.dir.str(), ChipSwimlaneLevel::TASK_TIMING));
    ASSERT_TRUE(fx.collector.abandon_run(kSecondRolled));
    EXPECT_EQ(fx.collector.retained_run_stats_for_test().open_slots, 1u);

    EXPECT_EQ(fx.collector.collected_aicore_records_for_test()[0].size(), 4u)
        << "a rollback took the predecessor's records";
    EXPECT_TRUE(published_files(fx.dir.path()).empty()) << "a run that submitted nothing published an artifact";
    EXPECT_FALSE(fx.collector.retained_run_stats_for_test().fatal);

    // And no rollback is remembered as a lost artifact: the predecessor
    // publishes and the flush is clean.
    fx.close(kKept, cores, 1);
    ASSERT_TRUE(fx.wait_for_files(1));
    std::string error;
    EXPECT_TRUE(fx.collector.flush_retained_runs(4000, &error)) << error;
    EXPECT_EQ(published_files(fx.dir.path()).size(), 1u) << "a withdrawn run left an artifact of its own";
}

// The other half of the same rollback: a withdrawal whose acknowledgement
// cannot land must publish that it failed before it reports anything. The
// deferral flag, the sticky fatal and the waiter wakeup are what a capacity
// claim and a flush barrier read, and this path runs where an allocation may
// have just failed — so the slot stays occupied and every one of those is set,
// rather than the bucket being left mid-withdrawal with nothing said about it.
TEST(ChipSwimlaneRetainedRunsTest, AnUnprovableWithdrawalQuarantinesAndPublishesItsFailure) {
    RetainedRunsFixture fx("abandonfail", /*cores=*/1, /*threads=*/1);
    constexpr uint64_t kEpoch = 9611;

    fx.begin(kEpoch);
    ASSERT_FALSE(fx.collector.retained_run_stats_for_test().fatal);

    // Joining the reader shards is what makes the withdrawal unprovable: no
    // shard is left to acknowledge that it dropped its reference.
    fx.collector.stop();
    EXPECT_FALSE(fx.collector.abandon_run(kEpoch)) << "a withdrawal no shard acknowledged reported success";

    const auto stats = fx.collector.retained_run_stats_for_test();
    EXPECT_EQ(stats.open_slots, 1u) << "storage was released without proof that its readers had let go";
    EXPECT_EQ(stats.quarantined, 1u) << "the unprovable withdrawal reached no quarantine verdict";
    EXPECT_TRUE(stats.release_deferred) << "storage was not marked as held pending the reader join";
    EXPECT_TRUE(stats.fatal) << "a waiter would have no way to learn the withdrawal failed";
    EXPECT_TRUE(published_files(fx.dir.path()).empty()) << "a withdrawn run published an artifact";

    // A flush reports the failure rather than reading the empty per-epoch rows
    // as a clean collector.
    std::string error;
    EXPECT_FALSE(fx.collector.flush_retained_runs(200, &error));
    EXPECT_NE(error.find("fatal"), std::string::npos) << error;
}

// A predecessor's AICore buffer that arrives *after* the successor was armed is
// still the predecessor's. The arm moves the collector's "most recently armed
// run", so comparing a record's stamp against that field classifies every late
// predecessor buffer as foreign — the normal case here, not an exception — and
// publishes an artifact that lists the rows while reporting none collected.
// AICore is its own path: it is the only producer class whose records carry a
// per-buffer identity decision, so the AICPU-record cases above do not cover
// it.
TEST(ChipSwimlaneRetainedRunsTest, LatePredecessorAicoreBufferCountsAsItsOwnEpochsRecords) {
    RetainedRunsFixture fx("lateaicore", /*cores=*/1, /*threads=*/1);
    const int cores[] = {0};
    constexpr uint64_t kFirst = 9111;
    constexpr uint64_t kSecond = 9112;

    fx.begin(kFirst);
    // Admitting and arming the successor is what moves the armed epoch off the
    // predecessor while the predecessor is still open.
    fx.begin(kSecond);

    // The predecessor's buffer, delivered now. Handed straight to the collector
    // so the arrival is ordered after the successor's arm by construction
    // rather than by timing; no producer is running on this shard, so this
    // thread is its only writer.
    ChipSwimlaneAicoreTaskBuffer late{};
    late.run_epoch = kFirst;
    late.count = 2;
    late.records[0].start_time = 4000;
    late.records[0].end_time = 5000;
    late.records[1].start_time = 4001;
    late.records[1].end_time = 5001;
    ReadyBufferInfo info{};
    info.type = ProfBufferType::AICORE_TASK;
    info.index = 0;
    info.dev_buffer_ptr = &late;
    info.host_buffer_ptr = &late;
    fx.collector.on_buffer_collected(info, /*collector_shard=*/0);

    fx.close(kFirst, cores, 1);
    fx.close(kSecond, cores, 1);
    ASSERT_TRUE(fx.wait_for_files(2));

    const auto stats = fx.collector.retained_run_stats_for_test();
    EXPECT_EQ(stats.aicore_collected, 2u) << "the late buffer's records were not counted as their epoch's";
    EXPECT_EQ(stats.aicore_foreign, 0u) << "a record was charged to an identity its own epoch did not have";
    EXPECT_EQ(stats.unknown_epoch, 0u);
    EXPECT_EQ(stats.late_after_seal, 0u);

    // And the rows are in the predecessor's file, so the count above describes
    // the same records the artifact publishes.
    bool saw_first = false;
    for (const auto &f : published_files(fx.dir.path())) {
        if (f.filename().string() != "records_e" + std::to_string(kFirst) + ".json") continue;
        saw_first = true;
        EXPECT_EQ(rows_for_epoch(read_file(f), kFirst), 2u);
    }
    EXPECT_TRUE(saw_first);
}

// Two device ready queues, one owner each. A queue's target may only be
// discharged by its own traffic: sustained work on queue 0 must not settle a
// run whose buffer is still sitting on queue 1.
TEST(ChipSwimlaneRetainedRunsTest, OneQueuesTrafficDoesNotDischargeAnother) {
    RetainedRunsFixture fx("twoqueue", /*cores=*/2, /*threads=*/2);
    const int all_cores[] = {0, 1};
    constexpr uint64_t kEpoch = 9201;

    fx.begin(kEpoch);
    // Core 0 is served by thread 0's queue, core 1 by thread 1's.
    fx.dispatch(/*core_id=*/0, 3);
    fx.dispatch(/*core_id=*/1, 5);
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, &all_cores[0], 1);
    chip_swimlane_aicpu_flush(/*thread_idx=*/1, &all_cores[1], 1);
    fx.collector.run_close(kEpoch, 0, true);

    ASSERT_TRUE(fx.wait_for_files(1));
    const auto files = published_files(fx.dir.path());
    ASSERT_EQ(files.size(), 1u);
    const std::string body = read_file(files[0]);
    // Both queues' records are in the one artifact, which is only true if the
    // cut waited for both targets.
    EXPECT_NE(body.find("\"processing_complete\": true"), std::string::npos) << body.substr(0, 400);
}

// A run whose device execution never completed has no terminal to read. It
// still has to reach a verdict and release its slot, or two such runs would
// strand the capacity for the rest of the process.
TEST(ChipSwimlaneRetainedRunsTest, RunWithoutTerminalStillPublishesAndReleasesItsSlot) {
    RetainedRunsFixture fx("noterminal", /*cores=*/1, /*threads=*/1);
    const int cores[] = {0};
    constexpr uint64_t kEpoch = 9301;

    fx.begin(kEpoch);
    fx.dispatch(/*core_id=*/0, 2);
    chip_swimlane_aicpu_flush(0, cores, 1);
    // device_execution_complete = false: the recovery path, where the bank says
    // nothing about this run.
    fx.collector.run_close(kEpoch, 0, /*device_execution_complete=*/false);

    ASSERT_TRUE(fx.wait_for_files(1));
    const std::string body = read_file(published_files(fx.dir.path())[0]);
    EXPECT_NE(body.find("\"processing_complete\": false"), std::string::npos) << body.substr(0, 400);
    EXPECT_NE(body.find("partial_cut_unknown"), std::string::npos);

    // The slot came back: a further run is admitted without waiting.
    const auto stats = fx.collector.retained_run_stats_for_test();
    EXPECT_EQ(stats.open_slots, 0u);
    EXPECT_FALSE(stats.fatal);
}

// A publication that cannot happen must not be silently forgotten: no file, an
// error from flush, and the slot still released so the collector keeps running.
TEST(ChipSwimlaneRetainedRunsTest, WriteFailureIsReportedAndStillReleasesTheSlot) {
    RetainedRunsFixture fx("writefail", /*cores=*/1, /*threads=*/1);
    const int cores[] = {0};
    constexpr uint64_t kEpoch = 9401;

    fx.begin(kEpoch);
    fx.dispatch(/*core_id=*/0, 2);

    // Occupy the exact path this run will publish to. `link` refuses to
    // replace, so the run reports a collision instead of overwriting.
    fs::path taken;
    for (const auto &entry : fs::directory_iterator(fx.dir.path())) {
        if (entry.is_directory()) taken = entry.path() / ("records_e" + std::to_string(kEpoch) + ".json");
    }
    ASSERT_FALSE(taken.empty());
    {
        std::ofstream squatter(taken);
        squatter << "not this run's artifact";
    }

    fx.close(kEpoch, cores, 1);

    std::string error;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
    bool reported = false;
    while (std::chrono::steady_clock::now() < deadline) {
        if (!fx.collector.flush_retained_runs(200, &error)) {
            reported = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_TRUE(reported) << "a run that produced no file was reported as a clean flush";
    EXPECT_NE(error.find("write_failed"), std::string::npos) << error;
    EXPECT_EQ(read_file(taken), "not this run's artifact") << "an existing artifact was overwritten";
    EXPECT_EQ(fx.collector.retained_run_stats_for_test().open_slots, 0u);
}

// The tombstone ring holds 16 epochs and exists to classify late buffers. A
// failure from further back than that must still be reported, which it is only
// because the collector's error summary is separate and never evicted.
TEST(ChipSwimlaneRetainedRunsTest, FailureIsRememberedPastTheTombstoneRing) {
    RetainedRunsFixture fx("memory", /*cores=*/1, /*threads=*/1);
    const int cores[] = {0};
    constexpr uint64_t kFailing = 9501;

    fx.begin(kFailing);

    // The artifact directory is reserved by the first admission, so it exists
    // only from here. The squatter occupies the name this epoch will publish
    // under, and publication is a `link` that cannot replace — so planting it
    // any time before the seal is what makes the write fail.
    fs::path taken;
    for (const auto &entry : fs::directory_iterator(fx.dir.path())) {
        if (entry.is_directory()) taken = entry.path() / ("records_e" + std::to_string(kFailing) + ".json");
    }
    ASSERT_FALSE(taken.empty());
    {
        std::ofstream squatter(taken);
        squatter << "occupied";
    }

    fx.dispatch(0, 1);
    fx.close(kFailing, cores, 1);

    // Cumulative, so every epoch below is actually awaited: a fixed count
    // stays satisfied after the first clean run and the loop would wait for
    // nothing. The squatter already matches the published-file predicate and
    // the failing epoch adds no file of its own, so the baseline is read from
    // the directory rather than assumed to be zero.
    size_t expected = published_files(fx.dir.path()).size();

    // More than a tombstone ring's worth of clean runs after the failure.
    for (uint64_t epoch = 9502; epoch <= 9522; epoch++) {
        fx.begin(epoch);
        fx.dispatch(0, 1);
        fx.close(epoch, cores, 1);
        ASSERT_TRUE(fx.wait_for_files(++expected)) << "epoch " << epoch << " never published";
    }

    std::string error;
    EXPECT_FALSE(fx.collector.flush_retained_runs(4000, &error)) << "the old failure was forgotten";
    EXPECT_NE(error.find("write_failed"), std::string::npos) << error;
    EXPECT_NE(error.find(std::to_string(kFailing)), std::string::npos) << error;
}

// A buffer whose run the collector has already sealed is counted and dropped.
// It must never append to a slot the writer has moved, and never reopen one.
TEST(ChipSwimlaneRetainedRunsTest, LateBufferForASealedEpochIsCountedNotAppended) {
    RetainedRunsFixture fx("late", /*cores=*/1, /*threads=*/1);
    const int cores[] = {0};
    constexpr uint64_t kEpoch = 9601;

    fx.begin(kEpoch);
    fx.dispatch(0, 2);
    fx.close(kEpoch, cores, 1);
    ASSERT_TRUE(fx.wait_for_files(1));

    ChipSwimlaneAicoreTaskBuffer late{};
    late.run_epoch = kEpoch;
    late.count = 3;
    ReadyBufferInfo info{};
    info.type = ProfBufferType::AICORE_TASK;
    info.index = 0;
    info.dev_buffer_ptr = &late;
    info.host_buffer_ptr = &late;
    fx.collector.on_buffer_collected(info, /*collector_shard=*/0);

    const auto stats = fx.collector.retained_run_stats_for_test();
    EXPECT_EQ(stats.late_after_seal, 1u);
    EXPECT_EQ(stats.unknown_epoch, 0u);
    EXPECT_EQ(published_files(fx.dir.path()).size(), 1u) << "a late buffer produced a second artifact";
}

// An epoch nobody opened is counted separately from one that was sealed: the
// two are different facts and the tombstone ring is what tells them apart.
TEST(ChipSwimlaneRetainedRunsTest, BufferForAnUnknownEpochIsCountedSeparately) {
    RetainedRunsFixture fx("unknown", /*cores=*/1, /*threads=*/1);
    fx.begin(9701);

    ChipSwimlaneAicoreTaskBuffer stray{};
    stray.run_epoch = 424242;
    stray.count = 1;
    ReadyBufferInfo info{};
    info.type = ProfBufferType::AICORE_TASK;
    info.index = 0;
    info.dev_buffer_ptr = &stray;
    info.host_buffer_ptr = &stray;
    fx.collector.on_buffer_collected(info, /*collector_shard=*/0);

    const auto stats = fx.collector.retained_run_stats_for_test();
    EXPECT_EQ(stats.unknown_epoch, 1u);
    EXPECT_EQ(stats.late_after_seal, 0u);
}

// A collector reserves its artifact directory by exclusive creation, so a
// second collector over the same output root cannot land on the first's files.
TEST(ChipSwimlaneRetainedRunsTest, TwoCollectorsOverOneRootReserveDistinctDirectories) {
    ArtifactRoot root("reserve");
    ChipSwimlaneCollector first;
    ChipSwimlaneCollector second;
    // Declared after both collectors, so each is torn down before it is
    // destroyed and a failed assertion below still joins their threads.
    RetainedRunsTeardown teardown_first(first, retained_free);
    RetainedRunsTeardown teardown_second(second, retained_free);
    ASSERT_EQ(first.initialize(1, 1, 0, ChipSwimlaneLevel::TASK_TIMING, retained_alloc, nullptr, retained_free), 0);
    ASSERT_EQ(second.initialize(1, 1, 0, ChipSwimlaneLevel::TASK_TIMING, retained_alloc, nullptr, retained_free), 0);
    first.configure_retained_runs(true, simpler::dfx::runs::kDefaultBudgetBytes);
    second.configure_retained_runs(true, simpler::dfx::runs::kDefaultBudgetBytes);
    // Reader shards before the first admission, the order both runner bases
    // take: admitting a run waits for every collector shard to acknowledge the
    // run table, and `initialize()` creates no shard to acknowledge it.
    first.start(retained_thread_factory);
    second.start(retained_thread_factory);
    EXPECT_TRUE(first.run_begin(1, root.str(), ChipSwimlaneLevel::TASK_TIMING));
    EXPECT_TRUE(second.run_begin(1, root.str(), ChipSwimlaneLevel::TASK_TIMING));

    int dirs = 0;
    for (const auto &entry : fs::directory_iterator(root.path())) {
        if (entry.is_directory()) dirs++;
    }
    EXPECT_EQ(dirs, 2) << "the second collector reused the first one's directory";
}

// A budget that cannot hold the fixed overhead plus a working set is refused
// when the first run asks to be retained, rather than discovered as emptiness
// later.
TEST(ChipSwimlaneRetainedRunsTest, AnUnworkableBudgetIsRefusedAtOpen) {
    ArtifactRoot root("budget");
    ChipSwimlaneCollector collector;
    ASSERT_EQ(collector.initialize(1, 1, 0, ChipSwimlaneLevel::TASK_TIMING, retained_alloc, nullptr, retained_free), 0);
    // Below the minimum working set by construction.
    collector.configure_retained_runs(/*retain_across_runs=*/true, /*budget_bytes=*/1024);
    EXPECT_FALSE(collector.run_begin(1, root.str(), ChipSwimlaneLevel::TASK_TIMING));
    // A refusal costs the collector nothing: the single-run path it would have
    // taken with retention off still works on it.
    collector.begin_run(root.str(), ChipSwimlaneLevel::TASK_TIMING);
    collector.finalize(nullptr, retained_free);
}

// Readiness is one latch, taken only after every step of the preparation has
// succeeded — so a preparation that failed leaves nothing a later run could
// mistake for a prepared collector, and no directory behind either. The retry
// then prepares for real, and the two idempotent exits can each be called
// twice without releasing anything a second time.
TEST(ChipSwimlaneRetainedRunsTest, AFailedPreparationLeavesNothingReadyAndRetries) {
    ArtifactRoot root("retry");
    ChipSwimlaneCollector collector;
    // The explicit exits below are this case's subject; this one only covers an
    // early return on a failed assertion, and finds nothing left to do
    // otherwise.
    RetainedRunsTeardown teardown(collector, retained_free);
    ASSERT_EQ(collector.initialize(1, 1, 0, ChipSwimlaneLevel::TASK_TIMING, retained_alloc, nullptr, retained_free), 0);

    auto artifact_dirs = [&root]() {
        int dirs = 0;
        std::error_code ec;
        for (const auto &entry : fs::directory_iterator(root.path(), ec)) {
            if (entry.is_directory()) dirs++;
        }
        return dirs;
    };

    // Reader shards before the first admission, which is the order both runner
    // bases take: admitting a run waits for every shard to acknowledge the run
    // table.
    collector.configure_retained_runs(/*retain_across_runs=*/true, /*budget_bytes=*/1024);
    collector.start(retained_thread_factory);

    // The budget is taken before the directory, so a budget this small refuses
    // before anything is reserved.
    EXPECT_FALSE(collector.run_begin(1, root.str(), ChipSwimlaneLevel::TASK_TIMING));
    EXPECT_EQ(artifact_dirs(), 0) << "a failed preparation left a directory that reads as this collector's";

    // The same collector, a budget that works: the retry prepares and reserves
    // exactly one directory.
    collector.configure_retained_runs(/*retain_across_runs=*/true, simpler::dfx::runs::kDefaultBudgetBytes);
    ASSERT_TRUE(collector.run_begin(2, root.str(), ChipSwimlaneLevel::TASK_TIMING));
    EXPECT_EQ(artifact_dirs(), 1) << "the retry did not reserve its own directory";

    collector.run_close(2, /*bank_index=*/0, /*device_execution_complete=*/true);

    // Both exits twice. The first pair publishes and stops admitting; the
    // second finds the watermark already set and the resources already gone.
    collector.finish_retained_runs();
    collector.finish_retained_runs();
    EXPECT_EQ(published_files(root.path()).size(), 1u) << "the retried run did not publish exactly one artifact";
    collector.stop();
    collector.finalize(nullptr, retained_free);
    collector.finalize(nullptr, retained_free);
    EXPECT_EQ(published_files(root.path()).size(), 1u) << "a second finalize changed what had been published";
}

// The per-kind paired cap is twice the bytes `init()` seeded, so the
// seed has to mean one initialization's allocations and nothing else. Two
// lifecycle events would otherwise inflate it: an aborted init whose rollback
// frees everything it charged, and the pool's own growth. Both are
// reachable — the rollback guard runs on any late init failure, and `finalize`
// permits re-initialization on the same collector.
TEST(ChipSwimlaneRetainedRunsTest, PairedSeedIsPerInitAndUnmovedByGrowth) {
    ArtifactRoot root("pairedseed");
    constexpr int kKind = static_cast<int>(ProfBufferType::AICPU_TASK);

    // What one clean initialization seeds, and what a finalize leaves behind.
    size_t clean_seed = 0;
    {
        ChipSwimlaneCollector control;
        ASSERT_EQ(
            control.initialize(1, 1, 0, ChipSwimlaneLevel::TASK_TIMING, retained_alloc, nullptr, retained_free), 0
        );
        clean_seed = control.manager().paired_initial(kKind);
        control.finalize(nullptr, retained_free);
        EXPECT_EQ(control.manager().paired_initial(kKind), 0u) << "finalize left a seed behind for a re-init to add to";
        EXPECT_EQ(control.manager().paired_charged(kKind), 0u);
    }
    ASSERT_GT(clean_seed, 0u) << "this kind seeds nothing, so the case cannot discriminate";

    ChipSwimlaneCollector collector;
    // Enough allocations to charge some of this kind's buffers, not enough to
    // finish: the shm block comes first and is charged to no kind.
    g_alloc_budget = 3;
    EXPECT_NE(
        collector.initialize(1, 1, 0, ChipSwimlaneLevel::TASK_TIMING, retained_alloc_limited, nullptr, retained_free), 0
    );
    g_alloc_budget = -1;
    ASSERT_EQ(collector.initialize(1, 1, 0, ChipSwimlaneLevel::TASK_TIMING, retained_alloc, nullptr, retained_free), 0);
    EXPECT_EQ(collector.manager().paired_initial(kKind), clean_seed)
        << "the aborted init's charges outlived the rollback that freed them";

    // Growth moves the live total and leaves the seed alone, so a collector
    // opened after it derives the same cap as one opened before.
    const size_t live_before = collector.manager().paired_charged(kKind);
    const size_t published =
        collector.manager().allocate_recycled_batch(kKind, sizeof(ChipSwimlaneAicpuTaskBuffer), 4, /*shard_index=*/0);
    ASSERT_GT(published, 0u);
    EXPECT_GT(collector.manager().paired_charged(kKind), live_before) << "growth was not charged";
    EXPECT_EQ(collector.manager().paired_initial(kKind), clean_seed) << "growth redefined the seed a cap comes from";

    collector.finalize(nullptr, retained_free);
}

// The device-side bound is a byte figure, so an emptied mapping table
// is not enough to reconcile it: the release surface reports a status per
// pointer and carries no size, and a release that did not succeed leaves bytes
// the pool may still hold. Occupancy therefore stays charged, and the collector
// refuses rather than admitting against a figure it cannot state.
TEST(ChipSwimlaneRetainedRunsTest, UnprovedReleaseKeepsOccupancyAndRefusesRetention) {
    ArtifactRoot root("unproved");
    constexpr int kKind = static_cast<int>(ProfBufferType::AICPU_TASK);

    ChipSwimlaneCollector collector;
    // Declared before the first assertion, so an early return still joins.
    RetainedRunsTeardown teardown(collector, retained_free);
    ASSERT_EQ(collector.initialize(1, 1, 0, ChipSwimlaneLevel::TASK_TIMING, retained_alloc, nullptr, retained_free), 0);
    const size_t seeded = collector.manager().paired_initial(kKind);
    ASSERT_GT(seeded, 0u);
    EXPECT_FALSE(collector.manager().release_unproven());

    // Retention is prepared against a proved-clean pool, and releasing it
    // leaves the seed a later preparation's cap comes from untouched. The
    // reader shards start first, the order both runner bases take: admitting a
    // run waits for every collector shard to acknowledge the run table.
    collector.configure_retained_runs(true, simpler::dfx::runs::kDefaultBudgetBytes);
    collector.start(retained_thread_factory);
    ASSERT_TRUE(collector.run_begin(1, root.str(), ChipSwimlaneLevel::TASK_TIMING));
    collector.finish_retained_runs();
    EXPECT_EQ(collector.manager().paired_initial(kKind), seeded) << "preparing retention moved the seed";

    // Finalize with a free that reports failure. The mapping table empties
    // either way, which is exactly why emptiness cannot be the proof. This
    // joins the writer and the reader shards before it frees, as production
    // does.
    collector.finalize(nullptr, retained_free_failing);
    EXPECT_TRUE(collector.manager().release_unproven());
    EXPECT_GT(collector.manager().paired_charged(kKind), 0u) << "occupancy was erased without proof of release";
    EXPECT_GT(collector.manager().paired_initial(kKind), 0u);

    // And no further run is retained against that occupancy. This refusal is
    // on the unproved release, which the preparation tests before it takes the
    // budget and before it publishes anything for a shard to acknowledge — so
    // it needs no reader shard, and starting one would prove nothing about it.
    ASSERT_EQ(collector.initialize(1, 1, 0, ChipSwimlaneLevel::TASK_TIMING, retained_alloc, nullptr, retained_free), 0);
    collector.configure_retained_runs(true, simpler::dfx::runs::kDefaultBudgetBytes);
    EXPECT_FALSE(collector.run_begin(2, root.str(), ChipSwimlaneLevel::TASK_TIMING))
        << "a run was retained while an unproved release was outstanding";
    // The single-run path stays usable, as it does for every other refusal.
    collector.begin_run(root.str(), ChipSwimlaneLevel::TASK_TIMING);
}

// The same admission rule has to hold for a buffer that never reached the
// manager's mapping table. `alloc_paired_buffer` cleans up its own device
// pointer when registration or the initial copy fails, and returns before
// registering it — so the rollback guard never sees that pointer, and this is
// the only place the outcome of its release can be recorded. A cleanup that did
// not report success leaves memory held just the same.
TEST(ChipSwimlaneRetainedRunsTest, UnregisteredInitCleanupFailureAlsoRefusesRetention) {
    ArtifactRoot root("initcleanup");
    ChipSwimlaneCollector collector;

    // Registration refuses on the very first paired allocation, and the free
    // that follows reports failure while still reclaiming the memory.
    EXPECT_NE(
        collector.initialize(
            1, 1, 0, ChipSwimlaneLevel::TASK_TIMING, retained_alloc, retained_register_failing, retained_free_failing
        ),
        0
    );
    EXPECT_TRUE(collector.manager().release_unproven())
        << "a cleanup the rollback guard cannot see reported nothing at all";

    // A later initialization succeeds, and the run is still not retained: the
    // unproved occupancy belongs to the pool, not to the failed attempt.
    ASSERT_EQ(collector.initialize(1, 1, 0, ChipSwimlaneLevel::TASK_TIMING, retained_alloc, nullptr, retained_free), 0);
    collector.configure_retained_runs(true, simpler::dfx::runs::kDefaultBudgetBytes);
    EXPECT_FALSE(collector.run_begin(1, root.str(), ChipSwimlaneLevel::TASK_TIMING))
        << "a run was retained after an unregistered buffer's cleanup failed";
    collector.begin_run(root.str(), ChipSwimlaneLevel::TASK_TIMING);
    collector.finalize(nullptr, retained_free);
}

// With retention off nothing about the existing path moves: the legacy
// artifact name, the legacy location, and no `collection` object.
TEST(ChipSwimlaneRetainedRunsTest, RetentionOffKeepsTheLegacyArtifactExactly) {
    ArtifactRoot root("legacy");
    ChipSwimlaneCollector collector;
    ASSERT_EQ(collector.initialize(1, 1, 0, ChipSwimlaneLevel::TASK_TIMING, retained_alloc, nullptr, retained_free), 0);
    EXPECT_FALSE(collector.retains_runs());

    collector.begin_run(root.str(), ChipSwimlaneLevel::TASK_TIMING);
    set_platform_run_result(0, 9801);
    set_chip_swimlane_enabled(true);
    set_platform_chip_swimlane_base(reinterpret_cast<uint64_t>(collector.get_chip_swimlane_setup_device_ptr()));
    set_platform_chip_swimlane_aicore_rotation_table(0);
    set_platform_chip_swimlane_run_terminal_bank(reinterpret_cast<uint64_t>(collector.arm_run_terminal_bank(0, 9801)));
    chip_swimlane_aicpu_init(1);
    chip_swimlane_aicpu_on_aicore_dispatch(0, 0, 1);
    auto *ac_state = get_aicore_buffer_state(collector.get_chip_swimlane_setup_device_ptr(), 0);
    auto *buf = reinterpret_cast<ChipSwimlaneAicoreTaskBuffer *>(ac_state->head.current_buf_ptr);
    ASSERT_NE(buf, nullptr);
    buf->records[0].start_time = 10;
    buf->records[0].end_time = 20;
    buf->records[0].reg_task_id = 1;
    const int cores[] = {0};
    chip_swimlane_aicpu_flush(0, cores, 1);

    collector.start(retained_thread_factory);
    collector.quiesce();
    collector.stop();
    collector.reconcile_counters();
    EXPECT_EQ(collector.export_swimlane_json(), 0);

    const fs::path legacy = root.path() / "chip_swimlane_records.json";
    ASSERT_TRUE(fs::exists(legacy)) << "the legacy artifact name or location changed";
    EXPECT_EQ(read_file(legacy).find("\"collection\""), std::string::npos)
        << "the default path emitted the retained-run metadata object";
    EXPECT_TRUE(published_files(root.path()).empty()) << "the default path created an artifact directory";

    collector.finalize(nullptr, retained_free);
}

// Six runs over two slots, so every slot is armed, retired and armed again
// three times while the drain owners and collector shards keep running. A cut
// slot handed back before its readers had left it would be reinitialized under
// one of them, and the run whose arrays were overwritten would have its records
// land in the wrong artifact or in none. Each run's own count in its own file is
// what rules that out.
TEST(ChipSwimlaneRetainedRunsTest, SlotReuseAcrossRunsKeepsEachRunsRecordsInItsOwnFile) {
    RetainedRunsFixture fx("reuse", /*cores=*/1, /*threads=*/1);
    const int cores[] = {0};
    constexpr uint64_t kBase = 9901;
    constexpr int kRuns = 6;

    for (int i = 0; i < kRuns; i++) {
        const uint64_t epoch = kBase + static_cast<uint64_t>(i);
        fx.begin(epoch);
        fx.dispatch(/*core_id=*/0, i + 1);
        fx.close(epoch, cores, 1);
    }
    ASSERT_TRUE(fx.wait_for_files(kRuns)) << "the collector did not publish every run";

    std::string error;
    EXPECT_TRUE(fx.collector.flush_retained_runs(8000, &error)) << error;
    for (int i = 0; i < kRuns; i++) {
        const uint64_t epoch = kBase + static_cast<uint64_t>(i);
        const fs::path expected = fx.dir.path() / "swimlane-0" / ("records_e" + std::to_string(epoch) + ".json");
        ASSERT_TRUE(fs::exists(expected)) << expected.string();
        EXPECT_EQ(rows_for_epoch(read_file(expected), epoch), static_cast<size_t>(i + 1))
            << "epoch " << epoch << " did not carry exactly its own records";
    }
    EXPECT_EQ(fx.collector.retained_run_stats_for_test().open_slots, 0u);
}

// The host budget is a bound on retained records, not a number in a log line.
// Past it the epoch stops retaining, keeps its receipts, and publishes an
// artifact that says its content is incomplete — and the charge never goes
// beyond the limit retention was configured with.
TEST(ChipSwimlaneRetainedRunsTest, BudgetExhaustionStopsRetentionAndPublishesAPartial) {
    // Fixed overhead plus a little over the minimum working set.
    constexpr size_t kBudget = 18ull * 1024 * 1024;
    RetainedRunsFixture fx("exhaust", /*cores=*/1, /*threads=*/1, kBudget);
    const int cores[] = {0};
    constexpr uint64_t kEpoch = 9801;
    fx.begin(kEpoch);

    // Fed straight to the collector rather than through the device queues: the
    // charge follows the records either way, and this keeps the volume
    // independent of how fast the pools recycle. Nothing else produces on
    // shard 0 here, so this thread is its only writer.
    const size_t per_buffer =
        static_cast<size_t>(PLATFORM_AICORE_BUFFER_SIZE) * sizeof(CollectedRecord<ChipSwimlaneAicoreTaskRecord>);
    const size_t cap = (kBudget / per_buffer) * 2 + 16;
    ChipSwimlaneAicoreTaskBuffer buf{};
    buf.run_epoch = kEpoch;
    buf.count = static_cast<uint32_t>(PLATFORM_AICORE_BUFFER_SIZE);
    for (int i = 0; i < PLATFORM_AICORE_BUFFER_SIZE; i++) {
        buf.records[i].start_time = 1000 + static_cast<uint64_t>(i);
        buf.records[i].end_time = 2000 + static_cast<uint64_t>(i);
    }
    ReadyBufferInfo info{};
    info.type = ProfBufferType::AICORE_TASK;
    info.index = 0;
    info.dev_buffer_ptr = &buf;
    info.host_buffer_ptr = &buf;

    size_t fed = 0;
    for (; fed < cap; fed++) {
        fx.collector.on_buffer_collected(info, /*collector_shard=*/0);
        if (fx.collector.retained_run_stats_for_test().budget_refusals > 0) break;
    }
    const auto stats = fx.collector.retained_run_stats_for_test();
    ASSERT_GT(stats.budget_refusals, 0u) << "fed " << fed << " buffers of " << per_buffer << " B without a refusal";
    EXPECT_LE(stats.host_charged, kBudget) << "the charge went past the budget it was opened with";
    EXPECT_FALSE(stats.fatal);

    fx.close(kEpoch, cores, 1);
    ASSERT_TRUE(fx.wait_for_files(1));
    const std::string body = read_file(published_files(fx.dir.path())[0]);
    EXPECT_NE(body.find("partial_safe"), std::string::npos) << body.substr(0, 600);
}

// A collector whose control handshake never completes is fatal before any epoch
// has a verdict of its own. Flush has to fail on the fatal itself: the
// per-epoch rows are empty, and reading empty as clean would report a
// successful flush over a collector that wrote nothing.
TEST(ChipSwimlaneRetainedRunsTest, FatalWithNoEpochVerdictStillFailsFlush) {
    ReaderlessCollector rs("fatalflush");

    // No collector shard is polling, so the epoch table can never be
    // acknowledged and admission ends in the collector's fatal.
    EXPECT_FALSE(rs.collector.run_begin(9001, rs.dir.str(), ChipSwimlaneLevel::TASK_TIMING));
    ASSERT_TRUE(rs.collector.retained_run_stats_for_test().fatal);
    EXPECT_EQ(rs.collector.retained_run_stats_for_test().published, 0u);

    std::string error;
    EXPECT_FALSE(rs.collector.flush_retained_runs(200, &error)) << "a fatal collector reported a successful flush";
    EXPECT_NE(error.find("fatal"), std::string::npos) << error;
    EXPECT_TRUE(published_files(rs.dir.path()).empty());

    rs.collector.finish_retained_runs();
    rs.collector.stop();
    rs.collector.finalize(nullptr, retained_free);
}

// A close cannot prove a stalled reader has let go, so it must not free that
// epoch's storage. The publisher's join says nothing about the collector
// shards, and the production close path runs before they are joined at all —
// so the release waits for `finalize()`, which joins them first.
//
// The shape: no shard ever polls, which is what makes every acknowledgement
// unprovable. That also makes admission itself end in the collector's fatal,
// so this case names that state rather than assuming a clean admission, and it
// awaits the seal's verdict on its own observable rather than through a
// `finish` that returns on the fatal.
TEST(ChipSwimlaneRetainedRunsTest, CloseDefersQuarantinedStorageUntilReadersAreJoined) {
    ReaderlessCollector rs("deferred");
    constexpr uint64_t kEpoch = 9002;

    // Admission publishes the fatal, because the epoch table can never be
    // acknowledged. The slot is claimed either way, which is what gives the
    // writer something to seal below.
    EXPECT_FALSE(rs.collector.run_begin(kEpoch, rs.dir.str(), ChipSwimlaneLevel::TASK_TIMING));
    ASSERT_TRUE(rs.collector.retained_run_stats_for_test().fatal);
    ASSERT_EQ(rs.collector.retained_run_stats_for_test().open_slots, 1u);

    // The epoch now has a target, so the writer seals it and cannot prove the
    // reference released. `finish_retained_runs` is still called, because that
    // is what production does, but its flush returns on the fatal — so the
    // verdict is awaited directly.
    rs.collector.run_close(kEpoch, /*bank_index=*/0, /*device_execution_complete=*/true);
    rs.collector.finish_retained_runs();
    ASSERT_TRUE(rs.wait_for_deferred_release()) << "the unprovable seal never reached a quarantine verdict";

    // Reached that verdict without freeing anything: the slot is still
    // occupied, the verdict says the reference was never proved released, and
    // no file was written. The verdict row is recorded before the flag waited
    // on above, so reading it here cannot race the seal.
    const auto after_close = rs.collector.retained_run_stats_for_test();
    EXPECT_EQ(after_close.open_slots, 1u) << "the seal freed storage a stalled reader may still hold";
    EXPECT_EQ(after_close.quarantined, 1u) << "the storage was held on something other than a quarantine";
    EXPECT_TRUE(published_files(rs.dir.path()).empty());

    // stop() joins every reader; only then may the storage go back.
    rs.collector.stop();
    rs.collector.finalize(nullptr, retained_free);
    const auto after_join = rs.collector.retained_run_stats_for_test();
    EXPECT_EQ(after_join.open_slots, 0u) << "the deferred release did not happen after the readers were joined";
    EXPECT_FALSE(after_join.release_deferred);
}

// `run_begin` blocks on the epoch table being acknowledged by every
// collector shard, and a shard with an empty ready ring is asleep in a 100 ms
// cv tick. Notifying that ring is not enough on its own: the notification
// advances no ready shard's state epoch, so a predicate that knows nothing
// about control re-tests false and the shard sleeps out the rest of its tick.
// With the control condition in the predicate, each handshake costs a wakeup
// instead of a timer. Timed with the ring deliberately empty and each epoch
// published before the next admission, so the interval measured contains the
// handshake and not a wait for capacity.
TEST(ChipSwimlaneRetainedRunsTest, ControlHandshakeWakesAnIdleCollectorWithoutItsRingTick) {
    RetainedRunsFixture fx("ctrlwake", /*cores=*/1, /*threads=*/1);
    const int cores[] = {0};
    constexpr uint64_t kBase = 9951;
    constexpr int kCycles = 8;
    // Eight wakeups against eight 100 ms ticks. Loose enough that ordinary load
    // cannot reach it, and a timer-bound handshake cannot come near it.
    constexpr int kBudgetMs = 200;

    std::chrono::nanoseconds admitting{0};
    for (int i = 0; i < kCycles; i++) {
        const uint64_t epoch = kBase + static_cast<uint64_t>(i);
        const auto started = std::chrono::steady_clock::now();
        ASSERT_TRUE(fx.collector.run_begin(epoch, fx.dir.str(), ChipSwimlaneLevel::TASK_TIMING));
        admitting += std::chrono::steady_clock::now() - started;

        set_platform_run_result(/*region_base=*/0, epoch);
        set_chip_swimlane_enabled(true);
        set_platform_chip_swimlane_base(reinterpret_cast<uint64_t>(fx.shm()));
        set_platform_chip_swimlane_aicore_rotation_table(0);
        set_platform_chip_swimlane_run_terminal_bank(
            reinterpret_cast<uint64_t>(fx.collector.arm_run_terminal_bank(/*bank_index=*/0, epoch))
        );
        chip_swimlane_aicpu_init(fx.num_aicore);
        fx.close(epoch, cores, 1);
        // Published before the next admission, so no `run_begin` below waits
        // for a slot and the measurement stays about the handshake.
        ASSERT_TRUE(fx.wait_for_files(static_cast<size_t>(i) + 1)) << "epoch " << epoch << " was not published";
    }

    const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(admitting).count();
    EXPECT_LT(total_ms, kBudgetMs) << kCycles << " control handshakes took " << total_ms
                                   << " ms, which is the ready-ring tick rather than a wakeup";
}

// Metadata a caller can size arbitrarily is charged at its sources *before* it
// is copied. A refusal discovered after the copy could only answer by keeping
// storage the budget said it could not pay for, or by freeing records whose
// readers have not been released — so the copy never happens, the artifact
// says its metadata is incomplete, and the charge stays inside the limit.
TEST(ChipSwimlaneRetainedRunsTest, OversizedRunMetadataIsRefusedBeforeItIsCopied) {
    constexpr size_t kBudget = 18ull * 1024 * 1024;
    RetainedRunsFixture fx("metabudget", /*cores=*/1, /*threads=*/1, kBudget);
    const int cores[] = {0};
    constexpr uint64_t kEpoch = 9851;

    fx.begin(kEpoch);
    fx.dispatch(/*core_id=*/0, 3);

    // One extension bigger than the whole budget, so its admission cannot be
    // paid whatever else this epoch holds. A lifecycle-records payload, which
    // no other section of this artifact can be confused with.
    const std::string marker = "\"ZZrefusedmetadataZZ\"";
    std::string oversized = "[";
    oversized.reserve(kBudget + 4096);
    while (oversized.size() < kBudget + 1024) {
        oversized += marker;
        oversized += ",";
    }
    oversized += "0]";
    ASSERT_TRUE(fx.collector.set_json_extension(ChipSwimlaneExtensionSection::AicpuLifecycleRecords, oversized));

    fx.close(kEpoch, cores, 1);
    ASSERT_TRUE(fx.wait_for_files(1));
    const std::string body = read_file(published_files(fx.dir.path())[0]);
    EXPECT_NE(body.find("\"metadata_complete\": false"), std::string::npos) << body.substr(0, 700);
    EXPECT_EQ(body.find(marker), std::string::npos) << "the refused metadata was copied into the artifact anyway";
    EXPECT_NE(body.find("partial_safe"), std::string::npos) << "an incomplete artifact was reported as settled";

    const auto stats = fx.collector.retained_run_stats_for_test();
    EXPECT_LE(stats.host_charged, kBudget) << "the refused charge was taken anyway";
    EXPECT_GT(stats.budget_refusals, 0u);
    // The epoch's own records survived the refusal: only the metadata was
    // declined, and nothing was freed ahead of its reference proof.
    EXPECT_EQ(rows_for_epoch(body, kEpoch), 3u) << "the refusal took this epoch's records with it";
}

// Two retained runs, each publishing host phase records of its own. The epoch's
// metadata snapshot copies whatever the collector holds when the run closes,
// and the collector holds one copy of it across every run it serves — so the
// artifact describes its own run only when the publication precedes the
// snapshot. Driven through the boundary helper both runner bases call, which
// is where that order lives. The two runs publish different counts, which is
// what the metadata reports at every level.
TEST(ChipSwimlaneRetainedRunsTest, HostPhaseRecordsReachTheEpochThatProducedThem) {
    RetainedRunsFixture fx("host-phase-own-epoch", /*cores=*/1, /*threads=*/1);
    const int cores[] = {0};
    constexpr uint64_t kFirst = 700;
    constexpr uint64_t kSecond = 701;

    fx.begin(kFirst);
    fx.dispatch(0, 1);
    fx.close_through_boundary(kFirst, cores, 1, [&fx] {
        fx.collector.set_host_phase_records(host_submit_rows(1), {}, 1, 1, 0);
    });

    fx.begin(kSecond);
    fx.dispatch(0, 1);
    fx.close_through_boundary(kSecond, cores, 1, [&fx] {
        fx.collector.set_host_phase_records(host_submit_rows(2), {}, 2, 2, 0);
    });

    ASSERT_TRUE(fx.wait_for_files(2));
    size_t seen = 0;
    for (const auto &f : published_files(fx.dir.path())) {
        const std::string name = f.filename().string();
        const std::string body = read_file(f);
        if (name == "records_e700.json") {
            seen++;
            EXPECT_NE(body.find(recorded_records_field(1)), std::string::npos)
                << "epoch 700 did not carry the host phase records it produced";
            EXPECT_EQ(body.find(recorded_records_field(2)), std::string::npos)
                << "epoch 700 carried its successor's host phase records";
        } else if (name == "records_e701.json") {
            seen++;
            EXPECT_NE(body.find(recorded_records_field(2)), std::string::npos)
                << "epoch 701 did not carry the host phase records it produced";
            EXPECT_EQ(body.find(recorded_records_field(1)), std::string::npos)
                << "epoch 701 carried its predecessor's host phase records";
        }
    }
    EXPECT_EQ(seen, 2u) << "one of the two epochs published no artifact";
}

// A run that produces no host phase records at all. The collector's copy is
// per run, so this one contributes none rather than inheriting what its
// predecessor left in the same fields.
TEST(ChipSwimlaneRetainedRunsTest, ARetainedRunWithoutHostPhaseRecordsInheritsNone) {
    RetainedRunsFixture fx("host-phase-empty-successor", /*cores=*/1, /*threads=*/1);
    const int cores[] = {0};
    constexpr uint64_t kFirst = 710;
    constexpr uint64_t kSecond = 711;

    fx.begin(kFirst);
    fx.dispatch(0, 1);
    fx.close_through_boundary(kFirst, cores, 1, [&fx] {
        fx.collector.set_host_phase_records(host_submit_rows(1), {}, 1, 1, 0);
    });

    // Nothing published: the production path skips the publication entirely
    // when a run's host phase store never finished a pass.
    fx.begin(kSecond);
    fx.dispatch(0, 1);
    fx.close_through_boundary(kSecond, cores, 1, [] {});

    ASSERT_TRUE(fx.wait_for_files(2));
    size_t seen = 0;
    for (const auto &f : published_files(fx.dir.path())) {
        const std::string name = f.filename().string();
        const std::string body = read_file(f);
        if (name == "records_e710.json") {
            seen++;
            EXPECT_NE(body.find(recorded_records_field(1)), std::string::npos)
                << "epoch 710 lost the host phase records it produced";
        } else if (name == "records_e711.json") {
            seen++;
            EXPECT_EQ(body.find("\"orchestrator_source\": \"host\""), std::string::npos)
                << "epoch 711 produced no host phase records but reported its predecessor's";
            EXPECT_EQ(body.find("\"host_capture\""), std::string::npos)
                << "epoch 711 reported a host capture it never made";
        }
    }
    EXPECT_EQ(seen, 2u) << "one of the two epochs published no artifact";
}

// The close is what hands an epoch to the writer and releases its
// slot, so a publication that throws must not skip it: two slots exist, and an
// run left open holds one for the collector's whole life. The failure still
// reaches the caller, and the epoch reports no host capture it did not make.
TEST(ChipSwimlaneRetainedRunsTest, AFailedHostPublicationStillClosesItsEpoch) {
    RetainedRunsFixture fx("host-phase-failed-publication", /*cores=*/1, /*threads=*/1);
    const int cores[] = {0};
    constexpr uint64_t kFirst = 720;
    constexpr uint64_t kSecond = 721;

    fx.begin(kFirst);
    fx.dispatch(0, 1);
    EXPECT_THROW(
        fx.close_through_boundary(
            kFirst, cores, 1,
            [] {
                throw std::runtime_error("host phase publication failed");
            }
        ),
        std::runtime_error
    );
    ASSERT_TRUE(fx.wait_for_files(1)) << "the epoch was never closed, so it never published";

    // The slot came back, so a later run is still admissible and publishes its
    // own host phase records.
    fx.begin(kSecond);
    fx.dispatch(0, 1);
    fx.close_through_boundary(kSecond, cores, 1, [&fx] {
        fx.collector.set_host_phase_records(host_submit_rows(1), {}, 1, 1, 0);
    });
    ASSERT_TRUE(fx.wait_for_files(2)) << "the failed publication's slot was never released";
    size_t seen = 0;
    for (const auto &f : published_files(fx.dir.path())) {
        const std::string name = f.filename().string();
        const std::string body = read_file(f);
        if (name == "records_e720.json") {
            seen++;
            EXPECT_EQ(body.find("\"host_capture\""), std::string::npos)
                << "the epoch whose publication threw reported a host capture anyway";
        } else if (name == "records_e721.json") {
            seen++;
            EXPECT_NE(body.find(recorded_records_field(1)), std::string::npos)
                << "the run after the failed publication lost its own host phase records";
        }
    }
    EXPECT_EQ(seen, 2u) << "one of the two epochs published no artifact";
}

// A publication that writes part of its host state and then fails — the shape
// the sim base has, where the phase records land and the runtime extensions
// throw part-way. What it published is kept, and the epoch says it is
// incomplete: a partial verdict rather than a publication, so a reader is
// never told a half-written capture is the whole run. The collector carries on,
// and the next epoch settles complete.
TEST(ChipSwimlaneRetainedRunsTest, APartialHostPublicationMarksItsEpochIncomplete) {
    RetainedRunsFixture fx("host-phase-partial-publication", /*cores=*/1, /*threads=*/1);
    const int cores[] = {0};
    constexpr uint64_t kPartial = 730;
    constexpr uint64_t kWhole = 731;

    fx.begin(kPartial);
    fx.dispatch(0, 1);
    EXPECT_THROW(
        fx.close_through_boundary(
            kPartial, cores, 1,
            [&fx] {
                fx.collector.set_host_phase_records(host_submit_rows(1), {}, 1, 1, 0);
                throw std::runtime_error("runtime extension publication failed");
            }
        ),
        std::runtime_error
    );

    fx.begin(kWhole);
    fx.dispatch(0, 1);
    fx.close_through_boundary(kWhole, cores, 1, [&fx] {
        fx.collector.set_host_phase_records(host_submit_rows(2), {}, 2, 2, 0);
    });

    ASSERT_TRUE(fx.wait_for_files(2)) << "the partial epoch stopped the collector from making progress";
    size_t seen = 0;
    for (const auto &f : published_files(fx.dir.path())) {
        const std::string name = f.filename().string();
        const std::string body = read_file(f);
        if (name == "records_e730.json") {
            seen++;
            EXPECT_NE(body.find("\"metadata_complete\": false"), std::string::npos)
                << "a publication that failed part-way was sealed as complete";
            EXPECT_NE(body.find("partial_safe"), std::string::npos)
                << "an epoch with incomplete metadata settled as a publication";
            // What did land is kept rather than discarded: the artifact
            // reports the one record the publication managed.
            EXPECT_NE(body.find(recorded_records_field(1)), std::string::npos)
                << "the part that published was thrown away with the part that did not";
        } else if (name == "records_e731.json") {
            seen++;
            EXPECT_NE(body.find("\"metadata_complete\": true"), std::string::npos)
                << "the epoch after a partial one inherited its incompleteness";
            EXPECT_NE(body.find(recorded_records_field(2)), std::string::npos)
                << "the epoch after a partial one lost its own host phase records";
        }
    }
    EXPECT_EQ(seen, 2u) << "one of the two epochs published no artifact";
}

namespace {
/** A failure the boundary's report cannot describe: it carries no message. */
struct UnnameableFailure {};
}  // namespace

// The boundary sets its state before it describes anything, so a failure the
// report cannot name costs neither the epoch's verdict nor its close. This one
// throws an object with no message at all, which is the shape every step of
// the diagnostic gives up on — and the epoch is still marked incomplete, still
// closed, still published, and its slot still comes back for the next run.
TEST(ChipSwimlaneRetainedRunsTest, AnUnnameableHostPublicationFailureStillClosesItsEpoch) {
    RetainedRunsFixture fx("host-phase-unnameable-failure", /*cores=*/1, /*threads=*/1);
    const int cores[] = {0};
    constexpr uint64_t kUnnameable = 740;
    constexpr uint64_t kNext = 741;

    fx.begin(kUnnameable);
    fx.dispatch(0, 1);
    EXPECT_THROW(
        fx.close_through_boundary(
            kUnnameable, cores, 1,
            [&fx] {
                fx.collector.set_host_phase_records(host_submit_rows(1), {}, 1, 1, 0);
                throw UnnameableFailure{};
            }
        ),
        UnnameableFailure
    );

    fx.begin(kNext);
    fx.dispatch(0, 1);
    fx.close_through_boundary(kNext, cores, 1, [&fx] {
        fx.collector.set_host_phase_records(host_submit_rows(2), {}, 2, 2, 0);
    });

    ASSERT_TRUE(fx.wait_for_files(2)) << "an unnameable failure cost the run its close or the collector its slot";
    size_t seen = 0;
    for (const auto &f : published_files(fx.dir.path())) {
        const std::string name = f.filename().string();
        const std::string body = read_file(f);
        if (name == "records_e740.json") {
            seen++;
            EXPECT_NE(body.find("\"metadata_complete\": false"), std::string::npos)
                << "a failure the report could not name was sealed as complete";
            EXPECT_NE(body.find("partial_safe"), std::string::npos);
            EXPECT_NE(body.find(recorded_records_field(1)), std::string::npos)
                << "the part that published before the failure was discarded";
        } else if (name == "records_e741.json") {
            seen++;
            EXPECT_NE(body.find("\"metadata_complete\": true"), std::string::npos)
                << "the epoch after an unnameable failure inherited its incompleteness";
        }
    }
    EXPECT_EQ(seen, 2u) << "one of the two epochs published no artifact";
    // The collector is still usable: a run-scoped failure is not a
    // collector-level one, so nothing here may have raised the sticky fatal.
    EXPECT_FALSE(fx.collector.retained_run_stats_for_test().fatal)
        << "a run's publication failure was escalated to a collector fatal";
}
