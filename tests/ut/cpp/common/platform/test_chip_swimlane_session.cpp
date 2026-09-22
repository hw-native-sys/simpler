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
 * Continuous-collection session: a run's host-side receipt, sealing and file
 * write continue while the next run executes.
 *
 * These drive the real thing — the collector's mgmt and poll threads, the
 * device-side producer, the per-queue transport cut and the session thread
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
#include "host/session_run_boundary.h"

namespace fs = std::filesystem;

namespace {

void *session_alloc(size_t size) { return std::calloc(1, size); }

// An allocator that runs out after a fixed number of successful calls, so a
// collector's `init()` fails part-way and its rollback guard runs.
int g_alloc_budget = -1;

void *session_alloc_limited(size_t size) {
    if (g_alloc_budget == 0) return nullptr;
    if (g_alloc_budget > 0) g_alloc_budget--;
    return std::calloc(1, size);
}

int session_free(void *ptr) {
    std::free(ptr);
    return 0;
}

// A free that reports failure while still reclaiming the memory, so the case
// leaks nothing and the collector sees only the status a real failure carries.
int session_free_failing(void *ptr) {
    std::free(ptr);
    return -1;
}

// A host-mapping registration that refuses, which is what drives `init()` into
// the cleanup path for a device pointer it never registered with the manager.
int session_register_failing(void *dev_ptr, size_t size, int device_id, void **host_ptr_out) {
    (void)dev_ptr;
    (void)size;
    (void)device_id;
    if (host_ptr_out != nullptr) *host_ptr_out = nullptr;
    return -1;
}

std::thread session_thread_factory(std::function<void()> fn) { return std::thread(std::move(fn)); }

/** A private output root per case, removed at teardown. */
class SessionDir {
public:
    explicit SessionDir(const char *name) {
        path_ =
            fs::temp_directory_path() / ("simpler-dfx-session-" + std::string(name) + "-" + std::to_string(::getpid()));
        std::error_code ec;
        fs::remove_all(path_, ec);
        fs::create_directories(path_, ec);
    }
    ~SessionDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    std::string str() const { return path_.string(); }
    fs::path path() const { return path_; }

private:
    fs::path path_;
};

/** Files a session published, across every reserved session directory. */
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
 * One collector with its threads running and a session open.
 *
 * `aicpu_thread_num` is what gives the run more than one device ready queue,
 * which the cross-queue case needs: a queue is owned by exactly one drain
 * thread and its target may only ever be discharged by its own traffic.
 */
struct SessionFixture {
    ChipSwimlaneCollector collector;
    SessionDir dir;
    int num_aicore;

    SessionFixture(const char *name, int cores, int threads, size_t budget_bytes = 0) :
        dir(name),
        num_aicore(cores) {
        EXPECT_EQ(
            collector.initialize(
                cores, threads, /*device_id=*/0, ChipSwimlaneLevel::TASK_TIMING, session_alloc, nullptr, session_free
            ),
            0
        );
        ChipSwimlaneCollector::SessionOptions options;
        options.enabled = true;
        if (budget_bytes != 0) options.budget_bytes = budget_bytes;
        EXPECT_TRUE(collector.session_open(options, dir.str()));
        collector.start(session_thread_factory);
    }

    ~SessionFixture() {
        collector.session_close();
        collector.stop();
        collector.finalize(nullptr, session_free);
    }

    void *shm() { return collector.get_chip_swimlane_setup_device_ptr(); }

    /** Open a run: arm its bank, admit the epoch, bring the device side up. */
    void begin(uint64_t epoch) {
        ASSERT_NE(shm(), nullptr);
        EXPECT_TRUE(collector.session_run_begin(epoch, dir.str(), ChipSwimlaneLevel::TASK_TIMING));
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

    /** Close the device side of a run, then hand the epoch to the session. */
    void close(uint64_t epoch, const int *cores, int core_num, int thread_idx = 0) {
        chip_swimlane_aicpu_flush(thread_idx, cores, core_num);
        collector.session_run_close(epoch, /*bank_index=*/0, /*device_execution_complete=*/true);
    }

    /**
     * Close a run the way both runner bases do: whatever this run produced on
     * the host reaches the collector through the shared boundary helper, which
     * is what orders it against the epoch's metadata snapshot.
     */
    template <typename PublishHostState>
    void close_through_boundary(uint64_t epoch, const int *cores, int core_num, PublishHostState &&publish) {
        chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, core_num);
        simpler::dfx::session::close_session_run(
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
 * A collector with a session open but no reader threads started.
 *
 * This is the shape every unprovable close has: with no collector shard
 * polling, no reference-release acknowledgement can ever land, so the session
 * reaches its fatal and quarantine paths through production code rather than
 * through an injected failure.
 */
struct ReaderlessSession {
    ChipSwimlaneCollector collector;
    SessionDir dir;

    explicit ReaderlessSession(const char *name) :
        dir(name) {
        EXPECT_EQ(
            collector.initialize(
                /*cores=*/1, /*threads=*/1, /*device_id=*/0, ChipSwimlaneLevel::TASK_TIMING, session_alloc, nullptr,
                session_free
            ),
            0
        );
        ChipSwimlaneCollector::SessionOptions options;
        options.enabled = true;
        EXPECT_TRUE(collector.session_open(options, dir.str()));
    }
};

}  // namespace

// Three runs in a row, with nobody calling flush. The session thread is what
// frees a slot, so if publication were only a caller's job the third
// `run_begin` would block for ever against a two-slot cap.
TEST(ChipSwimlaneSessionTest, ThreeRunsProgressWithoutAnyFlush) {
    SessionFixture fx("progress", /*cores=*/1, /*threads=*/1);
    const int cores[] = {0};
    for (uint64_t epoch = 9001; epoch <= 9003; epoch++) {
        fx.begin(epoch);
        fx.dispatch(/*core_id=*/0, 4);
        fx.close(epoch, cores, 1);
    }
    EXPECT_TRUE(fx.wait_for_files(3)) << "the session thread did not publish on its own";

    std::string error;
    EXPECT_TRUE(fx.collector.session_flush(4000, &error)) << error;
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
TEST(ChipSwimlaneSessionTest, PredecessorRecordsSurviveSuccessorAdmission) {
    SessionFixture fx("overlap", /*cores=*/1, /*threads=*/1);
    const int cores[] = {0};
    constexpr uint64_t kFirst = 9101;
    constexpr uint64_t kSecond = 9102;

    fx.begin(kFirst);
    fx.dispatch(/*core_id=*/0, 6);
    chip_swimlane_aicpu_flush(0, cores, 1);
    // The successor is admitted before the predecessor is sealed.
    fx.collector.session_run_close(kFirst, 0, true);
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

// A predecessor's AICore buffer that arrives *after* the successor was armed is
// still the predecessor's. The arm moves the collector's "most recently armed
// run", so comparing a record's stamp against that field classifies every late
// predecessor buffer as foreign — the normal case here, not an exception — and
// publishes an artifact that lists the rows while reporting none collected.
// AICore is its own path: it is the only producer class whose records carry a
// per-buffer identity decision, so the AICPU-record cases above do not cover
// it.
TEST(ChipSwimlaneSessionTest, LatePredecessorAicoreBufferCountsAsItsOwnEpochsRecords) {
    SessionFixture fx("lateaicore", /*cores=*/1, /*threads=*/1);
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

    const auto stats = fx.collector.session_stats_for_test();
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
TEST(ChipSwimlaneSessionTest, OneQueuesTrafficDoesNotDischargeAnother) {
    SessionFixture fx("twoqueue", /*cores=*/2, /*threads=*/2);
    const int all_cores[] = {0, 1};
    constexpr uint64_t kEpoch = 9201;

    fx.begin(kEpoch);
    // Core 0 is served by thread 0's queue, core 1 by thread 1's.
    fx.dispatch(/*core_id=*/0, 3);
    fx.dispatch(/*core_id=*/1, 5);
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, &all_cores[0], 1);
    chip_swimlane_aicpu_flush(/*thread_idx=*/1, &all_cores[1], 1);
    fx.collector.session_run_close(kEpoch, 0, true);

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
TEST(ChipSwimlaneSessionTest, RunWithoutTerminalStillPublishesAndReleasesItsSlot) {
    SessionFixture fx("noterminal", /*cores=*/1, /*threads=*/1);
    const int cores[] = {0};
    constexpr uint64_t kEpoch = 9301;

    fx.begin(kEpoch);
    fx.dispatch(/*core_id=*/0, 2);
    chip_swimlane_aicpu_flush(0, cores, 1);
    // device_execution_complete = false: the recovery path, where the bank says
    // nothing about this run.
    fx.collector.session_run_close(kEpoch, 0, /*device_execution_complete=*/false);

    ASSERT_TRUE(fx.wait_for_files(1));
    const std::string body = read_file(published_files(fx.dir.path())[0]);
    EXPECT_NE(body.find("\"processing_complete\": false"), std::string::npos) << body.substr(0, 400);
    EXPECT_NE(body.find("partial_cut_unknown"), std::string::npos);

    // The slot came back: a further run is admitted without waiting.
    const auto stats = fx.collector.session_stats_for_test();
    EXPECT_EQ(stats.open_slots, 0u);
    EXPECT_FALSE(stats.fatal);
}

// A publication that cannot happen must not be silently forgotten: no file, an
// error from flush, and the slot still released so the session keeps running.
TEST(ChipSwimlaneSessionTest, WriteFailureIsReportedAndStillReleasesTheSlot) {
    SessionFixture fx("writefail", /*cores=*/1, /*threads=*/1);
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
        if (!fx.collector.session_flush(200, &error)) {
            reported = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_TRUE(reported) << "a run that produced no file was reported as a clean flush";
    EXPECT_NE(error.find("write_failed"), std::string::npos) << error;
    EXPECT_EQ(read_file(taken), "not this run's artifact") << "an existing artifact was overwritten";
    EXPECT_EQ(fx.collector.session_stats_for_test().open_slots, 0u);
}

// The tombstone ring holds 16 epochs and exists to classify late buffers. A
// failure from further back than that must still be reported, which it is only
// because the session's error summary is separate and never evicted.
TEST(ChipSwimlaneSessionTest, FailureIsRememberedPastTheTombstoneRing) {
    SessionFixture fx("memory", /*cores=*/1, /*threads=*/1);
    const int cores[] = {0};
    constexpr uint64_t kFailing = 9501;

    fs::path taken;
    for (const auto &entry : fs::directory_iterator(fx.dir.path())) {
        if (entry.is_directory()) taken = entry.path() / ("records_e" + std::to_string(kFailing) + ".json");
    }
    ASSERT_FALSE(taken.empty());
    {
        std::ofstream squatter(taken);
        squatter << "occupied";
    }

    fx.begin(kFailing);
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
    EXPECT_FALSE(fx.collector.session_flush(4000, &error)) << "the old failure was forgotten";
    EXPECT_NE(error.find("write_failed"), std::string::npos) << error;
    EXPECT_NE(error.find(std::to_string(kFailing)), std::string::npos) << error;
}

// A buffer whose epoch the session has already sealed is counted and dropped.
// It must never append to a bucket the session has moved, and never reopen one.
TEST(ChipSwimlaneSessionTest, LateBufferForASealedEpochIsCountedNotAppended) {
    SessionFixture fx("late", /*cores=*/1, /*threads=*/1);
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

    const auto stats = fx.collector.session_stats_for_test();
    EXPECT_EQ(stats.late_after_seal, 1u);
    EXPECT_EQ(stats.unknown_epoch, 0u);
    EXPECT_EQ(published_files(fx.dir.path()).size(), 1u) << "a late buffer produced a second artifact";
}

// An epoch nobody opened is counted separately from one that was sealed: the
// two are different facts and the tombstone ring is what tells them apart.
TEST(ChipSwimlaneSessionTest, BufferForAnUnknownEpochIsCountedSeparately) {
    SessionFixture fx("unknown", /*cores=*/1, /*threads=*/1);
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

    const auto stats = fx.collector.session_stats_for_test();
    EXPECT_EQ(stats.unknown_epoch, 1u);
    EXPECT_EQ(stats.late_after_seal, 0u);
}

// The session reserves its directory by exclusive creation, so a second
// session over the same output root cannot land on the first one's files.
TEST(ChipSwimlaneSessionTest, TwoSessionsOverOneRootReserveDistinctDirectories) {
    SessionDir root("reserve");
    ChipSwimlaneCollector first;
    ChipSwimlaneCollector second;
    ChipSwimlaneCollector::SessionOptions options;
    options.enabled = true;
    ASSERT_EQ(first.initialize(1, 1, 0, ChipSwimlaneLevel::TASK_TIMING, session_alloc, nullptr, session_free), 0);
    ASSERT_EQ(second.initialize(1, 1, 0, ChipSwimlaneLevel::TASK_TIMING, session_alloc, nullptr, session_free), 0);
    EXPECT_TRUE(first.session_open(options, root.str()));
    EXPECT_TRUE(second.session_open(options, root.str()));

    int dirs = 0;
    for (const auto &entry : fs::directory_iterator(root.path())) {
        if (entry.is_directory()) dirs++;
    }
    EXPECT_EQ(dirs, 2) << "the second session reused the first one's directory";

    first.session_close();
    second.session_close();
    first.finalize(nullptr, session_free);
    second.finalize(nullptr, session_free);
}

// A budget that cannot hold the fixed overhead plus a working set is refused at
// open rather than discovered as emptiness later.
TEST(ChipSwimlaneSessionTest, AnUnworkableBudgetIsRefusedAtOpen) {
    SessionDir root("budget");
    ChipSwimlaneCollector collector;
    ASSERT_EQ(collector.initialize(1, 1, 0, ChipSwimlaneLevel::TASK_TIMING, session_alloc, nullptr, session_free), 0);
    ChipSwimlaneCollector::SessionOptions options;
    options.enabled = true;
    options.budget_bytes = 1024;  // below the minimum working set by construction
    EXPECT_FALSE(collector.session_open(options, root.str()));
    EXPECT_FALSE(collector.session_active());
    // Refusing to open leaves the collector usable on the legacy path.
    collector.begin_run(root.str(), ChipSwimlaneLevel::TASK_TIMING);
    collector.finalize(nullptr, session_free);
}

// A session's per-kind paired cap is twice the bytes `init()` seeded, so the
// seed has to mean one initialization's allocations and nothing else. Two
// lifecycle events would otherwise inflate it: an aborted init whose rollback
// frees everything it charged, and a session's own pool growth. Both are
// reachable — the rollback guard runs on any late init failure, and `finalize`
// permits re-initialization on the same collector.
TEST(ChipSwimlaneSessionTest, PairedSeedIsPerInitAndUnmovedByGrowth) {
    SessionDir root("pairedseed");
    constexpr int kKind = static_cast<int>(ProfBufferType::AICPU_TASK);

    // What one clean initialization seeds, and what a finalize leaves behind.
    size_t clean_seed = 0;
    {
        ChipSwimlaneCollector control;
        ASSERT_EQ(control.initialize(1, 1, 0, ChipSwimlaneLevel::TASK_TIMING, session_alloc, nullptr, session_free), 0);
        clean_seed = control.manager().paired_initial(kKind);
        control.finalize(nullptr, session_free);
        EXPECT_EQ(control.manager().paired_initial(kKind), 0u) << "finalize left a seed behind for a re-init to add to";
        EXPECT_EQ(control.manager().paired_charged(kKind), 0u);
    }
    ASSERT_GT(clean_seed, 0u) << "this kind seeds nothing, so the case cannot discriminate";

    ChipSwimlaneCollector collector;
    // Enough allocations to charge some of this kind's buffers, not enough to
    // finish: the shm block comes first and is charged to no kind.
    g_alloc_budget = 3;
    EXPECT_NE(
        collector.initialize(1, 1, 0, ChipSwimlaneLevel::TASK_TIMING, session_alloc_limited, nullptr, session_free), 0
    );
    g_alloc_budget = -1;
    ASSERT_EQ(collector.initialize(1, 1, 0, ChipSwimlaneLevel::TASK_TIMING, session_alloc, nullptr, session_free), 0);
    EXPECT_EQ(collector.manager().paired_initial(kKind), clean_seed)
        << "the aborted init's charges outlived the rollback that freed them";

    // Growth moves the live total and leaves the seed alone, so a session
    // opened after it derives the same cap as one opened before.
    const size_t live_before = collector.manager().paired_charged(kKind);
    const size_t published =
        collector.manager().allocate_recycled_batch(kKind, sizeof(ChipSwimlaneAicpuTaskBuffer), 4, /*shard_index=*/0);
    ASSERT_GT(published, 0u);
    EXPECT_GT(collector.manager().paired_charged(kKind), live_before) << "growth was not charged";
    EXPECT_EQ(collector.manager().paired_initial(kKind), clean_seed) << "growth redefined the seed a cap comes from";

    collector.finalize(nullptr, session_free);
}

// A session's device-side bound is a byte figure, so an emptied mapping table
// is not enough to reconcile it: the release surface reports a status per
// pointer and carries no size, and a release that did not succeed leaves bytes
// the pool may still hold. Occupancy therefore stays charged, and the session
// refuses rather than admitting against a figure it cannot state.
TEST(ChipSwimlaneSessionTest, UnprovedReleaseKeepsOccupancyAndRefusesAnotherSession) {
    SessionDir root("unproved");
    constexpr int kKind = static_cast<int>(ProfBufferType::AICPU_TASK);

    ChipSwimlaneCollector collector;
    ASSERT_EQ(collector.initialize(1, 1, 0, ChipSwimlaneLevel::TASK_TIMING, session_alloc, nullptr, session_free), 0);
    const size_t seeded = collector.manager().paired_initial(kKind);
    ASSERT_GT(seeded, 0u);
    EXPECT_FALSE(collector.manager().release_unproven());

    // A session opens against a proved-clean pool, and closing it leaves the
    // seed a reopened session's cap comes from untouched.
    ChipSwimlaneCollector::SessionOptions options;
    options.enabled = true;
    ASSERT_TRUE(collector.session_open(options, root.str()));
    collector.session_close();
    ASSERT_TRUE(collector.session_open(options, root.str()));
    collector.session_close();
    EXPECT_EQ(collector.manager().paired_initial(kKind), seeded) << "a session reopen moved the seed";

    // Finalize with a free that reports failure. The mapping table empties
    // either way, which is exactly why emptiness cannot be the proof.
    collector.finalize(nullptr, session_free_failing);
    EXPECT_TRUE(collector.manager().release_unproven());
    EXPECT_GT(collector.manager().paired_charged(kKind), 0u) << "occupancy was erased without proof of release";
    EXPECT_GT(collector.manager().paired_initial(kKind), 0u);

    // And no further session is admitted against that occupancy.
    ASSERT_EQ(collector.initialize(1, 1, 0, ChipSwimlaneLevel::TASK_TIMING, session_alloc, nullptr, session_free), 0);
    EXPECT_FALSE(collector.session_open(options, root.str()))
        << "a session was admitted while an unproved release was outstanding";
    EXPECT_FALSE(collector.session_active());
    // The legacy path stays usable, as it does for every other refusal.
    collector.begin_run(root.str(), ChipSwimlaneLevel::TASK_TIMING);
    collector.finalize(nullptr, session_free);
}

// The same admission rule has to hold for a buffer that never reached the
// manager's mapping table. `alloc_paired_buffer` cleans up its own device
// pointer when registration or the initial copy fails, and returns before
// registering it — so the rollback guard never sees that pointer, and this is
// the only place the outcome of its release can be recorded. A cleanup that did
// not report success leaves memory held just the same.
TEST(ChipSwimlaneSessionTest, UnregisteredInitCleanupFailureAlsoRefusesAnotherSession) {
    SessionDir root("initcleanup");
    ChipSwimlaneCollector collector;

    // Registration refuses on the very first paired allocation, and the free
    // that follows reports failure while still reclaiming the memory.
    EXPECT_NE(
        collector.initialize(
            1, 1, 0, ChipSwimlaneLevel::TASK_TIMING, session_alloc, session_register_failing, session_free_failing
        ),
        0
    );
    EXPECT_TRUE(collector.manager().release_unproven())
        << "a cleanup the rollback guard cannot see reported nothing at all";

    // A later initialization succeeds, and the session is still refused: the
    // unproved occupancy belongs to the pool, not to the failed attempt.
    ASSERT_EQ(collector.initialize(1, 1, 0, ChipSwimlaneLevel::TASK_TIMING, session_alloc, nullptr, session_free), 0);
    ChipSwimlaneCollector::SessionOptions options;
    options.enabled = true;
    EXPECT_FALSE(collector.session_open(options, root.str()))
        << "a session was admitted after an unregistered buffer's cleanup failed";
    EXPECT_FALSE(collector.session_active());
    collector.begin_run(root.str(), ChipSwimlaneLevel::TASK_TIMING);
    collector.finalize(nullptr, session_free);
}

// With the session off nothing about the existing path moves: the legacy
// artifact name, the legacy location, and no `collection` object.
TEST(ChipSwimlaneSessionTest, SessionOffKeepsTheLegacyArtifactExactly) {
    SessionDir root("legacy");
    ChipSwimlaneCollector collector;
    ASSERT_EQ(collector.initialize(1, 1, 0, ChipSwimlaneLevel::TASK_TIMING, session_alloc, nullptr, session_free), 0);
    EXPECT_FALSE(collector.session_active());

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

    collector.start(session_thread_factory);
    collector.quiesce();
    collector.stop();
    collector.reconcile_counters();
    EXPECT_EQ(collector.export_swimlane_json(), 0);

    const fs::path legacy = root.path() / "chip_swimlane_records.json";
    ASSERT_TRUE(fs::exists(legacy)) << "the legacy artifact name or location changed";
    EXPECT_EQ(read_file(legacy).find("\"collection\""), std::string::npos)
        << "the default path emitted the session's metadata object";
    EXPECT_TRUE(published_files(root.path()).empty()) << "the default path created a session directory";

    collector.finalize(nullptr, session_free);
}

// Six runs over two slots, so every slot is armed, retired and armed again
// three times while the drain owners and collector shards keep running. A cut
// slot handed back before its readers had left it would be reinitialized under
// one of them, and the run whose arrays were overwritten would have its records
// land in the wrong artifact or in none. Each run's own count in its own file is
// what rules that out.
TEST(ChipSwimlaneSessionTest, SlotReuseAcrossRunsKeepsEachRunsRecordsInItsOwnFile) {
    SessionFixture fx("reuse", /*cores=*/1, /*threads=*/1);
    const int cores[] = {0};
    constexpr uint64_t kBase = 9901;
    constexpr int kRuns = 6;

    for (int i = 0; i < kRuns; i++) {
        const uint64_t epoch = kBase + static_cast<uint64_t>(i);
        fx.begin(epoch);
        fx.dispatch(/*core_id=*/0, i + 1);
        fx.close(epoch, cores, 1);
    }
    ASSERT_TRUE(fx.wait_for_files(kRuns)) << "the session did not publish every run";

    std::string error;
    EXPECT_TRUE(fx.collector.session_flush(8000, &error)) << error;
    for (int i = 0; i < kRuns; i++) {
        const uint64_t epoch = kBase + static_cast<uint64_t>(i);
        const fs::path expected = fx.dir.path() / "swimlane-0" / ("records_e" + std::to_string(epoch) + ".json");
        ASSERT_TRUE(fs::exists(expected)) << expected.string();
        EXPECT_EQ(rows_for_epoch(read_file(expected), epoch), static_cast<size_t>(i + 1))
            << "epoch " << epoch << " did not carry exactly its own records";
    }
    EXPECT_EQ(fx.collector.session_stats_for_test().open_slots, 0u);
}

// The host budget is a bound on retained records, not a number in a log line.
// Past it the epoch stops retaining, keeps its receipts, and publishes an
// artifact that says its content is incomplete — and the charge never goes
// beyond the limit the session was opened with.
TEST(ChipSwimlaneSessionTest, BudgetExhaustionStopsRetentionAndPublishesAPartial) {
    // Fixed overhead plus a little over the minimum working set.
    constexpr size_t kBudget = 18ull * 1024 * 1024;
    SessionFixture fx("exhaust", /*cores=*/1, /*threads=*/1, kBudget);
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
        if (fx.collector.session_stats_for_test().budget_refusals > 0) break;
    }
    const auto stats = fx.collector.session_stats_for_test();
    ASSERT_GT(stats.budget_refusals, 0u) << "fed " << fed << " buffers of " << per_buffer << " B without a refusal";
    EXPECT_LE(stats.host_charged, kBudget) << "the charge went past the budget it was opened with";
    EXPECT_FALSE(stats.fatal);

    fx.close(kEpoch, cores, 1);
    ASSERT_TRUE(fx.wait_for_files(1));
    const std::string body = read_file(published_files(fx.dir.path())[0]);
    EXPECT_NE(body.find("partial_safe"), std::string::npos) << body.substr(0, 600);
}

// A session whose control handshake never completes is fatal before any epoch
// has a verdict of its own. Flush has to fail on the fatal itself: the
// per-epoch rows are empty, and reading empty as clean would report a
// successful flush over a session that wrote nothing.
TEST(ChipSwimlaneSessionTest, FatalWithNoEpochVerdictStillFailsFlush) {
    ReaderlessSession rs("fatalflush");

    // No collector shard is polling, so the epoch table can never be
    // acknowledged and admission ends in the session's fatal.
    EXPECT_FALSE(rs.collector.session_run_begin(9001, rs.dir.str(), ChipSwimlaneLevel::TASK_TIMING));
    ASSERT_TRUE(rs.collector.session_stats_for_test().fatal);
    EXPECT_EQ(rs.collector.session_stats_for_test().published, 0u);

    std::string error;
    EXPECT_FALSE(rs.collector.session_flush(200, &error)) << "a fatal session reported a successful flush";
    EXPECT_NE(error.find("fatal"), std::string::npos) << error;
    EXPECT_TRUE(published_files(rs.dir.path()).empty());

    rs.collector.session_close();
    rs.collector.stop();
    rs.collector.finalize(nullptr, session_free);
}

// A close cannot prove a stalled reader has let go, so it must not free that
// epoch's storage. The publisher's join says nothing about the collector
// shards, and the production close path runs before they are joined at all —
// so the release waits for `finalize()`, which joins them first.
TEST(ChipSwimlaneSessionTest, CloseDefersQuarantinedStorageUntilReadersAreJoined) {
    ReaderlessSession rs("deferred");
    constexpr uint64_t kEpoch = 9002;

    EXPECT_FALSE(rs.collector.session_run_begin(kEpoch, rs.dir.str(), ChipSwimlaneLevel::TASK_TIMING));
    // The epoch is open and now has a target, so the publisher tries to seal it
    // and cannot prove the reference released. `session_close` waits for that
    // attempt to reach its verdict before it returns.
    rs.collector.session_run_close(kEpoch, /*bank_index=*/0, /*device_execution_complete=*/true);
    rs.collector.session_close();

    const auto after_close = rs.collector.session_stats_for_test();
    EXPECT_EQ(after_close.open_slots, 1u) << "close freed storage a stalled reader may still hold";
    EXPECT_TRUE(after_close.release_deferred);
    EXPECT_TRUE(published_files(rs.dir.path()).empty());

    // stop() joins every reader; only then may the storage go back.
    rs.collector.stop();
    rs.collector.finalize(nullptr, session_free);
    const auto after_join = rs.collector.session_stats_for_test();
    EXPECT_EQ(after_join.open_slots, 0u) << "the deferred release did not happen after the readers were joined";
    EXPECT_FALSE(after_join.release_deferred);
}

// `session_run_begin` blocks on the epoch table being acknowledged by every
// collector shard, and a shard with an empty ready ring is asleep in a 100 ms
// cv tick. Notifying that ring is not enough on its own: the notification
// advances no ready shard's state epoch, so a predicate that knows nothing
// about control re-tests false and the shard sleeps out the rest of its tick.
// With the control condition in the predicate, each handshake costs a wakeup
// instead of a timer. Timed with the ring deliberately empty and each epoch
// published before the next admission, so the interval measured contains the
// handshake and not a wait for capacity.
TEST(ChipSwimlaneSessionTest, ControlHandshakeWakesAnIdleCollectorWithoutItsRingTick) {
    SessionFixture fx("ctrlwake", /*cores=*/1, /*threads=*/1);
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
        ASSERT_TRUE(fx.collector.session_run_begin(epoch, fx.dir.str(), ChipSwimlaneLevel::TASK_TIMING));
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
TEST(ChipSwimlaneSessionTest, OversizedRunMetadataIsRefusedBeforeItIsCopied) {
    constexpr size_t kBudget = 18ull * 1024 * 1024;
    SessionFixture fx("metabudget", /*cores=*/1, /*threads=*/1, kBudget);
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

    const auto stats = fx.collector.session_stats_for_test();
    EXPECT_LE(stats.host_charged, kBudget) << "the refused charge was taken anyway";
    EXPECT_GT(stats.budget_refusals, 0u);
    // The epoch's own records survived the refusal: only the metadata was
    // declined, and nothing was freed ahead of its reference proof.
    EXPECT_EQ(rows_for_epoch(body, kEpoch), 3u) << "the refusal took this epoch's records with it";
}

// Two session runs, each publishing host phase records of its own. The epoch's
// metadata snapshot copies whatever the collector holds when the run closes,
// and the collector holds one copy of it across every run it serves — so the
// artifact describes its own run only when the publication precedes the
// snapshot. Driven through the boundary helper both runner bases call, which
// is where that order lives. The two runs publish different counts, which is
// what the metadata reports at every level.
TEST(ChipSwimlaneSessionTest, HostPhaseRecordsReachTheEpochThatProducedThem) {
    SessionFixture fx("host-phase-own-epoch", /*cores=*/1, /*threads=*/1);
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
TEST(ChipSwimlaneSessionTest, ASessionRunWithoutHostPhaseRecordsInheritsNone) {
    SessionFixture fx("host-phase-empty-successor", /*cores=*/1, /*threads=*/1);
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

// The close is what hands an epoch to the session thread and releases its
// slot, so a publication that throws must not skip it: two slots exist, and an
// epoch left open holds one for the session's whole life. The failure still
// reaches the caller, and the epoch reports no host capture it did not make.
TEST(ChipSwimlaneSessionTest, AFailedHostPublicationStillClosesItsEpoch) {
    SessionFixture fx("host-phase-failed-publication", /*cores=*/1, /*threads=*/1);
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
// never told a half-written capture is the whole run. The session carries on,
// and the next epoch settles complete.
TEST(ChipSwimlaneSessionTest, APartialHostPublicationMarksItsEpochIncomplete) {
    SessionFixture fx("host-phase-partial-publication", /*cores=*/1, /*threads=*/1);
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

    ASSERT_TRUE(fx.wait_for_files(2)) << "the partial epoch stopped the session from making progress";
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
TEST(ChipSwimlaneSessionTest, AnUnnameableHostPublicationFailureStillClosesItsEpoch) {
    SessionFixture fx("host-phase-unnameable-failure", /*cores=*/1, /*threads=*/1);
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

    ASSERT_TRUE(fx.wait_for_files(2)) << "an unnameable failure cost the epoch its close or the session its slot";
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
    // The session is still usable: an epoch-scoped failure is not a
    // session-level one, so nothing here may have raised the sticky fatal.
    EXPECT_FALSE(fx.collector.session_stats_for_test().fatal)
        << "an epoch's publication failure was escalated to a session fatal";
}
