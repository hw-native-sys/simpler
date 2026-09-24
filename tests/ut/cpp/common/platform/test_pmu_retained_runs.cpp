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
 * PMU cross-run retention: a run's host-side receipt and CSV publication
 * continue in the background while the next run owns the device.
 *
 * These drive the real thing — the collector's mgmt and poll threads, the real
 * `pmu_aicpu_init` / `pmu_aicpu_flush_buffers` producer, the per-queue
 * transport cut, the reference-release handshake and the writer that merges and
 * renames — rather than a model of it. Host and device share process memory
 * here, so nothing below is evidence about device cache visibility.
 *
 * **Record contents are a fixture, for the reason test_pmu_run_identity.cpp
 * gives:** PMU records are written through MMIO on a2a3 and validated out of an
 * AICore staging ring on a5, neither of which belongs in a host unit test. What
 * is production here is everything that decides attribution, finality and the
 * flush conclusion.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include "aicpu/device_run_result_base_aicpu.h"
#include "aicpu/platform_regs.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "common/memory_barrier.h"
#include "common/pmu_profiling.h"
#include "host/pmu_collector.h"

// The platform register accessors, for the reason test_pmu_run_identity.cpp
// gives: the two architectures define them in translation units that reach
// further into the runtime than this test needs, and a zero register window
// makes `pmu_aicpu_init` skip the MMIO programming entirely. Reaching any of
// them would mean the test wandered into the counters, so they say so.
extern "C" {
namespace {
uint64_t g_pmu_retained_reg_addrs = 0;
}

void set_platform_pmu_reg_addrs(uint64_t pmu_regs) { g_pmu_retained_reg_addrs = pmu_regs; }
uint64_t get_platform_pmu_reg_addrs() { return g_pmu_retained_reg_addrs; }
uint64_t get_platform_regs() { return 0; }
}

uint64_t read_reg(uint64_t, RegId) {
    ADD_FAILURE() << "the test reached MMIO; PMU registers are out of scope here";
    return 0;
}

uint32_t reg_load_acquire(const volatile uint32_t *) {
    ADD_FAILURE() << "the test reached MMIO; PMU registers are out of scope here";
    return 0;
}

void reg_store_release(volatile uint32_t *, uint32_t) {
    ADD_FAILURE() << "the test reached MMIO; PMU registers are out of scope here";
}

#if defined(SIMPLER_TEST_PMU_ARCH_A5)
void write_reg(uint64_t, RegId, uint64_t) {
    ADD_FAILURE() << "the test reached MMIO; PMU registers are out of scope here";
}
#endif

namespace fs = std::filesystem;

namespace {

void *retained_alloc(size_t size) { return std::calloc(1, size); }

int retained_free(void *ptr) {
    std::free(ptr);
    return 0;
}

std::thread retained_thread_factory(std::function<void()> fn) { return std::thread(std::move(fn)); }

/** A private output root per case, removed at teardown. */
class OutputRoot {
public:
    explicit OutputRoot(const char *name) {
        path_ = fs::temp_directory_path() /
                ("simpler-pmu-retained-" + std::string(name) + "-" + std::to_string(::getpid()));
        std::error_code ec;
        fs::remove_all(path_, ec);
        fs::create_directories(path_, ec);
    }
    ~OutputRoot() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    fs::path path() const { return path_; }
    /** One destination per run, so a case can also reuse one deliberately. */
    fs::path csv(const char *leaf) const {
        fs::path dir = path_ / leaf;
        std::error_code ec;
        fs::create_directories(dir, ec);
        return dir / "pmu.csv";
    }

private:
    fs::path path_;
};

std::string read_file(const fs::path &p) {
    std::ifstream in(p);
    if (!in.is_open()) return {};
    return std::string{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

/** Every data row's value in one named column, in file order. */
std::vector<std::string> csv_column(const std::string &content, const std::string &want) {
    std::vector<std::string> out;
    std::istringstream lines(content);
    std::string header;
    if (!std::getline(lines, header)) return out;
    std::vector<std::string> cols;
    std::istringstream head_cols(header);
    for (std::string col; std::getline(head_cols, col, ',');)
        cols.push_back(col);
    size_t idx = cols.size();
    for (size_t i = 0; i < cols.size(); i++) {
        if (cols[i] == want) idx = i;
    }
    if (idx == cols.size()) return out;
    for (std::string row; std::getline(lines, row);) {
        if (row.empty()) continue;
        std::vector<std::string> fields;
        std::istringstream row_cols(row);
        for (std::string f; std::getline(row_cols, f, ',');)
            fields.push_back(f);
        if (idx < fields.size()) out.push_back(fields[idx]);
    }
    return out;
}

/** Temp files this contract preserves as a failed epoch's evidence. */
std::vector<fs::path> evidence_files(const fs::path &dir) {
    std::vector<fs::path> found;
    std::error_code ec;
    for (const auto &entry : fs::directory_iterator(dir, ec)) {
        const std::string name = entry.path().filename().string();
        if (name.find(".e") != std::string::npos && entry.path().extension() == ".tmp") {
            found.push_back(entry.path());
        }
    }
    return found;
}

/**
 * One collector with its threads running, retaining runs.
 *
 * The reader shards have to be running before the first admission: admitting a
 * run waits for every shard to acknowledge the new epoch table, and a shard
 * that has not been spawned cannot acknowledge anything.
 */
struct RetainedPmuFixture {
    PmuCollector collector;
    OutputRoot root;
    int num_cores;
    int num_threads;
    void *shm = nullptr;
    PmuDataHeader *header = nullptr;

    RetainedPmuFixture(const char *name, int cores = 1, int threads = 1) :
        root(name),
        num_cores(cores),
        num_threads(threads) {
        collector.configure_retained_runs(true);
        EXPECT_EQ(collector.init(cores, threads, retained_alloc, nullptr, retained_free, /*device_id=*/0), 0);
        shm = collector.get_pmu_shm_device_ptr();
        EXPECT_NE(shm, nullptr);
        header = get_pmu_header(shm);
        collector.start(retained_thread_factory);
        set_pmu_enabled(true);
        set_platform_pmu_base(reinterpret_cast<uint64_t>(shm));
        set_platform_pmu_reg_addrs(0);
    }

    ~RetainedPmuFixture() {
        set_pmu_enabled(false);
        set_platform_pmu_base(0);
        set_platform_run_result(0, 0);
        // Ordered teardown, each step idempotent: publish what is left, join the
        // writer and the readers, then free.
        collector.finish_retained_runs();
        collector.stop();
        collector.finalize(nullptr, retained_free);
    }

    PmuBufferState *state(int core) { return get_pmu_buffer_state(shm, core); }

    /** Admit one run. Returns what the collector answered. */
    bool begin(uint64_t epoch, const fs::path &csv, PmuEventType event_type = PmuEventType::PIPE_UTILIZATION) {
        return collector.run_begin(epoch, csv.string(), event_type);
    }

    /** The device side of one run: acquire a buffer, fill it, publish it. */
    void produce(uint64_t epoch, int records, int core = 0, int thread = 0) {
        set_platform_run_result(/*region_base=*/0, epoch);
        std::vector<uint32_t> core_ids;
        for (int c = 0; c < num_cores; c++)
            core_ids.push_back(static_cast<uint32_t>(c));
        pmu_aicpu_init(core_ids.data(), num_cores);
        if (records > 0) fill_records(state(core)->current_buf_ptr, records, core);
        const int cores[] = {core};
        pmu_aicpu_flush_buffers(thread, cores, /*core_num=*/1);
    }

    /**
     * The fixture half. `run_epoch` is deliberately untouched — the stamp is
     * production, written by `pmu_aicpu_init`.
     */
    void fill_records(uint64_t buf_ptr, int n, int core) {
        ASSERT_NE(buf_ptr, 0u);
        auto *buf = reinterpret_cast<PmuBuffer *>(buf_ptr);
        for (int i = 0; i < n; i++) {
            buf->records[i].task_id = 0x900 + static_cast<uint64_t>(i);
            buf->records[i].func_id = static_cast<uint32_t>(i);
            buf->records[i].pmu_total_cycles = 4000 + static_cast<uint64_t>(i);
        }
        buf->count = static_cast<uint32_t>(n);
        state(core)->total_record_count += static_cast<uint32_t>(n);
        wmb();
    }

    void close(uint64_t epoch, bool device_execution_complete = true) {
        collector.run_close(epoch, device_execution_complete);
    }

    /** What the device's own producer accounting stands at, for a case that
     *  needs to know its own fixture landed before the close reads it. */
    uint64_t device_total(int core = 0) { return state(core)->total_record_count; }

    /** One flush, with the budget a caller's own deadline would hand over. */
    bool flush(std::string *error) {
        return collector.flush_retained_runs(simpler::dfx::pmu::kCutAckBudgetMs * 8, error);
    }

    simpler::dfx::pmu::RetainedRunStats stats() const { return collector.retained_run_stats_for_test(); }
};

// ---------------------------------------------------------------------------
// Continuous runs, and each run's own configuration
// ---------------------------------------------------------------------------

/**
 * Three runs back to back, each with its own destination and event type.
 *
 * Publication is asynchronous, so the flush in the middle is what a caller that
 * wants a third run admitted has to do: two unpublished runs is the capacity,
 * and this is the shape that proves the first two both published while the
 * collector kept serving.
 */
TEST(PmuRetainedRuns, ThreeRunsPublishTheirOwnRowsAndColumns) {
    RetainedPmuFixture fx("three-runs");
    const fs::path a = fx.root.csv("run-a");
    const fs::path b = fx.root.csv("run-b");
    const fs::path c = fx.root.csv("run-c");

    ASSERT_TRUE(fx.begin(101, a));
    fx.produce(101, /*records=*/2);
    ASSERT_EQ(fx.device_total(), 2u) << "the fixture's own producer accounting did not land";
    fx.close(101);

    // Admitted while run 101 may still be publishing: two epochs is what the
    // overlap is made of.
    ASSERT_TRUE(fx.begin(102, b, PmuEventType::PIPE_UTILIZATION));
    fx.produce(102, /*records=*/3);
    fx.close(102);

    std::string error;
    ASSERT_TRUE(fx.flush(&error)) << error;

    ASSERT_TRUE(fx.begin(103, c));
    fx.produce(103, /*records=*/1);
    fx.close(103);
    error.clear();
    EXPECT_TRUE(fx.flush(&error)) << error;

    // Each destination holds exactly its own run's rows, under its own header.
    const std::vector<std::string> ea = csv_column(read_file(a), "run_epoch");
    const std::vector<std::string> eb = csv_column(read_file(b), "run_epoch");
    const std::vector<std::string> ec = csv_column(read_file(c), "run_epoch");
    ASSERT_EQ(ea.size(), 2u) << read_file(a);
    ASSERT_EQ(eb.size(), 3u) << read_file(b);
    ASSERT_EQ(ec.size(), 1u) << read_file(c);
    for (const std::string &v : ea)
        EXPECT_EQ(v, "101");
    for (const std::string &v : eb)
        EXPECT_EQ(v, "102");
    for (const std::string &v : ec)
        EXPECT_EQ(v, "103");

    // Published by rename, so no shard or publication temp is left behind.
    EXPECT_TRUE(evidence_files(a.parent_path()).empty());
    EXPECT_TRUE(evidence_files(b.parent_path()).empty());

    const auto st = fx.stats();
    EXPECT_FALSE(st.fatal);
    EXPECT_FALSE(st.errors.has_error);
    EXPECT_EQ(st.errors.published, 3u);
}

/**
 * A run that produced no records leaves no file — and that is a success only
 * because its counters and its cut both proved the zero.
 */
TEST(PmuRetainedRuns, AProvedEmptyRunLeavesNoFileAndStillSucceeds) {
    RetainedPmuFixture fx("empty-run");
    const fs::path a = fx.root.csv("run-a");

    ASSERT_TRUE(fx.begin(201, a));
    fx.produce(201, /*records=*/0);
    fx.close(201);

    std::string error;
    EXPECT_TRUE(fx.flush(&error)) << error;
    EXPECT_FALSE(fs::exists(a)) << "a zero-record run must not write a file";
    EXPECT_EQ(fx.stats().errors.published_empty, 1u);
    EXPECT_FALSE(fx.stats().errors.has_error);
}

/**
 * A run whose device execution never completed cannot be reported as complete,
 * even with every counter at zero: an unvisited ready queue is
 * indistinguishable from a clean one.
 */
TEST(PmuRetainedRuns, AnUnfinishedRunIsNotAnEmptyRun) {
    RetainedPmuFixture fx("unfinished-run");
    const fs::path a = fx.root.csv("run-a");

    ASSERT_TRUE(fx.begin(301, a));
    fx.produce(301, /*records=*/0);
    fx.close(301, /*device_execution_complete=*/false);

    std::string error;
    EXPECT_FALSE(fx.flush(&error)) << "an unproved run must not report success";
    EXPECT_NE(error.find("cut_unproved"), std::string::npos) << error;
    EXPECT_EQ(fx.stats().errors.cut_unproved, 1u);
}

// ---------------------------------------------------------------------------
// Admission refusals
// ---------------------------------------------------------------------------

/** Two unpublished runs is the capacity, and a third is refused before launch. */
TEST(PmuRetainedRuns, AThirdUnpublishedRunIsRefused) {
    RetainedPmuFixture fx("capacity");
    // Both epochs stay open: closing is what hands one to the writer, so a run
    // that is admitted and not closed holds its slot.
    ASSERT_TRUE(fx.begin(401, fx.root.csv("run-a")));
    ASSERT_TRUE(fx.begin(402, fx.root.csv("run-b")));
    EXPECT_EQ(fx.stats().open_epochs, 2u);

    EXPECT_FALSE(fx.begin(403, fx.root.csv("run-c"))) << "capacity must refuse rather than queue";
    // A refusal is not a failure of an earlier run: nothing is recorded against
    // one, and the two open epochs are untouched.
    EXPECT_EQ(fx.stats().open_epochs, 2u);
    EXPECT_FALSE(fx.stats().errors.has_error);

    fx.close(401);
    fx.close(402);
    std::string error;
    EXPECT_TRUE(fx.flush(&error)) << error;
}

/** A destination an open epoch owns cannot be handed to a second run. */
TEST(PmuRetainedRuns, ADestinationAnOpenRunOwnsIsRefused) {
    RetainedPmuFixture fx("destination-owned");
    const fs::path shared = fx.root.csv("shared");
    ASSERT_TRUE(fx.begin(501, shared));
    EXPECT_FALSE(fx.begin(502, shared)) << "two epochs cannot own one publication point";
    fx.close(501);
    std::string error;
    EXPECT_TRUE(fx.flush(&error)) << error;
}

/**
 * A run admitted and never launched gives its slot back, and owes no verdict:
 * it promised no file, so reporting one would make a rolled-back launch read as
 * a lost artifact.
 */
TEST(PmuRetainedRuns, AnUnlaunchedRunIsWithdrawnWithoutAVerdict) {
    RetainedPmuFixture fx("rollback");
    const fs::path a = fx.root.csv("run-a");
    ASSERT_TRUE(fx.begin(601, a));
    EXPECT_EQ(fx.stats().open_epochs, 1u);

    EXPECT_TRUE(fx.collector.abandon_run(601));
    EXPECT_EQ(fx.stats().open_epochs, 0u);
    EXPECT_FALSE(fx.stats().errors.has_error);
    EXPECT_FALSE(fs::exists(a));

    // The slot is genuinely back: a later run takes it and publishes.
    ASSERT_TRUE(fx.begin(602, a));
    fx.produce(602, /*records=*/1);
    fx.close(602);
    std::string error;
    EXPECT_TRUE(fx.flush(&error)) << error;
    EXPECT_EQ(csv_column(read_file(a), "run_epoch").size(), 1u);
}

/**
 * A destination still holding a failed epoch's temp files is refused, and the
 * evidence is never removed to make room.
 */
TEST(PmuRetainedRuns, PreservedFailureEvidenceRefusesTheNextRun) {
    RetainedPmuFixture fx("evidence");
    const fs::path a = fx.root.csv("run-a");
    // A previous epoch's preserved shard file, in the shape the writer leaves.
    const fs::path leftover = a.string() + ".e77.shard0.tmp";
    { std::ofstream(leftover) << "thread_id,core_id\n"; }

    EXPECT_FALSE(fx.begin(701, a)) << "evidence must be read before its destination is reused";
    EXPECT_TRUE(fs::exists(leftover)) << "the contract never deletes failure evidence";
}

// ---------------------------------------------------------------------------
// Partial and unattributable collection
// ---------------------------------------------------------------------------

/**
 * Records the device dropped are published — the rows that exist are real — but
 * the flush must not call that a success: a PMU CSV has no field in which to
 * say it is partial.
 */
TEST(PmuRetainedRuns, ADroppedRecordPublishesAndStillFailsTheFlush) {
    RetainedPmuFixture fx("dropped");
    const fs::path a = fx.root.csv("run-a");

    ASSERT_TRUE(fx.begin(801, a));
    fx.produce(801, /*records=*/2);
    // The device's own drop accounting, which its flush path bumps when a queue
    // has no room. Charged before the close snapshot reads it.
    fx.state(0)->dropped_record_count += 1;
    fx.state(0)->total_record_count += 1;
    wmb();
    fx.close(801);

    std::string error;
    EXPECT_FALSE(fx.flush(&error)) << "a short CSV must not report success";
    EXPECT_NE(error.find("published_short"), std::string::npos) << error;
    // The rows that did arrive are published all the same.
    EXPECT_EQ(csv_column(read_file(a), "run_epoch").size(), 2u) << read_file(a);
    EXPECT_EQ(fx.stats().errors.published_short, 1u);
}

/**
 * A buffer that arrives for an epoch whose verdict is already published cannot
 * be attributed to it, so it becomes a bounded collector-scoped error rather
 * than a rewritten conclusion or a retained per-run object.
 */
TEST(PmuRetainedRuns, ALateBufferForASealedRunIsABoundedCollectorError) {
    RetainedPmuFixture fx("late-buffer");
    const fs::path a = fx.root.csv("run-a");
    const fs::path b = fx.root.csv("run-b");

    ASSERT_TRUE(fx.begin(901, a));
    fx.produce(901, /*records=*/1);
    fx.close(901);
    std::string error;
    ASSERT_TRUE(fx.flush(&error)) << error;
    const std::string published = read_file(a);
    ASSERT_EQ(csv_column(published, "run_epoch").size(), 1u);

    // A second run, and a buffer still carrying the sealed run's identity.
    ASSERT_TRUE(fx.begin(902, b));
    set_platform_run_result(/*region_base=*/0, 902);
    const uint32_t core_ids[] = {0};
    pmu_aicpu_init(core_ids, /*num_cores=*/1);
    auto *buf = reinterpret_cast<PmuBuffer *>(fx.state(0)->current_buf_ptr);
    ASSERT_NE(buf, nullptr);
    buf->records[0].task_id = 0x1;
    buf->count = 1;
    buf->run_epoch = 901;  // the sealed run's identity, on a live buffer
    wmb();
    const int cores[] = {0};
    pmu_aicpu_flush_buffers(/*thread_idx=*/0, cores, /*core_num=*/1);
    fx.close(902);

    error.clear();
    EXPECT_FALSE(fx.flush(&error)) << "an unplaceable record must reach the caller";
    EXPECT_NE(error.find("unknown_epoch_records"), std::string::npos) << error;
    EXPECT_GE(fx.stats().errors.unknown_epoch_buffers, 1u);
    // The already-published conclusion is not rewritten, and its file stands.
    EXPECT_EQ(read_file(a), published);
    // And it is not a fatal: admission continues.
    EXPECT_FALSE(fx.stats().fatal);
}

// ---------------------------------------------------------------------------
// I/O failure, and the sticky record
// ---------------------------------------------------------------------------

/**
 * A publication that cannot be renamed keeps its temp files, leaves an earlier
 * successful CSV alone, and fails the flush.
 *
 * The destination's directory is replaced by a read-only one after the epoch is
 * admitted, so the merge's own open fails where a real ENOSPC would.
 */
TEST(PmuRetainedRuns, AWriteFailurePreservesEvidenceAndKeepsTheOldFile) {
    // Directory permissions are the failure injection here, and they do not
    // bind a process with root privileges — which many CI containers run as.
    // Under root the shard open would succeed and this case would assert the
    // opposite of what happened, so it reports no result rather than a wrong
    // one.
    if (::geteuid() == 0) GTEST_SKIP() << "directory permissions do not bind root; this case cannot inject its failure";
    RetainedPmuFixture fx("write-failure");
    const fs::path a = fx.root.csv("run-a");

    // An earlier successful run under the same destination.
    ASSERT_TRUE(fx.begin(1001, a));
    fx.produce(1001, /*records=*/1);
    fx.close(1001);
    std::string error;
    ASSERT_TRUE(fx.flush(&error)) << error;
    const std::string first = read_file(a);
    ASSERT_FALSE(first.empty());

    // A second run whose shard files cannot be written: the directory is made
    // unwritable once the epoch has been admitted with its paths frozen.
    ASSERT_TRUE(fx.begin(1002, a));
    fs::permissions(a.parent_path(), fs::perms::owner_read | fs::perms::owner_exec);
    fx.produce(1002, /*records=*/1);
    fx.close(1002);
    error.clear();
    const bool flushed = fx.flush(&error);
    fs::permissions(a.parent_path(), fs::perms::owner_read | fs::perms::owner_write | fs::perms::owner_exec);

    EXPECT_FALSE(flushed) << "a failed write must reach the caller";
    // The earlier file is untouched: rename publication never truncates its
    // destination, unlike the single-run in-place merge.
    EXPECT_EQ(read_file(a), first);
    const auto st = fx.stats();
    EXPECT_GE(st.errors.write_failed + st.errors.published_short, 1u);
    EXPECT_TRUE(st.errors.has_error);
}

/**
 * A recorded failure survives the collector being finalized and re-initialized
 * — a shape change, or PMU being turned off and on — because there is no
 * acknowledgement that could clear it.
 */
TEST(PmuRetainedRuns, AFailureSurvivesACollectorRebuild) {
    OutputRoot root("rebuild");
    PmuCollector collector;
    collector.configure_retained_runs(true);
    ASSERT_EQ(collector.init(1, 1, retained_alloc, nullptr, retained_free, /*device_id=*/0), 0);
    void *shm = collector.get_pmu_shm_device_ptr();
    ASSERT_NE(shm, nullptr);
    collector.start(retained_thread_factory);
    set_pmu_enabled(true);
    set_platform_pmu_base(reinterpret_cast<uint64_t>(shm));
    set_platform_pmu_reg_addrs(0);

    const fs::path a = root.csv("run-a");
    ASSERT_TRUE(collector.run_begin(1101, a.string(), PmuEventType::PIPE_UTILIZATION));
    // A run the device never finished, which cannot be proved complete.
    collector.run_close(1101, /*device_execution_complete=*/false);
    std::string error;
    ASSERT_FALSE(collector.flush_retained_runs(simpler::dfx::pmu::kCutAckBudgetMs * 8, &error)) << "expected failure";
    ASSERT_TRUE(collector.retained_run_stats_for_test().errors.has_error);

    // The rebuild: exactly what a shape change does — finalize, then init and
    // start the same object again.
    collector.finish_retained_runs();
    collector.stop();
    collector.finalize(nullptr, retained_free);
    ASSERT_EQ(collector.init(1, 1, retained_alloc, nullptr, retained_free, /*device_id=*/0), 0);
    shm = collector.get_pmu_shm_device_ptr();
    ASSERT_NE(shm, nullptr);
    set_platform_pmu_base(reinterpret_cast<uint64_t>(shm));
    collector.start(retained_thread_factory);

    // A flush across the rebuild still reports the earlier failure, and it does
    // so before any new run is admitted.
    error.clear();
    EXPECT_FALSE(collector.flush_retained_runs(simpler::dfx::pmu::kCutAckBudgetMs, &error))
        << "a rebuild must not clear an unreported failure";
    EXPECT_NE(error.find("cut_unproved"), std::string::npos) << error;
    EXPECT_TRUE(collector.retained_run_stats_for_test().errors.has_error);

    // A later successful run does not clear it either.
    const fs::path b = root.csv("run-b");
    ASSERT_TRUE(collector.run_begin(1102, b.string(), PmuEventType::PIPE_UTILIZATION));
    collector.run_close(1102, /*device_execution_complete=*/true);
    error.clear();
    EXPECT_FALSE(collector.flush_retained_runs(simpler::dfx::pmu::kCutAckBudgetMs * 8, &error));

    set_pmu_enabled(false);
    set_platform_pmu_base(0);
    set_platform_run_result(0, 0);
    collector.finish_retained_runs();
    collector.stop();
    collector.finalize(nullptr, retained_free);
}

/**
 * Retention off is the default and changes nothing: the single-run window, its
 * in-place merge and its per-run reconcile are what a run gets.
 */
TEST(PmuRetainedRuns, RetentionOffKeepsTheSingleRunPath) {
    OutputRoot root("default-path");
    PmuCollector collector;
    // configure_retained_runs is never called, which is the default.
    EXPECT_FALSE(collector.retains_runs());
    ASSERT_EQ(collector.init(1, 1, retained_alloc, nullptr, retained_free, /*device_id=*/0), 0);
    void *shm = collector.get_pmu_shm_device_ptr();
    ASSERT_NE(shm, nullptr);
    const fs::path a = root.csv("run-a");
    collector.begin_run(a.string(), PmuEventType::PIPE_UTILIZATION);

    set_pmu_enabled(true);
    set_platform_pmu_base(reinterpret_cast<uint64_t>(shm));
    set_platform_pmu_reg_addrs(0);
    set_platform_run_result(/*region_base=*/0, 1201);
    const uint32_t core_ids[] = {0};
    pmu_aicpu_init(core_ids, /*num_cores=*/1);
    auto *state = get_pmu_buffer_state(shm, 0);
    auto *buf = reinterpret_cast<PmuBuffer *>(state->current_buf_ptr);
    ASSERT_NE(buf, nullptr);
    buf->records[0].task_id = 0x5;
    buf->count = 1;
    state->total_record_count += 1;
    wmb();
    const int cores[] = {0};
    pmu_aicpu_flush_buffers(/*thread_idx=*/0, cores, /*core_num=*/1);

    // The single-run receive path, driven directly: no retained epoch exists to
    // route into, and none is created.
    PmuDataHeader *header = get_pmu_header(shm);
    const uint32_t tail = header->queue_tails[0];
    ASSERT_GT(tail, 0u);
    for (uint32_t i = 0; i < tail; i++) {
        const PmuReadyQueueEntry &entry = header->queues[0][i];
        PmuReadyBufferInfo info{};
        info.core_index = entry.core_index;
        info.thread_index = 0;
        info.dev_buffer_ptr = reinterpret_cast<void *>(entry.buffer_ptr);
        info.host_buffer_ptr = reinterpret_cast<void *>(entry.buffer_ptr);
        info.buffer_seq = entry.buffer_seq;
        collector.on_buffer_collected(info, /*collector_shard=*/0);
    }
    collector.reconcile_counters();

    // Merged in place under the final name, with no epoch in any temp name.
    EXPECT_EQ(csv_column(read_file(a), "run_epoch").size(), 1u) << read_file(a);
    EXPECT_TRUE(evidence_files(a.parent_path()).empty());
    // And a flush has nothing to report, because nothing was retained.
    std::string error;
    EXPECT_TRUE(collector.flush_retained_runs(simpler::dfx::pmu::kCutAckBudgetMs, &error)) << error;

    set_pmu_enabled(false);
    set_platform_pmu_base(0);
    set_platform_run_result(0, 0);
    collector.finalize(nullptr, retained_free);
}

}  // namespace
