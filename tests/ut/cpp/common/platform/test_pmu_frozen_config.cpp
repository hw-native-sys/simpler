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
 * Two production boundaries the retained-run cases next door do not reach.
 *
 * 1. A run's event type and column set are frozen at its admission, so a
 *    successor's admission — which rewrites the collector's own event type, CSV
 *    header and the device header — cannot reach a predecessor's rows. The
 *    predecessor's whole publication is deliberately still pending when that
 *    happens: it has not been closed, so nothing of it has been sealed, merged
 *    or named. No sleeping is involved, so the overlap is a fact of the
 *    sequence rather than of timing.
 *
 * 2. The default, non-retained path opens its run window *before* the threads
 *    that serve it start, which is the order the runner has always used. A
 *    collector started first would drop the run's shard state on the way in.
 *
 * **Record contents are a fixture, for the reason test_pmu_run_identity.cpp
 * gives:** PMU records are written through MMIO on a2a3 and validated out of an
 * AICore staging ring on a5. Everything that decides configuration identity,
 * ordering and publication here is production.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
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
// makes `pmu_aicpu_init` skip the MMIO programming entirely.
extern "C" {
namespace {
uint64_t g_pmu_frozen_reg_addrs = 0;
}

void set_platform_pmu_reg_addrs(uint64_t pmu_regs) { g_pmu_frozen_reg_addrs = pmu_regs; }
uint64_t get_platform_pmu_reg_addrs() { return g_pmu_frozen_reg_addrs; }
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

void *frozen_alloc(size_t size) { return std::calloc(1, size); }

int frozen_free(void *ptr) {
    std::free(ptr);
    return 0;
}

std::thread frozen_thread_factory(std::function<void()> fn) { return std::thread(std::move(fn)); }

class OutputRoot {
public:
    explicit OutputRoot(const char *name) {
        path_ =
            fs::temp_directory_path() / ("simpler-pmu-frozen-" + std::string(name) + "-" + std::to_string(::getpid()));
        std::error_code ec;
        fs::remove_all(path_, ec);
        fs::create_directories(path_, ec);
    }
    ~OutputRoot() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
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

std::string header_line(const std::string &content) {
    std::istringstream lines(content);
    std::string header;
    std::getline(lines, header);
    return header;
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

/**
 * One core's device-side production for a run: acquire a buffer, fill it,
 * publish it. The `run_epoch` stamp is production — `pmu_aicpu_init` writes it
 * and nothing here touches it.
 */
void produce_one_run(void *shm, uint64_t epoch, int records) {
    set_platform_run_result(/*region_base=*/0, epoch);
    const uint32_t core_ids[] = {0};
    pmu_aicpu_init(core_ids, /*num_cores=*/1);
    PmuBufferState *state = get_pmu_buffer_state(shm, 0);
    ASSERT_NE(state->current_buf_ptr, 0u);
    auto *buf = reinterpret_cast<PmuBuffer *>(state->current_buf_ptr);
    for (int i = 0; i < records; i++) {
        buf->records[i].task_id = 0xB00 + static_cast<uint64_t>(i);
        buf->records[i].func_id = static_cast<uint32_t>(i);
        buf->records[i].pmu_total_cycles = 7000 + static_cast<uint64_t>(i);
    }
    buf->count = static_cast<uint32_t>(records);
    state->total_record_count += static_cast<uint32_t>(records);
    wmb();
    const int cores[] = {0};
    pmu_aicpu_flush_buffers(/*thread_idx=*/0, cores, /*core_num=*/1);
}

/**
 * A run's configuration is frozen at its admission.
 *
 * Run A is admitted with one event type, produces its rows and closes at its
 * own boundary — which is where production closes it, because the successor's
 * admission zeroes the per-core counters its snapshot reads. Run B is then
 * admitted with a *different* event type, which rewrites the collector's live
 * event type, its CSV header and the device header. A's rows are still to be
 * merged and named from its own frozen copy, and the buffers carrying them are
 * consumed by a collector thread that runs against whatever configuration is
 * live at the time.
 *
 * So a live read instead of a frozen one shows up as B's columns, or B's
 * `event_type` value, inside A's file. What this case does **not** do is force
 * A's merge to land after B's admission: the writer publishes on its own
 * schedule, and there is no knob that holds it without also holding the
 * device-side order this contract depends on. It is a sound detector of the
 * defect rather than a guaranteed-window one, and the row-level assertions —
 * written by the collector shard, not by the writer — are the part that
 * overlaps by construction.
 */
TEST(PmuFrozenConfig, APendingRunKeepsItsColumnsWhileASuccessorResetsThem) {
    OutputRoot root("frozen-columns");
    PmuCollector collector;
    collector.configure_retained_runs(true);
    ASSERT_EQ(collector.init(1, 1, frozen_alloc, nullptr, frozen_free, /*device_id=*/0), 0);
    void *shm = collector.get_pmu_shm_device_ptr();
    ASSERT_NE(shm, nullptr);
    collector.start(frozen_thread_factory);
    set_pmu_enabled(true);
    set_platform_pmu_base(reinterpret_cast<uint64_t>(shm));
    set_platform_pmu_reg_addrs(0);

    const fs::path a = root.csv("run-a");
    const fs::path b = root.csv("run-b");

    // A, under PIPE_UTILIZATION: produced and closed while it owns the device,
    // so its counters are its own.
    ASSERT_TRUE(collector.run_begin(2101, a.string(), PmuEventType::PIPE_UTILIZATION));
    produce_one_run(shm, 2101, /*records=*/2);
    collector.run_close(2101, /*device_execution_complete=*/true);

    // B's admission is what rewrites the collector's own event type, its header
    // and the device header — while A's file is still to be written.
    ASSERT_TRUE(collector.run_begin(2102, b.string(), PmuEventType::MEMORY));
    produce_one_run(shm, 2102, /*records=*/1);
    collector.run_close(2102, /*device_execution_complete=*/true);

    std::string error;
    EXPECT_TRUE(collector.flush_retained_runs(simpler::dfx::pmu::kCutAckBudgetMs * 8, &error)) << error;

    const std::string content_a = read_file(a);
    const std::string content_b = read_file(b);
    ASSERT_FALSE(content_a.empty()) << "run A published nothing";
    ASSERT_FALSE(content_b.empty()) << "run B published nothing";

    // Two different event types name two different column sets, and each file
    // carries its own.
    EXPECT_NE(header_line(content_a), header_line(content_b))
        << "a successor's event type reached a predecessor's columns:\n"
        << header_line(content_a);

    const std::vector<std::string> type_a = csv_column(content_a, "event_type");
    const std::vector<std::string> type_b = csv_column(content_b, "event_type");
    ASSERT_EQ(type_a.size(), 2u) << content_a;
    ASSERT_EQ(type_b.size(), 1u) << content_b;
    for (const std::string &v : type_a) {
        EXPECT_EQ(v, std::to_string(static_cast<uint32_t>(PmuEventType::PIPE_UTILIZATION)));
    }
    for (const std::string &v : type_b) {
        EXPECT_EQ(v, std::to_string(static_cast<uint32_t>(PmuEventType::MEMORY)));
    }

    // And each run's rows are its own.
    const std::vector<std::string> epochs_a = csv_column(content_a, "run_epoch");
    const std::vector<std::string> epochs_b = csv_column(content_b, "run_epoch");
    for (const std::string &v : epochs_a)
        EXPECT_EQ(v, "2101");
    for (const std::string &v : epochs_b)
        EXPECT_EQ(v, "2102");

    set_pmu_enabled(false);
    set_platform_pmu_base(0);
    set_platform_run_result(0, 0);
    collector.finish_retained_runs();
    collector.stop();
    collector.finalize(nullptr, frozen_free);
}

/**
 * The default path opens its run window before its threads start.
 *
 * This is the runner's order for a collector that retains nothing, and it is
 * load-bearing: `start()` on that path resets the run's shard files, counters
 * and header, so starting first would discard what `begin_run` had just bound.
 * Driven with the real threads and the real drain, so the rows reach the final
 * CSV the way a run's do.
 */
TEST(PmuFrozenConfig, TheDefaultPathBeginsItsRunBeforeItsThreadsStart) {
    OutputRoot root("default-order");
    PmuCollector collector;
    // Never configured to retain, which is the default.
    ASSERT_FALSE(collector.retains_runs());
    ASSERT_EQ(collector.init(1, 1, frozen_alloc, nullptr, frozen_free, /*device_id=*/0), 0);
    void *shm = collector.get_pmu_shm_device_ptr();
    ASSERT_NE(shm, nullptr);

    const fs::path a = root.csv("run-a");
    // The runner's order, exactly: bind the window, then start the threads.
    collector.begin_run(a.string(), PmuEventType::MEMORY);
    collector.start(frozen_thread_factory);

    set_pmu_enabled(true);
    set_platform_pmu_base(reinterpret_cast<uint64_t>(shm));
    set_platform_pmu_reg_addrs(0);
    produce_one_run(shm, 2201, /*records=*/2);

    // The boundary the runner uses on this path: drain, then reconcile, which
    // is what merges the shards into the final CSV.
    collector.quiesce();
    collector.reconcile_counters();

    const std::string content = read_file(a);
    ASSERT_FALSE(content.empty()) << "the default path published nothing";
    const std::vector<std::string> epochs = csv_column(content, "run_epoch");
    EXPECT_EQ(epochs.size(), 2u) << content;
    for (const std::string &v : epochs)
        EXPECT_EQ(v, "2201");
    // This run's own event type named the columns, and the header is the one
    // `begin_run` bound rather than a default the start could have rebuilt.
    const std::vector<std::string> types = csv_column(content, "event_type");
    for (const std::string &v : types) {
        EXPECT_EQ(v, std::to_string(static_cast<uint32_t>(PmuEventType::MEMORY)));
    }

    set_pmu_enabled(false);
    set_platform_pmu_base(0);
    set_platform_run_result(0, 0);
    collector.stop();
    collector.finalize(nullptr, frozen_free);
}

/**
 * A row written *after* a successor reset the live configuration still carries
 * its own run's columns.
 *
 * The case above cannot prove when its rows were written: the drain and
 * collector threads choose that instant, and A's whole file can be finished
 * before B is admitted, so a live-configuration bug would pass whenever the
 * host keeps up. This one fixes both ends of the window instead of hoping for
 * it:
 *
 * - the background writer is **held**, so A is provably still open when B's
 *   admission returns — asserted, not assumed. The hold is on the writer and
 *   not on a collector shard, so B's reference handshake still lands and its
 *   admission still succeeds.
 * - A's record is delivered through the real routing path **on this thread**,
 *   after `run_begin(B)` has already rewritten the collector's live event type,
 *   its CSV header and the device header.
 *
 * A's close still precedes B's admission, so A's counters are its own. What is
 * left non-deterministic is nothing this case asserts.
 */
TEST(PmuFrozenConfig, ARowWrittenAfterASuccessorsResetKeepsItsOwnRunsColumns) {
    OutputRoot root("frozen-witness");
    PmuCollector collector;
    collector.configure_retained_runs(true);
    ASSERT_EQ(collector.init(1, 1, frozen_alloc, nullptr, frozen_free, /*device_id=*/0), 0);
    void *shm = collector.get_pmu_shm_device_ptr();
    ASSERT_NE(shm, nullptr);
    collector.start(frozen_thread_factory);
    set_pmu_enabled(true);
    set_platform_pmu_base(reinterpret_cast<uint64_t>(shm));
    set_platform_pmu_reg_addrs(0);

    const fs::path a = root.csv("run-a");
    const fs::path b = root.csv("run-b");

    // Nothing may be sealed until this case says so.
    collector.hold_retained_writer_for_test(true);

    // A, under PIPE_UTILIZATION. Its one record is accounted on the device now
    // and delivered later, so its close snapshot is complete and honest: one
    // record produced, no buffer left in a core's hand.
    ASSERT_TRUE(collector.run_begin(3101, a.string(), PmuEventType::PIPE_UTILIZATION));
    PmuBufferState *state = get_pmu_buffer_state(shm, 0);
    state->total_record_count += 1;
    wmb();
    collector.run_close(3101, /*device_execution_complete=*/true);

    // B's admission rewrites every live copy of the configuration.
    ASSERT_TRUE(collector.run_begin(3102, b.string(), PmuEventType::MEMORY));

    // The window this case is about: A is still open, so the row below is
    // written by an epoch whose configuration was frozen before B touched it.
    const simpler::dfx::pmu::RetainedRunStats held = collector.retained_run_stats_for_test();
    ASSERT_EQ(held.open_epochs, 2u) << "the writer did not hold A open; the witness window is gone";
    ASSERT_FALSE(held.fatal);

    auto late = std::make_unique<PmuBuffer>();
    late->records[0].task_id = 0xC01;
    late->records[0].func_id = 7;
    late->records[0].pmu_total_cycles = 9100;
    late->count = 1;
    late->run_epoch = 3101;
    collector.deliver_buffer_for_test(late.get(), /*core_id=*/0, /*thread_idx=*/0, /*collector_shard=*/0);
    // Routed into A, not dropped as unplaceable: that is what makes the
    // assertion below about A's columns rather than about nothing.
    EXPECT_EQ(collector.retained_run_stats_for_test().errors.unknown_epoch_buffers, 0u)
        << "the late row was not attributed to run A";

    // B's own record, through the real producer and the real drain.
    produce_one_run(shm, 3102, /*records=*/1);
    collector.run_close(3102, /*device_execution_complete=*/true);

    collector.hold_retained_writer_for_test(false);
    std::string error;
    EXPECT_TRUE(collector.flush_retained_runs(simpler::dfx::pmu::kCutAckBudgetMs * 8, &error)) << error;

    const std::string content_a = read_file(a);
    const std::string content_b = read_file(b);
    ASSERT_FALSE(content_a.empty()) << "run A published nothing";
    ASSERT_FALSE(content_b.empty()) << "run B published nothing";
    EXPECT_NE(header_line(content_a), header_line(content_b))
        << "a successor's event type reached a row written after its reset:\n"
        << header_line(content_a);

    const std::vector<std::string> type_a = csv_column(content_a, "event_type");
    ASSERT_EQ(type_a.size(), 1u) << content_a;
    EXPECT_EQ(type_a[0], std::to_string(static_cast<uint32_t>(PmuEventType::PIPE_UTILIZATION)))
        << "the row written after the reset took the live event type";
    const std::vector<std::string> epochs_a = csv_column(content_a, "run_epoch");
    ASSERT_EQ(epochs_a.size(), 1u);
    EXPECT_EQ(epochs_a[0], "3101");

    const std::vector<std::string> type_b = csv_column(content_b, "event_type");
    ASSERT_EQ(type_b.size(), 1u) << content_b;
    EXPECT_EQ(type_b[0], std::to_string(static_cast<uint32_t>(PmuEventType::MEMORY)));

    set_pmu_enabled(false);
    set_platform_pmu_base(0);
    set_platform_run_result(0, 0);
    collector.finish_retained_runs();
    collector.stop();
    collector.finalize(nullptr, frozen_free);
}

}  // namespace
