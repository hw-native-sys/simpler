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
 * DepGen cross-run retention: what the boundary owns, and what the writer is
 * allowed to publish.
 *
 * Both sides are production code. Records are produced through the real AICPU
 * module — `dep_gen_aicpu_init` / `_record_submit` / `_flush` acquire, stamp
 * and publish exactly as they do on device — consumed through the collector's
 * own `on_buffer_collected`, sealed by its own boundary, and published by its
 * own writer through the real replay. What a case supplies is the shape of the
 * run and what the device left behind.
 *
 * What these cannot cover, stated rather than simulated: device cache
 * visibility (every store here is plain host memory), and exhausting a real
 * 256 MiB budget — the budget refusals below shrink the budget instead, and
 * assert the refusal's *consequence* through the production classifier and
 * accounting.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "aicpu/dep_gen_collector_aicpu.h"
#include "aicpu/device_run_result_base_aicpu.h"
#include "common/dep_gen.h"
#include "common/memory_barrier.h"
#include "host/dep_gen_collector.h"
#include "support/test_task_id.h"
#include "dep_gen_replay.h"

namespace fs = std::filesystem;
namespace runs = simpler::dfx::runs;
namespace dg_runs = simpler::dfx::dep_gen_runs;

namespace {

constexpr uint8_t kCreatorEdgeDepKinds = 0x3;  // wait | retain

void *retained_alloc(size_t size) { return std::calloc(1, size); }

int retained_free(void *ptr) {
    std::free(ptr);
    return 0;
}

std::thread retained_thread_factory(std::function<void()> fn) { return std::thread(std::move(fn)); }

std::string read_file(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return {};
    return std::string{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

/** How many `"task_id"` keys a deps.json carries, i.e. its task count. */
size_t count_key(const std::string &json, const std::string &key) {
    size_t n = 0;
    for (size_t at = json.find(key); at != std::string::npos; at = json.find(key, at + key.size()))
        n++;
    return n;
}

struct OutputRoot {
    fs::path base;

    explicit OutputRoot(const char *name) {
        base = fs::temp_directory_path() / ("dep_gen_retained_" + std::string(name) + "_" + std::to_string(::getpid()));
        fs::remove_all(base);
        EXPECT_TRUE(fs::create_directories(base));
    }
    ~OutputRoot() { fs::remove_all(base); }

    fs::path prefix(const char *run) const { return base / run; }
    fs::path artifact(const char *run) const { return prefix(run) / "deps.json"; }
    fs::path temp(const char *run) const { return prefix(run) / "deps.json.tmp"; }
};

class DepGenRetainedTest : public ::testing::Test {
protected:
    void configure(bool retained, size_t budget = runs::kDefaultBudgetBytes) {
        collector_.configure_retained_runs(retained, budget);
        ASSERT_EQ(collector_.init(/*num_threads=*/1, retained_alloc, nullptr, retained_free, /*device_id=*/0), 0);
        shm_ = collector_.get_dep_gen_shm_device_ptr();
        ASSERT_NE(shm_, nullptr);
        state_ = get_dep_gen_buffer_state(shm_, 0);
        header_ = get_dep_gen_header(shm_);
        set_dep_gen_enabled(true);
        set_platform_dep_gen_base(reinterpret_cast<uint64_t>(shm_));
        dep_gen_aicpu_set_orch_thread_idx(0);
    }

    void TearDown() override {
        collector_.pause_writer_for_test(false);
        dep_gen_aicpu_finalize();
        set_dep_gen_enabled(false);
        set_platform_dep_gen_base(0);
        set_platform_run_result(0, 0);
        collector_.finalize(nullptr, retained_free);
    }

    bool admit(uint64_t epoch, const fs::path &prefix) {
        collector_.start(retained_thread_factory);
        return collector_.run_begin(epoch, prefix.string());
    }

    /**
     * One run's device-side sequence: a chain of `submits` tasks, each
     * depending on the one before, so the graph has real edges rather than an
     * isolated task set.
     */
    void produce_chain(uint64_t epoch, int submits, uint32_t first_task_id) {
        set_platform_run_result(/*region_base=*/0, epoch);
        dep_gen_aicpu_init();
        const int32_t kernel_ids[3] = {-1, -1, -1};
        for (int i = 0; i < submits; i++) {
            const TaskId me = simpler::ut::test_task_id(first_task_id + static_cast<uint32_t>(i));
            TaskId dep = simpler::ut::test_task_id(first_task_id + static_cast<uint32_t>(i) - 1);
            const uint8_t kind = kCreatorEdgeDepKinds;
            dep_gen_aicpu_record_submit(
                me, /*in_manual_scope=*/false, /*early_dispatch=*/false, /*sync_start=*/false, /*predicated=*/false,
                /*tensor_count=*/0,
                /*tensor_ptrs=*/nullptr, /*arg_types=*/nullptr,
                /*explicit_dep_count=*/i == 0 ? 0 : 1, /*explicit_deps=*/i == 0 ? nullptr : &dep,
                /*explicit_dep_kinds_raw=*/i == 0 ? nullptr : &kind, kCreatorEdgeDepKinds, /*block_num=*/1, kernel_ids
            );
        }
        dep_gen_aicpu_flush();
    }

    void close(uint64_t epoch, bool complete = true) { collector_.run_close(epoch, complete); }
    bool flush(std::string *error) { return collector_.flush_retained_runs(5000, error); }

    DepGenCollector collector_;
    void *shm_ = nullptr;
    DepGenBufferState *state_ = nullptr;
    DepGenDataHeader *header_ = nullptr;
};

}  // namespace

/**
 * A sealed run survives the next run's arming, and each artifact holds only
 * its own graph.
 *
 * The writer is paused across run B's whole sequence, so B is armed and run
 * while A is still unpublished — the overlap the public API cannot force. Task
 * ids are deliberately reused across the two runs: they are run-local, so the
 * graphs are told apart by their topology, not by identity.
 */
TEST_F(DepGenRetainedTest, ASealedRunSurvivesTheNextRunsArming) {
    configure(/*retained=*/true);
    OutputRoot root("sealed-survives");

    collector_.pause_writer_for_test(true);
    ASSERT_TRUE(admit(9001, root.prefix("run-a")));
    produce_chain(9001, /*submits=*/3, /*first_task_id=*/0x1000);
    close(9001);

    // B reuses A's task ids and adds two more, so a mixed-up graph is visible
    // as a task count rather than as an id collision.
    ASSERT_TRUE(admit(9002, root.prefix("run-b")));
    produce_chain(9002, /*submits=*/5, /*first_task_id=*/0x1000);
    close(9002);

    collector_.pause_writer_for_test(false);
    std::string error;
    ASSERT_TRUE(flush(&error)) << error;

    const std::string a = read_file(root.artifact("run-a"));
    const std::string b = read_file(root.artifact("run-b"));
    ASSERT_FALSE(a.empty()) << "run A's graph was destroyed by run B's arming";
    ASSERT_FALSE(b.empty());
    EXPECT_EQ(count_key(a, "\"task_id\""), 3u);
    EXPECT_EQ(count_key(b, "\"task_id\""), 5u);
    // A chain of n tasks has n-1 edges, and each run's edge count must be its
    // own: merged records would show up here before anywhere else.
    EXPECT_EQ(count_key(a, "\"pred\""), 2u);
    EXPECT_EQ(count_key(b, "\"pred\""), 4u);
    EXPECT_EQ(collector_.retained_run_stats_for_test().published, 2u);
    EXPECT_EQ(collector_.retained_run_stats_for_test().open_slots, 0u);
    EXPECT_FALSE(fs::exists(root.temp("run-a")));
    EXPECT_FALSE(fs::exists(root.temp("run-b")));
}

/**
 * A third admission is refused while two exports are unpublished, and succeeds
 * once one has been published.
 *
 * The bound covers the run being collected, so two is two — not two plus the
 * one filling.
 */
TEST_F(DepGenRetainedTest, TwoUnpublishedExportsAreAllowedAndAThirdIsRefused) {
    configure(/*retained=*/true);
    OutputRoot root("third-refused");

    collector_.pause_writer_for_test(true);
    ASSERT_TRUE(admit(9101, root.prefix("run-a")));
    produce_chain(9101, 2, 0x2000);
    close(9101);
    ASSERT_TRUE(admit(9102, root.prefix("run-b")));
    produce_chain(9102, 2, 0x2000);
    close(9102);

    EXPECT_FALSE(admit(9103, root.prefix("run-c"))) << "a third unpublished export must be refused";

    collector_.pause_writer_for_test(false);
    std::string error;
    ASSERT_TRUE(flush(&error)) << error;
    EXPECT_TRUE(admit(9104, root.prefix("run-d")));
    produce_chain(9104, 2, 0x2000);
    close(9104);
    ASSERT_TRUE(flush(&error)) << error;
    EXPECT_EQ(collector_.retained_run_stats_for_test().published, 3u);
}

/**
 * A run that submitted nothing publishes an empty graph under its own epoch.
 *
 * The records map has no entry for such a run, so an implementation that took
 * its identity from the map would have none — the epoch and the destination
 * come from the admission record instead.
 */
TEST_F(DepGenRetainedTest, AnEmptyRunPublishesAnEmptyGraphUnderItsOwnEpoch) {
    configure(/*retained=*/true);
    OutputRoot root("empty-run");

    ASSERT_TRUE(admit(9201, root.prefix("run-a")));
    set_platform_run_result(0, 9201);
    dep_gen_aicpu_init();
    dep_gen_aicpu_flush();
    close(9201);

    std::string error;
    ASSERT_TRUE(flush(&error)) << error;
    const std::string a = read_file(root.artifact("run-a"));
    ASSERT_FALSE(a.empty()) << "an empty graph is this run's answer, not a missing file";
    EXPECT_EQ(count_key(a, "\"task_id\""), 0u);
    EXPECT_EQ(count_key(a, "\"pred\""), 0u);
    EXPECT_EQ(collector_.retained_run_stats_for_test().published, 1u);
}

/**
 * A device counter at the saturation sentinel is counts-unknown: no file, and
 * the flush reports it.
 *
 * `UINT32_MAX` is the producer's clamp rather than a wrapped value, so no
 * identity over it is a proof of completeness.
 */
TEST_F(DepGenRetainedTest, ASaturatedCounterPublishesNothingAndFailsTheFlush) {
    configure(/*retained=*/true);
    OutputRoot root("saturated");

    ASSERT_TRUE(admit(9301, root.prefix("run-a")));
    produce_chain(9301, 2, 0x3000);
    state_->total_record_count = UINT32_MAX;
    wmb();
    close(9301);

    std::string error;
    EXPECT_FALSE(flush(&error)) << "a saturated counter must fail the flush";
    EXPECT_FALSE(error.empty());
    EXPECT_FALSE(fs::exists(root.artifact("run-a")));
    EXPECT_FALSE(fs::exists(root.temp("run-a")));
    const auto stats = collector_.retained_run_stats_for_test();
    EXPECT_EQ(stats.published, 0u);
    EXPECT_EQ(stats.open_slots, 0u) << "a refusal must settle its own slot";
}

/**
 * A failed shared-region read publishes nothing.
 *
 * The failure matters because the host shadow still holds the zeroes
 * `begin_run()` wrote, so the count identity balances over a run that
 * collected nothing — "the comparison balanced" is not "the comparison was
 * made".
 */
TEST_F(DepGenRetainedTest, AFailedRegionReadPublishesNothingAndFailsTheFlush) {
    configure(/*retained=*/true);
    OutputRoot root("region-read");

    ASSERT_TRUE(admit(9401, root.prefix("run-a")));
    produce_chain(9401, 2, 0x4000);
    collector_.fail_region_read_for_test(true);
    close(9401);
    collector_.fail_region_read_for_test(false);

    std::string error;
    EXPECT_FALSE(flush(&error));
    EXPECT_FALSE(fs::exists(root.artifact("run-a")));
    EXPECT_EQ(collector_.retained_run_stats_for_test().open_slots, 0u);
}

/** A counter reset that did not reach the device refuses the admission itself. */
TEST_F(DepGenRetainedTest, AnUnpublishedCounterResetRefusesTheAdmission) {
    configure(/*retained=*/true);
    OutputRoot root("reset-failed");

    collector_.fail_counter_reset_for_test(true);
    EXPECT_FALSE(admit(9501, root.prefix("run-a"))) << "the run must fail before anything is submitted";
    collector_.fail_counter_reset_for_test(false);

    // The refused admission left no slot behind, so the next run is admitted.
    EXPECT_EQ(collector_.retained_run_stats_for_test().open_slots, 0u);
    EXPECT_TRUE(admit(9502, root.prefix("run-b")));
    produce_chain(9502, 2, 0x5000);
    close(9502);
    std::string error;
    EXPECT_FALSE(flush(&error)) << "the earlier refusal stays sticky";
    EXPECT_TRUE(fs::exists(root.artifact("run-b"))) << "the later clean run still publishes";
}

/**
 * Records stamped with an epoch this collector never admitted are an
 * attribution failure, not a new graph.
 *
 * Grouping by whatever a buffer carries would file a stray buffer under a
 * fresh key and publish it as if it were a run.
 */
TEST_F(DepGenRetainedTest, RecordsFromANonAdmittedEpochRefuseTheRun) {
    configure(/*retained=*/true);
    OutputRoot root("foreign-epoch");

    ASSERT_TRUE(admit(9601, root.prefix("run-a")));
    // The producer stamps whatever epoch the host published, so publishing a
    // different one is how a stray buffer arises without touching the record.
    produce_chain(/*epoch=*/7777, 2, 0x6000);
    close(9601);

    std::string error;
    EXPECT_FALSE(flush(&error));
    EXPECT_FALSE(fs::exists(root.artifact("run-a")));
    EXPECT_GT(collector_.retained_run_stats_for_test().foreign_epoch_records, 0u);
}

/**
 * A budget too small to lay this run's records out contiguously publishes
 * nothing, and the charge is fully credited back.
 *
 * The refusal is the point: the replay indexes its input, so the contiguous
 * copy is not optional, and a graph that cannot be built is not written.
 */
TEST_F(DepGenRetainedTest, ABudgetRefusalPublishesNothingAndSettlesItsCharges) {
    // Sized so the run is admitted and its publication is not. `open()`
    // requires the budget to cover the fixed overhead (~1.01 MiB: two export
    // slots, their path allowances, the error record and the 1 MiB
    // serialization scratch) plus a 16 MiB working set, so 18 MiB opens. What
    // is left after the overhead is ~16.99 MiB, and the two replay tensormaps
    // alone reserve 98304 + 65536x272 + 24xSUM(W_r) bytes — over 17.9 MiB even
    // at this graph's window sizes. The refusal is therefore structural rather
    // than a guess about this record count.
    configure(/*retained=*/true, runs::kMinWorkingSetBytes + runs::kWriterScratchBytes * 2);
    OutputRoot root("budget-refusal");

    ASSERT_TRUE(admit(9701, root.prefix("run-a")));
    produce_chain(9701, 4, 0x7000);
    close(9701);

    std::string error;
    EXPECT_FALSE(flush(&error)) << "a refused charge must fail the flush";
    EXPECT_FALSE(fs::exists(root.artifact("run-a")));
    EXPECT_FALSE(fs::exists(root.temp("run-a"))) << "a refusal must leave no temporary behind";
    const auto stats = collector_.retained_run_stats_for_test();
    EXPECT_EQ(stats.published, 0u);
    EXPECT_EQ(stats.open_slots, 0u);
}

/**
 * Two exports' charges settle back to the collector's fixed overhead.
 *
 * The guard against crediting a released block twice, and against leaving a
 * publication's working storage charged after it ends.
 */
TEST_F(DepGenRetainedTest, TwoExportsChargesSettleBackToTheFixedOverhead) {
    configure(/*retained=*/true);
    OutputRoot root("charges-settle");

    // Taken after the first admission opens the budget, so it is the
    // collector's fixed reservation and nothing else.
    ASSERT_TRUE(admit(9801, root.prefix("run-a")));
    const size_t fixed_only = collector_.retained_run_stats_for_test().charged_bytes;
    produce_chain(9801, 3, 0x8000);
    close(9801);
    ASSERT_TRUE(admit(9802, root.prefix("run-b")));
    produce_chain(9802, 3, 0x8000);
    close(9802);

    std::string error;
    ASSERT_TRUE(flush(&error)) << error;
    const auto stats = collector_.retained_run_stats_for_test();
    EXPECT_EQ(stats.published, 2u);
    EXPECT_EQ(stats.open_slots, 0u);
    // Everything two runs charged — their record blocks, the contiguous
    // layouts and both replay working sets — is credited once they are
    // published, so what remains is the reservation the budget opened with and
    // not a byte more. A block credited twice would show up here as a figure
    // below the fixed reservation.
    EXPECT_EQ(stats.charged_bytes, fixed_only);
}

/**
 * A destination that already holds a deps.json fails the run without touching
 * the file there, and leaves no temporary.
 *
 * `link` never replaces a name, which is what makes the publication atomic and
 * the occupying file safe.
 */
TEST_F(DepGenRetainedTest, AnOccupiedDestinationFailsWithoutClobbering) {
    configure(/*retained=*/true);
    OutputRoot root("occupied");

    ASSERT_TRUE(fs::create_directories(root.prefix("run-a")));
    const std::string occupant = "{\"not\":\"ours\"}\n";
    {
        std::ofstream out(root.artifact("run-a"), std::ios::binary);
        out << occupant;
    }

    ASSERT_TRUE(admit(9901, root.prefix("run-a")));
    produce_chain(9901, 2, 0x9000);
    close(9901);

    std::string error;
    EXPECT_FALSE(flush(&error)) << "a publication failure must fail the flush";
    EXPECT_EQ(read_file(root.artifact("run-a")), occupant) << "the file already there was modified";
    EXPECT_FALSE(fs::exists(root.temp("run-a")));
}

/** A run with no completion proof reads nothing shared and publishes nothing. */
TEST_F(DepGenRetainedTest, UnprovenCompletionPublishesNothing) {
    configure(/*retained=*/true);
    OutputRoot root("unproven");

    ASSERT_TRUE(admit(10001, root.prefix("run-a")));
    produce_chain(10001, 2, 0xA000);
    close(10001, /*complete=*/false);

    std::string error;
    EXPECT_FALSE(flush(&error));
    EXPECT_FALSE(fs::exists(root.artifact("run-a")));
    const auto stats = collector_.retained_run_stats_for_test();
    EXPECT_EQ(stats.quarantined, 1u);
    EXPECT_EQ(stats.published, 0u);

    // The quarantine holds this collector's live records, so a later run is
    // refused rather than clearing them.
    EXPECT_FALSE(admit(10002, root.prefix("run-b")));
    collector_.discard_quarantined_runs();
    EXPECT_TRUE(admit(10003, root.prefix("run-c")));
}

/**
 * A run whose launch submitted nothing gives its slot back.
 *
 * Left held, two of them would exhaust admission for the runner's life.
 */
TEST_F(DepGenRetainedTest, AnAbandonedRunReleasesItsSlot) {
    configure(/*retained=*/true);
    OutputRoot root("abandoned");

    ASSERT_TRUE(admit(10101, root.prefix("run-a")));
    EXPECT_EQ(collector_.retained_run_stats_for_test().open_slots, 1u);
    collector_.abandon_run(10101);
    EXPECT_EQ(collector_.retained_run_stats_for_test().open_slots, 0u);

    ASSERT_TRUE(admit(10102, root.prefix("run-b")));
    ASSERT_TRUE(admit(10103, root.prefix("run-c")));
    EXPECT_FALSE(admit(10104, root.prefix("run-d"))) << "both slots are in use again";
}

/**
 * Retention off keeps the boundary write, and its bytes.
 *
 * The default artifact is the guard on the promise that a clean run's output
 * does not change: the same records through both modes produce the same file.
 */
TEST_F(DepGenRetainedTest, RetentionOffProducesTheSameBytesAsRetentionOn) {
    OutputRoot root("default-bytes");

    // Retained first, so its artifact is the reference.
    configure(/*retained=*/true);
    ASSERT_TRUE(admit(10201, root.prefix("run-retained")));
    produce_chain(10201, 4, 0xB000);
    close(10201);
    std::string error;
    ASSERT_TRUE(flush(&error)) << error;
    const std::string retained_bytes = read_file(root.artifact("run-retained"));
    ASSERT_FALSE(retained_bytes.empty());

    // A second collector on the default path, same records.
    DepGenCollector plain;
    ASSERT_EQ(plain.init(1, retained_alloc, nullptr, retained_free, 0), 0);
    void *shm = plain.get_dep_gen_shm_device_ptr();
    set_platform_dep_gen_base(reinterpret_cast<uint64_t>(shm));
    ASSERT_TRUE(plain.begin_run());
    plain.start(retained_thread_factory);
    set_platform_run_result(0, 10202);
    dep_gen_aicpu_init();
    const int32_t kernel_ids[3] = {-1, -1, -1};
    for (int i = 0; i < 4; i++) {
        const TaskId me = simpler::ut::test_task_id(0xB000 + static_cast<uint32_t>(i));
        TaskId dep = simpler::ut::test_task_id(0xB000 + static_cast<uint32_t>(i) - 1);
        const uint8_t kind = kCreatorEdgeDepKinds;
        dep_gen_aicpu_record_submit(
            me, false, false, false, false, 0, nullptr, nullptr, i == 0 ? 0 : 1, i == 0 ? nullptr : &dep,
            i == 0 ? nullptr : &kind, kCreatorEdgeDepKinds, 1, kernel_ids
        );
    }
    dep_gen_aicpu_flush();
    plain.quiesce();
    const auto report = plain.reconcile_report();
    ASSERT_TRUE(dg_runs::publishable(report));
    const std::string plain_path = (root.prefix("run-default") / "deps.json").string();
    fs::create_directories(root.prefix("run-default"));
    ASSERT_EQ(
        dep_gen_replay_emit_deps_json(
            plain.window_records()->data(), plain.window_records()->size(), plain_path.c_str()
        ),
        0
    );
    plain.finalize(nullptr, retained_free);

    EXPECT_EQ(read_file(plain_path), retained_bytes) << "the retained path changed a clean run's bytes";
}

// ---------------------------------------------------------------------------
// Per-run terminal banks, mapped
// ---------------------------------------------------------------------------

namespace {

/**
 * A retained collector whose producer publishes its own per-run accounting.
 *
 * The bank address normally reaches the producer through the launching run's
 * kernel arguments, which this collector does not own, so this pushes it and
 * calls the producer's close — exactly where the runner will. Its init already
 * runs per run, and is what starts this run's counters.
 */
class DepGenMappedTest : public DepGenRetainedTest {
protected:
    void configure_mapped() {
        configure(/*retained=*/true);
        ASSERT_TRUE(collector_.open_run_terminal_banks(DEP_GEN_TERMINAL_PRODUCERS));
        ASSERT_TRUE(collector_.device_publishes_run_terminal());
    }

    /**
     * Admit a run and push the bank its admission reserved, which is what the
     * host does at launch: the device is handed one resolved address per run,
     * so a successor publishes into its own bank rather than deriving one.
     */
    bool admit(uint64_t epoch, const fs::path &prefix) {
        if (!DepGenRetainedTest::admit(epoch, prefix)) return false;
        const uint64_t bank = collector_.run_terminal_bank_address(epoch);
        EXPECT_NE(bank, 0u) << "an admitted run resolved no terminal bank";
        set_platform_dep_gen_run_terminal(bank);
        return true;
    }

    void TearDown() override {
        set_platform_dep_gen_run_terminal(0);
        DepGenRetainedTest::TearDown();
    }

    /** One run's device side, closed through the producer's own entry. */
    void produce_run(uint64_t epoch, int submits, uint32_t first_task_id) {
        produce_chain(epoch, submits, first_task_id);
        dep_gen_aicpu_close_run();
    }
};

}  // namespace

TEST_F(DepGenMappedTest, AMappedRunIsSettledFromItsOwnProducerAccounting) {
    OutputRoot root("mapped-settled");
    configure_mapped();
    ASSERT_TRUE(admit(9001, root.prefix("run-a")));
    produce_run(9001, /*submits=*/3, 0x3100);
    close(9001);

    std::string error;
    ASSERT_TRUE(flush(&error)) << error;
    EXPECT_TRUE(fs::exists(root.artifact("run-a"))) << "a run whose producer reported published no graph";
}

TEST_F(DepGenMappedTest, AProducerThatNeverPublishedProducesNoGraph) {
    OutputRoot root("mapped-missing");
    configure_mapped();
    ASSERT_TRUE(admit(9101, root.prefix("run-a")));
    // The records are produced and received, but the producer's close never
    // runs. DepGen publishes one whole graph or none, so a run whose edges
    // cannot be shown to be all of its edges gets no file.
    produce_chain(9101, /*submits=*/2, 0x3200);
    close(9101);

    std::string error;
    EXPECT_FALSE(flush(&error));
    EXPECT_FALSE(fs::exists(root.artifact("run-a"))) << "a graph was published from a partial producer set";
}

TEST_F(DepGenMappedTest, AnOverflowChainStillPublishes) {
    OutputRoot root("mapped-overflow");
    configure_mapped();
    ASSERT_TRUE(admit(9201, root.prefix("run-a")));
    set_platform_run_result(/*region_base=*/0, 9201);
    dep_gen_aicpu_init();

    // A submit whose dependency list spills into a chain: the producer charges
    // the extra slots to its own counter, and the terminal reports buffer slots
    // so the two sides are in the same unit.
    constexpr int kDepCount = DEP_GEN_MAX_EXPLICIT_DEPS + 1;
    std::vector<TaskId> deps(kDepCount);
    for (int i = 0; i < kDepCount; ++i)
        deps[i] = simpler::ut::test_task_id(0x3300 + static_cast<uint32_t>(i));
    const int32_t kernel_ids[3] = {-1, -1, -1};
    dep_gen_aicpu_record_submit(
        simpler::ut::test_task_id(0x33FF), /*in_manual_scope=*/false, /*early_dispatch=*/false, /*sync_start=*/false,
        /*predicated=*/false, /*tensor_count=*/0,
        /*tensor_ptrs=*/nullptr, /*arg_types=*/nullptr, kDepCount, deps.data(), /*explicit_dep_kinds_raw=*/nullptr,
        kCreatorEdgeDepKinds, /*block_num=*/1, kernel_ids
    );
    dep_gen_aicpu_flush();
    dep_gen_aicpu_close_run();
    close(9201);

    std::string error;
    ASSERT_TRUE(flush(&error)) << error;
    EXPECT_TRUE(fs::exists(root.artifact("run-a"))) << "an overflow chain was reported as a unit mismatch";
}

TEST_F(DepGenMappedTest, ASecondAdmissionDoesNotEraseThePredecessorsEdges) {
    OutputRoot root("mapped-two-open");
    configure_mapped();
    // A is admitted and receiving. B is admitted before A seals, which is what
    // joining a successor to a running predecessor means: A's edges and
    // receipts are its own and must survive B's admission. They were not —
    // admission released the collector's single store.
    ASSERT_TRUE(admit(9301, root.prefix("run-a")));
    produce_run(9301, /*submits=*/3, 0x3400);

    ASSERT_TRUE(admit(9302, root.prefix("run-b")));
    produce_run(9302, /*submits=*/2, 0x3500);

    close(9301);
    close(9302);

    std::string error;
    ASSERT_TRUE(flush(&error)) << error;
    EXPECT_TRUE(fs::exists(root.artifact("run-a"))) << "a run lost its edges to its successor's admission";
    EXPECT_TRUE(fs::exists(root.artifact("run-b")));
}

TEST_F(DepGenMappedTest, ASuccessorsRecordsAreNotForeignWhileThePredecessorIsOpen) {
    OutputRoot root("mapped-not-foreign");
    configure_mapped();
    ASSERT_TRUE(admit(9401, root.prefix("run-a")));
    produce_run(9401, /*submits=*/1, 0x3600);

    // With one admitted epoch, B's arrivals were filed as foreign and made
    // both graphs untrustworthy. Routed by the identity its producer stamped,
    // B's records are B's.
    ASSERT_TRUE(admit(9402, root.prefix("run-b")));
    produce_run(9402, /*submits=*/2, 0x3700);
    close(9402);
    close(9401);

    std::string error;
    ASSERT_TRUE(flush(&error)) << error;
    EXPECT_TRUE(fs::exists(root.artifact("run-b"))) << "a successor's edges were refused as foreign";
    EXPECT_TRUE(fs::exists(root.artifact("run-a")));
}

TEST_F(DepGenMappedTest, ARefusedGraphReturnsItsReceiptStoreAndBudget) {
    OutputRoot root("mapped-refused-cleanup");
    configure_mapped();

    // A run whose producer never closed yields no graph. Its edges live in its
    // own receipt entry, so the refusal has to release that entry — releasing
    // the legacy store instead left the entry open with its blocks charged, and
    // later admissions then ran out of receipt entries while output slots sat
    // free.
    ASSERT_TRUE(admit(9501, root.prefix("run-a")));
    // Taken after the admission and before any record: the first admission is
    // what opens the budget and reserves this collector's fixed retained
    // overhead — the export slots, their path allowance, the permanent error
    // record and the writer's scratch — and none of that is a run's to give
    // back. What this case is about is the dynamic charge the refused run's
    // edges took, so the baseline has to already contain the fixed part or the
    // comparison asks a refusal to return storage the collector still owns.
    const size_t baseline_charged = collector_.retained_run_stats_for_test().charged_bytes;
    ASSERT_GT(baseline_charged, 0u) << "the first admission established no fixed retained overhead";
    produce_chain(9501, /*submits=*/2, 0x3800);
    close(9501);
    std::string error;
    EXPECT_FALSE(flush(&error));
    ASSERT_FALSE(fs::exists(root.artifact("run-a")));

    // The bytes the refused run's edges occupied are back in the budget, and
    // only those: the fixed overhead the baseline captured stays charged. An
    // entry left open holds its blocks, so this reads above the baseline long
    // before any later admission notices a shortage.
    EXPECT_EQ(collector_.retained_run_stats_for_test().charged_bytes, baseline_charged)
        << "a refused run's record blocks stayed charged to the budget";

    // And the entry itself is back. The discriminator is holding the whole
    // table at once: a single leaked entry still leaves one free, which serves
    // arbitrarily many admissions that close before the next one opens, so a
    // serial sequence cannot tell the two states apart.
    std::vector<uint64_t> held;
    for (size_t i = 0; i < runs::kMaxOpenEpochs; i++) {
        const uint64_t epoch = 9510 + static_cast<uint64_t>(i);
        const std::string leaf = "run-" + std::to_string(i);
        ASSERT_TRUE(admit(epoch, root.prefix(leaf.c_str())))
            << "admission " << (i + 1) << " of " << runs::kMaxOpenEpochs
            << " concurrent runs was refused, so a refused run kept its receipt entry";
        produce_run(epoch, /*submits=*/1, 0x3900 + static_cast<uint32_t>(i) * 0x10);
        held.push_back(epoch);
    }

    // Each of them is a whole graph of its own, so the recovered entry is one
    // a run can actually be published from rather than merely allocated. The
    // flush still reports failure, and that is the point: recovering the entry
    // must not retract the sticky report that the refused run's graph was lost.
    for (uint64_t epoch : held)
        close(epoch);
    std::string publish_error;
    EXPECT_FALSE(flush(&publish_error)) << "the refused run's loss was retracted by a later successful run";
    EXPECT_TRUE(collector_.retained_run_stats_for_test().has_error);
    for (size_t i = 0; i < held.size(); i++) {
        const std::string leaf = "run-" + std::to_string(i);
        EXPECT_TRUE(fs::exists(root.artifact(leaf.c_str())))
            << "a run admitted into the recovered entry published no graph";
    }
}
