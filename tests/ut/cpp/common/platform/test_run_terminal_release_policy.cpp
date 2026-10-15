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
 * Which callback each collector's per-run terminal bank array goes back
 * through, and how many times.
 *
 * A collector is handed a device allocator at init and a second one at
 * finalize, and they are not the same function on every teardown: the fatal
 * path supplies one that frees nothing, because the device has not been proved
 * stopped and the card is about to be reset. Every pooled buffer follows that
 * choice. The bank array is the collector's remaining device allocation and
 * must follow it too — releasing it through the init-time allocator would hand
 * one allocation back on the single path where the caller abandoned all the
 * others.
 *
 * The ledgers below are the production callbacks' only observable effect, so
 * each case names the callback it expects by the pointer that reached it.
 */

#include <gtest/gtest.h>

#include "host/args_dump_collector.h"
#include "host/dep_gen_collector.h"
#include "host/pmu_collector.h"
#include "host/scope_stats_collector.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace runs = simpler::dfx::runs;

namespace {

/**
 * One device allocator's record of what it was asked to free.
 *
 * `frees` is every pointer handed to it, including repeats, so a case can
 * distinguish "released once" from "released twice". `abandon` makes it report
 * success without freeing, which is what the fatal teardown's callback does;
 * the test owns the allocation either way and reclaims it at the end.
 */
struct FreeLedger {
    std::vector<void *> frees;
    std::vector<void *> live;
    bool abandon{false};

    void *allocate(size_t bytes) {
        void *p = std::calloc(1, bytes);
        if (p != nullptr) live.push_back(p);
        return p;
    }

    int free_one(void *p) {
        frees.push_back(p);
        if (abandon) return 0;
        reclaim(p);
        return 0;
    }

    size_t times_freed(void *p) const { return static_cast<size_t>(std::count(frees.begin(), frees.end(), p)); }
    bool saw(void *p) const { return times_freed(p) > 0; }

    void reclaim(void *p) {
        auto it = std::find(live.begin(), live.end(), p);
        if (it == live.end()) return;
        live.erase(it);
        std::free(p);
    }

    /** Anything an abandoning teardown left mapped, so the process leaks none. */
    ~FreeLedger() {
        for (void *p : live)
            std::free(p);
    }
};

auto alloc_from(FreeLedger *ledger) {
    return [ledger](size_t bytes) -> void * {
        return ledger->allocate(bytes);
    };
}

auto free_from(FreeLedger *ledger) {
    return [ledger](void *p) -> int {
        return ledger->free_one(p);
    };
}

}  // namespace

// --- ScopeStats -------------------------------------------------------------

TEST(ScopeStatsRunTerminalReleasePolicy, AProvedTeardownReleasesTheBankOnceThroughItsOwnCallback) {
    FreeLedger init_time;
    FreeLedger teardown;
    ScopeStatsCollector collector;
    collector.configure_retained_runs(true, runs::kDefaultBudgetBytes);
    ASSERT_EQ(collector.init(/*num_threads=*/1, alloc_from(&init_time), nullptr, free_from(&init_time), 0), 0);
    ASSERT_TRUE(collector.open_run_terminal_banks(SCOPE_STATS_TERMINAL_PRODUCERS));
    void *bank = reinterpret_cast<void *>(collector.run_terminal_device_address());
    ASSERT_NE(bank, nullptr);

    collector.finalize(nullptr, free_from(&teardown));

    EXPECT_EQ(teardown.times_freed(bank), 1u) << "the teardown's own callback did not release the bank exactly once";
    EXPECT_FALSE(init_time.saw(bank)) << "the bank went back through the init-time allocator";
    EXPECT_FALSE(collector.device_publishes_run_terminal());
}

TEST(ScopeStatsRunTerminalReleasePolicy, AnAbandoningTeardownDoesNotFreeTheBankThroughTheInitAllocator) {
    FreeLedger init_time;
    FreeLedger teardown;
    // What `finalize_collectors(/*abandon_device_resources=*/true)` supplies:
    // a callback that reports success and frees nothing, because the device was
    // never proved stopped.
    teardown.abandon = true;
    ScopeStatsCollector collector;
    collector.configure_retained_runs(true, runs::kDefaultBudgetBytes);
    ASSERT_EQ(collector.init(/*num_threads=*/1, alloc_from(&init_time), nullptr, free_from(&init_time), 0), 0);
    ASSERT_TRUE(collector.open_run_terminal_banks(SCOPE_STATS_TERMINAL_PRODUCERS));
    void *bank = reinterpret_cast<void *>(collector.run_terminal_device_address());
    ASSERT_NE(bank, nullptr);

    collector.finalize(nullptr, free_from(&teardown));

    // The bank is abandoned with everything else rather than handed back to an
    // allocator the caller has stopped using: a producer that is still running
    // would otherwise write into memory that allocator can hand to someone else.
    EXPECT_FALSE(init_time.saw(bank)) << "an abandoning teardown still freed the bank through the init allocator";
    EXPECT_EQ(teardown.times_freed(bank), 1u) << "the bank did not follow this teardown's own policy";
    init_time.reclaim(bank);
}

// --- DepGen -----------------------------------------------------------------

TEST(DepGenRunTerminalReleasePolicy, AProvedTeardownReleasesTheBankOnceThroughItsOwnCallback) {
    FreeLedger init_time;
    FreeLedger teardown;
    DepGenCollector collector;
    collector.configure_retained_runs(true, runs::kDefaultBudgetBytes);
    ASSERT_EQ(collector.init(/*num_threads=*/1, alloc_from(&init_time), nullptr, free_from(&init_time), 0), 0);
    ASSERT_TRUE(collector.open_run_terminal_banks(DEP_GEN_TERMINAL_PRODUCERS));
    void *bank = reinterpret_cast<void *>(collector.run_terminal_device_address());
    ASSERT_NE(bank, nullptr);

    collector.finalize(nullptr, free_from(&teardown));

    EXPECT_EQ(teardown.times_freed(bank), 1u) << "the teardown's own callback did not release the bank exactly once";
    EXPECT_FALSE(init_time.saw(bank)) << "the bank went back through the init-time allocator";
}

TEST(DepGenRunTerminalReleasePolicy, AnAbandoningTeardownDoesNotFreeTheBankThroughTheInitAllocator) {
    FreeLedger init_time;
    FreeLedger teardown;
    teardown.abandon = true;
    DepGenCollector collector;
    collector.configure_retained_runs(true, runs::kDefaultBudgetBytes);
    ASSERT_EQ(collector.init(/*num_threads=*/1, alloc_from(&init_time), nullptr, free_from(&init_time), 0), 0);
    ASSERT_TRUE(collector.open_run_terminal_banks(DEP_GEN_TERMINAL_PRODUCERS));
    void *bank = reinterpret_cast<void *>(collector.run_terminal_device_address());
    ASSERT_NE(bank, nullptr);

    collector.finalize(nullptr, free_from(&teardown));

    EXPECT_FALSE(init_time.saw(bank)) << "an abandoning teardown still freed the bank through the init allocator";
    EXPECT_EQ(teardown.times_freed(bank), 1u) << "the bank did not follow this teardown's own policy";
    init_time.reclaim(bank);
}

// --- ArgsDump ---------------------------------------------------------------

TEST(ArgsDumpRunTerminalReleasePolicy, AProvedTeardownReleasesTheBankOnceThroughItsOwnCallback) {
    FreeLedger init_time;
    FreeLedger teardown;
    ArgsDumpCollector collector;
    collector.configure_retained_runs(true, runs::kDefaultBudgetBytes);
    ASSERT_EQ(
        collector.initialize(
            /*num_dump_threads=*/1, /*device_id=*/0, DumpArgsLevel::FULL, alloc_from(&init_time), nullptr,
            free_from(&init_time)
        ),
        0
    );
    ASSERT_TRUE(collector.open_run_terminal_banks(/*expected_producers=*/1));
    void *bank = reinterpret_cast<void *>(collector.run_terminal_device_address());
    ASSERT_NE(bank, nullptr);

    (void)collector.finalize(nullptr, free_from(&teardown));

    EXPECT_EQ(teardown.times_freed(bank), 1u) << "the teardown's own callback did not release the bank exactly once";
    EXPECT_FALSE(init_time.saw(bank)) << "the bank went back through the init-time allocator";
}

TEST(ArgsDumpRunTerminalReleasePolicy, AnAbandoningTeardownDoesNotFreeTheBankThroughTheInitAllocator) {
    FreeLedger init_time;
    FreeLedger teardown;
    teardown.abandon = true;
    ArgsDumpCollector collector;
    collector.configure_retained_runs(true, runs::kDefaultBudgetBytes);
    ASSERT_EQ(
        collector.initialize(
            /*num_dump_threads=*/1, /*device_id=*/0, DumpArgsLevel::FULL, alloc_from(&init_time), nullptr,
            free_from(&init_time)
        ),
        0
    );
    ASSERT_TRUE(collector.open_run_terminal_banks(/*expected_producers=*/1));
    void *bank = reinterpret_cast<void *>(collector.run_terminal_device_address());
    ASSERT_NE(bank, nullptr);

    (void)collector.finalize(nullptr, free_from(&teardown));

    EXPECT_FALSE(init_time.saw(bank)) << "an abandoning teardown still freed the bank through the init allocator";
    EXPECT_EQ(teardown.times_freed(bank), 1u) << "the bank did not follow this teardown's own policy";
    init_time.reclaim(bank);
}

// --- PMU --------------------------------------------------------------------

TEST(PmuRunTerminalReleasePolicy, AProvedTeardownReleasesTheBankOnceThroughItsOwnCallback) {
    FreeLedger init_time;
    FreeLedger teardown;
    PmuCollector collector;
    ASSERT_EQ(
        collector.init(/*num_aicore=*/1, /*aicpu_threads=*/1, alloc_from(&init_time), nullptr, free_from(&init_time),
                       0),
        0
    );
    ASSERT_TRUE(collector.open_run_terminal_banks(/*expected_producers=*/1));
    void *bank = reinterpret_cast<void *>(collector.run_terminal_device_address());
    ASSERT_NE(bank, nullptr);

    collector.finalize(nullptr, free_from(&teardown));

    EXPECT_EQ(teardown.times_freed(bank), 1u) << "the teardown's own callback did not release the bank exactly once";
    EXPECT_FALSE(init_time.saw(bank)) << "the bank went back through the init-time allocator";
}

TEST(PmuRunTerminalReleasePolicy, AnAbandoningTeardownDoesNotFreeTheBankThroughTheInitAllocator) {
    FreeLedger init_time;
    FreeLedger teardown;
    teardown.abandon = true;
    PmuCollector collector;
    ASSERT_EQ(
        collector.init(/*num_aicore=*/1, /*aicpu_threads=*/1, alloc_from(&init_time), nullptr, free_from(&init_time),
                       0),
        0
    );
    ASSERT_TRUE(collector.open_run_terminal_banks(/*expected_producers=*/1));
    void *bank = reinterpret_cast<void *>(collector.run_terminal_device_address());
    ASSERT_NE(bank, nullptr);

    collector.finalize(nullptr, free_from(&teardown));

    EXPECT_FALSE(init_time.saw(bank)) << "an abandoning teardown still freed the bank through the init allocator";
    EXPECT_EQ(teardown.times_freed(bank), 1u) << "the bank did not follow this teardown's own policy";
    init_time.reclaim(bank);
}

// --- The failed free, on the path that does release ---------------------------

TEST(ScopeStatsRunTerminalReleasePolicy, ARefusedFreeRetainsTheBankAndReportsIt) {
    FreeLedger init_time;
    ScopeStatsCollector collector;
    collector.configure_retained_runs(true, runs::kDefaultBudgetBytes);
    ASSERT_EQ(collector.init(/*num_threads=*/1, alloc_from(&init_time), nullptr, free_from(&init_time), 0), 0);
    ASSERT_TRUE(collector.open_run_terminal_banks(SCOPE_STATS_TERMINAL_PRODUCERS));
    void *bank = reinterpret_cast<void *>(collector.run_terminal_device_address());
    ASSERT_NE(bank, nullptr);

    // An allocator that refuses the pointer may still own it, so the array is
    // retained rather than dropped and the collector reports the failure.
    collector.finalize(nullptr, [](void *) -> int {
        return -1;
    });

    EXPECT_FALSE(init_time.saw(bank)) << "a refused free fell back to the init-time allocator";
    init_time.reclaim(bank);
}
