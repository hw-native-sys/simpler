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

#include "runtime_c_api.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>

#include "aicpu/device_run_result_base_aicpu.h"
#include "aicpu/scope_stats_collector_aicpu.h"
#include "device_runner_base.h"
#include "host/profiling_copy.h"

namespace fs = std::filesystem;

namespace profiling_common {

struct ProfilerBaseTestPeer {
    template <typename Collector>
    static bool running(const Collector &collector) {
        return collector.mgmt_running_.load(std::memory_order_acquire);
    }
};

}  // namespace profiling_common

namespace {

constexpr uint64_t kRunEpoch = 8200;
using simpler::dfx::scope_stats_runs::Capacities;
using simpler::dfx::scope_stats_runs::CapacityMetadata;
std::atomic<const volatile void *> refused_source{nullptr};
std::atomic<uint32_t> refused_copies{0};

void *close_alloc(size_t size) { return std::calloc(1, size); }

int close_free(void *ptr) {
    std::free(ptr);
    return 0;
}

std::string read_file(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

template <typename Predicate>
bool wait_until(Predicate ready) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!ready()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::yield();
    }
    return true;
}

class DrainStartGate {
public:
    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] {
            return released_;
        });
    }
    void release() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            released_ = true;
        }
        cv_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool released_ = false;
};

class GatedClose {
public:
    GatedClose(DrainStartGate &gate, std::function<void()> close) :
        gate_(gate),
        thread_(std::move(close)) {}
    ~GatedClose() { join(); }
    void join() {
        gate_.release();
        if (thread_.joinable()) thread_.join();
    }

private:
    DrainStartGate &gate_;
    std::thread thread_;
};

// The four targets exercise the real runner boundary. The producer is injected
// through its production AICPU API; only TMR has this producer in real runs.
class OrdinaryScopeCloseRunner : public SimDeviceRunnerBase {
public:
    ScopeStatsCollector &collector() { return scope_stats_collector_; }
    int attach() { return attach_current_thread(0); }

    int prepare_execution(
        Runtime &, const CallConfig &, uint32_t, const NativeRunIdentity &, std::unique_ptr<PreparedExecution> *
    ) override {
        return PTO_RUNTIME_ERR_UNSUPPORTED;
    }
    LaunchTransactionResult launch_execution_impl(PreparedExecution &, LaunchPermit) override {
        return LaunchTransactionResult{};
    }
    void abandon_prepared_execution(PreparedExecution &) noexcept override {}
    int poll_execution(const ActiveExecution &) override { return SIMPLER_NATIVE_RUN_POLL_ERROR; }
    DrainOutcome drain_execution(ActiveExecution &) override { return DrainOutcome::device_error(-1); }
    int finalize() override { return 0; }
    int ensure_binaries_loaded() override { return 0; }
    int invoke_device_register(const RegisterCallableArgs &) override { return 0; }
};

class ScopeStatsOrdinaryClose : public ::testing::Test {
protected:
    void SetUp() override {
        refused_source.store(nullptr);
        refused_copies.store(0);
        root_ = fs::temp_directory_path() / ("scope_stats_ordinary_close_" + std::to_string(::getpid()) + "_" +
                                             ::testing::UnitTest::GetInstance()->current_test_info()->name());
        fs::remove_all(root_);
        ASSERT_TRUE(fs::create_directories(root_));
        ASSERT_EQ(runner_.attach(), 0);
        auto &collector = runner_.collector();
        ASSERT_FALSE(collector.retains_runs());
        ASSERT_EQ(collector.init(1, close_alloc, nullptr, close_free, 0), 0);
        Capacities values;
        values.task_window[0] = 23;
        values.dep_pool[0] = 31;
        values.heap[0] = 8192;
        values.tensormap = 47;
        capacities_.values = values;
        collector.begin_run(kRunEpoch, &capacities_);
        start_collector();
        ASSERT_TRUE(collector.collector_threads_started());
        shm_ = collector.get_scope_stats_shm_device_ptr();
        ASSERT_NE(shm_, nullptr);
        state_ = get_scope_stats_buffer_state(shm_, 0);
        set_scope_stats_enabled(true);
        set_platform_scope_stats_base(reinterpret_cast<uint64_t>(shm_));
        set_platform_scope_stats_run_terminal(0);
        set_platform_run_result(0, kRunEpoch);
        scope_stats_aicpu_set_orch_thread_idx(0);
        scope_stats_aicpu_begin_run();
    }

    void TearDown() override {
        refused_source.store(nullptr);
        set_scope_stats_enabled(false);
        set_platform_scope_stats_base(0);
        set_platform_scope_stats_run_terminal(0);
        set_platform_run_result(0, 0);
        runner_.collector().stop();
        runner_.collector().finalize(nullptr, close_free);
        fs::remove_all(root_);
    }

    virtual void start_collector() {
        runner_.collector().start([](std::function<void()> fn) {
            return std::thread(std::move(fn));
        });
    }

    void record(const char *site) {
        scope_stats_set_pending_site(site, 11);
        scope_stats_begin(0, 1, 2, 1024, 2048, 1, 2, 3);
    }

    void collect_published() {
        record("published.cpp");
        scope_stats_aicpu_flush_buffers();
        runner_.collector().quiesce();
        ASSERT_EQ(runner_.collector().total_collected(), 1u);
    }

    void expect_held() {
        ASSERT_NE(state_->current_buf_ptr, 0u);
        const auto *buffer = reinterpret_cast<const ScopeStatsBuffer *>(state_->current_buf_ptr);
        ASSERT_EQ(buffer->run_epoch, kRunEpoch);
        ASSERT_EQ(buffer->count, 1u);
    }

    void close(bool complete, bool stopped, uint64_t epoch = kRunEpoch) {
        DfxRunConfig dfx;
        dfx.output_prefix = root_.string();
        runner_.close_scope_stats_run_boundary(dfx, epoch, complete, stopped);
    }

    fs::path artifact() const { return root_ / "scope_stats" / "scope_stats.jsonl"; }

    void expect_unknown(const std::string &jsonl, uint64_t received = 1) {
        EXPECT_NE(jsonl.find("\"counts_unknown\": true"), std::string::npos) << jsonl;
        EXPECT_NE(jsonl.find("\"collection_verdict\": \"partial_cut_unknown\""), std::string::npos) << jsonl;
        EXPECT_NE(jsonl.find("\"fatal\": null"), std::string::npos) << jsonl;
        EXPECT_NE(jsonl.find("\"dropped\": null"), std::string::npos) << jsonl;
        EXPECT_NE(jsonl.find("\"total\": null"), std::string::npos) << jsonl;
        EXPECT_NE(jsonl.find("\"host_received_records\": " + std::to_string(received)), std::string::npos) << jsonl;
        EXPECT_NE(jsonl.find("\"host_retained_records\": " + std::to_string(received)), std::string::npos) << jsonl;
    }

    OrdinaryScopeCloseRunner runner_;
    fs::path root_;
    void *shm_ = nullptr;
    ScopeStatsBufferState *state_ = nullptr;
    CapacityMetadata capacities_;
};

class ScopeStatsOrdinaryCloseWithGatedDrain : public ScopeStatsOrdinaryClose {
protected:
    void start_collector() override {
        runner_.collector().start([this, created = 0](std::function<void()> fn) mutable {
            const bool drain = created++ == 0;
            return std::thread([this, drain, fn = std::move(fn)] {
                if (drain) gate_.wait();
                fn();
            });
        });
    }
    void TearDown() override {
        gate_.release();
        ScopeStatsOrdinaryClose::TearDown();
    }
    void collect_first_publication() {
        record("published.cpp");
        scope_stats_aicpu_flush_buffers();
        auto &collector = runner_.collector();
        auto *header = get_scope_stats_header(collector.manager().shared_mem_host());
        ScopeStatsReadyQueueEntry entry{};
        using Algorithms = profiling_common::ProfilerAlgorithms<ScopeStatsModule>;
        ASSERT_TRUE(Algorithms::try_peek_aicpu_entry(collector.manager(), header, 0, entry, true));
        // The gated drain grants exclusive access to the ready/free queues.
        ASSERT_EQ(
            Algorithms::process_entry(collector.manager(), header, 0, entry, nullptr, false),
            profiling_common::EntryOutcome::kDelivered
        );
        ASSERT_TRUE(wait_until([&collector] {
            return collector.total_collected() == 1;
        }));
    }
    bool stop_requested() { return !profiling_common::ProfilerBaseTestPeer::running(runner_.collector()); }

    void expect_restart_discards_previous_epoch(bool capacity_required) {
        auto &collector = runner_.collector();
        auto *header = get_scope_stats_header(shm_);
        record("shared_site.cpp");
        scope_stats_aicpu_flush_buffers();
        const uint32_t head = header->queue_heads[0];
        ASSERT_NE(head, header->queue_tails[0]);
        const uint64_t old_buffer = header->queues[0][head % PLATFORM_SCOPE_STATS_READYQUEUE_SIZE].buffer_ptr;
        ASSERT_NE(old_buffer, 0u);
        ASSERT_EQ(reinterpret_cast<const ScopeStatsBuffer *>(old_buffer)->run_epoch, kRunEpoch);
        ASSERT_EQ(collector.total_collected(), 0u);

        GatedClose closing(gate_, [this] {
            close(/*complete=*/false, /*stopped=*/false);
        });
        ASSERT_TRUE(wait_until([this] {
            return stop_requested();
        }));
        closing.join();
        ASSERT_FALSE(collector.collector_threads_started());
        ASSERT_EQ(header->queue_heads[0], head);
        ASSERT_NE(head, header->queue_tails[0]);

        CapacityMetadata next_capacities = capacities_;
        next_capacities.required = capacity_required;
        collector.begin_run(kRunEpoch + 1, &next_capacities);
        set_platform_run_result(0, kRunEpoch + 1);
        scope_stats_aicpu_begin_run();
        // The actual producer's open scope supplies the same site's end in
        // the new epoch. The old queued begin cannot belong to that scope.
        scope_stats_end(0, 5, 7, 4096, 8192, 4, 6, 8);
        ASSERT_NE(state_->current_buf_ptr, 0u);
        const auto *held = reinterpret_cast<const ScopeStatsBuffer *>(state_->current_buf_ptr);
        ASSERT_EQ(held->run_epoch, kRunEpoch + 1);
        ASSERT_EQ(held->count, 1u);
        ASSERT_EQ(held->records[0].phase, SCOPE_STATS_PHASE_END);
        start_collector();
        close(/*complete=*/true, /*stopped=*/false, kRunEpoch + 1);

        EXPECT_EQ(header->queue_heads[0], header->queue_tails[0]);
        const auto records = collector.collected_records();
        EXPECT_EQ(records.size(), 1u) << "a previous epoch's publication entered the new collection";
        EXPECT_EQ(collector.collected_for_run(kRunEpoch), 0u);
        EXPECT_EQ(collector.collected_for_run(kRunEpoch + 1), 1u);
        for (const auto &record : records) {
            EXPECT_EQ(record.run_epoch, kRunEpoch + 1);
            EXPECT_EQ(record.record.phase, SCOPE_STATS_PHASE_END);
        }
        const auto jsonl = read_file(artifact());
        EXPECT_NE(jsonl.find("\"site\": \"shared_site.cpp:11\""), std::string::npos) << jsonl;
        EXPECT_NE(jsonl.find("\"phase\": \"end\""), std::string::npos) << jsonl;
        EXPECT_EQ(jsonl.find("\"phase\": \"begin\""), std::string::npos) << jsonl;
        EXPECT_EQ(jsonl.find("\"run_epoch\": 8200"), std::string::npos) << jsonl;
        EXPECT_NE(jsonl.find("\"run_epoch\": 8201"), std::string::npos) << jsonl;
        const auto status = collector.retained_run_stats_for_test();
        EXPECT_TRUE(status.has_error) << "the discarded foreign publication must report the existing host error";
        EXPECT_GT(status.host_failures, 0u);
    }

    DrainStartGate gate_;
};

}  // namespace

// Detached shadows on both architectures make device reads observable without
// hardware. The real collector transport uses these copy functions unchanged.
int profiling_copy_to_device(volatile void *dst, const void *src, size_t bytes) {
    std::memcpy(const_cast<void *>(dst), src, bytes);
    return 0;
}
int profiling_copy_from_device(volatile void *dst, const volatile void *src, size_t bytes) {
    if (src == refused_source.load()) {
        refused_copies.fetch_add(1);
        return -1;
    }
    std::memcpy(const_cast<void *>(dst), const_cast<const void *>(src), bytes);
    return 0;
}
std::function<int(void *, const void *, size_t)> profiling_copy_to_device_or_null() {
    return profiling_copy_to_device_for_ops;
}
std::function<int(void *, const void *, size_t)> profiling_copy_from_device_or_null() {
    return profiling_copy_from_device_for_ops;
}

namespace {

TEST_F(ScopeStatsOrdinaryClose, CompletionRecoversAnUnpublishedHeldRecord) {
    record("held.cpp");
    expect_held();
    close(/*complete=*/true, /*stopped=*/false);
    EXPECT_TRUE(runner_.collector().collector_threads_started());
    const auto jsonl = read_file(artifact());
    EXPECT_NE(jsonl.find("\"site\": \"held.cpp:11\""), std::string::npos) << jsonl;
    EXPECT_NE(jsonl.find("\"total\": 1"), std::string::npos) << jsonl;
    EXPECT_EQ(jsonl.find("counts_unknown"), std::string::npos) << jsonl;
}

TEST_F(ScopeStatsOrdinaryClose, ProducerStopRecoversAnUnpublishedHeldRecord) {
    record("held.cpp");
    expect_held();
    close(/*complete=*/false, /*stopped=*/true);
    EXPECT_TRUE(runner_.collector().collector_threads_started());
    const auto jsonl = read_file(artifact());
    EXPECT_NE(jsonl.find("\"site\": \"held.cpp:11\""), std::string::npos) << jsonl;
    EXPECT_NE(jsonl.find("\"total\": 1"), std::string::npos) << jsonl;
    EXPECT_EQ(jsonl.find("counts_unknown"), std::string::npos) << jsonl;
}

TEST_F(ScopeStatsOrdinaryClose, UnknownCompletionExportsOnlyHostOwnedRecordsAndJoinsReaders) {
    collect_published();
    record("held.cpp");
    expect_held();
    close(/*complete=*/false, /*stopped=*/false);
    EXPECT_FALSE(runner_.collector().collector_threads_started());
    const auto jsonl = read_file(artifact());
    EXPECT_NE(jsonl.find("\"site\": \"published.cpp:11\""), std::string::npos) << jsonl;
    EXPECT_EQ(jsonl.find("held.cpp"), std::string::npos) << jsonl;
    EXPECT_NE(jsonl.find("\"run_epoch\": 8200"), std::string::npos) << jsonl;
    EXPECT_NE(jsonl.find("\"phase\": \"begin\""), std::string::npos) << jsonl;
    EXPECT_EQ(jsonl.find("\"phase\": \"end\""), std::string::npos) << jsonl;
    expect_unknown(jsonl);
}

TEST_F(ScopeStatsOrdinaryClose, UnknownCloseDoesNotReadDeviceAccountingAfterHostReadersJoin) {
    collect_published();
    record("held.cpp");
    runner_.collector().stop(false);
    refused_source.store(shm_);
    close(/*complete=*/false, /*stopped=*/false);
    EXPECT_EQ(refused_copies.load(), 0u) << "unknown boundary requested a terminal shared-memory copy";
    const auto jsonl = read_file(artifact());
    EXPECT_NE(jsonl.find("published.cpp"), std::string::npos) << jsonl;
    EXPECT_EQ(jsonl.find("held.cpp"), std::string::npos) << jsonl;
    expect_unknown(jsonl);
}

TEST_F(ScopeStatsOrdinaryClose, UnknownCloseUsesCapacitiesSavedAtAdmission) {
    collect_published();
    capacities_.values->task_window[0] = 999;
    capacities_.values->heap[0] = 999;
    capacities_.values->dep_pool[0] = 999;
    capacities_.values->tensormap = 999;
    auto *header = get_scope_stats_header(shm_);
    header->task_window_cap[0] = 999;
    header->heap_cap[0] = 999;
    header->dep_pool_cap[0] = 999;
    header->tensormap_cap = 999;
    close(/*complete=*/false, /*stopped=*/false);
    const auto jsonl = read_file(artifact());
    EXPECT_NE(jsonl.find("\"task_window_max\": [23,"), std::string::npos) << jsonl;
    EXPECT_NE(jsonl.find("\"heap_max\": [8192,"), std::string::npos) << jsonl;
    EXPECT_NE(jsonl.find("\"dep_pool_max\": [31,"), std::string::npos) << jsonl;
    EXPECT_NE(jsonl.find("\"tensormap_max\": 47"), std::string::npos) << jsonl;
    expect_unknown(jsonl);
}

TEST_F(ScopeStatsOrdinaryClose, UnknownCloseDoesNotReadAnUnpublishedHeldBuffer) {
    collect_published();
    record("held.cpp");
    expect_held();
    runner_.collector().stop(false);
    refused_source.store(reinterpret_cast<const void *>(state_->current_buf_ptr));
    close(/*complete=*/false, /*stopped=*/false);
    EXPECT_EQ(refused_copies.load(), 0u) << "unknown boundary requested an unpublished buffer copy";
    const auto jsonl = read_file(artifact());
    EXPECT_NE(jsonl.find("published.cpp"), std::string::npos) << jsonl;
    EXPECT_EQ(jsonl.find("held.cpp"), std::string::npos) << jsonl;
    expect_unknown(jsonl);
}

TEST_F(ScopeStatsOrdinaryClose, UnknownCloseWithoutSavedCapacitiesMarksThemUnknown) {
    runner_.collector().begin_run(kRunEpoch);
    collect_published();
    close(/*complete=*/false, /*stopped=*/false);
    const auto jsonl = read_file(artifact());
    EXPECT_NE(jsonl.find("\"task_window_max\": null"), std::string::npos) << jsonl;
    EXPECT_NE(jsonl.find("\"heap_max\": null"), std::string::npos) << jsonl;
    EXPECT_NE(jsonl.find("\"dep_pool_max\": null"), std::string::npos) << jsonl;
    EXPECT_NE(jsonl.find("\"tensormap_max\": null"), std::string::npos) << jsonl;
    expect_unknown(jsonl);
}

TEST_F(ScopeStatsOrdinaryClose, UnknownCloseWithoutHostRecordsEmitsNoArtifact) {
    record("held.cpp");
    expect_held();
    close(/*complete=*/false, /*stopped=*/false);
    EXPECT_FALSE(runner_.collector().collector_threads_started());
    EXPECT_FALSE(fs::exists(artifact()));
}

TEST_F(ScopeStatsOrdinaryClose, EmptyUnknownCloseRemovesOnlyThePreviousArtifactAtTheSamePrefix) {
    record("previous.cpp");
    close(/*complete=*/true, /*stopped=*/false);
    ASSERT_TRUE(fs::exists(artifact()));
    ASSERT_NE(read_file(artifact()).find("previous.cpp"), std::string::npos);
    const auto sibling = artifact().parent_path() / "unrelated.txt";
    {
        std::ofstream out(sibling);
        out << "preserve this sibling";
        ASSERT_TRUE(out.good());
    }

    auto &collector = runner_.collector();
    collector.begin_run(kRunEpoch + 1, &capacities_);
    set_platform_run_result(0, kRunEpoch + 1);
    scope_stats_aicpu_begin_run();
    record("unpublished_next.cpp");
    ASSERT_EQ(collector.total_collected(), 0u);
    close(/*complete=*/false, /*stopped=*/false, kRunEpoch + 1);

    EXPECT_FALSE(collector.collector_threads_started());
    EXPECT_FALSE(fs::exists(artifact())) << "the preceding run's artifact still appears to describe this run";
    EXPECT_EQ(read_file(sibling), "preserve this sibling");
}

TEST_F(ScopeStatsOrdinaryClose, ProducerlessNormalRunPreservesMetadataOnlyArtifact) {
    CapacityMetadata producerless;
    producerless.required = false;
    runner_.collector().begin_run(kRunEpoch + 1, &producerless);
    close(/*complete=*/true, /*stopped=*/false, kRunEpoch + 1);
    const auto jsonl = read_file(artifact());
    EXPECT_NE(jsonl.find("\"total\": 0"), std::string::npos) << jsonl;
    EXPECT_NE(jsonl.find("\"fatal\": false"), std::string::npos) << jsonl;
    EXPECT_EQ(jsonl.find("\"site\""), std::string::npos) << jsonl;
    EXPECT_EQ(jsonl.find("counts_unknown"), std::string::npos) << jsonl;
}

TEST_F(ScopeStatsOrdinaryClose, NextCompletedRunClearsUnknownCollectionState) {
    collect_published();
    close(/*complete=*/false, /*stopped=*/false);
    expect_unknown(read_file(artifact()));
    runner_.collector().begin_run(kRunEpoch + 1, &capacities_);
    start_collector();
    set_platform_run_result(0, kRunEpoch + 1);
    scope_stats_aicpu_begin_run();
    record("next.cpp");
    close(/*complete=*/true, /*stopped=*/false, kRunEpoch + 1);
    const auto jsonl = read_file(artifact());
    EXPECT_NE(jsonl.find("\"site\": \"next.cpp:11\""), std::string::npos) << jsonl;
    EXPECT_NE(jsonl.find("\"total\": 1"), std::string::npos) << jsonl;
    EXPECT_EQ(jsonl.find("published.cpp"), std::string::npos) << jsonl;
    EXPECT_EQ(jsonl.find("counts_unknown"), std::string::npos) << jsonl;
}

TEST_F(ScopeStatsOrdinaryCloseWithGatedDrain, UnknownCloseDoesNotChasePublicationAfterItsStopRequest) {
    collect_first_publication();
    ASSERT_FALSE(HasFatalFailure());
    record("late.cpp");
    auto *header = get_scope_stats_header(shm_);
    const uint32_t head = header->queue_heads[0];
    ASSERT_EQ(head, header->queue_tails[0]);
    GatedClose closing(gate_, [this] {
        close(/*complete=*/false, /*stopped=*/false);
    });
    ASSERT_TRUE(wait_until([this] {
        return stop_requested();
    }));
    scope_stats_aicpu_flush_buffers();
    ASSERT_NE(head, header->queue_tails[0]);
    closing.join();
    EXPECT_EQ(header->queue_heads[0], head) << "unknown close chased a stop-late device publication";
    EXPECT_FALSE(runner_.collector().collector_threads_started());
    const auto jsonl = read_file(artifact());
    EXPECT_NE(jsonl.find("published.cpp"), std::string::npos) << jsonl;
    EXPECT_EQ(jsonl.find("late.cpp"), std::string::npos) << jsonl;
    expect_unknown(jsonl);
}

// These are collector restart contracts. An onboard failed fence poisons its
// runner, so they do not qualify a second healthy public hardware run.
TEST_F(ScopeStatsOrdinaryCloseWithGatedDrain, RestartDiscardsAnUnreadPublicationFromThePreviousEpoch) {
    expect_restart_discards_previous_epoch(/*capacity_required=*/true);
}

TEST_F(ScopeStatsOrdinaryCloseWithGatedDrain, ProducerlessCapacityDoesNotAdmitForeignEpochRecordsOnRestart) {
    expect_restart_discards_previous_epoch(/*capacity_required=*/false);
}

TEST_F(ScopeStatsOrdinaryCloseWithGatedDrain, DefaultStopDrainsPublicationBeforeTheDrainThreadStarts) {
    collect_first_publication();
    ASSERT_FALSE(HasFatalFailure());
    record("late.cpp");
    auto *header = get_scope_stats_header(shm_);
    const uint32_t head = header->queue_heads[0];
    ASSERT_EQ(head, header->queue_tails[0]);
    GatedClose closing(gate_, [this] {
        runner_.collector().stop();
    });
    ASSERT_TRUE(wait_until([this] {
        return stop_requested();
    }));
    scope_stats_aicpu_flush_buffers();
    ASSERT_NE(head, header->queue_tails[0]);
    closing.join();
    EXPECT_EQ(header->queue_heads[0], header->queue_tails[0]);
    EXPECT_EQ(runner_.collector().total_collected(), 2u);
    close(/*complete=*/true, /*stopped=*/false);
    const auto jsonl = read_file(artifact());
    EXPECT_NE(jsonl.find("published.cpp"), std::string::npos) << jsonl;
    EXPECT_NE(jsonl.find("late.cpp"), std::string::npos) << jsonl;
    EXPECT_NE(jsonl.find("\"total\": 2"), std::string::npos) << jsonl;
}

}  // namespace
