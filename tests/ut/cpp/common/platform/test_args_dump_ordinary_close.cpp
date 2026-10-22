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

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>

#include "aicpu/args_dump_aicpu.h"
#include "aicpu/device_run_result_base_aicpu.h"
#include "device_runner_base.h"
#include "support/test_task_id.h"

namespace fs = std::filesystem;

namespace profiling_common {

struct ProfilerBaseTestPeer {
    template <typename Collector>
    static bool running(const Collector &collector) {
        return collector.mgmt_running_.load(std::memory_order_acquire);
    }
    template <typename Collector>
    static void track_processed(Collector &collector) {
        // Ordinary production leaves these progress-observation counters off.
        collector.set_run_counters(true);
    }
    template <typename Collector>
    static uint64_t processed(const Collector &collector) {
        return collector.ring_processed_[0].load(std::memory_order_acquire);
    }
    template <typename Collector>
    static void retire_unreadable_buffer(Collector &collector) {
        collector.note_handoff_retired({});
    }
};

}  // namespace profiling_common

namespace {

constexpr uint64_t kRunEpoch = 8100;
constexpr uint32_t kPublishedValue = 0x12345678;
constexpr uint32_t kHeldValue = 0xABCDEF01;

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

// Releases the drain before joining even when a fatal test assertion returns.
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

// The runner owns its production collector; only initialization and inspection
// are exposed here. The boundary decision is SimDeviceRunnerBase's own.
class OrdinaryCloseRunner : public SimDeviceRunnerBase {
public:
    ArgsDumpCollector &collector() { return dump_collector_; }
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

class ArgsDumpOrdinaryClose : public ::testing::Test {
protected:
    void SetUp() override {
        root_ = fs::temp_directory_path() / ("args_dump_ordinary_close_" + std::to_string(::getpid()) + "_" +
                                             ::testing::UnitTest::GetInstance()->current_test_info()->name());
        fs::remove_all(root_);
        ASSERT_TRUE(fs::create_directories(root_));
        ASSERT_EQ(runner_.attach(), 0);
        auto &collector = runner_.collector();
        ASSERT_FALSE(collector.retains_runs());
        ASSERT_EQ(collector.initialize(1, 0, DumpArgsLevel::FULL, close_alloc, nullptr, close_free), 0);
        collector.begin_run(root_.string(), DumpArgsLevel::FULL, kRunEpoch);
        start_collector();
        ASSERT_TRUE(collector.collector_threads_started());

        shm_ = collector.get_dump_shm_device_ptr();
        ASSERT_NE(shm_, nullptr);
        state_ = get_dump_buffer_state(shm_, 0);
        set_dump_args_enabled(true);
        set_platform_dump_base(reinterpret_cast<uint64_t>(shm_));
        set_platform_run_result(0, kRunEpoch);
        dump_args_init(1, DumpArgsLevel::FULL);
    }

    void TearDown() override {
        set_dump_args_enabled(false);
        set_platform_dump_base(0);
        set_platform_run_result(0, 0);
        runner_.collector().stop();
        (void)runner_.collector().finalize(nullptr, close_free);
        fs::remove_all(root_);
    }

    virtual void start_collector() {
        runner_.collector().start([](std::function<void()> fn) {
            return std::thread(std::move(fn));
        });
    }

    void record_tensor(uint32_t value, uint32_t arg_index) {
        ArgsDumpInfo info{};
        info.task_id = simpler::ut::test_task_id(0x810);
        info.role = ArgsDumpRole::INPUT;
        info.stage = ArgsDumpStage::BEFORE_DISPATCH;
        info.arg_index = arg_index;
        info.kind = static_cast<uint8_t>(ArgsDumpKind::TENSOR);
        info.dtype = static_cast<uint8_t>(DataType::UINT32);
        info.capture_payload = 1;
        info.buffer_addr = reinterpret_cast<uint64_t>(&value);
        info.func_count = 1;
        info.func_ids[0] = 5;
        info.ndims = 1;
        info.shapes[0] = info.strides[0] = 1;
        ASSERT_EQ(dump_arg_record(0, info), 0);
    }

    int close(bool complete, bool stopped) {
        return runner_.close_args_dump_run_boundary(DfxRunConfig{}, kRunEpoch, complete, stopped);
    }

    void expect_payload(uint32_t value) {
        const std::string bytes = read_file(root_ / "args_dump" / "args.bin");
        const std::string expected(reinterpret_cast<const char *>(&value), sizeof(value));
        EXPECT_EQ(bytes, expected);
    }

    void expect_held_record() {
        ASSERT_NE(state_->current_buf_ptr, 0u);
        const auto *buffer = reinterpret_cast<const DumpMetaBuffer *>(state_->current_buf_ptr);
        ASSERT_EQ(buffer->run_epoch, kRunEpoch);
        ASSERT_EQ(buffer->count, 1u);
        ASSERT_EQ(state_->published_payload_count, 0u);
    }

    OrdinaryCloseRunner runner_;
    fs::path root_;
    void *shm_ = nullptr;
    DumpBufferState *state_ = nullptr;
};

class ArgsDumpOrdinaryCloseWithGatedDrain : public ArgsDumpOrdinaryClose {
protected:
    void start_collector() override {
        auto &collector = runner_.collector();
        profiling_common::ProfilerBaseTestPeer::track_processed(collector);
        collector.start([this, created = 0](std::function<void()> fn) mutable {
            const bool drain = created++ == 0;
            return std::thread([this, drain, fn = std::move(fn)] {
                if (drain) gate_.wait();
                fn();
            });
        });
    }

    void TearDown() override {
        gate_.release();
        ArgsDumpOrdinaryClose::TearDown();
    }

    void collect_first_publication() {
        record_tensor(kPublishedValue, 0);
        dump_args_flush(0);
        auto &collector = runner_.collector();
        auto *header = get_dump_header(collector.manager().shared_mem_host());
        DumpReadyQueueEntry entry{};
        using Algorithms = profiling_common::ProfilerAlgorithms<DumpModule>;
        ASSERT_TRUE(Algorithms::try_peek_aicpu_entry(collector.manager(), header, 0, entry, true));
        // The real drain thread is gated, so this production drain algorithm
        // has exclusive access to the device queue and its free-queue writer.
        ASSERT_EQ(
            Algorithms::process_entry(collector.manager(), header, 0, entry, nullptr, false),
            profiling_common::EntryOutcome::kDelivered
        );
        ASSERT_TRUE(wait_until([&collector] {
            return profiling_common::ProfilerBaseTestPeer::processed(collector) == 1;
        })) << "the real collector shard did not take ownership of the first publication";
    }

    bool stop_requested() { return !profiling_common::ProfilerBaseTestPeer::running(runner_.collector()); }

    DrainStartGate gate_;
};

TEST_F(ArgsDumpOrdinaryClose, CompletionRecoversAnUnpublishedHeldRecord) {
    record_tensor(kHeldValue, 1);
    expect_held_record();

    ASSERT_EQ(close(/*complete=*/true, /*stopped=*/false), 0);

    EXPECT_TRUE(runner_.collector().collector_threads_started());
    expect_payload(kHeldValue);
    EXPECT_NE(read_file(root_ / "args_dump" / "args_dump.json").find("\"arg_index\": 1"), std::string::npos);
    EXPECT_NE(
        read_file(root_ / "args_dump" / "args_dump.json").find("\"dropped_records_complete\": true"), std::string::npos
    );
}

TEST_F(ArgsDumpOrdinaryClose, ProducerStopRecoversAnUnpublishedHeldRecord) {
    record_tensor(kHeldValue, 1);
    expect_held_record();

    ASSERT_EQ(close(/*complete=*/false, /*stopped=*/true), 0);

    EXPECT_TRUE(runner_.collector().collector_threads_started());
    expect_payload(kHeldValue);
    EXPECT_NE(read_file(root_ / "args_dump" / "args_dump.json").find("\"arg_index\": 1"), std::string::npos);
    EXPECT_NE(
        read_file(root_ / "args_dump" / "args_dump.json").find("\"dropped_records_complete\": true"), std::string::npos
    );
}

TEST_F(ArgsDumpOrdinaryClose, UnknownCompletionExportsOnlyHostOwnedRecordsAndJoinsReaders) {
    record_tensor(kPublishedValue, 0);
    dump_args_flush(0);
    ASSERT_EQ(state_->published_payload_count, 1u);
    // Quiesce proves this published buffer reached the real collector shard.
    runner_.collector().quiesce();
    record_tensor(kHeldValue, 1);
    ASSERT_NE(state_->current_buf_ptr, 0u);
    ASSERT_EQ(reinterpret_cast<const DumpMetaBuffer *>(state_->current_buf_ptr)->count, 1u);
    ASSERT_EQ(state_->published_payload_count, 1u);

    ASSERT_EQ(close(/*complete=*/false, /*stopped=*/false), 0);

    EXPECT_FALSE(runner_.collector().collector_threads_started())
        << "host readers can append to shards or restart the writer after export";
    expect_payload(kPublishedValue);
    const std::string manifest = read_file(root_ / "args_dump" / "args_dump.json");
    EXPECT_NE(manifest.find("\"dropped_records_complete\": false"), std::string::npos);
    EXPECT_NE(manifest.find("\"arg_index\": 0"), std::string::npos);
    EXPECT_EQ(manifest.find("\"arg_index\": 1"), std::string::npos)
        << "an unsealed buffer was recovered without either producer proof";

    // No host reader may collect this later publication. Its bytes still live
    // in the producer's arena, so an earlier export cannot acknowledge them.
    runner_.collector().stop();
    runner_.collector().publish_arena_acks();
    ASSERT_EQ(state_->completed_payload_count, 1u);
    dump_args_flush(0);
    ASSERT_EQ(state_->published_payload_count, 2u);
    runner_.collector().publish_arena_acks();
    EXPECT_EQ(state_->completed_payload_count, 1u)
        << "a recovered write supplied live transport credit for an uncollected publication";
}

TEST_F(ArgsDumpOrdinaryCloseWithGatedDrain, UnknownCloseDoesNotChasePublicationAfterItsStopRequest) {
    collect_first_publication();
    ASSERT_FALSE(HasFatalFailure());
    record_tensor(kHeldValue, 1);
    auto *header = get_dump_header(shm_);
    const uint32_t head = header->queue_heads[0];
    ASSERT_EQ(head, header->queue_tails[0]);

    int rc = -1;
    GatedClose closing(gate_, [this, &rc] {
        rc = close(/*complete=*/false, /*stopped=*/false);
    });
    ASSERT_TRUE(wait_until([this] {
        return stop_requested();
    }));
    dump_args_flush(0);
    ASSERT_NE(head, header->queue_tails[0]);
    closing.join();

    EXPECT_EQ(rc, 0);
    EXPECT_EQ(header->queue_heads[0], head) << "unknown close chased a device publication after requesting stop";
    EXPECT_FALSE(runner_.collector().collector_threads_started());
    expect_payload(kPublishedValue);
    const std::string manifest = read_file(root_ / "args_dump" / "args_dump.json");
    EXPECT_NE(manifest.find("\"arg_index\": 0"), std::string::npos);
    EXPECT_EQ(manifest.find("\"arg_index\": 1"), std::string::npos);
    EXPECT_NE(manifest.find("\"dropped_records_complete\": false"), std::string::npos);
}

TEST_F(ArgsDumpOrdinaryClose, UnknownCloseSettlesHostDrainLossBeforeTheNextRun) {
    auto &collector = runner_.collector();
    record_tensor(kPublishedValue, 0);
    dump_args_flush(0);
    collector.quiesce();
    profiling_common::ProfilerBaseTestPeer::retire_unreadable_buffer(collector);
    ASSERT_EQ(collector.drain_dropped_buffers(), 1u);

    ASSERT_EQ(close(/*complete=*/false, /*stopped=*/false), 0);
    EXPECT_EQ(collector.drain_dropped_buffers(), 0u);

    const auto next = root_ / "next";
    collector.begin_run(next.string(), DumpArgsLevel::FULL, kRunEpoch + 1);
    start_collector();
    set_platform_run_result(0, kRunEpoch + 1);
    dump_args_init(1, DumpArgsLevel::FULL);
    record_tensor(kHeldValue, 1);
    dump_args_flush(0);
    ASSERT_EQ(runner_.close_args_dump_run_boundary(DfxRunConfig{}, kRunEpoch + 1, true, false), 0);

    const auto manifest = read_file(next / "args_dump" / "args_dump.json");
    EXPECT_NE(manifest.find("\"dropped_records_complete\": true"), std::string::npos) << manifest;
}

TEST_F(ArgsDumpOrdinaryCloseWithGatedDrain, DefaultStopDrainsPublicationBeforeTheDrainThreadStarts) {
    collect_first_publication();
    ASSERT_FALSE(HasFatalFailure());
    record_tensor(kHeldValue, 1);
    auto *header = get_dump_header(shm_);
    const uint32_t head = header->queue_heads[0];
    ASSERT_EQ(head, header->queue_tails[0]);

    GatedClose closing(gate_, [this] {
        runner_.collector().stop();
    });
    ASSERT_TRUE(wait_until([this] {
        return stop_requested();
    }));
    dump_args_flush(0);
    ASSERT_NE(head, header->queue_tails[0]);
    closing.join();

    EXPECT_EQ(header->queue_heads[0], header->queue_tails[0]);
    ASSERT_EQ(runner_.collector().export_dump_files(), 0);
    const uint32_t values[] = {kPublishedValue, kHeldValue};
    EXPECT_EQ(
        read_file(root_ / "args_dump" / "args.bin"), std::string(reinterpret_cast<const char *>(values), sizeof(values))
    );
    const std::string manifest = read_file(root_ / "args_dump" / "args_dump.json");
    EXPECT_NE(manifest.find("\"arg_index\": 0"), std::string::npos);
    EXPECT_NE(manifest.find("\"arg_index\": 1"), std::string::npos);
}

}  // namespace
