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
 * What may contribute to a lane's arena acknowledgement, across retained runs.
 *
 * The device advances `published_payload_count` only for a payload it actually
 * placed in a ready queue. The host's side of that equation — the payloads it
 * has taken into its own storage plus the ones it wrote off — must therefore
 * count exactly those payloads and nothing else, because both sides are
 * monotonic for the collector's life and a single stray credit never expires:
 * one run's recovered payload would let a later run's genuinely published
 * payload be acknowledged before anyone had copied it, and the producer would
 * recycle an arena that still held it.
 *
 * A recovered buffer's records are real output and are accounted as this run's
 * own; they are not a transport acknowledgement. These cases pin that split at
 * the counter, for both a recovered payload that is kept and one that is lost.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "aicpu/args_dump_aicpu.h"
#include "aicpu/device_run_result_base_aicpu.h"
#include "common/args_dump.h"
#include "common/memory_barrier.h"
#include "host/args_dump_collector.h"
#include "support/test_task_id.h"
#include "profiler_test_allocator.h"

namespace fs = std::filesystem;

struct ArgsDumpCollectorTestPeer {
    static int transport_shards(const ArgsDumpCollector &collector) { return collector.manager_.shard_count(); }
    static size_t ordinary_shards(const ArgsDumpCollector &collector) {
        return collector.collected_by_collector_.size();
    }
    static bool wait_for_release(ArgsDumpCollector &collector, uint64_t epoch) {
        std::unique_lock lock(collector.retained_mu_);
        return collector.retained_cv_.wait_for(lock, std::chrono::seconds(5), [&] {
            return collector.retained_find_slot(epoch) < 0;
        });
    }
    static size_t charged(const ArgsDumpCollector &collector) { return collector.retained_budget_.charged(); }
    static size_t fixed(const ArgsDumpCollector &collector) { return collector.retained_budget_.fixed(); }
    static uint64_t ordinary_pending(const ArgsDumpCollector &collector) {
        return collector.ordinary_queued_payloads_.load();
    }
    // The caller has quiesced the collectors and has not closed this window.
    static std::vector<uint32_t> bucket_arguments(ArgsDumpCollector &collector, uint64_t epoch, size_t bucket) {
        std::scoped_lock lock(collector.retained_mu_);
        const int slot = collector.retained_find_slot(epoch);
        if (slot < 0) return {};
        std::vector<uint32_t> indices;
        for (const auto &arg : collector.retained_epochs_[static_cast<size_t>(slot)].buckets[bucket])
            indices.push_back(arg.arg_index);
        return indices;
    }
};

namespace {

void *attribution_alloc(size_t size) { return profiler_test_calloc(size); }

int attribution_free(void *ptr) {
    std::free(ptr);
    return 0;
}

std::thread attribution_thread_factory(std::function<void()> fn) { return std::thread(std::move(fn)); }

std::string read_file(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return {};
    return std::string{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

long manifest_number(const std::string &json, const std::string &key) {
    const std::string needle = "\"" + key + "\": ";
    const size_t at = json.find(needle);
    if (at == std::string::npos) return -1;
    return std::strtol(json.c_str() + at + needle.size(), nullptr, 10);
}

std::string manifest_string(const std::string &json, const std::string &key) {
    const std::string needle = "\"" + key + "\": \"";
    const size_t at = json.find(needle);
    if (at == std::string::npos) return {};
    const size_t begin = at + needle.size();
    const size_t end = json.find('"', begin);
    if (end == std::string::npos) return {};
    return json.substr(begin, end - begin);
}

constexpr int kLane = 0;
constexpr uint64_t kTensorElements = 64;
constexpr uint64_t kTensorBytes = kTensorElements * sizeof(int32_t);

struct OutputRoot {
    fs::path base;

    explicit OutputRoot(const char *name) {
        base = fs::temp_directory_path() /
               ("args_dump_attribution_" + std::string(name) + "_" + std::to_string(::getpid()));
        fs::remove_all(base);
        EXPECT_TRUE(fs::create_directories(base));
    }
    ~OutputRoot() {
        if (std::getenv("SIMPLER_UT_KEEP_OUTPUTS") == nullptr) fs::remove_all(base);
    }

    fs::path prefix(const char *run) const { return base / run; }
    fs::path run_dir(const char *run) const { return prefix(run) / "args_dump"; }
    fs::path manifest(const char *run) const { return run_dir(run) / "args_dump.json"; }
};

struct AttributionFixture {
    ArgsDumpCollector collector;
    OutputRoot root;
    void *shm = nullptr;
    DumpDataHeader *header = nullptr;
    std::vector<int32_t> tensor;
    int lane_count;

    explicit AttributionFixture(const char *name, bool retaining = true, int lanes = 1) :
        root(name),
        tensor(kTensorElements),
        lane_count(lanes) {
        for (size_t i = 0; i < tensor.size(); i++)
            tensor[i] = static_cast<int32_t>(i + 1);
        collector.configure_retained_runs(retaining, simpler::dfx::runs::kDefaultBudgetBytes);
        EXPECT_EQ(
            collector.initialize(
                lane_count, /*device_id=*/0, DumpArgsLevel::FULL, attribution_alloc, nullptr, attribution_free
            ),
            0
        );
        shm = collector.get_dump_shm_device_ptr();
        EXPECT_NE(shm, nullptr);
        header = get_dump_header(shm);
        collector.start(attribution_thread_factory);
        set_dump_args_enabled(true);
        set_platform_dump_base(reinterpret_cast<uint64_t>(shm));
    }

    ~AttributionFixture() {
        collector.fail_arena_copy_for_test(false);
        set_dump_args_enabled(false);
        set_platform_dump_base(0);
        set_platform_run_result(0, 0);
        collector.finish_retained_runs();
        collector.stop();
        (void)collector.finalize(nullptr, attribution_free);
    }

    DumpBufferState *state(int lane = kLane) { return get_dump_buffer_state(shm, lane); }
    uint64_t credit(int lane = kLane) const { return collector.retained_lane_transport_credit_for_test(lane); }

    bool begin(uint64_t epoch, const fs::path &prefix) {
        return collector.run_begin(epoch, prefix.string(), DumpArgsLevel::FULL);
    }
    int close(uint64_t epoch) { return collector.run_close(epoch, /*device_execution_complete=*/true); }
    bool flush(std::string *error) {
        return collector.flush_retained_runs(simpler::dfx::runs::kCutAckBudgetMs * 8, error);
    }
    ArgsDumpCollector::RetainedRunStats stats() const { return collector.retained_run_stats_for_test(); }

    void record_tensor(int32_t task_id, uint32_t arg_index, int lane = kLane) {
        ArgsDumpInfo info{};
        info.task_id = simpler::ut::test_task_id(task_id);
        info.role = ArgsDumpRole::INPUT;
        info.stage = ArgsDumpStage::BEFORE_DISPATCH;
        info.arg_index = arg_index;
        info.kind = static_cast<uint8_t>(ArgsDumpKind::TENSOR);
        info.dtype = static_cast<uint8_t>(DataType::INT32);
        info.capture_payload = 1;
        info.buffer_addr = reinterpret_cast<uint64_t>(tensor.data());
        info.func_count = 1;
        info.func_ids[0] = 5;
        info.ndims = 1;
        info.shapes[0] = static_cast<uint32_t>(kTensorElements);
        info.strides[0] = 1;
        ASSERT_EQ(dump_arg_record(lane, info), 0);
    }

    /** One run that records `args` tensors and never flushes them to a queue. */
    void produce_unpublished(uint64_t epoch, int args) {
        set_platform_run_result(/*region_base=*/0, epoch);
        dump_args_init(lane_count, DumpArgsLevel::FULL);
        for (int i = 0; i < args; i++)
            record_tensor(0xB00 + i, static_cast<uint32_t>(i));
    }

    /** One run that records `args` tensors and publishes them. */
    void produce_published(uint64_t epoch, int args) {
        produce_unpublished(epoch, args);
        dump_args_flush(kLane);
    }

    bool wait_for_collected(uint64_t records, int timeout_ms = 5000) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            if (stats().open_collected_records >= records) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return stats().open_collected_records >= records;
    }
};

void verify_destination_alias(bool symlink, bool retaining) {
    AttributionFixture fx("live-destination-alias", retaining);
    const auto owner = fx.root.prefix("owner");
    ASSERT_TRUE(fs::create_directories(fx.root.run_dir("owner")));
    fs::path alias = owner / ".";
    if (symlink) {
        alias = fx.root.prefix("alias");
        ASSERT_TRUE(fs::create_directories(alias));
        std::error_code error;
        fs::create_directory_symlink(fx.root.run_dir("owner"), alias / "args_dump", error);
        if (error == std::errc::operation_not_permitted || error == std::errc::function_not_supported)
            GTEST_SKIP() << "Directory symlinks unavailable: " << error.message();
        ASSERT_FALSE(error) << error.message();
    }
    ASSERT_TRUE(fx.collector.begin_recorded_window(4001, owner.string(), DumpArgsLevel::FULL));
    ASSERT_EQ(fx.stats().open_collected_records, 0u);
    {
        std::ofstream previous(fx.root.manifest("owner"));
        previous << "owner evidence";
    }
    if (retaining) {
        EXPECT_FALSE(fx.begin(4002, alias));
    } else {
        EXPECT_THROW(fx.collector.begin_run(alias.string(), DumpArgsLevel::FULL, 4002), std::runtime_error);
    }
    EXPECT_FALSE(fx.collector.begin_recorded_window(4002, alias.string(), DumpArgsLevel::FULL));
    EXPECT_EQ(read_file(fx.root.manifest("owner")), "owner evidence");
    EXPECT_TRUE(fx.collector.recorded_window_open(4001));
    EXPECT_EQ(fx.stats().open_epochs, 1u);

    ASSERT_TRUE(fx.collector.begin_recorded_window(4003, fx.root.prefix("distinct").string(), DumpArgsLevel::FULL));
    ASSERT_EQ(fx.collector.close_recorded_window(4003), 0);
    std::fill(fx.tensor.begin(), fx.tensor.end(), 711);
    fx.produce_published(4001, 1);
    ASSERT_EQ(fx.collector.close_recorded_window(4001), 0);
    std::string error;
    ASSERT_TRUE(fx.flush(&error)) << error;
    const auto manifest = read_file(fx.root.manifest("owner"));
    ASSERT_EQ(manifest_number(manifest, "total_args"), 1);
    const auto bytes = read_file(fx.root.run_dir("owner") / manifest_string(manifest, "bin_file"));
    ASSERT_EQ(bytes.size(), kTensorBytes);
    EXPECT_EQ(std::memcmp(bytes.data(), fx.tensor.data(), kTensorBytes), 0);
}

}  // namespace

TEST(ArgsDumpAckAttribution, DotDestinationAliasCannotReplaceARecordedOwner) { verify_destination_alias(false, false); }

TEST(ArgsDumpAckAttribution, SymlinkDestinationAliasCannotReplaceARecordedOwner) {
    verify_destination_alias(true, false);
}

TEST(ArgsDumpAckAttribution, DotDestinationAliasPreservesRetainedAdmissionRefusal) {
    verify_destination_alias(false, true);
}

TEST(ArgsDumpAckAttribution, SymlinkDestinationAliasPreservesRetainedAdmissionRefusal) {
    verify_destination_alias(true, true);
}

TEST(ArgsDumpAckAttribution, OrdinaryPayloadCreditProvesHostOwnershipBeforeDiskAndSurvivesRunReset) {
    AttributionFixture fx("ordinary-owned-before-disk", false);
    fx.collector.begin_run(fx.root.prefix("run-a").string(), DumpArgsLevel::FULL, 3201);
    fx.collector.hold_retained_writer_for_test(true);
    auto resume = [&] {
        fx.collector.hold_retained_writer_for_test(false);
    };
    struct ResumeWriter {
        std::function<void()> resume;
        ~ResumeWriter() { resume(); }
    } guard{resume};
    fx.produce_published(3201, 1);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (fx.credit() != 1 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    ASSERT_EQ(fx.credit(), 1u) << "the physical receipt must count ordinary copied bytes before disk completion";
    fx.collector.publish_arena_acks();
    ASSERT_EQ(fx.state()->completed_payload_count, 1u);
    EXPECT_FALSE(fx.collector.retains_runs());
    EXPECT_EQ(fx.stats().open_epochs, 0u);
    ASSERT_TRUE(fs::exists(fx.root.run_dir("run-a") / "args.bin"));
    EXPECT_EQ(fs::file_size(fx.root.run_dir("run-a") / "args.bin"), 0u);
    std::memset(reinterpret_cast<void *>(fx.state()->arena_base), 0xee, kTensorBytes);
    resume();
    ASSERT_EQ(fx.collector.close_ordinary_run(true), 0);
    const std::string bytes = read_file(fx.root.run_dir("run-a") / "args.bin");
    ASSERT_EQ(bytes.size(), kTensorBytes);
    EXPECT_EQ(std::memcmp(bytes.data(), fx.tensor.data(), kTensorBytes), 0);

    fx.collector.begin_run(fx.root.prefix("run-b").string(), DumpArgsLevel::FULL, 3202);
    EXPECT_EQ(fx.credit(), 1u) << "ordinary metadata reset must not reset physical arena ownership";
    fx.produce_published(3202, 1);
    ASSERT_EQ(fx.collector.close_ordinary_run(true), 0);
    EXPECT_EQ(fx.credit(), 2u);
    fx.collector.publish_arena_acks();
    EXPECT_EQ(fx.state()->completed_payload_count, 2u);
}

TEST(ArgsDumpAckAttribution, RecordedReplaysKeepEveryPayloadAcrossOrdinaryTrafficAndIndependentClose) {
    AttributionFixture fx("recorded-with-ordinary", false);
    ASSERT_TRUE(fx.collector.begin_recorded_window(3301, fx.root.prefix("model-a").string(), DumpArgsLevel::FULL));
    ASSERT_TRUE(fx.collector.begin_recorded_window(3302, fx.root.prefix("model-b").string(), DumpArgsLevel::FULL));
    EXPECT_FALSE(fx.collector.begin_recorded_window(3303, fx.root.prefix("no-slot").string(), DumpArgsLevel::FULL));
    EXPECT_FALSE(fx.collector.retains_runs());
    const size_t fixed_charge = fx.stats().charged_bytes;
    ASSERT_GT(fixed_charge, 0u);
    auto replay = [&](uint64_t epoch, int before, int after) {
        set_platform_run_result(0, epoch);
        set_dump_args_enabled(true);
        dump_args_init(1, DumpArgsLevel::FULL);
        ArgsDumpInfo info{};
        info.task_id = simpler::ut::test_task_id(0xB00);
        info.role = ArgsDumpRole::INOUT;
        info.kind = static_cast<uint8_t>(ArgsDumpKind::TENSOR);
        info.dtype = static_cast<uint8_t>(DataType::INT32);
        info.capture_payload = 1;
        info.buffer_addr = reinterpret_cast<uint64_t>(fx.tensor.data());
        info.func_count = 1;
        info.func_ids[0] = 5;
        info.ndims = 1;
        info.shapes[0] = kTensorElements;
        info.strides[0] = 1;
        std::fill(fx.tensor.begin(), fx.tensor.end(), before);
        info.stage = ArgsDumpStage::BEFORE_DISPATCH;
        EXPECT_EQ(dump_arg_record(kLane, info), 0);
        std::fill(fx.tensor.begin(), fx.tensor.end(), after);
        info.stage = ArgsDumpStage::AFTER_COMPLETION;
        EXPECT_EQ(dump_arg_record(kLane, info), 0);
        dump_args_flush(kLane);
    };
    replay(3301, 100, 200);
    replay(3302, 500, 600);
    ASSERT_TRUE(fx.wait_for_collected(4));
    fx.collector.begin_run(fx.root.prefix("ordinary").string(), DumpArgsLevel::FULL, 3304);
    fx.produce_published(3304, 1);
    ASSERT_EQ(fx.collector.close_ordinary_run(true), 0);
    EXPECT_FALSE(fx.collector.retains_runs());
    EXPECT_EQ(fx.stats().open_epochs, 2u);
    replay(3301, 101, 201);
    replay(3301, 102, 202);
    ASSERT_TRUE(fx.wait_for_collected(8));
    ASSERT_EQ(fx.collector.close_recorded_window(3301), 0);
    std::string error;
    ASSERT_TRUE(fx.flush(&error)) << error;
    EXPECT_TRUE(fx.collector.recorded_window_open(3302));
    replay(3302, 501, 601);
    ASSERT_TRUE(fx.wait_for_collected(4));
    ASSERT_EQ(fx.collector.close_recorded_window(3302), 0);
    ASSERT_TRUE(fx.flush(&error)) << error;
    for (const char *name : {"model-a", "model-b"}) {
        const std::string manifest = read_file(fx.root.manifest(name));
        EXPECT_NE(manifest.find("\"recorded_window\": true"), std::string::npos);
        EXPECT_NE(manifest.find("\"dropped_records\": null"), std::string::npos);
        EXPECT_EQ(manifest_string(manifest, "collection_verdict"), "aggregate_counts_unknown");
        EXPECT_EQ(manifest_number(manifest, "host_discarded_args"), 0);
        EXPECT_EQ(manifest_number(manifest, "metadata_discarded_records"), 0);
        const std::string bytes = read_file(fx.root.run_dir(name) / manifest_string(manifest, "bin_file"));
        const std::vector<int32_t> expected = std::string(name) == "model-a" ?
                                                  std::vector<int32_t>{100, 200, 101, 201, 102, 202} :
                                                  std::vector<int32_t>{500, 600, 501, 601};
        ASSERT_EQ(bytes.size(), expected.size() * kTensorBytes);
        EXPECT_EQ(manifest_number(manifest, "total_args"), static_cast<long>(expected.size()));
        std::set<size_t> observed_offsets;
        std::istringstream lines(manifest);
        std::string entry;
        while (std::getline(lines, entry)) {
            if (entry.find("\"run_epoch\":") == std::string::npos) continue;
            const long offset = manifest_number(entry, "bin_offset");
            ASSERT_GE(offset, 0);
            ASSERT_EQ(static_cast<size_t>(offset) % kTensorBytes, 0u);
            ASSERT_EQ(manifest_number(entry, "bin_size"), static_cast<long>(kTensorBytes));
            const size_t sample = static_cast<size_t>(offset) / kTensorBytes;
            ASSERT_LT(sample, expected.size());
            EXPECT_TRUE(observed_offsets.insert(static_cast<size_t>(offset)).second);
            EXPECT_EQ(manifest_string(entry, "stage"), sample % 2 == 0 ? "before_dispatch" : "after_completion");
            std::vector<int32_t> expected_tensor(kTensorElements, expected[sample]);
            EXPECT_EQ(std::memcmp(bytes.data() + offset, expected_tensor.data(), kTensorBytes), 0) << entry;
        }
        EXPECT_EQ(observed_offsets.size(), expected.size());
    }
    EXPECT_EQ(fx.credit(), 11u);
    EXPECT_EQ(fx.stats().errors.recorded_aggregate, 2u);
    EXPECT_EQ(fx.stats().open_epochs, 0u);
    EXPECT_EQ(fx.stats().charged_bytes, fixed_charge);
}

TEST(ArgsDumpAckAttribution, OrdinaryExportPreservesEachRecordedTransportShard) {
    AttributionFixture fx("recorded-two-shards-after-ordinary", false, 2);
    ASSERT_TRUE(fx.collector.begin_recorded_window(3311, fx.root.prefix("recorded").string(), DumpArgsLevel::FULL));
    fx.collector.begin_run(fx.root.prefix("ordinary").string(), DumpArgsLevel::FULL, 3312);
    fx.produce_published(3312, 1);
    ASSERT_EQ(fx.collector.close_ordinary_run(true), 0);
    ASSERT_EQ(ArgsDumpCollectorTestPeer::ordinary_shards(fx.collector), 0u);
    ASSERT_EQ(ArgsDumpCollectorTestPeer::transport_shards(fx.collector), 2);

    set_platform_run_result(0, 3311);
    dump_args_init(2, DumpArgsLevel::FULL);
    for (int lane : {1, 0}) {
        std::fill(fx.tensor.begin(), fx.tensor.end(), 110 + lane * 101);
        fx.record_tensor(0xB00 + lane, static_cast<uint32_t>(lane), lane);
        dump_args_flush(lane);
        fx.collector.quiesce();
        EXPECT_EQ(
            ArgsDumpCollectorTestPeer::bucket_arguments(fx.collector, 3311, static_cast<size_t>(lane)),
            std::vector<uint32_t>({static_cast<uint32_t>(lane)})
        );
        EXPECT_EQ(fx.credit(lane), lane == 0 ? 2u : 1u);
    }
    EXPECT_EQ(ArgsDumpCollectorTestPeer::bucket_arguments(fx.collector, 3311, 0), std::vector<uint32_t>({0}));
    EXPECT_EQ(ArgsDumpCollectorTestPeer::bucket_arguments(fx.collector, 3311, 1), std::vector<uint32_t>({1}));
    ASSERT_EQ(fx.collector.close_recorded_window(3311), 0);
    std::string error;
    ASSERT_TRUE(fx.flush(&error)) << error;
    const auto manifest = read_file(fx.root.manifest("recorded"));
    EXPECT_EQ(manifest_number(manifest, "total_args"), 2);
    const auto bytes = read_file(fx.root.run_dir("recorded") / manifest_string(manifest, "bin_file"));
    ASSERT_EQ(bytes.size(), 2 * kTensorBytes);
    for (size_t sample = 0; sample < 2; ++sample) {
        const std::vector<int32_t> expected(kTensorElements, sample == 0 ? 211 : 110);
        EXPECT_EQ(std::memcmp(bytes.data() + sample * kTensorBytes, expected.data(), kTensorBytes), 0);
    }
}

TEST(ArgsDumpAckAttribution, RecordedCollectionKeepsOrdinaryRetainedAccountingStrict) {
    AttributionFixture fx("recorded-with-retaining-ordinary");
    ASSERT_TRUE(fx.collector.begin_recorded_window(3401, fx.root.prefix("model").string(), DumpArgsLevel::FULL));
    fx.produce_published(3401, 1);
    ASSERT_TRUE(fx.wait_for_collected(1));
    ASSERT_TRUE(fx.begin(3402, fx.root.prefix("ordinary")));
    fx.produce_published(3402, 1);
    ASSERT_EQ(fx.close(3402), 0);
    std::string error;
    ASSERT_TRUE(fx.flush(&error)) << error;
    EXPECT_EQ(manifest_string(read_file(fx.root.manifest("ordinary")), "collection_verdict"), "published");
    EXPECT_TRUE(fx.collector.retains_runs());
    EXPECT_TRUE(fx.collector.recorded_window_open(3401));
    fx.produce_published(3401, 1);
    ASSERT_TRUE(fx.wait_for_collected(2));
    ASSERT_EQ(fx.collector.close_recorded_window(3401), 0);
    ASSERT_TRUE(fx.flush(&error)) << error;
    EXPECT_EQ(manifest_number(read_file(fx.root.manifest("model")), "total_args"), 2);
    EXPECT_EQ(fx.stats().errors.published, 1u);
    EXPECT_EQ(fx.stats().errors.recorded_aggregate, 1u);
    EXPECT_EQ(fx.credit(), 3u);
}

TEST(ArgsDumpAckAttribution, RecordedPayloadBudgetLossAcknowledgesDiscardButFailsDiagnosticFlush) {
    AttributionFixture fx("recorded-payload-pressure", false);
    fx.collector.configure_retained_runs(false, 17 * 1024 * 1024);
    ASSERT_TRUE(fx.collector.begin_recorded_window(3501, fx.root.prefix("model").string(), DumpArgsLevel::FULL));
    fx.tensor.assign(8 * 1024 * 1024, 42);
    set_platform_run_result(0, 3501);
    dump_args_init(1, DumpArgsLevel::FULL);
    ArgsDumpInfo info{};
    info.task_id = simpler::ut::test_task_id(0xB00);
    info.role = ArgsDumpRole::INPUT;
    info.stage = ArgsDumpStage::BEFORE_DISPATCH;
    info.kind = static_cast<uint8_t>(ArgsDumpKind::TENSOR);
    info.dtype = static_cast<uint8_t>(DataType::INT32);
    info.capture_payload = 1;
    info.buffer_addr = reinterpret_cast<uint64_t>(fx.tensor.data());
    info.ndims = 1;
    info.shapes[0] = fx.tensor.size();
    info.strides[0] = 1;
    ASSERT_EQ(dump_arg_record(kLane, info), 0);
    dump_args_flush(kLane);
    ASSERT_TRUE(fx.wait_for_collected(1));
    EXPECT_EQ(fx.stats().open_discarded_args, 1u);
    EXPECT_EQ(fx.credit(), 1u);
    fx.collector.publish_arena_acks();
    EXPECT_EQ(fx.state()->completed_payload_count, 1u);
    ASSERT_EQ(fx.collector.close_recorded_window(3501), 0);
    std::string error;
    EXPECT_FALSE(fx.flush(&error));
    EXPECT_EQ(fx.stats().errors.published_short, 1u);
    EXPECT_EQ(manifest_number(read_file(fx.root.manifest("model")), "host_discarded_args"), 1);
}

TEST(ArgsDumpAckAttribution, RecordedPublicationFailureDoesNotReopenSettledDeviceOwnership) {
    AttributionFixture fx("recorded-manifest-failure", false);
    ASSERT_TRUE(fx.collector.begin_recorded_window(3601, fx.root.prefix("model").string(), DumpArgsLevel::FULL));
    fx.produce_published(3601, 1);
    ASSERT_TRUE(fx.wait_for_collected(1));
    ASSERT_TRUE(fs::create_directory(fx.root.manifest("model")));
    ASSERT_EQ(fx.collector.close_recorded_window(3601), 0);
    std::string error;
    EXPECT_FALSE(fx.flush(&error));
    EXPECT_EQ(fx.stats().errors.write_failed, 1u);
    EXPECT_EQ(fx.stats().open_epochs, 0u);
    EXPECT_FALSE(fx.collector.recorded_window_open(3601));
}

TEST(ArgsDumpAckAttribution, RecordedTransportLossSurvivesOrdinaryReconciliation) {
    AttributionFixture fx("recorded-malformed-handoff", false);
    ASSERT_TRUE(fx.collector.begin_recorded_window(3701, fx.root.prefix("model").string(), DumpArgsLevel::FULL));
    fx.collector.stop(false);
    fx.produce_published(3701, 1);
    const uint32_t head = fx.header->queue_heads[kLane];
    ASSERT_NE(head, fx.header->queue_tails[kLane]);
    ASSERT_NE(fx.header->queues[kLane][head].buffer_ptr, 0u);
    fx.header->queues[kLane][head].thread_index = UINT32_MAX;
    fx.collector.start(attribution_thread_factory);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (fx.collector.drain_dropped_buffers() == 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    ASSERT_EQ(fx.collector.drain_dropped_buffers(), 1u);
    EXPECT_EQ(fx.credit(), 0u) << "an unresolvable payload has no host-owned bytes to acknowledge";
    fx.collector.begin_run(fx.root.prefix("ordinary").string(), DumpArgsLevel::FULL, 3702);
    set_platform_run_result(0, 3702);
    dump_args_init(1, DumpArgsLevel::FULL);
    EXPECT_EQ(fx.collector.close_ordinary_run(true), 0);
    ASSERT_EQ(fx.collector.close_recorded_window(3701), 0);
    std::string error;
    EXPECT_FALSE(fx.flush(&error));
    EXPECT_EQ(fx.stats().errors.counts_unknown, 1u);
    EXPECT_EQ(fx.stats().errors.recorded_aggregate, 0u);
    EXPECT_EQ(manifest_string(read_file(fx.root.manifest("model")), "collection_verdict"), "counts_unknown");
}

TEST(ArgsDumpAckAttribution, OrdinaryWriterFailureRemainsVisibleAfterItsSinkCloses) {
    AttributionFixture fx("ordinary-write-failure", false);
    fx.collector.begin_run(fx.root.prefix("ordinary").string(), DumpArgsLevel::FULL, 3801);
    ASSERT_TRUE(fs::create_directories(fx.root.run_dir("ordinary")));
    if (fs::exists("/dev/full")) fs::create_symlink("/dev/full", fx.root.run_dir("ordinary") / "args.bin");
    else ASSERT_TRUE(fs::create_directory(fx.root.run_dir("ordinary") / "args.bin"));
    fx.produce_published(3801, 1);
    EXPECT_NE(fx.collector.close_ordinary_run(true), 0);
    EXPECT_EQ(fx.credit(), 1u) << "the copied payload remains host-owned even when its file failed";
    std::string error;
    EXPECT_FALSE(fx.flush(&error)) << "closing an ordinary sink must not erase its writer failure";
}

TEST(ArgsDumpAckAttribution, RecordedZeroWindowPublishesFreshUnknownCountsAndOrdinaryReuseClearsTheMarker) {
    AttributionFixture fx("zero-window-destination-reuse", false);
    ASSERT_TRUE(fs::create_directories(fx.root.run_dir("shared")));
    {
        std::ofstream old(fx.root.manifest("shared"));
        old << "old successful artifact";
    }
    ASSERT_TRUE(fx.collector.begin_recorded_window(3901, fx.root.prefix("shared").string(), DumpArgsLevel::FULL));
    EXPECT_EQ(read_file(fx.root.manifest("shared")), "old successful artifact");
    ASSERT_EQ(fx.collector.close_recorded_window(3901), 0);
    std::string error;
    ASSERT_TRUE(fx.flush(&error)) << error;
    const std::string manifest = read_file(fx.root.manifest("shared"));
    EXPECT_EQ(manifest_number(manifest, "total_args"), 0);
    EXPECT_NE(manifest.find("\"bin_file\": null"), std::string::npos);
    EXPECT_NE(manifest.find("\"dropped_records\": null"), std::string::npos);
    EXPECT_EQ(manifest_string(manifest, "collection_verdict"), "aggregate_counts_unknown");
    fx.collector.begin_run(fx.root.prefix("shared").string(), DumpArgsLevel::PARTIAL, 3902);
    ASSERT_EQ(fx.collector.close_ordinary_run(true), 0);
    EXPECT_FALSE(fs::exists(fx.root.manifest("shared")));
}

/**
 * A recovered payload is output, not an acknowledgement — so the successor's
 * own published payload is still acknowledged on its own terms.
 *
 * Run N publishes nothing and has one tensor recovered from the buffer the
 * device never handed over: the device's `published_payload_count` stays at 0,
 * so the host's credit for that lane must stay at 0 too. Run N+1 then publishes
 * one tensor. With the recovered payload credited, the equation would already
 * read 1 >= 1 and `completed_payload_count` would advance before anyone had
 * copied N+1's bytes, leaving the producer free to recycle an arena still
 * holding them. It advances only once N+1's own payload is host-owned.
 */
TEST(ArgsDumpAckAttribution, ARecoveredPayloadDoesNotAcknowledgeASuccessorsUnreadPayload) {
    AttributionFixture fx("recovered-vs-published");

    // Run N: recorded, never published.
    ASSERT_TRUE(fx.begin(3001, fx.root.prefix("run-a")));
    fx.produce_unpublished(3001, /*args=*/1);
    ASSERT_NE(fx.state()->current_buf_ptr, 0u);
    ASSERT_EQ(fx.state()->published_payload_count, 0u) << "the fixture published something it should not have";
    ASSERT_EQ(fx.close(3001), 0);

    // The record is output: it is in this run's manifest with its bytes.
    std::string error;
    ASSERT_TRUE(fx.flush(&error)) << error;
    const std::string manifest_a = read_file(fx.root.manifest("run-a"));
    ASSERT_FALSE(manifest_a.empty());
    EXPECT_EQ(manifest_number(manifest_a, "total_args"), 1);
    EXPECT_EQ(manifest_string(manifest_a, "collection_verdict"), "published");
    EXPECT_EQ(fs::file_size(fx.root.run_dir("run-a") / manifest_string(manifest_a, "bin_file")), kTensorBytes);

    // And it is not an acknowledgement: the device never published it, so this
    // lane's credit is still zero.
    EXPECT_EQ(fx.state()->published_payload_count, 0u);
    EXPECT_EQ(fx.credit(), 0u) << "a recovered payload was credited to the lane's transport acknowledgement";
    fx.collector.publish_arena_acks();
    EXPECT_EQ(fx.state()->completed_payload_count, 0u);

    // Run N+1 publishes one payload. The device's count moves first, so this is
    // the instant a stray credit from N would pay for: the equation must still
    // be waiting for N+1's own bytes.
    ASSERT_TRUE(fx.begin(3002, fx.root.prefix("run-b")));
    fx.produce_published(3002, /*args=*/1);
    ASSERT_EQ(fx.state()->published_payload_count, 1u);

    // The only way `completed` may reach 1 is N+1's own payload being taken.
    ASSERT_TRUE(fx.wait_for_collected(1)) << "run N+1's payload was never received";
    EXPECT_EQ(fx.credit(), 1u) << "exactly one published payload has been accounted for";
    fx.collector.publish_arena_acks();
    EXPECT_EQ(fx.state()->completed_payload_count, 1u);

    ASSERT_EQ(fx.close(3002), 0);
    error.clear();
    ASSERT_TRUE(fx.flush(&error)) << error;
    const std::string manifest_b = read_file(fx.root.manifest("run-b"));
    ASSERT_FALSE(manifest_b.empty());
    EXPECT_EQ(manifest_number(manifest_b, "total_args"), 1);
    EXPECT_EQ(fs::file_size(fx.root.run_dir("run-b") / manifest_string(manifest_b, "bin_file")), kTensorBytes);

    const auto st = fx.stats();
    EXPECT_EQ(st.errors.published, 2u);
    EXPECT_FALSE(st.fatal);
}

/**
 * A recovered payload that is *lost* is not an acknowledgement either.
 *
 * The discard paths settle a lane so a published payload's arena is not held
 * behind bytes the host deliberately did not take. A recovered payload was
 * never in that equation, so losing one must credit nothing — the run reports
 * the loss and fails its flush, and the lane's credit stays where the device
 * left it.
 */
TEST(ArgsDumpAckAttribution, ALostRecoveredPayloadCreditsNothingToTheLane) {
    AttributionFixture fx("recovered-discard");
    // Makes the recovery's own arena read fail, so its payload is lost rather
    // than taken — the discard side of the same attribution rule.
    fx.collector.fail_arena_copy_for_test(true);

    ASSERT_TRUE(fx.begin(3101, fx.root.prefix("run-a")));
    fx.produce_unpublished(3101, /*args=*/1);
    ASSERT_NE(fx.state()->current_buf_ptr, 0u);
    ASSERT_EQ(fx.state()->published_payload_count, 0u);
    ASSERT_EQ(fx.close(3101), 0);

    std::string error;
    EXPECT_FALSE(fx.flush(&error)) << "a lost payload must fail the flush";

    // The loss is this run's: metadata kept, marked, counted.
    const std::string manifest = read_file(fx.root.manifest("run-a"));
    ASSERT_FALSE(manifest.empty());
    EXPECT_EQ(manifest_number(manifest, "total_args"), 1);
    EXPECT_EQ(manifest_number(manifest, "host_discarded_args"), 1);
    EXPECT_NE(manifest.find("\"host_discarded\": true"), std::string::npos);
    EXPECT_EQ(manifest_string(manifest, "collection_verdict"), "published_short");

    // And it is not an acknowledgement.
    EXPECT_EQ(fx.state()->published_payload_count, 0u);
    EXPECT_EQ(fx.credit(), 0u) << "a lost recovered payload was credited to the lane's acknowledgement";
    fx.collector.publish_arena_acks();
    EXPECT_EQ(fx.state()->completed_payload_count, 0u);

    // A successor's published payload still settles on its own disposition.
    fx.collector.fail_arena_copy_for_test(false);
    ASSERT_TRUE(fx.begin(3102, fx.root.prefix("run-b")));
    fx.produce_published(3102, /*args=*/1);
    ASSERT_EQ(fx.state()->published_payload_count, 1u);
    ASSERT_TRUE(fx.wait_for_collected(1));
    EXPECT_EQ(fx.credit(), 1u);
    fx.collector.publish_arena_acks();
    EXPECT_EQ(fx.state()->completed_payload_count, 1u);
    ASSERT_EQ(fx.close(3102), 0);
    error.clear();
    // Sticky: run A's loss is still reported by every later flush.
    EXPECT_FALSE(fx.flush(&error));
    EXPECT_EQ(fx.stats().errors.published_short, 1u);
    EXPECT_EQ(fx.stats().errors.published, 1u);
}

TEST(ArgsDumpAckAttribution, OrdinaryPayloadPressureDiscardsBeforeAcknowledgingWithoutUnboundedCopies) {
    AttributionFixture fx("ordinary-payload-pressure", false);
    fx.collector.configure_retained_runs(false, 17 * 1024 * 1024);
    fx.collector.begin_run(fx.root.prefix("ordinary").string(), DumpArgsLevel::FULL, 9901);
    fx.collector.hold_retained_writer_for_test(true);
    auto resume = [&] {
        fx.collector.hold_retained_writer_for_test(false);
    };
    struct ResumeWriter {
        std::function<void()> resume;
        ~ResumeWriter() { resume(); }
    } guard{resume};
    fx.tensor.assign(2 * 1024 * 1024, 42);
    set_platform_run_result(0, 9901);
    dump_args_init(1, DumpArgsLevel::FULL);
    ArgsDumpInfo info{};
    info.task_id = simpler::ut::test_task_id(0xB00);
    info.role = ArgsDumpRole::INPUT;
    info.stage = ArgsDumpStage::BEFORE_DISPATCH;
    info.kind = static_cast<uint8_t>(ArgsDumpKind::TENSOR);
    info.dtype = static_cast<uint8_t>(DataType::INT32);
    info.capture_payload = 1;
    info.buffer_addr = reinterpret_cast<uint64_t>(fx.tensor.data());
    info.ndims = 1;
    info.shapes[0] = fx.tensor.size();
    info.strides[0] = 1;
    for (uint32_t index = 0; index < 3; ++index) {
        info.arg_index = index;
        ASSERT_EQ(dump_arg_record(kLane, info), 0);
    }
    dump_args_flush(kLane);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (fx.credit() != 3 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    ASSERT_EQ(fx.credit(), 3u);
    EXPECT_GT(ArgsDumpCollectorTestPeer::charged(fx.collector), 16 * 1024 * 1024u);
    EXPECT_LE(ArgsDumpCollectorTestPeer::charged(fx.collector), 17 * 1024 * 1024u);
    EXPECT_EQ(ArgsDumpCollectorTestPeer::ordinary_pending(fx.collector), 2u);
    EXPECT_GT(fx.stats().budget_refusals, 0u);
    EXPECT_EQ(fx.credit(), 3u);
    fx.collector.publish_arena_acks();
    EXPECT_EQ(fx.state()->completed_payload_count, 3u);
    resume();
    EXPECT_NE(fx.collector.close_ordinary_run(true), 0);
    const auto manifest = read_file(fx.root.manifest("ordinary"));
    EXPECT_EQ(manifest_number(manifest, "host_discarded_args"), 1);
    EXPECT_NE(manifest.find("\"host_discarded\": true"), std::string::npos);
    EXPECT_EQ(ArgsDumpCollectorTestPeer::charged(fx.collector), ArgsDumpCollectorTestPeer::fixed(fx.collector));
    EXPECT_EQ(manifest_number(manifest, "total_args"), 3);
    const auto bytes = read_file(fx.root.run_dir("ordinary") / "args.bin");
    ASSERT_EQ(bytes.size(), 16 * 1024 * 1024u);
    EXPECT_EQ(std::memcmp(bytes.data(), fx.tensor.data(), fx.tensor.size() * sizeof(int32_t)), 0);
    EXPECT_EQ(std::memcmp(bytes.data() + 8 * 1024 * 1024, fx.tensor.data(), fx.tensor.size() * sizeof(int32_t)), 0);
}

TEST(ArgsDumpAckAttribution, RecordedTruncationDoesNotPolluteTheOrdinarySuccessor) {
    AttributionFixture fx("recorded-truncation", false);
    ASSERT_TRUE(fx.collector.begin_recorded_window(9911, fx.root.prefix("model").string(), DumpArgsLevel::FULL));
    fx.collector.begin_run(fx.root.prefix("ordinary").string(), DumpArgsLevel::FULL, 9912);
    fx.collector.stop(false);
    fx.produce_published(9911, 1);
    const auto &entry = fx.header->queues[kLane][fx.header->queue_heads[kLane]];
    reinterpret_cast<DumpMetaBuffer *>(entry.buffer_ptr)->records[0].truncated = 1;
    fx.collector.start(attribution_thread_factory);
    ASSERT_TRUE(fx.wait_for_collected(1));
    fx.produce_published(9912, 1);
    ASSERT_EQ(fx.collector.close_ordinary_run(true), 0);
    EXPECT_EQ(manifest_number(read_file(fx.root.manifest("ordinary")), "truncated_args"), 0);
    ASSERT_EQ(fx.collector.close_recorded_window(9911), 0);
    std::string error;
    (void)fx.flush(&error);
    EXPECT_EQ(manifest_number(read_file(fx.root.manifest("model")), "truncated_args"), 1);
}

TEST(ArgsDumpAckAttribution, RefusedRecordedAdmissionPreservesThePreviousPublishedManifest) {
    AttributionFixture fx("preserve-published-on-refusal", false);
    fs::create_directories(fx.root.run_dir("model"));
    {
        std::ofstream old(fx.root.manifest("model"));
        old << "previous published manifest";
    }
    fx.collector.stop(false);
    EXPECT_FALSE(fx.collector.begin_recorded_window(9921, fx.root.prefix("model").string(), DumpArgsLevel::FULL));
    EXPECT_EQ(read_file(fx.root.manifest("model")), "previous published manifest");
}

TEST(ArgsDumpAckAttribution, UnrelatedAdmissionFailureDoesNotPinAStoppedRecordedModel) {
    AttributionFixture fx("unrelated-admission-fatal", false);
    ASSERT_TRUE(fx.collector.begin_recorded_window(9971, fx.root.prefix("model").string(), DumpArgsLevel::FULL));
    fx.produce_published(9971, 1);
    ASSERT_TRUE(fx.wait_for_collected(1));
    fx.collector.stop(false);
    ASSERT_FALSE(fx.collector.begin_recorded_window(9972, fx.root.prefix("refused").string(), DumpArgsLevel::FULL));
    ASSERT_TRUE(fx.stats().fatal);
    fx.collector.start(attribution_thread_factory);
    EXPECT_EQ(fx.collector.close_recorded_window(9971), 0);
    EXPECT_TRUE(ArgsDumpCollectorTestPeer::wait_for_release(fx.collector, 9971));
    EXPECT_FALSE(fx.collector.recorded_window_open(9971));
    EXPECT_EQ(manifest_number(read_file(fx.root.manifest("model")), "total_args"), 1);
    std::string error;
    EXPECT_FALSE(fx.flush(&error));
}
