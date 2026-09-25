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
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "aicpu/args_dump_aicpu.h"
#include "aicpu/device_run_result_base_aicpu.h"
#include "common/args_dump.h"
#include "common/memory_barrier.h"
#include "host/args_dump_collector.h"

namespace fs = std::filesystem;

namespace {

void *attribution_alloc(size_t size) { return std::calloc(1, size); }

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
    ~OutputRoot() { fs::remove_all(base); }

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

    explicit AttributionFixture(const char *name) :
        root(name),
        tensor(kTensorElements) {
        for (size_t i = 0; i < tensor.size(); i++)
            tensor[i] = static_cast<int32_t>(i + 1);
        collector.configure_retained_runs(true, simpler::dfx::runs::kDefaultBudgetBytes);
        EXPECT_EQ(
            collector.initialize(
                /*num_dump_threads=*/1, /*device_id=*/0, DumpArgsLevel::FULL, attribution_alloc, nullptr,
                attribution_free
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

    void record_tensor(uint64_t task_id, uint32_t arg_index) {
        ArgsDumpInfo info{};
        info.task_id = task_id;
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
        ASSERT_EQ(dump_arg_record(kLane, info), 0);
    }

    /** One run that records `args` tensors and never flushes them to a queue. */
    void produce_unpublished(uint64_t epoch, int args) {
        set_platform_run_result(/*region_base=*/0, epoch);
        dump_args_init(/*num_dump_threads=*/1);
        for (int i = 0; i < args; i++)
            record_tensor(0xB00 + static_cast<uint64_t>(i), static_cast<uint32_t>(i));
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

}  // namespace

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
