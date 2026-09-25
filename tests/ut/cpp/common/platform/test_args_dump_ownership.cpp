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
 * Who owns a retained run's argument content, and when.
 *
 * Three invariants that the run boundary, not the file, has to hold:
 *
 *  - a buffer the device published is not recovered a second time just because
 *    the host has not processed it yet — the close proves processing before it
 *    reads any receipt;
 *  - the producer's arena is released by **host ownership**, not by disk, so a
 *    stalled writer does not hold the device behind bytes the host already has;
 *  - a failed device-to-host arena copy is recorded as loss rather than
 *    exported from whatever the host shadow still held.
 *
 * Deliberately a separate target from `test_args_dump_retained_runs`: these are
 * the ownership invariants, and they are asserted through the same production
 * entry points a runner uses.
 */

#include <gtest/gtest.h>

#include <atomic>
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

void *ownership_alloc(size_t size) { return std::calloc(1, size); }

int ownership_free(void *ptr) {
    std::free(ptr);
    return 0;
}

std::thread ownership_thread_factory(std::function<void()> fn) { return std::thread(std::move(fn)); }

std::string read_file(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return {};
    return std::string{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
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

long manifest_number(const std::string &json, const std::string &key) {
    const std::string needle = "\"" + key + "\": ";
    const size_t at = json.find(needle);
    if (at == std::string::npos) return -1;
    return std::strtol(json.c_str() + at + needle.size(), nullptr, 10);
}

size_t count_occurrences(const std::string &haystack, const std::string &needle) {
    size_t n = 0;
    for (size_t at = haystack.find(needle); at != std::string::npos; at = haystack.find(needle, at + 1))
        n++;
    return n;
}

constexpr int kLane = 0;
constexpr uint64_t kTensorElements = 64;
constexpr uint64_t kTensorBytes = kTensorElements * sizeof(int32_t);

/** One output root per case, removed with the fixture. */
struct OutputRoot {
    fs::path base;

    explicit OutputRoot(const char *name) {
        base =
            fs::temp_directory_path() / ("args_dump_ownership_" + std::string(name) + "_" + std::to_string(::getpid()));
        fs::remove_all(base);
        EXPECT_TRUE(fs::create_directories(base));
    }
    ~OutputRoot() { fs::remove_all(base); }

    fs::path prefix(const char *run) const { return base / run; }
    fs::path run_dir(const char *run) const { return prefix(run) / "args_dump"; }
    fs::path manifest(const char *run) const { return run_dir(run) / "args_dump.json"; }
};

/** The collector, its threads and the AICPU producer, wired as a runner wires them. */
struct OwnershipFixture {
    ArgsDumpCollector collector;
    OutputRoot root;
    void *shm = nullptr;
    DumpDataHeader *header = nullptr;
    std::vector<int32_t> tensor;

    explicit OwnershipFixture(const char *name) :
        root(name),
        tensor(kTensorElements) {
        for (size_t i = 0; i < tensor.size(); i++)
            tensor[i] = static_cast<int32_t>(i + 1);
        collector.configure_retained_runs(true, simpler::dfx::runs::kDefaultBudgetBytes);
        EXPECT_EQ(
            collector.initialize(
                /*num_dump_threads=*/1, /*device_id=*/0, DumpArgsLevel::FULL, ownership_alloc, nullptr, ownership_free
            ),
            0
        );
        shm = collector.get_dump_shm_device_ptr();
        EXPECT_NE(shm, nullptr);
        header = get_dump_header(shm);
        collector.start(ownership_thread_factory);
        set_dump_args_enabled(true);
        set_platform_dump_base(reinterpret_cast<uint64_t>(shm));
    }

    ~OwnershipFixture() {
        collector.fail_arena_copy_for_test(false);
        set_dump_args_enabled(false);
        set_platform_dump_base(0);
        set_platform_run_result(0, 0);
        collector.finish_retained_runs();
        collector.stop();
        (void)collector.finalize(nullptr, ownership_free);
    }

    DumpBufferState *state(int lane = kLane) { return get_dump_buffer_state(shm, lane); }

    bool begin(uint64_t epoch, const fs::path &prefix) {
        return collector.run_begin(epoch, prefix.string(), DumpArgsLevel::FULL);
    }

    int close(uint64_t epoch, bool device_execution_complete = true) {
        return collector.run_close(epoch, device_execution_complete);
    }

    bool flush(std::string *error) {
        return collector.flush_retained_runs(simpler::dfx::runs::kCutAckBudgetMs * 8, error);
    }

    ArgsDumpCollector::RetainedRunStats stats() const { return collector.retained_run_stats_for_test(); }

    /** The device side of one run: init, record `args` tensors, flush. */
    void produce(uint64_t epoch, int args) {
        set_platform_run_result(/*region_base=*/0, epoch);
        dump_args_init(/*num_dump_threads=*/1);
        for (int i = 0; i < args; i++)
            record_tensor(0x900 + static_cast<uint64_t>(i), static_cast<uint32_t>(i));
        dump_args_flush(kLane);
    }

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
 * The close decides a leftover only after this run's published buffers are
 * proved processed, so a publication the host has not caught up with yet is
 * never recovered a second time.
 *
 * The close follows the device flush with no wait of the test's own, which is
 * exactly the window a capture acknowledgement alone would leave open: the
 * ledger would still read `next_expected_seq == 0`, the device would still name
 * the buffer, and the leftover decision would take a handed-over buffer for an
 * unpublished one — duplicating its records and reading a mapping a collector
 * shard is inside.
 */
TEST(ArgsDumpOwnership, ACloseProvesProcessingBeforeItDecidesALeftover) {
    OwnershipFixture fx("processing-proof");

    ASSERT_TRUE(fx.begin(2001, fx.root.prefix("run-a")));
    fx.produce(2001, /*args=*/2);
    // Deliberately no `wait_for_collected` here: the proof has to come from the
    // close itself.
    EXPECT_EQ(fx.close(2001), 0) << "the close could not prove its own run's processing";

    std::string error;
    ASSERT_TRUE(fx.flush(&error)) << error;

    const std::string manifest = read_file(fx.root.manifest("run-a"));
    ASSERT_FALSE(manifest.empty());
    // Exactly two records and two payloads: the ordinary path delivered them and
    // the leftover decision added nothing.
    EXPECT_EQ(manifest_number(manifest, "total_args"), 2);
    EXPECT_EQ(count_occurrences(manifest, "\"task_id\""), 2u);
    EXPECT_EQ(manifest_string(manifest, "collection_verdict"), "published");
    EXPECT_EQ(fs::file_size(fx.root.run_dir("run-a") / manifest_string(manifest, "bin_file")), 2 * kTensorBytes);
    // A recovery would have put its records in the close boundary's own bucket
    // and counted them; nothing was recovered because nothing was unpublished.
    const auto st = fx.stats();
    EXPECT_EQ(st.errors.published, 1u);
    EXPECT_EQ(st.errors.counts_unknown, 0u);
    EXPECT_FALSE(st.fatal);
}

/**
 * A close returns only once every buffer this run published has been received.
 *
 * The sharper form of the invariant above: with 600 records the producer fills
 * and switches its metadata buffer twice before its end-of-run flush, so three
 * buffers are published and the host has three to process. A close that took a
 * capture acknowledgement for a processing proof could return with some of them
 * still in the ring — and would then read a ledger that does not describe them.
 * After a successful close the count is exact, which is only true because the
 * proof is taken inside the execution claim.
 */
TEST(ArgsDumpOwnership, ACloseReturnsOnlyAfterEveryPublishedBufferIsReceived) {
    OwnershipFixture fx("processing-proof-multi");
    // Two switches plus the end-of-run flush.
    constexpr int kRecords = 2 * PLATFORM_DUMP_RECORDS_PER_BUFFER + 88;

    ASSERT_TRUE(fx.begin(2301, fx.root.prefix("run-a")));
    set_platform_run_result(/*region_base=*/0, 2301);
    dump_args_init(/*num_dump_threads=*/1);
    for (int i = 0; i < kRecords; i++)
        fx.record_tensor(0xA00 + static_cast<uint64_t>(i), static_cast<uint32_t>(i % 8));
    dump_args_flush(kLane);
    ASSERT_GT(fx.header->queue_tails[kLane], 2u) << "the producer did not switch buffers as this case needs";

    // No wait of the test's own: the close is what has to establish this.
    ASSERT_EQ(fx.close(2301), 0) << "the close could not prove its own run's processing";
    EXPECT_EQ(fx.stats().open_collected_records, static_cast<uint64_t>(kRecords))
        << "the close returned with published buffers still unreceived";

    std::string error;
    ASSERT_TRUE(fx.flush(&error)) << error;
    const std::string manifest = read_file(fx.root.manifest("run-a"));
    ASSERT_FALSE(manifest.empty());
    EXPECT_EQ(manifest_number(manifest, "total_args"), kRecords);
    EXPECT_EQ(count_occurrences(manifest, "\"task_id\""), static_cast<size_t>(kRecords));
    EXPECT_EQ(manifest_string(manifest, "collection_verdict"), "published");
    EXPECT_EQ(
        fs::file_size(fx.root.run_dir("run-a") / manifest_string(manifest, "bin_file")),
        static_cast<uintmax_t>(kRecords) * kTensorBytes
    );
}

/**
 * The producer's arena is released by host ownership, not by disk progress.
 *
 * With the writer held nothing has been written, so the lane's disk count is
 * zero and its payload file is empty — and the acknowledgement still reaches
 * the device, because the bytes are already in storage this host owns. A slow
 * or failing disk must not hold a producer behind bytes it has handed over.
 */
TEST(ArgsDumpOwnership, ArenaIsAcknowledgedOnHostOwnershipNotOnDiskProgress) {
    OwnershipFixture fx("arena-ack");
    // Held means this writer does nothing: no payload write, no seal.
    fx.collector.hold_retained_writer_for_test(true);

    ASSERT_TRUE(fx.begin(2101, fx.root.prefix("run-a")));
    fx.produce(2101, /*args=*/1);
    ASSERT_TRUE(fx.wait_for_collected(1)) << "the payload was never taken into host storage";

    // The device counted one published payload; the disk holds none of it.
    fx.collector.publish_arena_acks();
    DumpBufferState *lane = fx.state();
    ASSERT_EQ(lane->published_payload_count, 1u);
    EXPECT_EQ(lane->completed_payload_count, lane->published_payload_count)
        << "the arena was held behind the disk even though the payload was host-owned";

    // Nothing reached the file while the writer was held, which is what makes
    // the acknowledgement above about ownership rather than about disk.
    const fs::path payload = fx.root.run_dir("run-a") / "args.e2101.bin";
    ASSERT_TRUE(fs::exists(payload)) << "the exclusive payload name is reserved at admission";
    EXPECT_EQ(fs::file_size(payload), 0u);

    // Released, the same run then publishes its bytes and its manifest.
    fx.collector.hold_retained_writer_for_test(false);
    EXPECT_EQ(fx.close(2101), 0);
    std::string error;
    ASSERT_TRUE(fx.flush(&error)) << error;
    EXPECT_EQ(fs::file_size(payload), kTensorBytes);
    const std::string manifest = read_file(fx.root.manifest("run-a"));
    ASSERT_FALSE(manifest.empty());
    EXPECT_EQ(manifest_string(manifest, "bin_file"), "args.e2101.bin");
    EXPECT_EQ(manifest_string(manifest, "collection_verdict"), "published");
}

/**
 * A failed arena read is loss, not stale content.
 *
 * The host shadow still holds whatever a previous transfer left in it, so
 * copying from it after a failed device-to-host read would publish another
 * run's bytes as this one's and call the run complete. The record keeps its
 * metadata, says its payload was discarded, and the run's flush fails — while
 * the lane still settles, because those arena bytes will not be read again.
 */
TEST(ArgsDumpOwnership, AFailedArenaReadIsRecordedAsLossNotExportedStale) {
    OwnershipFixture fx("arena-read-failure");
    fx.collector.fail_arena_copy_for_test(true);

    ASSERT_TRUE(fx.begin(2201, fx.root.prefix("run-a")));
    fx.produce(2201, /*args=*/1);
    ASSERT_TRUE(fx.wait_for_collected(1)) << "the record never reached the host";
    EXPECT_EQ(fx.close(2201), 0);

    std::string error;
    EXPECT_FALSE(fx.flush(&error)) << "a lost payload must fail the flush";

    const std::string manifest = read_file(fx.root.manifest("run-a"));
    ASSERT_FALSE(manifest.empty());
    // The metadata is here and says the bytes are not.
    EXPECT_EQ(manifest_number(manifest, "total_args"), 1);
    EXPECT_EQ(manifest_number(manifest, "host_discarded_args"), 1);
    EXPECT_NE(manifest.find("\"host_discarded\": true"), std::string::npos);
    EXPECT_EQ(manifest_string(manifest, "collection_verdict"), "published_short");
    EXPECT_NE(manifest.find("\"counts_unknown\": true"), std::string::npos);
    // No payload byte was invented from the shadow.
    EXPECT_NE(manifest.find("\"bin_file\": null"), std::string::npos);
    EXPECT_EQ(manifest.find("\"bin_size\": " + std::to_string(kTensorBytes)), std::string::npos);
    EXPECT_EQ(fs::file_size(fx.root.run_dir("run-a") / "args.e2201.bin"), 0u);

    // The lane settles anyway: nothing will read those arena bytes again, so
    // holding the producer on them would only spend its backstop.
    fx.collector.publish_arena_acks();
    DumpBufferState *lane = fx.state();
    EXPECT_EQ(lane->completed_payload_count, lane->published_payload_count);
    // And the loss is sticky, so a later flush still reports it.
    std::string second_error;
    EXPECT_FALSE(fx.flush(&second_error));
    EXPECT_EQ(fx.stats().errors.published_short, 1u);
}
