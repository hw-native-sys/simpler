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
 * ArgsDump across repeated runs, end to end through production code: the real
 * host collector allocates the pool and runs its own drain, collector and
 * writer threads; the real AICPU producer records args and publishes buffers;
 * and each run's payload file and manifest are published by the background
 * writer against that run's own exclusively owned name pair.
 *
 * `flush_retained_runs` is the synchronization point every case uses, because
 * it is the same barrier a caller has: after it returns, the runs closed before
 * it either have their files or have failed.
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

/** The value of one `"key": <number>` in a manifest, or -1 when absent. */
long manifest_number(const std::string &json, const std::string &key) {
    const std::string needle = "\"" + key + "\": ";
    const size_t at = json.find(needle);
    if (at == std::string::npos) return -1;
    return std::strtol(json.c_str() + at + needle.size(), nullptr, 10);
}

/** The value of one `"key": "<string>"` in a manifest, or "" when absent. */
std::string manifest_string(const std::string &json, const std::string &key) {
    const std::string needle = "\"" + key + "\": \"";
    const size_t at = json.find(needle);
    if (at == std::string::npos) return {};
    const size_t begin = at + needle.size();
    const size_t end = json.find('"', begin);
    if (end == std::string::npos) return {};
    return json.substr(begin, end - begin);
}

size_t count_occurrences(const std::string &haystack, const std::string &needle) {
    size_t n = 0;
    for (size_t at = haystack.find(needle); at != std::string::npos; at = haystack.find(needle, at + 1))
        n++;
    return n;
}

/** Files a failed run leaves as evidence at a destination. */
std::vector<std::string> evidence_files(const fs::path &dir) {
    std::vector<std::string> out;
    std::error_code ec;
    fs::directory_iterator it(dir, ec);
    if (ec) return out;
    for (const auto &entry : it) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("args_dump.json.", 0) == 0 && name.size() > 4 && name.compare(name.size() - 4, 4, ".tmp") == 0) {
            out.push_back(name);
        }
    }
    return out;
}

/** One output root per case, removed with the fixture. */
struct OutputRoot {
    fs::path base;

    explicit OutputRoot(const char *name) {
        base =
            fs::temp_directory_path() / ("args_dump_retained_" + std::string(name) + "_" + std::to_string(::getpid()));
        fs::remove_all(base);
        EXPECT_TRUE(fs::create_directories(base));
    }
    ~OutputRoot() { fs::remove_all(base); }

    /** A per-run output prefix; the collector puts `args_dump/` under it. */
    fs::path prefix(const char *run) const { return base / run; }
    fs::path run_dir(const char *run) const { return prefix(run) / "args_dump"; }
    fs::path manifest(const char *run) const { return run_dir(run) / "args_dump.json"; }
};

constexpr int kLane = 0;
constexpr uint64_t kTensorElements = 64;
constexpr uint64_t kTensorBytes = kTensorElements * sizeof(int32_t);

/**
 * The collector, its threads, and the AICPU producer's globals, wired the way a
 * retaining runner wires them.
 */
struct RetainedArgsDumpFixture {
    ArgsDumpCollector collector;
    OutputRoot root;
    void *shm = nullptr;
    DumpDataHeader *header = nullptr;
    std::vector<int32_t> tensor;

    explicit RetainedArgsDumpFixture(const char *name, size_t budget_bytes = simpler::dfx::runs::kDefaultBudgetBytes) :
        root(name),
        tensor(kTensorElements) {
        for (size_t i = 0; i < tensor.size(); i++)
            tensor[i] = static_cast<int32_t>(i + 1);
        collector.configure_retained_runs(true, budget_bytes);
        EXPECT_EQ(
            collector.initialize(
                /*num_dump_threads=*/1, /*device_id=*/0, DumpArgsLevel::FULL, retained_alloc, nullptr, retained_free
            ),
            0
        );
        shm = collector.get_dump_shm_device_ptr();
        EXPECT_NE(shm, nullptr);
        header = get_dump_header(shm);
        collector.start(retained_thread_factory);
        set_dump_args_enabled(true);
        set_platform_dump_base(reinterpret_cast<uint64_t>(shm));
    }

    ~RetainedArgsDumpFixture() {
        set_dump_args_enabled(false);
        set_platform_dump_base(0);
        set_platform_run_result(0, 0);
        // Ordered teardown, each step idempotent: publish what is left, join the
        // writer and the readers, then free.
        collector.finish_retained_runs();
        collector.stop();
        (void)collector.finalize(nullptr, retained_free);
    }

    DumpBufferState *state(int lane = kLane) { return get_dump_buffer_state(shm, lane); }

    bool begin(uint64_t epoch, const fs::path &prefix) {
        return collector.run_begin(epoch, prefix.string(), DumpArgsLevel::FULL);
    }

    void close(uint64_t epoch, bool device_execution_complete = true) {
        collector.run_close(epoch, device_execution_complete);
    }

    bool flush(std::string *error) {
        return collector.flush_retained_runs(simpler::dfx::runs::kCutAckBudgetMs * 8, error);
    }

    ArgsDumpCollector::RetainedRunStats stats() const { return collector.retained_run_stats_for_test(); }

    /** The device side of one run: init, record `args` tensors, flush. */
    void produce(uint64_t epoch, int args, bool flush_device = true) {
        set_platform_run_result(/*region_base=*/0, epoch);
        dump_args_init(/*num_dump_threads=*/1);
        for (int i = 0; i < args; i++)
            record_tensor(0x700 + static_cast<uint64_t>(i), static_cast<uint32_t>(i));
        if (flush_device) dump_args_flush(kLane);
    }

    void record_tensor(uint64_t task_id, uint32_t arg_index, uint64_t elements = kTensorElements) {
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
        info.shapes[0] = static_cast<uint32_t>(elements);
        info.strides[0] = 1;
        ASSERT_EQ(dump_arg_record(kLane, info), 0);
    }

    /**
     * Wait until the open runs have received `records`, which is what a case
     * that must act *after* a receipt needs; the drain and collector threads
     * choose the instant otherwise.
     */
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

// ---------------------------------------------------------------------------
// Two runs, their own files, and the capacity between them
// ---------------------------------------------------------------------------

/**
 * Two runs publish to two destinations, each naming and holding only its own
 * payload, and each run's offsets start at its own file's beginning.
 */
TEST(ArgsDumpRetainedRuns, TwoRunsKeepTheirOwnPayloadFileAndManifest) {
    RetainedArgsDumpFixture fx("two-runs");

    ASSERT_TRUE(fx.begin(201, fx.root.prefix("run-a")));
    fx.produce(201, /*args=*/2);
    fx.close(201);

    // Admitted while run 201 may still be publishing: two unpublished runs is
    // the capacity this overlap is made of.
    ASSERT_TRUE(fx.begin(202, fx.root.prefix("run-b")));
    fx.produce(202, /*args=*/1);
    fx.close(202);

    std::string error;
    ASSERT_TRUE(fx.flush(&error)) << error;

    const std::string manifest_a = read_file(fx.root.manifest("run-a"));
    const std::string manifest_b = read_file(fx.root.manifest("run-b"));
    ASSERT_FALSE(manifest_a.empty());
    ASSERT_FALSE(manifest_b.empty());
    EXPECT_EQ(manifest_number(manifest_a, "total_args"), 2);
    EXPECT_EQ(manifest_number(manifest_b, "total_args"), 1);
    EXPECT_EQ(manifest_string(manifest_a, "collection_verdict"), "published");
    EXPECT_EQ(manifest_string(manifest_b, "collection_verdict"), "published");
    EXPECT_NE(manifest_a.find("\"counts_unknown\": false"), std::string::npos);

    // Each manifest names its own payload file, and that file holds exactly its
    // own run's bytes.
    const std::string bin_a = manifest_string(manifest_a, "bin_file");
    const std::string bin_b = manifest_string(manifest_b, "bin_file");
    EXPECT_EQ(bin_a, "args.e201.bin");
    EXPECT_EQ(bin_b, "args.e202.bin");
    EXPECT_EQ(fs::file_size(fx.root.run_dir("run-a") / bin_a), 2 * kTensorBytes);
    EXPECT_EQ(fs::file_size(fx.root.run_dir("run-b") / bin_b), kTensorBytes);

    // Per-run offsets: B's only payload starts at 0 in B's own file.
    EXPECT_NE(manifest_b.find("\"bin_offset\": 0"), std::string::npos);
    EXPECT_EQ(count_occurrences(manifest_a, "\"bin_offset\": 0"), 1u);
    EXPECT_EQ(count_occurrences(manifest_a, "\"bin_offset\": " + std::to_string(kTensorBytes)), 1u);

    // Published by rename, so no temporary manifest is left at either.
    EXPECT_TRUE(evidence_files(fx.root.run_dir("run-a")).empty());
    EXPECT_TRUE(evidence_files(fx.root.run_dir("run-b")).empty());

    const auto st = fx.stats();
    EXPECT_FALSE(st.fatal);
    EXPECT_EQ(st.errors.published, 2u);
    EXPECT_EQ(st.errors.counts_unknown, 0u);
    // Both runs' charges came back with their slots.
    EXPECT_EQ(st.open_epochs, 0u);
    EXPECT_EQ(st.open_collected_records, 0u);
}

/**
 * A third unpublished run is refused before the device is handed anything, and
 * the two it waited behind still publish.
 */
TEST(ArgsDumpRetainedRuns, AThirdUnpublishedRunIsRefusedBeforeLaunch) {
    RetainedArgsDumpFixture fx("third-refused");
    fx.collector.hold_retained_writer_for_test(true);

    ASSERT_TRUE(fx.begin(301, fx.root.prefix("run-a")));
    fx.produce(301, /*args=*/1);
    fx.close(301);
    ASSERT_TRUE(fx.begin(302, fx.root.prefix("run-b")));
    fx.produce(302, /*args=*/1);
    fx.close(302);

    // Both slots are occupied and the writer is held, so this is a refusal and
    // not a wait: the caller fails the launch instead of queueing behind disk.
    EXPECT_FALSE(fx.begin(303, fx.root.prefix("run-c")));
    EXPECT_FALSE(fs::exists(fx.root.manifest("run-c")));
    EXPECT_EQ(fx.stats().open_epochs, 2u);

    fx.collector.hold_retained_writer_for_test(false);
    std::string error;
    ASSERT_TRUE(fx.flush(&error)) << error;
    EXPECT_TRUE(fs::exists(fx.root.manifest("run-a")));
    EXPECT_TRUE(fs::exists(fx.root.manifest("run-b")));

    // A refused admission promised no file, so it records no verdict.
    const auto st = fx.stats();
    EXPECT_EQ(st.errors.published, 2u);
    EXPECT_EQ(st.errors.abandoned, 0u);
    EXPECT_FALSE(st.fatal);
}

// ---------------------------------------------------------------------------
// The leftover buffer
// ---------------------------------------------------------------------------

/**
 * A buffer the device never published is recovered — its records **and** the
 * payload bytes they name — while the run still holds its execution claim, so
 * the successor's reuse of the buffer and the arena cannot reach it.
 */
TEST(ArgsDumpRetainedRuns, AnUnpublishedLeftoverBufferIsRecoveredBeforeTheSuccessor) {
    RetainedArgsDumpFixture fx("leftover-recovered");

    ASSERT_TRUE(fx.begin(401, fx.root.prefix("run-a")));
    // No device-side flush: the buffer stays named by `current_buf_ptr` with
    // its records in it, which is what a run reaped before its own flush
    // leaves behind.
    fx.produce(401, /*args=*/2, /*flush_device=*/false);
    ASSERT_NE(fx.state()->current_buf_ptr, 0u);
    ASSERT_EQ(fx.state()->current_buf_seq, 0u) << "the first buffer of a run carries sequence 0";

    fx.close(401);
    std::string error;
    ASSERT_TRUE(fx.flush(&error)) << error;

    const std::string manifest = read_file(fx.root.manifest("run-a"));
    ASSERT_FALSE(manifest.empty());
    // Sequence 0 with nothing delivered is a *candidate*, not a skip: both args
    // are here exactly once, and so are their bytes.
    EXPECT_EQ(manifest_number(manifest, "total_args"), 2);
    EXPECT_EQ(count_occurrences(manifest, "\"task_id\""), 2u);
    EXPECT_EQ(manifest_string(manifest, "collection_verdict"), "published");
    EXPECT_EQ(fs::file_size(fx.root.run_dir("run-a") / manifest_string(manifest, "bin_file")), 2 * kTensorBytes);
    EXPECT_FALSE(fx.stats().fatal);
}

/**
 * A buffer whose publication the ledger has already seen is not read again,
 * even though the device still names it: that is the window between the
 * publish and the pointer clear, and reading it could reach a recycled
 * incarnation.
 */
TEST(ArgsDumpRetainedRuns, AnAlreadyDeliveredLeftoverIsNotReadTwice) {
    RetainedArgsDumpFixture fx("leftover-delivered");

    ASSERT_TRUE(fx.begin(501, fx.root.prefix("run-a")));
    fx.produce(501, /*args=*/2);
    // The receipt must have landed before the close, because the close's
    // decision is against the ledger the receipt builds.
    ASSERT_TRUE(fx.wait_for_collected(2)) << "the published buffer was never delivered";
    const uint64_t published_ptr = fx.header->queues[kLane][0].buffer_ptr;
    ASSERT_NE(published_ptr, 0u);

    // The device's clearing store lost: published, and still named. The
    // fixture plays the device here, which is the only way to stand in that
    // window deterministically.
    fx.state()->current_buf_ptr = published_ptr;
    wmb();

    fx.close(501);
    std::string error;
    ASSERT_TRUE(fx.flush(&error)) << error;

    const std::string manifest = read_file(fx.root.manifest("run-a"));
    ASSERT_FALSE(manifest.empty());
    // Exactly two args: the ordinary path delivered them, and the leftover
    // decision added nothing.
    EXPECT_EQ(manifest_number(manifest, "total_args"), 2);
    EXPECT_EQ(count_occurrences(manifest, "\"task_id\""), 2u);
    EXPECT_EQ(manifest_string(manifest, "collection_verdict"), "published");
    EXPECT_EQ(fs::file_size(fx.root.run_dir("run-a") / manifest_string(manifest, "bin_file")), 2 * kTensorBytes);
}

/**
 * A sequence ahead of the ledger means a publication is missing from it, so
 * completeness cannot be asserted: nothing is read and the run's flush fails.
 */
TEST(ArgsDumpRetainedRuns, AnIdentityAheadOfTheLedgerIsIncompleteAndUnread) {
    RetainedArgsDumpFixture fx("leftover-unknown");

    ASSERT_TRUE(fx.begin(601, fx.root.prefix("run-a")));
    fx.produce(601, /*args=*/1);
    ASSERT_TRUE(fx.wait_for_collected(1));

    // A leftover naming a sequence this run never handed over. Its contents
    // must not be read: the address may belong to another incarnation.
    DumpMetaBuffer *leftover = reinterpret_cast<DumpMetaBuffer *>(fx.header->queues[kLane][0].buffer_ptr);
    ASSERT_NE(leftover, nullptr);
    fx.state()->current_buf_ptr = reinterpret_cast<uint64_t>(leftover);
    fx.state()->current_buf_seq = 7;
    wmb();

    fx.close(601);
    std::string error;
    EXPECT_FALSE(fx.flush(&error)) << "an undecidable leftover must fail the flush";
    EXPECT_NE(error.find("counts_unknown=1"), std::string::npos) << error;

    // The manifest is published and says on its face that it is incomplete.
    const std::string manifest = read_file(fx.root.manifest("run-a"));
    ASSERT_FALSE(manifest.empty());
    EXPECT_EQ(manifest_string(manifest, "collection_verdict"), "counts_unknown");
    EXPECT_NE(manifest.find("\"counts_unknown\": true"), std::string::npos);
    EXPECT_EQ(manifest_number(manifest, "total_args"), 1) << "only what was delivered is in it";

    const auto st = fx.stats();
    EXPECT_EQ(st.errors.counts_unknown, 1u);
    EXPECT_EQ(st.errors.published, 0u);
}

/**
 * Without an observed device fence nothing device-side is read at all, and the
 * run's own records are published as an incomplete result.
 */
TEST(ArgsDumpRetainedRuns, AnUnobservedFenceReadsNothingAndFailsTheFlush) {
    RetainedArgsDumpFixture fx("no-fence");

    ASSERT_TRUE(fx.begin(701, fx.root.prefix("run-a")));
    fx.produce(701, /*args=*/1);
    ASSERT_TRUE(fx.wait_for_collected(1));
    // A second, unpublished buffer's worth of records, which a recovery would
    // have taken had one been attempted.
    fx.record_tensor(0x7FF, /*arg_index=*/9);
    ASSERT_NE(fx.state()->current_buf_ptr, 0u);
    const DumpMetaBuffer *leftover = reinterpret_cast<const DumpMetaBuffer *>(fx.state()->current_buf_ptr);
    const uint32_t leftover_count = leftover->count;
    ASSERT_GT(leftover_count, 0u);

    fx.close(701, /*device_execution_complete=*/false);
    std::string error;
    EXPECT_FALSE(fx.flush(&error)) << "an unproven device stop must fail the flush";

    // Nothing read: the leftover still holds its records, and the manifest has
    // only what the ordinary path delivered.
    EXPECT_EQ(leftover->count, leftover_count) << "the leftover buffer was read on an unproven stop";
    const std::string manifest = read_file(fx.root.manifest("run-a"));
    ASSERT_FALSE(manifest.empty());
    EXPECT_EQ(manifest_number(manifest, "total_args"), 1);
    EXPECT_EQ(manifest_string(manifest, "collection_verdict"), "counts_unknown");
    EXPECT_NE(manifest.find("\"counts_unknown\": true"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Budget, output identity and the default path
// ---------------------------------------------------------------------------

/**
 * A payload the budget cannot admit keeps its metadata, marks the arg, settles
 * its lane and fails that run's flush — and allocates nothing while doing it.
 */
TEST(ArgsDumpRetainedRuns, ABudgetRefusedPayloadKeepsItsMetadataAndFailsTheFlush) {
    // Just past the accountant's minimum working set, so a payload larger than
    // it is refused while the metadata slot is not.
    RetainedArgsDumpFixture fx("budget-refused", simpler::dfx::runs::kMinWorkingSetBytes + (1u << 20));

    ASSERT_TRUE(fx.begin(801, fx.root.prefix("run-a")));
    set_platform_run_result(/*region_base=*/0, 801);
    dump_args_init(/*num_dump_threads=*/1);
    // One tensor whose payload is twice the whole working set, so no ordering
    // of the other charges can make room for it.
    const uint64_t huge_elements = (2 * simpler::dfx::runs::kMinWorkingSetBytes) / sizeof(int32_t);
    std::vector<int32_t> huge(huge_elements, 7);
    ArgsDumpInfo info{};
    info.task_id = 0x800;
    info.role = ArgsDumpRole::INPUT;
    info.stage = ArgsDumpStage::BEFORE_DISPATCH;
    info.arg_index = 0;
    info.kind = static_cast<uint8_t>(ArgsDumpKind::TENSOR);
    info.dtype = static_cast<uint8_t>(DataType::INT32);
    info.capture_payload = 1;
    info.buffer_addr = reinterpret_cast<uint64_t>(huge.data());
    info.func_count = 1;
    info.func_ids[0] = 5;
    info.ndims = 1;
    info.shapes[0] = static_cast<uint32_t>(huge_elements);
    info.strides[0] = 1;
    ASSERT_EQ(dump_arg_record(kLane, info), 0);
    dump_args_flush(kLane);

    fx.close(801);
    std::string error;
    EXPECT_FALSE(fx.flush(&error)) << "a host-discarded payload must fail the flush";

    const std::string manifest = read_file(fx.root.manifest("run-a"));
    ASSERT_FALSE(manifest.empty());
    // The record is there, marked, and its bytes are not.
    EXPECT_EQ(manifest_number(manifest, "total_args"), 1);
    EXPECT_EQ(manifest_number(manifest, "host_discarded_args"), 1);
    EXPECT_NE(manifest.find("\"host_discarded\": true"), std::string::npos);
    EXPECT_EQ(manifest_string(manifest, "collection_verdict"), "published_short");
    EXPECT_NE(manifest.find("\"counts_unknown\": true"), std::string::npos);
    // No payload byte reached disk, so no file is named.
    EXPECT_NE(manifest.find("\"bin_file\": null"), std::string::npos);

    // The lane settled anyway: the discard is what keeps the device's arena
    // barrier from waiting on bytes the host deliberately did not take.
    fx.collector.publish_arena_acks();
    EXPECT_EQ(fx.state()->completed_payload_count, fx.state()->published_payload_count);
    EXPECT_GT(fx.stats().budget_refusals, 0u);
}

/**
 * A destination that already holds a file under this run's first candidate name
 * is not truncated: the run takes the next exclusive pair instead.
 */
TEST(ArgsDumpRetainedRuns, AnOccupiedOutputNameMakesTheRunTakeTheNextPair) {
    RetainedArgsDumpFixture fx("token-collision");
    const fs::path run_dir = fx.root.run_dir("run-a");
    ASSERT_TRUE(fs::create_directories(run_dir));
    const fs::path squatter = run_dir / "args.e901.bin";
    {
        std::ofstream out(squatter, std::ios::binary);
        out << "not mine";
    }
    const std::string before = read_file(squatter);

    ASSERT_TRUE(fx.begin(901, fx.root.prefix("run-a")));
    fx.produce(901, /*args=*/1);
    fx.close(901);
    std::string error;
    ASSERT_TRUE(fx.flush(&error)) << error;

    const std::string manifest = read_file(fx.root.manifest("run-a"));
    ASSERT_FALSE(manifest.empty());
    EXPECT_EQ(manifest_string(manifest, "bin_file"), "args.e901.1.bin");
    EXPECT_EQ(read_file(squatter), before) << "an occupied name must never be truncated";
    EXPECT_EQ(fs::file_size(run_dir / "args.e901.1.bin"), kTensorBytes);
}

/**
 * A run proved to have produced no record publishes a manifest that says so,
 * with no payload file named, and that is a success.
 */
TEST(ArgsDumpRetainedRuns, ARunWithNoRecordsPublishesAnExplicitlyEmptyManifest) {
    RetainedArgsDumpFixture fx("empty-run");

    ASSERT_TRUE(fx.begin(1001, fx.root.prefix("run-a")));
    // The device side runs and flushes, with nothing recorded.
    fx.produce(1001, /*args=*/0);
    fx.close(1001);
    std::string error;
    ASSERT_TRUE(fx.flush(&error)) << error;

    const std::string manifest = read_file(fx.root.manifest("run-a"));
    ASSERT_FALSE(manifest.empty());
    EXPECT_EQ(manifest_number(manifest, "total_args"), 0);
    EXPECT_NE(manifest.find("\"bin_file\": null"), std::string::npos);
    EXPECT_EQ(manifest_string(manifest, "collection_verdict"), "published_empty");
    EXPECT_EQ(fx.stats().errors.published_empty, 1u);
}

/**
 * The failure record is sticky and reaches the caller through `finalize`'s
 * return, which is the only channel left once the caller's own flush has
 * already run.
 */
TEST(ArgsDumpRetainedRuns, FinalizeCarriesAFailureTheEarlierFlushAlreadyReported) {
    OutputRoot root("finalize-rc");
    ArgsDumpCollector collector;
    collector.configure_retained_runs(true, simpler::dfx::runs::kDefaultBudgetBytes);
    ASSERT_EQ(
        collector.initialize(
            /*num_dump_threads=*/1, /*device_id=*/0, DumpArgsLevel::FULL, retained_alloc, nullptr, retained_free
        ),
        0
    );
    void *shm = collector.get_dump_shm_device_ptr();
    ASSERT_NE(shm, nullptr);
    collector.start(retained_thread_factory);
    set_dump_args_enabled(true);
    set_platform_dump_base(reinterpret_cast<uint64_t>(shm));

    ASSERT_TRUE(collector.run_begin(1101, root.prefix("run-a").string(), DumpArgsLevel::FULL));
    set_platform_run_result(/*region_base=*/0, 1101);
    dump_args_init(/*num_dump_threads=*/1);
    dump_args_flush(kLane);
    // No fence: the run is incomplete and the flush says so.
    collector.run_close(1101, /*device_execution_complete=*/false);
    std::string error;
    EXPECT_FALSE(collector.flush_retained_runs(simpler::dfx::runs::kCutAckBudgetMs * 8, &error));

    set_dump_args_enabled(false);
    set_platform_dump_base(0);
    set_platform_run_result(0, 0);
    collector.finish_retained_runs();
    collector.stop();
    // The same failure, on the one channel a caller has after its flush: a
    // later error cannot be reported by an earlier call that already returned.
    EXPECT_NE(collector.finalize(nullptr, retained_free), 0);
    // Idempotent, and a collector with nothing left reports nothing.
    EXPECT_EQ(collector.finalize(nullptr, retained_free), 0);
}

/**
 * With retention off the collector keeps its single-run behaviour: one
 * `args.bin`, the manifest that names it, and both complete at the boundary.
 */
TEST(ArgsDumpRetainedRuns, RetentionOffKeepsTheSingleRunOutput) {
    OutputRoot root("retention-off");
    ArgsDumpCollector collector;
    collector.configure_retained_runs(false, simpler::dfx::runs::kDefaultBudgetBytes);
    ASSERT_EQ(
        collector.initialize(
            /*num_dump_threads=*/1, /*device_id=*/0, DumpArgsLevel::FULL, retained_alloc, nullptr, retained_free
        ),
        0
    );
    void *shm = collector.get_dump_shm_device_ptr();
    ASSERT_NE(shm, nullptr);
    DumpDataHeader *header = get_dump_header(shm);
    set_dump_args_enabled(true);
    set_platform_dump_base(reinterpret_cast<uint64_t>(shm));

    collector.begin_run(root.prefix("run-a").string(), DumpArgsLevel::FULL);
    std::vector<int32_t> tensor(kTensorElements, 3);
    set_platform_run_result(/*region_base=*/0, 1201);
    dump_args_init(/*num_dump_threads=*/1);
    ArgsDumpInfo info{};
    info.task_id = 0x1200;
    info.role = ArgsDumpRole::INPUT;
    info.stage = ArgsDumpStage::BEFORE_DISPATCH;
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
    dump_args_flush(kLane);

    // The single-run path's own delivery, with no collector threads running.
    const DumpReadyQueueEntry &entry = header->queues[kLane][0];
    DumpReadyBufferInfo delivered{};
    delivered.thread_index = entry.thread_index;
    delivered.dev_buffer_ptr = reinterpret_cast<void *>(entry.buffer_ptr);
    delivered.host_buffer_ptr = reinterpret_cast<void *>(entry.buffer_ptr);
    delivered.buffer_seq = entry.buffer_seq;
    collector.on_buffer_collected(delivered, /*collector_shard=*/0);
    EXPECT_EQ(collector.export_dump_files(), 0);

    const fs::path run_dir = root.run_dir("run-a");
    const std::string manifest = read_file(run_dir / "args_dump.json");
    ASSERT_FALSE(manifest.empty());
    EXPECT_EQ(manifest_string(manifest, "bin_file"), "args.bin");
    EXPECT_EQ(manifest_number(manifest, "total_args"), 1);
    // None of the retained path's fields appear on this path.
    EXPECT_EQ(manifest.find("collection_verdict"), std::string::npos);
    EXPECT_EQ(manifest.find("host_discarded_args"), std::string::npos);
    EXPECT_EQ(fs::file_size(run_dir / "args.bin"), kTensorBytes);
    EXPECT_TRUE(fs::exists(run_dir / "args.bin"));

    set_dump_args_enabled(false);
    set_platform_dump_base(0);
    set_platform_run_result(0, 0);
    collector.stop();
    EXPECT_EQ(collector.finalize(nullptr, retained_free), 0);
}
