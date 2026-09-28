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
 * ScopeStats cross-run retention: what the boundary owns, and what the writer
 * is allowed to publish.
 *
 * Both sides are production code. Records are produced through the real AICPU
 * module — `scope_stats_begin` / `_end` / `scope_stats_aicpu_flush_buffers`
 * stamp, rotate and publish exactly as they do on device — and consumed
 * through the collector's own threads, boundary and writer. What a case
 * supplies is the shape of the run and what the device left behind.
 *
 * What these cannot cover, stated rather than simulated: device cache
 * visibility (every store here is plain host memory), and exhausting a real
 * 256 MiB budget — the budget refusal's *consequence* is asserted through the
 * production `RecordBlocks` and `classify` instead of by a multi-hundred-
 * thousand-record soak.
 */

#include "host/scope_stats_collector.h"

#include <gtest/gtest.h>

#include "aicpu/device_run_result_base_aicpu.h"
#include "aicpu/scope_stats_collector_aicpu.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <unistd.h>

namespace fs = std::filesystem;
namespace runs = simpler::dfx::runs;
namespace scope_runs = simpler::dfx::scope_stats_runs;

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

/** Count the record lines, which are every line after the metadata line. */
size_t record_lines(const std::string &jsonl) {
    if (jsonl.empty()) return 0;
    size_t lines = 0;
    for (char c : jsonl) {
        if (c == '\n') lines++;
    }
    return lines == 0 ? 0 : lines - 1;
}

struct OutputRoot {
    fs::path base;

    explicit OutputRoot(const char *name) {
        base = fs::temp_directory_path() /
               ("scope_stats_retained_" + std::string(name) + "_" + std::to_string(::getpid()));
        fs::remove_all(base);
        EXPECT_TRUE(fs::create_directories(base));
    }
    ~OutputRoot() { fs::remove_all(base); }

    fs::path prefix(const char *run) const { return base / run; }
    fs::path artifact(const char *run) const { return prefix(run) / "scope_stats" / "scope_stats.jsonl"; }
    fs::path temp(const char *run) const { return prefix(run) / "scope_stats" / "scope_stats.jsonl.tmp"; }
};

struct RetainedFixture {
    ScopeStatsCollector collector;
    OutputRoot root;
    void *shm = nullptr;

    explicit RetainedFixture(const char *name) :
        root(name) {
        collector.configure_retained_runs(true, runs::kDefaultBudgetBytes);
        EXPECT_EQ(collector.init(/*num_threads=*/1, retained_alloc, nullptr, retained_free, /*device_id=*/0), 0);
        shm = collector.get_scope_stats_shm_device_ptr();
        EXPECT_NE(shm, nullptr);
        set_scope_stats_enabled(true);
        set_platform_scope_stats_base(reinterpret_cast<uint64_t>(shm));
        scope_stats_aicpu_set_orch_thread_idx(0);
    }

    ~RetainedFixture() {
        collector.pause_writer_for_test(false);
        set_scope_stats_enabled(false);
        set_platform_scope_stats_base(0);
        set_platform_run_result(0, 0);
        collector.finalize(nullptr, retained_free);
    }

    ScopeStatsBufferState *state() { return get_scope_stats_buffer_state(shm, 0); }
    ScopeStatsDataHeader *header() { return get_scope_stats_header(shm); }

    bool begin(uint64_t epoch, const fs::path &prefix) {
        collector.start(retained_thread_factory);
        return collector.run_begin(epoch, prefix.string());
    }
    void close(uint64_t epoch, bool complete = true) { collector.run_close(epoch, complete); }
    bool flush(std::string *error) { return collector.flush_retained_runs(5000, error); }

    /** `pairs` begin/end pairs through the real producer, optionally published. */
    void produce(uint64_t epoch, int pairs, bool flush_at_end, const char *site) {
        set_platform_run_result(/*region_base=*/0, epoch);
        for (int i = 0; i < pairs; i++) {
            scope_stats_set_pending_site(site, 100 + i);
            scope_stats_begin(0, 1, 2, 1024, 2048, 1, 2, 3);
            scope_stats_end(0, 1, 3, 1024, 3072, 1, 3, 4);
        }
        if (flush_at_end) scope_stats_aicpu_flush_buffers();
    }

    bool wait_for_collected(uint64_t records, int timeout_ms = 5000) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            if (collector.total_collected() >= records) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        return collector.total_collected() >= records;
    }
};

}  // namespace

/**
 * A buffer the producer published and one it never published each contribute
 * their records exactly once.
 *
 * The producer clears `current_buf_ptr` the instant an enqueue succeeds, so a
 * non-zero pointer naming a non-empty buffer is its own evidence that the
 * buffer was never handed over. That is what lets the boundary recover it with
 * no de-duplication bookkeeping — and what stops a published buffer being
 * counted a second time.
 */
TEST(ScopeStatsRetainedRuns, PublishedAndUnpublishedBuffersAreEachCountedOnce) {
    RetainedFixture fx("counted-once");
    ASSERT_TRUE(fx.begin(4001, fx.root.prefix("run-a")));

    // Published: the producer's own flush enqueues the buffer and clears the
    // pointer behind it.
    fx.produce(4001, /*pairs=*/3, /*flush_at_end=*/true, "published.cpp");
    ASSERT_EQ(fx.state()->current_buf_ptr, 0u) << "the producer did not clear the pointer it published";
    ASSERT_TRUE(fx.wait_for_collected(6));

    // Unpublished: more records, no flush, so the pointer still names them.
    fx.produce(4001, /*pairs=*/2, /*flush_at_end=*/false, "unpublished.cpp");
    ASSERT_NE(fx.state()->current_buf_ptr, 0u);
    const uint32_t device_total = fx.state()->total_record_count;
    ASSERT_EQ(device_total, 10u);

    fx.close(4001);
    std::string error;
    ASSERT_TRUE(fx.flush(&error)) << error;

    const std::string jsonl = read_file(fx.root.artifact("run-a"));
    ASSERT_FALSE(jsonl.empty());
    EXPECT_EQ(record_lines(jsonl), 10u) << "every record appears exactly once";
    EXPECT_NE(jsonl.find("\"total\": 10"), std::string::npos);
    EXPECT_NE(jsonl.find("\"dropped\": 0"), std::string::npos);
    EXPECT_NE(jsonl.find("\"collection_verdict\": \"published\""), std::string::npos);
    EXPECT_NE(jsonl.find("\"counts_unknown\": false"), std::string::npos);
    EXPECT_NE(jsonl.find("\"host_received_records\": 10"), std::string::npos);
    EXPECT_NE(jsonl.find("\"host_retained_records\": 10"), std::string::npos);
    EXPECT_NE(jsonl.find("published.cpp"), std::string::npos);
    EXPECT_NE(jsonl.find("unpublished.cpp"), std::string::npos);
    EXPECT_EQ(fx.collector.retained_run_stats_for_test().published, 1u);
}

/**
 * A saturated device counter settles counts-unknown; a latched fatal with a
 * balanced identity stays count-known.
 *
 * These are the two classifications that must not be confused. A counter at
 * the saturation sentinel says only "at or past the limit", so nothing built
 * on it can be asserted. A device fatal says the run went wrong, but the
 * accounting still balances — so the artifact keeps its true counts and the
 * fatal travels in the field it already has.
 */
TEST(ScopeStatsRetainedRuns, SaturationIsCountsUnknownAndAFatalStaysCountKnown) {
    {
        RetainedFixture fx("saturated");
        ASSERT_TRUE(fx.begin(4101, fx.root.prefix("run-a")));
        fx.produce(4101, /*pairs=*/1, /*flush_at_end=*/true, "sat.cpp");
        ASSERT_TRUE(fx.wait_for_collected(2));
        fx.state()->total_record_count = UINT32_MAX;
        fx.close(4101);

        std::string error;
        EXPECT_FALSE(fx.flush(&error)) << "a saturated counter must fail the flush";
        const std::string jsonl = read_file(fx.root.artifact("run-a"));
        ASSERT_FALSE(jsonl.empty()) << "the artifact is still published, carrying its verdict";
        EXPECT_NE(jsonl.find("\"counts_unknown\": true"), std::string::npos);
        EXPECT_NE(jsonl.find("\"collection_verdict\": \"partial_cut_unknown\""), std::string::npos);
        EXPECT_NE(jsonl.find("\"total\": 4294967295"), std::string::npos);
        EXPECT_EQ(fx.collector.retained_run_stats_for_test().counts_unknown, 1u);
    }
    {
        RetainedFixture fx("fatal");
        ASSERT_TRUE(fx.begin(4102, fx.root.prefix("run-b")));
        fx.produce(4102, /*pairs=*/2, /*flush_at_end=*/true, "fatal.cpp");
        ASSERT_TRUE(fx.wait_for_collected(4));
        scope_stats_on_fatal();
        ASSERT_NE(fx.header()->fatal_latched, 0u);
        fx.close(4102);

        std::string error;
        EXPECT_FALSE(fx.flush(&error)) << "a device fatal must still fail the flush";
        const std::string jsonl = read_file(fx.root.artifact("run-b"));
        ASSERT_FALSE(jsonl.empty());
        EXPECT_NE(jsonl.find("\"fatal\": true"), std::string::npos);
        EXPECT_NE(jsonl.find("\"counts_unknown\": false"), std::string::npos)
            << "a fatal does not make the counts unknown";
        EXPECT_NE(jsonl.find("\"collection_verdict\": \"partial_safe\""), std::string::npos);
        EXPECT_EQ(record_lines(jsonl), 4u);
        EXPECT_EQ(fx.collector.retained_run_stats_for_test().partial, 1u);
    }
}

/**
 * A refused block charge keeps the record out of storage and out of nothing
 * else: it is still received, and the difference is what the artifact reports.
 *
 * Asserted against the production store and the production classifier rather
 * than by exhausting a 256 MiB budget, which a unit test cannot do in bounded
 * time. What it does establish is the rule the budget path relies on — a
 * refusal allocates nothing, drops the record rather than the run, and lands
 * as a count-known partial.
 */
TEST(ScopeStatsRetainedRuns, ABudgetRefusalRetainsLessThanItReceivedAndStaysCountKnown) {
    scope_runs::RecordBlocks store;
    const scope_runs::Record record{};
    size_t granted = 0;
    auto grant_one_block = [&granted](size_t bytes) {
        if (granted != 0) return false;
        granted = bytes;
        return true;
    };
    size_t credited = 0;
    auto give_back = [&credited](size_t bytes) {
        credited += bytes;
    };
    // The first record buys the only block this budget will pay for.
    ASSERT_TRUE(store.append(record, grant_one_block, give_back));
    EXPECT_EQ(granted, scope_runs::kRecordBlockBytes);
    EXPECT_EQ(store.charged_bytes(), scope_runs::kRecordBlockBytes);
    for (size_t i = 1; i < scope_runs::kRecordsPerBlock; i++) {
        ASSERT_TRUE(store.append(record, grant_one_block, give_back))
            << "a record inside the paid block needs no new charge";
    }
    // The next one needs a second block, which the budget refuses.
    EXPECT_FALSE(store.append(record, grant_one_block, give_back));
    EXPECT_EQ(store.size(), scope_runs::kRecordsPerBlock);
    EXPECT_EQ(store.charged_bytes(), scope_runs::kRecordBlockBytes) << "a refusal allocated nothing";
    EXPECT_EQ(credited, 0u) << "a budget refusal charges nothing, so it credits nothing back";
    EXPECT_EQ(store.release(), scope_runs::kRecordBlockBytes);

    scope_runs::DeviceSnapshot device;
    device.valid = true;
    device.total_records = 10;
    device.dropped_records = 0;
    const scope_runs::Collection partial = scope_runs::classify(device, /*received=*/10, /*retained=*/8);
    EXPECT_EQ(partial.verdict, runs::Verdict::PartialSafe);
    EXPECT_FALSE(partial.counts_unknown) << "host loss is known loss, not unknown counts";
    EXPECT_EQ(partial.received, 10u);
    EXPECT_EQ(partial.retained, 8u);
}

/**
 * Two runs may be unpublished at once; a third is refused before launch.
 *
 * A slot is held from admission until the artifact exists, so the bound is on
 * what the collector owns rather than on what is merely still filling. The
 * writer is held so both slots are provably occupied at the moment of the
 * third admission.
 */
TEST(ScopeStatsRetainedRuns, TwoUnpublishedExportsAreAllowedAndAThirdIsRefused) {
    RetainedFixture fx("three-runs");
    fx.collector.pause_writer_for_test(true);

    ASSERT_TRUE(fx.begin(4201, fx.root.prefix("run-a")));
    fx.produce(4201, 1, true, "a.cpp");
    ASSERT_TRUE(fx.wait_for_collected(2));
    fx.close(4201);

    ASSERT_TRUE(fx.begin(4202, fx.root.prefix("run-b")));
    fx.produce(4202, 1, true, "b.cpp");
    ASSERT_TRUE(fx.wait_for_collected(2));
    fx.close(4202);

    EXPECT_EQ(fx.collector.retained_run_stats_for_test().open_slots, 2u);
    EXPECT_FALSE(fx.begin(4203, fx.root.prefix("run-c"))) << "a third unpublished export must be refused";
    EXPECT_FALSE(fs::exists(fx.root.artifact("run-c")));

    // Releasing the writer publishes both and frees their slots.
    fx.collector.pause_writer_for_test(false);
    std::string error;
    ASSERT_TRUE(fx.flush(&error)) << error;
    EXPECT_EQ(fx.collector.retained_run_stats_for_test().open_slots, 0u);
    EXPECT_TRUE(fs::exists(fx.root.artifact("run-a")));
    EXPECT_TRUE(fs::exists(fx.root.artifact("run-b")));
    EXPECT_TRUE(fx.begin(4204, fx.root.prefix("run-d")));
    fx.close(4204);
}

/**
 * A destination that already exists is neither removed nor overwritten, the
 * run fails, and no temporary evidence is left behind.
 *
 * Publication is a `link` onto a name nothing else owns, so a collision is a
 * failure rather than a silent replacement — and a publication failure fails
 * the run whatever its collection verdict said.
 */
TEST(ScopeStatsRetainedRuns, AnOccupiedOutputNameFailsWithoutClobbering) {
    RetainedFixture fx("occupied");
    const fs::path dir = fx.root.prefix("run-a") / "scope_stats";
    ASSERT_TRUE(fs::create_directories(dir));
    const std::string keep = "{\"someone\": \"else\"}\n";
    {
        std::ofstream out(dir / "scope_stats.jsonl", std::ios::binary);
        out << keep;
    }

    ASSERT_TRUE(fx.begin(4301, fx.root.prefix("run-a")));
    fx.produce(4301, 1, true, "occupied.cpp");
    ASSERT_TRUE(fx.wait_for_collected(2));
    fx.close(4301);

    std::string error;
    EXPECT_FALSE(fx.flush(&error)) << "a publication failure must fail the flush";
    EXPECT_EQ(read_file(fx.root.artifact("run-a")), keep) << "the file already there was modified";
    EXPECT_FALSE(fs::exists(fx.root.temp("run-a"))) << "a failed publication left temporary debris";
    const auto stats = fx.collector.retained_run_stats_for_test();
    EXPECT_EQ(stats.write_failed, 1u);
    EXPECT_EQ(stats.published, 0u) << "a complete collection cannot report success through a missing file";
    EXPECT_EQ(stats.open_slots, 0u) << "a failed publication still returns its slot";
}

/**
 * Without a completion proof: no artifact, nothing shared is read, the host
 * copies survive until the collector threads are joined, and the next
 * ScopeStats run is refused.
 *
 * The device's own counters are the witness that nothing was reset: a run that
 * had been admitted would have zeroed them.
 */
TEST(ScopeStatsRetainedRuns, UnprovenCompletionPublishesNothingAndDiscardsAfterTheJoin) {
    OutputRoot root("unproven");
    ScopeStatsCollector collector;
    collector.configure_retained_runs(true, runs::kDefaultBudgetBytes);
    ASSERT_EQ(collector.init(1, retained_alloc, nullptr, retained_free, 0), 0);
    void *shm = collector.get_scope_stats_shm_device_ptr();
    set_scope_stats_enabled(true);
    set_platform_scope_stats_base(reinterpret_cast<uint64_t>(shm));
    scope_stats_aicpu_set_orch_thread_idx(0);

    collector.start(retained_thread_factory);
    ASSERT_TRUE(collector.run_begin(4401, root.prefix("run-a").string()));
    set_platform_run_result(0, 4401);
    for (int i = 0; i < 2; i++) {
        scope_stats_set_pending_site("unproven.cpp", 200 + i);
        scope_stats_begin(0, 1, 2, 1024, 2048, 1, 2, 3);
        scope_stats_end(0, 1, 3, 1024, 3072, 1, 3, 4);
    }
    auto *state = get_scope_stats_buffer_state(shm, 0);
    const uint64_t in_flight = state->current_buf_ptr;
    ASSERT_NE(in_flight, 0u);

    collector.run_close(4401, /*device_execution_complete=*/false);

    EXPECT_FALSE(fs::exists(root.artifact("run-a"))) << "no artifact is promised without a completion proof";
    EXPECT_FALSE(fs::exists(root.temp("run-a")));
    EXPECT_EQ(state->current_buf_ptr, in_flight) << "the producer's buffer was recovered on a path that must not read";
    EXPECT_EQ(state->total_record_count, 4u) << "the device counters were touched on a path that must not write";

    std::string error;
    EXPECT_FALSE(collector.flush_retained_runs(5000, &error));
    EXPECT_FALSE(error.empty());
    const auto stats = collector.retained_run_stats_for_test();
    EXPECT_EQ(stats.quarantined, 1u);
    EXPECT_GT(stats.charged_bytes, 0u) << "the quarantined copies are still charged";

    // Refused in any mode while the quarantine holds the live store.
    EXPECT_FALSE(collector.run_begin(4402, root.prefix("run-b").string()));

    // Freeing them is legal only after the collector threads are joined, which
    // is the `stop()` finalize already performs.
    collector.finalize(nullptr, retained_free);
    EXPECT_FALSE(fs::exists(root.artifact("run-a")));
    set_scope_stats_enabled(false);
    set_platform_scope_stats_base(0);
    set_platform_run_result(0, 0);
}

/**
 * Three consecutive runs each publish their own artifact, holding only their
 * own records, and `flush_diagnostics` is what makes them all present.
 */
TEST(ScopeStatsRetainedRuns, ConsecutiveRunsEachPublishTheirOwnRecords) {
    RetainedFixture fx("three-artifacts");
    const char *names[] = {"run-a", "run-b", "run-c"};
    const char *sites[] = {"first.cpp", "second.cpp", "third.cpp"};
    for (int i = 0; i < 3; i++) {
        const uint64_t epoch = 4500 + static_cast<uint64_t>(i);
        ASSERT_TRUE(fx.begin(epoch, fx.root.prefix(names[i])));
        fx.produce(epoch, /*pairs=*/i + 1, /*flush_at_end=*/true, sites[i]);
        ASSERT_TRUE(fx.wait_for_collected(static_cast<uint64_t>(2 * (i + 1))));
        fx.close(epoch);
    }
    std::string error;
    ASSERT_TRUE(fx.flush(&error)) << error;

    for (int i = 0; i < 3; i++) {
        const std::string jsonl = read_file(fx.root.artifact(names[i]));
        ASSERT_FALSE(jsonl.empty()) << names[i];
        EXPECT_EQ(record_lines(jsonl), static_cast<size_t>(2 * (i + 1))) << names[i];
        EXPECT_NE(jsonl.find(sites[i]), std::string::npos) << names[i];
        for (int other = 0; other < 3; other++) {
            if (other == i) continue;
            EXPECT_EQ(jsonl.find(sites[other]), std::string::npos) << "records leaked between runs";
        }
    }
    EXPECT_EQ(fx.collector.retained_run_stats_for_test().published, 3u);
}

/**
 * Retention off keeps the boundary write and today's metadata line exactly.
 *
 * The background-mode keys are additive, so their absence here is what proves
 * the default artifact is unchanged.
 */
TEST(ScopeStatsRetainedRuns, RetentionOffKeepsTheSingleRunOutput) {
    OutputRoot root("default-off");
    ScopeStatsCollector collector;
    ASSERT_EQ(collector.init(1, retained_alloc, nullptr, retained_free, 0), 0);
    EXPECT_FALSE(collector.retains_runs());
    void *shm = collector.get_scope_stats_shm_device_ptr();
    set_scope_stats_enabled(true);
    set_platform_scope_stats_base(reinterpret_cast<uint64_t>(shm));
    scope_stats_aicpu_set_orch_thread_idx(0);

    collector.begin_run();
    collector.start(retained_thread_factory);
    set_platform_run_result(0, 4601);
    scope_stats_set_pending_site("default.cpp", 300);
    scope_stats_begin(0, 1, 2, 1024, 2048, 1, 2, 3);
    scope_stats_end(0, 1, 3, 1024, 3072, 1, 3, 4);
    scope_stats_aicpu_flush_buffers();

    collector.quiesce();
    collector.reconcile_counters();
    ASSERT_EQ(collector.write_jsonl(root.prefix("run-a").string()), 0);

    const std::string jsonl = read_file(root.artifact("run-a"));
    ASSERT_FALSE(jsonl.empty());
    EXPECT_EQ(record_lines(jsonl), 2u);
    EXPECT_NE(jsonl.find("\"tensormap_max\":"), std::string::npos);
    EXPECT_EQ(jsonl.find("collection_verdict"), std::string::npos) << "the default artifact gained a key";
    EXPECT_EQ(jsonl.find("counts_unknown"), std::string::npos) << "the default artifact gained a key";
    EXPECT_EQ(jsonl.find("host_received_records"), std::string::npos) << "the default artifact gained a key";

    collector.finalize(nullptr, retained_free);
    set_scope_stats_enabled(false);
    set_platform_scope_stats_base(0);
    set_platform_run_result(0, 0);
}

/**
 * A host allocation failure on either thread becomes a reported error, and
 * strands nothing.
 *
 * Neither thread entry point has an exception boundary of its own —
 * `ProfilerBase::consume` calls the collector callback directly, and the
 * writer's publish allocates a path and a staging block — so an escape here
 * terminates the chip subprocess instead of failing the flush. Both failures
 * are injected at the production call site rather than imitated.
 *
 * What this cannot inject, and what the code carries instead: the credit
 * rollback inside `RecordBlocks::append` when the block allocation itself
 * throws. That branch needs `make_unique` to fail, which a test cannot force
 * without a hook in the store; it is a `catch` that credits the same figure it
 * charged, verified by inspection.
 */
TEST(ScopeStatsRetainedRuns, AllocationFailuresBecomeErrorsInsteadOfEscapingThreads) {
    RetainedFixture fx("alloc-failure");

    // The collector thread's append throws while the run is live.
    ASSERT_TRUE(fx.begin(4701, fx.root.prefix("run-a")));
    fx.collector.throw_in_collector_for_test(true);
    fx.produce(4701, /*pairs=*/1, /*flush_at_end=*/true, "escape.cpp");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        if (fx.collector.retained_run_stats_for_test().host_failures > 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    fx.collector.throw_in_collector_for_test(false);
    EXPECT_GT(fx.collector.retained_run_stats_for_test().host_failures, 0u)
        << "the collector thread's failure was neither caught nor reported";
    // Reported, not fatal to the process: the run still closes, and received
    // stays what the host actually took — the throw retained nothing.
    EXPECT_EQ(fx.collector.total_collected(), 0u);
    fx.close(4701);
    std::string error;
    EXPECT_FALSE(fx.flush(&error)) << "a host failure must fail the flush";
    EXPECT_FALSE(error.empty());

    // The writer thread's publish throws for the next run.
    fx.collector.throw_in_writer_for_test(true);
    ASSERT_TRUE(fx.begin(4702, fx.root.prefix("run-b")));
    fx.produce(4702, /*pairs=*/1, /*flush_at_end=*/true, "writer.cpp");
    ASSERT_TRUE(fx.wait_for_collected(2));
    fx.close(4702);
    std::string writer_error;
    EXPECT_FALSE(fx.flush(&writer_error));
    fx.collector.throw_in_writer_for_test(false);

    const auto stats = fx.collector.retained_run_stats_for_test();
    EXPECT_GE(stats.write_failed, 1u) << "a throwing publish settles as a write failure";
    EXPECT_EQ(stats.open_slots, 0u) << "a throwing publish stranded its export slot";
    EXPECT_FALSE(fs::exists(fx.root.artifact("run-b")));
    EXPECT_FALSE(fs::exists(fx.root.temp("run-b"))) << "a throwing publish left temporary debris";

    // Every slot and every charged block came back, so a third run is still
    // admissible and a later run still publishes normally.
    EXPECT_TRUE(fx.begin(4703, fx.root.prefix("run-c")));
    fx.produce(4703, /*pairs=*/1, /*flush_at_end=*/true, "after.cpp");
    ASSERT_TRUE(fx.wait_for_collected(2));
    fx.close(4703);
    std::string tail_error;
    EXPECT_FALSE(fx.flush(&tail_error)) << "the earlier failures are sticky for the runner's life";
    EXPECT_TRUE(fs::exists(fx.root.artifact("run-c"))) << "a later run still publishes normally";
}

/**
 * A sticky failure survives a reconfiguration that turns retention off, and is
 * still reported by the flush the runner calls.
 *
 * The runner's public arm is deliberately ungated on `retains_runs()` for this
 * reason — a collector rebuild may reconfigure retention off, and a failure
 * recorded before it must not become unreportable. This drives the collector
 * side of that: the reconfiguration is the real entry point the rebuild path
 * uses, and the flush is the one the runner calls into.
 */
TEST(ScopeStatsRetainedRuns, StickyErrorsSurviveAReconfigurationThatTurnsRetentionOff) {
    RetainedFixture fx("sticky-rebuild");
    const fs::path dir = fx.root.prefix("run-a") / "scope_stats";
    ASSERT_TRUE(fs::create_directories(dir));
    {
        std::ofstream out(dir / "scope_stats.jsonl", std::ios::binary);
        out << "{\"someone\": \"else\"}\n";
    }

    ASSERT_TRUE(fx.begin(4801, fx.root.prefix("run-a")));
    fx.produce(4801, /*pairs=*/1, /*flush_at_end=*/true, "sticky.cpp");
    ASSERT_TRUE(fx.wait_for_collected(2));
    fx.close(4801);

    std::string first;
    ASSERT_FALSE(fx.flush(&first)) << "the occupied destination must fail this run";
    ASSERT_FALSE(first.empty());

    // What a rebuild does to this collector: retention is reconfigured off.
    fx.collector.configure_retained_runs(false, runs::kDefaultBudgetBytes);
    EXPECT_FALSE(fx.collector.retains_runs());

    std::string after;
    EXPECT_FALSE(fx.flush(&after)) << "the failure became unreportable once retention was turned off";
    EXPECT_EQ(after, first) << "the same sticky record is reported, not a fresh empty one";

    // And a collector that never retained anything reports nothing, which is
    // what makes the runner's arm safe to leave ungated.
    ScopeStatsCollector never_retained;
    ASSERT_EQ(never_retained.init(1, retained_alloc, nullptr, retained_free, 0), 0);
    std::string quiet = "unset";
    EXPECT_TRUE(never_retained.flush_retained_runs(1000, &quiet));
    EXPECT_EQ(quiet, "unset") << "a collector with nothing to report wrote an error anyway";
    never_retained.finalize(nullptr, retained_free);
}

/**
 * A writer-thread construction failure leaves no phantom consumer behind.
 *
 * `writer_running_` used to be published before the `std::thread` was
 * constructed, so a `system_error` there left it true: the next admission
 * believed a consumer was running, its export went into a queue nothing
 * drained, the flush timed out and `finish_retained_runs` waited forever. The
 * flag is now published only after the thread exists, which is what makes the
 * recovery below possible at all.
 */
TEST(ScopeStatsRetainedRuns, AFailedWriterStartLeavesNoPhantomConsumer) {
    RetainedFixture fx("writer-start");

    fx.collector.fail_writer_start_for_test(true);
    fx.collector.start(retained_thread_factory);
    EXPECT_FALSE(fx.collector.run_begin(4901, fx.root.prefix("run-a").string()))
        << "a run must not be admitted when its writer could not be started";
    EXPECT_EQ(fx.collector.retained_run_stats_for_test().open_slots, 0u) << "the refused admission stranded its slot";
    EXPECT_GT(fx.collector.retained_run_stats_for_test().host_failures, 0u) << "the failure is not observable";

    // The failure persists while the cause does, and is refused the same way
    // rather than silently admitted against a consumer that does not exist.
    EXPECT_FALSE(fx.collector.run_begin(4902, fx.root.prefix("run-b").string()));
    EXPECT_EQ(fx.collector.retained_run_stats_for_test().open_slots, 0u);

    // With the cause gone a real writer starts, and the proof is that the
    // export is actually published — the flag cannot have been left true by
    // the failures above, or nothing would drain this queue.
    fx.collector.fail_writer_start_for_test(false);
    ASSERT_TRUE(fx.begin(4903, fx.root.prefix("run-c")));
    fx.produce(4903, /*pairs=*/1, /*flush_at_end=*/true, "recovered.cpp");
    ASSERT_TRUE(fx.wait_for_collected(2));
    fx.close(4903);
    std::string error;
    EXPECT_FALSE(fx.flush(&error)) << "the earlier host failures stay sticky";
    EXPECT_TRUE(fs::exists(fx.root.artifact("run-c"))) << "no writer consumed the queued export";

    // And the teardown that waits for the writer returns rather than hanging.
    fx.collector.finish_retained_runs();
}

/**
 * A handoff that throws returns the run's slot whichever state it was in.
 *
 * The records have already left the collector's store by then, so the failure
 * has to credit their bytes back and reclaim the slot by run identity: the
 * path copy runs while the slot still reads `Open`, and a guard that matched
 * only `Publishing` stranded it — two such failures exhausted admission.
 */
TEST(ScopeStatsRetainedRuns, AThrowingHandoffReturnsTheSlotAndTheBudget) {
    RetainedFixture fx("handoff-throw");
    // Read after the first admission, not before it: the budget is opened by
    // `run_begin`, so before that it reports nothing and no record block has
    // been charged yet either way.
    size_t fixed = 0;

    for (int attempt = 0; attempt < 2; attempt++) {
        const uint64_t epoch = 5001 + static_cast<uint64_t>(attempt);
        ASSERT_TRUE(fx.begin(epoch, fx.root.prefix(attempt == 0 ? "run-a" : "run-b")))
            << "attempt " << attempt << ": admission was exhausted by an earlier stranded slot";
        if (attempt == 0) fixed = fx.collector.retained_run_stats_for_test().charged_bytes;
        fx.produce(epoch, /*pairs=*/1, /*flush_at_end=*/true, "handoff.cpp");
        ASSERT_TRUE(fx.wait_for_collected(2));
        fx.collector.fail_handoff_for_test(true);
        fx.close(epoch);
        fx.collector.fail_handoff_for_test(false);

        const auto stats = fx.collector.retained_run_stats_for_test();
        EXPECT_EQ(stats.open_slots, 0u) << "attempt " << attempt << ": the slot was not reclaimed";
        EXPECT_EQ(stats.charged_bytes, fixed) << "attempt " << attempt << ": the record bytes were not credited back";
        EXPECT_GE(stats.write_failed, static_cast<uint64_t>(attempt + 1)) << "the failure was not recorded";
    }

    std::string error;
    EXPECT_FALSE(fx.flush(&error)) << "a failed handoff must fail the flush";
    EXPECT_FALSE(fs::exists(fx.root.artifact("run-a")));
    EXPECT_FALSE(fs::exists(fx.root.artifact("run-b")));

    // Two failures in a row did not consume the capacity, so a third run is
    // still admitted and still publishes.
    ASSERT_TRUE(fx.begin(5003, fx.root.prefix("run-c")));
    fx.produce(5003, /*pairs=*/1, /*flush_at_end=*/true, "after_handoff.cpp");
    ASSERT_TRUE(fx.wait_for_collected(2));
    fx.close(5003);
    std::string tail;
    EXPECT_FALSE(fx.flush(&tail)) << "the earlier failures stay sticky";
    EXPECT_TRUE(fs::exists(fx.root.artifact("run-c")));
}
