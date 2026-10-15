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
 * own records. Publication of the first two returns the slots needed for the
 * third admission; closing a run alone does not release its slot.
 */
TEST(ScopeStatsRetainedRuns, ConsecutiveRunsEachPublishTheirOwnRecords) {
    RetainedFixture fx("three-artifacts");
    fx.collector.pause_writer_for_test(true);
    const char *names[] = {"run-a", "run-b", "run-c"};
    const char *sites[] = {"first.cpp", "second.cpp", "third.cpp"};
    for (int i = 0; i < 3; i++) {
        SCOPED_TRACE(names[i]);
        if (i == 2) {
            ASSERT_EQ(fx.collector.retained_run_stats_for_test().published, 0u);
            fx.collector.pause_writer_for_test(false);
            std::string error;
            ASSERT_TRUE(fx.flush(&error)) << error;
            ASSERT_EQ(fx.collector.retained_run_stats_for_test().published, 2u);
        }
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
 * A recovered buffer is not the next run's to append to, empty next run or not.
 *
 * The producer clears `current_buf_ptr` only when it publishes, so a run that
 * ends without its last buffer handed over leaves that buffer named — with the
 * previous run's records still in it and the previous run's identity stamped on
 * it. Reusing a non-zero pointer skips the pop, and the pop is the only thing
 * that resets `count` or re-stamps the buffer, so nothing else in the producer
 * would ever undo either. Both halves are asserted, because they fail
 * differently: an empty second run would re-recover the first run's records
 * into its own artifact, and a non-empty third run would append behind them.
 *
 * The pointer itself must survive. Clearing it would orphan a pool buffer and
 * pushing it onto the free queue would add a second writer of a queue the
 * drain shard owns, so what changes is the header alone.
 */
TEST(ScopeStatsRetainedRuns, ARecoveredBufferIsNotTheNextRunsToAppendTo) {
    RetainedFixture fx("two-run-recovery");

    // Run a: no flush, so the boundary recovers the buffer and the pointer
    // stays behind naming it.
    ASSERT_TRUE(fx.begin(5101, fx.root.prefix("run-a")));
    fx.produce(5101, /*pairs=*/2, /*flush_at_end=*/false, "run-a.cpp");
    const uint64_t carried = fx.state()->current_buf_ptr;
    ASSERT_NE(carried, 0u) << "the producer published the buffer, so nothing is carried over";
    ASSERT_EQ(reinterpret_cast<const ScopeStatsBuffer *>(carried)->count, 4u);
    fx.close(5101);

    // Run b produces nothing at all. Admission is what re-stamps the carried
    // buffer, so this run owns an empty one rather than the previous run's.
    ASSERT_TRUE(fx.begin(5102, fx.root.prefix("run-b")));
    EXPECT_EQ(fx.state()->current_buf_ptr, carried) << "the pointer must be left in place";
    const auto *buf = reinterpret_cast<const ScopeStatsBuffer *>(carried);
    EXPECT_EQ(buf->count, 0u) << "run b would append behind run a's records";
    EXPECT_EQ(buf->run_epoch, 5102u) << "run b's records would carry run a's identity";
    fx.close(5102);

    std::string error;
    ASSERT_TRUE(fx.flush(&error)) << error;

    // Run c appends into that same carried buffer and publishes it.
    ASSERT_TRUE(fx.begin(5103, fx.root.prefix("run-c")));
    ASSERT_EQ(fx.state()->current_buf_ptr, carried);
    fx.produce(5103, /*pairs=*/3, /*flush_at_end=*/true, "run-c.cpp");
    ASSERT_TRUE(fx.wait_for_collected(6));
    fx.close(5103);
    ASSERT_TRUE(fx.flush(&error)) << error;

    const std::string a = read_file(fx.root.artifact("run-a"));
    const std::string b = read_file(fx.root.artifact("run-b"));
    const std::string c = read_file(fx.root.artifact("run-c"));
    ASSERT_FALSE(a.empty());
    ASSERT_FALSE(b.empty());
    ASSERT_FALSE(c.empty());

    EXPECT_EQ(record_lines(a), 4u) << "the recovery is what puts run a's records in run a's artifact";
    EXPECT_EQ(record_lines(b), 0u) << "run b re-recovered run a's buffer";
    EXPECT_EQ(record_lines(c), 6u) << "run c holds records that are not its own";
    EXPECT_EQ(b.find("run-a.cpp"), std::string::npos) << "run a's records reached run b's artifact";
    EXPECT_EQ(c.find("run-a.cpp"), std::string::npos) << "run a's records reached run c's artifact";
    EXPECT_NE(c.find("run-c.cpp"), std::string::npos);

    // The device counters and the host's received total agree for every run,
    // which is what a duplicated append would break.
    EXPECT_NE(a.find("\"total\": 4"), std::string::npos);
    EXPECT_NE(a.find("\"host_received_records\": 4"), std::string::npos);
    EXPECT_NE(b.find("\"total\": 0"), std::string::npos);
    EXPECT_NE(b.find("\"host_received_records\": 0"), std::string::npos);
    EXPECT_NE(c.find("\"total\": 6"), std::string::npos);
    EXPECT_NE(c.find("\"host_received_records\": 6"), std::string::npos);
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

    collector.begin_run(4601);
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

// ---------------------------------------------------------------------------
// Per-run terminal banks, mapped
// ---------------------------------------------------------------------------

namespace {

/**
 * A retained collector whose producer publishes its own per-run accounting.
 *
 * The bank address normally reaches the producer through the launching run's
 * kernel arguments, which this collector does not own, so the fixture pushes it
 * and calls the producer's own run entries — exactly where the runner will. The
 * rest is production code: the collector reserves the bank at admission, the
 * producer publishes into it, and the boundary settles it.
 */
struct MappedFixture : RetainedFixture {
    scope_runs::CapacityMetadata capacities{
        true, scope_runs::Capacities{{8, 16, 32, 64}, {5, 7, 9, 11}, {1025, 2049, 4097, 8193}, 17}
    };
    explicit MappedFixture(const char *name) :
        RetainedFixture(name) {
        EXPECT_TRUE(collector.open_run_terminal_banks(SCOPE_STATS_TERMINAL_PRODUCERS));
        EXPECT_TRUE(collector.device_publishes_run_terminal());
    }

    ~MappedFixture() { set_platform_scope_stats_run_terminal(0); }

    /**
     * Admit a run and push the bank its admission reserved, which is what the
     * host does at launch: the device is handed one resolved address per run,
     * so a successor publishes into its own bank rather than deriving one.
     */
    bool begin(uint64_t epoch, const fs::path &prefix) {
        collector.start(retained_thread_factory);
        if (!collector.run_begin(epoch, prefix.string(), nullptr, &capacities)) return false;
        const uint64_t bank = collector.run_terminal_bank_address(epoch);
        EXPECT_NE(bank, 0u) << "an admitted run resolved no terminal bank";
        set_platform_scope_stats_run_terminal(bank);
        return true;
    }

    /** One run's device side, through the producer's own entries. */
    void produce_run(uint64_t epoch, int pairs, bool flush_at_end, const char *site) {
        set_platform_run_result(/*region_base=*/0, epoch);
        scope_stats_aicpu_begin_run();
        produce(epoch, pairs, flush_at_end, site);
        scope_stats_aicpu_close_run();
    }
};

}  // namespace

TEST(ScopeStatsRunTerminal, AMappedRunIsSettledFromItsOwnProducerAccounting) {
    MappedFixture fx("mapped-settled");
    ASSERT_TRUE(fx.begin(7001, fx.root.prefix("run-a")));
    fx.produce_run(7001, /*pairs=*/3, /*flush_at_end=*/true, "mapped.cpp");
    ASSERT_TRUE(fx.wait_for_collected(6));
    fx.close(7001);

    std::string error;
    ASSERT_TRUE(fx.flush(&error)) << error;
    const auto stats = fx.collector.retained_run_stats_for_test();
    EXPECT_EQ(stats.published, 1u) << "a run whose producer reported was not published";
    EXPECT_EQ(stats.counts_unknown, 0u);
    EXPECT_EQ(stats.quarantined, 0u);
}

TEST(ScopeStatsRunTerminal, AProducerThatNeverPublishedLeavesTheRunUnknown) {
    MappedFixture fx("mapped-missing");
    ASSERT_TRUE(fx.begin(7201, fx.root.prefix("run-a")));
    // Records are produced and received, but the producer's close never runs —
    // the shape of a run killed before its operator finished. The receipts
    // alone must not be read as a complete run.
    set_platform_run_result(0, 7201);
    fx.produce(7201, /*pairs=*/2, /*flush_at_end=*/true, "mapped.cpp");
    ASSERT_TRUE(fx.wait_for_collected(4));
    fx.close(7201);

    std::string error;
    (void)fx.flush(&error);
    const auto stats = fx.collector.retained_run_stats_for_test();
    EXPECT_EQ(stats.counts_unknown, 1u) << "a run with no producer accounting was reported as counted";
    EXPECT_EQ(stats.published, 0u);
}

TEST(ScopeStatsRunTerminal, AHeldBufferIsReportedByTheRunRatherThanRecoveredByTheHost) {
    MappedFixture fx("mapped-held-buffer");
    ASSERT_TRUE(fx.begin(7301, fx.root.prefix("run-a")));
    // No flush: the producer still holds the records at close and reports them
    // as live. The host must not read that buffer — a successor adopts it.
    fx.produce_run(7301, /*pairs=*/2, /*flush_at_end=*/false, "mapped.cpp");
    const uint64_t held = fx.state()->current_buf_ptr;
    ASSERT_NE(held, 0u);
    fx.close(7301);
    EXPECT_EQ(fx.state()->current_buf_ptr, held) << "the producer's buffer was taken by the host";

    std::string error;
    (void)fx.flush(&error);
    const auto stats = fx.collector.retained_run_stats_for_test();
    EXPECT_EQ(stats.counts_unknown, 0u) << "a reported live buffer is a known loss, not an unknown one";
    EXPECT_EQ(stats.partial, 1u);
}

TEST(ScopeStatsRunTerminal, TheSuccessorsProducerOwnsTheCarriedBuffer) {
    MappedFixture fx("mapped-carried");
    ASSERT_TRUE(fx.begin(7401, fx.root.prefix("run-a")));
    fx.produce_run(7401, /*pairs=*/2, /*flush_at_end=*/false, "run-a.cpp");
    const uint64_t carried = fx.state()->current_buf_ptr;
    ASSERT_NE(carried, 0u);
    ASSERT_EQ(reinterpret_cast<const ScopeStatsBuffer *>(carried)->count, 4u);
    fx.close(7401);

    // The successor's own device init re-stamps and empties it, with no host
    // admission step in between.
    ASSERT_TRUE(fx.begin(7402, fx.root.prefix("run-b")));
    set_platform_run_result(0, 7402);
    scope_stats_aicpu_begin_run();
    const auto *buf = reinterpret_cast<const ScopeStatsBuffer *>(carried);
    EXPECT_EQ(buf->count, 0u) << "run b would append behind run a's records";
    EXPECT_EQ(buf->run_epoch, 7402u) << "run b's records would carry run a's identity";
    scope_stats_aicpu_close_run();
    fx.close(7402);

    std::string error;
    (void)fx.flush(&error);
}

TEST(ScopeStatsRunTerminal, ASecondAdmissionDoesNotEraseThePredecessorsRecords) {
    MappedFixture fx("mapped-two-open");
    // A is admitted and receiving. B is admitted before A seals, which is the
    // whole point of joining a successor to a running predecessor: A's records
    // and receipts are its own and must survive B's admission.
    ASSERT_TRUE(fx.begin(7501, fx.root.prefix("run-a")));
    fx.produce_run(7501, /*pairs=*/3, /*flush_at_end=*/true, "run-a.cpp");
    ASSERT_TRUE(fx.wait_for_collected(6));

    ASSERT_TRUE(fx.begin(7502, fx.root.prefix("run-b")));
    fx.produce_run(7502, /*pairs=*/1, /*flush_at_end=*/true, "run-b.cpp");

    // A seals after B was admitted and after B produced, and must still publish
    // its own six records rather than an artifact B emptied.
    fx.close(7501);
    fx.close(7502);

    std::string error;
    ASSERT_TRUE(fx.flush(&error)) << error;
    const auto stats = fx.collector.retained_run_stats_for_test();
    EXPECT_EQ(stats.published, 2u) << "a run lost its records to its successor's admission";
    EXPECT_EQ(stats.counts_unknown, 0u);
    EXPECT_TRUE(fs::exists(fx.root.artifact("run-a")));
    EXPECT_TRUE(fs::exists(fx.root.artifact("run-b")));
}

TEST(ScopeStatsRunTerminal, EachOpenRunKeepsItsOwnReceiptCount) {
    MappedFixture fx("mapped-receipt-isolation");
    ASSERT_TRUE(fx.begin(7601, fx.root.prefix("run-a")));
    fx.produce_run(7601, /*pairs=*/2, /*flush_at_end=*/true, "run-a.cpp");
    ASSERT_TRUE(fx.wait_for_collected(4));

    // B's arrivals are routed by the identity its producer stamped, so they are
    // counted against B even while A is still open. A's verdict therefore
    // compares its own four records against its own terminal, not against the
    // collector's running total.
    ASSERT_TRUE(fx.begin(7602, fx.root.prefix("run-b")));
    fx.produce_run(7602, /*pairs=*/5, /*flush_at_end=*/true, "run-b.cpp");
    fx.close(7601);

    std::string error;
    (void)fx.flush(&error);
    const auto stats = fx.collector.retained_run_stats_for_test();
    EXPECT_EQ(stats.counts_unknown, 0u) << "a run's receipts were charged with another run's records";
    fx.close(7602);
    (void)fx.flush(&error);
}

TEST(ScopeStatsRunTerminal, RecordsNamingNoAdmittedRunAreReportedNotAbsorbed) {
    MappedFixture fx("mapped-foreign-records");
    ASSERT_TRUE(fx.begin(7701, fx.root.prefix("run-a")));
    fx.produce_run(7701, /*pairs=*/1, /*flush_at_end=*/true, "run-a.cpp");
    ASSERT_TRUE(fx.wait_for_collected(2));
    fx.close(7701);

    // Records stamped with a run this collector never admitted: they belong to
    // no artifact, so they are neither added to an open run nor dropped in
    // silence. Produced after the only admitted run has closed, which is the
    // shape a late buffer takes.
    set_platform_run_result(0, 7799);
    scope_stats_aicpu_begin_run();
    fx.produce(7799, /*pairs=*/2, /*flush_at_end=*/true, "stray.cpp");
    scope_stats_aicpu_close_run();
    ASSERT_TRUE(fx.wait_for_collected(6)) << "the stray records never reached the collector";

    std::string error;
    EXPECT_FALSE(fx.flush(&error)) << "a record naming no admitted run was absorbed silently";
    EXPECT_NE(error.find("no admitted run"), std::string::npos) << "the loss was counted but not reported: " << error;
    EXPECT_GT(fx.collector.retained_run_stats_for_test().host_failures, 0u);
}

TEST(ScopeStatsRunTerminal, RetainsCapacitiesAcrossSuccessorInitAndDelayedExport) {
    MappedFixture fx("mapped-capacities");
    fx.collector.pause_writer_for_test(true);
    ASSERT_TRUE(fx.begin(7801, fx.root.prefix("run-a")));
    fx.produce_run(7801, 2, true, "a.cpp");
    ASSERT_TRUE(fx.wait_for_collected(4));

    fx.capacities.values = scope_runs::Capacities{{16, 32, 64, 128}, {13, 15, 17, 19}, {2049, 4097, 8193, 16385}, 29};
    ASSERT_TRUE(fx.begin(7802, fx.root.prefix("run-b")));
    fx.produce_run(7802, 1, true, "b.cpp");
    ASSERT_TRUE(fx.wait_for_collected(6));
    for (int r = 0; r < SCOPE_STATS_MAX_RING_DEPTH; r++)
        scope_stats_set_ring_capacity(r, 999, 999, 999);
    scope_stats_set_tensormap_capacity(999);
    fx.capacities.values.reset();
    fx.close(7801);
    fx.close(7802);
    fx.collector.pause_writer_for_test(false);
    std::string error;
    ASSERT_TRUE(fx.flush(&error)) << error;
    const auto a = read_file(fx.root.artifact("run-a"));
    const auto b = read_file(fx.root.artifact("run-b"));
    EXPECT_NE(a.find("\"task_window_max\": [8, 16, 32, 64]"), std::string::npos);
    EXPECT_NE(a.find("\"heap_max\": [1025, 2049, 4097, 8193]"), std::string::npos);
    EXPECT_NE(a.find("\"dep_pool_max\": [5, 7, 9, 11]"), std::string::npos);
    EXPECT_NE(a.find("\"tensormap_max\": 17"), std::string::npos);
    EXPECT_NE(a.find("\"total\": 4"), std::string::npos);
    EXPECT_NE(b.find("\"task_window_max\": [16, 32, 64, 128]"), std::string::npos);
    EXPECT_NE(b.find("\"heap_max\": [2049, 4097, 8193, 16385]"), std::string::npos);
    EXPECT_NE(b.find("\"dep_pool_max\": [13, 15, 17, 19]"), std::string::npos);
    EXPECT_NE(b.find("\"tensormap_max\": 29"), std::string::npos);
    EXPECT_NE(b.find("\"total\": 2"), std::string::npos);
}

TEST(ScopeStatsRunTerminal, EmptyCompletedRunKeepsBoundCapacities) {
    MappedFixture fx("mapped-empty-capacities");
    ASSERT_TRUE(fx.begin(7811, fx.root.prefix("run-a")));
    fx.produce_run(7811, 0, true, "empty.cpp");
    fx.close(7811);
    std::string error;
    ASSERT_TRUE(fx.flush(&error)) << error;
    const auto json = read_file(fx.root.artifact("run-a"));
    EXPECT_EQ(record_lines(json), 0u);
    EXPECT_NE(json.find("\"total\": 0"), std::string::npos);
    EXPECT_NE(json.find("\"task_window_max\": [8, 16, 32, 64]"), std::string::npos);
    EXPECT_NE(json.find("\"heap_max\": [1025, 2049, 4097, 8193]"), std::string::npos);
    EXPECT_NE(json.find("\"dep_pool_max\": [5, 7, 9, 11]"), std::string::npos);
    EXPECT_NE(json.find("\"tensormap_max\": 17"), std::string::npos);
}

TEST(ScopeStatsRunTerminal, MissingCapacityRefusesBeforeTakingCreditAndRetryKeepsItsOwnValues) {
    MappedFixture fx("mapped-missing-capacities");
    ASSERT_TRUE(fx.begin(7821, fx.root.prefix("withdrawn")));
    fx.collector.abandon_run(7821);
    EXPECT_EQ(fx.collector.run_terminal_banks_in_use(), 0u);
    const auto saved = fx.capacities;
    fx.capacities.values.reset();
    runs::AdmissionRefusal why = runs::AdmissionRefusal::None;
    EXPECT_FALSE(fx.collector.run_begin(7821, fx.root.prefix("missing").string(), &why, &fx.capacities));
    EXPECT_EQ(why, runs::AdmissionRefusal::Fatal);
    EXPECT_FALSE(fx.collector.run_begin(7821, fx.root.prefix("missing").string(), &why));
    EXPECT_EQ(fx.collector.run_terminal_banks_in_use(), 0u);
    EXPECT_FALSE(fs::exists(fx.root.artifact("withdrawn")));
    EXPECT_FALSE(fs::exists(fx.root.artifact("missing")));
    fx.capacities = saved;
    fx.capacities.values->tensormap = 31;
    ASSERT_TRUE(fx.begin(7821, fx.root.prefix("retry")));
    fx.produce_run(7821, 0, true, "empty.cpp");
    fx.close(7821);
    std::string error;
    ASSERT_TRUE(fx.flush(&error)) << error;
    EXPECT_NE(read_file(fx.root.artifact("retry")).find("\"tensormap_max\": 31"), std::string::npos);
}

TEST(ScopeStatsRunTerminal, CapacitySnapshotDoesNotAuthorizeAnUnprovedCompletion) {
    MappedFixture fx("mapped-capacities-unproved");
    ASSERT_TRUE(fx.begin(7831, fx.root.prefix("run-a")));
    fx.produce_run(7831, 1, true, "unproved.cpp");
    ASSERT_TRUE(fx.wait_for_collected(2));
    fx.close(7831, false);
    EXPECT_FALSE(fs::exists(fx.root.artifact("run-a")));
    EXPECT_EQ(fx.collector.retained_run_stats_for_test().quarantined, 1u);
    EXPECT_EQ(fx.collector.run_terminal_banks_in_use(), 1u);
    fx.collector.discard_run_terminal_bank(7831);
    EXPECT_EQ(fx.collector.run_terminal_banks_in_use(), 0u);
    EXPECT_FALSE(fs::exists(fx.root.artifact("run-a")));
}

TEST(ScopeStatsRunTerminal, RuntimeWithoutRingCapacitiesKeepsItsExistingMetadata) {
    MappedFixture fx("mapped-capacities-inapplicable");
    fx.capacities = {false, std::nullopt};
    ASSERT_TRUE(fx.begin(7841, fx.root.prefix("run-a")));
    fx.produce_run(7841, 0, true, "empty.cpp");
    fx.close(7841);
    std::string error;
    ASSERT_TRUE(fx.flush(&error)) << error;
    EXPECT_NE(read_file(fx.root.artifact("run-a")).find("\"tensormap_max\": 0"), std::string::npos);
}
