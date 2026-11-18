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

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "common/l0_profiling.h"
#include "aicpu/l0_dfx_aicpu.h"
#include "host/l0_capture_collector.h"
#include "host/l0_trace_records.h"

namespace {

struct alignas(64) TestShm {
    L0DataHeader header{};
    L0BufferState states[2]{};
};

static_assert(sizeof(TestShm) == sizeof(L0DataHeader) + 2 * sizeof(L0BufferState));

TEST(L0CapturePipeline, PublishesRangeToReadyQueue) {
    TestShm shm{};
    shm.header.num_cores = 2;
    shm.header.num_threads = 2;
    auto *state = l0_dfx_state(&shm, 1);
    state->physical_core_id = 107;
    ASSERT_EQ(l0_dfx_publish_range(&shm.header, 1, 1, 16, 16, 19, 19, false, 64, 64, L0_DFX_FLAG_LIVE), 0);
    EXPECT_EQ(shm.header.queue_tails[0], 0u);
    EXPECT_EQ(shm.header.queue_tails[1], 1u);
    const auto &entry = shm.header.queues[1][0];
    EXPECT_EQ(entry.core_index, 1u);
    EXPECT_EQ(entry.start_wptr, 16u);
    EXPECT_EQ(entry.end_wptr, 19u);
    EXPECT_EQ(entry.start_samp_wrt, 16u);
    EXPECT_EQ(entry.end_samp_wrt, 19u);
    EXPECT_EQ(entry.sequence, 0u);
    EXPECT_EQ(state->published_wptr, 19u);
    EXPECT_EQ(state->published_samp_wrt, 19u);
    EXPECT_EQ(state->batch_count, 1u);
    EXPECT_EQ(state->published_bytes, 3u);
}

TEST(L0CapturePipeline, RejectsOutOfRangeOrOverrun) {
    TestShm shm{};
    shm.header.num_cores = 1;
    shm.header.num_threads = 1;
    EXPECT_EQ(l0_dfx_publish_range(&shm.header, 0, 2, 0, 0, 4, 4, false, 64, 64, 0), -1);
    EXPECT_EQ(l0_dfx_publish_range(&shm.header, 0, 0, 0, 0, 100, 64, false, 64, 64, 0), -1);
    EXPECT_EQ(l0_dfx_state(&shm, 0)->capture_error, L0_DFX_CAPTURE_RANGE_ERROR);
}

TEST(L0CapturePipeline, RejectsFullReadyQueue) {
    TestShm shm{};
    shm.header.num_cores = 1;
    shm.header.num_threads = 1;
    // Fill the single-thread ready queue to its capacity (one slot is reserved
    // to distinguish full from empty).
    for (uint32_t i = 0; i + 1 < L0_DFX_READY_QUEUE_SIZE; ++i)
        ASSERT_EQ(l0_dfx_publish_range(&shm.header, 0, 0, 0, 0, 4, 4, false, 64, 64, 0), 0);
    EXPECT_EQ(l0_dfx_publish_range(&shm.header, 0, 0, 0, 0, 4, 4, false, 64, 64, 0), -1);
}

TEST(L0CapturePipeline, CollectorInitBeginRunFinalize) {
    L0CaptureCollector collector;
    auto allocate = [](size_t size) -> void * {
        return std::malloc(size);
    };
    auto release = [](void *ptr) -> int {
        std::free(ptr);
        return 0;
    };
    std::vector<void *> rings;
    void *ring = std::malloc(64);
    rings.push_back(ring);
    ASSERT_EQ(collector.init({107}, 1, rings, allocate, release, 0), 0);
    ASSERT_NE(collector.device_shm(), nullptr);
    auto *header = static_cast<L0DataHeader *>(collector.device_shm());
    ASSERT_EQ(header->num_cores, 1u);
    ASSERT_EQ(header->num_threads, 1u);
    ASSERT_EQ(l0_dfx_state(header, 0)->physical_core_id, 107u);

    const auto root = std::filesystem::temp_directory_path() / "simpler_l0_capture_collector_ut";
    std::filesystem::remove_all(root);
    ASSERT_EQ(collector.begin_run(42, root, 64, 64, 1500000000ull, 1200000000ull), 0);
    collector.finalize(release);
    EXPECT_FALSE(collector.output_failed());
    std::filesystem::remove_all(root);
    std::free(ring);
}

TEST(L0CapturePipeline, DrainRemainingConsumesPublishedRangesAndReconciles) {
    L0CaptureCollector collector;
    auto allocate = [](size_t size) -> void * {
        return std::malloc(size);
    };
    auto release = [](void *ptr) -> int {
        std::free(ptr);
        return 0;
    };
    std::vector<void *> rings;
    void *ring = std::malloc(64);
    std::memset(ring, 0xAB, 64);
    rings.push_back(ring);
    ASSERT_EQ(collector.init({0}, 1, rings, allocate, release, 0), 0);
    auto *header = static_cast<L0DataHeader *>(collector.device_shm());

    // A live checkpoint range followed by a terminal range, both on the core's
    // owning thread queue. The terminal drain must consume them in sequence and
    // reconcile must match the published byte count.
    ASSERT_EQ(l0_dfx_publish_range(header, 0, 0, 0, 0, 16, 16, false, 64, 64, L0_DFX_FLAG_LIVE), 0);
    ASSERT_EQ(l0_dfx_publish_range(header, 0, 0, 16, 16, 32, 32, false, 64, 64, 0), 0);
    auto *state = l0_dfx_state(header, 0);
    state->verified_samp_wrt = 32;
    state->producer_stopped = 1;

    const auto root = std::filesystem::temp_directory_path() / "simpler_l0_drain_ut";
    std::filesystem::remove_all(root);
    ASSERT_EQ(collector.begin_run(42, root, 64, 64, 1500000000ull, 1200000000ull), 0);
    collector.drain_remaining();
    ASSERT_TRUE(collector.reconcile(true));
    EXPECT_FALSE(collector.output_failed());
    collector.finalize(release);

    std::ifstream bin(root / "l0_raw" / "run_42" / "core_0.bin", std::ios::binary | std::ios::ate);
    ASSERT_TRUE(bin);
    EXPECT_EQ(bin.tellg(), std::streampos(32));

    std::filesystem::remove_all(root);
    std::free(ring);
}

TEST(L0CapturePipeline, MissingTerminalProducerMakesCaptureIncomplete) {
    L0CaptureCollector collector;
    auto allocate = [](size_t size) -> void * {
        return std::malloc(size);
    };
    auto release = [](void *ptr) -> int {
        std::free(ptr);
        return 0;
    };
    void *ring = std::malloc(64);
    ASSERT_NE(ring, nullptr);
    std::memset(ring, 0xAB, 64);
    ASSERT_EQ(collector.init({0}, 1, {ring}, allocate, release, 0), 0);
    auto *header = static_cast<L0DataHeader *>(collector.device_shm());
    ASSERT_EQ(l0_dfx_publish_range(header, 0, 0, 0, 0, 16, 16, false, 64, 64, 0), 0);
    const auto root = std::filesystem::temp_directory_path() / "simpler_l0_missing_terminal_ut";
    std::filesystem::remove_all(root);
    ASSERT_EQ(collector.begin_run(42, root, 64, 64), 0);
    collector.drain_remaining();

    EXPECT_FALSE(collector.reconcile(true));
    EXPECT_FALSE(collector.output_failed());
    collector.finalize(release);
    std::ifstream manifest(root / "l0_raw" / "run_42" / "manifest.json");
    ASSERT_TRUE(manifest);
    const std::string contents{std::istreambuf_iterator<char>(manifest), std::istreambuf_iterator<char>()};
    EXPECT_NE(contents.find("\"producer_stopped\": 0"), std::string::npos);
    EXPECT_NE(contents.find("\"raw_transport_complete\": true"), std::string::npos);
    EXPECT_NE(contents.find("\"capture_complete\": false"), std::string::npos);
    std::filesystem::remove_all(root);
    std::free(ring);
}

TEST(L0CapturePipeline, InheritedCounterWrapCopiesOnlyCurrentRunBytes) {
    L0CaptureCollector collector;
    auto allocate = [](size_t size) -> void * {
        return std::malloc(size);
    };
    auto release = [](void *ptr) -> int {
        std::free(ptr);
        return 0;
    };
    auto *ring = static_cast<uint8_t *>(std::malloc(64));
    ASSERT_NE(ring, nullptr);
    for (uint32_t i = 0; i < 64; ++i)
        ring[i] = static_cast<uint8_t>(i);
    ASSERT_EQ(collector.init({0}, 1, {ring}, allocate, release, 0), 0);
    auto *header = static_cast<L0DataHeader *>(collector.device_shm());
    auto *state = l0_dfx_state(header, 0);
    constexpr uint32_t initial = UINT32_MAX - 31;
    constexpr uint32_t current_run_bytes = 48;
    state->consumed_samp_wrt = initial;
    state->published_samp_wrt = initial;
    ASSERT_TRUE(l0_dfx_wrap_unambiguous(initial, initial, initial + current_run_bytes, 64));
    ASSERT_EQ(
        l0_dfx_publish_range(header, 0, 0, 0, initial, current_run_bytes, initial + current_run_bytes, true, 64, 64, 0),
        0
    );
    state->producer_stopped = 1;

    const auto root = std::filesystem::temp_directory_path() / "simpler_l0_counter_wrap_ut";
    std::filesystem::remove_all(root);
    ASSERT_EQ(collector.begin_run(42, root, 64, 64, 1500000000ull, 1200000000ull), 0);
    collector.drain_remaining();
    ASSERT_TRUE(collector.reconcile(true));
    collector.finalize(release);

    std::ifstream raw(root / "l0_raw" / "run_42" / "core_0.bin", std::ios::binary);
    ASSERT_TRUE(raw);
    const std::vector<uint8_t> actual{std::istreambuf_iterator<char>(raw), std::istreambuf_iterator<char>()};
    ASSERT_EQ(actual.size(), current_run_bytes);
    for (uint32_t i = 0; i < current_run_bytes; ++i)
        EXPECT_EQ(actual[i], i);

    std::filesystem::remove_all(root);
    std::free(ring);
}

TEST(L0CapturePipeline, DrainRemainingConsumesPerThreadQueuesAcrossCores) {
    L0CaptureCollector collector;
    auto allocate = [](size_t size) -> void * {
        return std::malloc(size);
    };
    auto release = [](void *ptr) -> int {
        std::free(ptr);
        return 0;
    };
    std::vector<void *> rings;
    void *ring0 = std::malloc(64);
    void *ring1 = std::malloc(64);
    std::memset(ring0, 0xAB, 64);
    std::memset(ring1, 0xAB, 64);
    rings.push_back(ring0);
    rings.push_back(ring1);
    // Two cores across two scheduler threads: core i is owned by thread i,
    // matching the checkpoint/snapshot owner-thread mapping (core % num_threads).
    ASSERT_EQ(collector.init({0, 1}, 2, rings, allocate, release, 0), 0);
    auto *header = static_cast<L0DataHeader *>(collector.device_shm());

    // A live range then a terminal range per core, sequences 0 and 1 within
    // each core, on that core's owning thread queue.
    ASSERT_EQ(l0_dfx_publish_range(header, 0, 0, 0, 0, 16, 16, false, 64, 64, L0_DFX_FLAG_LIVE), 0);
    ASSERT_EQ(l0_dfx_publish_range(header, 1, 1, 0, 0, 16, 16, false, 64, 64, L0_DFX_FLAG_LIVE), 0);
    ASSERT_EQ(l0_dfx_publish_range(header, 0, 0, 16, 16, 32, 32, false, 64, 64, 0), 0);
    ASSERT_EQ(l0_dfx_publish_range(header, 1, 1, 16, 16, 32, 32, false, 64, 64, 0), 0);
    for (uint32_t core = 0; core < 2; ++core) {
        auto *state = l0_dfx_state(header, core);
        state->verified_samp_wrt = 32;
        state->producer_stopped = 1;
    }

    const auto root = std::filesystem::temp_directory_path() / "simpler_l0_drain_multi_ut";
    std::filesystem::remove_all(root);
    ASSERT_EQ(collector.begin_run(42, root, 64, 64, 1500000000ull, 1200000000ull), 0);
    collector.drain_remaining();
    ASSERT_TRUE(collector.reconcile(true));
    EXPECT_FALSE(collector.output_failed());
    collector.finalize(release);

    for (int core = 0; core < 2; ++core) {
        std::ifstream bin(
            root / "l0_raw" / "run_42" / ("core_" + std::to_string(core) + ".bin"), std::ios::binary | std::ios::ate
        );
        ASSERT_TRUE(bin);
        EXPECT_EQ(bin.tellg(), std::streampos(32));
    }

    std::filesystem::remove_all(root);
    std::free(ring0);
    std::free(ring1);
}

TEST(L0CapturePipeline, RejectsLiveRangeOverwrittenBeforeHostCopyCompletes) {
    L0CaptureCollector collector;
    auto allocate = [](size_t size) -> void * {
        return std::malloc(size);
    };
    auto release = [](void *ptr) -> int {
        std::free(ptr);
        return 0;
    };
    void *ring = std::malloc(64);
    ASSERT_NE(ring, nullptr);
    std::memset(ring, 0xAB, 64);
    ASSERT_EQ(collector.init({0}, 1, {ring}, allocate, release, 0), 0);
    auto *header = static_cast<L0DataHeader *>(collector.device_shm());
    ASSERT_EQ(l0_dfx_publish_range(header, 0, 0, 0, 0, 16, 16, false, 64, 64, L0_DFX_FLAG_LIVE), 0);
    auto *state = l0_dfx_state(header, 0);
    state->verified_samp_wrt = 64;
    state->producer_stopped = 1;

    const auto root = std::filesystem::temp_directory_path() / "simpler_l0_overwrite_ut";
    std::filesystem::remove_all(root);
    ASSERT_EQ(collector.begin_run(42, root, 64, 64, 1500000000ull, 1200000000ull), 0);
    collector.drain_remaining();
    EXPECT_FALSE(collector.reconcile(true));
    EXPECT_TRUE(collector.output_failed());
    collector.finalize(release);
    std::ifstream bin(root / "l0_raw" / "run_42" / "core_0.bin", std::ios::binary | std::ios::ate);
    ASSERT_TRUE(bin);
    EXPECT_EQ(bin.tellg(), std::streampos(0));

    std::filesystem::remove_all(root);
    std::free(ring);
}

TEST(L0CapturePipeline, OverrunStillExportsPhysicalGmContents) {
    L0CaptureCollector collector;
    auto allocate = [](size_t size) -> void * {
        return std::malloc(size);
    };
    auto release = [](void *ptr) -> int {
        std::free(ptr);
        return 0;
    };
    std::vector<uint8_t> ring(80);
    for (size_t i = 0; i < ring.size(); ++i)
        ring[i] = static_cast<uint8_t>(i);
    ASSERT_EQ(collector.init({7}, 1, {ring.data()}, allocate, release, 0), 0);
    auto *state = l0_dfx_state(static_cast<L0DataHeader *>(collector.device_shm()), 0);
    state->capture_error = L0_DFX_CAPTURE_OVERRUN;
    state->producer_stopped = 1;

    const auto root = std::filesystem::temp_directory_path() / "simpler_l0_overrun_gm_ut";
    std::filesystem::remove_all(root);
    ASSERT_EQ(collector.begin_run(42, root, 64, 80), 0);
    ASSERT_TRUE(collector.dump_gm_rings());
    EXPECT_FALSE(collector.reconcile(true));
    collector.finalize(release);

    const auto run_dir = root / "l0_raw" / "run_42";
    std::ifstream gm(run_dir / "core_7.gm.bin", std::ios::binary);
    ASSERT_TRUE(gm);
    const std::vector<uint8_t> actual{std::istreambuf_iterator<char>(gm), std::istreambuf_iterator<char>()};
    EXPECT_EQ(actual, ring);
    std::ifstream raw(run_dir / "core_7.bin", std::ios::binary | std::ios::ate);
    ASSERT_TRUE(raw);
    EXPECT_EQ(raw.tellg(), std::streampos(0));
    std::ifstream manifest(run_dir / "manifest.json");
    ASSERT_TRUE(manifest);
    const std::string contents{std::istreambuf_iterator<char>(manifest), std::istreambuf_iterator<char>()};
    EXPECT_NE(contents.find("\"overrun\": true"), std::string::npos);
    EXPECT_NE(contents.find("\"gm_snapshot_bytes\": 80"), std::string::npos);
    EXPECT_NE(contents.find("\"gm_snapshot_complete\": true"), std::string::npos);
    EXPECT_NE(contents.find("\"raw_bytes\": 0"), std::string::npos);
    std::filesystem::remove_all(root);
}

TEST(L0CapturePipeline, StructuredEventExportFromRawFile) {
    const auto root = std::filesystem::temp_directory_path() / "simpler_l0_export_ut";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    const uint32_t words[]{
        0xE0120004u, 0xE0340003u, 0xE0000002u, 0xE0000001u, 0xF0000005u,
        0xF0010003u, 0xF0000007u, 0x40070006u, 0xEFFFFFFFu,
    };
    std::ofstream raw(root / "core_84.bin", std::ios::binary);
    for (uint32_t word : words)
        raw.write(reinterpret_cast<const char *>(&word), sizeof(word));
    raw.close();

    std::vector<uint32_t> core_types(PLATFORM_MAX_CORES, L0_DFX_CORE_TYPE_UNKNOWN);
    core_types[84] = L0_DFX_CORE_TYPE_AIV;
    ASSERT_TRUE(export_l0_event_records(root, 42, 1500000000ull, 1200000000ull, core_types));
    const auto json_path = root / "core_84.events.json";
    std::ifstream json_file(json_path);
    ASSERT_TRUE(json_file);
    const std::string json(std::istreambuf_iterator<char>{json_file}, {});
    EXPECT_NE(json.find("\"event_count\":6"), std::string::npos);
    EXPECT_NE(json.find("\"core_frequency_hz\":1200000000"), std::string::npos);
    std::filesystem::remove_all(root);
}

}  // namespace
