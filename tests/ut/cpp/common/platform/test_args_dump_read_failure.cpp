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

#include "host/args_dump_collector.h"

#include <gtest/gtest.h>

#include "aicpu/args_dump_aicpu.h"
#include "aicpu/device_run_result_base_aicpu.h"
#include "host/profiling_copy.h"
#include "support/test_task_id.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <unistd.h>

namespace {
const volatile void *refused_source = nullptr;
void *last_buffer_shadow = nullptr;
const volatile void *tracked_buffer_source = nullptr;
int refused_copies = 0;
void *allocate(size_t bytes) { return std::calloc(1, bytes); }
int release(void *ptr) {
    std::free(ptr);
    return 0;
}
}  // namespace

int profiling_copy_to_device(volatile void *dst, const void *src, size_t bytes) {
    std::memcpy(const_cast<void *>(dst), src, bytes);
    return 0;
}
int profiling_copy_from_device(volatile void *dst, const volatile void *src, size_t bytes) {
    if (src == refused_source) {
        ++refused_copies;
        return -1;
    }
    std::memcpy(const_cast<void *>(dst), const_cast<const void *>(src), bytes);
    if (src == tracked_buffer_source) last_buffer_shadow = const_cast<void *>(dst);
    return 0;
}
std::function<int(void *, const void *, size_t)> profiling_copy_to_device_or_null() {
    return profiling_copy_to_device_for_ops;
}
std::function<int(void *, const void *, size_t)> profiling_copy_from_device_or_null() {
    return profiling_copy_from_device_for_ops;
}

class TestArgsDumpCollector : public ArgsDumpCollector {
public:
    void *host_ptr() const { return shm_host_; }
    void retire_unreadable_buffer() { note_handoff_retired({}); }
};

class ArgsDumpReadFailure : public ::testing::TestWithParam<bool> {
protected:
    void SetUp() override {
        refused_source = nullptr;
        tracked_buffer_source = nullptr;
        last_buffer_shadow = nullptr;
        ASSERT_EQ(collector.initialize(dump_thread_count(), 0, DumpArgsLevel::FULL, allocate, nullptr, release), 0);
        device = collector.get_dump_shm_device_ptr();
        host = collector.host_ptr();
        ASSERT_NE(device, host);
        set_platform_dump_base(reinterpret_cast<uint64_t>(device));
        set_platform_dump_args_run_terminal(0);
        set_dump_args_enabled(true);
        root = std::filesystem::temp_directory_path() /
               ("args_dump_copy_" + std::to_string(::getpid()) + "_" + std::to_string(GetParam()));
        std::filesystem::remove_all(root);
    }
    void TearDown() override {
        refused_source = nullptr;
        tracked_buffer_source = nullptr;
        set_dump_args_enabled(false);
        set_platform_dump_base(0);
        set_platform_run_result(0, 0);
        collector.finalize(nullptr, release);
        std::filesystem::remove_all(root);
    }
    TestArgsDumpCollector collector;
    void *device = nullptr;
    void *host = nullptr;
    std::filesystem::path root;

    virtual int dump_thread_count() const { return 1; }

    void begin(uint64_t epoch) {
        collector.begin_run(root.string(), DumpArgsLevel::FULL, epoch);
        set_platform_run_result(0, epoch);
        dump_args_init(dump_thread_count(), DumpArgsLevel::FULL);
    }

    int record(uint32_t value) {
        ArgsDumpInfo info{};
        info.task_id = simpler::ut::test_task_id(value);
        info.kind = static_cast<uint8_t>(ArgsDumpKind::SCALAR);
        info.dtype = static_cast<uint8_t>(DataType::UINT32);
        info.scalar_value = value;
        return dump_arg_record(0, info);
    }

    void collect_published_record() {
        ASSERT_EQ(record(41), 0);
        dump_args_flush(0);
        auto *header = get_dump_header(device);
        ASSERT_EQ(header->queue_tails[0], 1u);
        const auto &entry = header->queues[0][0];
        DumpMetaBuffer snapshot{};
        ASSERT_EQ(
            profiling_copy_from_device(&snapshot, reinterpret_cast<void *>(entry.buffer_ptr), sizeof(snapshot)), 0
        );
        DumpReadyBufferInfo info{};
        info.thread_index = entry.thread_index;
        info.dev_buffer_ptr = reinterpret_cast<void *>(entry.buffer_ptr);
        info.host_buffer_ptr = &snapshot;
        info.buffer_seq = entry.buffer_seq;
        collector.on_buffer_collected(info, 0);
    }

    std::string export_manifest() {
        EXPECT_EQ(collector.export_dump_files(), 0);
        std::ifstream json(root / "args_dump" / "args_dump.json");
        EXPECT_TRUE(json.is_open());
        return {std::istreambuf_iterator<char>(json), std::istreambuf_iterator<char>()};
    }
};

TEST_P(ArgsDumpReadFailure, SuccessfulReconcileReportsCompleteZeroLossCount) {
    begin(51);
    ASSERT_EQ(record(42), 0);
    collector.reconcile_counters();
    const auto manifest = export_manifest();
    EXPECT_NE(manifest.find("\"value\": 42"), std::string::npos) << manifest;
    EXPECT_NE(manifest.find("\"dropped_records\": 0,"), std::string::npos) << manifest;
    EXPECT_NE(manifest.find("\"dropped_records_complete\": true"), std::string::npos) << manifest;
}

TEST_P(ArgsDumpReadFailure, FailedReadPreservesCollectedRecordsAndMarksLossCountIncomplete) {
    begin(52);
    collect_published_record();
    ASSERT_EQ(record(42), 0);
    auto *state = get_dump_buffer_state(device, 0);
    state->run_dropped_record_count = 7;
    refused_source = GetParam() ? device : reinterpret_cast<void *>(state->current_buf_ptr);
    refused_copies = 0;
    collector.reconcile_counters();
    EXPECT_EQ(refused_copies, 1);
    const auto manifest = export_manifest();
    EXPECT_NE(manifest.find("\"value\": 41"), std::string::npos) << manifest;
    EXPECT_EQ(manifest.find("\"value\": 42"), std::string::npos) << manifest;
    EXPECT_NE(manifest.find(GetParam() ? "\"dropped_records\": 0," : "\"dropped_records\": 7,"), std::string::npos)
        << manifest;
    EXPECT_NE(manifest.find("\"dropped_records_complete\": false"), std::string::npos) << manifest;
}

TEST_P(ArgsDumpReadFailure, ExportWithoutReconcileReportsIncompleteLossCount) {
    begin(53);
    collect_published_record();
    const auto manifest = export_manifest();
    EXPECT_NE(manifest.find("\"value\": 41"), std::string::npos) << manifest;
    EXPECT_NE(manifest.find("\"dropped_records\": 0,"), std::string::npos) << manifest;
    EXPECT_NE(manifest.find("\"dropped_records_complete\": false"), std::string::npos) << manifest;
}

TEST_P(ArgsDumpReadFailure, DeviceRefusedRecordIsCountedExactly) {
    begin(54);
    collect_published_record();
    auto *state = get_dump_buffer_state(device, 0);
    state->free_queue.buffer_ptrs[state->free_queue.head % PLATFORM_DUMP_SLOT_COUNT] = 0;
    EXPECT_NE(record(42), 0);
    ASSERT_EQ(state->run_dropped_record_count, 1u);
    collector.reconcile_counters();
    const auto manifest = export_manifest();
    EXPECT_NE(manifest.find("\"dropped_records\": 1,"), std::string::npos) << manifest;
    EXPECT_NE(manifest.find("\"dropped_records_complete\": true"), std::string::npos) << manifest;
}

TEST_P(ArgsDumpReadFailure, RetryReplacesCounterSnapshotAndCanCompleteLossCount) {
    begin(55);
    collect_published_record();
    ASSERT_EQ(record(42), 0);
    auto *state = get_dump_buffer_state(device, 0);
    state->run_dropped_record_count = 7;
    refused_source = GetParam() ? device : reinterpret_cast<void *>(state->current_buf_ptr);
    collector.reconcile_counters();
    refused_source = nullptr;
    collector.reconcile_counters();
    const auto manifest = export_manifest();
    EXPECT_NE(manifest.find("\"value\": 42"), std::string::npos) << manifest;
    EXPECT_NE(manifest.find("\"dropped_records\": 7,"), std::string::npos) << manifest;
    EXPECT_NE(manifest.find("\"dropped_records_complete\": true"), std::string::npos) << manifest;
}

TEST_P(ArgsDumpReadFailure, RetiredTransportBufferKeepsLossCountIncompleteAcrossReconcileRetry) {
    begin(56);
    collect_published_record();
    collector.retire_unreadable_buffer();
    collector.reconcile_counters();
    collector.reconcile_counters();
    const auto manifest = export_manifest();
    EXPECT_NE(manifest.find("\"dropped_records_complete\": false"), std::string::npos) << manifest;
}

TEST_P(ArgsDumpReadFailure, NewRunDoesNotInheritCompleteLossAccounting) {
    begin(57);
    ASSERT_EQ(record(42), 0);
    collector.reconcile_counters();
    EXPECT_NE(export_manifest().find("\"dropped_records_complete\": true"), std::string::npos);
    begin(58);
    collect_published_record();
    const auto manifest = export_manifest();
    EXPECT_NE(manifest.find("\"dropped_records_complete\": false"), std::string::npos) << manifest;
}

TEST_P(ArgsDumpReadFailure, UnattributableOrMalformedHeldBufferLeavesLossCountIncomplete) {
    begin(59);
    collect_published_record();
    ASSERT_EQ(record(42), 0);
    auto *buffer = reinterpret_cast<DumpMetaBuffer *>(get_dump_buffer_state(device, 0)->current_buf_ptr);
    if (GetParam()) {
        buffer->run_epoch++;
    } else {
        buffer->count = PLATFORM_DUMP_RECORDS_PER_BUFFER + 1;
    }
    collector.reconcile_counters();
    const auto manifest = export_manifest();
    EXPECT_EQ(manifest.find("\"value\": 42"), std::string::npos) << manifest;
    EXPECT_NE(manifest.find("\"dropped_records_complete\": false"), std::string::npos) << manifest;
}

TEST_P(ArgsDumpReadFailure, SaturatedRunCounterIsOnlyALowerBound) {
    begin(60);
    collect_published_record();
    get_dump_buffer_state(device, 0)->run_dropped_record_count = std::numeric_limits<uint32_t>::max();
    collector.reconcile_counters();
    const auto manifest = export_manifest();
    EXPECT_NE(
        manifest.find("\"dropped_records\": " + std::to_string(std::numeric_limits<uint32_t>::max()) + ","),
        std::string::npos
    ) << manifest;
    EXPECT_NE(manifest.find("\"dropped_records_complete\": false"), std::string::npos) << manifest;
}

TEST_P(ArgsDumpReadFailure, FailedReadCannotRecoverCachedRecordsAndCanRetry) {
    auto record = [](uint32_t value) {
        ArgsDumpInfo info{};
        info.task_id = simpler::ut::test_task_id(0x900);
        info.kind = static_cast<uint8_t>(ArgsDumpKind::SCALAR);
        info.dtype = static_cast<uint8_t>(DataType::UINT32);
        info.scalar_value = value;
        EXPECT_EQ(dump_arg_record(0, info), 0);
    };
    collector.begin_run((root / "old").string(), DumpArgsLevel::FULL, 11);
    set_platform_run_result(0, 11);
    dump_args_init(1, DumpArgsLevel::FULL);
    record(11);
    tracked_buffer_source = reinterpret_cast<const void *>(get_dump_buffer_state(device, 0)->current_buf_ptr);
    collector.reconcile_counters();
    ASSERT_EQ(collector.export_dump_files(), 0);
    ASSERT_NE(last_buffer_shadow, nullptr);

    collector.begin_run((root / "new").string(), DumpArgsLevel::FULL, 22);
    set_platform_run_result(0, 22);
    dump_args_init(1, DumpArgsLevel::FULL);
    record(42);
    // Matching identity isolates the copy-result guard from the epoch guard:
    // cached bytes alone are not evidence that a recovery read succeeded.
    static_cast<DumpMetaBuffer *>(last_buffer_shadow)->run_epoch = 22;
    get_dump_buffer_state(host, 0)->run_epoch = 22;
    refused_source =
        GetParam() ? device : reinterpret_cast<const void *>(get_dump_buffer_state(device, 0)->current_buf_ptr);
    refused_copies = 0;
    collector.reconcile_counters();
    EXPECT_EQ(refused_copies, 1);
    EXPECT_EQ(collector.export_dump_files(), 0);
    ASSERT_FALSE(std::filesystem::exists(root / "new" / "args_dump" / "args_dump.json"));

    refused_source = nullptr;
    collector.reconcile_counters();
    ASSERT_EQ(collector.export_dump_files(), 0);
    std::ifstream json(root / "new" / "args_dump" / "args_dump.json");
    const std::string manifest{std::istreambuf_iterator<char>(json), std::istreambuf_iterator<char>()};
    EXPECT_NE(manifest.find("\"value\": 42"), std::string::npos) << manifest;
}

class ArgsDumpMultiLaneReadFailure : public ArgsDumpReadFailure {
protected:
    int dump_thread_count() const override { return 2; }

    void record_tensor(int lane, uint32_t value) {
        ArgsDumpInfo info{};
        info.task_id = simpler::ut::test_task_id(0xB00 + lane);
        info.arg_index = static_cast<uint32_t>(lane);
        info.kind = static_cast<uint8_t>(ArgsDumpKind::TENSOR);
        info.dtype = static_cast<uint8_t>(DataType::UINT32);
        info.capture_payload = 1;
        info.buffer_addr = reinterpret_cast<uint64_t>(&value);
        info.ndims = 1;
        info.shapes[0] = info.strides[0] = 1;
        ASSERT_EQ(dump_arg_record(lane, info), 0);
    }

    size_t occurrences(const std::string &text, const std::string &needle) {
        size_t count = 0;
        for (size_t pos = text.find(needle); pos != std::string::npos; pos = text.find(needle, pos + needle.size())) {
            count++;
        }
        return count;
    }

    void expect_payloads(uint32_t first, uint32_t second) {
        std::ifstream bin(root / "args_dump" / "args.bin", std::ios::binary);
        ASSERT_TRUE(bin.is_open());
        const std::string bytes{std::istreambuf_iterator<char>(bin), std::istreambuf_iterator<char>()};
        const std::string expected = std::string(reinterpret_cast<const char *>(&first), sizeof(first)) +
                                     std::string(reinterpret_cast<const char *>(&second), sizeof(second));
        EXPECT_EQ(bytes, expected);
    }
};

TEST_P(ArgsDumpMultiLaneReadFailure, PerLaneDroppedCountsAreSummed) {
    begin(71);
    record_tensor(0, 41);
    record_tensor(1, 42);
    get_dump_buffer_state(device, 0)->run_dropped_record_count = 3;
    get_dump_buffer_state(device, 1)->run_dropped_record_count = 7;

    collector.reconcile_counters();
    const auto manifest = export_manifest();
    EXPECT_NE(manifest.find("\"dropped_records\": 10,"), std::string::npos) << manifest;
    EXPECT_NE(manifest.find("\"dropped_records_complete\": true"), std::string::npos) << manifest;
    EXPECT_NE(manifest.find("\"total_args\": 2,"), std::string::npos) << manifest;
    expect_payloads(41, 42);
}

TEST_P(ArgsDumpMultiLaneReadFailure, RetryRecoversEachHeldBufferExactlyOnce) {
    begin(72);
    record_tensor(0, 41);
    record_tensor(1, 42);
    get_dump_buffer_state(device, 0)->run_dropped_record_count = 3;
    auto *second_state = get_dump_buffer_state(device, 1);
    second_state->run_dropped_record_count = 7;
    const int failing_lane = GetParam() ? 1 : 0;
    refused_source = reinterpret_cast<void *>(get_dump_buffer_state(device, failing_lane)->current_buf_ptr);
    refused_copies = 0;
    collector.reconcile_counters();
    EXPECT_EQ(refused_copies, 1);

    refused_source = nullptr;
    collector.reconcile_counters();
    collector.reconcile_counters();
    const auto manifest = export_manifest();
    EXPECT_NE(manifest.find("\"total_args\": 2,"), std::string::npos) << manifest;
    EXPECT_EQ(occurrences(manifest, "\"arg_index\": 0"), 1u) << manifest;
    EXPECT_EQ(occurrences(manifest, "\"arg_index\": 1"), 1u) << manifest;
    EXPECT_NE(manifest.find("\"dropped_records\": 10,"), std::string::npos) << manifest;
    EXPECT_NE(manifest.find("\"dropped_records_complete\": true"), std::string::npos) << manifest;
    expect_payloads(GetParam() ? 41 : 42, GetParam() ? 42 : 41);
}

TEST_P(ArgsDumpMultiLaneReadFailure, UnknownLanePreservesOtherLaneKnownLossCount) {
    begin(73);
    record_tensor(0, 41);
    record_tensor(1, 42);
    const int unknown_lane = GetParam() ? 1 : 0;
    auto *unknown_state = get_dump_buffer_state(device, unknown_lane);
    unknown_state->run_epoch = 72;
    unknown_state->run_dropped_record_count = 3;
    get_dump_buffer_state(device, 1 - unknown_lane)->run_dropped_record_count = 7;

    collector.reconcile_counters();
    const auto manifest = export_manifest();
    EXPECT_NE(manifest.find("\"total_args\": 1,"), std::string::npos) << manifest;
    EXPECT_NE(manifest.find("\"dropped_records\": 7,"), std::string::npos) << manifest;
    EXPECT_NE(manifest.find("\"dropped_records_complete\": false"), std::string::npos) << manifest;
}

TEST_P(ArgsDumpMultiLaneReadFailure, UnreadHeldBufferKeepsKnownCounterSumIncomplete) {
    begin(76);
    record_tensor(0, 41);
    record_tensor(1, 42);
    get_dump_buffer_state(device, 0)->run_dropped_record_count = 3;
    get_dump_buffer_state(device, 1)->run_dropped_record_count = 7;
    const int unread_lane = GetParam() ? 1 : 0;
    refused_source = reinterpret_cast<void *>(get_dump_buffer_state(device, unread_lane)->current_buf_ptr);

    collector.reconcile_counters();
    const auto manifest = export_manifest();
    EXPECT_NE(manifest.find("\"total_args\": 1,"), std::string::npos) << manifest;
    EXPECT_NE(manifest.find("\"dropped_records\": 10,"), std::string::npos) << manifest;
    EXPECT_NE(manifest.find("\"dropped_records_complete\": false"), std::string::npos) << manifest;
}

TEST_P(ArgsDumpMultiLaneReadFailure, SaturatedLaneKeepsCombinedCountCappedAndIncomplete) {
    begin(77);
    record_tensor(0, 41);
    record_tensor(1, 42);
    const int saturated_lane = GetParam() ? 1 : 0;
    get_dump_buffer_state(device, saturated_lane)->run_dropped_record_count = std::numeric_limits<uint32_t>::max();
    get_dump_buffer_state(device, 1 - saturated_lane)->run_dropped_record_count = 7;

    collector.reconcile_counters();
    const auto manifest = export_manifest();
    EXPECT_NE(
        manifest.find("\"dropped_records\": " + std::to_string(std::numeric_limits<uint32_t>::max()) + ","),
        std::string::npos
    ) << manifest;
    EXPECT_NE(manifest.find("\"dropped_records_complete\": false"), std::string::npos) << manifest;
    EXPECT_NE(manifest.find("\"total_args\": 2,"), std::string::npos) << manifest;
}

TEST_P(ArgsDumpMultiLaneReadFailure, NewRunRecoversReusedHeldBuffers) {
    begin(74);
    record_tensor(0, 41);
    record_tensor(1, 42);
    const uint64_t held0 = get_dump_buffer_state(device, 0)->current_buf_ptr;
    const uint64_t held1 = get_dump_buffer_state(device, 1)->current_buf_ptr;
    collector.reconcile_counters();
    EXPECT_NE(export_manifest().find("\"total_args\": 2,"), std::string::npos);

    begin(75);
    record_tensor(0, 51);
    record_tensor(1, 52);
    ASSERT_EQ(get_dump_buffer_state(device, 0)->current_buf_ptr, held0);
    ASSERT_EQ(get_dump_buffer_state(device, 1)->current_buf_ptr, held1);
    collector.reconcile_counters();
    const auto manifest = export_manifest();
    EXPECT_NE(manifest.find("\"total_args\": 2,"), std::string::npos) << manifest;
    EXPECT_NE(manifest.find("\"dropped_records_complete\": true"), std::string::npos) << manifest;
    expect_payloads(51, 52);
}

INSTANTIATE_TEST_SUITE_P(
    Transport, ArgsDumpReadFailure, ::testing::Bool(), [](const ::testing::TestParamInfo<bool> &info) {
        return info.param ? "SharedState" : "HeldBuffer";
    }
);

INSTANTIATE_TEST_SUITE_P(
    HeldRecovery, ArgsDumpMultiLaneReadFailure, ::testing::Bool(), [](const ::testing::TestParamInfo<bool> &info) {
        return info.param ? "Lane1" : "Lane0";
    }
);
