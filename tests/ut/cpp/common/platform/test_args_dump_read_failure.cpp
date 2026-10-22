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
};

class ArgsDumpReadFailure : public ::testing::TestWithParam<bool> {
protected:
    void SetUp() override {
        refused_source = nullptr;
        tracked_buffer_source = nullptr;
        last_buffer_shadow = nullptr;
        ASSERT_EQ(collector.initialize(1, 0, DumpArgsLevel::FULL, allocate, nullptr, release), 0);
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
};

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

INSTANTIATE_TEST_SUITE_P(
    Transport, ArgsDumpReadFailure, ::testing::Bool(), [](const ::testing::TestParamInfo<bool> &info) {
        return info.param ? "SharedState" : "HeldBuffer";
    }
);
