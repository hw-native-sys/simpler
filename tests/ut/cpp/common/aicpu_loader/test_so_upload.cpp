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
#include <signal.h>
#include <sys/resource.h>
#include <unistd.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include "aicpu/aicpu_schedule/aicpu_context.h"

namespace {
aicpu::aicpuContext_t context{1, 0, 123, 7};
aicpu::status_t context_status = aicpu::AICPU_ERROR_NONE;
}  // namespace
extern "C" void DlogRecord(int, int, const char *, ...) {}
namespace aicpu {
status_t aicpuGetContext(aicpuContext_t *ctx) {
    *ctx = context;
    return context_status;
}
uint32_t GetUniqueVfId() { return 9; }
}  // namespace aicpu
namespace simpler_dispatcher {
std::string MakeInnerSoPath(uint64_t fp, uint64_t device_id);
bool WriteBytes(const std::string &path, const char *data, uint64_t len);
}  // namespace simpler_dispatcher

namespace {
namespace fs = std::filesystem;
using namespace simpler_dispatcher;  // NOLINT(build/namespaces)

TEST(SoUpload, UsesCannOwnerDirectory) {
    context = {1, 0, 123, 7};
    context_status = aicpu::AICPU_ERROR_NONE;
    const auto path = MakeInnerSoPath(1, 12);
    EXPECT_EQ(path, "/usr/lib64/aicpu_kernels/9/aicpu_kernels_device/123_1/simpler_inner_0000000000000001_12.so");
    context.hostPid = 124;
    EXPECT_NE(MakeInnerSoPath(1, 12), path);
    context.hostPid = 123;
    context.deviceId = 0;
    EXPECT_NE(MakeInnerSoPath(1, 12), path);
    context_status = aicpu::AICPU_ERROR_FAILED;
    EXPECT_TRUE(MakeInnerSoPath(1, 12).empty());
    context_status = aicpu::AICPU_ERROR_NONE;
    context.hostPid = 0;
    EXPECT_TRUE(MakeInnerSoPath(1, 12).empty());
}

class SoUploadFiles : public testing::Test {
protected:
    std::string dir;
    void SetUp() override {
        char path[] = "/tmp/simpler-so-upload-XXXXXX";
        ASSERT_NE(mkdtemp(path), nullptr);
        dir = path;
    }
    void TearDown() override { fs::remove_all(dir); }
};

TEST_F(SoUploadFiles, ConcurrentRepeatedWritesAreAtomicAndLeaveOtherFilesAlone) {
    const std::string path = dir + "/owned.so", history = dir + "/simpler_inner_1_0.so";
    std::ofstream(history) << "history";
    std::array<bool, 4> good{true, true, true, true};
    std::array<std::thread, 4> writers;
    for (size_t i = 0; i < writers.size(); ++i)
        writers[i] = std::thread([&, i] {
            const std::string bytes(65536, 'a' + i);
            for (int k = 0; k < 20; ++k)
                if (!WriteBytes(path, bytes.data(), bytes.size())) good[i] = false;
        });
    for (auto &writer : writers)
        writer.join();
    for (bool result : good)
        EXPECT_TRUE(result);
    std::ifstream file(path, std::ios::binary);
    const std::string actual((std::istreambuf_iterator<char>(file)), {});
    ASSERT_EQ(actual.size(), 65536);
    EXPECT_GE(actual[0], 'a');
    EXPECT_LE(actual[0], 'd');
    EXPECT_EQ(actual, std::string(actual.size(), actual[0]));
    EXPECT_TRUE(fs::exists(history));
    EXPECT_EQ(std::distance(fs::directory_iterator(dir), fs::directory_iterator()), 2);
}

TEST_F(SoUploadFiles, FailedUploadPreservesExistingTargetAndAllowsRetry) {
    const std::string path = dir + "/owned.so";
    const std::string bytes(1024 * 1024, 'x');
    for (bool existing : {false, true}) {
        SCOPED_TRACE(existing);
        if (existing) std::ofstream(path) << "old";
        ASSERT_EXIT(
            {
                signal(SIGXFSZ, SIG_IGN);
                struct rlimit limit;
                limit.rlim_cur = limit.rlim_max = 4096;
                if (setrlimit(RLIMIT_FSIZE, &limit) != 0) _exit(1);
                _exit(WriteBytes(path, bytes.data(), bytes.size()) ? 2 : 0);
            },
            testing::ExitedWithCode(0), ""
        );
        EXPECT_EQ(std::distance(fs::directory_iterator(dir), fs::directory_iterator()), existing ? 1 : 0);
        if (existing) {
            std::ifstream previous(path, std::ios::binary);
            EXPECT_EQ(std::string(std::istreambuf_iterator<char>(previous), {}), "old");
        }
        ASSERT_TRUE(WriteBytes(path, bytes.data(), bytes.size()));
        std::ifstream retried(path, std::ios::binary);
        EXPECT_EQ(std::string(std::istreambuf_iterator<char>(retried), {}), bytes);
        ASSERT_TRUE(fs::remove(path));
    }
}
}  // namespace
