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
#include <sys/statvfs.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include "aicpu/aicpu_schedule/aicpu_context.h"
#include "aicpu_loader/device/aicpu_dispatcher.h"

namespace {
aicpu::aicpuContext_t context{1, 0, 123, 7};
aicpu::status_t context_status = aicpu::AICPU_ERROR_NONE;
#if defined(__linux__)
int fault_write_errno;
int fault_close_errno;
int fault_link_errno;
bool fail_capacity;
int capacity_calls;
#endif
}  // namespace

#if defined(__linux__)
extern "C" ssize_t __real_write(int, const void *, size_t);
extern "C" int __real_close(int);
extern "C" int __real_link(const char *, const char *);
extern "C" int __real_statvfs(const char *, struct statvfs *);
extern "C" ssize_t __wrap_write(int fd, const void *data, size_t count) {
    if (fault_write_errno == 0) return __real_write(fd, data, count);
    errno = fault_write_errno;
    return -1;
}
extern "C" int __wrap_close(int fd) {
    const int rc = __real_close(fd);
    if (fault_close_errno == 0) return rc;
    errno = fault_close_errno;
    return -1;
}
extern "C" int __wrap_link(const char *source, const char *target) {
    if (fault_link_errno == 0) return __real_link(source, target);
    errno = fault_link_errno;
    return -1;
}
extern "C" int __wrap_statvfs(const char *path, struct statvfs *capacity) {
    if (fault_write_errno == 0) return __real_statvfs(path, capacity);
    ++capacity_calls;
    if (fail_capacity) {
        errno = ENOSYS;
        return -1;
    }
    *capacity = {};
    capacity->f_frsize = 4096;
    capacity->f_ffree = 11;
    return 0;
}
#endif
extern "C" void DlogRecord(int, int, const char *format, ...) {
    va_list ap;
    va_start(ap, format);
    vfprintf(stderr, format, ap);
    va_end(ap);
}
namespace aicpu {
status_t aicpuGetContext(aicpuContext_t *ctx) {
    *ctx = context;
    return context_status;
}
uint32_t GetUniqueVfId() { return 9; }
}  // namespace aicpu

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

TEST(SoUpload, ContextFailurePublishesWithoutUsingAnUnrelatedErrno) {
    BootstrapDeviceArgs device;
    BootstrapResult result;
    struct {
        uint64_t unused[5]{};
        void *device_args;
    } args{};
    const char bytes[] = "runtime";
    device.device_id = 12;
    device.inner_so_bin = reinterpret_cast<uint64_t>(bytes);
    device.inner_so_len = sizeof(bytes);
    device.result_addr = reinterpret_cast<uint64_t>(&result);
    args.device_args = &device;
    context_status = aicpu::AICPU_ERROR_FAILED;
    errno = ENOSPC;
    EXPECT_EQ(DynTileFwkBackendKernelServerInit(&args), 0u);
    EXPECT_EQ(result.status, BootstrapStatus::Failed);
    EXPECT_EQ(result.stage, BootstrapStage::Context);
    EXPECT_EQ(result.context_error, static_cast<int32_t>(aicpu::AICPU_ERROR_FAILED));
    EXPECT_EQ(result.error_number, 0);
    EXPECT_EQ(result.device_id, 12);
    EXPECT_EQ(result.requested_bytes, sizeof(bytes));
    device.result_addr = 0;
    EXPECT_EQ(DynTileFwkBackendKernelServerInit(&args), 1u);
    context_status = aicpu::AICPU_ERROR_NONE;
}

class SoUploadFiles : public testing::Test {
protected:
    std::string dir;
    void SetUp() override {
#if defined(__linux__)
        fault_write_errno = fault_close_errno = fault_link_errno = capacity_calls = 0;
        fail_capacity = false;
#endif
        char path[] = "/tmp/simpler-so-upload-XXXXXX";
        ASSERT_NE(mkdtemp(path), nullptr);
        dir = path;
    }
    void TearDown() override {
#if defined(__linux__)
        fault_write_errno = fault_close_errno = fault_link_errno = 0;
#endif
        fs::remove_all(dir);
    }
};

#if defined(__linux__)
TEST_F(SoUploadFiles, CapacityErrorsPublishOneBoundedQueryAndPreserveTarget) {
    const std::string path = dir + "/owned.so", previous_path = dir + "/history.so";
    std::ofstream(previous_path) << "old";
    for (int error : {ENOSPC, EDQUOT}) {
        SCOPED_TRACE(error);
        fault_write_errno = error;
        capacity_calls = 0;
        BootstrapResult result;
        EXPECT_FALSE(WriteBytes(path, "x", 1, result));
        EXPECT_EQ(result.status, BootstrapStatus::Failed);
        EXPECT_EQ(result.stage, BootstrapStage::Write);
        EXPECT_EQ(result.error_number, error);
        EXPECT_STREQ(result.path, path.c_str());
        EXPECT_EQ(result.requested_bytes, 1u);
        EXPECT_EQ(result.written_bytes, 0u);
        EXPECT_EQ(capacity_calls, 1);
        EXPECT_EQ(result.capacity_valid, 1u);
        EXPECT_EQ(result.capacity_errno, 0);
        EXPECT_EQ(result.block_bytes, 4096u);
        EXPECT_EQ(result.free_blocks, 0u);
        EXPECT_EQ(result.available_blocks, 0u);
        EXPECT_EQ(result.free_inodes, 11u);
        std::ifstream previous(previous_path);
        EXPECT_EQ(std::string(std::istreambuf_iterator<char>(previous), {}), "old");
        EXPECT_EQ(std::distance(fs::directory_iterator(dir), fs::directory_iterator()), 1);
    }
}

TEST_F(SoUploadFiles, CapacityQueryFailureDoesNotReplaceTheUploadErrno) {
    fault_write_errno = ENOSPC;
    fail_capacity = true;
    BootstrapResult result;
    EXPECT_FALSE(WriteBytes(dir + "/owned.so", "x", 1, result));
    EXPECT_EQ(result.status, BootstrapStatus::Failed);
    EXPECT_EQ(result.stage, BootstrapStage::Write);
    EXPECT_EQ(result.error_number, ENOSPC);
    EXPECT_EQ(result.capacity_errno, ENOSYS);
    EXPECT_EQ(result.capacity_valid, 0u);
    EXPECT_EQ(capacity_calls, 1);
    EXPECT_TRUE(fs::is_empty(dir));
}
#endif

TEST_F(SoUploadFiles, FailedCreateReportsItsStageAndErrno) {
    BootstrapResult result;
    EXPECT_FALSE(WriteBytes(dir + "/missing/owned.so", "x", 1, result));
    EXPECT_EQ(result.status, BootstrapStatus::Failed);
    EXPECT_EQ(result.stage, BootstrapStage::Create);
    EXPECT_EQ(result.error_number, ENOENT);
    EXPECT_EQ(result.requested_bytes, 1u);
    EXPECT_EQ(result.written_bytes, 0u);
}

TEST_F(SoUploadFiles, FailedWritePreservesErrnoBeforeCleanup) {
    const std::string path = dir + "/owned.so", bytes(1024 * 1024, 'x');
    ASSERT_EXIT(
        {
            signal(SIGXFSZ, SIG_IGN);
            struct rlimit limit;
            limit.rlim_cur = limit.rlim_max = 4096;
            if (setrlimit(RLIMIT_FSIZE, &limit) != 0) _exit(1);
            BootstrapResult result;
            if (WriteBytes(path, bytes.data(), bytes.size(), result)) _exit(2);
            _exit(
                result.status == BootstrapStatus::Failed && result.stage == BootstrapStage::Write &&
                        result.error_number == EFBIG && result.requested_bytes == bytes.size() &&
                        result.written_bytes == limit.rlim_cur ?
                    0 :
                    3
            );
        },
        testing::ExitedWithCode(0), ""
    );
}

TEST_F(SoUploadFiles, SequentialGenerationsKeepOneImmutableImage) {
    const std::string path = dir + "/owned.so";
    const std::string bytes(65536, 'a');
    BootstrapResult result;
    ASSERT_TRUE(WriteBytes(path, bytes.data(), bytes.size(), result));
    struct stat original{};
    ASSERT_EQ(stat(path.c_str(), &original), 0);
    for (int generation = 0; generation < 8; ++generation) {
        BootstrapResult hit;
        ASSERT_TRUE(WriteBytes(path, bytes.data(), bytes.size(), hit));
        EXPECT_EQ(hit.stage, BootstrapStage::CacheHit);
        EXPECT_EQ(hit.written_bytes, 0u);
        struct stat current{};
        ASSERT_EQ(stat(path.c_str(), &current), 0);
        EXPECT_EQ(current.st_ino, original.st_ino);
        EXPECT_EQ(current.st_dev, original.st_dev);
        EXPECT_EQ(std::distance(fs::directory_iterator(dir), fs::directory_iterator()), 1);
    }
    const std::string collision(bytes.size(), 'b');
    BootstrapResult collision_result;
    EXPECT_FALSE(WriteBytes(path, collision.data(), collision.size(), collision_result));
    EXPECT_EQ(collision_result.stage, BootstrapStage::Verify);
    EXPECT_EQ(collision_result.error_number, EINVAL);
    std::ifstream file(path, std::ios::binary);
    EXPECT_TRUE(std::string(std::istreambuf_iterator<char>(file), {}) == bytes);
}

TEST_F(SoUploadFiles, ConcurrentPublicationChoosesOneImmutableImageAndLeavesOtherFilesAlone) {
    const std::string path = dir + "/owned.so", history = dir + "/simpler_inner_1_0.so";
    std::ofstream(history) << "history";
    std::array<int, 4> successes{};
    std::array<std::thread, 4> writers;
    for (size_t i = 0; i < writers.size(); ++i)
        writers[i] = std::thread([&, i] {
            const std::string bytes(65536, 'a' + i);
            BootstrapResult result;
            for (int k = 0; k < 20; ++k)
                if (WriteBytes(path, bytes.data(), bytes.size(), result)) ++successes[i];
        });
    for (auto &writer : writers)
        writer.join();
    EXPECT_EQ(std::count(successes.begin(), successes.end(), 20), 1);
    EXPECT_EQ(std::count(successes.begin(), successes.end(), 0), 3);
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
                BootstrapResult result;
                _exit(WriteBytes(path, bytes.data(), bytes.size(), result) ? 2 : 0);
            },
            testing::ExitedWithCode(0), ""
        );
        EXPECT_EQ(std::distance(fs::directory_iterator(dir), fs::directory_iterator()), existing ? 1 : 0);
        if (existing) {
            std::ifstream previous(path, std::ios::binary);
            EXPECT_EQ(std::string(std::istreambuf_iterator<char>(previous), {}), "old");
        }
        BootstrapResult result;
        if (existing) {
            EXPECT_FALSE(WriteBytes(path, bytes.data(), bytes.size(), result));
            EXPECT_EQ(result.stage, BootstrapStage::Verify);
            EXPECT_EQ(result.error_number, EINVAL);
            ASSERT_TRUE(fs::remove(path));
        }
        ASSERT_TRUE(WriteBytes(path, bytes.data(), bytes.size(), result));
        EXPECT_EQ(result.written_bytes, bytes.size());
        std::ifstream retried(path, std::ios::binary);
        EXPECT_EQ(std::string(std::istreambuf_iterator<char>(retried), {}), bytes);
        ASSERT_TRUE(fs::remove(path));
    }
}
#if defined(__linux__)
TEST_F(SoUploadFiles, FailedCloseOrLinkPreservesOriginalStageAndLeavesNoPartialFile) {
    const auto path = dir + "/owned.so";
    for (bool close_failure : {true, false}) {
        fault_close_errno = close_failure ? EIO : 0;
        fault_link_errno = close_failure ? 0 : EACCES;
        BootstrapResult result;
        EXPECT_FALSE(WriteBytes(path, "runtime", 7, result));
        EXPECT_EQ(result.stage, close_failure ? BootstrapStage::Close : BootstrapStage::Link);
        EXPECT_EQ(result.error_number, close_failure ? EIO : EACCES);
        EXPECT_TRUE(fs::is_empty(dir));
    }
}
#endif

TEST_F(SoUploadFiles, VerificationRefusesSymlinksAndNonRegularFiles) {
    const auto path = dir + "/owned.so";
    fs::create_directory(path);
    BootstrapResult directory;
    EXPECT_FALSE(WriteBytes(path, "x", 1, directory));
    EXPECT_EQ(directory.stage, BootstrapStage::Verify);
    EXPECT_EQ(directory.error_number, EINVAL);
    fs::remove(path);
    std::ofstream(dir + "/target.so") << "x";
    fs::create_symlink(dir + "/target.so", path);
    BootstrapResult symlink;
    EXPECT_FALSE(WriteBytes(path, "x", 1, symlink));
    EXPECT_EQ(symlink.stage, BootstrapStage::Verify);
    EXPECT_EQ(symlink.error_number, ELOOP);
}
}  // namespace
