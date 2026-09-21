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

#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <string>
#include <thread>
#include <unistd.h>

#include "host/file_marker_handshake.h"

namespace {

class HandshakeDirectory {
public:
    HandshakeDirectory() {
        path_ = std::filesystem::temp_directory_path() /
                ("simpler_file_marker_handshake_" + std::to_string(static_cast<long long>(getpid())));
        std::filesystem::remove_all(path_);
        std::filesystem::create_directories(path_);
    }
    ~HandshakeDirectory() { std::filesystem::remove_all(path_); }

    std::string rootinfo_path() const { return (path_ / "rootinfo.bin").string(); }

private:
    std::filesystem::path path_;
};

bool wait_for_path(const std::string &path, std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!std::filesystem::exists(path)) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

}  // namespace

TEST(FileMarkerHandshakeTest, RankZeroWaitsForFollowerDepartureBeforeCleanup) {
    HandshakeDirectory directory;
    constexpr uint64_t kRunToken = 0xA5A3;
    const std::string rootinfo_path = directory.rootinfo_path();
    const std::string rank_zero_arrival = file_marker_handshake::marker_path(rootinfo_path, kRunToken, "destroy", 0);
    const std::string follower_arrival = file_marker_handshake::marker_path(rootinfo_path, kRunToken, "destroy", 1);
    const std::string rank_zero_departure =
        file_marker_handshake::marker_path(rootinfo_path, kRunToken, "destroy_departed", 0);
    const std::string follower_departure =
        file_marker_handshake::marker_path(rootinfo_path, kRunToken, "destroy_departed", 1);

    auto rank_zero = std::async(std::launch::async, [&]() {
        auto result = file_marker_handshake::destroy_barrier(
            rootinfo_path, 0, 2, kRunToken, {std::chrono::seconds(2), std::chrono::milliseconds(1)}
        );
        if (result.ok()) (void)file_marker_handshake::cleanup(rootinfo_path);
        return result;
    });

    EXPECT_TRUE(wait_for_path(rank_zero_arrival, std::chrono::seconds(1)));
    EXPECT_TRUE(file_marker_handshake::publish_marker(follower_arrival));
    EXPECT_TRUE(wait_for_path(rank_zero_departure, std::chrono::seconds(1)));

    EXPECT_EQ(rank_zero.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
    EXPECT_TRUE(std::filesystem::exists(rank_zero_arrival));
    EXPECT_TRUE(std::filesystem::exists(follower_arrival));

    const auto follower_result = file_marker_handshake::destroy_barrier(
        rootinfo_path, 1, 2, kRunToken, {std::chrono::seconds(2), std::chrono::milliseconds(1)}
    );
    const auto result = rank_zero.get();

    EXPECT_TRUE(follower_result.ok());
    EXPECT_TRUE(result.ok());
    EXPECT_FALSE(std::filesystem::exists(rank_zero_arrival));
    EXPECT_FALSE(std::filesystem::exists(follower_arrival));
    EXPECT_FALSE(std::filesystem::exists(rank_zero_departure));
    EXPECT_FALSE(std::filesystem::exists(follower_departure));
}

TEST(FileMarkerHandshakeTest, FollowerReturnsOnlyAfterRootinfoRetirement) {
    HandshakeDirectory directory;
    constexpr uint64_t kRunToken = 0x1234;
    const std::string rootinfo_path = directory.rootinfo_path();
    { std::ofstream(rootinfo_path) << "old generation"; }

    auto follower = std::async(std::launch::async, [&]() {
        return file_marker_handshake::release_after_cleanup(
            rootinfo_path, 1, 2, kRunToken, {std::chrono::seconds(2), std::chrono::milliseconds(1)}
        );
    });

    EXPECT_EQ(follower.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
    EXPECT_TRUE(std::filesystem::exists(rootinfo_path));

    const auto root_result = file_marker_handshake::release_after_cleanup(
        rootinfo_path, 0, 2, kRunToken, {std::chrono::seconds(2), std::chrono::milliseconds(1)}
    );
    const auto follower_result = follower.get();

    EXPECT_TRUE(root_result.ok());
    EXPECT_TRUE(follower_result.ok());
    EXPECT_FALSE(std::filesystem::exists(rootinfo_path));
    EXPECT_FALSE(std::filesystem::exists(file_marker_handshake::release_path(rootinfo_path, kRunToken, 1)));
}

TEST(FileMarkerHandshakeTest, FollowerReleaseSurvivesNextGenerationSweep) {
    HandshakeDirectory directory;
    constexpr uint64_t kOldRunToken = 0x1234;
    constexpr uint64_t kNextRunToken = 0x5678;
    const std::string rootinfo_path = directory.rootinfo_path();
    const std::string old_marker = file_marker_handshake::marker_path(rootinfo_path, kOldRunToken, "destroy", 0);
    const std::string release = file_marker_handshake::release_path(rootinfo_path, kOldRunToken, 1);
    { std::ofstream(rootinfo_path) << "old generation"; }
    ASSERT_TRUE(file_marker_handshake::publish_marker(old_marker));

    const auto root_result = file_marker_handshake::release_after_cleanup(
        rootinfo_path, 0, 2, kOldRunToken, {std::chrono::seconds(2), std::chrono::milliseconds(1)}
    );
    ASSERT_TRUE(root_result.ok());
    ASSERT_TRUE(std::filesystem::exists(release));

    // A sequential rank-0 init performs the same broad sweep before writing
    // its new rootinfo. The old follower's private release must survive it.
    ASSERT_TRUE(file_marker_handshake::cleanup(rootinfo_path).ok());
    ASSERT_TRUE(std::filesystem::exists(release));

    const std::string next_marker =
        file_marker_handshake::marker_path(rootinfo_path, kNextRunToken, "rootinfo_ready", 0);
    { std::ofstream(rootinfo_path) << "next generation"; }
    ASSERT_TRUE(file_marker_handshake::publish_marker(next_marker));

    const auto follower_result = file_marker_handshake::release_after_cleanup(
        rootinfo_path, 1, 2, kOldRunToken, {std::chrono::seconds(2), std::chrono::milliseconds(1)}
    );
    EXPECT_TRUE(follower_result.ok());
    EXPECT_FALSE(std::filesystem::exists(release));
    EXPECT_TRUE(std::filesystem::exists(rootinfo_path));
    EXPECT_TRUE(std::filesystem::exists(next_marker));
}

TEST(FileMarkerHandshakeTest, CleanupFailureDoesNotReleaseFollowers) {
    HandshakeDirectory directory;
    constexpr uint64_t kRunToken = 0x1234;
    const std::string rootinfo_path = directory.rootinfo_path();
    std::filesystem::create_directory(rootinfo_path);
    { std::ofstream(std::filesystem::path(rootinfo_path) / "keep") << "not empty"; }

    const auto result = file_marker_handshake::release_after_cleanup(
        rootinfo_path, 0, 2, kRunToken, {std::chrono::milliseconds(20), std::chrono::milliseconds(1)}
    );

    EXPECT_EQ(result.error, file_marker_handshake::Error::CLEANUP_FAILED);
    EXPECT_EQ(result.stage, file_marker_handshake::Stage::CLEANUP);
    EXPECT_FALSE(std::filesystem::exists(file_marker_handshake::release_path(rootinfo_path, kRunToken, 1)));
}

TEST(FileMarkerHandshakeTest, UnrelatedEntryErrorDoesNotBlockRelease) {
    HandshakeDirectory directory;
    constexpr uint64_t kRunToken = 0x1234;
    const std::string rootinfo_path = directory.rootinfo_path();
    const auto loop = std::filesystem::path(rootinfo_path).parent_path() / "status_loop";
    { std::ofstream(rootinfo_path) << "old generation"; }
    std::filesystem::create_symlink(loop.filename(), loop);

    const auto result = file_marker_handshake::release_after_cleanup(
        rootinfo_path, 0, 2, kRunToken, {std::chrono::milliseconds(20), std::chrono::milliseconds(1)}
    );

    EXPECT_TRUE(result.ok());
    EXPECT_TRUE(std::filesystem::exists(file_marker_handshake::release_path(rootinfo_path, kRunToken, 1)));
}

TEST(FileMarkerHandshakeTest, FailedAtomicPublishLeavesNoVisibleMarker) {
    HandshakeDirectory directory;
    const std::string path = directory.rootinfo_path() + "/missing/marker";

    EXPECT_FALSE(file_marker_handshake::publish_marker(path));
    EXPECT_FALSE(std::filesystem::exists(path));
    EXPECT_FALSE(std::filesystem::exists(path + ".tmp"));
}
