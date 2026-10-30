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

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "a5/platform/onboard/host/comm_hccl.cpp"

namespace {
std::vector<int> calls;
int fail_call = -1;
aclError injected(int operation) {
    calls.push_back(operation);
    if (fail_call == 0) return static_cast<aclError>(40 + calls.size());
    return static_cast<int>(calls.size()) == fail_call ? 42 : ACL_SUCCESS;
}
}  // namespace

extern "C" aclError aclrtUnmapMem(void *) { return injected(1); }
extern "C" aclError aclrtReleaseMemAddress(void *) { return injected(2); }
extern "C" aclError aclrtFreePhysical(aclrtDrvMemHandle) { return injected(3); }

class A5DomainRelease : public testing::TestWithParam<int> {};

TEST_P(A5DomainRelease, ErrorsBlockRetirementAndAdmission) {
    const auto directory = std::filesystem::temp_directory_path() /
                           ("simpler_a5_release_" + std::to_string(getpid()) + "_" + std::to_string(GetParam()));
    ASSERT_TRUE(std::filesystem::create_directory(directory));
    CommHandle_ handle{};
    handle.rootinfo_path = (directory / "root.bin").string();
    handle.run_token = 1;
    auto allocation = std::make_unique<DomainAllocation>();
    allocation->rank = 0;
    allocation->nranks = 1;
    int local{}, peer{};
    allocation->local_buf = &local;
    allocation->own_handle = reinterpret_cast<aclrtDrvMemHandle>(&local);
    allocation->peer_windows.emplace_back(&peer, reinterpret_cast<aclrtDrvMemHandle>(&peer));
    handle.domain_allocations.emplace(7, std::move(allocation));
    calls.clear();
    fail_call = GetParam();
    const int result = comm_release_domain_windows(&handle, 7, 1, 0);
    EXPECT_EQ(calls, (std::vector<int>{1, 2, 3, 1, 2, 3}));
    EXPECT_TRUE(handle.domain_allocations.empty());
    const auto arrival = file_marker_handshake::marker_path(handle.rootinfo_path, 1, "alloc_7_release", 0);
    if (GetParam() > 0) {
        EXPECT_NE(result, 0);
        EXPECT_TRUE(handle.domain_admission_stopped);
        EXPECT_TRUE(std::filesystem::exists(arrival));
        EXPECT_FALSE(
            std::filesystem::exists(
                file_marker_handshake::marker_path(handle.rootinfo_path, 1, "alloc_7_release_consumed", 0)
            )
        );
    } else {
        EXPECT_EQ(result, 0);
        EXPECT_FALSE(handle.domain_admission_stopped);
        EXPECT_FALSE(std::filesystem::exists(arrival));
    }
    std::filesystem::remove_all(directory);
}

INSTANTIATE_TEST_SUITE_P(AclFailures, A5DomainRelease, testing::Values(-1, 1, 2, 3, 4, 5, 6));

TEST(A5DomainReleaseHelper, PreservesFirstErrorAndAttemptsAllReleases) {
    int buffer{};
    calls.clear();
    fail_call = 0;
    EXPECT_EQ(release_own_vmm_window(&buffer, reinterpret_cast<aclrtDrvMemHandle>(&buffer)), 41);
    EXPECT_EQ(calls, (std::vector<int>{1, 2, 3}));
}
