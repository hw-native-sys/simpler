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
// Regression test for the per-callable_id orch SO file naming contract.
//
// The onboard variants of `create_orch_so_file` (src/{a2a3,a5}/platform/
// onboard/aicpu/orch_so_file.cpp) historically used pid-only naming, which
// silently broke once multi-callable dispatch was introduced on the same
// device process: the second cid's `O_TRUNC` open
// shredded the first cid's already-dlopen'd SO image and the next launch
// on cid=0 SIGBUS'd inside the AICPU executor (manifesting as
// `rtStreamSynchronize (AICPU) failed: 507018` on the host).
//
// The fix is to embed `callable_id` in the file name when cid >= 0. This
// test exercises the contract directly: distinct cids must produce distinct
// paths, and the legacy cid=-1 path must remain pid-only (no behavioural
// change for variants that never adopt per-cid dispatch).

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <gtest/gtest.h>

#include "aicpu/orch_so_file.h"

namespace {

std::string mkscratch_dir() {
    char templ[] = "/tmp/orch_so_file_ut_XXXXXX";
    const char *dir = mkdtemp(templ);
    if (dir == nullptr) {
        std::abort();
    }
    return std::string(dir);
}

void rmtree(const std::string &dir) {
    std::string cmd = "rm -rf '" + dir + "'";
    (void)std::system(cmd.c_str());
}

}  // namespace

TEST(OrchSoFile, DistinctCallableIdsProduceDistinctPaths) {
    // Repro for the 507018 SIGBUS bug: with pid-only naming, cid=0 and
    // cid=1 collide on `libdevice_orch_<pid>.so` and the second
    // O_TRUNC open silently shreds the first cid's already-dlopen'd
    // image. Embedding the cid restores per-callable file isolation.
    const std::string dir = mkscratch_dir();
    char path0[256] = {};
    char path1[256] = {};

    int32_t fd0 = create_orch_so_file(dir.c_str(), /*callable_id=*/0, path0, sizeof(path0));
    ASSERT_GE(fd0, 0) << "create_orch_so_file(cid=0) failed";
    close(fd0);

    int32_t fd1 = create_orch_so_file(dir.c_str(), /*callable_id=*/1, path1, sizeof(path1));
    ASSERT_GE(fd1, 0) << "create_orch_so_file(cid=1) failed";
    close(fd1);

    EXPECT_STRNE(path0, path1) << "Distinct cids must yield distinct file paths "
                                  "(otherwise O_TRUNC would corrupt the first SO).";

    rmtree(dir);
}

TEST(OrchSoFile, LegacySentinelKeepsPidOnlyNaming) {
    // Variants that never adopt per-cid dispatch pass cid=-1; the file
    // name must remain pid-only so existing callers see no change.
    const std::string dir = mkscratch_dir();
    char path[256] = {};

    int32_t fd = create_orch_so_file(dir.c_str(), /*callable_id=*/-1, path, sizeof(path));
    ASSERT_GE(fd, 0);
    close(fd);

    char expected[256];
    std::snprintf(expected, sizeof(expected), "%s/libdevice_orch_%d.so", dir.c_str(), getpid());
    EXPECT_STREQ(path, expected) << "Legacy (cid=-1) path must remain pid-only";

    rmtree(dir);
}
