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

/*
 * Hardware UT for the comm_* C API end-to-end lifecycle on a2a3 onboard.
 *
 * Drives the full call chain
 *   dlopen -> create_device_context -> ensure_acl_ready_ctx ->
 *   aclrtCreateStream -> comm_init -> comm_alloc_windows ->
 *   comm_get_local_window_base / comm_get_window_size ->
 *   comm_barrier -> comm_destroy
 * and then copies the device-side CommContext back to host to assert the
 * Fabric V2 backend populated it correctly:
 *
 *   rankId          == (this process's rank)
 *   rankNum         == (nranks we passed to comm_init)
 *   winSize         == comm_get_window_size(h)          (cross-API consistency)
 *   windowsIn[rank] == comm_get_local_window_base(h)
 *   windowsIn[0..nranks-1] all non-zero                 (every peer GVA populated)
 *
 * What this guards in the Fabric V2 backend:
 *
 *   - Fabric announcement timing out on a peer but the function still returning
 *     0 -- the windowsIn[peer]==0
 * assertion would catch the resulting empty
 *     slot.
 *   - comm_get_window_size / winSize implementation drift --
 * the cross-API
 *     consistency check pins them together.
 *   - a Fabric import succeeding without populating the
 * corresponding peer
 *     GVA -- host-side byte-level memcpy of the device ctx catches a null slot.
 * This is *not*
 * an HCCL-private ABI canary (that role died with the MC2 reverse-parse code); it is a black-box lifecycle test of the
 * comm_* C API and the device-side CommContext it produces.
 *
 * Hardware classification: requires_hardware_a2a3 (ctest label) + CMake
 * gate SIMPLER_ENABLE_HARDWARE_TESTS.  Device allocation is driven by
 * CTest RESOURCE_GROUPS + --resource-spec-file.
 *
 * Linking strategy: libhost_runtime.so is dlopen'd -- it is the subject
 * under test and mirrors how ChipWorker loads a runtime backend in
 * production.  libascendcl.so is linked directly at compile time because
 * it is generic CANN infra; going through dlsym for acl* here buys nothing
 * and only hides types behind void pointers.
 *
 * PTO_HOST_RUNTIME_LIB_PATH is baked in at configure time by
 * tests/ut/cpp/CMakeLists.txt.
 */

#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "acl/acl.h"
#include "platform_comm/comm.h"
#include "platform_comm/comm_context.h"
#include "comm_lifecycle_rank.h"

namespace {

/// Read device ids allocated by CTest resource allocation.
///
/// CTest sets CTEST_RESOURCE_GROUP_COUNT and per-group env vars when
/// --resource-spec-file is provided.  With RESOURCE_GROUPS "npus:2",
/// there is one group (group 0) containing two NPU allocations:
///   CTEST_RESOURCE_GROUP_0_NPUS = "id:4,slots:1;id:5,slots:1"
///
/// Returns the extracted device ids.
std::vector<int> read_ctest_devices() {
    std::vector<int> ids;
    const char *count_str = std::getenv("CTEST_RESOURCE_GROUP_COUNT");
    if (count_str == nullptr) return ids;

    int group_count = std::atoi(count_str);
    for (int g = 0; g < group_count; ++g) {
        std::string var = "CTEST_RESOURCE_GROUP_" + std::to_string(g) + "_NPUS";
        const char *val = std::getenv(var.c_str());
        if (val == nullptr) continue;

        // Parse "id:<N>,slots:<M>;id:<N>,slots:<M>;..."
        std::string s(val);
        size_t pos = 0;
        while ((pos = s.find("id:", pos)) != std::string::npos) {
            pos += 3;
            ids.push_back(std::atoi(s.c_str() + pos));
        }
    }
    return ids;
}

}  // namespace

class CommLifecycleTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Paths are baked in by tests/ut/cpp/CMakeLists.txt; the only way they
        // can be wrong at test time is if someone ran ctest without first
        // building the onboard runtime.
        if (!std::filesystem::exists(PTO_HOST_RUNTIME_LIB_PATH)) {
            GTEST_SKIP() << "libhost_runtime.so not built: " << PTO_HOST_RUNTIME_LIB_PATH
                         << "\n(build the a2a3 onboard tensormap_and_ringbuffer runtime first)";
        }
    }
};

TEST_F(CommLifecycleTest, TwoRankInitAllocBarrierDestroy) {
    constexpr int kNranks = 2;
    auto devices = read_ctest_devices();
    ASSERT_GE(devices.size(), static_cast<size_t>(kNranks))
        << "need " << kNranks << " NPU devices; run with --resource-spec-file";

    const std::string rootinfo_path = "/tmp/pto_comm_ut_rootinfo_" + std::to_string(getpid()) + ".bin";

    std::vector<pid_t> pids;
    pids.reserve(kNranks);
    for (int rank = 0; rank < kNranks; ++rank) {
        pid_t pid = fork();
        ASSERT_GE(pid, 0) << "fork failed: " << strerror(errno);
        if (pid == 0) {
            std::_Exit(comm_lifecycle_test::run_rank(rank, kNranks, devices[rank], rootinfo_path.c_str()));
        }
        pids.push_back(pid);
    }

    for (int rank = 0; rank < kNranks; ++rank) {
        int status = 0;
        pid_t waited = waitpid(pids[rank], &status, 0);
        ASSERT_EQ(waited, pids[rank]);
        ASSERT_TRUE(WIFEXITED(status)) << "rank " << rank << " did not exit normally (status=" << status << ")";
        EXPECT_EQ(WEXITSTATUS(status), 0)
            << "rank " << rank << " failed at stage with exit code " << WEXITSTATUS(status)
            << " (10=dlopen, 15=dev_ctx, 18=acl_ready, 19=stream, 20=init, 30=alloc, "
            << "40=base, 50=size, 55=ctx_memcpy, 56=ctx_fields, 60=barrier, 70=destroy)";
    }

    std::error_code ec;
    std::filesystem::remove(rootinfo_path, ec);
}
