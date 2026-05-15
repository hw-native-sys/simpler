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
 * Path D backend populated it correctly:
 *
 *   rankId          == (this process's rank)
 *   rankNum         == (nranks we passed to comm_init)
 *   winSize         == comm_get_window_size(h)          (cross-API consistency)
 *   windowsIn[rank] == comm_get_local_window_base(h)
 *   windowsIn[0..nranks-1] all non-zero                 (every peer GVA populated)
 *
 * What this guards in the Path D world:
 *
 *   - ipc_read_announce timing out on a peer but the function still returning
 *     0 -- the windowsIn[peer]==0 assertion would catch the resulting empty
 *     slot.
 *   - comm_get_window_size / winSize implementation drift -- the cross-API
 *     consistency check pins them together.
 *   - aclrtDeviceEnablePeerAccess silently failing but the warning being
 *     fprintf'd and continuing (alloc_windows_via_ipc does this on
 *     "already enabled" returns) -- a peer slot can still get populated with
 *     an unmapped VA; host-side byte-level memcpy of the device ctx catches
 *     a wholly null slot.
 *
 * This is *not* an HCCL-private ABI canary (that role died with the MC2
 * reverse-parse code); it is a black-box lifecycle test of the comm_*
 * C API and the device-side CommContext it produces.
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

namespace {

// Function pointers from libhost_runtime.so (comm_* C API + DeviceRunner wiring)
struct HostRuntimeApi {
    void *(*create_device_context)();
    void (*destroy_device_context)(void *);
    int (*ensure_acl_ready_ctx)(void *, int);

    CommHandle (*comm_init)(int, int, void *, const char *);
    int (*comm_alloc_windows)(CommHandle, size_t, uint64_t *);
    int (*comm_get_local_window_base)(CommHandle, uint64_t *);
    int (*comm_get_window_size)(CommHandle, size_t *);
    int (*comm_barrier)(CommHandle);
    int (*comm_destroy)(CommHandle);
};

template <typename F>
F resolve(void *handle, const char *name) {
    dlerror();
    void *sym = dlsym(handle, name);
    if (dlerror() != nullptr) return nullptr;
    return reinterpret_cast<F>(sym);
}

bool load_host_runtime_api(void *handle, HostRuntimeApi &api) {
    api.create_device_context = resolve<decltype(api.create_device_context)>(handle, "create_device_context");
    api.destroy_device_context = resolve<decltype(api.destroy_device_context)>(handle, "destroy_device_context");
    api.ensure_acl_ready_ctx = resolve<decltype(api.ensure_acl_ready_ctx)>(handle, "ensure_acl_ready_ctx");
    api.comm_init = resolve<decltype(api.comm_init)>(handle, "comm_init");
    api.comm_alloc_windows = resolve<decltype(api.comm_alloc_windows)>(handle, "comm_alloc_windows");
    api.comm_get_local_window_base =
        resolve<decltype(api.comm_get_local_window_base)>(handle, "comm_get_local_window_base");
    api.comm_get_window_size = resolve<decltype(api.comm_get_window_size)>(handle, "comm_get_window_size");
    api.comm_barrier = resolve<decltype(api.comm_barrier)>(handle, "comm_barrier");
    api.comm_destroy = resolve<decltype(api.comm_destroy)>(handle, "comm_destroy");
    return api.create_device_context && api.destroy_device_context && api.ensure_acl_ready_ctx && api.comm_init &&
           api.comm_alloc_windows && api.comm_get_local_window_base && api.comm_get_window_size && api.comm_barrier &&
           api.comm_destroy;
}

// Exit codes for the rank child. Each stage gets a distinct code so the
// parent's waitpid surface pinpoints exactly which step broke.
constexpr int EXIT_DLERR = 10;
constexpr int EXIT_DEV_CTX = 15;
constexpr int EXIT_ACL_READY = 18;
constexpr int EXIT_STREAM = 19;
constexpr int EXIT_INIT = 20;
constexpr int EXIT_ALLOC = 30;
constexpr int EXIT_WINDOW_BASE = 40;
constexpr int EXIT_WINDOW_SIZE = 50;
// EXIT_CTX_{MEMCPY,FIELDS} guard the device-side CommContext that Path D
// fills in alloc_windows_via_ipc: cross-API consistency between
// comm_get_window_size / winSize, and every peer's windowsIn slot
// actually getting a non-zero GVA.
constexpr int EXIT_CTX_MEMCPY = 55;
constexpr int EXIT_CTX_FIELDS = 56;
constexpr int EXIT_BARRIER = 60;
constexpr int EXIT_DESTROY = 70;

int run_rank(int rank, int nranks, int device_id, const char *rootinfo_path) {
    // libsimpler_log.so must be loaded RTLD_GLOBAL first so libhost_runtime.so's
    // undefined HostLogger / unified_log_* symbols resolve at load time.
    // Mirrors ChipWorker::init in production.
    void *log_handle = dlopen(SIMPLER_LOG_LIB_PATH, RTLD_NOW | RTLD_GLOBAL);
    if (log_handle == nullptr) {
        fprintf(stderr, "[rank %d] dlopen simpler_log failed: %s\n", rank, dlerror());
        return EXIT_DLERR;
    }

    // libhost_runtime.so is the subject under test -- dlopen mirrors
    // ChipWorker.  libascendcl is linked in, so acl* is available directly.
    void *host_handle = dlopen(PTO_HOST_RUNTIME_LIB_PATH, RTLD_NOW | RTLD_LOCAL);
    if (host_handle == nullptr) {
        fprintf(stderr, "[rank %d] dlopen host lib failed: %s\n", rank, dlerror());
        return EXIT_DLERR;
    }

    HostRuntimeApi api{};
    if (!load_host_runtime_api(host_handle, api)) {
        fprintf(stderr, "[rank %d] required symbols missing from libhost_runtime.so\n", rank);
        dlclose(host_handle);
        return EXIT_DLERR;
    }

    // Caller step 1: stand up DeviceRunner to own ACL lifecycle.
    void *dev_ctx = api.create_device_context();
    if (dev_ctx == nullptr) {
        fprintf(stderr, "[rank %d] create_device_context returned null\n", rank);
        dlclose(host_handle);
        return EXIT_DEV_CTX;
    }

    // Caller step 2: aclInit + aclrtSetDevice via DeviceRunner.
    if (api.ensure_acl_ready_ctx(dev_ctx, device_id) != 0) {
        fprintf(stderr, "[rank %d] ensure_acl_ready_ctx(%d) failed\n", rank, device_id);
        api.destroy_device_context(dev_ctx);
        dlclose(host_handle);
        return EXIT_ACL_READY;
    }

    // Caller step 3: caller creates its own stream; comm never touches it.
    aclrtStream stream = nullptr;
    aclError aRet = aclrtCreateStream(&stream);
    if (aRet != ACL_SUCCESS || stream == nullptr) {
        fprintf(stderr, "[rank %d] aclrtCreateStream failed: %d\n", rank, static_cast<int>(aRet));
        api.destroy_device_context(dev_ctx);
        dlclose(host_handle);
        return EXIT_STREAM;
    }

    // Caller step 4: drive comm_* against the injected stream.
    int stage = 0;
    int exit_code = 0;
    CommHandle h = api.comm_init(rank, nranks, stream, rootinfo_path);
    if (h == nullptr) {
        stage = EXIT_INIT;
    } else {
        uint64_t device_ctx_ptr = 0;
        if (api.comm_alloc_windows(h, 4096, &device_ctx_ptr) != 0 || device_ctx_ptr == 0) {
            stage = EXIT_ALLOC;
        } else {
            uint64_t local_base = 0;
            if (api.comm_get_local_window_base(h, &local_base) != 0 || local_base == 0) {
                stage = EXIT_WINDOW_BASE;
            } else {
                size_t win_size = 0;
                if (api.comm_get_window_size(h, &win_size) != 0 || win_size < 4096) {
                    stage = EXIT_WINDOW_SIZE;
                } else {
                    // Black-box contract check on the device-side CommContext
                    // produced by Path D: cross-API consistency (winSize ==
                    // comm_get_window_size), every peer's windowsIn slot
                    // populated, and the local slot matching
                    // comm_get_local_window_base.
                    CommContext host_ctx{};
                    aclError mc_rc = aclrtMemcpy(
                        &host_ctx, sizeof(host_ctx), reinterpret_cast<void *>(device_ctx_ptr), sizeof(host_ctx),
                        ACL_MEMCPY_DEVICE_TO_HOST
                    );
                    if (mc_rc != ACL_SUCCESS) {
                        fprintf(
                            stderr, "[rank %d] aclrtMemcpy(device_ctx) failed: %d\n", rank, static_cast<int>(mc_rc)
                        );
                        stage = EXIT_CTX_MEMCPY;
                    } else if (host_ctx.rankId != static_cast<uint32_t>(rank) ||
                               host_ctx.rankNum != static_cast<uint32_t>(nranks) ||
                               host_ctx.winSize != static_cast<uint64_t>(win_size) ||
                               host_ctx.windowsIn[rank] != local_base) {
                        fprintf(
                            stderr,
                            "[rank %d] CommContext field mismatch\n"
                            "  got:      rankId=%u rankNum=%u winSize=%lu windowsIn[%d]=0x%lx\n"
                            "  expected: rankId=%d rankNum=%d winSize=%zu windowsIn[%d]=0x%lx\n",
                            rank, host_ctx.rankId, host_ctx.rankNum, static_cast<unsigned long>(host_ctx.winSize), rank,
                            static_cast<unsigned long>(host_ctx.windowsIn[rank]), rank, nranks, win_size, rank,
                            static_cast<unsigned long>(local_base)
                        );
                        stage = EXIT_CTX_FIELDS;
                    } else {
                        // Every peer's window GVA must be non-zero.  A zero
                        // entry means ipc_read_announce / ImportByKey didn't
                        // populate that slot but the function still returned
                        // success.
                        for (int i = 0; i < nranks; ++i) {
                            if (host_ctx.windowsIn[i] == 0) {
                                fprintf(
                                    stderr, "[rank %d] CommContext.windowsIn[%d] == 0 (peer GVA missing)\n", rank, i
                                );
                                stage = EXIT_CTX_FIELDS;
                                break;
                            }
                        }
                    }

                    if (stage == 0 && api.comm_barrier(h) != 0) {
                        stage = EXIT_BARRIER;
                    }
                }
            }
        }
        if (api.comm_destroy(h) != 0 && stage == 0) {
            stage = EXIT_DESTROY;
        }
    }
    exit_code = stage;

    // Caller step 5: cleanup in reverse order.  destroy_device_context
    // eventually drives DeviceRunner::finalize which calls aclrtResetDevice
    // and aclFinalize, so we do not call them ourselves.
    aclrtDestroyStream(stream);
    api.destroy_device_context(dev_ctx);
    dlclose(host_handle);
    return exit_code;
}

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
        if (!std::filesystem::exists(SIMPLER_LOG_LIB_PATH)) {
            GTEST_SKIP() << "libsimpler_log.so not built: " << SIMPLER_LOG_LIB_PATH
                         << "\n(build the a2a3 onboard runtime first — runs simpler_log too)";
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
            std::_Exit(run_rank(rank, kNranks, devices[rank], rootinfo_path.c_str()));
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
