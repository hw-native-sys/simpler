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
 * Hardware UT for the HCCL backend of the comm_* C API.
 *
 * The test drives the expected caller contract end-to-end:
 *   1. Caller brings up an ACL context (here: DeviceRunner::ensure_acl_ready
 *      via the ensure_acl_ready_ctx C export).
 *   2. Caller creates its own aclrtStream (here: dlsym'd aclrtCreateStream).
 *   3. Caller passes the stream into comm_init, which does nothing with ACL
 *      lifecycle or stream ownership.
 *   4. Caller tears everything down in reverse order; DeviceRunner::finalize
 *      is responsible for aclrtResetDevice + aclFinalize (wired via
 *      destroy_device_context).
 *
 * This flow mirrors exactly what ChipWorker (follow-up PR) will do.  Each
 * rank runs in its own forked child so every process owns one ACL context.
 *
 * Hardware classification: requires_hardware_a2a3 (ctest label).
 * No-hw runners exclude this test at ctest selection time via -LE.
 *
 * Device allocation: CTest RESOURCE_GROUPS ("npus:2") + --resource-spec-file.
 * CTest allocates NPU device ids and passes them via CTEST_RESOURCE_GROUP_*
 * environment variables.  The test reads these to determine which devices
 * to use for each rank.
 *
 * Environment:
 *   PTO_HOST_RUNTIME_LIB  absolute path to libhost_runtime.so (onboard)
 *   PTO_ASCENDCL_LIB      optional override; defaults to "libascendcl.so"
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

#include "host/comm.h"

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

// Minimal ACL surface we need via dlopen libascendcl.so
struct AclApi {
    int (*aclrtCreateStream)(void **);
    int (*aclrtDestroyStream)(void *);
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

bool load_acl_api(void *handle, AclApi &api) {
    api.aclrtCreateStream = resolve<decltype(api.aclrtCreateStream)>(handle, "aclrtCreateStream");
    api.aclrtDestroyStream = resolve<decltype(api.aclrtDestroyStream)>(handle, "aclrtDestroyStream");
    return api.aclrtCreateStream && api.aclrtDestroyStream;
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
constexpr int EXIT_BARRIER = 60;
constexpr int EXIT_DESTROY = 70;

int run_rank(
    const char *host_lib_path, const char *acl_lib_path, int rank, int nranks, int device_id, const char *rootinfo_path
) {
    void *host_handle = dlopen(host_lib_path, RTLD_NOW | RTLD_LOCAL);
    if (host_handle == nullptr) {
        fprintf(stderr, "[rank %d] dlopen host lib failed: %s\n", rank, dlerror());
        return EXIT_DLERR;
    }
    void *acl_handle = dlopen(acl_lib_path, RTLD_NOW | RTLD_GLOBAL);
    if (acl_handle == nullptr) {
        fprintf(stderr, "[rank %d] dlopen acl lib failed: %s\n", rank, dlerror());
        dlclose(host_handle);
        return EXIT_DLERR;
    }

    HostRuntimeApi api{};
    AclApi acl{};
    if (!load_host_runtime_api(host_handle, api) || !load_acl_api(acl_handle, acl)) {
        fprintf(stderr, "[rank %d] required symbols missing\n", rank);
        dlclose(acl_handle);
        dlclose(host_handle);
        return EXIT_DLERR;
    }

    // Caller step 1: stand up DeviceRunner to own ACL lifecycle.
    void *dev_ctx = api.create_device_context();
    if (dev_ctx == nullptr) {
        fprintf(stderr, "[rank %d] create_device_context returned null\n", rank);
        dlclose(acl_handle);
        dlclose(host_handle);
        return EXIT_DEV_CTX;
    }

    // Caller step 2: aclInit + aclrtSetDevice via DeviceRunner.
    if (api.ensure_acl_ready_ctx(dev_ctx, device_id) != 0) {
        fprintf(stderr, "[rank %d] ensure_acl_ready_ctx(%d) failed\n", rank, device_id);
        api.destroy_device_context(dev_ctx);
        dlclose(acl_handle);
        dlclose(host_handle);
        return EXIT_ACL_READY;
    }

    // Caller step 3: caller creates its own stream; comm never touches it.
    void *stream = nullptr;
    if (acl.aclrtCreateStream(&stream) != 0 || stream == nullptr) {
        fprintf(stderr, "[rank %d] aclrtCreateStream failed\n", rank);
        api.destroy_device_context(dev_ctx);
        dlclose(acl_handle);
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
                } else if (api.comm_barrier(h) != 0) {
                    stage = EXIT_BARRIER;
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
    acl.aclrtDestroyStream(stream);
    api.destroy_device_context(dev_ctx);
    dlclose(acl_handle);
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

class HcclCommTest : public ::testing::Test {
protected:
    void SetUp() override {
        host_lib_path_ = std::getenv("PTO_HOST_RUNTIME_LIB");
        if (host_lib_path_ == nullptr || *host_lib_path_ == '\0') {
            GTEST_SKIP() << "PTO_HOST_RUNTIME_LIB not set; cannot locate libhost_runtime.so.";
        }
        if (!std::filesystem::exists(host_lib_path_)) {
            GTEST_SKIP() << "PTO_HOST_RUNTIME_LIB does not exist: " << host_lib_path_;
        }

        const char *acl_override = std::getenv("PTO_ASCENDCL_LIB");
        acl_lib_path_ = (acl_override != nullptr && *acl_override != '\0') ? acl_override : "libascendcl.so";
    }

    const char *host_lib_path_ = nullptr;
    const char *acl_lib_path_ = nullptr;
};

TEST_F(HcclCommTest, TwoRankInitAllocBarrierDestroy) {
    constexpr int kNranks = 2;
    auto devices = read_ctest_devices();
    ASSERT_GE(devices.size(), static_cast<size_t>(kNranks))
        << "need " << kNranks << " NPU devices; run with --resource-spec-file";

    const std::string rootinfo_path = "/tmp/pto_hccl_ut_rootinfo_" + std::to_string(getpid()) + ".bin";

    std::vector<pid_t> pids;
    pids.reserve(kNranks);
    for (int rank = 0; rank < kNranks; ++rank) {
        pid_t pid = fork();
        ASSERT_GE(pid, 0) << "fork failed: " << strerror(errno);
        if (pid == 0) {
            std::_Exit(run_rank(host_lib_path_, acl_lib_path_, rank, kNranks, devices[rank], rootinfo_path.c_str()));
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
            << "40=base, 50=size, 60=barrier, 70=destroy)";
    }

    std::error_code ec;
    std::filesystem::remove(rootinfo_path, ec);
}
