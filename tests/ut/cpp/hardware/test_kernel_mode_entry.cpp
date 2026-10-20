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
 * Hardware UT for the kernel-mode entry layer on a2a3 onboard: the four C ABI
 * entries reached through a real libhost_runtime.so, on a device and stream the
 * test owns rather than simpler.
 *
 * The borrowing is the point. The cases that reach a device bind it themselves
 * (aclInit + aclrtSetDevice) and create any stream themselves, and launch is the
 * only entry handed one, so what is under test is the borrowed-device shape a
 * kernel-mode caller presents. A test that
 * let simpler stand the device up would exercise the program-mode shape and
 * pass for the wrong reason.
 *
 * Real TMR binaries must initialize and close successfully. Another live
 * context using the same runtime is refused
 * until that owner closes. Invalid
 * arguments and unprepared launches retain their precise entry error codes.
 * The
 * ChipWorker cases exercise its dlsym forwarding and lifecycle as well.
 * After close, the caller verifies that its
 * device and stream remain usable.
 * Numerical execution and successful callable preparation are covered by the
 *
 * Python kernel-mode hardware test, which compiles a real orchestration/kernel
 * pair; this target requires only the
 * prebuilt runtime artifacts below.
 * Hardware classification: requires_hardware_a2a3 (ctest label) + CMake gate
 * SIMPLER_ENABLE_HARDWARE_TESTS. Device allocation is driven by CTest
 * RESOURCE_GROUPS + --resource-spec-file.
 *
 * PTO_HOST_RUNTIME_LIB_PATH and the three binary paths are baked in at
 * configure time by tests/ut/cpp/CMakeLists.txt.
 */

#include <dlfcn.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "acl/acl.h"
#include "call_config.h"
#include "callable.h"
// ChipWorker holds a unique_ptr<ChipRunLane>, whose deleter needs the complete
// type wherever a ChipWorker is destroyed.
#include "chip_run_lane.h"
#include "chip_worker.h"
#include "runtime_c_api.h"
#include "task_args.h"
#include "host/raii_scope_guard.h"

namespace {

/// The four kernel-mode entries, resolved out of libhost_runtime.so the way
/// ChipWorker resolves them.
struct KernelModeApi {
    void *(*create_device_context)();
    void (*destroy_device_context)(void *);
    int (*finalize_device)(void *);
    decltype(&simpler_kernel_mode_supported) supported;
    decltype(&simpler_kernel_mode_init) init;
    decltype(&simpler_kernel_mode_prepare_callable) prepare_callable;
    decltype(&simpler_kernel_mode_launch) launch;
};

template <typename F>
F resolve(void *handle, const char *name) {
    dlerror();
    void *sym = dlsym(handle, name);
    if (dlerror() != nullptr) return nullptr;
    return reinterpret_cast<F>(sym);
}

bool load_kernel_mode_api(void *handle, KernelModeApi &api) {
    api.create_device_context = resolve<decltype(api.create_device_context)>(handle, "create_device_context");
    api.destroy_device_context = resolve<decltype(api.destroy_device_context)>(handle, "destroy_device_context");
    api.finalize_device = resolve<decltype(api.finalize_device)>(handle, "finalize_device");
    api.supported = resolve<decltype(api.supported)>(handle, "simpler_kernel_mode_supported");
    api.init = resolve<decltype(api.init)>(handle, "simpler_kernel_mode_init");
    api.prepare_callable = resolve<decltype(api.prepare_callable)>(handle, "simpler_kernel_mode_prepare_callable");
    api.launch = resolve<decltype(api.launch)>(handle, "simpler_kernel_mode_launch");
    return api.create_device_context && api.destroy_device_context && api.finalize_device && api.supported &&
           api.init && api.prepare_callable && api.launch;
}

std::vector<uint8_t> read_binary(const char *path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error(std::string("cannot read runtime artifact: ") + path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

struct RuntimeBinaries {
    std::vector<uint8_t> aicpu{read_binary(PTO_KERNEL_UT_AICPU_PATH)};
    std::vector<uint8_t> aicore{read_binary(PTO_KERNEL_UT_AICORE_PATH)};
    std::vector<uint8_t> dispatcher{read_binary(PTO_KERNEL_UT_DISPATCHER_PATH)};

    int init(const KernelModeApi &api, void *ctx, int device_id, uint64_t generation) const {
        const CallConfig config{};
        return api.init(
            ctx, device_id, aicpu.data(), aicpu.size(), aicore.data(), aicore.size(), dispatcher.data(),
            dispatcher.size(), &config, generation, 0, nullptr, 0
        );
    }
};

/// Device ids allocated by CTest resource allocation, parsed from
/// CTEST_RESOURCE_GROUP_<n>_NPUS ("id:<N>,slots:<M>;..."). Mirrors
/// test_comm_lifecycle.cpp::read_ctest_devices().
std::vector<int> read_ctest_devices() {
    std::vector<int> ids;
    const char *count_str = std::getenv("CTEST_RESOURCE_GROUP_COUNT");
    if (count_str == nullptr) return ids;

    int group_count = std::atoi(count_str);
    for (int g = 0; g < group_count; ++g) {
        std::string var = "CTEST_RESOURCE_GROUP_" + std::to_string(g) + "_NPUS";
        const char *val = std::getenv(var.c_str());
        if (val == nullptr) continue;

        std::string s(val);
        size_t pos = 0;
        while ((pos = s.find("id:", pos)) != std::string::npos) {
            pos += 3;
            ids.push_back(std::atoi(s.c_str() + pos));
        }
    }
    return ids;
}

/// A zeroed, correctly aligned ChipCallable image. The entry validation checks
/// alignment, the size floor and the output pointer; it does not walk the
/// flexible-array offsets, so a zeroed header is an image it accepts.
struct AlignedCallableImage {
    alignas(ChipCallable) unsigned char bytes[sizeof(ChipCallable)];

    AlignedCallableImage() { std::memset(bytes, 0, sizeof(bytes)); }
    const void *data() const { return bytes; }
    static size_t size() { return sizeof(ChipCallable); }
};

/// The caller's device and stream, stood up and torn down by the test. This is
/// the state a kernel-mode caller already holds and lends to simpler; nothing
/// under test may reset or finalize any of it.
class BorrowedDevice {
public:
    explicit BorrowedDevice(int device_id) :
        device_id_(device_id) {
        aclError rc = aclInit(nullptr);
        // aclInit is process-wide and another owner in this process may have
        // run it already.
        acl_owned_ = (rc == ACL_SUCCESS);
        if (rc != ACL_SUCCESS && static_cast<int>(rc) != ACL_ERROR_REPEAT_INITIALIZE) return;
        if (aclrtSetDevice(device_id_) != ACL_SUCCESS) return;
        device_set_ = true;
        if (aclrtCreateStream(&stream_) != ACL_SUCCESS) stream_ = nullptr;
    }

    ~BorrowedDevice() {
        if (stream_ != nullptr) aclrtDestroyStream(stream_);
        if (device_set_) aclrtResetDevice(device_id_);
        if (acl_owned_) aclFinalize();
    }

    BorrowedDevice(const BorrowedDevice &) = delete;
    BorrowedDevice &operator=(const BorrowedDevice &) = delete;

    bool ready() const { return device_set_ && stream_ != nullptr; }
    void *stream() const { return stream_; }
    int device_id() const { return device_id_; }

    void verify_still_usable() const {
        int32_t current = -1;
        ASSERT_EQ(aclrtGetDevice(&current), ACL_SUCCESS);
        ASSERT_EQ(current, device_id_);
        void *buffer = nullptr;
        ASSERT_EQ(aclrtMalloc(&buffer, 64, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
        auto cleanup = RAIIScopeGuard([&]() {
            EXPECT_EQ(aclrtFree(buffer), ACL_SUCCESS);
        });
        ASSERT_EQ(aclrtMemsetAsync(buffer, 64, 0, 64, stream_), ACL_SUCCESS);
        EXPECT_EQ(aclrtSynchronizeStreamWithTimeout(stream_, 1000), ACL_SUCCESS);
    }

private:
    int device_id_;
    bool acl_owned_{false};
    bool device_set_{false};
    aclrtStream stream_{nullptr};
};

/// Runs `call` and checks that it throws a ChipWorkerError carrying `code`.
template <typename Call>
void expect_chip_worker_error(Call &&call, int code) {
    try {
        call();
        ADD_FAILURE() << "expected a ChipWorkerError with code " << code;
    } catch (const ChipWorkerError &error) {
        EXPECT_EQ(error.code(), code) << error.what();
    }
}

class KernelModeEntryTest : public ::testing::Test {
protected:
    void SetUp() override {
        for (const char *path :
             {PTO_HOST_RUNTIME_LIB_PATH, PTO_KERNEL_UT_AICPU_PATH, PTO_KERNEL_UT_AICORE_PATH,
              PTO_KERNEL_UT_DISPATCHER_PATH}) {
            if (!std::filesystem::exists(path)) {
                GTEST_SKIP() << "onboard runtime artifact not built: " << path
                             << "\n(build the a2a3 onboard tensormap_and_ringbuffer runtime first)";
            }
        }
        auto devices = read_ctest_devices();
        if (devices.empty()) {
            GTEST_SKIP() << "no NPU device allocated; run ctest with --resource-spec-file";
        }
        device_id_ = devices[0];
    }

    int device_id_{-1};
};

// ---------------------------------------------------------------------------
// C ABI surface: all four entries, on a borrowed device and stream.
// ---------------------------------------------------------------------------

TEST_F(KernelModeEntryTest, SupportsBorrowedContextAndRejectsUnpreparedEntries) {
    BorrowedDevice borrowed(device_id_);
    ASSERT_TRUE(borrowed.ready()) << "could not stand up the caller's device/stream on device " << device_id_;

    void *handle = dlopen(PTO_HOST_RUNTIME_LIB_PATH, RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(handle, nullptr) << "dlopen failed: " << dlerror();

    KernelModeApi api{};
    ASSERT_TRUE(load_kernel_mode_api(handle, api)) << "kernel-mode entries missing from libhost_runtime.so";

    void *ctx = api.create_device_context();
    ASSERT_NE(ctx, nullptr);
    auto cleanup = RAIIScopeGuard([&]() {
        EXPECT_EQ(api.finalize_device(ctx), 0);
        api.destroy_device_context(ctx);
        dlclose(handle);
    });

    EXPECT_EQ(api.supported(ctx), 1);

    AlignedCallableImage image;
    int32_t minted = 99;
    EXPECT_EQ(
        api.prepare_callable(ctx, image.data(), AlignedCallableImage::size(), &minted), PTO_RUNTIME_ERR_INVALID_STATE
    ) << "no kernel context is live on this device context";
    EXPECT_EQ(minted, -1) << "a refused registration must not leave a usable id behind";

    ChipStorageTaskArgs args{};
    EXPECT_EQ(api.launch(ctx, 0, &args, borrowed.stream()), PTO_RUNTIME_ERR_INVALID_STATE)
        << "no kernel context is live on this device context";

    const RuntimeBinaries binaries;
    ASSERT_EQ(binaries.init(api, ctx, device_id_, 1), 0);
    EXPECT_EQ(binaries.init(api, ctx, device_id_, 2), PTO_RUNTIME_ERR_INVALID_STATE);
    // The declared binary cannot fit in this image. Admission must reject it
    // before registration enqueues any device work.
    reinterpret_cast<ChipCallable *>(image.bytes)->binary_size_ = 1;
    EXPECT_EQ(
        api.prepare_callable(ctx, image.data(), AlignedCallableImage::size(), &minted), PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(minted, -1);
    EXPECT_EQ(api.launch(ctx, 0, &args, borrowed.stream()), PTO_RUNTIME_ERR_INVALID_STATE);
    ASSERT_EQ(api.finalize_device(ctx), 0);
    EXPECT_EQ(api.launch(ctx, 0, &args, borrowed.stream()), PTO_RUNTIME_ERR_INVALID_STATE);
    borrowed.verify_still_usable();
}

TEST_F(KernelModeEntryTest, LiveContextClaimRejectsAnotherOwnerAndClosePermitsFreshContext) {
    BorrowedDevice borrowed(device_id_);
    ASSERT_TRUE(borrowed.ready());
    void *handle = dlopen(PTO_HOST_RUNTIME_LIB_PATH, RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(handle, nullptr) << dlerror();
    KernelModeApi api{};
    ASSERT_TRUE(load_kernel_mode_api(handle, api));
    auto unload = RAIIScopeGuard([&]() {
        dlclose(handle);
    });
    void *owner = api.create_device_context();
    void *refused = api.create_device_context();
    void *next = api.create_device_context();
    auto cleanup = RAIIScopeGuard([&]() {
        for (void *ctx : {next, refused, owner}) {
            if (ctx == nullptr) continue;
            EXPECT_EQ(api.finalize_device(ctx), 0);
            api.destroy_device_context(ctx);
        }
    });
    ASSERT_NE(owner, nullptr);
    ASSERT_NE(refused, nullptr);
    ASSERT_NE(next, nullptr);
    const RuntimeBinaries binaries;
    ASSERT_EQ(binaries.init(api, owner, device_id_, 1), 0);
    EXPECT_EQ(binaries.init(api, refused, device_id_, 2), PTO_RUNTIME_ERR_INVALID_STATE);
    ChipStorageTaskArgs args{};
    EXPECT_EQ(api.launch(refused, 0, &args, borrowed.stream()), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(api.finalize_device(refused), 0);
    ASSERT_EQ(api.finalize_device(owner), 0);
    ASSERT_EQ(binaries.init(api, next, device_id_, 3), 0);
    ASSERT_EQ(api.finalize_device(next), 0);
    borrowed.verify_still_usable();
}

TEST_F(KernelModeEntryTest, MalformedArgumentsAreRejectedBeforeTheStateCheck) {
    BorrowedDevice borrowed(device_id_);
    ASSERT_TRUE(borrowed.ready());

    void *handle = dlopen(PTO_HOST_RUNTIME_LIB_PATH, RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(handle, nullptr) << "dlopen failed: " << dlerror();
    KernelModeApi api{};
    ASSERT_TRUE(load_kernel_mode_api(handle, api));
    void *ctx = api.create_device_context();
    ASSERT_NE(ctx, nullptr);

    const CallConfig config{};
    const std::vector<uint8_t> aicpu(16, 0);

    // Generation zero is the ABI's invalid generation.
    EXPECT_EQ(
        api.init(ctx, device_id_, aicpu.data(), aicpu.size(), nullptr, 0, nullptr, 0, &config, 0, 0, nullptr, 0),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    // A binary and its size describe one object: present-together or absent-together.
    EXPECT_EQ(
        api.init(ctx, device_id_, aicpu.data(), 0, nullptr, 0, nullptr, 0, &config, 1, 0, nullptr, 0),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        api.init(ctx, -1, aicpu.data(), aicpu.size(), nullptr, 0, nullptr, 0, &config, 1, 0, nullptr, 0),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );

    AlignedCallableImage image;
    int32_t minted = 99;
    EXPECT_EQ(
        api.prepare_callable(ctx, nullptr, AlignedCallableImage::size(), &minted), PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(minted, -1);
    // Below the size floor the image cannot hold a ChipCallable header.
    minted = 99;
    EXPECT_EQ(
        api.prepare_callable(ctx, image.data(), sizeof(ChipCallable) - 1, &minted), PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(minted, -1);
    // Registration mints the id, so without somewhere to write it the call
    // cannot report one.
    EXPECT_EQ(
        api.prepare_callable(ctx, image.data(), AlignedCallableImage::size(), nullptr), PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );

    ChipStorageTaskArgs args{};
    EXPECT_EQ(api.launch(ctx, 0, &args, nullptr), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(api.launch(ctx, -1, &args, borrowed.stream()), PTO_RUNTIME_ERR_INVALID_ARGUMENT);

    api.destroy_device_context(ctx);
    dlclose(handle);
}

// ---------------------------------------------------------------------------
// ChipWorker surface: the host half of the wiring.
// ---------------------------------------------------------------------------

TEST_F(KernelModeEntryTest, WorkerKernelInitAndClosePreserveBorrowedDeviceAndStream) {
    BorrowedDevice borrowed(device_id_);
    ASSERT_TRUE(borrowed.ready());

    ChipWorker worker;
    const CallConfig config{};
    ASSERT_NO_THROW(worker.kernel_init(
        PTO_HOST_RUNTIME_LIB_PATH, PTO_KERNEL_UT_AICPU_PATH, PTO_KERNEL_UT_AICORE_PATH, PTO_KERNEL_UT_DISPATCHER_PATH,
        device_id_, config, ChipWorker::next_kernel_context_generation()
    ));

    EXPECT_TRUE(worker.initialized());
    EXPECT_EQ(worker.device_id(), device_id_);
    EXPECT_TRUE(worker.kernel_mode_supported());
    ChipWorker refused;
    EXPECT_THROW(
        refused.kernel_init(
            PTO_HOST_RUNTIME_LIB_PATH, PTO_KERNEL_UT_AICPU_PATH, PTO_KERNEL_UT_AICORE_PATH,
            PTO_KERNEL_UT_DISPATCHER_PATH, device_id_, config, ChipWorker::next_kernel_context_generation()
        ),
        ChipWorkerError
    );
    EXPECT_FALSE(refused.initialized());
    EXPECT_EQ(refused.device_id(), -1);
    EXPECT_NO_THROW(refused.finalize());
    EXPECT_TRUE(worker.initialized());

    AlignedCallableImage image;
    reinterpret_cast<ChipCallable *>(image.bytes)->binary_size_ = 1;
    EXPECT_THROW((void)worker.kernel_prepare_callable(image.data(), AlignedCallableImage::size()), std::runtime_error);
    ChipStorageTaskArgs args{};
    EXPECT_THROW(worker.kernel_launch(0, &args, borrowed.stream()), std::runtime_error);
    // Only launch takes a caller stream, and a null one never reaches the runtime.
    expect_chip_worker_error(
        [&] {
            worker.kernel_launch(0, &args, nullptr);
        },
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );

    EXPECT_NO_THROW(worker.finalize());
    EXPECT_FALSE(worker.initialized());
    borrowed.verify_still_usable();
}

TEST_F(KernelModeEntryTest, WorkerKernelStateMachineRejectsOutOfOrderUse) {
    BorrowedDevice borrowed(device_id_);
    ASSERT_TRUE(borrowed.ready());

    const CallConfig config{};
    AlignedCallableImage image;
    ChipStorageTaskArgs args{};

    // Before any init no runtime is bound: supported() reports false, and the
    // entries that need a context refuse with INVALID_STATE on the host side.
    {
        ChipWorker worker;
        EXPECT_FALSE(worker.kernel_mode_supported());
        expect_chip_worker_error(
            [&] {
                worker.kernel_prepare_callable(image.data(), AlignedCallableImage::size());
            },
            PTO_RUNTIME_ERR_INVALID_STATE
        );
        expect_chip_worker_error(
            [&] {
                worker.kernel_launch(0, &args, borrowed.stream());
            },
            PTO_RUNTIME_ERR_INVALID_STATE
        );
    }

    // Generation zero is not a valid context identity.
    {
        ChipWorker worker;
        expect_chip_worker_error(
            [&] {
                worker.kernel_init(
                    PTO_HOST_RUNTIME_LIB_PATH, PTO_KERNEL_UT_AICPU_PATH, PTO_KERNEL_UT_AICORE_PATH,
                    PTO_KERNEL_UT_DISPATCHER_PATH, device_id_, config, 0
                );
            },
            PTO_RUNTIME_ERR_INVALID_ARGUMENT
        );
    }

    // After finalize the worker is terminal for both identities.
    {
        ChipWorker worker;
        worker.finalize();
        EXPECT_FALSE(worker.kernel_mode_supported());
        expect_chip_worker_error(
            [&] {
                worker.kernel_init(
                    PTO_HOST_RUNTIME_LIB_PATH, PTO_KERNEL_UT_AICPU_PATH, PTO_KERNEL_UT_AICORE_PATH,
                    PTO_KERNEL_UT_DISPATCHER_PATH, device_id_, config, ChipWorker::next_kernel_context_generation()
                );
            },
            PTO_RUNTIME_ERR_INVALID_STATE
        );
    }
}

TEST_F(KernelModeEntryTest, ContextGenerationsAreNonzeroAndIncreasing) {
    const uint64_t first = ChipWorker::next_kernel_context_generation();
    const uint64_t second = ChipWorker::next_kernel_context_generation();
    EXPECT_NE(first, 0u) << "generation zero is what the C ABI rejects as invalid";
    EXPECT_GT(second, first);
}

}  // namespace
