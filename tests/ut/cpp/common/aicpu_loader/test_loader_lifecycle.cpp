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
// RTS is an external transport. Real runtime modules remain mapped beyond
// registration unload; production file publication and TMR callable cleanup
// are under test. This control makes no physical device-unmap claim.
#include <dlfcn.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "acl/acl.h"
#include "aicpu/aicpu_schedule/aicpu_context.h"
#include "aicpu_loader/finalize_args.h"
#include "aicpu_loader/device/aicpu_dispatcher.h"
#include <set>
#include <sys/stat.h>
#include "aicpu_loader/host/load_aicpu_op.h"
#include "utils/elf_build_id.h"

namespace {
namespace fs = std::filesystem;
std::vector<char> ReadBytes(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), {}};
}
struct Binary {
    void *module{};
    std::map<std::string, void *> entries;
    bool loaded = true;
};
struct Transport {
    std::string dir;
    int bootstraps = 0;
    std::set<void *> allocations;
    int free_error = 0;
    int launch_error = 0;
    bool execute_on_launch_error = false;
    int copy_error = 0;
    std::vector<std::string> json_paths;
    std::vector<std::unique_ptr<Binary>> binaries;
    int sync_error = 0;
    int unload_error = 0;
    int resolve_failure_at = -1;
    int resolve_calls = 0;
    int launches = 0;
    ~Transport() {
        for (auto &binary : binaries)
            dlclose(binary->module);
        for (void *ptr : allocations)
            std::free(ptr);
        for (const auto &path : json_paths)
            fs::remove(path);
        fs::remove_all(dir);
    }
};
Transport *transport;
}  // namespace

extern "C" void DlogRecord(int, int, const char *, ...) {}
extern "C" void unified_log_info(const char *, const char *, ...) {}
extern "C" void unified_log_error(const char *, const char *, ...) {}
namespace aicpu {
status_t aicpuGetContext(aicpuContext_t *ctx) {
    *ctx = {0, 0, getpid(), 0};
    return AICPU_ERROR_NONE;
}
uint32_t GetUniqueVfId() { return 0; }
}  // namespace aicpu
extern "C" aclError aclrtMalloc(void **ptr, size_t bytes, int) {
    *ptr = std::malloc(bytes);
    if (*ptr != nullptr) transport->allocations.insert(*ptr);
    return *ptr == nullptr ? 1 : 0;
}
extern "C" aclError aclrtFree(void *ptr) {
    if (transport->free_error != 0) return transport->free_error;
    EXPECT_EQ(transport->allocations.erase(ptr), 1);
    std::free(ptr);
    return 0;
}
extern "C" aclError aclrtMemcpy(void *dst, size_t, const void *src, size_t bytes, int kind) {
    if (kind == ACL_MEMCPY_DEVICE_TO_HOST && transport->copy_error != 0) return transport->copy_error;
    std::memcpy(dst, src, bytes);
    return 0;
}
extern "C" aclError aclrtSynchronizeStream(void *) { return transport->sync_error; }
extern "C" rtError_t rtAicpuKernelLaunchExWithArgs(
    rtKernelType_t, const char *, uint32_t, rtAicpuArgsEx_t *args, void *, rtStream_t, uint32_t
) {
    struct KernelArgs {
        uint64_t unused[5];
        uint64_t device_args;
    };
    const auto *kernel = static_cast<const KernelArgs *>(args->args);
    const auto *device = reinterpret_cast<const simpler_dispatcher::BootstrapDeviceArgs *>(kernel->device_args);
    ++transport->bootstraps;
    const auto *bytes = reinterpret_cast<const char *>(device->inner_so_bin);
    const auto size = device->inner_so_len;
    const auto path =
        simpler_dispatcher::MakeInnerSoPath(simpler::common::utils::elf_build_id_64(bytes, size), device->device_id);
    auto &result = *reinterpret_cast<simpler_dispatcher::BootstrapResult *>(device->result_addr);
    result.device_id = static_cast<int32_t>(device->device_id);
    const bool success =
        simpler_dispatcher::WriteBytes(transport->dir + "/" + fs::path(path).filename().string(), bytes, size, result);
    result.status =
        success ? simpler_dispatcher::BootstrapStatus::Success : simpler_dispatcher::BootstrapStatus::Failed;
    return 0;  // Expected upload failures are acknowledged without faulting the stream.
}
extern "C" rtError_t rtsBinaryLoadFromFile(const char *path, const rtLoadBinaryConfig_t *config, void **handle) {
    EXPECT_EQ(config->options[0].value.cpuKernelMode, 0);
    transport->json_paths.emplace_back(path);
    const auto bytes = ReadBytes(path);
    const std::string json(bytes.begin(), bytes.end());
    const std::regex entry(
        R"REGEX("([^"]+)": \{\s*"opInfo": \{\s*"functionName": "([^"]+)",\s*"kernelSo": "([^"]+)")REGEX"
    );
    auto binary = std::make_unique<Binary>();
    for (auto i = std::sregex_iterator(json.begin(), json.end(), entry); i != std::sregex_iterator(); ++i) {
        if (binary->module == nullptr)
            binary->module = dlopen((transport->dir + "/" + (*i)[3].str()).c_str(), RTLD_NOW | RTLD_LOCAL);
        if (binary->module == nullptr) return 2;
        binary->entries[(*i)[1]] = dlsym(binary->module, (*i)[2].str().c_str());
    }
    if (binary->module == nullptr) return 3;
    *handle = binary.get();
    transport->binaries.push_back(std::move(binary));
    return 0;
}
extern "C" rtError_t rtsBinaryUnload(void *handle) {
    if (transport->unload_error != 0) return transport->unload_error;
    auto *binary = static_cast<Binary *>(handle);
    EXPECT_TRUE(binary->loaded);
    // CANN registration unload does not promise to unload its immutable DSO.
    // Keep the real local module mapped until the provider fixture ends.
    binary->loaded = false;
    return 0;
}
extern "C" rtError_t rtsFuncGetByName(void *handle, const char *name, rtFuncHandle *function) {
    if (transport->resolve_calls++ == transport->resolve_failure_at) return 71;
    const auto &entries = static_cast<Binary *>(handle)->entries;
    const auto entry = entries.find(name);
    if (entry == entries.end() || entry->second == nullptr) return 72;
    *function = entry->second;
    return 0;
}
extern "C" rtError_t
rtsLaunchCpuKernel(rtFuncHandle function, uint32_t, rtStream_t, const rtKernelLaunchCfg_t *, rtCpuKernelArgs_t *args) {
    ++transport->launches;
    if (transport->launch_error != 0 && !transport->execute_on_launch_error) return transport->launch_error;
    const int rc = reinterpret_cast<int (*)(void *)>(function)(args->baseArgs.args);
    return transport->launch_error != 0 ? transport->launch_error : rc;
}

namespace {
class LoadOwnership : public testing::Test {
protected:
    Transport state;
    std::vector<char> image;
    std::vector<char> orchestration;
    std::vector<char> second_orchestration;
    void SetUp() override {
        char path[] = "/tmp/simpler-load-ownership-XXXXXX";
        ASSERT_NE(mkdtemp(path), nullptr);
        state.dir = path;
        transport = &state;
        image = ReadBytes(TEST_RUNTIME_IMAGE);
        orchestration = ReadBytes(TEST_ORCHESTRATION);
        second_orchestration = ReadBytes(TEST_SECOND_ORCHESTRATION);
        ASSERT_FALSE(image.empty());
        ASSERT_FALSE(orchestration.empty());
        ASSERT_NE(orchestration, second_orchestration);
    }
    int Bootstrap(host::LoadAicpuOp &loader) {
        const char dispatcher = 'd';
        return loader.BootstrapDispatcher(&dispatcher, 1, image.data(), image.size(), reinterpret_cast<void *>(1), 0);
    }
    int Init(host::LoadAicpuOp &loader) { return loader.Init({host::KernelNames::RegisterCallableName}); }
    int Register(host::LoadAicpuOp &loader, int slot, const char *name, const std::vector<char> &bytes) {
        RegisterCallableArgs args{};
        args.active_callable_id = slot;
        args.dev_orch_so_addr = reinterpret_cast<uint64_t>(bytes.data());
        args.dev_orch_so_size = bytes.size();
        std::snprintf(args.device_orch_func_name, sizeof(args.device_orch_func_name), "%s", name);
        std::snprintf(args.device_orch_config_name, sizeof(args.device_orch_config_name), "%s_config", name);
        args.residency_generation = 1;
        return loader.LaunchBuiltInOp(nullptr, &args, sizeof(args), 1, host::KernelNames::RegisterCallableName);
    }
    int Finalize(host::LoadAicpuOp &loader) { return loader.Finalize(reinterpret_cast<void *>(1)); }
    int Live(Binary &binary) { return reinterpret_cast<int (*)()>(dlsym(binary.module, "probe_live_handles"))(); }
    void FailClose(Binary &binary, int attempt) {
        reinterpret_cast<void (*)(int)>(dlsym(binary.module, "probe_fail_close_at"))(attempt);
    }
    int Selected(Binary &binary, int slot) {
        auto probe = reinterpret_cast<int (*)(int)>(dlsym(binary.module, "probe_selected_config"));
        EXPECT_NE(probe, nullptr);
        return probe == nullptr ? -1 : probe(slot);
    }
};

TEST_F(LoadOwnership, SequentialGenerationsReuseImageAndReleaseActualCallableHandles) {
    void *resident = nullptr;
    ino_t inode = 0;
    for (int generation = 0; generation < 3; ++generation) {
        host::LoadAicpuOp loader;
        ASSERT_EQ(Bootstrap(loader), 0);
        ASSERT_EQ(Init(loader), 0);
        auto &binary = *state.binaries.back();
        EXPECT_EQ(Live(binary), 0);
        if (resident != nullptr) EXPECT_EQ(binary.module, resident);
        resident = binary.module;
        const auto bytes = ReadBytes(state.json_paths.back());
        const std::string json(bytes.begin(), bytes.end());
        std::smatch name;
        ASSERT_TRUE(std::regex_search(json, name, std::regex(R"REGEX("kernelSo": "([^"]+)")REGEX")));
        struct stat info{};
        ASSERT_EQ(stat((state.dir + "/" + name[1].str()).c_str(), &info), 0);
        if (inode != 0) EXPECT_EQ(info.st_ino, inode);
        inode = info.st_ino;
        const bool opposite = generation % 2;
        ASSERT_EQ(Register(loader, opposite ? 1 : 0, "owner_x", orchestration), 0);
        ASSERT_EQ(Register(loader, opposite ? 0 : 1, "owner_y", second_orchestration), 0);
        EXPECT_EQ(Selected(binary, opposite ? 1 : 0), 3);
        EXPECT_EQ(Selected(binary, opposite ? 0 : 1), 17);
        EXPECT_EQ(Live(binary), 2);
        ASSERT_EQ(Finalize(loader), 0);
        EXPECT_EQ(Live(binary), 0);
        EXPECT_FALSE(loader.has_live_resources());
        EXPECT_FALSE(fs::exists(state.json_paths.back()));
        EXPECT_EQ(std::distance(fs::directory_iterator(state.dir), fs::directory_iterator()), 1);
    }
}

TEST_F(LoadOwnership, PartialDeviceCleanupFailureRetainsRegistrationAndRetriesOnlyRemainingHandles) {
    host::LoadAicpuOp loader;
    ASSERT_EQ(Bootstrap(loader), 0);
    ASSERT_EQ(Init(loader), 0);
    auto &binary = *state.binaries.back();
    ASSERT_EQ(Register(loader, 0, "owner_x", orchestration), 0);
    ASSERT_EQ(Register(loader, 1, "owner_y", second_orchestration), 0);
    FailClose(binary, 0);
    EXPECT_NE(Finalize(loader), 0);  // enqueue succeeded; device result failed
    EXPECT_EQ(Live(binary), 1);
    EXPECT_TRUE(binary.loaded);
    EXPECT_TRUE(loader.has_live_resources());
    EXPECT_TRUE(fs::exists(state.json_paths.back()));
    FailClose(binary, -1);
    ASSERT_EQ(Finalize(loader), 0);
    EXPECT_EQ(Live(binary), 0);
    EXPECT_FALSE(binary.loaded);
}

TEST_F(LoadOwnership, AmbiguousWaitAndCopyRetainOneRequestUntilAcknowledged) {
    host::LoadAicpuOp loader;
    ASSERT_EQ(Bootstrap(loader), 0);
    ASSERT_EQ(Init(loader), 0);
    ASSERT_EQ(Register(loader, 0, "owner_x", orchestration), 0);
    state.sync_error = 73;
    EXPECT_EQ(Finalize(loader), 73);
    const int launches = state.launches;
    EXPECT_EQ(state.allocations.size(), 1);
    EXPECT_TRUE(state.binaries.back()->loaded);
    state.sync_error = 0;
    state.copy_error = 74;
    EXPECT_EQ(Finalize(loader), 74);
    EXPECT_EQ(state.launches, launches);
    EXPECT_EQ(state.allocations.size(), 1);
    state.copy_error = 0;
    ASSERT_EQ(Finalize(loader), 0);
    EXPECT_EQ(state.launches, launches);
    EXPECT_TRUE(state.allocations.empty());
}

TEST_F(LoadOwnership, FailedSubmissionFreeAndUnloadPreserveRetryDebt) {
    host::LoadAicpuOp loader;
    ASSERT_EQ(Bootstrap(loader), 0);
    ASSERT_EQ(Init(loader), 0);
    ASSERT_EQ(Register(loader, 0, "owner_x", orchestration), 0);
    state.launch_error = 75;
    EXPECT_EQ(Finalize(loader), 75);
    EXPECT_EQ(Live(*state.binaries.back()), 1);
    EXPECT_TRUE(state.binaries.back()->loaded);
    // The transport's enqueue error did not prove non-submission. With no
    // device acknowledgement, retry retains it and never submits again.
    const int refused_launches = state.launches;
    state.launch_error = 0;
    EXPECT_NE(Finalize(loader), 0);
    EXPECT_EQ(state.launches, refused_launches);
    loader.ForgetWithoutUnload();  // terminal abandonment control owns retained module/buffer
}

TEST_F(LoadOwnership, FreeAndUnloadFailuresRetryAcknowledgedCleanupWithoutAnotherLaunch) {
    host::LoadAicpuOp loader;
    ASSERT_EQ(Bootstrap(loader), 0);
    ASSERT_EQ(Init(loader), 0);
    ASSERT_EQ(Register(loader, 0, "owner_x", orchestration), 0);
    state.free_error = 76;
    EXPECT_EQ(Finalize(loader), 76);
    EXPECT_EQ(Live(*state.binaries.back()), 0);
    const int launches = state.launches;
    state.free_error = 0;
    state.unload_error = 77;
    EXPECT_EQ(Finalize(loader), 77);
    EXPECT_EQ(state.launches, launches);
    EXPECT_TRUE(loader.has_live_resources());
    EXPECT_TRUE(fs::exists(state.json_paths.back()));
    state.unload_error = 0;
    ASSERT_EQ(Finalize(loader), 0);
    EXPECT_EQ(state.launches, launches);
    EXPECT_FALSE(loader.has_live_resources());
}
TEST_F(LoadOwnership, BootstrapWaitFailureRetainsItsReadBuffersUntilCloseProvesCompletion) {
    host::LoadAicpuOp loader;
    state.sync_error = 78;
    EXPECT_EQ(Bootstrap(loader), 78);
    EXPECT_TRUE(loader.has_live_resources());
    EXPECT_EQ(state.allocations.size(), 4);
    EXPECT_NE(Init(loader), 0);
    EXPECT_EQ(Finalize(loader), 78);
    EXPECT_EQ(state.allocations.size(), 4);
    state.sync_error = 0;
    ASSERT_EQ(Finalize(loader), 0);
    EXPECT_TRUE(state.allocations.empty());
    EXPECT_FALSE(loader.has_live_resources());
}

TEST_F(LoadOwnership, BootstrapCopyAndFreeFailureKeepTheSameFourBufferRequest) {
    host::LoadAicpuOp loader;
    state.copy_error = 81;
    EXPECT_EQ(Bootstrap(loader), 81);
    EXPECT_EQ(state.allocations.size(), 4);
    EXPECT_NE(Init(loader), 0);
    EXPECT_EQ(state.bootstraps, 1);
    state.copy_error = 0;
    state.free_error = 82;
    EXPECT_EQ(Finalize(loader), 82);
    EXPECT_EQ(state.allocations.size(), 4);
    state.free_error = 0;
    // Acknowledgement is already saved, so another copy failure must not
    // force rereading a result that a partial free could already have retired.
    state.copy_error = 83;
    EXPECT_EQ(Finalize(loader), 0);
    EXPECT_TRUE(state.allocations.empty());
    EXPECT_EQ(state.bootstraps, 1);
}

TEST_F(LoadOwnership, PartialInitRetainsJsonAndRejectsLaunchUntilUnloadRetry) {
    host::LoadAicpuOp loader;
    ASSERT_EQ(Bootstrap(loader), 0);
    state.resolve_failure_at = 1;
    state.unload_error = 79;
    EXPECT_EQ(Init(loader), 71);
    ASSERT_EQ(state.json_paths.size(), 1);
    EXPECT_TRUE(fs::exists(state.json_paths.back()));
    int args = 0;
    EXPECT_NE(loader.LaunchBuiltInOp(nullptr, &args, sizeof(args), 1, host::KernelNames::RunName), 0);
    EXPECT_EQ(state.launches, 0);
    EXPECT_NE(Bootstrap(loader), 0);
    EXPECT_EQ(Finalize(loader), 79);
    state.unload_error = 0;
    ASSERT_EQ(Finalize(loader), 0);
    EXPECT_FALSE(fs::exists(state.json_paths.back()));
    EXPECT_FALSE(loader.has_live_resources());
}
TEST_F(LoadOwnership, EnqueueErrorAfterExecutingCleanupUsesOriginalAcknowledgement) {
    host::LoadAicpuOp loader;
    ASSERT_EQ(Bootstrap(loader), 0);
    ASSERT_EQ(Init(loader), 0);
    ASSERT_EQ(Register(loader, 0, "owner_x", orchestration), 0);
    state.launch_error = 80;
    state.execute_on_launch_error = true;
    EXPECT_EQ(Finalize(loader), 80);
    const int launches = state.launches;
    EXPECT_EQ(Live(*state.binaries.back()), 0);
    EXPECT_EQ(state.allocations.size(), 1);
    EXPECT_TRUE(state.binaries.back()->loaded);
    state.launch_error = 0;
    ASSERT_EQ(Finalize(loader), 0);
    EXPECT_EQ(state.launches, launches);
    EXPECT_TRUE(state.allocations.empty());
}
}  // namespace
