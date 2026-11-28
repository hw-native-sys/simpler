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

// Test-only facade over an unchanged runtime, linked as libhost_runtime_impl.so.
// All unwrapped exports resolve through that DT_NEEDED dependency. A facade is
// deliberately limited to one two-rank communicator and serial native runs.
#include <acl/acl.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <climits>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include "platform_comm/comm.h"
#include "platform_comm/comm_context.h"
#include "worker/runtime_c_api.h"

namespace {
namespace fs = std::filesystem;
constexpr size_t kWindowBytes = 512;
constexpr size_t kResultOffset = 256;
constexpr int kHostTimeoutSeconds = 10;
constexpr int kStreamTimeoutMs = 50000;

fs::path module_directory() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void *>(&module_directory), &info) == 0 || info.dli_fname == nullptr)
        throw std::runtime_error("cannot locate device-start-gate facade");
    return fs::absolute(info.dli_fname).parent_path();
}

template <typename Fn>
Fn required(void *module, const char *name) {
    dlerror();
    void *symbol = dlsym(module, name);
    const char *error = dlerror();
    if (error != nullptr || symbol == nullptr) throw std::runtime_error(std::string("missing implementation: ") + name);
    return reinterpret_cast<Fn>(symbol);
}

struct Api {
    void *module;
    decltype(&comm_init) init;
    decltype(&comm_destroy) destroy;
    decltype(&comm_alloc_windows) alloc;
    decltype(&simpler_launch_run_on_stream) launch;

    Api() {
        const auto path = module_directory() / "libhost_runtime_impl.so";
        module = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (module == nullptr) throw std::runtime_error(std::string("cannot load implementation: ") + dlerror());
        init = required<decltype(init)>(module, "comm_init");
        destroy = required<decltype(destroy)>(module, "comm_destroy");
        alloc = required<decltype(alloc)>(module, "comm_alloc_windows");
        launch = required<decltype(launch)>(module, "simpler_launch_run_on_stream");
    }
};

Api &api() {
    static Api instance;
    return instance;
}

void check(int rc, const char *operation) {
    if (rc != 0) throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(rc));
}

struct Gate {
    CommHandle business{};
    CommHandle communicator{};
    aclrtStream stream{};
    aclrtBinHandle binary{};
    aclrtFuncHandle function{};
    aclrtContext context{};
    DeviceContextHandle runner{};
    fs::path markers;
    uint64_t local_base{};
    uint64_t peer_base{};
    uint64_t epoch{};
    uint64_t verified_epoch{};
    int rank{-1};
    int device{-1};
    bool failed{};
};

std::mutex gate_mutex;
std::unique_ptr<Gate> gate;

void marker(const fs::path &path) {
    const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) throw std::runtime_error("cannot create unique gate marker: " + path.string());
    if (close(fd) != 0) throw std::runtime_error("cannot close gate marker: " + path.string());
}

fs::path marker_path(const Gate &state, const std::string &phase, int rank) {
    return state.markers / (phase + ".rank" + std::to_string(rank));
}

void rendezvous(const Gate &state, const std::string &phase) {
    marker(marker_path(state, phase, state.rank));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(kHostTimeoutSeconds);
    const auto peer = marker_path(state, phase, 1 - state.rank);
    const auto failed = marker_path(state, "failed", 1 - state.rank);
    while (!fs::exists(peer)) {
        if (fs::exists(failed)) throw std::runtime_error("peer failed before device-start gate release");
        if (std::chrono::steady_clock::now() >= deadline)
            throw std::runtime_error("device-start gate enqueue rendezvous timed out: " + phase);
    }
    if (fs::exists(failed)) throw std::runtime_error("peer rejected device-start gate");
}

fs::path marker_directory(const std::string &rootinfo) {
    // The initialized communicator's rootinfo contains a fresh run identity.
    // Hash its bytes without depending on its private header representation.
    std::ifstream input(rootinfo, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read initialized gate rootinfo");
    uint64_t hash = 14695981039346656037ULL;
    size_t count = 0;
    char byte;
    while (input.get(byte)) {
        hash = (hash ^ static_cast<unsigned char>(byte)) * 1099511628211ULL;
        ++count;
    }
    if (!input.eof() || count == 0) throw std::runtime_error("invalid initialized gate rootinfo");
    auto path = fs::path(rootinfo + ".markers-" + std::to_string(hash));
    fs::create_directories(path);
    return path;
}

void report_failure(const char *operation, const char *reason) noexcept {
    std::fprintf(stderr, "[device-start-gate] %s failed: %s; sample invalid\n", operation, reason);
    if (!gate) return;
    gate->failed = true;
    try {
        if (!gate->markers.empty()) {
            const auto path = marker_path(*gate, "failed", gate->rank);
            const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_CLOEXEC, 0600);
            if (fd >= 0) close(fd);
        }
    } catch (...) {}
}

void require_context(const Gate &state) {
    aclrtContext current{};
    check(aclrtGetCurrentContext(&current), "aclrtGetCurrentContext");
    if (current != state.context) throw std::runtime_error("multiple execution contexts are unsupported");
}

void verify_previous_gate(Gate &state) {
    if (state.epoch == state.verified_epoch) return;
    std::array<uint64_t, 4> result{};
    check(
        aclrtMemcpy(
            result.data(), sizeof(result), reinterpret_cast<void *>(state.local_base + kResultOffset), sizeof(result),
            ACL_MEMCPY_DEVICE_TO_HOST
        ),
        "read completed device-start gate evidence"
    );
    if (result[0] != state.epoch || result[1] != state.epoch || result[2] == 0 || result[3] < result[2])
        throw std::runtime_error("device-start gate did not confirm this epoch's peer arrival");
    std::fprintf(
        stderr,
        "[DEVICE_START_GATE] rank=%d device=%d epoch=%llu peer_epoch=%llu gate_begin_cycles=%llu "
        "gate_end_cycles=%llu\n",
        state.rank, state.device, static_cast<unsigned long long>(result[0]),
        static_cast<unsigned long long>(result[1]), static_cast<unsigned long long>(result[2]),
        static_cast<unsigned long long>(result[3])
    );
    state.verified_epoch = state.epoch;
}
}  // namespace

extern "C" __attribute__((visibility("default"))) CommHandle
comm_init(int rank, int nranks, void *stream, const char *rootinfo_path) {
    std::unique_lock<std::mutex> lock(gate_mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        std::fprintf(stderr, "[device-start-gate] concurrent communicator initialization is unsupported\n");
        return nullptr;
    }
    try {
        if (gate || nranks != 2 || rank < 0 || rank > 1 || stream == nullptr || rootinfo_path == nullptr ||
            *rootinfo_path == '\0')
            throw std::runtime_error("exactly one two-rank communicator per process is required");
        auto state = std::make_unique<Gate>();
        state->rank = rank;
        check(aclrtGetCurrentContext(&state->context), "aclrtGetCurrentContext");
        check(aclrtGetDevice(&state->device), "aclrtGetDevice");
        gate = std::move(state);
        gate->business = api().init(rank, nranks, stream, rootinfo_path);
        if (gate->business == nullptr) throw std::runtime_error("business comm_init returned null");
        check(aclrtSetCurrentContext(gate->context), "restore execution context");
        const std::string gate_root = std::string(rootinfo_path) + ".device-start-gate";
        gate->communicator = api().init(rank, nranks, stream, gate_root.c_str());
        if (gate->communicator == nullptr) throw std::runtime_error("gate comm_init returned null");
        check(aclrtSetCurrentContext(gate->context), "restore gate context");
        uint64_t context_address{};
        check(api().alloc(gate->communicator, kWindowBytes, &context_address), "gate comm_alloc_windows");
        check(aclrtSetCurrentContext(gate->context), "restore allocated gate context");
        check(aclrtCreateStream(&gate->stream), "aclrtCreateStream");
        CommContext context{};
        check(
            aclrtMemcpy(
                &context, sizeof(context), reinterpret_cast<void *>(context_address), sizeof(context),
                ACL_MEMCPY_DEVICE_TO_HOST
            ),
            "read gate CommContext"
        );
        if (context.rankId != static_cast<uint32_t>(rank) || context.rankNum != 2 || context.winSize < kWindowBytes)
            throw std::runtime_error("gate communication topology mismatch");
        gate->local_base = context.windowsIn[rank];
        gate->peer_base = context.windowsIn[1 - rank];
        if (gate->local_base == 0 || gate->peer_base == 0 || gate->local_base % 128 != 0 || gate->peer_base % 128 != 0)
            throw std::runtime_error("gate requires non-null 128-byte-aligned peer windows");
        check(
            aclrtMemset(reinterpret_cast<void *>(gate->local_base), kWindowBytes, 0, kWindowBytes), "zero gate window"
        );
        const auto kernel = module_directory() / "device_start_gate.o";
        check(aclrtBinaryLoadFromFile(kernel.c_str(), nullptr, &gate->binary), "load gate binary");
        check(aclrtBinaryGetFunction(gate->binary, "dsv41_device_start_gate", &gate->function), "resolve gate entry");
        gate->markers = marker_directory(gate_root);
        rendezvous(*gate, "initialized");
        std::fprintf(
            stderr, "[device-start-gate] rank=%d device=%d ready; device barrier precedes RunWall\n", rank, gate->device
        );
        return gate->business;
    } catch (const std::exception &error) {
        report_failure("comm_init", error.what());
        return nullptr;
    }
}

extern "C" __attribute__((visibility("default"))) int
simpler_launch_run(DeviceContextHandle ctx, RuntimeHandle runtime) {
    std::unique_lock<std::mutex> lock(gate_mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        std::fprintf(stderr, "[device-start-gate] concurrent launch is unsupported; sample invalid\n");
        return PTO_RUNTIME_ERR_UNSUPPORTED;
    }
    try {
        if (!gate || gate->failed) throw std::runtime_error("no healthy two-rank device-start gate");
        require_context(*gate);
        if (gate->runner != nullptr && gate->runner != ctx)
            throw std::runtime_error("multiple runners are unsupported");
        gate->runner = ctx;
        aclrtStreamStatus status{};
        check(aclrtStreamQuery(gate->stream, &status), "query gate caller stream");
        if (status != ACL_STREAM_STATUS_COMPLETE) throw std::runtime_error("overlapping native runs are unsupported");
        verify_previous_gate(*gate);
        if (gate->epoch >= INT_MAX) throw std::runtime_error("gate epoch exhausted");
        const uint64_t epoch = ++gate->epoch;
        std::array<uint64_t, 3> args{gate->local_base, gate->peer_base, epoch};
        aclrtLaunchKernelAttr engine{};
        engine.id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
        engine.value.engineType = ACL_RT_ENGINE_TYPE_AIV;
        aclrtLaunchKernelCfg config{&engine, 1};
        check(
            aclrtLaunchKernelWithHostArgs(
                gate->function, 1, gate->stream, &config, args.data(), sizeof(args), nullptr, 0
            ),
            "enqueue device-start gate"
        );
        check(api().launch(ctx, runtime, gate->stream, nullptr), "enqueue native run after device-start gate");
        rendezvous(*gate, "enqueued-" + std::to_string(epoch));
        // A synchronous copy uses its own transfer path. Queueing this behind
        // the prelude on gate->stream would deadlock its local release wait.
        const int32_t release = static_cast<int32_t>(epoch);
        check(
            aclrtMemcpy(
                reinterpret_cast<void *>(gate->local_base), sizeof(release), &release, sizeof(release),
                ACL_MEMCPY_HOST_TO_DEVICE
            ),
            "release local device-start gate"
        );
        return 0;
    } catch (const std::exception &error) {
        report_failure("simpler_launch_run", error.what());
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}

// Explicit rejection prevents callers from bypassing the facade through its
// implementation dependency. The facade calls the implementation symbol itself.
extern "C" __attribute__((visibility("default"))) int
simpler_launch_run_on_stream(DeviceContextHandle, RuntimeHandle, void *, RuntimeHandle) {
    std::fprintf(stderr, "[device-start-gate] external caller streams are unsupported; sample invalid\n");
    return PTO_RUNTIME_ERR_UNSUPPORTED;
}

extern "C" __attribute__((visibility("default"))) int
simpler_launch_run_joined(DeviceContextHandle, RuntimeHandle, RuntimeHandle) {
    std::fprintf(stderr, "[device-start-gate] joined launches are unsupported; sample invalid\n");
    return PTO_RUNTIME_ERR_UNSUPPORTED;
}

extern "C" __attribute__((visibility("default"))) int comm_destroy(CommHandle handle) {
    std::unique_lock<std::mutex> lock(gate_mutex, std::try_to_lock);
    if (!lock.owns_lock()) return COMM_RECLAMATION_UNCONFIRMED;
    try {
        if (!gate || gate->business != handle) throw std::runtime_error("unknown communicator");
        if (gate->failed) throw std::runtime_error("failed gate resources are retained until process teardown");
        require_context(*gate);
        check(aclrtSynchronizeStreamWithTimeout(gate->stream, kStreamTimeoutMs), "drain gate caller stream");
        verify_previous_gate(*gate);
        // The communicator owns its window; it cannot be released until all
        // commands that might access either peer's window have completed.
        rendezvous(*gate, "drained");
        check(api().destroy(gate->communicator), "destroy gate communicator");
        gate->communicator = nullptr;
        check(aclrtSetCurrentContext(gate->context), "restore destruction context");
        check(aclrtBinaryUnLoad(gate->binary), "unload gate binary");
        gate->binary = nullptr;
        check(aclrtDestroyStream(gate->stream), "destroy gate caller stream");
        gate->stream = nullptr;
        check(api().destroy(gate->business), "destroy business communicator");
        check(aclrtSetCurrentContext(gate->context), "restore business destruction context");
        gate.reset();
        return 0;
    } catch (const std::exception &error) {
        report_failure("comm_destroy", error.what());
        return COMM_RECLAMATION_UNCONFIRMED;
    }
}
