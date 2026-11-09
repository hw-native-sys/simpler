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
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "platform_comm/ccu/ccu_host.h"

namespace {
using Clock = std::chrono::steady_clock;
constexpr uint32_t TIMEOUT_MS = 120000;
constexpr size_t WINDOW_BYTES = 24576;
constexpr size_t RESULT_BYTES = 128;

void Check(int code, const char *operation) {
    if (code != 0) throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(code));
}

void Check(CcuStatus status, const char *operation, const CcuSession &session) {
    if (status != CcuStatus::OK) {
        throw std::runtime_error(
            std::string(operation) + " failed: status=" + std::to_string(static_cast<uint32_t>(status)) +
            " native=" + std::to_string(session.native_error())
        );
    }
}

void Publish(const std::filesystem::path &path, const void *data, size_t size) {
    const auto temporary = path.string() + ".tmp";
    std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
    if (!file || !file.write(static_cast<const char *>(data), size)) {
        throw std::runtime_error("cannot publish rendezvous file " + path.string());
    }
    file.close();
    if (!file) throw std::runtime_error("cannot flush rendezvous file " + path.string());
    std::filesystem::rename(temporary, path);
}

void Receive(const std::filesystem::path &path, void *data, size_t size, Clock::time_point deadline) {
    while (!std::filesystem::exists(path)) {
        if (Clock::now() >= deadline) throw std::runtime_error("rendezvous timeout: " + path.string());
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (std::filesystem::file_size(path) != size) {
        throw std::runtime_error("rendezvous size mismatch: " + path.string());
    }
    std::ifstream file(path, std::ios::binary);
    if (!file || !file.read(static_cast<char *>(data), size)) {
        throw std::runtime_error("cannot read rendezvous file " + path.string());
    }
}

std::vector<uint8_t> ReadModule(const char *path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open embedded module");
    const auto size = file.tellg();
    if (size <= 0) throw std::runtime_error("empty embedded module");
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char *>(bytes.data()), bytes.size())) {
        throw std::runtime_error("cannot read embedded module");
    }
    return bytes;
}

struct Bootstrap {
    std::filesystem::path directory;
    uint32_t rank;
    uint64_t epoch{};

    std::vector<uint8_t> RootInfo() {
        std::filesystem::create_directories(directory);
        if (!std::filesystem::create_directory(directory / ("rank" + std::to_string(rank)))) {
            throw std::runtime_error("rendezvous directory must be fresh for each invocation");
        }
        struct Root {
            uint64_t epoch;
            HcclRootInfo info;
        } root{};
        const auto path = directory / "root.bin";
        if (rank == 0) {
            std::random_device random;
            root.epoch = (uint64_t{random()} << 32) | random();
            if (!root.epoch) root.epoch = 1;
            Check(HcclGetRootInfo(&root.info), "HcclGetRootInfo");
            Publish(path, &root, sizeof(root));
        } else {
            Receive(path, &root, sizeof(root), Clock::now() + std::chrono::milliseconds(TIMEOUT_MS));
        }
        epoch = root.epoch;
        if (!epoch) throw std::runtime_error("invalid rendezvous epoch");
        std::vector<uint8_t> bytes(sizeof(root.info));
        std::memcpy(bytes.data(), &root.info, bytes.size());
        return bytes;
    }

    int Exchange(
        uint64_t generation, uint32_t phase, const void *input, uint32_t input_bytes, void *output, uint32_t rank_bytes,
        uint32_t timeout_ms
    ) noexcept {
        try {
            if (!input || !output || !input_bytes || input_bytes != rank_bytes) return 1;
            const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
            const auto prefix =
                std::to_string(epoch) + "-" + std::to_string(generation) + "-" + std::to_string(phase) + "-";
            Publish(directory / (prefix + std::to_string(rank)), input, input_bytes);
            for (uint32_t peer = 0; peer < 2; ++peer) {
                Receive(
                    directory / (prefix + std::to_string(peer)), static_cast<uint8_t *>(output) + peer * rank_bytes,
                    rank_bytes, deadline
                );
            }
            return 0;
        } catch (const std::exception &error) {
            std::cerr << "rank=" << rank << " bootstrap: " << error.what() << '\n';
            return 1;
        }
    }
};

// Cleanup is explicit and requires proof that both AIV and CCU have stopped.
struct Resources {
    void *window{};
    void *result{};
    aclrtStream stream{};
    aclrtBinHandle binary{};
    std::shared_ptr<Bootstrap> bootstrap;
};
}  // namespace

int main(int argc, char **argv) {
    if (argc != 7) {
        std::cerr << "usage: " << argv[0] << " device rank rendezvous_dir rounds kernel.o module.so\n";
        return 2;
    }
    auto resources = std::make_shared<Resources>();
    std::unique_ptr<CcuSession> session;
    bool run_retained = false;
    bool aiv_may_be_running = false;
    int rank = -1;
    try {
        const int device = std::stoi(argv[1]);
        rank = std::stoi(argv[2]);
        const uint64_t rounds = std::stoull(argv[4]);
        if (device < 0 || rank < 0 || rank > 1 || !rounds || rounds > 4096) {
            throw std::runtime_error("invalid device, rank or rounds (supported: 1..4096)");
        }
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(device), "aclrtSetDevice");
        resources->bootstrap = std::make_shared<Bootstrap>(Bootstrap{argv[3], static_cast<uint32_t>(rank)});
        CcuSessionConfig config;
        config.root_info = resources->bootstrap->RootInfo();
        config.allocation_id = 1;
        config.epoch = resources->bootstrap->epoch;
        config.domain_owner = 1;
        config.callable_id = 0;
        config.device_id = device;
        config.rank = rank;
        config.timeout_ms = TIMEOUT_MS;
        config.parent_owner = resources;
        config.exchange = [bootstrap = resources->bootstrap](
                              uint64_t generation, uint32_t phase, const void *input, uint32_t input_bytes,
                              void *output, uint32_t rank_bytes, uint32_t timeout_ms
                          ) {
            return bootstrap->Exchange(generation, phase, input, input_bytes, output, rank_bytes, timeout_ms);
        };
        session = std::make_unique<CcuSession>(std::move(config));
        Check(session->initialize(), "initialize", *session);
        Check(aclrtMalloc(&resources->window, WINDOW_BYTES, ACL_MEM_MALLOC_HUGE_FIRST), "allocate window");
        Check(aclrtMalloc(&resources->result, RESULT_BYTES, ACL_MEM_MALLOC_HUGE_FIRST), "allocate result");
        Check(aclrtMemset(resources->result, RESULT_BYTES, 0xff, RESULT_BYTES), "initialize result");
        Check(
            session->prepare_memory(1, reinterpret_cast<uint64_t>(resources->window), WINDOW_BYTES), "prepare memory",
            *session
        );
        const auto module = ReadModule(argv[6]);
        Check(
            session->register_functions({{module, "broadcast_root0"}, {module, "broadcast_root1"}}),
            "register embedded functions", *session
        );
        CcuDeviceContextView context{};
        Check(session->get_device_context(context), "publish context", *session);
        Check(aclrtCreateStream(&resources->stream), "create AIV stream");
        Check(aclrtBinaryLoadFromFile(argv[5], nullptr, &resources->binary), "load AIV kernel");
        aclrtFuncHandle function{};
        Check(aclrtBinaryGetFunction(resources->binary, "ccu_session_onboard", &function), "find AIV kernel");
        Check(session->retain_run(), "retain run", *session);
        run_retained = true;
        std::array<uint64_t, 6> arguments{
            context.addr,
            context.size,
            reinterpret_cast<uint64_t>(resources->window),
            reinterpret_cast<uint64_t>(resources->result),
            rounds,
            static_cast<uint64_t>(rank)
        };
        aclrtLaunchKernelAttr engine{};
        engine.id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
        engine.value.engineType = ACL_RT_ENGINE_TYPE_AIV;
        aclrtLaunchKernelCfg launch_config{&engine, 1};
        aiv_may_be_running = true;
        Check(
            aclrtLaunchKernelWithHostArgs(
                function, 1, resources->stream, &launch_config, arguments.data(), sizeof(arguments), nullptr, 0
            ),
            "launch AIV"
        );
        Check(aclrtSynchronizeStreamWithTimeout(resources->stream, TIMEOUT_MS), "wait AIV");
        aiv_may_be_running = false;
        std::array<uint64_t, RESULT_BYTES / sizeof(uint64_t)> result{};
        Check(
            aclrtMemcpy(result.data(), sizeof(result), resources->result, sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST),
            "read result"
        );
        Check(session->release_run(), "release run", *session);
        run_retained = false;
        std::array<uint64_t, 2> stopped{};
        const uint64_t ready = 1;
        Check(
            resources->bootstrap->Exchange(
                UINT64_MAX, 0, &ready, sizeof(ready), stopped.data(), sizeof(ready), TIMEOUT_MS
            ),
            "AIV stop rendezvous"
        );
        Check(session->close(CcuCloseMode::NORMAL), "close session", *session);
        Check(aclrtDestroyStream(resources->stream), "destroy AIV stream");
        resources->stream = nullptr;
        Check(aclrtBinaryUnLoad(resources->binary), "unload AIV binary");
        resources->binary = nullptr;
        Check(aclrtFree(resources->result), "free result");
        resources->result = nullptr;
        Check(aclrtFree(resources->window), "free window");
        resources->window = nullptr;
        std::cout << "rank=" << rank << " status=" << result[0] << " completed_rounds=" << result[1]
                  << " requested_rounds=" << rounds << " mismatch_index=" << result[2] << std::endl;
        Check(aclFinalize(), "aclFinalize");
        return result[0] == 0 && result[1] == rounds ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr << "rank=" << rank << " " << error.what() << " aiv_may_be_running=" << aiv_may_be_running
                  << " run_retained=" << run_retained << "; no device reset or unconfirmed resource release\n";
        if (session) (void)session.release();
        return 1;
    }
}
