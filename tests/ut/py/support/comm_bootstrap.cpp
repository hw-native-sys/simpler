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

#include <cassert>
#include <filesystem>
#include <future>

#include "pto/comm/async_common/sdma_constants.hpp"

#include "comm_hccl.cpp"

// Bootstrap file helpers own no device allocations. Registry destructors may
// reference these entry points, but a call would violate this CPU-only fixture.
extern "C" aclError aclrtUnmapMem(void *) { std::abort(); }
extern "C" aclError aclrtReleaseMemAddress(void *) { std::abort(); }
extern "C" aclError aclrtFreePhysical(aclrtDrvMemHandle) { std::abort(); }
extern "C" aclError aclrtFree(void *) { std::abort(); }

namespace {
void publish_rootinfo(const std::string &path, uint64_t token) {
    RootInfoFileHeader header{};
    header.run_token = token;
    HcclRootInfo rootinfo{};
    rootinfo.internal[0] = static_cast<char>(token);
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(&header), sizeof(header));
    file.write(rootinfo.internal, HCCL_ROOT_INFO_BYTES);
    file.close();
    assert(file.good());
}

void await_marker(const std::string &path) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!std::filesystem::exists(path)) {
        assert(std::chrono::steady_clock::now() < deadline);
        std::this_thread::yield();
    }
}
}  // namespace

int main(int argc, char **argv) {
    assert(argc == 4);
    const std::string stale = argv[1];
    const std::string current = argv[2];
    constexpr uint64_t kOldToken = 0x111;
    constexpr uint64_t kNewToken = 0x222;
    if (std::string(argv[3]) == "timeout") {
        assert(!file_barrier(current, 0, 2, "rootinfo_ready", kNewToken, 1));
        cleanup_handshake_files(current);
        return 0;
    }
    const int first_rank = std::stoi(argv[3]);
    publish_rootinfo(stale, kOldToken);
    const auto stale_marker = barrier_marker_path(stale, kOldToken, "rootinfo_ready", 0);
    assert(file_marker_handshake::publish_marker(stale_marker));
    // A reused path exposes a fully formed predecessor's RootInfo to a follower.
    assert(stale != current);
    assert(!std::filesystem::exists(current));
    cleanup_handshake_files(current);
    assert(std::filesystem::exists(stale));
    assert(std::filesystem::exists(stale_marker));
    publish_rootinfo(current, kNewToken);
    HcclRootInfo rootinfo{};
    uint64_t observed_token = 0;
    assert(wait_for_rootinfo(current, &rootinfo, &observed_token, 5));
    assert(observed_token == kNewToken);
    assert(rootinfo.internal[0] == static_cast<char>(kNewToken));

    auto first = std::async(std::launch::async, [&]() {
        return file_barrier(current, first_rank, 2, "rootinfo_ready", observed_token, 5);
    });
    await_marker(barrier_marker_path(current, observed_token, "rootinfo_ready", first_rank));
    assert(file_barrier(current, 1 - first_rank, 2, "rootinfo_ready", observed_token, 5));
    assert(first.get());
    cleanup_handshake_files(current);
    assert(!std::filesystem::exists(current));
    for (int rank = 0; rank < 2; ++rank) {
        assert(!std::filesystem::exists(barrier_marker_path(current, kNewToken, "rootinfo_ready", rank)));
    }
    assert(std::filesystem::exists(stale));
    assert(std::filesystem::exists(stale_marker));
    cleanup_handshake_files(stale);
}
