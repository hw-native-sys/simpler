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

// Test-only SDK adapter. It never changes a production launch argument or result.
#include <acl/acl.h>
#include <dlfcn.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <thread>

#include "runtime/kernel.h"  // NOLINT(build/include_subdir)

namespace {
struct Tag {
    uint64_t token{};
    std::array<uint64_t, 8> identity{};
    bool issued{};
};
struct Gate {
    Tag tag;
    aclrtContext context{};
    aclrtStream compute{};
    aclrtNotify notify{};
    aclrtEvent entered{};
};
thread_local Tag current;
std::mutex state_mutex, send_mutex;
std::atomic<int> channel{-1};
char socket_name[100]{};
std::map<uint64_t, Gate> gates;
std::set<uint64_t> cancelled;
aclrtContext control_context{};
aclrtStream control{};

void check(int rc, const char *operation) {
    if (rc) {
        std::fprintf(stderr, "DFX Notify %s failed: %d\n", operation, rc);
        std::fflush(stderr);
        std::_Exit(80);
    }
}

void emit(const char *kind, const Tag &tag, int activated = 0, int launched = 0, int disposition = 0) {
    char data[768];
    int size = std::snprintf(
        data, sizeof(data),
        "{\"type\":\"%s\",\"pid\":%d,\"token\":%llu,\"identity\":[%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu],"
        "\"run_id\":%llu,\"activated\":%d,\"launched\":%d,\"disposition\":%d}",
        kind, getpid(), static_cast<unsigned long long>(tag.token), static_cast<unsigned long long>(tag.identity[0]),
        static_cast<unsigned long long>(tag.identity[1]), static_cast<unsigned long long>(tag.identity[2]),
        static_cast<unsigned long long>(tag.identity[3]), static_cast<unsigned long long>(tag.identity[4]),
        static_cast<unsigned long long>(tag.identity[5]), static_cast<unsigned long long>(tag.identity[6]),
        static_cast<unsigned long long>(tag.identity[7]), static_cast<unsigned long long>(tag.identity[1]), activated,
        launched, disposition
    );
    std::lock_guard<std::mutex> lock(send_mutex);
    if (send(channel.load(), data, size, MSG_NOSIGNAL) != size) std::_Exit(81);
}

void serve() {
    char command[128];
    for (;;) {
        auto size = recv(channel.load(), command, sizeof(command) - 1, 0);
        if (size <= 0) return;
        command[size] = 0;
        char verb[16]{};
        unsigned long long token = 0;
        if (std::strcmp(command, "close") == 0) {
            check(gates.empty() ? 0 : 1, "unretired gates");
            if (control) {
                check(aclrtSetCurrentContext(control_context), "close context");
                check(aclrtDestroyStream(control), "destroy control stream");
            }
            emit("closed", {});
            close(channel.exchange(-1));
            return;
        }
        if (std::sscanf(command, "%15s %llu", verb, &token) != 2) std::_Exit(82);
        Gate gate;
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            if (std::strcmp(verb, "release") == 0 && !gates.count(token)) {
                cancelled.insert(token);
                Tag tag;
                tag.token = token;
                emit("released", tag);
                continue;
            }
            gate = gates.at(token);
        }
        check(aclrtSetCurrentContext(gate.context), "control context");
        if (std::strcmp(verb, "entered") == 0) {
            check(aclrtSynchronizeEventWithTimeout(gate.entered, 30000), "entered marker");
            emit("entered", gate.tag);
        } else if (std::strcmp(verb, "release") == 0) {
            check(aclrtRecordNotify(gate.notify, control), "record release");
            check(aclrtSynchronizeStreamWithTimeout(control, 30000), "release stream");
            emit("released", gate.tag);
        } else if (std::strcmp(verb, "retire") == 0) {
            check(aclrtSynchronizeStreamWithTimeout(gate.compute, 30000), "retire compute stream");
            check(aclrtDestroyEvent(gate.entered), "destroy marker");
            check(aclrtDestroyNotify(gate.notify), "destroy notify");
            {
                std::lock_guard<std::mutex> lock(state_mutex);
                gates.erase(token);
            }
            emit("retired", gate.tag);
        } else std::_Exit(83);
    }
}

void connect_control() {
    if (channel.load() >= 0) return;
    int fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path + 1, socket_name, std::strlen(socket_name));
    if (connect(
            fd, reinterpret_cast<sockaddr *>(&address), offsetof(sockaddr_un, sun_path) + 1 + std::strlen(socket_name)
        ))
        std::_Exit(84);
    channel.store(fd);
    std::thread(serve).detach();
}
}  // namespace

extern "C" void joined_notify_configure(const char *name) {
    if (std::strlen(name) >= sizeof(socket_name)) std::_Exit(85);
    std::strcpy(socket_name, name);  // NOLINT(runtime/printf)
}

extern "C" void joined_notify_begin(uint64_t token, const uint64_t *identity) {
    current = {};
    current.token = token;
    std::memcpy(current.identity.data(), identity, sizeof(current.identity));
    connect_control();
}

extern "C" void joined_notify_end(int activated, int launched, int disposition) {
    emit("activation", current, activated, launched, disposition);
    current = {};
}

// A notification of the lane's actual launch state, including progress retries.
extern "C" void joined_notify_launched(const uint64_t *identity) {
    Tag tag;
    std::memcpy(tag.identity.data(), identity, sizeof(tag.identity));
    emit("launch", tag, 1, 1);
}

extern "C" rtError_t rtKernelLaunchWithHandleV2(
    void *handle, uint64_t key, uint32_t blocks, rtArgsEx_t *args, rtSmDesc_t *sm, rtStream_t stream,
    const rtTaskCfgInfo_t *config
) {
    static auto real =
        reinterpret_cast<decltype(&rtKernelLaunchWithHandleV2)>(dlsym(RTLD_NEXT, "rtKernelLaunchWithHandleV2"));
    if (!real) std::_Exit(86);
    if (current.token) {
        if (current.issued) std::_Exit(87);  // One real AICore worker launch per activated run.
        current.issued = true;
        std::lock_guard<std::mutex> lock(state_mutex);
        if (!cancelled.count(current.token)) {
            Gate gate;
            gate.tag = current;
            gate.compute = stream;
            check(aclrtGetCurrentContext(&gate.context), "launch context");
            if (!control) {
                control_context = gate.context;
                check(aclrtCreateStream(&control), "create control stream");
            }
            check(control_context == gate.context ? 0 : 1, "changed device context");
            check(aclrtCreateNotify(&gate.notify, 0), "create notify");
            check(aclrtCreateEventExWithFlag(&gate.entered, ACL_EVENT_SYNC), "create marker");
            check(aclrtRecordEvent(gate.entered, stream), "record marker");
            check(aclrtWaitAndResetNotify(gate.notify, stream, 0), "enqueue wait");
            gates.emplace(current.token, gate);
            emit("armed", current);
        }
    }
    return real(handle, key, blocks, args, sm, stream, config);
}
