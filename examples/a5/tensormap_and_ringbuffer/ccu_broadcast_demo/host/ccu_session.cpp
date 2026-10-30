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
#include "ccu_session.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_ccu_res.h"
#include "hccl/hccl_rank_graph.h"
#include "hccl/hccl_res.h"
#include "hcomm/ccu/ccu_launch.h"
#include "hcomm/ccu/ccu_primitives.hpp"
#include "hcomm/ccu/ccu_res.h"

namespace {

constexpr uint32_t kRanks = 2;
constexpr uint32_t kMaxCount = 1024;
constexpr uint32_t kGuardCount = 32;
constexpr uint32_t kOutputOffset = kMaxCount + 2 * kGuardCount;
constexpr uint32_t kWindowFloats = 2 * kMaxCount + 3 * kGuardCount;
constexpr float kSentinel = -987654.0f;
constexpr uint32_t kTriggered = 0x43435531;
thread_local std::string last_error;

void Check(int rc, const char *operation) {
    if (rc != 0) throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(rc));
}

template <typename Fn>
int Protect(Fn &&fn) noexcept {
    last_error.clear();
    try {
        std::forward<Fn>(fn)();
        return 0;
    } catch (const std::exception &error) {
        last_error = error.what();
    } catch (...) {
        last_error = "Unknown CCU session failure";
    }
    return -1;
}

float InputValue(uint32_t rank, uint32_t iteration, uint32_t index) {
    return static_cast<float>(rank * 10000 + iteration * 1000 + index);
}

CcuResult ChannelDieProbe(const ChannelHandle *channel) { return AscendC::ccu::NotifyWait(*channel, 0); }

uint8_t FindChannelDie(ChannelHandle channel) {
    const void *arguments[] = {&channel};
    for (uint32_t die = 0; die < 2; ++die) {
        HcommCcuResDescHandle descriptor = 0;
        Check(HcommCcuInsResDescCreate(die, &descriptor), "HcommCcuInsResDescCreate");
        // Query validates channel placement without registering or launching a task.
        const CcuResult query =
            HcommCcuKernelQueryResReq(reinterpret_cast<const void *>(ChannelDieProbe), arguments, 1, descriptor);
        Check(HcommCcuInsResDescDestroy(descriptor), "HcommCcuInsResDescDestroy");
        if (query == CCU_SUCCESS) return static_cast<uint8_t>(die);
        if (query != CCU_E_PARA && query != CCU_E_UNAVAIL) Check(query, "HcommCcuKernelQueryResReq(channel die)");
    }
    throw std::runtime_error("CCU channel has no enabled matching IO die");
}

struct BroadcastArgs {
    ChannelHandle channel;
    CcuEventHandle gate;
    bool is_root;
};

// This Host function records a CCU instruction program during registration.
CcuResult BroadcastProgram(const BroadcastArgs *args) {
    namespace ccu = AscendC::ccu;
    try {
        ccu::Variable input, output, token, peer_output, peer_token, length;
        const std::array<ccu::Variable, 6> variables{input, output, token, peer_output, peer_token, length};
        for (uint32_t i = 0; i < variables.size(); ++i)
            Check(ccu::LoadArg(variables[i], i), "ccu::LoadArg");
        ccu::Event gate(args->gate);
        Check(ccu::EventWait(gate), "ccu::EventWait(AIV gate)");
        Check(ccu::NotifyRecord(args->channel, 0, 1), "ccu::NotifyRecord(ready)");
        Check(ccu::NotifyWait(args->channel, 0, 1), "ccu::NotifyWait(ready)");
        if (args->is_root) {
            ccu::LocalAddr source, destination;
            ccu::RemoteAddr remote;
            source.addr = input;
            source.token = token;
            destination.addr = output;
            destination.token = token;
            remote.addr = peer_output;
            remote.token = peer_token;
            ccu::Event copied;
            ccu::CcuBuffer buffer;
            Check(ccu::LocalCopy(buffer, source, length, copied), "ccu::LocalCopy(HBM to CCU buffer)");
            Check(ccu::EventWait(copied), "ccu::EventWait(local read)");
            Check(ccu::LocalCopy(destination, buffer, length, copied), "ccu::LocalCopy(CCU buffer to HBM)");
            Check(ccu::EventWait(copied), "ccu::EventWait(local write)");
            Check(ccu::Write(args->channel, remote, source, length, copied), "ccu::Write(peer HBM)");
            Check(ccu::EventWait(copied), "ccu::EventWait(remote write)");
        }
        Check(ccu::NotifyRecord(args->channel, 0, 2), "ccu::NotifyRecord(done)");
        Check(ccu::NotifyWait(args->channel, 0, 2), "ccu::NotifyWait(done)");
        return CCU_SUCCESS;
    } catch (const std::exception &error) {
        last_error = error.what();
        return CCU_E_INTERNAL;
    }
}

struct Session {
    int device = -1;
    uint32_t rank = 0;
    uint32_t count = 0;
    uint32_t iteration = 0;
    uint32_t root = 0;
    HcclComm comm = nullptr;
    aclrtStream stream = nullptr;
    ThreadHandle thread = 0;
    CcuInsHandle instance = 0;
    CcuEventHandle gate = 0;
    void *window = nullptr;
    void *trigger_status = nullptr;
    CcuPeerInfo peer{};
    std::vector<ChannelHandle> channels;
    bool launched = false;
    bool drained = false;

    int Close() {
        if (launched && !drained) throw std::runtime_error("CCU session has no proven completion; owners retained");
        Check(aclrtSetDevice(device), "aclrtSetDevice(close)");
        if (comm != nullptr) {
            Check(HcclCommDestroy(comm), "HcclCommDestroy");
            comm = nullptr;
        }
        if (stream != nullptr) {
            Check(aclrtDestroyStream(stream), "aclrtDestroyStream");
            stream = nullptr;
        }
        if (trigger_status != nullptr) {
            Check(aclrtFree(trigger_status), "aclrtFree(trigger_status)");
            trigger_status = nullptr;
        }
        if (window != nullptr) {
            Check(aclrtFree(window), "aclrtFree(window)");
            window = nullptr;
        }
        return 0;
    }
};

Session &Get(void *handle) {
    if (handle == nullptr) throw std::invalid_argument("CCU session is null");
    return *static_cast<Session *>(handle);
}

void AcquireChannels(Session &session) {
    std::vector<HcclChannelDesc> requests;
    const uint32_t remote = 1 - session.rank;
    CommLink *links = nullptr;
    uint32_t link_count = 0;
    Check(HcclRankGraphGetLinks(session.comm, 0, session.rank, remote, &links, &link_count), "HcclRankGraphGetLinks");
    for (uint32_t i = 0; i < link_count; ++i) {
        if (links[i].linkAttr.linkProtocol != COMM_PROTOCOL_UBC_CTP) continue;
        HcclChannelDesc desc;
        Check(HcclChannelDescInit(&desc, 1), "HcclChannelDescInit");
        desc.remoteRank = remote;
        desc.notifyNum = 4;
        desc.channelProtocol = links[i].linkAttr.linkProtocol;
        desc.localEndpoint = links[i].srcEndpointDesc;
        desc.remoteEndpoint = links[i].dstEndpointDesc;
        requests.push_back(desc);
        break;
    }
    if (requests.empty()) throw std::runtime_error("No UBC_CTP link to peer; CCU Broadcast requires UB connectivity");
    session.channels.resize(requests.size());
    Check(
        HcclChannelAcquire(session.comm, COMM_ENGINE_CCU, requests.data(), requests.size(), session.channels.data()),
        "HcclChannelAcquire(CCU)"
    );
}

}  // namespace

extern "C" const char *ccu_demo_last_error() { return last_error.c_str(); }
extern "C" size_t ccu_demo_root_info_size() { return sizeof(HcclRootInfo); }

extern "C" int ccu_demo_get_root_info(int device, void *root_info, size_t size) {
    return Protect([&] {
        if (root_info == nullptr || size != sizeof(HcclRootInfo)) throw std::invalid_argument("Root-info buffer size");
        Check(aclrtSetDevice(device), "aclrtSetDevice(root)");
        Check(HcclGetRootInfo(static_cast<HcclRootInfo *>(root_info)), "HcclGetRootInfo");
    });
}

extern "C" int ccu_demo_open(
    int device, uint32_t rank, const void *root_info, size_t root_info_size, uint32_t count, uint32_t iteration,
    void **handle, CcuPeerInfo *peer
) {
    return Protect([&] {
        if (root_info == nullptr || root_info_size != sizeof(HcclRootInfo) || handle == nullptr || *handle != nullptr ||
            peer == nullptr || rank >= kRanks || count == 0 || count > kMaxCount || iteration > 10000) {
            throw std::invalid_argument("Invalid CCU session arguments");
        }
        *handle = nullptr;
        auto session = std::make_unique<Session>();
        session->device = device;
        session->rank = rank;
        session->count = count;
        session->iteration = iteration;
        Check(aclrtSetDevice(device), "aclrtSetDevice(open)");
        try {
            HcclRootInfo root{};
            std::memcpy(&root, root_info, sizeof(root));
            Check(HcclCommInitRootInfo(kRanks, &root, rank, &session->comm), "HcclCommInitRootInfo");
            Check(aclrtCreateStream(&session->stream), "aclrtCreateStream(CCU)");
            Check(
                HcclThreadAcquireWithStream(session->comm, COMM_ENGINE_CCU, session->stream, 1, &session->thread),
                "HcclThreadAcquireWithStream(CCU)"
            );
            AcquireChannels(*session);

            aclrtMallocAttrValue value{};
            value.moduleId = 3;
            aclrtMallocAttribute attribute{ACL_RT_MEM_ATTR_MODULE_ID, value};
            aclrtMallocConfig config{&attribute, 1};
            const size_t bytes = kWindowFloats * sizeof(float);
            Check(
                aclrtMallocWithCfg(&session->window, bytes, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &config),
                "aclrtMallocWithCfg(CCU window)"
            );
            Check(
                aclrtMalloc(&session->trigger_status, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST),
                "aclrtMalloc(trigger_status)"
            );
            std::vector<float> initial(kWindowFloats, kSentinel);
            for (uint32_t i = 0; i < count; ++i)
                initial[kGuardCount + i] = InputValue(rank, iteration, i);
            Check(
                aclrtMemcpy(session->window, bytes, initial.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE),
                "aclrtMemcpy(window H2D)"
            );
            const uint32_t status = 0;
            Check(
                aclrtMemcpy(
                    session->trigger_status, sizeof(status), &status, sizeof(status), ACL_MEMCPY_HOST_TO_DEVICE
                ),
                "aclrtMemcpy(status H2D)"
            );
            const auto base = reinterpret_cast<uint64_t>(session->window);
            session->peer.input = base + kGuardCount * sizeof(float);
            session->peer.output = base + kOutputOffset * sizeof(float);
            Check(HcommCcuGetMemToken(base, bytes, &session->peer.token), "HcommCcuGetMemToken");
            session->peer.trigger_status = reinterpret_cast<uint64_t>(session->trigger_status);
            *peer = session->peer;
            *handle = session.release();
        } catch (...) {
            session->Close();
            throw;
        }
    });
}

extern "C" int ccu_demo_prepare(
    void *handle, const CcuPeerInfo *peers, size_t peer_count, uint32_t root, uint64_t *gate_va, uint32_t *gate_mask
) {
    return Protect([&] {
        auto &session = Get(handle);
        if (peers == nullptr || peer_count != kRanks || root >= kRanks || gate_va == nullptr || gate_mask == nullptr ||
            session.launched) {
            throw std::invalid_argument("Invalid CCU prepare arguments");
        }
        std::array<uint64_t, kRanks> inputs{}, outputs{}, tokens{};
        for (uint32_t i = 0; i < kRanks; ++i) {
            if (peers[i].input == 0 || peers[i].output == 0) throw std::invalid_argument("Missing peer window");
            inputs[i] = peers[i].input;
            outputs[i] = peers[i].output;
            tokens[i] = peers[i].token;
        }
        if (peers[session.rank].input != session.peer.input || peers[session.rank].output != session.peer.output ||
            peers[session.rank].token != session.peer.token) {
            throw std::invalid_argument("Peer metadata does not match this rank");
        }
        uint32_t instance_count = 0;
        Check(HcclCommQueryCcuIns(session.comm, &session.instance, &instance_count), "HcclCommQueryCcuIns");
        if (instance_count != 1) throw std::runtime_error("Expected one communicator-owned CCU instance");
        const uint8_t die = FindChannelDie(session.channels.at(0));
        Check(HcommCcuEventAlloc(session.instance, die, 1, &session.gate), "HcommCcuEventAlloc(AIV gate)");
        std::fprintf(stderr, "CCU_SETUP rank=%u io_die=%u protocol=UBC_CTP\n", session.rank, die);
        Check(HcommCcuEventGetAddr(session.gate, 0, gate_va), "HcommCcuEventGetAddr(AIV gate)");
        if (*gate_va == 0) throw std::runtime_error("HCOMM returned an empty CKE mapping");
        *gate_mask = 1;
        const BroadcastArgs program{session.channels.at(0), session.gate, session.rank == root};
        const void *program_args[] = {&program};
        CcuKernelHandle kernel = 0;
        Check(HcommCcuKernelRegisterStart(session.instance), "HcommCcuKernelRegisterStart");
        Check(
            HcommCcuKernelRegister(
                session.instance, die, "SimplerAivGatedBroadcast", reinterpret_cast<const void *>(BroadcastProgram),
                program_args, 1, &kernel
            ),
            "HcommCcuKernelRegister"
        );
        Check(HcommCcuKernelRegisterEnd(session.instance), "HcommCcuKernelRegisterEnd");
        const uint32_t remote = 1 - session.rank;
        const std::array<uint64_t, 6> task{inputs[session.rank], outputs[session.rank], tokens[session.rank],
                                           outputs[remote],      tokens[remote],        session.count * sizeof(float)};
        uint32_t task_count = 0;
        Check(HcommCcuGetTaskArgsNum(kernel, &task_count), "HcommCcuGetTaskArgsNum");
        if (task_count != task.size()) throw std::runtime_error("Registered CCU task argument count differs");
        session.root = root;
        session.launched = true;
        Check(HcommCcuKernelLaunch(session.thread, kernel, task.data(), task.size()), "HcommCcuKernelLaunch");
    });
}

extern "C" int ccu_demo_wait_and_verify(void *handle, uint32_t root, int timeout_ms) {
    return Protect([&] {
        auto &session = Get(handle);
        if (!session.launched || root != session.root || timeout_ms <= 0) {
            throw std::invalid_argument("Invalid CCU completion arguments");
        }
        uint32_t status = 0;
        Check(
            aclrtMemcpy(&status, sizeof(status), session.trigger_status, sizeof(status), ACL_MEMCPY_DEVICE_TO_HOST),
            "aclrtMemcpy(trigger proof)"
        );
        if (status != kTriggered) throw std::runtime_error("simpler AIV trigger did not publish its completion marker");
        Check(aclrtSynchronizeStreamWithTimeout(session.stream, timeout_ms), "CCU stream completion");
        session.drained = true;
        std::vector<float> actual(kWindowFloats);
        Check(
            aclrtMemcpy(
                actual.data(), actual.size() * sizeof(float), session.window, actual.size() * sizeof(float),
                ACL_MEMCPY_DEVICE_TO_HOST
            ),
            "aclrtMemcpy(result D2H)"
        );
        for (uint32_t i = 0; i < kWindowFloats; ++i) {
            float expected = kSentinel;
            if (i >= kGuardCount && i < kGuardCount + session.count) {
                expected = InputValue(session.rank, session.iteration, i - kGuardCount);
            } else if (i >= kOutputOffset && i < kOutputOffset + session.count) {
                expected = InputValue(root, session.iteration, i - kOutputOffset);
            }
            if (actual[i] != expected) {
                throw std::runtime_error(
                    "Broadcast result/guard mismatch at float " + std::to_string(i) + ": expected " +
                    std::to_string(expected) + ", got " + std::to_string(actual[i])
                );
            }
        }
    });
}

extern "C" int ccu_demo_close(void *handle) {
    return Protect([&] {
        if (handle == nullptr) return;
        auto *session = static_cast<Session *>(handle);
        session->Close();
        delete session;
    });
}
