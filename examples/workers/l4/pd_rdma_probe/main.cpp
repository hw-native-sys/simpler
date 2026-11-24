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

#include <acl/acl.h>
#include <arpa/inet.h>
#include <cs/hixl_cs.h>
#include <hcomm/hcomm_res.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
constexpr size_t kAllocation = 12288;
constexpr uint8_t kGuard = 0xCA;
constexpr uint32_t kTimeout = 12000;
constexpr char kTag[] = "pd-probe-existing-hbm";

struct CopyRange {
    size_t src;
    size_t dst;
    size_t bytes;
};

constexpr std::array<CopyRange, 3> kCopies{{{256, 512, 64}, {1024, 2048, 4096}, {8192, 9216, 128}}};

bool report(const char *api, int64_t status) {
    std::cout << "operation=" << api << " status=" << status << std::endl;
    return status == 0;
}

void check(const char *api, int64_t status) {
    if (!report(api, status)) {
        throw std::runtime_error(std::string(api) + " failed with status " + std::to_string(status));
    }
}

int number(const char *arg, int max) {
    size_t consumed = 0;
    const auto parsed = std::stol(arg, &consumed);
    if (arg[consumed] != '\0' || parsed < 0 || parsed > max) {
        throw std::invalid_argument("invalid numeric argument");
    }
    return static_cast<int>(parsed);
}

EndpointDesc roce_endpoint(int physical, const char *addr) {
    EndpointDesc endpoint{};
    check("EndpointDescInit", EndpointDescInit(&endpoint, 1));
    endpoint.protocol = COMM_PROTOCOL_ROCE;
    endpoint.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    if (inet_pton(AF_INET, addr, &endpoint.commAddr.addr) != 1) {
        throw std::invalid_argument("invalid device RoCE IPv4 address");
    }
    endpoint.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    endpoint.loc.device.devPhyId = static_cast<uint32_t>(physical);
    return endpoint;
}

uint8_t pattern(size_t index) { return static_cast<uint8_t>((index * 73 + (index >> 5) * 29) % 251); }

struct Resources {
    bool acl_initialized = false;
    uint8_t *hbm = nullptr;
    HixlServerHandle server = nullptr;
    HixlClientHandle client = nullptr;
    MemHandle memory = nullptr;
    bool access_unknown = false;

    bool close() {
        if (access_unknown) {
            return false;
        }
        if (memory) {
            const auto status = client ? HixlCSClientUnregMem(client, memory) : HixlCSServerUnregMem(server, memory);
            if (!report("HixlCSUnregMem", status)) {
                return false;
            }
            memory = nullptr;
        }
        return (!client || report("HixlCSClientDestroy", HixlCSClientDestroy(client))) &&
               (!server || report("HixlCSServerDestroy", HixlCSServerDestroy(server))) &&
               (!hbm || report("aclrtFree", aclrtFree(hbm))) &&
               (!acl_initialized || report("aclFinalize", aclFinalize()));
    }
};

void prepare_hbm(Resources &resources, bool receiver) {
    void *buffer = nullptr;
    check("aclrtMalloc", aclrtMalloc(&buffer, kAllocation, ACL_MEM_MALLOC_HUGE_FIRST));
    resources.hbm = static_cast<uint8_t *>(buffer);
    std::array<uint8_t, kAllocation> initial{};
    for (size_t index = 0; index < initial.size(); ++index) {
        initial[index] = receiver ? kGuard : pattern(index);
    }
    check(
        "aclrtMemcpy.initialize",
        aclrtMemcpy(buffer, kAllocation, initial.data(), initial.size(), ACL_MEMCPY_HOST_TO_DEVICE)
    );
}

void receive(Resources &resources, const EndpointDesc &local, const char *server_ip, uint32_t port) {
    HixlServerDesc desc{};
    desc.endpoint_list = &local;
    desc.endpoint_list_num = 1;
    desc.server_ip = server_ip;
    desc.server_port = port;
    const HixlServerConfig config{};
    check("HixlCSServerCreate", HixlCSServerCreate(&desc, &config, &resources.server));
    const CommMem mem{COMM_MEM_TYPE_DEVICE, resources.hbm, kAllocation};
    check("HixlCSServerRegMem", HixlCSServerRegMem(resources.server, kTag, &mem, &resources.memory));
    check("HixlCSServerListen", HixlCSServerListen(resources.server, 1));
    resources.access_unknown = true;
    std::cout << "receiver_ready=true transport=device_roce" << std::endl;

    std::string command;
    if (!std::getline(std::cin, command) || (command != "VERIFY" && command != "NO_SUBMISSION")) {
        throw std::runtime_error("client retirement was not confirmed");
    }
    resources.access_unknown = false;
    if (command == "NO_SUBMISSION") {
        throw std::runtime_error("client did not submit data");
    }

    std::array<uint8_t, kAllocation> received{};
    check(
        "aclrtMemcpy.verify",
        aclrtMemcpy(received.data(), received.size(), resources.hbm, kAllocation, ACL_MEMCPY_DEVICE_TO_HOST)
    );
    std::array<uint8_t, kAllocation> expected{};
    expected.fill(kGuard);
    size_t payload = 0;
    for (const auto &copy : kCopies) {
        for (size_t index = 0; index < copy.bytes; ++index) {
            expected[copy.dst + index] = pattern(copy.src + index);
        }
        payload += copy.bytes;
    }
    const auto mismatch = std::mismatch(received.begin(), received.end(), expected.begin());
    if (mismatch.first != received.end()) {
        throw std::runtime_error("receiver mismatch at offset " + std::to_string(mismatch.first - received.begin()));
    }
    std::cout << "device_payload_verified=" << payload << " ranges=" << kCopies.size() << " guards_verified=true"
              << std::endl;
}

void send(
    Resources &resources, const EndpointDesc &local, const EndpointDesc &remote, const char *server_ip, uint32_t port
) {
    HixlClientDesc desc{};
    desc.local_endpoint = &local;
    desc.remote_endpoint = &remote;
    desc.server_ip = server_ip;
    desc.server_port = port;
    const HixlClientConfig config{};
    check("HixlCSClientCreate", HixlCSClientCreate(&desc, &config, &resources.client));
    const CommMem mem{COMM_MEM_TYPE_DEVICE, resources.hbm, kAllocation};
    check("HixlCSClientRegMem", HixlCSClientRegMem(resources.client, kTag, &mem, &resources.memory));
    check("HixlCSClientConnect", HixlCSClientConnect(resources.client, kTimeout));

    CommMem *regions = nullptr;
    char **tags = nullptr;
    uint32_t count = 0;
    check("HixlCSClientGetRemoteMem", HixlCSClientGetRemoteMem(resources.client, &regions, &tags, &count, kTimeout));
    if (!regions || !tags) {
        throw std::runtime_error("peer did not publish registered regions");
    }
    uintptr_t remote_base = 0;
    for (uint32_t index = 0; index < count; ++index) {
        const auto &region = regions[index];
        const bool matched = tags[index] && std::string(tags[index]) == kTag;
        std::cout << "peer_region index=" << index << " tag_matches=" << matched
                  << " type=" << static_cast<int>(region.type) << " bytes=" << region.size << std::endl;
        if (matched) {
            const auto address = reinterpret_cast<uintptr_t>(region.addr);
            if (remote_base || region.type != COMM_MEM_TYPE_DEVICE || region.size != kAllocation || !address ||
                address > UINTPTR_MAX - kAllocation) {
                throw std::runtime_error("remote region does not match the registered device buffer");
            }
            remote_base = address;
        }
    }
    if (!remote_base) {
        throw std::runtime_error("peer did not publish the requested device buffer tag");
    }

    std::array<HixlOneSideOpDesc, kCopies.size()> ops{};
    for (size_t index = 0; index < ops.size(); ++index) {
        const auto &copy = kCopies[index];
        ops[index] = {reinterpret_cast<void *>(remote_base + copy.dst), resources.hbm + copy.src, copy.bytes};
    }
    resources.access_unknown = true;
    std::cout << "payload_attempted=true transport=device_roce" << std::endl;
    check("HixlCSClientBatchPutSync", HixlCSClientBatchPutSync(resources.client, ops.size(), ops.data(), kTimeout));
    resources.access_unknown = false;
}
}  // namespace

int main(int argc, char **argv) {
    if (argc != 9) {
        std::cerr
            << "usage: pd_rdma_probe ROLE DEVICE PHYSICAL LOCAL_NPU_IP PEER_PHYSICAL PEER_NPU_IP SERVER_IP PORT\n";
        return 2;
    }
    Resources resources;
    bool passed = false;
    try {
        const std::string role = argv[1];
        if (role != "prefill" && role != "decode") {
            throw std::invalid_argument("role must be prefill or decode");
        }
        const bool receiver = role == "decode";
        const auto device = number(argv[2], INT32_MAX);
        const auto physical = number(argv[3], INT32_MAX);
        const auto port = static_cast<uint32_t>(number(argv[8], UINT16_MAX));
        const auto local = roce_endpoint(physical, argv[4]);
        check("aclInit", aclInit(nullptr));
        resources.acl_initialized = true;
        int32_t mapped = -1;
        check("aclrtGetPhyDevIdByUserDevId", aclrtGetPhyDevIdByUserDevId(device, &mapped));
        if (physical != mapped || port == 0) {
            throw std::invalid_argument("device mapping or port is invalid");
        }
        check("aclrtSetDevice", aclrtSetDevice(device));
        prepare_hbm(resources, receiver);
        if (receiver) {
            receive(resources, local, argv[7], port);
        } else {
            const auto remote = roce_endpoint(number(argv[5], INT32_MAX), argv[6]);
            send(resources, local, remote, argv[7], port);
        }
        passed = true;
    } catch (const std::exception &error) {
        std::cerr << "probe_error=" << error.what() << std::endl;
    }
    const bool retired = resources.close();
    std::cout << "retirement_confirmed=" << (retired ? "true" : "false") << std::endl;
    return passed && retired ? 0 : 1;
}
