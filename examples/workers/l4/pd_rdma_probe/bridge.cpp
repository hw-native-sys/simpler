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

#include <arpa/inet.h>
#include <cs/hixl_cs.h>
#include <hcomm/hcomm_res.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

namespace {
constexpr uint32_t kTimeoutMs = 12000;
constexpr uint32_t kListenBacklog = 16;
static_assert(sizeof(uintptr_t) == sizeof(uint64_t));
static_assert(sizeof(size_t) == sizeof(uint64_t));

int32_t roce_endpoint(const char *ip, int32_t physical, EndpointDesc &endpoint) {
    if (!ip || physical < 0) {
        return HIXL_PARAM_INVALID;
    }
    const auto status = EndpointDescInit(&endpoint, 1);
    if (status != 0) {
        return status;
    }
    endpoint.protocol = COMM_PROTOCOL_ROCE;
    endpoint.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    if (inet_pton(AF_INET, ip, &endpoint.commAddr.addr) != 1) {
        return HIXL_PARAM_INVALID;
    }
    endpoint.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    endpoint.loc.device.devPhyId = static_cast<uint32_t>(physical);
    return HIXL_SUCCESS;
}

bool valid_range(uint64_t address, uint64_t extent) {
    return address != 0 && extent != 0 && extent <= UINTPTR_MAX - address;
}

int32_t register_region(
    decltype(&HixlCSServerRegMem) register_mem, void *handle, const char *tag, uint64_t address, uint64_t extent,
    void **mr
) {
    if (!mr) {
        return HIXL_PARAM_INVALID;
    }
    *mr = nullptr;
    if (!handle || !tag || !tag[0] || !valid_range(address, extent)) {
        return HIXL_PARAM_INVALID;
    }
    const CommMem memory{COMM_MEM_TYPE_DEVICE, reinterpret_cast<void *>(address), static_cast<size_t>(extent)};
    return register_mem(handle, tag, &memory, mr);
}
}  // namespace

// The caller binds the allocation owner's ACL context and retains every registered buffer.
extern "C" {

int32_t hixl_pd_server_create(const char *nic_ip, int32_t physical, const char *host, int32_t port, void **handle) {
    if (!handle) {
        return HIXL_PARAM_INVALID;
    }
    *handle = nullptr;
    if (!host || !host[0] || port <= 0 || port > UINT16_MAX) {
        return HIXL_PARAM_INVALID;
    }
    EndpointDesc endpoint{};
    const auto status = roce_endpoint(nic_ip, physical, endpoint);
    if (status != HIXL_SUCCESS) {
        return status;
    }
    HixlServerDesc desc{};
    desc.endpoint_list = &endpoint;
    desc.endpoint_list_num = 1;
    desc.server_ip = host;
    desc.server_port = static_cast<uint32_t>(port);
    const HixlServerConfig config{};
    return HixlCSServerCreate(&desc, &config, handle);
}

int32_t hixl_pd_server_register(void *handle, const char *tag, uint64_t address, uint64_t extent, void **mr) {
    return register_region(HixlCSServerRegMem, handle, tag, address, extent, mr);
}

int32_t hixl_pd_listen(void *handle) {
    return handle ? HixlCSServerListen(handle, kListenBacklog) : HIXL_PARAM_INVALID;
}

int32_t hixl_pd_client_create(
    const char *local_ip, int32_t local_phy, const char *peer_ip, int32_t peer_phy, const char *host, int32_t port,
    void **handle
) {
    if (!handle) {
        return HIXL_PARAM_INVALID;
    }
    *handle = nullptr;
    if (!host || !host[0] || port <= 0 || port > UINT16_MAX) {
        return HIXL_PARAM_INVALID;
    }
    EndpointDesc local{};
    auto status = roce_endpoint(local_ip, local_phy, local);
    if (status != HIXL_SUCCESS) {
        return status;
    }
    EndpointDesc peer{};
    status = roce_endpoint(peer_ip, peer_phy, peer);
    if (status != HIXL_SUCCESS) {
        return status;
    }
    HixlClientDesc desc{};
    desc.local_endpoint = &local;
    desc.remote_endpoint = &peer;
    desc.server_ip = host;
    desc.server_port = static_cast<uint32_t>(port);
    const HixlClientConfig config{};
    return HixlCSClientCreate(&desc, &config, handle);
}

int32_t hixl_pd_client_register(void *handle, const char *tag, uint64_t address, uint64_t extent, void **mr) {
    return register_region(HixlCSClientRegMem, handle, tag, address, extent, mr);
}

int32_t hixl_pd_connect(void *handle) { return handle ? HixlCSClientConnect(handle, kTimeoutMs) : HIXL_PARAM_INVALID; }

int32_t hixl_pd_remote_region(void *handle, const char *tag, uint64_t *address, uint64_t *extent) {
    if (!address || !extent) {
        return HIXL_PARAM_INVALID;
    }
    *address = 0;
    *extent = 0;
    if (!handle || !tag || !tag[0]) {
        return HIXL_PARAM_INVALID;
    }

    CommMem *regions = nullptr;
    char **tags = nullptr;
    uint32_t count = 0;
    const auto status = HixlCSClientGetRemoteMem(handle, &regions, &tags, &count, kTimeoutMs);
    if (status != HIXL_SUCCESS) {
        return status;
    }
    if (!regions || !tags) {
        return HIXL_FAILED;
    }
    const CommMem *match = nullptr;
    for (uint32_t index = 0; index < count; ++index) {
        if (tags[index] && std::strcmp(tags[index], tag) == 0) {
            if (match) {
                return HIXL_FAILED;
            }
            match = &regions[index];
        }
    }
    if (!match || match->type != COMM_MEM_TYPE_DEVICE) {
        return HIXL_FAILED;
    }
    const auto base = reinterpret_cast<uintptr_t>(match->addr);
    if (!valid_range(base, match->size)) {
        return HIXL_FAILED;
    }
    *address = base;
    *extent = match->size;
    return HIXL_SUCCESS;
}

int32_t
hixl_pd_write(void *handle, const uint64_t *source, const uint64_t *dest, const uint64_t *length, uint32_t count) {
    if (!handle || !source || !dest || !length || !count) {
        return HIXL_PARAM_INVALID;
    }
    try {
        std::vector<HixlOneSideOpDesc> operations;
        operations.reserve(count);
        for (uint32_t index = 0; index < count; ++index) {
            if (!valid_range(source[index], length[index]) || !valid_range(dest[index], length[index])) {
                return HIXL_PARAM_INVALID;
            }
            operations.push_back({
                reinterpret_cast<void *>(dest[index]),
                reinterpret_cast<void *>(source[index]),
                static_cast<size_t>(length[index]),
            });
        }
        return HixlCSClientBatchPutSync(handle, count, operations.data(), kTimeoutMs);
    } catch (const std::bad_alloc &) {
        return HIXL_FAILED;
    }
}

int32_t hixl_pd_server_unregister(void *handle, void *mr) {
    return handle && mr ? HixlCSServerUnregMem(handle, mr) : HIXL_PARAM_INVALID;
}

int32_t hixl_pd_client_unregister(void *handle, void *mr) {
    return handle && mr ? HixlCSClientUnregMem(handle, mr) : HIXL_PARAM_INVALID;
}

int32_t hixl_pd_server_destroy(void *handle) { return handle ? HixlCSServerDestroy(handle) : HIXL_PARAM_INVALID; }

int32_t hixl_pd_client_destroy(void *handle) { return handle ? HixlCSClientDestroy(handle) : HIXL_PARAM_INVALID; }

}  // extern "C"
