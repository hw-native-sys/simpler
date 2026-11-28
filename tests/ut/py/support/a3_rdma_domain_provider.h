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
#pragma once

#include <vector>

#include "acl/acl.h"
#include "communication/comm_device_context.hpp"
#include "communication/comm_domain_types.hpp"
#include "pto/comm/async/rdma/backends/in_die/in_die_types.hpp"
#include "pto/comm/async/rdma/rdma_workspace_types.hpp"

namespace rdma_fixture {
inline bool fail_initialization = false;
inline bool fail_cleanup = false;
inline int providers = 0;
}  // namespace rdma_fixture

namespace pto::comm::domain {

struct CommContext {
    RdmaInitReport rdmaReport{};
    CommDeviceContext rdmaHostCtx{};
    bool provider_owned = false;
};

inline bool TryDestroyComm(CommContext &context) {
    if (context.provider_owned && rdma_fixture::fail_cleanup) return false;
    if (context.provider_owned) --rdma_fixture::providers;
    context.provider_owned = false;
    return true;
}

inline aclError GetCommWorkspaceSize(const CommConfig &config, CommWorkspaceSizes *sizes) {
    sizes->rdma = 65536;
    sizes->rdmaSymmetric = config.symBytes;
    sizes->rdmaDeviceContext = sizeof(CommDeviceContext);
    return ACL_SUCCESS;
}

inline bool BuildComm(
    const CommConfig &config, const CommWorkspaceAddresses &addresses, const CommWorkspaceSizes &, CommContext &context
) {
    ++rdma_fixture::providers;
    context.provider_owned = true;
    if (rdma_fixture::fail_initialization) {
        const bool cleaned = TryDestroyComm(context);
        context.rdmaReport = {cleaned ? RdmaInitStatus::LinkFailed : RdmaInitStatus::CleanupFailed, config.rankId};
        return false;
    }
    const uint64_t local = reinterpret_cast<uint64_t>(addresses.rdmaSymmetric);
    std::vector<uint64_t> peers(config.rankNum);
    if (!config.exchange.allGather(config.exchange.context, &local, sizeof(local), peers.data())) return false;
    auto &header = *static_cast<rdma::RdmaInfo *>(addresses.rdma);
    header = {};
    header.magic = rdma::kRdmaWorkspaceMagic;
    header.version = rdma::kRdmaWorkspaceVersion;
    header.backend = RdmaBackend::IN_DIE;
    header.rankCount = config.rankNum;
    header.qpNum = config.rdma.qpNum;
    header.sqPtr = reinterpret_cast<uint64_t>(addresses.rdma) + sizeof(header);
    header.scqPtr = header.sqPtr;
    header.memPtr = header.sqPtr + uint64_t(config.rankNum) * config.rdma.qpNum * sizeof(rdma::in_die::Queue);
    auto &host = context.rdmaHostCtx;
    host.rankId = config.rankId;
    host.rankNum = config.rankNum;
    host.winSize = config.symBytes;
    for (int rank = 0; rank < config.rankNum; ++rank)
        host.windowsIn[rank] = host.windowsOut[rank] = peers[rank];
    context.rdmaReport = {RdmaInitStatus::Ready, -1};
    return true;
}

}  // namespace pto::comm::domain
