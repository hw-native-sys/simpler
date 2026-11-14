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

#include <acl/acl_base.h>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "pto/comm/async/urma/urma_types.hpp"
#include "pto/comm/async_common/sdma_constants.hpp"

namespace workspace_fixture {
inline int started[3]{};
inline int live[3]{};
inline int fail_engine = -1;
inline int zero_size_engine = -1;
inline int throw_engine = -1;
inline int invalid_addr_engine = -1;
inline int invalid_size_engine = -1;
inline void *last_workspace[3]{};
inline uint32_t device_aiv_count = 2;
inline uint32_t sized_jetties = 0, initialized_jetties = 0, initialized_sq_depth = 0;
inline uint32_t sized_aivs = 0, initialized_aivs = 0;

class Manager {
public:
    explicit Manager(int engine) :
        engine_(engine) {}
    ~Manager() {
        if (addr_) {
            --live[engine_];
        }
    }
    bool create(void *addr, uint64_t size) {
        last_workspace[engine_] = addr;
        ++started[engine_];
        if (!addr || size < GetWorkspaceSize()) return false;
        addr_ = addr;
        ++live[engine_];
        if (throw_engine == engine_) throw std::runtime_error("workspace init failure");
        return fail_engine != engine_;
    }
    void *GetWorkspaceAddr() const { return invalid_addr_engine == engine_ ? nullptr : addr_; }
    uint64_t GetWorkspaceSize() const {
        return invalid_size_engine == engine_ ? 0 : engine_ == 0 ? pto::comm::sdma::kSdmaWorkspaceBytes : 8192;
    }

private:
    int engine_;
    void *addr_ = nullptr;
};
}  // namespace workspace_fixture

namespace pto::comm::sdma {
class SdmaWorkspaceManager : public workspace_fixture::Manager {
public:
    SdmaWorkspaceManager() :
        Manager(0) {}
    using Manager::GetWorkspaceSize;
    static aclError GetWorkspaceSize(uint64_t *size) {
        if (workspace_fixture::zero_size_engine == 0) {
            *size = 0;
            return ACL_ERROR_INVALID_PARAM;
        }
        *size = pto::comm::sdma::kSdmaWorkspaceBytes;
        return ACL_SUCCESS;
    }
    bool Init(void *addr, uint64_t size) { return create(addr, size); }
};
}  // namespace pto::comm::sdma

namespace pto::comm::urma {
class UrmaWorkspaceManager : public workspace_fixture::Manager {
public:
    UrmaWorkspaceManager() :
        Manager(1) {}
    using Manager::GetWorkspaceSize;
    static uint32_t QueryDeviceAivCount() { return workspace_fixture::device_aiv_count; }
    static aclError GetWorkspaceSize(
        uint32_t, uint64_t *size, UrmaLayout layout = UrmaLayout::PER_PEER, uint32_t aivs = kUrmaAutoAivCount,
        uint32_t jetties = 1
    ) {
        workspace_fixture::sized_jetties = jetties;
        workspace_fixture::sized_aivs = aivs;
        if (workspace_fixture::zero_size_engine == 1) {
            *size = 0;
            return ACL_ERROR_INVALID_PARAM;
        }
        *size = 8192;
        return ACL_SUCCESS;
    }
    bool Init(
        void *, uint32_t, uint32_t ranks, void *, uint64_t, void *addr, uint64_t size,
        UrmaLayout layout = UrmaLayout::PER_PEER, uint32_t aivs = kUrmaAutoAivCount, uint32_t jetties = 1,
        uint32_t depth = 0
    ) {
        workspace_fixture::initialized_jetties = jetties;
        workspace_fixture::initialized_sq_depth = depth;
        workspace_fixture::initialized_aivs = aivs;
        const bool ok = create(addr, size);
        auto *info = static_cast<UrmaInfo *>(GetWorkspaceAddr());
        if (info == nullptr) return ok;
        info->rankCount = ranks;
        info->jettiesPerCore = jetties;
        info->rowsPerPeer = (aivs == kUrmaAutoAivCount ? QueryDeviceAivCount() : aivs) * jetties;
        info->jettyCount = ranks * info->rowsPerPeer;
        info->sqPtr = reinterpret_cast<uint64_t>(info + 1);
        info->scqPtr = info->sqPtr + info->jettyCount * sizeof(UrmaWQCtx);
        auto *sq = reinterpret_cast<UrmaWQCtx *>(info->sqPtr);
        auto *cq = reinterpret_cast<UrmaCqCtx *>(info->scqPtr);
        for (uint32_t row = 0; row < info->jettyCount; ++row) {
            sq[row].headAddr = sq[row].tailAddr = reinterpret_cast<uint64_t>(&sq[row].submittedWqeCount);
            cq[row].tailAddr = sq[row].headAddr;
        }
        return ok;
    }
};
}  // namespace pto::comm::urma

namespace pto::comm::rdma {
enum class WorkspaceInitResult { READY, FAILED };
struct WorkspaceConfig {
    uint32_t rankId = 0, rankCount = 0, phyId = 0, qpNum = 1;
    std::string localIp;
    uint16_t basePort = 0;
    void *symmetricAddr = nullptr;
    uint64_t symmetricSize = 0;
    std::vector<std::string> peerIps;
    std::vector<uint32_t> peerPhyIds;
    std::vector<uint64_t> peerSymAddrs;
};
struct RdmaInfo {
    uint64_t sqPtr;
};
struct RdmaMemInfo {
    uint64_t base, size;
};
namespace hns_1825 {
struct RoceSqCtx {
    uint8_t dbCos;
};
struct RoceCqCtx {
    uint64_t value;
};
}  // namespace hns_1825
class RdmaWorkspaceManager : public workspace_fixture::Manager {
public:
    RdmaWorkspaceManager() :
        Manager(2) {}
    using Manager::GetWorkspaceSize;
    static aclError GetWorkspaceSize(uint32_t, uint64_t *size, uint32_t) {
        if (workspace_fixture::zero_size_engine == 2) {
            *size = 0;
            return ACL_ERROR_INVALID_PARAM;
        }
        *size = 8192;
        return ACL_SUCCESS;
    }
    WorkspaceInitResult Init(const WorkspaceConfig &, void *addr, uint64_t size) {
        if (!create(addr, size)) return WorkspaceInitResult::FAILED;
        auto *info = static_cast<RdmaInfo *>(GetWorkspaceAddr());
        info->sqPtr = reinterpret_cast<uint64_t>(info + 1);
        return WorkspaceInitResult::READY;
    }
};
}  // namespace pto::comm::rdma
