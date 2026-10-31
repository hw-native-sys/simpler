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

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace workspace_fixture {
inline int started[3]{};
inline int live[3]{};
inline int fail_engine = -1;
inline int throw_engine = -1;

class Manager {
public:
    explicit Manager(int engine) :
        engine_(engine) {}
    ~Manager() {
        if (addr_) {
            --live[engine_];
            std::free(addr_);
        }
    }
    bool create() {
        ++started[engine_];
        addr_ = std::calloc(1, GetWorkspaceSize());
        if (!addr_) throw std::bad_alloc();
        ++live[engine_];
        if (throw_engine == engine_) throw std::runtime_error("workspace init failure");
        return fail_engine != engine_;
    }
    void *GetWorkspaceAddr() const { return addr_; }
    uint64_t GetWorkspaceSize() const { return engine_ == 0 ? 16384 : 8192; }

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
    bool Init() { return create(); }
};
}  // namespace pto::comm::sdma

namespace pto::comm::urma {
class UrmaWorkspaceManager : public workspace_fixture::Manager {
public:
    UrmaWorkspaceManager() :
        Manager(1) {}
    template <typename... Args>
    bool Init(Args...) {
        return create();
    }
};
}  // namespace pto::comm::urma

namespace pto::comm::rdma {
enum class WorkspaceInitResult { READY, FAILED };
struct WorkspaceConfig {
    uint32_t rankId = 0, rankCount = 0, phyId = 0;
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
    WorkspaceInitResult Init(const WorkspaceConfig &) {
        if (!create()) return WorkspaceInitResult::FAILED;
        auto *info = static_cast<RdmaInfo *>(GetWorkspaceAddr());
        info->sqPtr = reinterpret_cast<uint64_t>(info + 1);
        return WorkspaceInitResult::READY;
    }
};
}  // namespace pto::comm::rdma
