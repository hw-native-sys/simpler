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
#include <memory>
#include <string>
#include <vector>

#include "hccl/hccl_comm.h"
#include "platform_comm/ccu/ccu_domain_owner.h"

namespace simpler::ccu {

bool query_window_token(uint64_t address, uint64_t size, uint64_t &token);

struct HcommPreparedMission {
    uint32_t ccu_id{};
    uint32_t kernel_id{};
    uint64_t program_handle{};
    uint64_t trigger_event_handle{};
    uint64_t ready_event_handle{};
    uint64_t done_event_handle{};
    uint64_t trigger_gate_address{};
    uint64_t ready_gate_address{};
    uint64_t done_gate_address{};
    uint64_t stream_handle{};
    uint32_t module_index{};
    uint64_t parameter_size{};
    uint32_t parameter_alignment{};
};

class HcommDomainSession : public DomainResourceBackend {
public:
    virtual uint64_t prepare(const std::vector<HcommPreparedMission> &missions, uint64_t generation) = 0;
    virtual void discard(uint64_t generation) = 0;
    virtual uint64_t begin(
        DomainCcuOwner &owner, const std::vector<HcommPreparedMission> &missions, uint64_t generation,
        uint64_t domain_id
    ) = 0;
    virtual bool stop() = 0;
    virtual void abort() = 0;
};

std::vector<HcommPreparedMission> prepare_hcomm_ccu_callable(
    HcclComm base_communicator, uint64_t domain_id, const std::vector<uint32_t> &rank_ids, uint32_t domain_rank,
    const ExpectedBuildIdentity &expected_identity, const std::vector<std::vector<uint8_t>> &module_bytes,
    const std::string &program_namespace, std::unique_ptr<DomainCcuOwner> &owner, HcommDomainSession *&session
);

}  // namespace simpler::ccu
