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

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "host/ccu_execution.h"
#include "host/ccu_device.h"
#include "host/ccu_request_state.h"
#include "host/ccu_request_validation.h"

struct CcuOwnerLeaseV1 {
    void *owner{};
    CcuOwnerOpsV1 ops{};
    bool retained{};
    bool Retain();
    bool Release();
    ~CcuOwnerLeaseV1();
};

struct CcuWindowLeaseV1 {
    CcuWindowId id{};
    CcuWindowImportV1 import{};
    CcuOwnerLeaseV1 owner;
    uint64_t token{};
};

struct CcuPeerWindowV1 {
    uint64_t allocation_id{};
    uint64_t allocation_generation{};
    uint64_t base_va{};
    uint64_t allocation_size{};
    uint64_t window_offset{};
    uint64_t window_size{};
    int32_t device_id{};
    uint32_t reserved{};
    uint64_t token{};
};

struct CcuRequestEntryV1 {
    CcuRequestId id{};
    CcuRequestDescriptorV1 descriptor{};
    CcuRequestStateV1 state;
    CcuWindowId input_id{};
    CcuWindowId output_id{};
    std::array<uint64_t, 8> task_args{};
    bool launch_attempted{};
    CcuSubmissionV1 submission{CcuSubmissionV1::NOT_SUBMITTED};
    bool trigger_attempted{};
    CcuSubmissionV1 trigger_submission{CcuSubmissionV1::NOT_SUBMITTED};
    uint64_t observed{};
    bool data_ready{};
    bool drained{};
};

struct CcuContextEntryV1 {
    CcuContextId id{};
    uint64_t generation{1};
    CcuCreateOptionsV1 options{};
    std::vector<uint8_t> root_info;
    CcuBootstrapOpsV1 bootstrap{};
    void *bootstrap_owner{};
    CcuOwnerLeaseV1 bootstrap_lease;
    std::mutex mutex;
    bool bound{};
    bool closing{};
    bool retained{};
    bool busy{};
    bool ready{};
    std::shared_ptr<CcuDevice> device;
    std::string trigger_path;
    uint64_t next_sequence{1};
    uint64_t binding_generation{};
    std::unordered_map<CcuWindowId, std::shared_ptr<CcuWindowLeaseV1>> windows;
    CcuWindowId input_id{};
    CcuWindowId output_id{};
    std::array<CcuPeerWindowV1, 2> peer_windows{};
    std::array<CcuPeerWindowV1, 2> peer_inputs{};
    std::array<uint64_t, 2> peer_contexts{}, peer_generations{};
    std::shared_ptr<CcuRequestEntryV1> request;
};

class CcuExecutionRegistryV1 {
public:
    static CcuExecutionRegistryV1 &Instance();
    uint64_t NextId();
    bool AddContext(const std::shared_ptr<CcuContextEntryV1> &context);
    std::shared_ptr<CcuContextEntryV1> GetContext(CcuContextId id);
    std::shared_ptr<CcuRequestEntryV1> GetRequest(CcuRequestId id);
    bool AddRequest(const std::shared_ptr<CcuRequestEntryV1> &request);
    void RemoveRequest(CcuRequestId id);
    void RemoveContext(CcuContextId id);

private:
    std::mutex mutex_;
    uint64_t next_id_{1};
    std::unordered_map<CcuContextId, std::shared_ptr<CcuContextEntryV1>> contexts_;
    std::unordered_map<CcuRequestId, std::shared_ptr<CcuRequestEntryV1>> requests_;
};
