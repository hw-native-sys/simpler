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

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "platform_comm/ccu/ccu_builder.h"

struct CcuTensorParam {
    std::string name;
    CcuArgumentType type{CcuArgumentType::FLOAT32};
};
struct CcuScalarParam {
    std::string name;
    CcuArgumentType type{CcuArgumentType::UINT64};
};
struct CcuSignature {
    std::vector<CcuTensorParam> inputs, outputs, workspaces;
    std::vector<CcuScalarParam> scalars;
    std::vector<std::pair<std::string, std::string>> aliases;
};
struct CcuFunctionSpec {
    std::string name, library_path, entry;
    CcuSignature signature;
    CcuProgramKey program_key{};
};
struct CcuSessionConfig {
    uint64_t allocation_id{}, epoch{};
    int32_t device_id{-1};
    uint32_t rank{}, rank_count{2};
    uint32_t timeout_ms{5000};
    std::vector<uint8_t> root_info;
    // Parent owns the bootstrap transport, device/runtime environment and callbacks.
    std::shared_ptr<void> parent_owner;
    std::function<int(uint64_t, uint32_t, const void *, uint32_t, void *, uint32_t, uint32_t)> exchange;
};

// Key canonicalization includes library bytes, entry, signature and build ABI.
// PR-3's generated Binding and Host metadata consume this same calculation.
CcuStatus ccu_compute_program_key(const CcuFunctionSpec &function, CcuProgramKey &out);
CcuFunctionSpec ccu_builtin_broadcast(uint32_t root);

class CcuSession {
public:
    explicit CcuSession(CcuSessionConfig config);
    ~CcuSession();
    CcuSession(const CcuSession &) = delete;
    CcuSession &operator=(const CcuSession &) = delete;
    CcuStatus initialize();
    // Registration returns an opaque Host transaction reference (context_addr=0).
    // Device references are resolved by key after get_device_context publication.
    CcuStatus register_function(uint64_t registration_id, const CcuFunctionSpec &function, CcuProgramRef &out);
    CcuStatus abort_registration(uint64_t registration_id);
    CcuStatus prepare_memory(uint64_t memory_id, uint64_t addr, uint64_t byte_size);
    CcuStatus release_memory(uint64_t memory_id);
    CcuStatus get_device_context(CcuDeviceContextView &out) const;
    CcuStatus quiesce(uint64_t timeout_ms);
    CcuStatus close(CcuCloseMode mode);

    // PR-3 holds a run reference from before submission until all task readers exit.
    CcuStatus retain_run();
    CcuStatus release_run();
    int32_t native_error() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
