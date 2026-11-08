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
    std::vector<uint8_t> module_binary;
    uint64_t positional_id{};  // Zero selects the legacy named-parameter ABI; otherwise ccu_id + 1.
};
enum class CcuEmbeddedModuleKind : uint32_t { HOST_BUILDER = 1 };
struct CcuEmbeddedFunction {
    std::vector<uint8_t> binary;
    std::string entry;
    CcuEmbeddedModuleKind kind{CcuEmbeddedModuleKind::HOST_BUILDER};
    uint32_t abi_version{CCU_POSITIONAL_ABI_VERSION};
};
struct CcuSessionConfig {
    uint64_t allocation_id{}, epoch{};
    uint64_t domain_owner{};
    int32_t callable_id{-1};
    int32_t device_id{-1};
    uint32_t rank{}, rank_count{2};
    uint32_t timeout_ms{5000};
    std::vector<uint8_t> root_info;
    // Parent owns the bootstrap transport, device/runtime environment and callbacks.
    std::shared_ptr<void> parent_owner;
    std::function<int(uint64_t, uint32_t, const void *, uint32_t, void *, uint32_t, uint32_t)> exchange;
};

// Legacy named-parameter registration canonicalizes library bytes, entry,
// signature and build ABI. Embedded callable definitions use ordered IDs instead.
CcuStatus ccu_compute_program_key(const CcuFunctionSpec &function, CcuProgramKey &out);
CcuFunctionSpec ccu_builtin_broadcast(uint32_t root);

class CcuSession {
public:
    explicit CcuSession(CcuSessionConfig config);
    ~CcuSession();
    CcuSession(const CcuSession &) = delete;
    CcuSession &operator=(const CcuSession &) = delete;
    CcuStatus initialize();
    // Legacy registration returns an opaque Host transaction reference (context_addr=0).
    // Its device references are resolved by key after get_device_context publication.
    CcuStatus register_function(uint64_t registration_id, const CcuFunctionSpec &function, CcuProgramRef &out);
    // The ordered list is registered once per callable/domain session, before publication.
    CcuStatus register_functions(const std::vector<CcuEmbeddedFunction> &functions);
    CcuStatus abort_registration(uint64_t registration_id);
    CcuStatus prepare_memory(uint64_t memory_id, uint64_t addr, uint64_t byte_size);
    CcuStatus release_memory(uint64_t memory_id);
    CcuStatus get_device_context(CcuDeviceContextView &out) const;
    CcuStatus quiesce(uint64_t timeout_ms);
    CcuStatus close(CcuCloseMode mode);

    // The callable provider retains a run before submission until all task readers exit.
    CcuStatus retain_run();
    CcuStatus release_run();
    CcuPollResult poll_idle();
    // Call only after ordinary task readers have stopped. Successful drain permits reuse.
    CcuStatus drain(uint64_t timeout_ms);
    int32_t native_error() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
