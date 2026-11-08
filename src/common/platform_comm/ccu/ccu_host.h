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
#include <vector>

#include "platform_comm/ccu/ccu_builder.h"

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

class CcuSession {
public:
    explicit CcuSession(CcuSessionConfig config);
    ~CcuSession();
    CcuSession(const CcuSession &) = delete;
    CcuSession &operator=(const CcuSession &) = delete;
    CcuStatus initialize();
    // The ordered list is registered once per callable/domain session, before publication.
    CcuStatus register_functions(const std::vector<CcuEmbeddedFunction> &functions);
    CcuStatus prepare_memory(uint64_t memory_id, uint64_t addr, uint64_t byte_size);
    CcuStatus get_device_context(CcuDeviceContextView &out) const;
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
