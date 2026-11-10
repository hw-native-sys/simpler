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

#include "platform_comm/ccu/ccu_host.h"
#include <stdexcept>

struct CcuBackendError : std::runtime_error {
    CcuStatus status;
    int32_t native_error;
    CcuBackendError(CcuStatus status, int32_t error, const char *operation) :
        std::runtime_error(operation),
        status(status),
        native_error(error) {}
};

struct CcuRegisteredFunction {
    CcuFunctionSpec spec;
    CcuBuildEntry entry{};
    std::vector<CcuParameterLayout> parameters;
    std::vector<std::string> names;
    uint32_t argument_bytes{};
};
struct alignas(128) CcuImageBlock {
    uint8_t bytes[128];
};

// Ownership of every partially acquired SDK resource stays in this backend.
class CcuSessionBackend {
public:
    virtual ~CcuSessionBackend() = default;
    virtual void Initialize(const CcuSessionConfig &config) = 0;
    virtual uint64_t PrepareMemory(uint64_t addr, uint64_t size) = 0;
    virtual void ReleaseMemory(uint64_t memory_id) = 0;
    virtual bool Idle() = 0;
    virtual void PreventRequests() = 0;
    virtual void Publish(
        const std::vector<CcuRegisteredFunction> &functions, std::vector<CcuImageBlock> image,
        CcuDeviceContextView &view
    ) = 0;
    virtual bool Stop(uint64_t timeout_ms) = 0;
    virtual void Close() = 0;
};

std::unique_ptr<CcuSessionBackend> ccu_make_session_backend();
const char *ccu_session_sdk_build_id();
