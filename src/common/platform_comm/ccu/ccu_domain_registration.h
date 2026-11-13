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
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include "ccu_native_module.h"

namespace simpler::ccu {

struct MissionRegistrationResources {
    uint64_t ready_event_handle{};
    uint64_t trigger_event_handle{};
    uint64_t done_event_handle{};
};

struct DomainRegistrationRequest {
    uint64_t domain_id{};
    uint32_t rank_id{};
    uint32_t rank_count{};
    std::array<uint64_t, SIMPLER_CCU_MAX_RANKS> channels{};
    std::string program_namespace;
    std::vector<MissionRegistrationResources> missions;
};

struct RegisteredKernel {
    uint32_t kernel_id{};
    uint32_t ccu_id{};
    uint64_t program_handle{};
};

class DomainRegistrationSdk {
public:
    virtual ~DomainRegistrationSdk() = default;

    virtual int query_resources(const void *entry, const void *const *arguments, uint32_t argument_count) = 0;
    virtual int validate_capacity() = 0;
    virtual int register_start() = 0;
    virtual int register_kernel(
        const std::string &name, const void *entry, const void *const *arguments, uint32_t argument_count,
        uint64_t *program_handle
    ) = 0;
    virtual int register_end() = 0;
    virtual int get_task_argument_count(uint64_t program_handle, uint32_t *argument_count) = 0;
};

class DomainRegistrationError final : public std::runtime_error {
public:
    DomainRegistrationError(std::string operation, int native_error, int cleanup_error = 0);

    const std::string &operation() const noexcept { return operation_; }
    int native_error() const noexcept { return native_error_; }
    int cleanup_error() const noexcept { return cleanup_error_; }

private:
    std::string operation_;
    int native_error_{};
    int cleanup_error_{};
};

std::vector<RegisteredKernel> register_domain_kernels(
    const std::vector<NativeKernelDefinition> &kernels, const DomainRegistrationRequest &request,
    DomainRegistrationSdk &sdk
);

}  // namespace simpler::ccu
