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
#include <cstddef>
#include <cstdint>
#include <string>

#include "platform_comm/ccu/ccu_domain_registration.h"

namespace simpler::ccu {

class HcommDomainRegistrationSdk final : public DomainRegistrationSdk {
public:
    static constexpr size_t kResourceTypeCount = 8;

    HcommDomainRegistrationSdk(uint64_t instance_handle, uint32_t die_id);

    int query_resources(const void *entry, const void *const *arguments, uint32_t argument_count) override;
    int validate_capacity() override;
    int register_start() override;
    int register_kernel(
        const std::string &name, const void *entry, const void *const *arguments, uint32_t argument_count,
        uint64_t *program_handle
    ) override;
    int register_end() override;
    int get_task_argument_count(uint64_t program_handle, uint32_t *argument_count) override;

    const std::array<uint64_t, kResourceTypeCount> &required_resources() const noexcept { return required_resources_; }

private:
    uint64_t instance_handle_{};
    uint32_t die_id_{};
    std::array<uint64_t, kResourceTypeCount> required_resources_{};
};

}  // namespace simpler::ccu
