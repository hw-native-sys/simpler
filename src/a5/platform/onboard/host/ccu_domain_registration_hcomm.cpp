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

#include "ccu_domain_registration_hcomm.h"

#include <limits>

#include "hcomm/ccu/ccu_launch.h"
#include "hcomm/ccu/ccu_primitives.hpp"
#include "hcomm/ccu/ccu_res.h"

namespace simpler::ccu {
namespace {

struct ReadyDefinition {
    SimplerCcuNativeKernelEntry entry;
    const SimplerCcuRegistrationBinding *binding;
};

CcuResult ready_definition(const ReadyDefinition *definition) {
    namespace ccu = AscendC::ccu;
    const auto &binding = *definition->binding;
    ccu::Event ready(binding.ready_event_handle);
    const CcuResult rc = ccu::EventRecord(ready);
    if (rc != CCU_SUCCESS) return rc;
    return static_cast<CcuResult>(definition->entry(definition->binding));
}

int finish_descriptor(HcommCcuResDescHandle descriptor, int primary_error) {
    const int cleanup_error = HcommCcuInsResDescDestroy(descriptor);
    return primary_error != CCU_SUCCESS ? primary_error : cleanup_error;
}

}  // namespace

HcommDomainRegistrationSdk::HcommDomainRegistrationSdk(uint64_t instance_handle, uint32_t die_id) :
    instance_handle_(instance_handle),
    die_id_(die_id) {}

int HcommDomainRegistrationSdk::query_resources(
    const void *entry, const void *const *arguments, uint32_t argument_count
) {
    HcommCcuResDescHandle descriptor{};
    int rc = HcommCcuInsResDescCreate(die_id_, &descriptor);
    if (rc != CCU_SUCCESS) return rc;

    if (argument_count != 1 || arguments == nullptr || arguments[0] == nullptr)
        return finish_descriptor(descriptor, CCU_E_PARA);
    const ReadyDefinition definition{
        reinterpret_cast<SimplerCcuNativeKernelEntry>(const_cast<void *>(entry)),
        static_cast<const SimplerCcuRegistrationBinding *>(arguments[0])
    };
    const void *ready_arguments[]{&definition};
    rc = HcommCcuKernelQueryResReq(reinterpret_cast<const void *>(ready_definition), ready_arguments, 1, descriptor);
    for (size_t i = 0; rc == CCU_SUCCESS && i < required_resources_.size(); ++i) {
        uint32_t count{};
        rc = HcommCcuInsResDescQueryNum(descriptor, static_cast<HcommCcuResType>(i), &count);
        if (rc == CCU_SUCCESS) {
            if (required_resources_[i] > std::numeric_limits<uint64_t>::max() - count) {
                rc = CCU_E_PARA;
            } else {
                required_resources_[i] += count;
            }
        }
    }
    return finish_descriptor(descriptor, rc);
}

int HcommDomainRegistrationSdk::validate_capacity() {
    HcommCcuResDescHandle instance_descriptor{};
    int rc = HcommCcuInsResDescCreate(die_id_, &instance_descriptor);
    if (rc != CCU_SUCCESS) return rc;

    HcommCcuResDescHandle remaining_descriptor{};
    rc = HcommCcuInsResDescCreate(die_id_, &remaining_descriptor);
    if (rc != CCU_SUCCESS) return finish_descriptor(instance_descriptor, rc);

    rc = HcommCcuInsQueryResDesc(instance_handle_, instance_descriptor);
    if (rc == CCU_SUCCESS) rc = HcommCcuQueryRemainResDesc(remaining_descriptor);
    for (size_t i = 0; rc == CCU_SUCCESS && i < required_resources_.size(); ++i) {
        const auto type = static_cast<HcommCcuResType>(i);
        // Instance descriptors report resources already reserved by the instance. Instructions are the exception:
        // HCOMM allocates them while registering a kernel, so their instance count is zero before registration. Check
        // the device's remaining consecutive instruction space instead. Cascading counters are also absent from the
        // instance descriptor in the current SDK; registration remains the authoritative capacity check for them.
        if (type == HCOMM_CCU_RES_TYPE_CASC_CNT) continue;
        const auto descriptor = type == HCOMM_CCU_RES_TYPE_INSTRUCTION ? remaining_descriptor : instance_descriptor;
        uint32_t available{};
        rc = HcommCcuInsResDescQueryNum(descriptor, type, &available);
        if (rc == CCU_SUCCESS && required_resources_[i] > available) rc = CCU_E_UNAVAIL;
    }
    const int remaining_cleanup_error = HcommCcuInsResDescDestroy(remaining_descriptor);
    const int instance_cleanup_error = HcommCcuInsResDescDestroy(instance_descriptor);
    if (rc != CCU_SUCCESS) return rc;
    return remaining_cleanup_error != CCU_SUCCESS ? remaining_cleanup_error : instance_cleanup_error;
}

int HcommDomainRegistrationSdk::register_start() {
    if (!HcommCcuKernelRegisterStart) return CCU_E_UNAVAIL;
    return HcommCcuKernelRegisterStart(instance_handle_);
}

int HcommDomainRegistrationSdk::register_kernel(
    const std::string &name, const void *entry, const void *const *arguments, uint32_t argument_count,
    uint64_t *program_handle
) {
    if (!HcommCcuKernelRegister) return CCU_E_UNAVAIL;
    if (argument_count != 1 || arguments == nullptr || arguments[0] == nullptr) return CCU_E_PARA;
    const ReadyDefinition definition{
        reinterpret_cast<SimplerCcuNativeKernelEntry>(const_cast<void *>(entry)),
        static_cast<const SimplerCcuRegistrationBinding *>(arguments[0])
    };
    const void *ready_arguments[]{&definition};
    return HcommCcuKernelRegister(
        instance_handle_, die_id_, name.c_str(), reinterpret_cast<const void *>(ready_definition), ready_arguments, 1,
        program_handle
    );
}

int HcommDomainRegistrationSdk::register_end() {
    if (!HcommCcuKernelRegisterEnd) return CCU_E_UNAVAIL;
    return HcommCcuKernelRegisterEnd(instance_handle_);
}

int HcommDomainRegistrationSdk::get_task_argument_count(uint64_t program_handle, uint32_t *argument_count) {
    return HcommCcuGetTaskArgsNum(program_handle, argument_count);
}

}  // namespace simpler::ccu
