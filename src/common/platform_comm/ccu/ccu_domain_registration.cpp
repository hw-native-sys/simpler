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

#include "ccu_domain_registration.h"

#include <map>
#include <set>
#include <algorithm>
#include <sstream>
#include <utility>

namespace simpler::ccu {
namespace {

constexpr int kProtocolError = -1;

[[noreturn]] void fail_protocol(const std::string &operation) {
    throw DomainRegistrationError(operation, kProtocolError);
}

std::string error_message(const std::string &operation, int native_error, int cleanup_error) {
    std::ostringstream message;
    message << operation << " failed with code " << native_error;
    if (cleanup_error != 0) message << "; RegisterEnd cleanup failed with code " << cleanup_error;
    return message.str();
}

std::string program_name(const DomainRegistrationRequest &request, const NativeKernelDefinition &kernel) {
    std::ostringstream name;
    name << request.program_namespace << "_d" << request.domain_id << "_k" << kernel.record.kernel_id << "_r"
         << request.rank_id;
    return name.str();
}

}  // namespace

DomainRegistrationError::DomainRegistrationError(std::string operation, int native_error, int cleanup_error) :
    std::runtime_error(error_message(operation, native_error, cleanup_error)),
    operation_(std::move(operation)),
    native_error_(native_error),
    cleanup_error_(cleanup_error) {}

std::vector<RegisteredKernel> register_domain_kernels(
    const std::vector<NativeKernelDefinition> &kernels, const DomainRegistrationRequest &request,
    DomainRegistrationSdk &sdk
) {
    if (request.rank_count < 2 || request.rank_count > SIMPLER_CCU_MAX_RANKS || request.rank_id >= request.rank_count)
        fail_protocol("CCU domain logical rank");
    for (uint32_t peer = 0; peer < SIMPLER_CCU_MAX_RANKS; ++peer) {
        const bool required = peer < request.rank_count && peer != request.rank_id;
        if ((request.channels[peer] != 0) != required) fail_protocol("CCU all-peer channel binding");
    }
    if (request.program_namespace.empty()) fail_protocol("CCU program namespace");

    if (kernels.empty() || request.missions.size() != kernels.size()) fail_protocol("CCU control count");
    if (!request.ccu_ids.empty() && request.ccu_ids.size() != kernels.size()) fail_protocol("CCU ID count");
    std::set<uint32_t> ids;
    for (uint32_t id : request.ccu_ids)
        if (id == UINT32_MAX || !ids.insert(id).second) fail_protocol("CCU duplicate or invalid ID");
    std::vector<const NativeKernelDefinition *> selected;
    std::vector<SimplerCcuRegistrationBinding> bindings;
    for (const auto &mission : request.missions) {
        if (mission.ready_event_handle == 0 || mission.trigger_event_handle == 0 || mission.done_event_handle == 0 ||
            mission.ready_event_handle == mission.trigger_event_handle ||
            mission.ready_event_handle == mission.done_event_handle ||
            mission.trigger_event_handle == mission.done_event_handle)
            fail_protocol("CCU mission event binding");
        const auto &kernel = kernels[bindings.size()];
        if (kernel.entry == nullptr ||
            kernel.record.registration_binding_size != sizeof(SimplerCcuRegistrationBinding) ||
            kernel.record.registration_argument_count != SIMPLER_CCU_REGISTRATION_ARGUMENT_COUNT ||
            kernel.record.task_argument_count != SIMPLER_CCU_TASK_ARGUMENT_COUNT)
            fail_protocol("CCU kernel ABI");
        if (kernel.record.required_rank_count != 0 && kernel.record.required_rank_count != request.rank_count)
            fail_protocol("CCU kernel domain rank specialization");
        selected.push_back(&kernel);
        bindings.push_back(
            {sizeof(SimplerCcuRegistrationBinding),
             request.ccu_ids.empty() ? static_cast<uint32_t>(bindings.size()) : request.ccu_ids[bindings.size()],
             request.rank_id,
             request.rank_count,
             {},
             mission.ready_event_handle,
             mission.trigger_event_handle,
             mission.done_event_handle}
        );
        std::copy(request.channels.begin(), request.channels.end(), bindings.back().channels);
    }

    for (size_t i = 0; i < selected.size(); ++i) {
        const void *arguments[] = {&bindings[i]};
        const int rc = sdk.query_resources(selected[i]->entry, arguments, SIMPLER_CCU_REGISTRATION_ARGUMENT_COUNT);
        if (rc != 0) throw DomainRegistrationError("HcommCcuKernelQueryResReq", rc);
    }
    const int capacity_rc = sdk.validate_capacity();
    if (capacity_rc != 0) throw DomainRegistrationError("CCU instance capacity", capacity_rc);

    const int start_rc = sdk.register_start();
    if (start_rc != 0) throw DomainRegistrationError("HcommCcuKernelRegisterStart", start_rc);

    std::vector<RegisteredKernel> registered;
    registered.reserve(selected.size());
    try {
        for (size_t i = 0; i < selected.size(); ++i) {
            const void *arguments[] = {&bindings[i]};
            uint64_t program_handle{};
            const int rc = sdk.register_kernel(
                program_name(request, *selected[i]) + "_m" + std::to_string(bindings[i].ccu_id), selected[i]->entry,
                arguments, SIMPLER_CCU_REGISTRATION_ARGUMENT_COUNT, &program_handle
            );
            if (rc != 0 || program_handle == 0)
                throw DomainRegistrationError("HcommCcuKernelRegister", rc != 0 ? rc : kProtocolError);
            registered.push_back({selected[i]->record.kernel_id, bindings[i].ccu_id, program_handle});
        }
    } catch (const DomainRegistrationError &error) {
        const int cleanup = sdk.register_end();
        throw DomainRegistrationError(error.operation(), error.native_error(), cleanup);
    } catch (...) {
        (void)sdk.register_end();
        throw;
    }

    const int end_rc = sdk.register_end();
    if (end_rc != 0) throw DomainRegistrationError("HcommCcuKernelRegisterEnd", end_rc);

    for (const auto &kernel : registered) {
        uint32_t argument_count{};
        const int rc = sdk.get_task_argument_count(kernel.program_handle, &argument_count);
        if (rc != 0) throw DomainRegistrationError("HcommCcuGetTaskArgsNum", rc);
        if (argument_count != SIMPLER_CCU_TASK_ARGUMENT_COUNT)
            throw DomainRegistrationError("CCU task argument ABI", kProtocolError);
    }
    return registered;
}

}  // namespace simpler::ccu
