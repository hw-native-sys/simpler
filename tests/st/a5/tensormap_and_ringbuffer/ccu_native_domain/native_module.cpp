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

#include <array>
#include <cstring>
#include <stdexcept>

#include "ccu_manifest_identity.h"
#include "hcomm/ccu/ccu_primitives.hpp"
#include "hcomm/ccu/ccu_res.h"
#include "platform_comm/ccu/ccu_native_abi.h"

namespace {
namespace ccu = AscendC::ccu;

void check(CcuResult result) {
    if (result != CCU_SUCCESS) throw std::runtime_error("native CCU instruction rejected");
}

ccu::Variable address(ccu::Variable base, uint64_t offset) {
    ccu::Variable constant;
    constant = offset;
    ccu::Variable result;
    result = base + constant;
    return result;
}

void store(ccu::Variable base, uint64_t offset, uint64_t value) {
    ccu::Variable source;
    source = value;
    check(ccu::Store(address(base, offset), source));
}

int32_t native_mission(const SimplerCcuRegistrationBinding *binding) noexcept {
    try {
        std::array<ccu::Variable, SIMPLER_CCU_TASK_ARGUMENT_COUNT> args;
        for (uint32_t i = 0; i < args.size(); ++i)
            check(ccu::LoadArg(args[i], i));
        ccu::Event trigger(binding->trigger_event_handle), done(binding->done_event_handle);
        const uint32_t operation_notify = binding->ccu_id;
        const auto channel = binding->channels[(binding->rank_id + 1) % binding->rank_count];
        ccu::Variable output, value, again;
        again = 1;
        CCU_WHILE(again == 1) {
            check(ccu::EventWait(trigger, 1));
            check(ccu::Load(args[0], output));
            check(ccu::Load(address(args[0], 8), value));
            check(ccu::Load(address(args[0], 16), again));
            CCU_IF(value == 0) {
                // Deliberately omit done to exercise the L2 execution timeout.
                store(output, 0, 2971);
            }
            CCU_ELSE {
                check(ccu::Store(output, value));
                CCU_IF(again == 0) {
                    check(ccu::NotifyRecord(channel, operation_notify, 1));
                    check(ccu::NotifyWait(channel, operation_notify, 1));
                }
                check(ccu::EventRecord(done, 1));
            }
        }
        return CCU_SUCCESS;
    } catch (...) {
        return CCU_E_INTERNAL;
    }
}

struct Manifest {
    SimplerCcuModuleManifest header;
    SimplerCcuKernelRecord kernels[2];
    char symbols[26];
};

Manifest make_manifest() {
    Manifest result{};
    result.header.struct_size = sizeof(result.header);
    result.header.manifest_size = sizeof(result);
    result.header.kernel_record_size = sizeof(result.kernels[0]);
    result.header.kernel_count = 2;
    result.header.kernel_records_offset = offsetof(Manifest, kernels);
    result.header.string_table_offset = offsetof(Manifest, symbols);
    result.header.string_table_size = sizeof(result.symbols);
    std::memcpy(result.header.sdk_build_id, simpler::ccu::sdk_build_id, sizeof(simpler::ccu::sdk_build_id));
    std::memcpy(result.header.pto_isa_build_id, simpler::ccu::pto_isa_build_id, sizeof(simpler::ccu::pto_isa_build_id));
    constexpr char names[] = "ccu_mission_accu_mission_b";
    static_assert(sizeof(names) - 1 == sizeof(result.symbols));
    std::memcpy(result.symbols, names, sizeof(result.symbols));
    for (uint32_t i = 0; i < 2; ++i) {
        result.kernels[i] = {
            sizeof(SimplerCcuKernelRecord),
            i + 1,
            24,
            64,
            SIMPLER_CCU_KERNEL_CAP_MASK,
            static_cast<uint32_t>(offsetof(Manifest, symbols)) + i * 13,
            13,
            sizeof(SimplerCcuRegistrationBinding),
            SIMPLER_CCU_REGISTRATION_ARGUMENT_COUNT,
            SIMPLER_CCU_TASK_ARGUMENT_COUNT
        };
    }
    return result;
}
}  // namespace

extern "C" int32_t ccu_mission_a(const SimplerCcuRegistrationBinding *binding) noexcept {
    return native_mission(binding);
}
extern "C" int32_t ccu_mission_b(const SimplerCcuRegistrationBinding *binding) noexcept {
    return native_mission(binding);
}
extern "C" const SimplerCcuModuleManifest *simpler_ccu_module_manifest() noexcept {
    static const Manifest manifest = make_manifest();
    return &manifest.header;
}
