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
void check(CcuResult rc) {
    if (rc != CCU_SUCCESS) throw std::runtime_error("CCU window instruction rejected");
}
int32_t transfer(const SimplerCcuRegistrationBinding *binding) noexcept {
    try {
        std::array<ccu::Variable, SIMPLER_CCU_TASK_ARGUMENT_COUNT> args;
        for (uint32_t i = 0; i < args.size(); ++i)
            check(ccu::LoadArg(args[i], i));
        ccu::Event done(binding->done_event_handle), copied;
        ccu::Event trigger(binding->trigger_event_handle);
        check(ccu::EventWait(trigger));
        std::array<ccu::Variable, 7> words;
        for (uint32_t i = 0; i < words.size(); ++i) {
            ccu::Variable offset, location;
            offset = uint64_t{i} * 8;
            location = args[0] + offset;
            check(ccu::Load(location, words[i]));
        }
        CCU_IF(words[4] != 0) {
            ccu::LocalAddr local;
            ccu::RemoteAddr remote;
            local.addr = words[0];
            local.token = words[1];
            remote.addr = words[2];
            remote.token = words[3];
            for (uint32_t peer = 0; peer < binding->rank_count; ++peer) {
                if (peer == binding->rank_id) continue;
                CCU_IF(words[5] == peer) {
                    check(ccu::Write(binding->channels[peer], remote, local, words[4], copied));
                    check(ccu::EventWait(copied));
                    check(ccu::NotifyRecord(binding->channels[peer], binding->ccu_id, 1));
                }
            }
            for (uint32_t peer = 0; peer < binding->rank_count; ++peer) {
                if (peer == binding->rank_id) continue;
                CCU_IF(words[6] == peer) { check(ccu::NotifyWait(binding->channels[peer], binding->ccu_id, 1)); }
            }
        }
        check(ccu::EventRecord(done));
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
    Manifest m{};
    m.header.struct_size = sizeof(m.header);
    m.header.manifest_size = sizeof(m);
    m.header.abi_version = SIMPLER_CCU_ABI_VERSION;
    m.header.kernel_record_size = sizeof(m.kernels[0]);
    m.header.kernel_count = 2;
    m.header.kernel_records_offset = offsetof(Manifest, kernels);
    m.header.string_table_offset = offsetof(Manifest, symbols);
    m.header.string_table_size = sizeof(m.symbols);
    std::memcpy(m.header.sdk_build_id, simpler::ccu::sdk_build_id, sizeof(simpler::ccu::sdk_build_id));
    std::memcpy(m.header.pto_isa_build_id, simpler::ccu::pto_isa_build_id, sizeof(simpler::ccu::pto_isa_build_id));
    std::memcpy(m.symbols, "ccu_mission_accu_mission_b", sizeof(m.symbols));
    for (uint32_t i = 0; i < 2; ++i) {
        m.kernels[i] = {
            sizeof(SimplerCcuKernelRecord),
            i + 1,
            56,
            64,
            SIMPLER_CCU_KERNEL_CAP_MASK,
            static_cast<uint32_t>(offsetof(Manifest, symbols)) + i * 13,
            13,
            sizeof(SimplerCcuRegistrationBinding),
            SIMPLER_CCU_REGISTRATION_ARGUMENT_COUNT,
            SIMPLER_CCU_TASK_ARGUMENT_COUNT
        };
    }
    return m;
}
}  // namespace
extern "C" int32_t ccu_mission_a(const SimplerCcuRegistrationBinding *b) noexcept { return transfer(b); }
extern "C" int32_t ccu_mission_b(const SimplerCcuRegistrationBinding *b) noexcept { return transfer(b); }
extern "C" const SimplerCcuModuleManifest *simpler_ccu_module_manifest() noexcept {
    static const auto m = make_manifest();
    return &m.header;
}
