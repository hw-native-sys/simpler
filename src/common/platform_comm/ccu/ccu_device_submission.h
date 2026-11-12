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
#include "ccu_native_abi.h"
#include "platform_comm/comm_context.h"

#if defined(__CCE_AICORE__)
#define SIMPLER_CCU_SUBMIT_DEVICE __aicore__
#define SIMPLER_CCU_SUBMIT_GM __gm__
#else
#define SIMPLER_CCU_SUBMIT_DEVICE
#define SIMPLER_CCU_SUBMIT_GM
#endif

namespace simpler::ccu {
struct CcuCallArg {
    enum class Kind : uint32_t { Scalar, Window, PeerRank };
    Kind kind;
    uint32_t rank;
    uint64_t value;  // Scalar value or byte offset in W.
    uint64_t size;
};

inline SIMPLER_CCU_SUBMIT_DEVICE const SIMPLER_CCU_SUBMIT_GM SimplerCcuWorkspace *
ccu_workspace(const SIMPLER_CCU_SUBMIT_GM CommContext *domain) {
    if (domain == nullptr) return nullptr;
    const auto &slot = comm_workspace<CommWorkspaceKind::CCU>(*domain);
    if (slot.addr == 0 || slot.addr % alignof(SimplerCcuWorkspace) != 0 || slot.size != sizeof(SimplerCcuWorkspace))
        return nullptr;
    const auto *workspace = reinterpret_cast<const SIMPLER_CCU_SUBMIT_GM SimplerCcuWorkspace *>(slot.addr);
    return workspace->struct_size == sizeof(*workspace) ? workspace : nullptr;
}

// Positional wire arguments: scalars occupy one word; windows occupy address, token.
template <size_t N>
SIMPLER_CCU_SUBMIT_DEVICE bool resolve_call_args(
    const SIMPLER_CCU_SUBMIT_GM CommContext *domain, const CcuCallArg (&args)[N], uint64_t (&words)[N * 2],
    uint64_t &bytes
) {
    if (domain == nullptr || domain->rankNum == 0 || domain->rankNum > COMM_MAX_RANK_NUM ||
        domain->rankId >= domain->rankNum)
        return false;
    const auto *workspace = ccu_workspace(domain);
    size_t count = 0;
    for (size_t i = 0; i < N; ++i) {
        const auto &arg = args[i];
        if (arg.kind == CcuCallArg::Kind::Scalar) {
            words[count++] = arg.value;
        } else if (arg.kind == CcuCallArg::Kind::Window || arg.kind == CcuCallArg::Kind::PeerRank) {
            if (workspace == nullptr || arg.rank >= domain->rankNum || workspace->windows_addr == 0 ||
                workspace->windows_addr % alignof(CommCcuWindow) != 0 || workspace->window_count != domain->rankNum ||
                workspace->window_size != sizeof(CommCcuWindow))
                return false;
            const auto *windows =
                reinterpret_cast<const SIMPLER_CCU_SUBMIT_GM CommCcuWindow *>(workspace->windows_addr);
            const auto &window = windows[arg.rank];
            if (window.reserved != 0 || window.l2_rank >= COMM_MAX_RANK_NUM) return false;
            if (arg.kind == CcuCallArg::Kind::PeerRank) {
                words[count++] = window.l2_rank;
                continue;
            }
            if (window.address == 0 || arg.size == 0 || arg.value > window.size || arg.size > window.size - arg.value ||
                window.address > UINT64_MAX - arg.value || arg.size - 1 > UINT64_MAX - (window.address + arg.value))
                return false;
            words[count++] = window.address + arg.value;
            words[count++] = window.token;
        } else {
            return false;
        }
    }
    bytes = count * sizeof(uint64_t);
    return true;
}

inline SIMPLER_CCU_SUBMIT_DEVICE const volatile SIMPLER_CCU_SUBMIT_GM SimplerCcuControl *
ccu_control(const volatile SIMPLER_CCU_SUBMIT_GM SimplerCcuWorkspace *workspace, uint32_t ccu_id) {
    if (workspace == nullptr || workspace->struct_size != sizeof(*workspace) || workspace->run_generation == 0 ||
        workspace->controls_addr == 0 || workspace->controls_addr % alignof(SimplerCcuControl) != 0 ||
        ccu_id >= workspace->ccu_count || workspace->control_size != sizeof(SimplerCcuControl) || workspace->flags != 0)
        return nullptr;
    const auto *controls =
        reinterpret_cast<const volatile SIMPLER_CCU_SUBMIT_GM SimplerCcuControl *>(workspace->controls_addr);
    const auto *control = &controls[ccu_id];
    if (control->ccu_id != ccu_id || control->run_generation != workspace->run_generation) return nullptr;
    return control;
}

struct DeviceSubmissionIo {
    static SIMPLER_CCU_SUBMIT_DEVICE void publish(volatile SIMPLER_CCU_SUBMIT_GM void *address, uint64_t bytes) {
#if defined(__CCE_AICORE__)
        (void)bytes;
        dcci(const_cast<SIMPLER_CCU_SUBMIT_GM void *>(address), cache_line_t::ENTIRE_DATA_CACHE);
        dsb(DSB_DDR);
#else
        (void)address;
        (void)bytes;
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
    }
};

template <typename Io = DeviceSubmissionIo, typename RegisterCompletion>
SIMPLER_CCU_SUBMIT_DEVICE bool submit_mission(
    const volatile SIMPLER_CCU_SUBMIT_GM SimplerCcuControl *mission, const void *parameters, uint64_t parameter_size,
    RegisterCompletion &&register_completion
) {
    if (mission == nullptr || mission->struct_size != sizeof(*mission) || mission->run_generation == 0 ||
        mission->flags != 0 || mission->parameter_addr == 0 || mission->parameter_alignment == 0 ||
        (mission->parameter_alignment & (mission->parameter_alignment - 1)) != 0 ||
        mission->parameter_addr % mission->parameter_alignment != 0 || mission->parameter_size != parameter_size ||
        (parameter_size != 0 && parameters == nullptr))
        return false;
    const uint64_t addresses[]{mission->ready.address, mission->trigger.address, mission->done.address};
    if (mission->ready.mask != SIMPLER_CCU_EVENT_MASK || mission->trigger.mask != SIMPLER_CCU_EVENT_MASK ||
        mission->done.mask != SIMPLER_CCU_EVENT_MASK)
        return false;
    for (uint32_t i = 0; i < 3; ++i) {
        if (addresses[i] == 0 || addresses[i] % alignof(uint64_t) != 0) return false;
        for (uint32_t j = 0; j < i; ++j)
            if (addresses[i] == addresses[j]) return false;
    }
    if (!register_completion(mission->done.address, mission->done.mask)) return false;
    auto *destination = reinterpret_cast<volatile SIMPLER_CCU_SUBMIT_GM uint8_t *>(mission->parameter_addr);
    const auto *source = static_cast<const uint8_t *>(parameters);
    for (uint64_t i = 0; i < parameter_size; ++i)
        destination[i] = source[i];
    Io::publish(destination, parameter_size);
    auto *trigger = reinterpret_cast<volatile SIMPLER_CCU_SUBMIT_GM uint64_t *>(mission->trigger.address);
    *trigger = (uint64_t{1} << 63) | mission->trigger.mask;
    Io::publish(trigger, sizeof(*trigger));
    return true;
}
}  // namespace simpler::ccu
#undef SIMPLER_CCU_SUBMIT_DEVICE
#undef SIMPLER_CCU_SUBMIT_GM
