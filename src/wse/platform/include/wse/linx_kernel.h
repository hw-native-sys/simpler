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
#include <pto/pto-inst.hpp>
#include "wse/linx_kernel_abi.h"

#pragma GCC visibility push(hidden)
namespace wse::detail {
inline void *create_linx_core(uint32_t pe_mask) {
    auto *core = new pto::linx_cpu::CoreContext;
    core->activeMask = pe_mask;
    return core;
}

inline void destroy_linx_core(void *opaque) { delete static_cast<pto::linx_cpu::CoreContext *>(opaque); }

inline void cancel_linx_core(void *opaque, const std::exception_ptr &error) {
    auto &core = *static_cast<pto::linx_cpu::CoreContext *>(opaque);
    std::lock_guard lock(core.mutex);
    if (!core.error) core.error = error;
    core.changed.notify_all();
}

inline void invoke_linx_kernel(void *opaque, uint32_t pe, KernelEntry entry, int64_t *args) {
    namespace linx = pto::linx_cpu;
    auto &core = *static_cast<linx::CoreContext *>(opaque);
    linx::currentCore = &core;
    linx::peIndex = pe;
    std::exception_ptr error;
    try {
        {
            std::lock_guard lock(core.mutex);
            if (core.error) std::rethrow_exception(core.error);
        }
        entry(args);
    } catch (...) {
        error = std::current_exception();
    }
    {
        std::lock_guard lock(core.mutex);
        core.finished |= 1u << pe;
        if (!core.error) {
            if (error) core.error = error;
            else if (core.arrived || core.barrierArrived)
                core.error = std::make_exception_ptr(std::invalid_argument("divergent Linx Core collective"));
        }
        error = core.error;
        core.changed.notify_all();
    }
    // WSE workers execute successive Blocks on the same OS thread.
    linx::sharedStorage.clear();
    linx::sharedSlots.fill(0);
    linx::currentCore = nullptr;
    linx::peIndex = 0;
    if (error) std::rethrow_exception(error);
}

// Each loaded image binds its own entry, even when another kernel is globally visible.
inline const LinxKernel *bind_linx_kernel(KernelEntry entry) {
    static const LinxKernel kernel{entry, create_linx_core, destroy_linx_core, invoke_linx_kernel, cancel_linx_core};
    return &kernel;
}
}  // namespace wse::detail
#pragma GCC visibility pop

extern "C" __attribute__((used, visibility("default"))) inline const wse::LinxKernel *
simpler_wse_linx_kernel(wse::KernelEntry entry) {
    return wse::detail::bind_linx_kernel(entry);
}
