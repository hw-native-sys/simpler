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

#include "host/kernel_ctx_control.h"

namespace {

bool has_nonzero_reserved(const SimplerKernelCtxControl &control) {
    for (uint64_t word : control.reserved) {
        if (word != 0) return true;
    }
    return false;
}

}  // namespace

int KernelCtxControlState::apply(
    const SimplerKernelCtxControl *control, const Environment &env, const Capabilities &caps
) {
    if (control == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
    if (control->abi_version != SIMPLER_KERNEL_CTX_CONTROL_ABI_VERSION) return PTO_RUNTIME_ERR_INTERNAL;
    if (control->struct_size != sizeof(SimplerKernelCtxControl)) return PTO_RUNTIME_ERR_INTERNAL;
    if (control->action != SIMPLER_KERNEL_CTX_CONFIGURE && control->action != SIMPLER_KERNEL_CTX_FREEZE) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (has_nonzero_reserved(*control)) return PTO_RUNTIME_ERR_INTERNAL;

    std::scoped_lock lock(mutex_);
    if (control->action == SIMPLER_KERNEL_CTX_CONFIGURE) {
        if (control->mode != SIMPLER_MODE_PROGRAM && control->mode != SIMPLER_MODE_KERNEL) {
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        if (env.init_done) return PTO_RUNTIME_ERR_INVALID_STATE;
        const EffectiveTuple requested{
            control->mode, control->gm_heap_bytes, control->gm_sm_bytes, control->runtime_arena_bytes
        };
        if (configured_ && !(requested == tuple_)) return PTO_RUNTIME_ERR_INVALID_STATE;
        /* Capability split only after every shared check passed. */
        if (requested.mode == SIMPLER_MODE_KERNEL && !caps.kernel_mode) return PTO_RUNTIME_ERR_UNSUPPORTED;
        const bool wants_capacity =
            requested.gm_heap_bytes != 0 || requested.gm_sm_bytes != 0 || requested.runtime_arena_bytes != 0;
        if (wants_capacity && !caps.capacity_intent) return PTO_RUNTIME_ERR_UNSUPPORTED;
        tuple_ = requested;
        configured_ = true;
        return 0;
    }

    /* FREEZE carries no payload: mode and every capacity field must be zero. */
    if (control->mode != 0 || control->gm_heap_bytes != 0 || control->gm_sm_bytes != 0 ||
        control->runtime_arena_bytes != 0) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    /* Freezing is kernel-mode machinery: only a context whose kernel-mode
       CONFIGURE was accepted has capacity the frozen bit protects. Rejecting
       everything else keeps a successful FREEZE synonymous with "the freeze
       guard will enforce this", and keeps program-mode contexts structurally
       out of that guard's reach. */
    if (!configured_ || tuple_.mode != SIMPLER_MODE_KERNEL) return PTO_RUNTIME_ERR_INVALID_STATE;
    if (!env.init_done || !env.capacity_established) return PTO_RUNTIME_ERR_INVALID_STATE;
    if (frozen_) return PTO_RUNTIME_ERR_INVALID_STATE;
    frozen_ = true;
    return 0;
}

SimplerExecutionMode KernelCtxControlState::configured_mode() const {
    std::scoped_lock lock(mutex_);
    return static_cast<SimplerExecutionMode>(tuple_.mode);
}

bool KernelCtxControlState::frozen() const {
    std::scoped_lock lock(mutex_);
    return frozen_;
}
