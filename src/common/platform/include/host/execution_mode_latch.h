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

#include <mutex>

#include "execution_mode.h"
#include "runtime_c_api.h"

/**
 * Write-once execution identity of one device context.
 *
 * A context's mode is a property of its construction, not a state that
 * evolves: the C ABI creates contexts mode-less (`create_device_context`
 * takes no parameters), so the first init entry to run completes the
 * construction by latching the mode, and the latch never changes afterwards
 * — not on finalize, not on error. `simpler_init` latches PROGRAM before
 * touching any runner state, so a live program context is never unlatched
 * and the program/kernel mutual exclusion is enforced on every init, not by
 * anyone remembering a separate claim step. Re-initializing a finalized
 * context under the same mode stays possible (latching the held mode is
 * idempotent); changing a context's identity is not. That permission is real
 * for PROGRAM, whose finalize resets device_id_ to -1; for KERNEL the latch
 * would allow the re-latch but KernelExecutionState's phase machine refuses
 * the re-init, since close() leaves the phase Closed and initialize() accepts
 * only New.
 *
 * There is no unlatch and no rollback: an init that fails after latching
 * leaves the context on that identity permanently, so a handle from a failed
 * kernel init can never be recycled into a program context. Latching is
 * therefore the step an init takes once it has decided which identity it is
 * constructing, and an init that adopts device state must roll that state
 * back itself.
 *
 * Every kernel-mode guard keys on is_kernel(), which is false until an init
 * latches KERNEL — so the guards arm exactly when a kernel context comes
 * into existence.
 */
class ExecutionModeLatch {
public:
    /** Latch the identity. Idempotent for the held mode; a different mode
        returns PTO_RUNTIME_ERR_INVALID_STATE. */
    int latch(SimplerExecutionMode mode) {
        std::scoped_lock lock(mutex_);
        if (latched_ && mode_ != mode) return PTO_RUNTIME_ERR_INVALID_STATE;
        mode_ = mode;
        latched_ = true;
        return 0;
    }

    bool is_kernel() const {
        std::scoped_lock lock(mutex_);
        return latched_ && mode_ == SIMPLER_MODE_KERNEL;
    }

    bool is_latched() const {
        std::scoped_lock lock(mutex_);
        return latched_;
    }

    /** The held mode; meaningful only once is_latched(). */
    SimplerExecutionMode latched_mode() const {
        std::scoped_lock lock(mutex_);
        return mode_;
    }

private:
    mutable std::mutex mutex_;
    bool latched_{false};
    SimplerExecutionMode mode_{SIMPLER_MODE_PROGRAM};
};
