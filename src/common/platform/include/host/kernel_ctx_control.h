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
#include <mutex>

#include "runtime_c_api.h"

/**
 * Per-context state machine behind simpler_kernel_mode_ctx_control.
 *
 * One instance lives on each device runner (onboard and sim). Every
 * host-runtime variant — real implementation or unsupported stub — routes its
 * simpler_kernel_mode_ctx_control through this one class, so structural
 * validation, ordering checks, and idempotency behave identically across all
 * eight components; only the capability flags differ. That parity is what
 * lets the fail-closed rules be tested on any component and hold on all.
 *
 * Ordering rules (all violations return PTO_RUNTIME_ERR_INVALID_STATE):
 *   - CONFIGURE is accepted only before init. Repeating it with the same
 *     effective tuple is idempotent; a different tuple is rejected.
 *   - FREEZE is accepted only on a context whose kernel-mode CONFIGURE was
 *     accepted, after init once capacity is established, and only once: a
 *     second FREEZE is rejected, and CONFIGURE after FREEZE is unreachable
 *     (freeze implies init). A never-configured or program-mode context has
 *     nothing the frozen bit protects, so its FREEZE is an ordering
 *     violation — which also means a component whose kernel CONFIGURE
 *     reports unsupported can never reach a successful FREEZE.
 *
 * Structural violations (null/version/size/action/mode/reserved/FREEZE
 * payload) return PTO_RUNTIME_ERR_INTERNAL before any state mutation.
 * Structurally valid, well-ordered requests the component cannot honor
 * return PTO_RUNTIME_ERR_UNSUPPORTED — after the shared checks, never
 * instead of them.
 *
 * The tuple comparison covers (mode, gm_heap_bytes, gm_sm_bytes,
 * runtime_arena_bytes) only; abi_version, struct_size, action, and reserved
 * do not participate.
 */
class KernelCtxControlState {
public:
    /* Context facts the runner supplies per call; this class owns none of them. */
    struct Environment {
        bool init_done{false};
        bool capacity_established{false};
    };

    /* What this component can honor once the shared validation has passed.
       A component that supports neither is a pure stub and still runs every
       check above. */
    struct Capabilities {
        bool kernel_mode{false};
        bool capacity_intent{false};
    };

    int apply(const SimplerKernelCtxControl *control, const Environment &env, const Capabilities &caps);

    SimplerExecutionMode configured_mode() const;
    bool frozen() const;

private:
    struct EffectiveTuple {
        uint32_t mode{SIMPLER_MODE_PROGRAM};
        uint64_t gm_heap_bytes{0};
        uint64_t gm_sm_bytes{0};
        uint64_t runtime_arena_bytes{0};

        bool operator==(const EffectiveTuple &other) const {
            return mode == other.mode && gm_heap_bytes == other.gm_heap_bytes && gm_sm_bytes == other.gm_sm_bytes &&
                   runtime_arena_bytes == other.runtime_arena_bytes;
        }
    };

    mutable std::mutex mutex_;
    bool configured_{false};
    bool frozen_{false};
    EffectiveTuple tuple_{};
};
