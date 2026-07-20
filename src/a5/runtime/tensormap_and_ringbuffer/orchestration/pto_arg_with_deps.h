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
/**
 * Convenience layer over Arg: bundles a fixed-capacity dependency buffer with
 * an Arg and exposes an incremental add_dep(...) API on top of the runtime
 * primitive L0TaskArgs::set_dependencies(ptr, count).
 *
 * Layering:
 *   - Primitive:   Arg + set_dependencies(ptr, count) in pto_types.h.
 *                  No cap, caller owns the deps buffer.
 *   - Convenience: L0TaskArgsWithDeps<N> in this header. Owns a stack-sized dep
 *                  buffer of capacity N (default 16); provides add_dep().
 *                  Submitted via the rt_submit_*_task overloads below, which
 *                  forward the bundled deps into the underlying Arg.
 *
 * This file is auto-included at the bottom of pto_orchestration_api.h so
 * orchestration sources see L0TaskArgsWithDeps after a single `#include
 * "pto_orchestration_api.h"`. The split is purely organizational —
 * orchestration code should not include this header directly. Code generated
 * from pypto can ignore the convenience layer entirely and target Arg +
 * set_dependencies(ptr, count) directly.
 *
 * L0TaskArgsWithDeps uses private inheritance from Arg so that set_dependencies and
 * the explicit_dep* accessors are NOT reachable on a wrapper instance — users
 * who pick the convenience layer cannot accidentally mix it with the
 * primitive layer's dep API on the same object.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <type_traits>

#include "pto_orchestration_api.h"  // Arg, MixedKernels, rt_submit_* primitives

template <size_t MAX_DEP_COUNT = 16>
class L0TaskArgsWithDeps : private L0TaskArgs {
public:
    // Tensor / scalar setters — forward to Arg
    using L0TaskArgs::add_inout;
    using L0TaskArgs::add_input;
    using L0TaskArgs::add_no_dep;
    using L0TaskArgs::add_output;
    using L0TaskArgs::add_scalar;
    using L0TaskArgs::add_scalars;
    using L0TaskArgs::add_scalars_i32;
    using L0TaskArgs::allow_early_resolve;  // task hint (getter)
    using L0TaskArgs::copy_scalars_from;
    using L0TaskArgs::set_allow_early_resolve;  // task hint (setter)
    using L0TaskArgs::set_task_timing_slot;     // selective task-timing slot (setter)
    using L0TaskArgs::task_timing_slot;         // selective task-timing slot (getter)

    // Error / status — forward to Arg
    using L0TaskArgs::error_msg;
    using L0TaskArgs::has_error;
    using L0TaskArgs::launch_spec;
    using L0TaskArgs::set_error;

    // NOT exposed: set_dependencies, explicit_dep_count, explicit_dep,
    // explicit_deps_data — these are the primitive-layer dep API. Users of
    // the convenience layer reach dependencies only through add_dep() below.

    /**
     * Append one or more dependencies to the bundled buffer. May be called
     * multiple times; deps accumulate. Variadic accepts any non-zero number
     * of PTO2TaskId arguments.
     *
     * Overflow (more than MAX_DEP_COUNT total) records an error on the
     * underlying Arg; the error surfaces at submit time.
     */
    template <typename... Ids>
    void add_dep(Ids... ids) {
        static_assert(sizeof...(Ids) >= 1, "add_dep: at least one task id is required");
        static_assert(
            (std::is_same_v<std::decay_t<Ids>, PTO2TaskId> && ...), "add_dep: all arguments must be PTO2TaskId"
        );
        if (count_ + sizeof...(Ids) > MAX_DEP_COUNT) {
            L0TaskArgs::set_error(
                "L0TaskArgsWithDeps::add_dep: dep count exceeds MAX_DEP_COUNT (bump the template arg)"
            );
            return;
        }
        ((deps_[count_++] = ids), ...);
    }

    /**
     * Clear the bundled dep buffer and reset the underlying Arg.
     * Use this to recycle an L0TaskArgsWithDeps across loop iterations.
     */
    void reset() {
        L0TaskArgs::reset();
        count_ = 0;
    }

    /**
     * Submit-only hook: bind the bundled deps onto the underlying Arg and
     * return it as Arg&. Called by the rt_submit_*_task overloads below;
     * orchestration code does not invoke this directly.
     *
     * Idempotent: explicitly clears any prior dep binding before re-setting,
     * so a wrapper can be re-finalized (e.g. resubmitted) without tripping
     * the primitive layer's single-shot check.
     */
    L0TaskArgs &finalize_for_submit() {
        L0TaskArgs::set_dependencies(nullptr, 0);
        L0TaskArgs::set_dependencies(deps_, count_);
        return *this;
    }

private:
    PTO2TaskId deps_[MAX_DEP_COUNT];
    uint32_t count_ = 0;
};

// =============================================================================
// Submit overloads — accept L0TaskArgsWithDeps<N> transparently
// =============================================================================

template <size_t N>
static inline TaskOutputTensors rt_submit_task(const MixedKernels &mixed_kernels, L0TaskArgsWithDeps<N> &awd) {
    return rt_submit_task(mixed_kernels, awd.finalize_for_submit());
}

template <size_t N>
static inline TaskOutputTensors rt_submit_aic_task(int32_t kernel_id, L0TaskArgsWithDeps<N> &awd) {
    return rt_submit_aic_task(kernel_id, awd.finalize_for_submit());
}

template <size_t N>
static inline TaskOutputTensors rt_submit_aiv_task(int32_t kernel_id, L0TaskArgsWithDeps<N> &awd) {
    return rt_submit_aiv_task(kernel_id, awd.finalize_for_submit());
}
