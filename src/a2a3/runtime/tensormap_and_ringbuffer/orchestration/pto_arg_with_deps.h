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

#include <stddef.h>
#include <stdint.h>

#include <type_traits>

#include "pto_orchestration_api.h"  // Arg, MixedKernels, rt_submit_* primitives

template <size_t MAX_DEP_COUNT = 16>
class ArgWithDeps : private Arg
{
public:
    // Tensor / scalar setters — forward to Arg
    using Arg::add_inout;
    using Arg::add_input;
    using Arg::add_no_dep;
    using Arg::add_output;
    using Arg::add_scalar;
    using Arg::add_scalars;
    using Arg::add_scalars_i32;
    using Arg::copy_scalars_from;

    // Error / status — forward to Arg
    using Arg::error_msg;
    using Arg::has_error;
    using Arg::launch_spec;
    using Arg::set_error;

    template <typename... Ids>
    void add_dep(Ids... ids)
    {
        static_assert(sizeof...(Ids) >= 1, "add_dep: at least one task id is required");
        static_assert((std::is_same_v<std::decay_t<Ids>, PTO2TaskId> && ...), "add_dep: all arguments must be PTO2TaskId");
        if (count_ + sizeof...(Ids) > MAX_DEP_COUNT)
        {
            Arg::set_error("ArgWithDeps::add_dep: dep count exceeds MAX_DEP_COUNT (bump the template arg)");
            return;
        }
        ((deps_[count_++] = ids), ...);
    }

    void reset()
    {
        Arg::reset();
        count_ = 0;
    }

    Arg &finalize_for_submit()
    {
        Arg::set_dependencies(nullptr, 0);
        Arg::set_dependencies(deps_, count_);
        return *this;
    }

private:
    PTO2TaskId deps_[MAX_DEP_COUNT];
    uint32_t count_ = 0;
};

template <size_t N>
static inline TaskOutputTensors rt_submit_task(const MixedKernels &mixed_kernels, ArgWithDeps<N> &awd)
{
    return rt_submit_task(mixed_kernels, awd.finalize_for_submit());
}

template <size_t N>
static inline TaskOutputTensors rt_submit_aic_task(int32_t kernel_id, ArgWithDeps<N> &awd)
{
    return rt_submit_aic_task(kernel_id, awd.finalize_for_submit());
}

template <size_t N>
static inline TaskOutputTensors rt_submit_aiv_task(int32_t kernel_id, ArgWithDeps<N> &awd)
{
    return rt_submit_aiv_task(kernel_id, awd.finalize_for_submit());
}
