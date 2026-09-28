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
 * A TaskId for a test that is built against both runtimes.
 *
 * The two runtimes mint through different factories -- host_build_graph takes a
 * task-table local id, tensormap_and_ringbuffer takes a (ring, local) pair -- so a
 * case compiled for both cannot name either one. This picks whichever the build's
 * TaskId offers.
 */

#pragma once

#include <cstdint>
#include <type_traits>

// The owning runtime's task handle, resolved by the bare name the way every other
// runtime-agnostic source reaches it: src/common/<runtime> is on the build's include
// path, and reaching both headers from one scope is a compile error.
#include "task_id.h"

namespace simpler::ut {

template <typename T, typename = void>
struct HasMakeGlobal : std::false_type {};

template <typename T>
struct HasMakeGlobal<T, std::void_t<decltype(T::make_global(int32_t{0}))>> : std::true_type {};

/**
 * A handle distinct for each `n`, and nothing more.
 *
 * What a caller may rely on: distinct inputs give handles that compare unequal, the
 * same input gives the same handle, and every result is a valid mint of the build's
 * runtime. Which bits move is not part of that -- a test that asserts on the layout
 * belongs in that runtime's own directory, where it can name the factory directly.
 *
 * Templated on the handle so the branch not taken is discarded rather than compiled:
 * `if constexpr` only drops a branch inside a template, and each runtime declares
 * exactly one of these two factories.
 */
template <typename T = TaskId>
inline T test_task_id(int32_t n) {
    if constexpr (HasMakeGlobal<T>::value) {
        return T::make_global(n);
    } else {
        return T::make(0, n);
    }
}

}  // namespace simpler::ut
