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
 * Shared helper: set perf_level and legacy enable_profiling on a Runtime struct.
 *
 * Used by both onboard and sim pto_runtime_c_api.cpp implementations.
 * Some runtime structs still carry a bool enable_profiling member alongside
 * the newer int perf_level.  This template detects the legacy member at
 * compile time and keeps both in sync.
 */

#pragma once

#include <type_traits>

template <typename T, typename = void>
struct HasEnableProfilingMember : std::false_type {};

template <typename T>
struct HasEnableProfilingMember<T, std::void_t<decltype(std::declval<T &>().enable_profiling)>> : std::true_type {};

template <typename R>
static inline void set_runtime_profiling_mode(R *runtime, int enable_profiling) {
    runtime->perf_level = enable_profiling;
    if constexpr (HasEnableProfilingMember<R>::value) {
        runtime->enable_profiling = (enable_profiling > 0);
    }
}
