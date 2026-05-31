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

#include <cstddef>

// Platform-specific copy hooks for profiling collectors.
//
// Implementations live in each arch's onboard/sim host directory:
//   - a2a3: stubs that return 0 (SVM — host and device share address space, so
//     no mirror is needed and collectors never call these). Stubs exist only
//     to satisfy the symbol references the framework header pulls in.
//   - a5 onboard: rtMemcpy-based.
//   - a5 sim: plain memcpy.
//
// The shared profiling framework (`profiler_base.h`) only invokes these
// through `MemoryOps::copy_to_device` / `copy_from_device` lambdas. Collectors
// that don't install those lambdas (i.e. all a2a3 collectors) never reach the
// stubs at runtime.
int profiling_copy_to_device(volatile void *dev_dst, const void *host_src, size_t size);
int profiling_copy_from_device(volatile void *host_dst, const volatile void *dev_src, size_t size);

// Non-volatile-signature shims for use with MemoryOps's `copy_to_device` /
// `copy_from_device` callback slots (the framework's std::function shape).
inline int profiling_copy_to_device_for_ops(void *dev_dst, const void *host_src, size_t size) {
    return profiling_copy_to_device(dev_dst, host_src, size);
}
inline int profiling_copy_from_device_for_ops(void *host_dst, const void *dev_src, size_t size) {
    return profiling_copy_from_device(host_dst, dev_src, size);
}
