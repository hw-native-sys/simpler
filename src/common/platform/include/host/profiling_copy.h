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
#include <functional>

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

// Per-arch callback selector for leaf collectors that live in `common/`
// (e.g. scope_stats, tensor_dump). Returns the copy function on non-SVM
// platforms (a5: real rtMemcpy/memcpy) and an empty `std::function` on SVM
// platforms (a2a3: host and device share addresses, the framework's
// null-check then short-circuits all mirror/range/buffer ops to no-ops).
//
// Why this exists: the common leaf collector is compiled once per platform
// target but cannot inspect the arch at runtime. Each arch's
// `profiling_copy.cpp` defines these to either return the real shim or an
// empty `std::function`, so the leaf collector can call them blindly and
// `MemoryOps::copy_to_device`/`copy_from_device` ends up correctly null on
// SVM. Without this, passing `&profiling_copy_to_device_for_ops`
// unconditionally on a2a3 sim makes `ProfilerBase::start()` pick the
// host-shadow malloc path even though the per-arch copy stubs are no-ops;
// the shadow ends up detached from the device buffer and AICPU writes
// never surface to the host (segfault on read).
std::function<int(void *, const void *, size_t)> profiling_copy_to_device_or_null();
std::function<int(void *, const void *, size_t)> profiling_copy_from_device_or_null();
