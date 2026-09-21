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
 * Test-local `profiling_copy` implementation with distinct host and device
 * storage.
 *
 * Takes the place of an arch's `profiling_copy.cpp` at link time, so the
 * collector under test follows the non-SVM path: `alloc_paired_buffer` mallocs
 * a host shadow separate from the device allocation, and every host read of
 * device state goes through a real copy that can fail.
 *
 * The arch stubs a test would otherwise link are no-ops over aliased pointers,
 * which makes both failure branches unreachable — a copy that cannot fail
 * cannot show what the collector does when one does.
 *
 * Failure injection lives here rather than in production: the collector has no
 * flag, environment variable or macro that changes its copy behaviour.
 */

#include "profiling_copy_fault.h"

#include <cstring>

#include "host/profiling_copy.h"

namespace copy_fault {
namespace {

Plan g_plan{};
Counts g_counts{};

}  // namespace

void arm(const Plan &plan) {
    g_plan = plan;
    g_counts = Counts{};
}

void reset() {
    g_plan = Plan{};
    g_counts = Counts{};
}

Counts counts() { return g_counts; }

// Real transfer between the separate host shadow and device allocation, except
// for the one size a test armed to fail. A failed copy leaves the destination
// untouched, which is what makes the host shadow's stale contents plausible to
// a reader that ignores the return code.
int from_device(void *host_dst, const void *dev_src, size_t size) {
    g_counts.from_device_calls++;
    if (g_plan.fail_from_device_size != 0 && size == g_plan.fail_from_device_size) {
        g_counts.from_device_failures++;
        return g_plan.fail_rc;
    }
    std::memcpy(host_dst, dev_src, size);
    return 0;
}

}  // namespace copy_fault

int profiling_copy_to_device(volatile void *dev_dst, const void *host_src, size_t size) {
    std::memcpy(const_cast<void *>(dev_dst), host_src, size);
    return 0;
}

int profiling_copy_from_device(volatile void *host_dst, const volatile void *dev_src, size_t size) {
    return copy_fault::from_device(const_cast<void *>(host_dst), const_cast<const void *>(dev_src), size);
}

std::function<int(void *, const void *, size_t)> profiling_copy_to_device_or_null() {
    return &profiling_copy_to_device_for_ops;
}

std::function<int(void *, const void *, size_t)> profiling_copy_from_device_or_null() {
    return &profiling_copy_from_device_for_ops;
}
