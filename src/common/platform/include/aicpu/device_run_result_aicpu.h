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
 * @file device_run_result_aicpu.h
 * @brief AICPU-side publication into one run's result region.
 *
 * See `common/device_run_result.h` for why the region exists. The publisher's
 * obligations, all of which this helper carries:
 *
 *   - Run **once per run, on the thread that observes every other thread's
 *     writes**, so the payload is the run's final state rather than one
 *     thread's view of it.
 *   - Run **before the kernel returns**, so the copy precedes the fence that
 *     releases any successor to reset the shared state it was copied from.
 *   - Make the payload visible to the host **before** the epoch that marks it
 *     valid. A thread-ordering fence does not do this: the host reads through a
 *     D2H copy of device memory, so the bytes have to leave this core's cache.
 *     Hence an explicit flush of the payload, then of the epoch.
 *
 * A caller publishes only what it wants preserved. Today that is a failing
 * run's error state: this region preserves an error scene, and is deliberately
 * not a per-run status channel — whether a run failed is still decided by the
 * host's execution error path, which for a normal drain is still the stream
 * synchronize (#2267 is open).
 */

#ifndef SRC_COMMON_PLATFORM_INCLUDE_AICPU_DEVICE_RUN_RESULT_AICPU_H_
#define SRC_COMMON_PLATFORM_INCLUDE_AICPU_DEVICE_RUN_RESULT_AICPU_H_

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "aicpu/cache_maintenance.h"
#include "common/device_run_result.h"

/**
 * Copy `bytes` from `src` into the run-result region at `region_base` and
 * publish it under `run_epoch`.
 *
 * No-op when the host provided no region (`region_base == 0`), when the run has
 * no epoch, or when the payload does not fit — a producer that outgrows the
 * capacity must widen `DEVICE_RUN_RESULT_PAYLOAD_BYTES` rather than publish a
 * truncated result, so this refuses instead of clamping. Refusing leaves the
 * region holding some earlier run's epoch, which the host reads as "this run
 * published nothing" rather than as a result.
 */
inline void aicpu_publish_run_result(uint64_t region_base, uint64_t run_epoch, const void *src, size_t bytes) {
    if (region_base == 0 || run_epoch == 0 || src == nullptr) return;
    if (bytes == 0 || bytes > DEVICE_RUN_RESULT_PAYLOAD_BYTES) return;

    auto *region = reinterpret_cast<DeviceRunResultRegion *>(region_base);
    std::memcpy(region->payload, src, bytes);
    region->payload_bytes = static_cast<uint32_t>(bytes);
    region->reserved = 0;
    // Payload and its length out of cache first; only then the epoch that says
    // they are readable. A host that sees the epoch must already be able to see
    // what it vouches for.
    cache_flush_range(region->payload, bytes);
    cache_flush_range(&region->payload_bytes, sizeof(region->payload_bytes));
    region->published = run_epoch;
    cache_flush_range(&region->published, sizeof(region->published));
}

#endif  // SRC_COMMON_PLATFORM_INCLUDE_AICPU_DEVICE_RUN_RESULT_AICPU_H_
