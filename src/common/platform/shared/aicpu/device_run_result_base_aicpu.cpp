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
 * @file device_run_result_base_aicpu.cpp
 * @brief Published run-result region base and epoch for the AICPU publisher.
 *
 * Globals inside the AICPU SO rather than `thread_local` (per
 * docs/dynamic-linking.md), so they survive the host<->dlopen'd runtime SO
 * boundary on sim.
 *
 * **Every launched AICPU thread stores here, concurrently.** The kernel entry
 * runs once per thread and each one publishes from its own KernelArgs copy, so
 * the stores carry identical values but still overlap in time — which is a data
 * race on a plain object no matter what value is written. Hence `std::atomic`
 * with relaxed ordering: relaxed is enough because no reader depends on these
 * stores for ordering. The publisher runs on the thread the run's finish count
 * elects, and that election is an acq_rel read-modify-write ordered after every
 * other thread's, so a value stored before any thread reached the count is
 * already visible without these stores carrying ordering of their own.
 *
 * The epoch travels with the base because the two are only meaningful together:
 * a base without this run's epoch cannot be published into, since the host
 * decides what belongs to this run by comparing epochs.
 */

#include "aicpu/device_run_result_base_aicpu.h"

#include <atomic>

namespace {
std::atomic<uint64_t> g_platform_run_result_base{0};
std::atomic<uint64_t> g_platform_run_result_epoch{0};
}  // namespace

extern "C" void set_platform_run_result(uint64_t region_base, uint64_t run_epoch) {
    g_platform_run_result_base.store(region_base, std::memory_order_relaxed);
    g_platform_run_result_epoch.store(run_epoch, std::memory_order_relaxed);
}

extern "C" uint64_t get_platform_run_result_base() {
    return g_platform_run_result_base.load(std::memory_order_relaxed);
}

extern "C" uint64_t get_platform_run_result_epoch() {
    return g_platform_run_result_epoch.load(std::memory_order_relaxed);
}
