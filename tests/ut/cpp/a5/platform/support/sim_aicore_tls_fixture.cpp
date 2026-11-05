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

#include <atomic>
#include <cstdint>
#include <pthread.h>
#include <thread>
#include <vector>

#include "aicore/aicore.h"
#include "aicore/aicore_profiling_state.h"
#include "runtime.h"

namespace {
constexpr uint32_t kWorkers = 6;
std::atomic<uint32_t> calls{0};
std::atomic<uint32_t> errors{0};
uint64_t expected_epoch;
uint64_t register_addresses[kWorkers];
}  // namespace

// The wrapper reads only worker_count; this fixture uses the real Runtime type
// without bringing an orchestrator or scheduler into the loadable module.
Runtime::Runtime() = default;

void aicore_execute(Runtime *, int block_idx, CoreType) {
    if (get_aicore_report_epoch() != expected_epoch || sim_get_physical_core_id() != static_cast<uint32_t>(block_idx) ||
        reinterpret_cast<uintptr_t>(sim_get_reg_base()) != register_addresses[block_idx] ||
        get_aicore_pmu_reg_base() != register_addresses[block_idx] || get_aicore_profiling_flag() != 0 ||
        get_chip_swimlane_aicore_head() != nullptr || get_aicore_pmu_ring() != nullptr ||
        platform_ssbuf_hardware_base_address() != UINT64_C(0x8000) + static_cast<uint64_t>(block_idx) * 128)
        errors.fetch_add(1, std::memory_order_relaxed);
    calls.fetch_add(1, std::memory_order_relaxed);
}

extern "C" void
aicore_execute_wrapper(Runtime *, int, CoreType, uint32_t, uint64_t, uint32_t, uint64_t, uint64_t, uint64_t, uint64_t);

extern "C" uint64_t run_sim_tls_wrapper_probe(uint64_t epoch, pthread_key_t protected_key) {
    Runtime runtime;
    runtime.dev.worker_count = kWorkers;
    expected_epoch = epoch;
    calls.store(0);
    errors.store(0);
    for (uint32_t worker = 0; worker < kWorkers; ++worker)
        register_addresses[worker] = UINT64_C(0x10000) + worker * 4096;
    std::vector<std::thread> workers;
    for (uint32_t worker = 0; worker < kWorkers; ++worker) {
        workers.emplace_back([&, worker] {
            void *sentinel = reinterpret_cast<void *>(UINT64_C(0x12345678));
            if (pthread_setspecific(protected_key, sentinel) != 0) errors.fetch_add(1);
            aicore_execute_wrapper(
                &runtime, worker, worker < 2 ? CoreType::AIC : CoreType::AIV, worker,
                reinterpret_cast<uint64_t>(register_addresses), 0, 0, 0, epoch, UINT64_C(0x8000) + worker * 128
            );
            if (pthread_getspecific(protected_key) != sentinel) errors.fetch_add(1);
        });
    }
    for (auto &worker : workers)
        worker.join();
    return static_cast<uint64_t>(calls.load()) << 32 | errors.load();
}
