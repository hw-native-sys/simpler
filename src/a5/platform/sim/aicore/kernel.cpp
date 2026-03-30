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
 * AICore Kernel Wrapper for Simulation
 *
 * Provides a wrapper around aicore_execute for dlsym lookup.
 * Sets up per-thread simulated register base before calling the executor.
 */

#include <dlfcn.h>

#include <cstdint>

#include "aicore/aicore.h"
#include "common/core_type.h"
#include "common/platform_config.h"
#include "runtime.h"

// Thread-local simulated register base (declared in inner_kernel.h)
thread_local volatile uint8_t* g_sim_reg_base = nullptr;

// Thread-local simulated physical core ID (declared in inner_kernel.h)
thread_local uint32_t g_sim_physical_core_id = 0;

// Declare the original function (defined in aicore_executor.cpp with weak linkage)
void aicore_execute(__gm__ Runtime* runtime, int block_idx, CoreType core_type);

namespace {
using CpuSimSetExecutionContextHook = void (*)(uint32_t, uint32_t, uint32_t);

CpuSimSetExecutionContextHook resolve_cpu_sim_set_execution_context_hook() {
    static auto hook =
        reinterpret_cast<CpuSimSetExecutionContextHook>(dlsym(RTLD_DEFAULT, "pto_cpu_sim_set_execution_context"));
    return hook;
}
}  // namespace

// Wrapper with extern "C" for dlsym lookup
// NOTE: physical_core_id stays in wrapper signature (DeviceRunner passes it for register indexing)
extern "C" void aicore_execute_wrapper(
    __gm__ Runtime* runtime, int block_idx, CoreType core_type, uint32_t physical_core_id, uint64_t regs) {
    // Set up simulated register base for this thread.
    // regs points to an array of uint64_t base addresses (one per core).
    // physical_core_id indexes into it to get this core's register block.
    if (regs != 0) {
        uint64_t* regs_array = reinterpret_cast<uint64_t*>(regs);
        g_sim_reg_base = reinterpret_cast<volatile uint8_t*>(regs_array[physical_core_id]);
    }

    g_sim_physical_core_id = physical_core_id;
    const uint32_t num_aic = static_cast<uint32_t>(runtime->worker_count / PLATFORM_CORES_PER_BLOCKDIM);
    uint32_t cpu_block_idx = static_cast<uint32_t>(block_idx);
    uint32_t subblock_id = 0;
    uint32_t subblock_dim = 1;

    if (core_type == CoreType::AIV && physical_core_id >= num_aic) {
        const uint32_t aiv_offset = physical_core_id - num_aic;
        cpu_block_idx = aiv_offset / PLATFORM_AIV_CORES_PER_BLOCKDIM;
        subblock_id = aiv_offset % PLATFORM_AIV_CORES_PER_BLOCKDIM;
        subblock_dim = PLATFORM_AIV_CORES_PER_BLOCKDIM;
    } else {
        cpu_block_idx = physical_core_id;
    }

    if (auto hook = resolve_cpu_sim_set_execution_context_hook(); hook != nullptr) {
        hook(cpu_block_idx, subblock_id, subblock_dim);
    }
    aicore_execute(runtime, block_idx, core_type);
}
