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
 * Sets up per-thread simulated register base and core identity before calling
 * the executor.
 */

#include <cstdint>
#include <pthread.h>

#include "inner_kernel.h"
#include "aicore/aicore.h"
#include "aicore/aicore_profiling_state.h"
#include "common/core_type.h"
#include "common/l2_perf_profiling.h"
#include "common/platform_config.h"
#include "runtime.h"

// Per-thread simulated register state — use pthread TLS instead of C++
// thread_local to avoid glibc TLSDESC issues when the AICore SO is loaded
// with RTLD_LOCAL on aarch64.
static pthread_key_t g_reg_base_key;
static pthread_key_t g_core_id_key;
static pthread_key_t g_aicore_profiling_flag_key;
static pthread_key_t g_aicore_l2_perf_ring_key;
static pthread_once_t g_tls_once = PTHREAD_ONCE_INIT;

static void create_tls_keys() {
    pthread_key_create(&g_reg_base_key, nullptr);
    pthread_key_create(&g_core_id_key, nullptr);
    pthread_key_create(&g_aicore_profiling_flag_key, nullptr);
    pthread_key_create(&g_aicore_l2_perf_ring_key, nullptr);
}

volatile uint8_t *sim_get_reg_base() { return static_cast<volatile uint8_t *>(pthread_getspecific(g_reg_base_key)); }

uint32_t sim_get_physical_core_id() {
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(pthread_getspecific(g_core_id_key)));
}

// Per-core profiling state setters/getters. Same contract as the onboard
// [[block_local]] backing in src/a2a3/platform/onboard/aicore/kernel.cpp —
// AICPU never reaches into the runtime's Handshake for profiling, runtime's
// aicore_execute keeps its original signature, and adding profiling fields
// only touches KernelArgs + this state surface.
__aicore__ void set_aicore_profiling_flag(uint32_t flag) {
    pthread_setspecific(g_aicore_profiling_flag_key, reinterpret_cast<void *>(static_cast<uintptr_t>(flag)));
}
__aicore__ uint32_t get_aicore_profiling_flag() {
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(pthread_getspecific(g_aicore_profiling_flag_key)));
}

__aicore__ void set_aicore_l2_perf_ring(__gm__ L2PerfAicoreRing *ring) {
    pthread_setspecific(g_aicore_l2_perf_ring_key, reinterpret_cast<void *>(ring));
}
__aicore__ __gm__ L2PerfAicoreRing *get_aicore_l2_perf_ring() {
    return reinterpret_cast<__gm__ L2PerfAicoreRing *>(pthread_getspecific(g_aicore_l2_perf_ring_key));
}

// Core identity setter function pointers — set by DeviceRunner after dlopen.
// These point into cpu_sim_context.cpp (host_runtime SO) and set per-thread
// subblock_id and cluster_id for pto-isa's TPUSH/TPOP hooks.
using SimSetUint32Fn = void (*)(uint32_t);

static SimSetUint32Fn g_set_subblock_id_fn = nullptr;
static SimSetUint32Fn g_set_cluster_id_fn = nullptr;

extern "C" void set_sim_core_identity_helpers(void *set_subblock_id, void *set_cluster_id) {
    g_set_subblock_id_fn = reinterpret_cast<SimSetUint32Fn>(set_subblock_id);
    g_set_cluster_id_fn = reinterpret_cast<SimSetUint32Fn>(set_cluster_id);
}

// Declare the original function (defined in aicore_executor.cpp with weak linkage)
void aicore_execute(__gm__ Runtime *runtime, int block_idx, CoreType core_type);

// Wrapper with extern "C" for dlsym lookup. Mirrors the onboard kernel.cpp
// KERNEL_ENTRY: receives launch arguments straight from the host, populates
// the per-thread profiling slots via set_aicore_*(), then invokes the runtime
// executor with its original signature.
extern "C" void aicore_execute_wrapper(
    __gm__ Runtime *runtime, int block_idx, CoreType core_type, uint32_t physical_core_id, uint64_t regs,
    uint32_t enable_profiling_flag, uint64_t aicore_ring_addr
) {
    pthread_once(&g_tls_once, create_tls_keys);

    // Set up simulated register base for this thread.
    // regs points to an array of uint64_t base addresses (one per core).
    // physical_core_id indexes into it to get this core's register block.
    if (regs != 0) {
        uint64_t *regs_array = reinterpret_cast<uint64_t *>(regs);
        pthread_setspecific(g_reg_base_key, reinterpret_cast<void *>(regs_array[physical_core_id]));
    }

    pthread_setspecific(g_core_id_key, reinterpret_cast<void *>(static_cast<uintptr_t>(physical_core_id)));

    // Publish per-core profiling state before the executor runs.
    set_aicore_profiling_flag(enable_profiling_flag);
    if (aicore_ring_addr != 0) {
        uint64_t *ring_table = reinterpret_cast<uint64_t *>(aicore_ring_addr);
        set_aicore_l2_perf_ring(reinterpret_cast<__gm__ L2PerfAicoreRing *>(ring_table[block_idx]));
    } else {
        set_aicore_l2_perf_ring(nullptr);
    }

    // Set core identity for pto-isa TPUSH/TPOP simulation.
    // Core layout in sim DeviceRunner:
    //   physical_core_id [0..block_dim-1]              = AIC of cluster i
    //   physical_core_id [block_dim..3*block_dim-1]    = AIV pairs
    //                                                    (AIV0_c0, AIV1_c0, AIV0_c1, AIV1_c1, ...)
    // This mirrors the runtime's CoreTracker cluster assignment.
    uint32_t block_dim = static_cast<uint32_t>(runtime->worker_count) / PLATFORM_CORES_PER_BLOCKDIM;
    uint32_t cluster_id;
    uint32_t subblock_id;
    if (core_type == CoreType::AIC) {
        cluster_id = physical_core_id;
        subblock_id = 0;
    } else {
        uint32_t aiv_idx = physical_core_id - block_dim;
        cluster_id = aiv_idx / 2;
        subblock_id = aiv_idx % 2;
    }

    if (g_set_subblock_id_fn != nullptr) {
        g_set_subblock_id_fn(subblock_id);
    }
    if (g_set_cluster_id_fn != nullptr) {
        g_set_cluster_id_fn(cluster_id);
    }

    aicore_execute(runtime, block_idx, core_type);
}
