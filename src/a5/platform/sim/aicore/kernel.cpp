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
#include <cstdio>
#include <cstdlib>
#include <pthread.h>

#include "inner_kernel.h"
#include "aicore/aicore.h"
#include "aicore/aicore_profiling_state.h"
#include "common/core_type.h"
#include "common/l2_swimlane_profiling.h"
#include "common/platform_config.h"
#include "common/pmu_profiling.h"
#include "runtime.h"

// Per-thread simulated register state — use pthread TLS instead of C++
// thread_local to avoid glibc TLSDESC issues when the AICore SO is loaded
// with RTLD_LOCAL on aarch64.
static pthread_key_t g_reg_base_key;
static pthread_key_t g_core_id_key;
static pthread_key_t g_block_idx_key;
static pthread_key_t g_aicore_profiling_flag_key;
// Slot pointer (NOT the dereferenced head address) — see
// aicore_profiling_state.h for the lazy-deref contract.
static pthread_key_t g_l2_swimlane_aicore_head_slot_key;
static pthread_key_t g_l2_swimlane_aicore_head_key;
static pthread_key_t g_aicore_pmu_ring_key;
static pthread_key_t g_pmu_reg_base_key;
static pthread_once_t g_tls_once = PTHREAD_ONCE_INIT;
// True once create_tls_keys() has successfully created ALL keys; gates the
// unload-time delete so we never pthread_key_delete a stale/uncreated key.
static bool g_tls_keys_ready = false;

// All pthread keys owned by this DSO, in creation order. destroy_tls_keys()
// rolls these back at unload so a per-run dlopen/dlclose cycle is net-zero on
// the process-wide TLS key pool (see destroy_tls_keys()).
static pthread_key_t *const g_all_keys[] = {
    &g_reg_base_key,
    &g_core_id_key,
    &g_block_idx_key,
    &g_aicore_profiling_flag_key,
    &g_l2_swimlane_aicore_head_slot_key,
    &g_l2_swimlane_aicore_head_key,
    &g_aicore_pmu_ring_key,
    &g_pmu_reg_base_key,
};
constexpr int kNumTlsKeys = sizeof(g_all_keys) / sizeof(g_all_keys[0]);

static void create_tls_keys() {
    for (int i = 0; i < kNumTlsKeys; i++) {
        if (pthread_key_create(g_all_keys[i], nullptr) != 0) {
            // The process-wide pthread key pool (PTHREAD_KEYS_MAX, 1024) is
            // exhausted. Roll back what we created and fail loudly: silently
            // leaving a key at 0 makes sim_get_reg_base() return NULL and
            // crashes write_reg() on a NULL register base (hard-to-debug
            // SIGSEGV). With destroy_tls_keys() reclaiming keys on unload this
            // path should never be hit.
            for (int j = 0; j < i; j++) pthread_key_delete(*g_all_keys[j]);
            fprintf(stderr, "[aicore_sim] FATAL: pthread_key_create failed at key %d/%d — TLS key pool exhausted\n", i,
                    kNumTlsKeys);
            abort();
        }
    }
    g_tls_keys_ready = true;
}

// Release this DSO's pthread TLS keys when it is unloaded (dlclose). The AICore
// kernel .so is dlopen/dlclose'd once per run (device_runner.cpp reloads it
// because the kernel binary can vary per case), and glibc does NOT reclaim a
// DSO's pthread keys on unload. Without this, every run leaked these keys and
// after ~PTHREAD_KEYS_MAX/kNumTlsKeys runs pthread_key_create() began failing
// (EAGAIN), leaving the keys at 0 → sim_get_reg_base() == NULL → write_reg()
// NULL-deref SIGSEGV mid-sweep. All AICore worker threads are joined before the
// DSO is dlclose'd, so deleting the keys here is race-free.
__attribute__((destructor)) static void destroy_tls_keys() {
    if (!g_tls_keys_ready) return;
    for (int i = 0; i < kNumTlsKeys; i++) pthread_key_delete(*g_all_keys[i]);
    g_tls_keys_ready = false;
}

volatile uint8_t *sim_get_reg_base() { return static_cast<volatile uint8_t *>(pthread_getspecific(g_reg_base_key)); }

uint32_t sim_get_physical_core_id() {
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(pthread_getspecific(g_core_id_key)));
}

// Per-core profiling state setters/getters. Same contract as the onboard
// [[block_local]] backing in src/a5/platform/onboard/aicore/kernel.cpp —
// AICore never reaches into the runtime's Handshake for profiling, runtime's
// aicore_execute keeps its original signature, and adding profiling fields
// only touches KernelArgs + this state surface.
__aicore__ void set_aicore_profiling_flag(uint32_t flag) {
    pthread_setspecific(g_aicore_profiling_flag_key, reinterpret_cast<void *>(static_cast<uintptr_t>(flag)));
}
__aicore__ uint32_t get_aicore_profiling_flag() {
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(pthread_getspecific(g_aicore_profiling_flag_key)));
}

__aicore__ void set_l2_swimlane_aicore_head_slot(__gm__ uint64_t *slot_ptr) {
    pthread_setspecific(g_l2_swimlane_aicore_head_slot_key, reinterpret_cast<void *>(slot_ptr));
    pthread_setspecific(g_l2_swimlane_aicore_head_key, nullptr);  // force lazy resolve on next get
}
__aicore__ __gm__ L2SwimlaneActiveHead *get_l2_swimlane_aicore_head() {
    auto *cached = reinterpret_cast<__gm__ L2SwimlaneActiveHead *>(pthread_getspecific(g_l2_swimlane_aicore_head_key));
    if (cached != nullptr) return cached;
    auto *slot = reinterpret_cast<__gm__ uint64_t *>(pthread_getspecific(g_l2_swimlane_aicore_head_slot_key));
    if (slot == nullptr) return nullptr;
    // Lazy first-call resolve — see aicore_profiling_state.h.
    cached = reinterpret_cast<__gm__ L2SwimlaneActiveHead *>(*slot);
    pthread_setspecific(g_l2_swimlane_aicore_head_key, reinterpret_cast<void *>(cached));
    return cached;
}

__aicore__ void set_aicore_pmu_ring(__gm__ PmuAicoreRing *ring) {
    pthread_setspecific(g_aicore_pmu_ring_key, reinterpret_cast<void *>(ring));
}
__aicore__ __gm__ PmuAicoreRing *get_aicore_pmu_ring() {
    return reinterpret_cast<__gm__ PmuAicoreRing *>(pthread_getspecific(g_aicore_pmu_ring_key));
}

__aicore__ void set_aicore_pmu_reg_base(uint64_t reg_base) {
    pthread_setspecific(g_pmu_reg_base_key, reinterpret_cast<void *>(static_cast<uintptr_t>(reg_base)));
}
__aicore__ uint64_t get_aicore_pmu_reg_base() {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pthread_getspecific(g_pmu_reg_base_key)));
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
    uint32_t enable_profiling_flag, uint64_t l2_swimlane_aicore_rotation_table, uint64_t aicore_pmu_ring_addrs
) {
    pthread_once(&g_tls_once, create_tls_keys);

    // Set up simulated register base for this thread.
    // regs points to an array of uint64_t base addresses (one per core).
    // physical_core_id indexes into it to get this core's register block.
    uint64_t this_core_reg_base = 0;
    if (regs != 0) {
        uint64_t *regs_array = reinterpret_cast<uint64_t *>(regs);
        this_core_reg_base = regs_array[physical_core_id];
        pthread_setspecific(g_reg_base_key, reinterpret_cast<void *>(this_core_reg_base));
    }

    pthread_setspecific(g_core_id_key, reinterpret_cast<void *>(static_cast<uintptr_t>(physical_core_id)));
    pthread_setspecific(g_block_idx_key, reinterpret_cast<void *>(static_cast<uintptr_t>(block_idx)));

    // Publish per-core profiling state before the executor runs.
    set_aicore_profiling_flag(enable_profiling_flag);
    if ((enable_profiling_flag & PROFILING_FLAG_L2_SWIMLANE) && l2_swimlane_aicore_rotation_table != 0) {
        // Stash only the slot pointer; deref happens lazily inside
        // get_l2_swimlane_aicore_head() once AICPU has populated the table. See
        // aicore_profiling_state.h.
        uint64_t *head_table = reinterpret_cast<uint64_t *>(l2_swimlane_aicore_rotation_table);
        set_l2_swimlane_aicore_head_slot(reinterpret_cast<__gm__ uint64_t *>(&head_table[block_idx]));
    } else {
        set_l2_swimlane_aicore_head_slot(nullptr);
    }
    if ((enable_profiling_flag & PROFILING_FLAG_PMU) && aicore_pmu_ring_addrs != 0) {
        uint64_t *pmu_ring_table = reinterpret_cast<uint64_t *>(aicore_pmu_ring_addrs);
        set_aicore_pmu_ring(reinterpret_cast<__gm__ PmuAicoreRing *>(pmu_ring_table[block_idx]));
    } else {
        set_aicore_pmu_ring(nullptr);
    }
    set_aicore_pmu_reg_base(this_core_reg_base);

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
