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

#include <cstddef>
#include <cstdint>
#include "aicore/sim_core_abi.h"
#include <pthread.h>

#include "inner_kernel.h"
#include "aicore/sim_tls_state.h"
#include "aicore/aicore.h"
#include "aicore/aicore_profiling_state.h"
#include "common/core_type.h"
#include "common/chip_swimlane_profiling.h"
#include "common/platform_config.h"
#include "runtime.h"

// Per-thread simulated register state — use pthread TLS instead of C++
// thread_local to avoid glibc TLSDESC issues when the AICore SO is loaded
// with RTLD_LOCAL on aarch64.
static pthread_key_t g_reg_base_key;
static pthread_key_t g_core_id_key;
static pthread_key_t g_aicore_profiling_flag_key;
// Slot pointer (NOT the dereferenced head address) — see
// aicore_profiling_state.h for the lazy-deref contract.
static pthread_key_t g_chip_swimlane_aicore_head_slot_key;
static pthread_key_t g_chip_swimlane_aicore_head_key;
static pthread_key_t g_aicore_report_epoch_key;
static pthread_once_t g_tls_once = PTHREAD_ONCE_INIT;
static pthread_key_t *const g_tls_keys[] = {
    &g_reg_base_key,
    &g_core_id_key,
    &g_aicore_profiling_flag_key,
    &g_chip_swimlane_aicore_head_slot_key,
    &g_chip_swimlane_aicore_head_key,
    &g_aicore_report_epoch_key,
};
static size_t g_tls_key_count = 0;
static int g_tls_init_result = 0;

static void release_tls_keys() {
    while (g_tls_key_count != 0)
        pthread_key_delete(*g_tls_keys[--g_tls_key_count]);
}

// DeviceRunner joins every AICore thread before unloading its per-run SO.
// POSIX keys belong to the process and need explicit deletion at that boundary.
namespace {
struct SimTlsKeyOwner {
    ~SimTlsKeyOwner() { release_tls_keys(); }
};
SimTlsKeyOwner g_tls_key_owner;
}  // namespace

static void create_tls_keys() {
    for (pthread_key_t *key : g_tls_keys) {
        g_tls_init_result = pthread_key_create(key, nullptr);
        if (g_tls_init_result != 0) {
            release_tls_keys();
            return;
        }
        ++g_tls_key_count;
    }
}

extern "C" int initialize_aicore_sim_tls() {
    const int result = pthread_once(&g_tls_once, create_tls_keys);
    return result != 0 ? result : g_tls_init_result;
}

volatile uint8_t *sim_get_reg_base() { return static_cast<volatile uint8_t *>(sim_tls_value(g_reg_base_key)); }

uint32_t sim_get_physical_core_id() {
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(sim_tls_value(g_core_id_key)));
}

// Per-core profiling state setters/getters. Same contract as the onboard
// [[block_local]] backing in src/a2a3/platform/onboard/aicore/kernel.cpp —
// AICPU never reaches into the runtime's Handshake for profiling, runtime's
// aicore_execute keeps its original signature, and adding profiling fields
// only touches KernelArgs + this state surface.
__aicore__ void set_aicore_profiling_flag(uint32_t flag) {
    sim_set_tls_value(g_aicore_profiling_flag_key, reinterpret_cast<void *>(static_cast<uintptr_t>(flag)));
}
__aicore__ uint32_t get_aicore_profiling_flag() {
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(sim_tls_value(g_aicore_profiling_flag_key)));
}

__aicore__ void set_aicore_report_epoch(uint64_t epoch) {
    sim_set_tls_value(g_aicore_report_epoch_key, reinterpret_cast<void *>(static_cast<uintptr_t>(epoch)));
}
__aicore__ uint64_t get_aicore_report_epoch() {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(sim_tls_value(g_aicore_report_epoch_key)));
}

__aicore__ void set_chip_swimlane_aicore_head_slot(__gm__ uint64_t *slot_ptr) {
    sim_set_tls_value(g_chip_swimlane_aicore_head_slot_key, reinterpret_cast<void *>(slot_ptr));
    sim_set_tls_value(g_chip_swimlane_aicore_head_key, nullptr);  // force lazy resolve on next get
}
__aicore__ __gm__ ChipSwimlaneActiveHead *get_chip_swimlane_aicore_head() {
    auto *cached = reinterpret_cast<__gm__ ChipSwimlaneActiveHead *>(sim_tls_value(g_chip_swimlane_aicore_head_key));
    if (cached != nullptr) return cached;
    auto *slot = reinterpret_cast<__gm__ uint64_t *>(sim_tls_value(g_chip_swimlane_aicore_head_slot_key));
    if (slot == nullptr) return nullptr;
    // Lazy first-call resolve — see aicore_profiling_state.h.
    cached = reinterpret_cast<__gm__ ChipSwimlaneActiveHead *>(*slot);
    sim_set_tls_value(g_chip_swimlane_aicore_head_key, reinterpret_cast<void *>(cached));
    return cached;
}

// Core identity setter function pointers — set by DeviceRunner after dlopen.
// These point into cpu_sim_context.cpp (host_runtime SO) and set per-thread
// subblock_id and cluster_id for pto-isa's TPUSH/TPOP hooks.

static SimIdentitySetFn g_set_subblock_id_fn = nullptr;
static SimIdentitySetFn g_set_cluster_id_fn = nullptr;

extern "C" void sim_install_core_identity_setters(SimIdentitySetFn set_subblock_id, SimIdentitySetFn set_cluster_id) {
    g_set_subblock_id_fn = set_subblock_id;
    g_set_cluster_id_fn = set_cluster_id;
}

// Declare the original function (defined in aicore_executor.cpp with weak linkage)
void aicore_execute(__gm__ Runtime *runtime, int block_idx, CoreType core_type);

// Wrapper with extern "C" for dlsym lookup. Mirrors the onboard kernel.cpp
// KERNEL_ENTRY: receives launch arguments straight from the host, populates
// the per-thread profiling slots via set_aicore_*(), then invokes the runtime
// executor with its original signature.
extern "C" int aicore_execute_wrapper_checked(
    __gm__ Runtime *runtime, int block_idx, CoreType core_type, uint32_t physical_core_id, uint64_t regs,
    uint32_t enable_profiling_flag, uint64_t chip_swimlane_aicore_rotation_table, uint64_t report_epoch,
    SimAdmissionReportFn ready, void *startup
) {
    SimAicoreTlsBinding state(g_tls_keys);
    int error = initialize_aicore_sim_tls();
    if (error == 0) error = state.bind();
    if (error != 0) {
        ready(startup, error);
        return error;
    }

    // Set up simulated register base for this thread.
    // regs points to an array of uint64_t base addresses (one per core).
    // physical_core_id indexes into it to get this core's register block.
    if (regs != 0) {
        uint64_t *regs_array = reinterpret_cast<uint64_t *>(regs);
        sim_set_tls_value(g_reg_base_key, reinterpret_cast<void *>(regs_array[physical_core_id]));
    }

    sim_set_tls_value(g_core_id_key, reinterpret_cast<void *>(static_cast<uintptr_t>(physical_core_id)));

    // Publish per-core profiling state before the executor runs.
    set_aicore_profiling_flag(enable_profiling_flag);
    set_aicore_report_epoch(report_epoch);
    if (chip_swimlane_aicore_rotation_table != 0) {
        // Stash only the slot pointer; the executor dereferences it via
        // get_chip_swimlane_aicore_head() after observing Phase 2 window-open.
        // See aicore_profiling_state.h.
        uint64_t *head_table = reinterpret_cast<uint64_t *>(chip_swimlane_aicore_rotation_table);
        set_chip_swimlane_aicore_head_slot(reinterpret_cast<__gm__ uint64_t *>(&head_table[block_idx]));
    } else {
        set_chip_swimlane_aicore_head_slot(nullptr);
    }

    // Set core identity for pto-isa TPUSH/TPOP simulation.
    // Core layout in sim DeviceRunner:
    //   physical_core_id [0..block_dim-1]              = AIC of cluster i
    //   physical_core_id [block_dim..3*block_dim-1]    = AIV pairs
    //                                                    (AIV0_c0, AIV1_c0, AIV0_c1, AIV1_c1, ...)
    // This mirrors the runtime's CoreTracker cluster assignment.
    uint32_t block_dim = static_cast<uint32_t>(runtime->get_worker_count()) / PLATFORM_CORES_PER_BLOCKDIM;
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
        error = g_set_subblock_id_fn(subblock_id);
    }
    if (error == 0 && g_set_cluster_id_fn != nullptr) {
        error = g_set_cluster_id_fn(cluster_id);
    }

    const bool admitted = ready(startup, error);
    if (error == 0 && admitted) aicore_execute(runtime, block_idx, core_type);
    const int clear_error = state.clear();
    return error != 0 ? error : clear_error;
}
