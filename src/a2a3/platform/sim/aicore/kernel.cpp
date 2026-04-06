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

#include <cstdint>
#include <pthread.h>

#include "inner_kernel.h"  // NOLINT(build/include_subdir)
#include "aicore/aicore.h"
#include "common/core_type.h"
#include "common/platform_config.h"
#include "runtime.h"

// Per-thread simulated register state — use pthread TLS instead of C++
// thread_local to avoid glibc TLSDESC issues when the AICore SO is loaded
// with RTLD_LOCAL on aarch64.
static pthread_key_t g_reg_base_key;
static pthread_key_t g_core_id_key;
static pthread_once_t g_tls_once = PTHREAD_ONCE_INIT;

static void create_tls_keys() {
    pthread_key_create(&g_reg_base_key, nullptr);
    pthread_key_create(&g_core_id_key, nullptr);
}

volatile uint8_t *sim_get_reg_base() { return static_cast<volatile uint8_t *>(pthread_getspecific(g_reg_base_key)); }

uint32_t sim_get_physical_core_id() {
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(pthread_getspecific(g_core_id_key)));
}

// Sim context function pointers — set by DeviceRunner after dlopen.
SimSetExecCtxFn g_sim_set_exec_ctx_fn = nullptr;
SimSetTaskCookieFn g_sim_set_task_cookie_fn = nullptr;
SimGetTaskCookieFn g_sim_get_task_cookie_fn = nullptr;

extern "C" void set_sim_context_helpers(void *set_exec_ctx, void *set_task_cookie, void *get_task_cookie) {
    g_sim_set_exec_ctx_fn = reinterpret_cast<SimSetExecCtxFn>(set_exec_ctx);
    g_sim_set_task_cookie_fn = reinterpret_cast<SimSetTaskCookieFn>(set_task_cookie);
    g_sim_get_task_cookie_fn = reinterpret_cast<SimGetTaskCookieFn>(get_task_cookie);
}

// Declare the original function (defined in aicore_executor.cpp with weak linkage)
void aicore_execute(__gm__ Runtime *runtime, int block_idx, CoreType core_type);

// Wrapper with extern "C" for dlsym lookup
// NOTE: physical_core_id stays in wrapper signature (DeviceRunner passes it for register indexing)
extern "C" void aicore_execute_wrapper(
    __gm__ Runtime *runtime, int block_idx, CoreType core_type, uint32_t physical_core_id, uint64_t regs
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

    aicore_execute(runtime, block_idx, core_type);
}
