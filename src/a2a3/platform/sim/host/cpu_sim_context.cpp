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
 * @file cpu_sim_context.cpp
 * @brief CPU simulation context for CANN intrinsic emulation
 *
 * Provides per-thread execution context (block_idx, subblock_id, subblock_dim)
 * and task cookie storage backed by pthread keys for AICore kernels running in
 * host-based simulation.
 * Also provides shared storage for cross-core data exchange in simulation.
 *
 * Functions are exported with extern "C" linkage so that AICore kernel .so files
 * can resolve them via dlsym(RTLD_DEFAULT, ...) at runtime.
 */

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <pthread.h>
#include <string>

#include "cpu_sim_context.h"

namespace {
struct CpuSimExecutionContext {
    uint32_t block_idx = 0;
    uint32_t subblock_id = 0;
    uint32_t subblock_dim = 1;
    uint64_t task_cookie = 0;
};

void free_cpu_sim_execution_context(void *ptr) { std::free(ptr); }

// Use pthread TLS instead of C++ thread_local so the dlclose'd host_runtime.so
// does not rely on ELF TLS descriptors across reloads.
std::mutex g_cpu_sim_context_key_mutex;
pthread_key_t g_cpu_sim_context_key{};
std::atomic<bool> g_cpu_sim_context_key_initialized{false};

CpuSimExecutionContext *get_cpu_sim_execution_context() {
    if (!g_cpu_sim_context_key_initialized.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(g_cpu_sim_context_key_mutex);
        if (!g_cpu_sim_context_key_initialized.load(std::memory_order_relaxed)) {
            if (pthread_key_create(&g_cpu_sim_context_key, free_cpu_sim_execution_context) != 0) {
                return nullptr;
            }
            g_cpu_sim_context_key_initialized.store(true, std::memory_order_release);
        }
    }

    auto *context = static_cast<CpuSimExecutionContext *>(pthread_getspecific(g_cpu_sim_context_key));
    if (context != nullptr) {
        return context;
    }

    context = static_cast<CpuSimExecutionContext *>(std::calloc(1, sizeof(CpuSimExecutionContext)));
    if (context == nullptr) {
        return nullptr;
    }
    context->subblock_dim = 1;

    if (pthread_setspecific(g_cpu_sim_context_key, context) != 0) {
        std::free(context);
        return nullptr;
    }

    return context;
}

void reset_cpu_sim_execution_context_key() {
    if (!g_cpu_sim_context_key_initialized.load(std::memory_order_acquire)) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_cpu_sim_context_key_mutex);
    if (!g_cpu_sim_context_key_initialized.load(std::memory_order_relaxed)) {
        return;
    }

    void *current_context = pthread_getspecific(g_cpu_sim_context_key);
    if (current_context != nullptr) {
        std::free(current_context);
        (void)pthread_setspecific(g_cpu_sim_context_key, nullptr);
    }

    (void)pthread_key_delete(g_cpu_sim_context_key);
    g_cpu_sim_context_key_initialized.store(false, std::memory_order_release);
}

std::mutex g_cpu_sim_shared_storage_mutex;
std::map<std::string, void *> g_cpu_sim_shared_storage;
std::mutex g_cpu_sim_task_cookie_mutex;
std::map<uint64_t, uint64_t> g_cpu_sim_task_cookies;

uint64_t make_task_cookie_key(uint32_t core_id, uint32_t reg_task_id) {
    return (static_cast<uint64_t>(core_id) << 32) | static_cast<uint64_t>(reg_task_id);
}
}  // namespace

void clear_cpu_sim_shared_storage() {
    reset_cpu_sim_execution_context_key();

    {
        std::lock_guard<std::mutex> lock(g_cpu_sim_task_cookie_mutex);
        g_cpu_sim_task_cookies.clear();
    }

    std::lock_guard<std::mutex> lock(g_cpu_sim_shared_storage_mutex);
    for (auto &[key, storage] : g_cpu_sim_shared_storage) {
        (void)key;
        std::free(storage);
    }
    g_cpu_sim_shared_storage.clear();
}

extern "C" void pto_cpu_sim_set_execution_context(uint32_t block_idx, uint32_t subblock_id, uint32_t subblock_dim) {
    auto *context = get_cpu_sim_execution_context();
    if (context == nullptr) {
        return;
    }

    context->block_idx = block_idx;
    context->subblock_id = subblock_id;
    context->subblock_dim = (subblock_dim == 0) ? 1u : subblock_dim;
}

extern "C" void pto_cpu_sim_set_task_cookie(uint64_t task_cookie) {
    auto *context = get_cpu_sim_execution_context();
    if (context == nullptr) {
        return;
    }

    context->task_cookie = task_cookie;
}

extern "C" void pto_cpu_sim_get_execution_context(uint32_t *block_idx, uint32_t *subblock_id, uint32_t *subblock_dim) {
    auto *context = get_cpu_sim_execution_context();
    uint32_t current_block_idx = 0;
    uint32_t current_subblock_id = 0;
    uint32_t current_subblock_dim = 1;
    if (context != nullptr) {
        current_block_idx = context->block_idx;
        current_subblock_id = context->subblock_id;
        current_subblock_dim = context->subblock_dim;
    }

    if (block_idx != nullptr) {
        *block_idx = current_block_idx;
    }
    if (subblock_id != nullptr) {
        *subblock_id = current_subblock_id;
    }
    if (subblock_dim != nullptr) {
        *subblock_dim = current_subblock_dim;
    }
}

extern "C" uint64_t pto_cpu_sim_get_task_cookie() {
    auto *context = get_cpu_sim_execution_context();
    return (context != nullptr) ? context->task_cookie : 0;
}

extern "C" void platform_set_cpu_sim_task_cookie(uint32_t core_id, uint32_t reg_task_id, uint64_t task_cookie) {
    std::lock_guard<std::mutex> lock(g_cpu_sim_task_cookie_mutex);
    g_cpu_sim_task_cookies[make_task_cookie_key(core_id, reg_task_id)] = task_cookie;
}

extern "C" uint64_t platform_get_cpu_sim_task_cookie(uint32_t core_id, uint32_t reg_task_id) {
    std::lock_guard<std::mutex> lock(g_cpu_sim_task_cookie_mutex);
    uint64_t key = make_task_cookie_key(core_id, reg_task_id);
    auto it = g_cpu_sim_task_cookies.find(key);
    if (it == g_cpu_sim_task_cookies.end()) {
        return 0;
    }

    uint64_t task_cookie = it->second;
    g_cpu_sim_task_cookies.erase(it);
    return task_cookie;
}

extern "C" void *pto_cpu_sim_get_shared_storage(const char *key, size_t size) {
    if (key == nullptr || size == 0) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(g_cpu_sim_shared_storage_mutex);
    auto it = g_cpu_sim_shared_storage.find(key);
    if (it != g_cpu_sim_shared_storage.end()) {
        return it->second;
    }

    void *storage = std::calloc(1, size);
    g_cpu_sim_shared_storage.emplace(key, storage);
    return storage;
}
