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
 * @brief Per-device CPU simulation context
 *
 * Provides per-thread execution context (block_idx, subblock_id, subblock_dim)
 * and per-device shared storage / task cookie maps.
 *
 * Each simulated device has an independent DeviceSimContext so that multiple
 * ChipWorkers (each simulating a different device) can run concurrently.
 *
 * The current device is bound to each thread via a pthread key set in
 * pto_cpu_sim_bind_device(). All pto_cpu_sim_* functions route through
 * this binding to find the correct DeviceSimContext.
 *
 * Functions are exported with extern "C" linkage so that AICore kernel SOs
 * can resolve them via dlsym(RTLD_DEFAULT, ...) at runtime.
 */

#include "cpu_sim_context.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <pthread.h>
#include <string>
#include <unordered_map>

namespace {

// ---------------------------------------------------------------------------
// Per-device context
// ---------------------------------------------------------------------------

struct DeviceSimContext {
    std::mutex shared_storage_mutex;
    std::map<std::string, void *> shared_storage;

    std::mutex task_cookie_mutex;
    std::map<uint64_t, uint64_t> task_cookies;
};

std::mutex g_registry_mutex;
std::unordered_map<int, DeviceSimContext *> g_device_contexts;

DeviceSimContext *lookup_device_context(int device_id) {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    auto it = g_device_contexts.find(device_id);
    return (it != g_device_contexts.end()) ? it->second : nullptr;
}

// ---------------------------------------------------------------------------
// Per-thread device binding (pthread key, not thread_local)
// ---------------------------------------------------------------------------

// Encode device_id as (void*)(intptr_t)(device_id + 1) so that
// device_id 0 is distinguishable from "not set" (nullptr).
constexpr intptr_t DEVICE_ID_OFFSET = 1;

std::mutex g_device_key_mutex;
pthread_key_t g_device_id_key{};
std::atomic<bool> g_device_key_initialized{false};

void ensure_device_key() {
    if (g_device_key_initialized.load(std::memory_order_acquire)) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_device_key_mutex);
    if (!g_device_key_initialized.load(std::memory_order_relaxed)) {
        if (pthread_key_create(&g_device_id_key, nullptr) != 0) {
            return;
        }
        g_device_key_initialized.store(true, std::memory_order_release);
    }
}

int get_current_device_id() {
    if (!g_device_key_initialized.load(std::memory_order_acquire)) {
        return -1;
    }
    auto val = reinterpret_cast<intptr_t>(pthread_getspecific(g_device_id_key));
    return (val != 0) ? static_cast<int>(val - DEVICE_ID_OFFSET) : -1;
}

DeviceSimContext *get_current_device_context() {
    int id = get_current_device_id();
    return (id >= 0) ? lookup_device_context(id) : nullptr;
}

// ---------------------------------------------------------------------------
// Per-thread execution context (block_idx, subblock_id, etc.)
// ---------------------------------------------------------------------------

struct CpuSimExecutionContext {
    uint32_t block_idx = 0;
    uint32_t subblock_id = 0;
    uint32_t subblock_dim = 1;
    uint64_t task_cookie = 0;
};

void free_cpu_sim_execution_context(void *ptr) { std::free(ptr); }

std::mutex g_exec_ctx_key_mutex;
pthread_key_t g_exec_ctx_key{};
std::atomic<bool> g_exec_ctx_key_initialized{false};

CpuSimExecutionContext *get_cpu_sim_execution_context() {
    if (!g_exec_ctx_key_initialized.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(g_exec_ctx_key_mutex);
        if (!g_exec_ctx_key_initialized.load(std::memory_order_relaxed)) {
            if (pthread_key_create(&g_exec_ctx_key, free_cpu_sim_execution_context) != 0) {
                return nullptr;
            }
            g_exec_ctx_key_initialized.store(true, std::memory_order_release);
        }
    }

    auto *ctx = static_cast<CpuSimExecutionContext *>(pthread_getspecific(g_exec_ctx_key));
    if (ctx != nullptr) {
        return ctx;
    }

    ctx = static_cast<CpuSimExecutionContext *>(std::calloc(1, sizeof(CpuSimExecutionContext)));
    if (ctx == nullptr) {
        return nullptr;
    }
    ctx->subblock_dim = 1;

    if (pthread_setspecific(g_exec_ctx_key, ctx) != 0) {
        std::free(ctx);
        return nullptr;
    }
    return ctx;
}

uint64_t make_task_cookie_key(uint32_t core_id, uint32_t reg_task_id) {
    return (static_cast<uint64_t>(core_id) << 32) | static_cast<uint64_t>(reg_task_id);
}

}  // namespace

// ---------------------------------------------------------------------------
// Device lifecycle
// ---------------------------------------------------------------------------

extern "C" void pto_cpu_sim_bind_device(int device_id) {
    ensure_device_key();
    pthread_setspecific(g_device_id_key, reinterpret_cast<void *>(static_cast<intptr_t>(device_id + DEVICE_ID_OFFSET)));
}

extern "C" int pto_cpu_sim_get_bound_device(void) { return get_current_device_id(); }

extern "C" void pto_cpu_sim_acquire_device(int device_id) {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    if (g_device_contexts.find(device_id) == g_device_contexts.end()) {
        g_device_contexts[device_id] = new DeviceSimContext();
    }
}

/** Release and destroy the context for device_id.
 *
 * Safety: the caller (finalize_device in pto_runtime_c_api.cpp) must ensure
 * that all DeviceRunner worker threads for this device have been joined
 * before calling this function. This is guaranteed by DeviceRunner::finalize()
 * which joins all threads before returning.
 */
extern "C" void pto_cpu_sim_release_device(int device_id) {
    DeviceSimContext *ctx = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_registry_mutex);
        auto it = g_device_contexts.find(device_id);
        if (it == g_device_contexts.end()) {
            return;
        }
        ctx = it->second;
        g_device_contexts.erase(it);
    }

    {
        std::lock_guard<std::mutex> lock(ctx->shared_storage_mutex);
        for (auto &[key, storage] : ctx->shared_storage) {
            (void)key;
            std::free(storage);
        }
    }
    delete ctx;
}

void clear_cpu_sim_shared_storage() {
    DeviceSimContext *ctx = get_current_device_context();
    if (ctx == nullptr) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(ctx->task_cookie_mutex);
        ctx->task_cookies.clear();
    }

    {
        std::lock_guard<std::mutex> lock(ctx->shared_storage_mutex);
        for (auto &[key, storage] : ctx->shared_storage) {
            (void)key;
            std::free(storage);
        }
        ctx->shared_storage.clear();
    }
}

// ---------------------------------------------------------------------------
// Per-thread execution context
// ---------------------------------------------------------------------------

extern "C" void pto_cpu_sim_set_execution_context(uint32_t block_idx, uint32_t subblock_id, uint32_t subblock_dim) {
    auto *ctx = get_cpu_sim_execution_context();
    if (ctx == nullptr) {
        return;
    }
    ctx->block_idx = block_idx;
    ctx->subblock_id = subblock_id;
    ctx->subblock_dim = (subblock_dim == 0) ? 1u : subblock_dim;
}

extern "C" void pto_cpu_sim_set_task_cookie(uint64_t task_cookie) {
    auto *ctx = get_cpu_sim_execution_context();
    if (ctx == nullptr) {
        return;
    }
    ctx->task_cookie = task_cookie;
}

extern "C" void pto_cpu_sim_get_execution_context(uint32_t *block_idx, uint32_t *subblock_id, uint32_t *subblock_dim) {
    auto *ctx = get_cpu_sim_execution_context();
    uint32_t b = 0, s = 0, d = 1;
    if (ctx != nullptr) {
        b = ctx->block_idx;
        s = ctx->subblock_id;
        d = ctx->subblock_dim;
    }
    if (block_idx != nullptr) *block_idx = b;
    if (subblock_id != nullptr) *subblock_id = s;
    if (subblock_dim != nullptr) *subblock_dim = d;
}

extern "C" uint64_t pto_cpu_sim_get_task_cookie() {
    auto *ctx = get_cpu_sim_execution_context();
    return (ctx != nullptr) ? ctx->task_cookie : 0;
}

// ---------------------------------------------------------------------------
// Per-device shared storage and task cookies
// ---------------------------------------------------------------------------

extern "C" void platform_set_cpu_sim_task_cookie(uint32_t core_id, uint32_t reg_task_id, uint64_t task_cookie) {
    DeviceSimContext *dev = get_current_device_context();
    if (dev == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(dev->task_cookie_mutex);
    dev->task_cookies[make_task_cookie_key(core_id, reg_task_id)] = task_cookie;
}

extern "C" uint64_t platform_get_cpu_sim_task_cookie(uint32_t core_id, uint32_t reg_task_id) {
    DeviceSimContext *dev = get_current_device_context();
    if (dev == nullptr) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(dev->task_cookie_mutex);
    uint64_t key = make_task_cookie_key(core_id, reg_task_id);
    auto it = dev->task_cookies.find(key);
    if (it == dev->task_cookies.end()) {
        return 0;
    }
    uint64_t val = it->second;
    dev->task_cookies.erase(it);
    return val;
}

extern "C" void *pto_cpu_sim_get_shared_storage(const char *key, size_t size) {
    if (key == nullptr || size == 0) {
        return nullptr;
    }

    DeviceSimContext *dev = get_current_device_context();
    if (dev == nullptr) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(dev->shared_storage_mutex);
    auto it = dev->shared_storage.find(key);
    if (it != dev->shared_storage.end()) {
        return it->second;
    }

    void *storage = std::calloc(1, size);
    dev->shared_storage.emplace(key, storage);
    return storage;
}
