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
 * @brief Per-device CPU simulation context for TPUSH/TPOP
 *
 * Provides per-thread core identity (subblock_id, cluster_id, dispatch_id)
 * and per-device pipe shared state maps.
 *
 * Each simulated device has an independent DeviceSimContext so that multiple
 * ChipWorkers (each simulating a different device) can run concurrently.
 *
 * The current device is bound to each thread via a pthread key set in
 * pto_cpu_sim_bind_device(). Hooks called by pto-isa route through this
 * binding to find the correct DeviceSimContext.
 *
 * Exported hooks (resolved by pto-isa via dlsym(RTLD_DEFAULT)):
 *   - pto_sim_get_subblock_id: returns current thread's AIV lane (0 or 1)
 *   - pto_sim_get_pipe_shared_state: returns per-device per-cluster
 *     per-dispatch pipe shared memory keyed by a uint32 pipe configuration
 *
 * Per-thread TLS values (subblock_id, cluster_id, dispatch_id) are set by
 * the sim aicore platform code via the sim_context_set_* setter functions.
 */

#include "cpu_sim_context.h"

#include "common/unified_log.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <pthread.h>
#include <unordered_map>

namespace {

// ---------------------------------------------------------------------------
// Per-device pipe shared state
// ---------------------------------------------------------------------------

// Key identifying a pipe shared state entry: per-cluster, per-pipe configuration.
// pipe_key encodes FlagID, DirType, SlotNum, LocalSlotNum, SlotSize from pto-isa.
// Fixed number of entries (one per cluster per pipe type) — no growth over time.
struct PipeStateKey {
    uint32_t cluster_id;
    uint64_t pipe_key;

    bool operator==(const PipeStateKey &o) const { return cluster_id == o.cluster_id && pipe_key == o.pipe_key; }
};

struct PipeStateKeyHash {
    size_t operator()(const PipeStateKey &k) const {
        return static_cast<size_t>(k.cluster_id) * 0x9e3779b97f4a7c15ULL ^ static_cast<size_t>(k.pipe_key);
    }
};

struct DeviceSimContext {
    std::mutex pipe_state_mutex;
    std::unordered_map<PipeStateKey, void *, PipeStateKeyHash> pipe_states;
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
// Per-thread core identity TLS (subblock_id, cluster_id, dispatch_id)
// ---------------------------------------------------------------------------

std::mutex g_identity_key_mutex;
pthread_key_t g_subblock_id_key{};
pthread_key_t g_cluster_id_key{};
std::atomic<bool> g_identity_keys_initialized{false};

void ensure_identity_keys() {
    if (g_identity_keys_initialized.load(std::memory_order_acquire)) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_identity_key_mutex);
    if (g_identity_keys_initialized.load(std::memory_order_relaxed)) {
        return;
    }
    if (pthread_key_create(&g_subblock_id_key, nullptr) != 0) {
        return;
    }
    if (pthread_key_create(&g_cluster_id_key, nullptr) != 0) {
        pthread_key_delete(g_subblock_id_key);
        return;
    }
    g_identity_keys_initialized.store(true, std::memory_order_release);
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
        // Verifies process-wide HostLogger singleton: this LOG_INFO_V0 call
        // resolves into libsimpler_log.so loaded by ChipWorker with
        // RTLD_GLOBAL — same instance as host_runtime.so writes to.
        LOG_INFO_V0("cpu_sim_context: acquired device %d", device_id);
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
        std::lock_guard<std::mutex> lock(ctx->pipe_state_mutex);
        for (auto &[key, storage] : ctx->pipe_states) {
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
    std::lock_guard<std::mutex> lock(ctx->pipe_state_mutex);
    for (auto &[key, storage] : ctx->pipe_states) {
        (void)key;
        std::free(storage);
    }
    ctx->pipe_states.clear();
}

// ---------------------------------------------------------------------------
// Per-thread core identity setters (called by sim aicore platform code)
// ---------------------------------------------------------------------------

void sim_context_set_subblock_id(uint32_t subblock_id) {
    ensure_identity_keys();
    pthread_setspecific(g_subblock_id_key, reinterpret_cast<void *>(static_cast<uintptr_t>(subblock_id)));
}

void sim_context_set_cluster_id(uint32_t cluster_id) {
    ensure_identity_keys();
    pthread_setspecific(g_cluster_id_key, reinterpret_cast<void *>(static_cast<uintptr_t>(cluster_id)));
}

// ---------------------------------------------------------------------------
// Hooks resolved by pto-isa via function pointer injection
// ---------------------------------------------------------------------------

extern "C" uint32_t pto_sim_get_subblock_id() {
    if (!g_identity_keys_initialized.load(std::memory_order_acquire)) {
        return 0;
    }
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(pthread_getspecific(g_subblock_id_key)));
}

extern "C" void *pto_sim_get_pipe_shared_state(uint64_t pipe_key, size_t size) {
    if (size == 0) {
        return nullptr;
    }

    DeviceSimContext *dev = get_current_device_context();
    if (dev == nullptr) {
        return nullptr;
    }

    uint32_t cluster_id = 0;
    if (g_identity_keys_initialized.load(std::memory_order_acquire)) {
        cluster_id = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(pthread_getspecific(g_cluster_id_key)));
    }

    PipeStateKey key{cluster_id, pipe_key};

    std::lock_guard<std::mutex> lock(dev->pipe_state_mutex);
    auto it = dev->pipe_states.find(key);
    if (it != dev->pipe_states.end()) {
        return it->second;
    }

    void *storage = std::calloc(1, size);
    dev->pipe_states.emplace(key, storage);
    return storage;
}
