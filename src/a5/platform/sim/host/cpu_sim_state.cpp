/**
 * @file cpu_sim_state.cpp
 * @brief CPU simulation state management for AICore execution context
 *
 * Provides thread-local execution context and shared storage APIs for
 * the simulation environment. These extern "C" functions are called by
 * AICore simulation code (via dlsym) to emulate hardware-provided
 * block/subblock identity and cross-core shared state (e.g., VEC_FIFO).
 *
 * On real hardware these calls are compiled out via no-op macros in
 * onboard/aicore/inner_kernel.h.
 */

#include <cstdint>
#include <cstdlib>
#include <cstddef>
#include <map>
#include <mutex>
#include <string>

#include "cpu_sim_state.h"

namespace {
thread_local uint32_t g_cpu_sim_block_idx = 0;
thread_local uint32_t g_cpu_sim_subblock_id = 0;
thread_local uint32_t g_cpu_sim_subblock_dim = 1;
thread_local uint64_t g_cpu_sim_task_cookie = 0;
std::mutex g_cpu_sim_shared_storage_mutex;
std::map<std::string, void*> g_cpu_sim_shared_storage;
}  // namespace

void clear_cpu_sim_shared_storage()
{
    std::lock_guard<std::mutex> lock(g_cpu_sim_shared_storage_mutex);
    for (auto& [key, storage] : g_cpu_sim_shared_storage) {
        (void)key;
        std::free(storage);
    }
    g_cpu_sim_shared_storage.clear();
}

extern "C" void pto_cpu_sim_set_execution_context(uint32_t block_idx, uint32_t subblock_id,
                                                   uint32_t subblock_dim)
{
    g_cpu_sim_block_idx = block_idx;
    g_cpu_sim_subblock_id = subblock_id;
    g_cpu_sim_subblock_dim = (subblock_dim == 0) ? 1u : subblock_dim;
}

extern "C" void pto_cpu_sim_set_task_cookie(uint64_t task_cookie)
{
    g_cpu_sim_task_cookie = task_cookie;
}

extern "C" void pto_cpu_sim_get_execution_context(uint32_t* block_idx, uint32_t* subblock_id,
                                                   uint32_t* subblock_dim)
{
    if (block_idx != nullptr) {
        *block_idx = g_cpu_sim_block_idx;
    }
    if (subblock_id != nullptr) {
        *subblock_id = g_cpu_sim_subblock_id;
    }
    if (subblock_dim != nullptr) {
        *subblock_dim = g_cpu_sim_subblock_dim;
    }
}

extern "C" uint64_t pto_cpu_sim_get_task_cookie()
{
    return g_cpu_sim_task_cookie;
}

extern "C" void* pto_cpu_sim_get_shared_storage(const char* key, size_t size)
{
    if (key == nullptr || size == 0) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(g_cpu_sim_shared_storage_mutex);
    auto it = g_cpu_sim_shared_storage.find(key);
    if (it != g_cpu_sim_shared_storage.end()) {
        return it->second;
    }

    void* storage = std::calloc(1, size);
    g_cpu_sim_shared_storage.emplace(key, storage);
    return storage;
}
