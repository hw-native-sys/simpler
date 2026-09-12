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

#pragma once

#if __has_include(<pto/common/cpu_stub.hpp>)
#include <pto/common/cpu_stub.hpp>
#endif

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <thread>
#include <type_traits>

inline constexpr uint16_t FFTS_MODE_VAL = 2;

namespace pto {
static inline uint16_t getFFTSMsg(uint16_t mode, uint16_t event_id, uint16_t base_count = 1) {
    return (base_count & 0xf) | ((mode & 0x3) << 4) | ((event_id & 0xf) << 8);
}
}  // namespace pto

namespace {
namespace sim_ffts {

using SubblockHook = uint32_t (*)();
using StorageHook = void *(*)(uint64_t, size_t);
SubblockHook subblock_hook = nullptr;
StorageHook storage_hook = nullptr;

#if defined(__DAV_CUBE__)
constexpr bool is_cube = true;
#elif defined(__DAV_VEC__)
constexpr bool is_cube = false;
#else
#error "FFTS simulation requires the existing AIC or AIV compiler specialization"
#endif

struct EventState {
    std::atomic<uint32_t> cube_to_vector[2]{};
    std::atomic<uint32_t> vector_to_cube[2]{};
};
static_assert(std::is_trivially_destructible_v<EventState>);

struct EventStorage {
    uint32_t initialized;
    alignas(EventState) unsigned char payload[sizeof(EventState)];
};

EventState &event_state(int event_id) {
    if (event_id < 0 || event_id >= 16) {
        throw std::runtime_error("FFTS simulation requires an event ID in [0, 15]");
    }
    if (storage_hook == nullptr || subblock_hook == nullptr) {
        throw std::runtime_error("FFTS simulation requires runtime hook injection");
    }
    // The prefix is outside the hardware tile-pipe flag namespace. The registry
    // isolates devices/clusters and clears storage before each simulated run.
    constexpr uint64_t key_prefix = 0xff46465453000000ULL;
    auto *storage = static_cast<EventStorage *>(storage_hook(key_prefix | event_id, sizeof(EventStorage)));
    if (storage == nullptr) {
        throw std::runtime_error("FFTS simulation requires a bound runtime device");
    }
    std::atomic_ref<uint32_t> initialized(storage->initialized);
    uint32_t expected = 0;
    if (initialized.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
        new (storage->payload) EventState{};
        initialized.store(2, std::memory_order_release);
    } else {
        while (initialized.load(std::memory_order_acquire) != 2) {}
    }
    return *std::launder(reinterpret_cast<EventState *>(storage->payload));
}

uint32_t vector_lane() {
    const uint32_t lane = subblock_hook();
    if (lane >= 2) {
        throw std::runtime_error("FFTS mode 2 requires AIV subblock 0 or 1");
    }
    return lane;
}

void publish(std::atomic<uint32_t> &credits) {
    uint32_t value = credits.load(std::memory_order_relaxed);
    do {
        if (value == std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("FFTS simulation credit counter overflow");
        }
    } while (!credits.compare_exchange_weak(value, value + 1, std::memory_order_release, std::memory_order_relaxed));
}

void consume(std::atomic<uint32_t> &credits) {
    uint32_t value = credits.load(std::memory_order_acquire);
    for (;;) {
        if (value == 0) {
            std::this_thread::yield();
            value = credits.load(std::memory_order_acquire);
        } else if (credits.compare_exchange_weak(value, value - 1, std::memory_order_acquire)) {
            return;
        }
    }
}

void signal(uint16_t message) {
    if (((message >> 4) & 0x3) != FFTS_MODE_VAL || (message & 0xf) != 1) {
        throw std::runtime_error("FFTS simulation supports mode 2 with base count 1 only");
    }
    auto &state = event_state((message >> 8) & 0xf);
    if constexpr (is_cube) {
        publish(state.cube_to_vector[0]);
        publish(state.cube_to_vector[1]);
    } else {
        publish(state.vector_to_cube[vector_lane()]);
    }
}

void wait(int event_id) {
    auto &state = event_state(event_id);
    if constexpr (is_cube) {
        // Two notifications from one AIV lane cannot satisfy the other lane.
        consume(state.vector_to_cube[0]);
        consume(state.vector_to_cube[1]);
    } else {
        consume(state.cube_to_vector[vector_lane()]);
    }
}

}  // namespace sim_ffts
}  // namespace

// Each RTLD_LOCAL kernel DSO receives the same runtime-owned registry hooks.
extern "C" __attribute__((visibility("default"))) void
pto_sim_register_hooks(void *get_subblock_id, void *get_pipe_shared_state) {
    sim_ffts::subblock_hook = reinterpret_cast<sim_ffts::SubblockHook>(get_subblock_id);
    sim_ffts::storage_hook = reinterpret_cast<sim_ffts::StorageHook>(get_pipe_shared_state);
#if __has_include(<pto/common/cpu_stub.hpp>)
    pto::cpu_sim::register_hooks(get_subblock_id, get_pipe_shared_state);
#endif
}

static inline void __builtin_cce_ffts_cross_core_sync(int, uint16_t message) { sim_ffts::signal(message); }

static inline void __builtin_cce_wait_flag_dev(int event_id) { sim_ffts::wait(event_id); }

static inline void ffts_cross_core_sync(int pipe, uint16_t message) {
    __builtin_cce_ffts_cross_core_sync(pipe, message);
}

static inline void wait_flag_dev(int event_id) { __builtin_cce_wait_flag_dev(event_id); }
