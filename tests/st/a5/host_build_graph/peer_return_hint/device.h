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

#include <cstdint>
#if !defined(__CCE_AICORE__)
#include <chrono>
#endif

#include "intrinsic.h"
#include "scheduler/calendar_wire.h"
#include "scheduler/gm.h"
#include "scheduler/profiling_types.h"
#include "scheduler/wire_types.h"
#include "tensor.h"

namespace peer_hint_test {

inline constexpr int kProduced = 0;
inline constexpr int kSignal = 16;
inline constexpr int kConsumed = 32;
inline constexpr int kObservedSignal = 33;

inline __aicore__ __gm__ int32_t *state(__gm__ int64_t *args) {
    const auto *tensor = reinterpret_cast<const __gm__ Tensor *>(args[0]);
    return reinterpret_cast<__gm__ int32_t *>(tensor->buffer.addr) + tensor->start_offset;
}

inline __aicore__ uint64_t now() {
#if defined(__CCE_AICORE__)
    return get_sys_cnt();
#else
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count()
    );
#endif
}

inline __aicore__ void wait_ticks(uint64_t ticks) {
    const uint64_t began = now();
    while (now() - began < ticks) {}
}

inline __aicore__ int32_t observe(__gm__ int32_t *address) {
#if defined(__CCE_AICORE__)
    dcci(address, SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
    return *reinterpret_cast<volatile __gm__ int32_t *>(address);
#else
    return __atomic_load_n(address, __ATOMIC_ACQUIRE);
#endif
}

inline __aicore__ void publish(__gm__ int32_t *address, int32_t value) {
#if defined(__CCE_AICORE__)
    *address = value;
    dcci(address, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dsb(DSB_DDR);
#else
    __atomic_store_n(address, value, __ATOMIC_RELEASE);
#endif
}

template <typename Ready>
inline __aicore__ bool wait_for(Ready ready) {
    const auto start = peer_hint_test::now();
    while (!ready()) {
        if (peer_hint_test::now() - start > 10000000) return false;
    }
    return true;
}

inline __aicore__ bool receiver_checked_pending(__gm__ int64_t *args, bool require_token) {
    using namespace simpler::percounter;
    auto *image = reinterpret_cast<__gm__ uint8_t *>(load_global_context(args).async_image_address);
    auto &control = *reinterpret_cast<__gm__ PercounterControl *>(image);
    const auto *calendar = reinterpret_cast<const __gm__ CalendarRangeHeader *>(image + control.calendar_offset);
    if (calendar->peer_endpoints_offset == 0 || control.profile_buffers_offset == 0) {
        if (!require_token) return true;
        auto *counters = reinterpret_cast<__gm__ PercounterCounter *>(image + control.counters_offset);
        return wait_for([&]() __aicore__ {
            return percounter_gm_read(counters[0].calendar_defer_head) != 0;
        });
    }

    const auto *endpoints = reinterpret_cast<const __gm__ CalendarPeerEndpoint *>(
        reinterpret_cast<const __gm__ uint8_t *>(calendar) + calendar->peer_endpoints_offset
    );
    const auto *profiles =
        reinterpret_cast<const __gm__ PercounterProfileBuffer *>(image + control.profile_buffers_offset);
    auto *timing = reinterpret_cast<__gm__ PercounterAicpuTiming *>(image + control.aicpu_profile_offset);
    uint64_t start = 0;
    // Event storage is reused. START is published before any event from this run.
    if (!wait_for([&]() __aicore__ {
            start = percounter_gm_read(timing->ready_publish_start_cycles);
            return start != 0;
        }))
        return false;
    for (uint32_t worker = 0; worker < calendar->worker_count; ++worker) {
        if (endpoints[worker].input_count != 1) continue;
        const uint64_t expected = uint64_t{endpoints[worker].token_base} << 32;
        const auto &profile = profiles[worker];
        auto *events = reinterpret_cast<__gm__ PercounterTaskEvent *>(image + profile.event_records_offset);
        return wait_for([&]() __aicore__ {
            for (uint32_t i = 0; i < profile.event_capacity; ++i) {
                auto &event = events[i];
                const auto kind = static_cast<PercounterTaskEventKind>(
                    percounter_gm_read(reinterpret_cast<__gm__ uint32_t &>(event.kind))
                );
                if (kind != PercounterTaskEventKind::PeerInitialProof &&
                    (!require_token || kind != PercounterTaskEventKind::PeerObserved))
                    continue;
                if (percounter_gm_read(event.cycles) < start) continue;
                const uint64_t details = percounter_gm_read(event.detail);
                if (require_token ? details == expected : static_cast<uint32_t>(details) == 0) return true;
            }
            return false;
        });
    }
    return false;
}

}  // namespace peer_hint_test
