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

#include "aicpu/cache_maintenance.h"
#include "common/memory_barrier.h"
#include "scheduler/calendar_wire.h"

namespace simpler::percounter {

// A burst is admitted only if every destination in the arithmetic progression
// is present in the bound subscriber list. MTE3 uses a positive byte stride.
inline CalendarPeerBurst percounter_peer_burst(const uint64_t *addresses, uint32_t count, uint32_t step) {
    CalendarPeerBurst burst{addresses[0], 0, 1};
    if (count <= step) return burst;
    const bool descending = addresses[step] < addresses[0];
    const uint64_t stride = descending ? addresses[0] - addresses[step] : addresses[step] - addresses[0];
    if (stride == 0 || stride > UINT32_MAX || stride % sizeof(uint32_t) != 0) return burst;
    burst.stride_bytes = static_cast<uint32_t>(stride);
    for (uint32_t i = step; i < count; i += step) {
        const auto previous = addresses[i - step], current = addresses[i];
        if (descending ? (current >= previous || previous - current != stride) :
                         (current <= previous || current - previous != stride))
            break;
        ++burst.targets;
    }
    if (descending) burst.start = addresses[(burst.targets - 1) * step];
    return burst;
}

inline uint16_t percounter_pack_peer_bursts(const uint64_t *addresses, uint32_t count, CalendarPeerBurst *out) {
    const uint32_t capacity = count * sizeof(uint32_t) / sizeof(CalendarPeerBurst);
    uint32_t used = 0;
    for (uint32_t i = 0; i < count;) {
        if (used == capacity) return 0;
        const uint32_t left = count - i;
        auto direct = percounter_peer_burst(addresses + i, left, 1);
        // Adjacent AIV siblings often form two interleaved progressions.
        if (left >= 4 && capacity - used >= 2) {
            auto even = percounter_peer_burst(addresses + i, left, 2);
            auto odd = percounter_peer_burst(addresses + i + 1, left - 1, 2);
            const uint32_t pairs = even.targets < odd.targets ? even.targets : odd.targets;
            if (pairs > direct.targets) {
                auto pair_burst = [&](CalendarPeerBurst burst, uint32_t first) {
                    burst.targets = pairs;
                    const auto last = addresses[first + (pairs - 1) * 2];
                    burst.start = addresses[first] < last ? addresses[first] : last;
                    return burst;
                };
                out[used++] = pair_burst(even, i);
                out[used++] = pair_burst(odd, i + 1);
                i += 2 * pairs;
                continue;
            }
        }
        out[used++] = direct;
        i += direct.targets;
    }
    return static_cast<uint16_t>(used);
}

// The host fixes the recipients and their order. Physical endpoints and their
// initial DMB words become known only after the workers report for this launch.
inline bool
percounter_bind_peer_fanouts(CalendarRangeHeader &calendar, uint64_t bytes, const CalendarPeerEndpoint *endpoints) {
    auto *image = reinterpret_cast<uint8_t *>(&calendar);
    uint32_t offset = calendar.peer_fanouts_offset;
    while (offset != 0) {
        if (offset < sizeof(calendar) || offset % alignof(CalendarPeerFanout) != 0 || offset > bytes ||
            bytes - offset < sizeof(CalendarPeerFanout))
            return false;
        auto *fanout = reinterpret_cast<CalendarPeerFanout *>(image + offset);
        cache_invalidate_range(fanout, sizeof(*fanout));
        rmb();
        const uint32_t count = fanout->count;
        const uint32_t previous = fanout->previous_offset;
        const size_t size = sizeof(*fanout) + uint64_t{count} * CALENDAR_FANOUT_TARGET_BYTES;
        if (count < 2 || count > calendar.worker_count || size > bytes - offset || previous >= offset) return false;
        auto *addresses = reinterpret_cast<uint64_t *>(fanout + 1);
        auto *seeds = reinterpret_cast<uint32_t *>(addresses + count);
        const auto *workers = seeds + count;
        cache_invalidate_range(fanout, size);
        rmb();
        uint64_t common = 0;
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t worker = workers[i];
            if (worker >= calendar.worker_count || (i != 0 && worker <= workers[i - 1])) return false;
            const auto &endpoint = endpoints[worker];
            if (endpoint.input_count != CALENDAR_PEER_SPMD_ENDPOINT || endpoint.address == 0) return false;
            addresses[i] = endpoint.address;
            seeds[i] = endpoint.token_base;
            if (i == 0) common = seeds[i];
            else if (common != seeds[i]) common = UINT64_MAX;
        }
        fanout->common_token_base = common;
        fanout->burst_count =
            common == UINT64_MAX ?
                0 :
                percounter_pack_peer_bursts(addresses, count, reinterpret_cast<CalendarPeerBurst *>(seeds));
        cache_flush_range(fanout, size);
        offset = previous;
    }
    return true;
}

}  // namespace simpler::percounter
