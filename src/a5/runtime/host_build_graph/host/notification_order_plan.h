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
#include <functional>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "task_catalog_plan.h"

struct SchedulerNotificationIdentity {
    uint64_t domain_owner{0};
    uint64_t domain_id{0};
    uint64_t domain_generation{0};
    uint64_t allocation_id{0};
    uint64_t run_epoch{0};
    uint64_t counter_byte_offset{0};
    uint32_t target_rank{0};
    // A nonzero scope is supplied only by a proven notification/reset protocol.
    // Zero means all writes to this counter in the run; expected is not a scope.
    uint64_t notification_epoch{0};

    auto key() const {
        return std::tie(
            domain_owner, domain_id, domain_generation, allocation_id, run_epoch, counter_byte_offset, target_rank,
            notification_epoch
        );
    }
    bool operator<(const SchedulerNotificationIdentity &other) const { return key() < other.key(); }
};

enum class SchedulerNotificationKind { SET, ADD, WAIT, INITIAL_VALUE };

struct SchedulerNotificationEvent {
    SchedulerNotificationIdentity counter;
    uint32_t actor_rank{0};
    int64_t node_ref{-1};
    // Distinguishes real contributions, including different blocks of one task.
    uint64_t event_index{0};
    SchedulerNotificationKind kind{SchedulerNotificationKind::WAIT};
    uint64_t value{0};
    uint32_t block_index{0};
    uint32_t lane{0};
};

struct SchedulerNotificationOrderPlan {
    std::vector<SchedulerTaskOrderEdge> local_edges;
    // Missing remote information or an unproved local protocol keeps the
    // original defer_wait behavior; it does not create a scheduling edge.
    std::vector<SchedulerNotificationEvent> unresolved_waits;
    struct Match {
        SchedulerNotificationEvent wait;
        std::vector<SchedulerNotificationEvent> sources;
    };
    std::vector<Match> matches;
};

// Actual data/Graph reachability only, excluding edges introduced by this matcher.
using SchedulerNotificationDataBefore = std::function<bool(uint32_t, int64_t, int64_t)>;

inline bool scheduler_plan_notification_order(
    uint32_t local_rank, uint64_t node_count, const std::vector<SchedulerNotificationEvent> &events,
    SchedulerNotificationOrderPlan *plan, std::string *error = nullptr,
    const SchedulerNotificationDataBefore &data_before = {}
) {
    const auto fail = [&](const char *message) {
        if (error != nullptr) *error = message;
        return false;
    };
    if (plan == nullptr || node_count > INT32_MAX) return fail("invalid notification plan dimensions");
    using EventId = std::tuple<int64_t, uint32_t, uint32_t, uint64_t>;
    std::map<EventId, SchedulerNotificationEvent> unique;
    std::map<SchedulerNotificationIdentity, std::vector<SchedulerNotificationEvent>> writes;
    std::set<SchedulerNotificationIdentity> ambiguous;
    std::vector<SchedulerNotificationEvent> waits;
    for (const auto &event : events) {
        if (event.actor_rank != local_rank) continue;
        const auto &counter = event.counter;
        const bool initial = event.kind == SchedulerNotificationKind::INITIAL_VALUE;
        if (counter.counter_byte_offset % sizeof(uint32_t) != 0 || event.value > INT32_MAX ||
            (initial ? event.node_ref != -1 : event.node_ref < 0 || uint64_t(event.node_ref) >= node_count))
            return fail("invalid local notification event identity or value");
        if (event.kind != SchedulerNotificationKind::SET && event.kind != SchedulerNotificationKind::ADD &&
            event.kind != SchedulerNotificationKind::WAIT && !initial)
            return fail("unknown notification event kind");
        if (counter.target_rank != local_rank) continue;
        const EventId id{event.node_ref, event.block_index, event.lane, event.event_index};
        const auto [found, inserted] = unique.emplace(id, event);
        if (!inserted) {
            const auto &previous = found->second;
            if (previous.counter.key() != counter.key() || previous.kind != event.kind ||
                previous.value != event.value) {
                ambiguous.insert(previous.counter);
                ambiguous.insert(counter);
            }
            continue;
        }
        if (event.kind == SchedulerNotificationKind::WAIT) waits.push_back(event);
        else writes[counter].push_back(event);
    }

    SchedulerNotificationOrderPlan next;
    std::set<std::pair<int64_t, int64_t>> edges;
    for (const auto &wait : waits) {
        const auto source = writes.find(wait.counter);
        if (source == writes.end() || ambiguous.count(wait.counter) != 0) {
            next.unresolved_waits.push_back(wait);
            continue;
        }
        const auto &contributors = source->second;
        const SchedulerNotificationEvent *initial = nullptr;
        const SchedulerNotificationEvent *setter = nullptr;
        std::map<int64_t, const SchedulerNotificationEvent *> setters;
        bool proved = true, has_add = false;
        uint64_t added = 0;
        for (const auto &event : contributors) {
            if (event.kind == SchedulerNotificationKind::INITIAL_VALUE) {
                if (initial != nullptr) proved = false;
                initial = &event;
            } else if (event.kind == SchedulerNotificationKind::SET) {
                const auto [found, inserted] = setters.emplace(event.node_ref, &event);
                if (!inserted && found->second->value != event.value) proved = false;
                setter = &event;
            } else {
                has_add = true;
                if (event.value > uint64_t{INT32_MAX} - added) proved = false;
                else added += event.value;
            }
        }
        if (setter != nullptr && (has_add || initial != nullptr)) proved = false;
        if (proved && setters.size() > 1) {
            if (!data_before) proved = false;
            std::vector<const SchedulerNotificationEvent *> ordered;
            for (const auto &[ref, candidate] : setters) {
                if (!proved) break;
                auto position = ordered.begin();
                for (; position != ordered.end(); ++position) {
                    const bool before = data_before(local_rank, ref, (*position)->node_ref);
                    const bool after = data_before(local_rank, (*position)->node_ref, ref);
                    if (before == after) {
                        proved = false;
                        break;
                    }
                    if (before) break;
                }
                ordered.insert(position, candidate);
            }
            if (proved) {
                for (size_t i = 1; i < ordered.size(); ++i)
                    if (ordered[i - 1]->value > ordered[i]->value) proved = false;
                setter = ordered.back();
                for (const auto *candidate : ordered)
                    if (candidate->value >= wait.value) {
                        setter = candidate;
                        break;
                    }
            }
        }
        if (setter == nullptr && initial == nullptr) proved = false;
        if (initial != nullptr && added > uint64_t{INT32_MAX} - initial->value) proved = false;
        const uint64_t guaranteed = setter != nullptr ? setter->value : initial != nullptr ? initial->value + added : 0;
        const bool initially_satisfied = initial != nullptr && initial->value >= wait.value;
        if (guaranteed < wait.value || (has_add && !initially_satisfied && guaranteed != wait.value)) proved = false;
        std::set<int64_t> producers;
        if (proved && !initially_satisfied)
            for (const auto &event : contributors) {
                if (event.kind == SchedulerNotificationKind::INITIAL_VALUE ||
                    (event.kind == SchedulerNotificationKind::ADD && event.value == 0) ||
                    (event.kind == SchedulerNotificationKind::SET && event.node_ref != setter->node_ref))
                    continue;
                if (event.node_ref == wait.node_ref ||
                    (data_before && data_before(local_rank, wait.node_ref, event.node_ref)))
                    proved = false;
                producers.insert(event.node_ref);
            }
        if (!proved) {
            next.unresolved_waits.push_back(wait);
            continue;
        }
        next.matches.push_back({wait, contributors});
        for (const auto producer : producers)
            edges.emplace(producer, wait.node_ref);
    }
    for (const auto &[producer, waiter] : edges)
        next.local_edges.push_back({producer, waiter});
    *plan = std::move(next);
    if (error != nullptr) error->clear();
    return true;
}
