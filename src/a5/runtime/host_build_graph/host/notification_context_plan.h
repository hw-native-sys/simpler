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

#include <map>
#include <memory>
#include <string>

#include "common/notification_context.h"
#include "notification_metadata_plan.h"
#include "notification_order_plan.h"

using SchedulerNotificationContexts = std::map<uint64_t, std::shared_ptr<const HostNotificationContext>>;

inline bool scheduler_normalize_notification_event(
    const SchedulerBoundNotificationEvent &bound, const HostNotificationContext &context, uint64_t run_epoch,
    SchedulerNotificationEvent *event, std::string *error
) {
    using namespace simpler::hbg;
    const auto fail = [&](const char *message) {
        if (error != nullptr) *error = std::string(message) + " at task " + std::to_string(bound.task_ref);
        return false;
    };
    if (event == nullptr || context.bootstrap_token == 0 || context.rank_count == 0 ||
        context.rank_count > context.windows_in.size() || context.local_rank >= context.rank_count ||
        bound.effect.comm_context != context.context_address || run_epoch == 0)
        return fail("invalid owned notification context");
    const uint64_t base = context.windows_in[context.local_rank];
    if (base == 0 || bound.local_counter_address < base || bound.local_counter_address - base > context.window_bytes ||
        sizeof(uint32_t) > context.window_bytes - (bound.local_counter_address - base))
        return fail("notification counter is outside its owned window");
    const bool waiting = bound.effect.kind == NotificationEventKind::WAIT_GE;
    if (!waiting && (bound.effect.peer < 0 || static_cast<uint64_t>(bound.effect.peer) >= context.rank_count))
        return fail("notification peer is outside its owned domain");
    SchedulerNotificationEvent next;
    next.counter.domain_owner = context.bootstrap_token;
    next.counter.allocation_id = context.allocation_id;
    next.counter.domain_generation = context.generation;
    next.counter.run_epoch = run_epoch;
    next.counter.counter_byte_offset = bound.local_counter_address - base;
    next.counter.target_rank = waiting ? context.local_rank : static_cast<uint32_t>(bound.effect.peer);
    next.actor_rank = context.local_rank;
    next.node_ref = bound.task_ref;
    next.block_index = bound.block_index;
    next.lane = bound.lane;
    next.event_index = bound.event_index;
    next.kind = waiting                                         ? SchedulerNotificationKind::WAIT :
                bound.effect.kind == NotificationEventKind::SET ? SchedulerNotificationKind::SET :
                                                                  SchedulerNotificationKind::ADD;
    next.value = static_cast<uint64_t>(bound.effect.value);
    *event = next;
    return true;
}

// Only relationships proved from this Host plan add ordering edges. Unresolved
// remote waits and unavailable notification metadata retain their device behavior.
inline bool scheduler_resolve_notification_order(
    const SchedulerNotificationMetadataPlan &metadata, const SchedulerNotificationContexts &contexts,
    uint64_t run_epoch, const SchedulerGraphAccess &graph, const std::vector<int64_t> &completed,
    std::vector<SchedulerTaskOrderEdge> *edges, std::string *error
) {
    using namespace simpler::hbg;
    if (edges == nullptr) return false;
    const auto unchanged = [&] {
        edges->clear();
        if (error != nullptr) error->clear();
        return true;
    };
    for (const auto &missing : metadata.unavailable)
        if ((missing.kernel_flags & NOTIFICATION_KERNEL_NOTIFY) != 0) return unchanged();
    struct DomainEvents {
        std::shared_ptr<const HostNotificationContext> context;
        std::vector<SchedulerNotificationEvent> events;
        bool proved{true};
    };
    using DomainKey = std::tuple<uint64_t, uint64_t, uint64_t, uint32_t>;
    const auto domain_key = [](const HostNotificationContext &context) {
        return DomainKey{context.bootstrap_token, context.allocation_id, context.generation, context.local_rank};
    };
    std::map<DomainKey, DomainEvents> domains;
    for (const auto &bound : metadata.events) {
        if (bound.effect.kind != NotificationEventKind::WAIT_GE || bound.effect.value == 0) continue;
        const auto found = contexts.find(bound.effect.comm_context);
        if (found == contexts.end() || !found->second) continue;
        domains[domain_key(*found->second)].context = found->second;
    }
    if (domains.empty()) return unchanged();
    for (const auto &bound : metadata.events) {
        if (bound.effect.kind == NotificationEventKind::WAIT_GE && bound.effect.value == 0) continue;
        const auto found = contexts.find(bound.effect.comm_context);
        if (found == contexts.end() || !found->second) {
            if (bound.effect.kind != NotificationEventKind::WAIT_GE) return unchanged();
            continue;
        }
        const auto &context = *found->second;
        const auto selected = domains.find(domain_key(context));
        if (selected == domains.end()) continue;
        auto &domain = selected->second;
        SchedulerNotificationEvent event;
        if (domain.context->rank_count != context.rank_count ||
            !scheduler_normalize_notification_event(bound, context, run_epoch, &event, nullptr)) {
            domain.proved = false;
            continue;
        }
        domain.events.push_back(event);
    }

    std::vector<bool> active(graph.task_count, true);
    for (const auto ref : completed) {
        if (!scheduler_task_ref_valid(graph, ref)) {
            if (error != nullptr) *error = "invalid completed notification node";
            return false;
        }
        active[ref] = false;
    }
    std::vector<std::vector<int64_t>> successors(graph.task_count);
    for (uint64_t ref = 0; ref < graph.task_count; ++ref) {
        if (!active[ref]) continue;
        SchedulerFaninView fanin{};
        if (scheduler_node_fanin_view(graph, ref, &fanin) != SchedulerGraphResult::OK) {
            if (error != nullptr) *error = "invalid notification data dependency view";
            return false;
        }
        for (uint32_t index = 0; index < fanin.count; ++index) {
            int64_t producer = -1;
            if (scheduler_fanin_ref(fanin, ref, index, &producer) != SchedulerGraphResult::OK) {
                if (error != nullptr) *error = "invalid notification data dependency";
                return false;
            }
            successors[producer].push_back(static_cast<int64_t>(ref));
        }
    }
    std::map<std::pair<int64_t, int64_t>, bool> reachable;
    const SchedulerNotificationDataBefore data_before = [&](uint32_t, int64_t producer, int64_t consumer) {
        const auto key = std::make_pair(producer, consumer);
        const auto cached = reachable.find(key);
        if (cached != reachable.end()) return cached->second;
        std::vector<uint8_t> visited(graph.task_count);
        std::vector<int64_t> pending{producer};
        bool found = false;
        while (!pending.empty() && !found) {
            const auto ref = pending.back();
            pending.pop_back();
            if (visited[ref]) continue;
            visited[ref] = 1;
            for (const auto next : successors[ref]) {
                if (next == consumer) {
                    found = true;
                    break;
                }
                pending.push_back(next);
            }
        }
        reachable.emplace(key, found);
        return found;
    };
    std::set<std::pair<int64_t, int64_t>> merged;
    for (const auto &[key, domain] : domains) {
        if (!domain.proved) continue;
        SchedulerNotificationOrderPlan plan;
        if (!scheduler_plan_notification_order(
                domain.context->local_rank, graph.task_count, domain.events, &plan, error, data_before
            ))
            return false;
        for (const auto &edge : plan.local_edges)
            merged.emplace(edge.predecessor, edge.successor);
    }
    edges->clear();
    for (const auto &[producer, consumer] : merged)
        edges->push_back({producer, consumer});
    if (error != nullptr) error->clear();
    return true;
}
