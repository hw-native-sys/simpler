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

#include <limits>
#include <string>
#include <vector>

#include "host_build_graph/host/notification_metadata.h"

struct SchedulerBoundNotificationEvent {
    int64_t task_ref;
    uint32_t block_index;
    uint32_t lane;
    uint32_t event_index;
    uint64_t local_counter_address;
    uint64_t local_buffer_address;
    simpler::hbg::NotificationPlanEvent effect;
};

struct SchedulerUnavailableNotificationMetadata {
    int64_t task_ref;
    uint32_t kernel_id;
    uint32_t block_index;
    uint32_t lane;
    uint32_t kernel_flags;
    simpler::hbg::NotificationPlanStatus status;
};

struct SchedulerNotificationMetadataPlan {
    std::vector<SchedulerBoundNotificationEvent> events;
    std::vector<SchedulerUnavailableNotificationMetadata> unavailable;
};

// Addresses still refer to the local window. Remote peer normalization and
// provenance checks must precede matching; a peer number never changes an address here.
inline simpler::hbg::NotificationPlanStatus scheduler_collect_notification_metadata(
    simpler::hbg::NotificationPlanFunction query, uint32_t kernel_id, int64_t task_ref,
    const simpler::hbg::NotificationPlanInput &input, std::vector<SchedulerBoundNotificationEvent> *events,
    std::string *error
) {
    using namespace simpler::hbg;
    if (query == nullptr || events == nullptr || task_ref < 0 || input.version != NOTIFICATION_PLAN_VERSION ||
        input.block_count == 0 || input.block_index >= input.block_count || input.lane >= 3 ||
        (input.tensor_count != 0 && input.tensors == nullptr) ||
        (input.scalar_count != 0 && input.scalars == nullptr)) {
        if (error != nullptr) *error = "invalid notification metadata invocation";
        return NotificationPlanStatus::INVALID;
    }
    struct SinkState {
        const NotificationPlanInput &input;
        int64_t task_ref;
        std::vector<SchedulerBoundNotificationEvent> staged;
        bool invalid{false};
    } state{input, task_ref, {}, false};
    const auto sink = [](void *opaque, const NotificationPlanEvent *event) {
        auto &state = *static_cast<SinkState *>(opaque);
        const auto invalid = [&state]() {
            state.invalid = true;
            return false;
        };
        if (event == nullptr || event->tensor_index >= state.input.tensor_count || event->ndims == 0 ||
            event->ndims > MAX_TENSOR_DIMS || event->value < 0 || event->value > INT32_MAX ||
            (event->kind != NotificationEventKind::SET && event->kind != NotificationEventKind::ADD &&
             event->kind != NotificationEventKind::WAIT_GE) ||
            (event->kind == NotificationEventKind::WAIT_GE ? event->peer != -1 : event->peer < 0))
            return invalid();
        const auto &tensor = state.input.tensors[event->tensor_index];
        if (tensor.ndims != event->ndims || tensor.dtype != DataType::INT32 ||
            tensor.address_space != AddressSpace::DEVICE || tensor.buffer.addr == 0)
            return invalid();
        uint64_t offset = tensor.start_offset;
        for (uint32_t axis = 0; axis < event->ndims; ++axis) {
            if (event->offsets[axis] < 0 || static_cast<uint64_t>(event->offsets[axis]) >= tensor.shapes[axis])
                return invalid();
            const uint64_t term = static_cast<uint64_t>(event->offsets[axis]) * tensor.strides[axis];
            if (term > UINT64_MAX - offset) return invalid();
            offset += term;
        }
        if (offset > UINT64_MAX / sizeof(int32_t)) return invalid();
        offset *= sizeof(int32_t);
        if (offset > tensor.buffer.size || sizeof(int32_t) > tensor.buffer.size - offset ||
            offset > UINT64_MAX - tensor.buffer.addr || (tensor.buffer.addr + offset) % alignof(int32_t) != 0 ||
            state.staged.size() >= UINT32_MAX)
            return invalid();
        state.staged.push_back(
            {state.task_ref, state.input.block_index, state.input.lane, static_cast<uint32_t>(state.staged.size()),
             tensor.buffer.addr + offset, tensor.buffer.addr, *event}
        );
        return true;
    };
    const auto status = query(kernel_id, &input, sink, &state);
    if (state.invalid) {
        if (error != nullptr) *error = "invalid notification event at task " + std::to_string(task_ref);
        return NotificationPlanStatus::INVALID;
    }
    if (status != NotificationPlanStatus::COMPLETE) {
        if (error != nullptr)
            *error = "notification metadata unavailable at task " + std::to_string(task_ref) + " kernel " +
                     std::to_string(kernel_id);
        return status;
    }
    events->insert(events->end(), state.staged.begin(), state.staged.end());
    if (error != nullptr) error->clear();
    return NotificationPlanStatus::COMPLETE;
}

inline bool scheduler_collect_notification_invocation(
    simpler::hbg::NotificationPlanFunction query, uint32_t kernel_flags, uint32_t kernel_id, int64_t task_ref,
    const simpler::hbg::NotificationPlanInput &input, SchedulerNotificationMetadataPlan *plan, std::string *error
) {
    using namespace simpler::hbg;
    if (plan == nullptr) return false;
    // Opaque ordinary kernels do not acquire a notification scheduling contract.
    if (kernel_flags == UINT32_MAX || (kernel_flags & (NOTIFICATION_KERNEL_NOTIFY | NOTIFICATION_KERNEL_WAIT)) == 0)
        return true;
    const size_t before = plan->events.size();
    const auto status =
        query == nullptr ?
            NotificationPlanStatus::OPAQUE :
            scheduler_collect_notification_metadata(query, kernel_id, task_ref, input, &plan->events, error);
    if (status != NotificationPlanStatus::COMPLETE) {
        plan->unavailable.push_back({task_ref, kernel_id, input.block_index, input.lane, kernel_flags, status});
        if (status == NotificationPlanStatus::INVALID) {
            if (error != nullptr && (query == nullptr || error->empty()))
                *error = "defer_wait metadata unavailable at task " + std::to_string(task_ref) + " kernel " +
                         std::to_string(kernel_id);
            return false;
        }
        return status == NotificationPlanStatus::OPAQUE || status == NotificationPlanStatus::UNSUPPORTED;
    }
    for (size_t i = before; i < plan->events.size(); ++i) {
        const uint32_t required = plan->events[i].effect.kind == NotificationEventKind::WAIT_GE ?
                                      NOTIFICATION_KERNEL_WAIT :
                                      NOTIFICATION_KERNEL_NOTIFY;
        if ((kernel_flags & required) == 0) {
            plan->events.resize(before);
            if (error != nullptr)
                *error = "notification event contradicts kernel flags at task " + std::to_string(task_ref);
            return false;
        }
    }
    return true;
}
