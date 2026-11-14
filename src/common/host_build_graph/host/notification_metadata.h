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

#include "host_build_graph/tensor_data.h"

namespace simpler::hbg {

inline constexpr uint32_t NOTIFICATION_PLAN_VERSION = 1;
inline constexpr uint32_t NOTIFICATION_KERNEL_NOTIFY = 1;
inline constexpr uint32_t NOTIFICATION_KERNEL_WAIT = 2;
inline constexpr uint32_t NOTIFICATION_KERNEL_OPAQUE = 4;

enum class NotificationPlanStatus : uint32_t { COMPLETE = 0, OPAQUE = 1, UNSUPPORTED = 2, INVALID = 3 };
enum class NotificationEventKind : uint32_t { SET = 0, ADD = 1, WAIT_GE = 2 };

// Host mirrors of invocation arguments. No callback may dereference a DEVICE
// tensor or a device communication-context address.
struct NotificationPlanInput {
    uint32_t version{NOTIFICATION_PLAN_VERSION};
    uint32_t tensor_count{0};
    uint32_t scalar_count{0};
    uint32_t block_index{0};
    uint32_t block_count{1};
    uint32_t lane{0};
    const TensorData *tensors{nullptr};
    const uint64_t *scalars{nullptr};
    bool (*query_context)(void *, uint64_t, int32_t *, uint32_t *){nullptr};
    void *context_user{nullptr};
};

// Offsets are logical coordinates in tensor_index's bound view. WAIT_GE's
// peer=-1 denotes its local window; SET/ADD preserve the original peer rank.
struct NotificationPlanEvent {
    NotificationEventKind kind{NotificationEventKind::WAIT_GE};
    uint32_t tensor_index{0};
    uint32_t ndims{0};
    int64_t offsets[MAX_TENSOR_DIMS]{};
    int64_t peer{-1};
    int64_t value{0};
    uint64_t comm_context{0};
};

using NotificationEventSink = bool (*)(void *, const NotificationPlanEvent *);
using NotificationPlanFunction =
    NotificationPlanStatus (*)(uint32_t, const NotificationPlanInput *, NotificationEventSink, void *);
using NotificationKernelFlagsFunction = uint32_t (*)(uint32_t);
}  // namespace simpler::hbg
