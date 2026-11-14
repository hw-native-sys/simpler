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

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "host_build_graph/task_id.h"
#include "host_build_graph/self_relative_ptr.h"
#include "task_image_predicate.h"

namespace simpler::hbg {
struct Tensor;
}

struct TaskImageDescriptor {
    TaskId task_id{};
    int32_t kernel_id[3]{-1, -1, -1};
};

struct TaskImagePayload {
    int32_t tensor_count{0};
    int32_t scalar_count{0};
    int32_t fanin_count{0};
    simpler::hbg::SelfRelativePtr<simpler::hbg::Tensor> tensors;
    simpler::hbg::SelfRelativePtr<uint64_t> scalars;
    simpler::hbg::SelfRelativePtr<int32_t> fanin;
    SchedulerDispatchPredicate predicate{};

    simpler::hbg::Tensor *tensor_data() const { return tensors.get(); }
    uint64_t *scalar_data() const { return scalars.get(); }
    int32_t *fanin_data() const { return fanin.get(); }
    void bind_regions(simpler::hbg::Tensor *tensor_region, uint64_t *scalar_region, int32_t *fanin_region) {
        tensors.set(tensor_region);
        scalars.set(scalar_region);
        fanin.set(fanin_region);
    }
};

struct alignas(16) TaskImageStorage {
    TaskImageDescriptor task;
    TaskImagePayload payload;
};

static_assert(sizeof(SchedulerDispatchPredicate) == 24);
static_assert(sizeof(TaskImageDescriptor) == 24);
static_assert(sizeof(TaskImagePayload) == 48);
static_assert(sizeof(TaskImageStorage) == 80);
static_assert(std::is_standard_layout_v<TaskImageStorage> && std::is_trivially_destructible_v<TaskImageStorage>);
