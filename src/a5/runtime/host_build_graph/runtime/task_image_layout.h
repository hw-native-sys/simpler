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

#include "task_image_types.h"

inline constexpr uint32_t SCHEDULER_GRAPH_TASK_STORAGE_STRIDE = sizeof(TaskImageStorage);
inline constexpr uint32_t SCHEDULER_GRAPH_DESCRIPTOR_OFFSET = offsetof(TaskImageStorage, task);
inline constexpr uint32_t SCHEDULER_GRAPH_PAYLOAD_OFFSET = offsetof(TaskImageStorage, payload);
inline constexpr uint32_t SCHEDULER_GRAPH_TASK_ID_OFFSET = offsetof(TaskImageDescriptor, task_id);
inline constexpr uint32_t SCHEDULER_GRAPH_KERNEL_IDS_OFFSET = offsetof(TaskImageDescriptor, kernel_id);
inline constexpr uint32_t SCHEDULER_GRAPH_FANIN_COUNT_OFFSET = offsetof(TaskImagePayload, fanin_count);
inline constexpr uint32_t SCHEDULER_GRAPH_PREDICATE_OFFSET = offsetof(TaskImagePayload, predicate);
inline constexpr uint32_t TASK_IMAGE_TENSOR_COUNT_OFFSET = offsetof(TaskImagePayload, tensor_count);
inline constexpr uint32_t TASK_IMAGE_SCALAR_COUNT_OFFSET = offsetof(TaskImagePayload, scalar_count);
inline constexpr uint32_t TASK_IMAGE_TENSORS_DELTA_OFFSET = offsetof(TaskImagePayload, tensors);
inline constexpr uint32_t TASK_IMAGE_SCALARS_DELTA_OFFSET = offsetof(TaskImagePayload, scalars);
inline constexpr uint32_t TASK_IMAGE_FANIN_DELTA_OFFSET = offsetof(TaskImagePayload, fanin);
inline constexpr uint32_t TASK_IMAGE_ERROR_CODE_OFFSET = 0;
