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

// Percounter's AICPU Graph materializer constructs ChipTaskStorage records.
inline constexpr uint32_t SCHEDULER_GRAPH_TASK_STORAGE_STRIDE = 320;
inline constexpr uint32_t SCHEDULER_GRAPH_DESCRIPTOR_OFFSET = 0;
inline constexpr uint32_t SCHEDULER_GRAPH_PAYLOAD_OFFSET = 128;
inline constexpr uint32_t SCHEDULER_GRAPH_TASK_ID_OFFSET = 0;
inline constexpr uint32_t SCHEDULER_GRAPH_KERNEL_IDS_OFFSET = 8;
inline constexpr uint32_t SCHEDULER_GRAPH_FANIN_COUNT_OFFSET = 8;
inline constexpr uint32_t SCHEDULER_GRAPH_PREDICATE_OFFSET = 128;
inline constexpr uint32_t TASK_IMAGE_TENSOR_COUNT_OFFSET = 0;
inline constexpr uint32_t TASK_IMAGE_SCALAR_COUNT_OFFSET = 4;
inline constexpr uint32_t TASK_IMAGE_TENSORS_DELTA_OFFSET = 12;
inline constexpr uint32_t TASK_IMAGE_SCALARS_DELTA_OFFSET = 16;
inline constexpr uint32_t TASK_IMAGE_FANIN_DELTA_OFFSET = 20;
inline constexpr uint32_t TASK_IMAGE_ERROR_CODE_OFFSET = 68;
