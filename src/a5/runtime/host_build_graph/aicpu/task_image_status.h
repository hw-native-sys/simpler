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

#include <atomic>
#include <cstdint>

#include "task_image_layout.h"

struct alignas(64) TaskImageStatus {
    std::atomic<int32_t> error_code{0};
};
static_assert(sizeof(TaskImageStatus) == 64);

inline std::atomic<int32_t> *task_image_error_code(void *image) {
    return image == nullptr ?
               nullptr :
               reinterpret_cast<std::atomic<int32_t> *>(static_cast<uint8_t *>(image) + TASK_IMAGE_ERROR_CODE_OFFSET);
}
