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

#include <cstring>

#include "common/hbg_execution_image.h"

template <typename Invalidate, typename Flush>
int32_t restore_hbg_execution_image(const void *packet, Invalidate invalidate, Flush flush) {
    if (packet == nullptr) return 0;
    HbgExecutionImage image{};
    invalidate(packet, sizeof(image));
    std::memcpy(&image, packet, sizeof(image));
    if (image.working_address == 0 || image.bytes == 0 || image.bytes > UINT32_MAX) return -1;
    const auto *source = static_cast<const unsigned char *>(packet) + sizeof(image);
    auto *target = reinterpret_cast<void *>(image.working_address);
    invalidate(source, static_cast<size_t>(image.bytes));
    std::memcpy(target, source, static_cast<size_t>(image.bytes));
    flush(target, static_cast<size_t>(image.bytes));
    return 0;
}
