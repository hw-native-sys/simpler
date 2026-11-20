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

#include "common/launch_entry_args.h"

template <typename Invalidate, typename Flush>
int32_t restore_hbg_execution_image(const void *packet, size_t packet_bytes, Invalidate invalidate, Flush flush) {
    if (packet == nullptr) return packet_bytes == 0 ? 0 : -1;
    if (packet_bytes < sizeof(HbgExecutionImage) || packet_bytes > UINT32_MAX) return -1;
    HbgExecutionImage image{};
    invalidate(packet, sizeof(image));
    std::memcpy(&image, packet, sizeof(image));
    if (image.working_address == 0 || image.bytes == 0 || image.bytes > packet_bytes - sizeof(image) ||
        image.mutable_prefix_bytes > image.bytes ||
        image.reference_count > (packet_bytes - sizeof(image) - image.bytes) / sizeof(HbgImageReference) ||
        sizeof(image) + image.bytes + image.reference_count * sizeof(HbgImageReference) != packet_bytes)
        return -1;
    const auto *source = static_cast<const unsigned char *>(packet) + sizeof(image);
    auto *target = reinterpret_cast<void *>(image.working_address);
    // The scheduler passes aligned tensor descriptors to AICore callables.
    if (image.reference_count != 0 && reinterpret_cast<uintptr_t>(source) % HBG_IMAGE_ALIGNMENT != 0) return -1;
    invalidate(source, image.bytes + image.reference_count * sizeof(HbgImageReference));
    if (image.mutable_prefix_bytes != 0) {
        std::memcpy(target, source, image.mutable_prefix_bytes);
        flush(target, image.mutable_prefix_bytes);
    }
    const auto *references = source + image.bytes;
    for (size_t i = 0; i < image.reference_count; ++i) {
        HbgImageReference ref{};
        std::memcpy(&ref, references + i * sizeof(ref), sizeof(ref));
        if (ref.device_field_address == 0 || ref.device_field_address % alignof(uint64_t) != 0 ||
            ref.image_offset >= image.bytes)
            return -1;
        auto *field = reinterpret_cast<uint64_t *>(ref.device_field_address);
        invalidate(field, sizeof(*field));
        *field = reinterpret_cast<uint64_t>(source + ref.image_offset);
        flush(field, sizeof(*field));
    }
    return 0;
}
