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
#include <cstring>
#include <type_traits>

#include "graph_execution.h"

/**
 * One reading thread's view of the Graph Definition section, decoded by value.
 *
 * The section is the tail of this run's AICPU launch package, which RTS copies,
 * or — in simulation, where there is no launch package — the run-owned host
 * snapshot its runner publishes. Either way the bytes belong to whoever
 * delivered them and this view only reads them.
 *
 * Two properties make it a view rather than a pointer holder:
 *
 *   - **No typed pointer is ever formed into the section.** Every access is a
 *     fixed-length `memcpy` into a caller-supplied aligned local, which is
 *     defined for any source address, so the section's own alignment is not a
 *     precondition and a legal package at any base decodes.
 *   - **It is never stored in shared state.** A view names the launch arguments
 *     of the thread that built it, and that thread may return while its peers
 *     are still reading, so a shared copy would outlive its bytes. Each reader
 *     builds its own and resolves the offsets shared state carries.
 *
 * Every read is bounded twice: against the section, and against the framed
 * Definition the offset claims to be inside. A field that points into a
 * different valid object in the same section is as invalid as one pointing
 * past the end.
 */
struct GraphImageView {
    const std::byte *section{nullptr};
    uint32_t bytes{0};

    bool valid() const { return section != nullptr && bytes >= sizeof(GraphDefinitionHeader); }

    /** Copy `len` bytes from `offset`, or fail without touching `dst`. */
    bool read(uint32_t offset, void *dst, uint32_t len) const {
        if (!valid() || dst == nullptr || len == 0) return false;
        if (offset > bytes || len > bytes - offset) return false;
        std::memcpy(dst, section + offset, len);
        return true;
    }

    /** Copy one trivially-copyable value into an aligned local. */
    template <typename T>
    bool load(uint32_t offset, T *out) const {
        static_assert(std::is_trivially_copyable_v<T>, "a Definition value must be memcpy-able");
        // The read length is a uint32_t, so the narrowing below has to be
        // lossless for the bounds `read` applies to it to be the ones this
        // value needs.
        constexpr size_t value_bytes = sizeof(T);
        static_assert(
            value_bytes == static_cast<size_t>(static_cast<uint32_t>(value_bytes)),
            "a Definition value must fit the section's read length"
        );
        return read(offset, out, static_cast<uint32_t>(value_bytes));
    }
};

/**
 * One framed Definition object, decoded out of a section by value.
 *
 * `image_offset` is where the Definition image starts, which is what an outer
 * Graph task carries and what every section offset below is relative to. The
 * framing header sits immediately in front of it.
 */
struct GraphDefinitionValue {
    uint32_t image_offset{0};
    GraphDefinition definition{};

    bool valid() const { return image_offset != 0 && definition.total_bytes >= sizeof(GraphDefinition); }

    /** Absolute section offset of `offset_in_image`, checked against this object and the section. */
    bool absolute(const GraphImageView &view, uint32_t offset_in_image, uint32_t len, uint32_t *out) const {
        if (out == nullptr || !valid() || !view.valid() || len == 0) return false;
        // Inside the image: the header's own offsets are relative to the image
        // base, so a zero is "absent" and anything past total_bytes is corrupt.
        if (offset_in_image == 0 || offset_in_image > definition.total_bytes) return false;
        if (len > definition.total_bytes - offset_in_image) return false;
        // And inside the section: the two bounds are independent, because a
        // corrupt offset can land inside another object that is itself framed.
        if (image_offset > view.bytes || definition.total_bytes > view.bytes - image_offset) return false;
        *out = image_offset + offset_in_image;
        return true;
    }
};

/**
 * Decode and validate one framed Definition.
 *
 * This replaces the pointer-returning framing check for a section that must not
 * be dereferenced: the header and the image are copied into locals first and
 * the three framing facts are then compared between those two values — the
 * object is one of ours, it is the size the packing recorded, and it holds the
 * Graph the caller named.
 */
inline bool
graph_definition_decode_framed(const GraphImageView &view, uint32_t image_offset, GraphDefinitionValue *out) {
    if (out == nullptr || !view.valid()) return false;
    if (image_offset < sizeof(GraphDefinitionHeader)) return false;
    GraphDefinitionHeader header{};
    if (!view.load(image_offset - static_cast<uint32_t>(sizeof(GraphDefinitionHeader)), &header)) return false;
    if (header.magic != GRAPH_DEFINITION_OBJECT_MAGIC) return false;
    if (header.definition_bytes < sizeof(GraphDefinition)) return false;
    GraphDefinition definition{};
    if (!view.load(image_offset, &definition)) return false;
    if (definition.total_bytes != header.definition_bytes) return false;
    if (definition.graph_key != header.graph_key) return false;
    // The whole image has to be inside the section before any of its offsets is
    // used, so a truncated package is rejected here rather than per section.
    if (image_offset > view.bytes || definition.total_bytes > view.bytes - image_offset) return false;
    out->image_offset = image_offset;
    out->definition = definition;
    return true;
}

/**
 * Copy one element of an image array into an aligned local.
 *
 * `offset_in_image` is the section's own `off_*` field and `count` its length,
 * both taken from the decoded header, so the element's extent is checked
 * against the array, the image and the section before a byte is read.
 */
template <typename T>
inline bool graph_definition_load_element(
    const GraphImageView &view, const GraphDefinitionValue &object, uint32_t offset_in_image, int32_t count,
    int32_t index, T *out
) {
    static_assert(std::is_trivially_copyable_v<T>, "a Definition array element must be memcpy-able");
    if (out == nullptr || count <= 0 || index < 0 || index >= count) return false;
    // Element extent first, so the multiplication cannot wrap before the bound
    // that would have caught it: total_bytes is a uint32 and sizeof(T) a small
    // constant, so 64-bit arithmetic covers every representable array.
    const uint64_t array_bytes = static_cast<uint64_t>(count) * sizeof(T);
    const uint64_t element_offset = static_cast<uint64_t>(offset_in_image) + static_cast<uint64_t>(index) * sizeof(T);
    if (array_bytes > object.definition.total_bytes) return false;
    if (element_offset + sizeof(T) > static_cast<uint64_t>(offset_in_image) + array_bytes) return false;
    if (element_offset > UINT32_MAX) return false;
    uint32_t absolute = 0;
    if (!object.absolute(view, static_cast<uint32_t>(element_offset), static_cast<uint32_t>(sizeof(T)), &absolute)) {
        return false;
    }
    return view.load(absolute, out);
}

/**
 * Copy a whole image array into caller storage.
 *
 * Used where an array is read often enough that the execution keeps its own
 * copy — the fanin CSR — rather than decoding per element. The destination is
 * the caller's; this only bounds the source.
 */
template <typename T>
inline bool graph_definition_copy_array(
    const GraphImageView &view, const GraphDefinitionValue &object, uint32_t offset_in_image, int32_t count, T *dst
) {
    static_assert(std::is_trivially_copyable_v<T>, "a Definition array element must be memcpy-able");
    if (dst == nullptr || count < 0) return false;
    if (count == 0) return true;
    const uint64_t array_bytes = static_cast<uint64_t>(count) * sizeof(T);
    if (array_bytes > UINT32_MAX) return false;
    uint32_t absolute = 0;
    if (!object.absolute(view, offset_in_image, static_cast<uint32_t>(array_bytes), &absolute)) return false;
    return view.read(absolute, dst, static_cast<uint32_t>(array_bytes));
}
