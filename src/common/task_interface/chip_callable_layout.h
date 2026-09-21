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
/**
 * Platform-agnostic ChipCallable layout / content-hash / function-table helpers
 * used by DeviceRunner::upload_chip_callable_buffer on every platform variant.
 *
 * The byte-size math (mirroring make_callable<>()'s layout), the FNV-1a dedup
 * hash, and the derivation of the callable's two func_id-indexed tables are
 * identical on onboard and sim. Only the H2D mechanism diverges: onboard
 * rtMemcpy's the scratch into device GM after rewriting each child's
 * resolved_addr_ to a device offset; sim instead dlopen's each child kernel
 * and writes the resulting function pointer into resolved_addr_. The
 * device-offset patch is exposed here so onboard can share it; the dlopen
 * path stays in sim's device_runner.cpp.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include "callable.h"
#include "utils/fnv1a_64.h"

struct ChipCallableLayout {
    size_t header_size;          // offsetof(ChipCallable, storage_)
    size_t total_size;           // header_size + storage_used (matches make_callable())
    uint64_t content_hash;       // FNV-1a 64 over [callable, total_size)
    uint64_t aicore_image_hash;  // FNV-1a 64 over func ids and child binaries
};

/**
 * Compute byte-size and content hash for a ChipCallable buffer.
 *
 * `storage_used` is max(binary_size, child_offset[i] + CoreCallable header +
 * child binary_size) over all children — same arithmetic make_callable<>()
 * uses when emitting the host buffer.
 */
inline ChipCallableLayout compute_chip_callable_layout(const ChipCallable *callable) {
    constexpr size_t kHeaderSize = offsetof(ChipCallable, storage_);
    size_t storage_used = static_cast<size_t>(callable->binary_size());
    const int32_t child_count = callable->child_count();
    uint64_t aicore_image_hash = simpler::common::utils::fnv1a_64(&child_count, sizeof(child_count));
    for (int32_t i = 0; i < callable->child_count(); ++i) {
        const CoreCallable &c = callable->child(i);
        const int32_t func_id = callable->child_func_id(i);
        const uint32_t binary_size = c.binary_size();
        aicore_image_hash = simpler::common::utils::fnv1a_64_append(aicore_image_hash, &func_id, sizeof(func_id));
        aicore_image_hash =
            simpler::common::utils::fnv1a_64_append(aicore_image_hash, &binary_size, sizeof(binary_size));
        aicore_image_hash = simpler::common::utils::fnv1a_64_append(
            aicore_image_hash, c.binary_data(), static_cast<size_t>(binary_size)
        );
        size_t child_total = CoreCallable::binary_data_offset() + static_cast<size_t>(c.binary_size());
        size_t end = static_cast<size_t>(callable->child_offset(i)) + child_total;
        if (end > storage_used) storage_used = end;
    }
    const size_t total_size = kHeaderSize + storage_used;
    const uint64_t hash = simpler::common::utils::fnv1a_64(reinterpret_cast<const uint8_t *>(callable), total_size);
    return ChipCallableLayout{kHeaderSize, total_size, hash, aicore_image_hash};
}

/**
 * Length of the dense func_id-indexed table `callable`'s children need: one
 * past their largest func_id, or 0 for a callable with no children.
 *
 * `max_func_id` is the exclusive bound the consuming runtime's tables can
 * address. Returns false without writing `*length` when a child names a
 * func_id outside [0, max_func_id), reporting it in `*bad_func_id` so the
 * caller can refuse the registration before it allocates anything.
 */
inline bool
chip_callable_table_length(const ChipCallable *callable, uint32_t max_func_id, uint32_t *length, int32_t *bad_func_id) {
    uint32_t longest = 0;
    for (int32_t i = 0; i < callable->child_count(); ++i) {
        const int32_t func_id = callable->child_func_id(i);
        if (func_id < 0 || static_cast<uint32_t>(func_id) >= max_func_id) {
            *bad_func_id = func_id;
            return false;
        }
        const uint32_t needed = static_cast<uint32_t>(func_id) + 1;
        if (needed > longest) longest = needed;
    }
    *length = longest;
    return true;
}

/**
 * Fill the two func_id-indexed views of `callable` from `scratch`, a byte copy
 * of it whose children's resolved_addr_ already hold the address the dispatch
 * path will use.
 *
 * `object_base` is the address that scratch becomes readable at — a device base
 * onboard, the scratch's own host address in sim — so `object[func_id]` is
 * where that child's CoreCallable lands there. `entry[func_id]` takes the
 * child's resolved_addr_ verbatim, which is that same object plus
 * CoreCallable::binary_data_offset() onboard and the dlopen'd host function
 * pointer in sim; one formula therefore serves both platforms. Entries no child
 * claims stay 0, which is what an unmapped func_id must read.
 *
 * Both spans hold `length` entries, as chip_callable_table_length() computed
 * for this same callable. `entry` may be null on a runtime whose consumers
 * resolve the entry out of the object themselves.
 */
inline void chip_callable_fill_tables(
    const ChipCallable *callable, const ChipCallableLayout &layout, const uint8_t *scratch, uint64_t object_base,
    uint32_t length, uint64_t *object, uint64_t *entry
) {
    for (uint32_t i = 0; i < length; ++i) {
        object[i] = 0;
        if (entry != nullptr) entry[i] = 0;
    }
    for (int32_t i = 0; i < callable->child_count(); ++i) {
        const uint32_t off = callable->child_offset(i);
        const auto *child = reinterpret_cast<const CoreCallable *>(scratch + layout.header_size + off);
        const uint32_t func_id = static_cast<uint32_t>(callable->child_func_id(i));
        object[func_id] = object_base + layout.header_size + off;
        if (entry != nullptr) entry[func_id] = child->resolved_addr();
    }
}

/**
 * Onboard-style scratch patch: rewrite each child's resolved_addr_ in the
 * host scratch buffer to the device-side code address of the child's binary,
 * computed as `target_base + header_size + child_offset(i) +
 * CoreCallable::binary_data_offset()`.
 *
 * `scratch` already holds a byte-copy of `callable` of `layout.total_size`
 * bytes; this helper only flips the child resolved_addr_ words. Sim does not
 * call this — it writes host function pointers into resolved_addr_ instead.
 */
inline void patch_chip_callable_scratch_for_device(
    const ChipCallable *callable, const ChipCallableLayout &layout, uint64_t target_base, uint8_t *scratch
) {
    for (int32_t i = 0; i < callable->child_count(); ++i) {
        const uint32_t off = callable->child_offset(i);
        auto *child = reinterpret_cast<CoreCallable *>(scratch + layout.header_size + off);
        uint64_t child_dev = target_base + layout.header_size + off;
        child->set_resolved_addr(child_dev + CoreCallable::binary_data_offset());
    }
}
