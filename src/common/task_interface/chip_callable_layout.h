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
 * Platform-agnostic ChipCallable layout / content-hash helpers used by
 * DeviceRunner::upload_chip_callable_buffer on every platform variant.
 *
 * The byte-size math (mirroring make_callable<>()'s layout) and FNV-1a dedup
 * hash are identical on onboard and sim. Only the H2D mechanism diverges:
 * onboard rtMemcpy's the scratch into device GM after rewriting each child's
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
    size_t header_size;     // offsetof(ChipCallable, storage_)
    size_t total_size;      // header_size + storage_used (matches make_callable())
    uint64_t content_hash;  // FNV-1a 64 over [callable, total_size)
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
    for (int32_t i = 0; i < callable->child_count(); ++i) {
        const CoreCallable &c = callable->child(i);
        size_t child_total = CoreCallable::binary_data_offset() + static_cast<size_t>(c.binary_size());
        size_t end = static_cast<size_t>(callable->child_offset(i)) + child_total;
        if (end > storage_used) storage_used = end;
    }
    const size_t total_size = kHeaderSize + storage_used;
    const uint64_t hash = simpler::common::utils::fnv1a_64(reinterpret_cast<const uint8_t *>(callable), total_size);
    return ChipCallableLayout{kHeaderSize, total_size, hash};
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
