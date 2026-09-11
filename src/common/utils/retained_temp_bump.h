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

#include "common/host_api.h"

/**
 * Per-run bump allocator over the runner's retained temporary buffer, shared by
 * every host runtime that stages device arguments.
 *
 * This is the whole temporary-buffer mechanism: the platform only remembers a
 * {addr, size} slot across runs (HostApi get/set_retained_temp_buffer); the
 * grow and slice logic lives here. A bind packs its own staged tensors to a
 * required size, calls begin() to grow the slot to it, and acquire()s one slice
 * per tensor. The slot is per pipeline slot, so two runs holding different slot
 * leases never share a staging buffer.
 *
 * Kernels require 1024-byte-aligned device pointers, and this class is what
 * makes every slice one. It does not assume the backend returns an aligned
 * allocation: onboard device_malloc happens to, but the sim backend is
 * std::malloc (src/common/platform/sim/host/memory_allocator.cpp), which
 * guarantees only max_align_t. begin() therefore over-allocates by
 * kAlignment - 1 and hands out an aligned base inside it, so aligning the slice
 * offsets is enough — a misaligned base cannot reach a caller.
 *
 * The slot keeps the raw allocation, not the aligned base: that is the pointer
 * device_free must receive, and re-aligning it on every begin() is free.
 *
 * Reports failure through return values rather than logging: this header is
 * included by translation units that do not all share one logging backend, and
 * the caller has the tensor index the message needs anyway.
 */
class RetainedTempBump {
public:
    static constexpr size_t kAlignment = 1024;

    // Wraps to a smaller value for v > SIZE_MAX - kAlignment + 1, which a real
    // byte count never reaches. Left unchecked because acquire() is where an
    // undersized total has to be caught anyway: a checked form here would only
    // move the same failure earlier, at the cost of an error path through every
    // caller's packing loop.
    static size_t align_up(size_t v) { return (v + (kAlignment - 1)) & ~(kAlignment - 1); }

    /**
     * Grow the retained slot to `required` bytes if it is too small (free old +
     * malloc new + write back) and reset the slice cursor.
     *
     * @param required  packed size of this run's slices, each aligned up to
     *                  kAlignment by the caller
     * @return false only if the (grow) device_malloc fails; a run needing 0
     *         bytes leaves the slot untouched and succeeds
     */
    bool begin(const HostApi *api, size_t required) {
        offset_ = 0;
        base_ = nullptr;
        capacity_ = 0;

        void *raw = nullptr;
        size_t size = 0;
        api->get_retained_temp_buffer(&raw, &size);

        if (required != 0) {
            // The extra kAlignment - 1 is the headroom the base alignment below
            // may consume, so a buffer of this size yields `required` usable
            // bytes whatever address the backend returns.
            const size_t wanted = required + kAlignment - 1;
            if (wanted > size) {
                if (raw != nullptr) {
                    api->device_free(raw);
                }
                raw = api->device_malloc(wanted);
                if (raw == nullptr) {
                    // The old buffer is already released, so the slot must stop
                    // naming it — a later run would otherwise free it twice.
                    api->set_retained_temp_buffer(nullptr, 0);
                    return false;
                }
                api->set_retained_temp_buffer(raw, wanted);
                size = wanted;
            }
        }
        if (raw == nullptr) {
            return true;  // nothing retained, and this run needs nothing
        }

        // Align inside the allocation rather than trusting the backend to have
        // aligned it. The slot keeps `raw` — that is the pointer device_free
        // must get — and the aligned base is re-derived on every begin().
        const uintptr_t raw_addr = reinterpret_cast<uintptr_t>(raw);
        const uintptr_t base_addr = static_cast<uintptr_t>(align_up(static_cast<size_t>(raw_addr)));
        const size_t head = static_cast<size_t>(base_addr - raw_addr);
        if (head < size) {
            base_ = reinterpret_cast<void *>(base_addr);
            capacity_ = size - head;
        }
        return true;
    }

    /**
     * Slice `bytes` from the retained buffer at the next kAlignment-aligned
     * offset. Must fit because begin() was given the packed size of the same
     * slices; a miss is a caller bug (plan/slice mismatch), reported as nullptr.
     *
     * This is the only bound the returned pointer has, so it is written as a
     * subtraction rather than as `aligned + bytes > capacity_`: a caller's byte
     * count comes from `ChipTensor::nbytes()`, an unchecked `uint64_t` product,
     * and a sum that wrapped would compare small and hand back a pointer past
     * the buffer. Every overflow upstream — in a size, in a packed total, or in
     * align_up itself — therefore lands here as a miss instead.
     */
    void *acquire(size_t bytes) {
        const size_t aligned = align_up(offset_);
        if (base_ == nullptr || aligned > capacity_ || bytes > capacity_ - aligned) {
            return nullptr;
        }
        void *ptr = static_cast<char *>(base_) + aligned;
        // Bounded by the test above, so this cannot wrap: aligned + bytes <= capacity_.
        offset_ = aligned + bytes;
        return ptr;
    }

    /** Offset the next acquire() would slice at; names the miss in a caller's diagnostic. */
    size_t next_offset() const { return align_up(offset_); }
    size_t capacity() const { return capacity_; }

private:
    void *base_ = nullptr;
    size_t capacity_ = 0;
    size_t offset_ = 0;
};
