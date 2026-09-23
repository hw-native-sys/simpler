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
 * Where the `tmr` runtime keeps the arguments its orchestration entry was called
 * with.
 *
 * This header is runtime-internal and separate from tensor.h. It reaches only
 * this runtime's `Tensor` and the capacity constants, never the L3+ argument
 * surface in src/common/task_interface/task_args.h and so never the address-free
 * wire `Tensor` that buffer.h declares at global scope. A kernel translation
 * unit needs this runtime's `Tensor` but none of that, and gets it from
 * tensor.h, which stays clear of both.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>

#include "task_interface/arg_direction.h"
#include "tensor.h"

namespace simpler::tmr {

/**
 * Entry arguments as this runtime's device descriptor carries them.
 *
 * `Runtime::set_orch_args` adopts the boundary ChipStorageTaskArgs into this
 * once, on the host, before orchestration runs; from there inward nothing holds
 * a bare ChipTensor.
 *
 * Member order is the upload contract, not a preference. The scalars and the
 * two counts come first and `tensors_` last, so the bytes a run actually filled
 * are one contiguous range starting at offset 0 — which is what lets
 * `runtime_device_copy_size` stop after `tensor_count_` tensors and still hand
 * `rtMemcpy` a single prefix. The tensor array is the reason: it is
 * `CHIP_MAX_TENSOR_ARGS` 128-byte slots, two orders of magnitude larger than
 * everything else in the descriptor put together, and an entry typically fills
 * a handful. The scalars are copied in full at 1 KiB rather than bounded too,
 * because a second bounded array would need a second range and the whole point
 * is to keep one copy.
 *
 * This is deliberately not `TaskArgsTpl`: that template puts its tensors first,
 * it is shared with host_build_graph, the mailbox and the task-args wire, and
 * the only thing this needs from it is the six accessors below. Reordering it
 * there to serve one descriptor would move layout under all of them.
 *
 * Slots at or past `tensor_count_` hold whatever an earlier run of the same
 * allocation left, because a short publication does not reach them. Every
 * consumer is bounded by the counts and must stay so; see
 * `ChipTaskArgs::create_from_entry_storage`.
 */
struct alignas(64) EntryArgsStorage {
    uint64_t scalars_[CHIP_MAX_SCALAR_ARGS];
    int32_t tensor_count_{0};
    int32_t scalar_count_{0};
    // Last, and 64-byte aligned in its own right: `used_prefix_bytes()` returns
    // an offset into this array, so its start is the fixed part of that length.
    alignas(64) Tensor tensors_[CHIP_MAX_TENSOR_ARGS];

    void add_tensor(const Tensor &t) {
        if (scalar_count_ > 0) throw std::logic_error("TaskArgs: cannot add tensor after scalar");
        if (static_cast<size_t>(tensor_count_) >= CHIP_MAX_TENSOR_ARGS) {
            throw std::out_of_range("TaskArgs: tensor capacity exceeded");
        }
        tensors_[tensor_count_++] = t;
    }

    void add_scalar(uint64_t s) {
        if (static_cast<size_t>(scalar_count_) >= CHIP_MAX_SCALAR_ARGS) {
            throw std::out_of_range("TaskArgs: scalar capacity exceeded");
        }
        scalars_[scalar_count_++] = s;
    }

    const Tensor &tensor(int32_t i) const { return tensors_[i]; }
    uint64_t scalar(int32_t i) const { return scalars_[i]; }

    int32_t tensor_count() const { return tensor_count_; }
    int32_t scalar_count() const { return scalar_count_; }

    void clear() {
        tensor_count_ = 0;
        scalar_count_ = 0;
    }

    /**
     * Bytes from the start of this object that hold values this object's own
     * `add_*` calls wrote: everything up to `tensors_`, plus the filled tensor
     * slots. A count outside [0, CHIP_MAX_TENSOR_ARGS] yields the full object,
     * so a descriptor whose counts were never set — or were corrupted — is
     * published whole rather than truncated to a length derived from a bad
     * value. `add_tensor` already refuses to produce such a count; this is the
     * arithmetic's own bound, not a second gate.
     */
    size_t used_prefix_bytes() const {
        if (tensor_count_ < 0 || static_cast<size_t>(tensor_count_) > CHIP_MAX_TENSOR_ARGS) {
            return sizeof(EntryArgsStorage);
        }
        return offsetof(EntryArgsStorage, tensors_) + static_cast<size_t>(tensor_count_) * sizeof(Tensor);
    }

    /**
     * Bytes a launch package carries for `tensor_count` tensors and
     * `scalar_count` scalars: the values themselves, with none of this object's
     * padding or unfilled slots.
     */
    static constexpr size_t wire_bytes(uint32_t tensor_count, uint32_t scalar_count) {
        return static_cast<size_t>(tensor_count) * sizeof(Tensor) +
               static_cast<size_t>(scalar_count) * sizeof(uint64_t);
    }

    /**
     * Adopt `tensor_count` tensors followed by `scalar_count` scalars from raw
     * wire bytes.
     *
     * `payload` is read as bytes and never addressed as a `Tensor *`: it points
     * into a buffer whose base alignment belongs to whoever allocated it, and
     * this type is 64-byte aligned. The copy target carries that alignment
     * because it is this object, so the values are typed only after landing
     * here — the same rule `TaskArgsView::tensors` follows for the L3+ blob.
     *
     * Either both counts fit or nothing is written: a partially decoded entry
     * would leave counts and values disagreeing, which every consumer trusts.
     * Returns whether the counts were in range.
     */
    bool load_from_wire(const void *payload, uint32_t tensor_count, uint32_t scalar_count) {
        if (tensor_count > CHIP_MAX_TENSOR_ARGS || scalar_count > CHIP_MAX_SCALAR_ARGS) return false;
        if (payload == nullptr && wire_bytes(tensor_count, scalar_count) != 0) return false;
        const auto *bytes = static_cast<const unsigned char *>(payload);
        if (tensor_count > 0) {
            memcpy(tensors_, bytes, static_cast<size_t>(tensor_count) * sizeof(Tensor));
        }
        if (scalar_count > 0) {
            memcpy(
                scalars_, bytes + static_cast<size_t>(tensor_count) * sizeof(Tensor),
                static_cast<size_t>(scalar_count) * sizeof(uint64_t)
            );
        }
        tensor_count_ = static_cast<int32_t>(tensor_count);
        scalar_count_ = static_cast<int32_t>(scalar_count);
        return true;
    }
};

static_assert(
    std::is_standard_layout_v<EntryArgsStorage> && std::is_trivially_copyable_v<EntryArgsStorage>,
    "EntryArgsStorage travels to the device inside the run descriptor's rtMemcpy"
);
// `used_prefix_bytes()` is an offset into `tensors_`, so the array's alignment
// is what keeps every length it returns a whole number of cache lines.
static_assert(alignof(Tensor) == 64 && offsetof(EntryArgsStorage, tensors_) % 64 == 0);
// The tensors are last: a length that stops inside them stops inside the object,
// and the full count reaches exactly its end. Both halves of the upload contract.
static_assert(
    offsetof(EntryArgsStorage, tensors_) + CHIP_MAX_TENSOR_ARGS * sizeof(Tensor) == sizeof(EntryArgsStorage),
    "tensors_ must be the last member, so a count-bounded prefix is contiguous from offset 0"
);

}  // namespace simpler::tmr
