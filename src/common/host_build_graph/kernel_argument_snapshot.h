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

#include "task_interface/task_args.h"
#include "utils/fnv1a_64.h"

namespace hbg {

template <typename T>
inline uint64_t hash_argument_value(uint64_t hash, const T &value) noexcept {
    return simpler::common::utils::fnv1a_64_append(hash, &value, sizeof(value));
}

// Hash the semantic argument snapshot baked into an HBG graph. Padding and
// unused dimensions are deliberately excluded from the identity.
inline uint64_t kernel_argument_snapshot_hash(const ChipStorageTaskArgs &args) noexcept {
    if (args.tensor_count_ < 0 || args.tensor_count_ > CHIP_MAX_TENSOR_ARGS || args.scalar_count_ < 0 ||
        args.scalar_count_ > CHIP_MAX_SCALAR_ARGS)
        return 0;
    uint64_t hash = simpler::common::utils::fnv1a_64(&args.tensor_count_, sizeof(args.tensor_count_));
    hash = hash_argument_value(hash, args.scalar_count_);
    for (int32_t i = 0; i < args.tensor_count_; ++i) {
        const ChipTensor &tensor = args.tensor(i);
        if (tensor.ndims == 0 || tensor.ndims > MAX_TENSOR_DIMS) return 0;
        hash = hash_argument_value(hash, tensor.buffer.addr);
        hash = hash_argument_value(hash, tensor.buffer.size);
        hash = hash_argument_value(hash, tensor.start_offset);
        hash = hash_argument_value(hash, tensor.ndims);
        const uint8_t dtype = static_cast<uint8_t>(tensor.dtype);
        hash = hash_argument_value(hash, dtype);
        hash = hash_argument_value(hash, tensor.address_space);
        for (uint32_t d = 0; d < tensor.ndims; ++d)
            hash = hash_argument_value(hash, tensor.shapes[d]);
        for (uint32_t d = 0; d < tensor.ndims; ++d)
            hash = hash_argument_value(hash, tensor.strides[d]);
    }
    for (int32_t i = 0; i < args.scalar_count_; ++i)
        hash = hash_argument_value(hash, args.scalar(i));
    return hash;
}

inline bool same_kernel_argument_snapshot(const ChipStorageTaskArgs &a, const ChipStorageTaskArgs &b) noexcept {
    if (a.tensor_count_ != b.tensor_count_ || a.scalar_count_ != b.scalar_count_ ||
        kernel_argument_snapshot_hash(a) == 0 || kernel_argument_snapshot_hash(b) == 0)
        return false;
    for (int32_t i = 0; i < a.tensor_count_; ++i) {
        const ChipTensor &x = a.tensor(i);
        const ChipTensor &y = b.tensor(i);
        if (x.buffer.addr != y.buffer.addr || x.buffer.size != y.buffer.size || x.start_offset != y.start_offset ||
            x.ndims != y.ndims || x.dtype != y.dtype || x.address_space != y.address_space)
            return false;
        for (uint32_t d = 0; d < x.ndims; ++d)
            if (x.shapes[d] != y.shapes[d] || x.strides[d] != y.strides[d]) return false;
    }
    for (int32_t i = 0; i < a.scalar_count_; ++i)
        if (a.scalar(i) != b.scalar(i)) return false;
    return true;
}

}  // namespace hbg
