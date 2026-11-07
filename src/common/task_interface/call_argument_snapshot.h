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
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "arg_direction.h"
#include "tensor_geometry.h"
#include "task_args.h"

// Owns only call-value bytes. Admission, execution and retirement belong to the caller.
// Copies retain native storage; no Python object or GIL-dependent destructor is present.
class CallArgumentSnapshot {
public:
    CallArgumentSnapshot(const ChipStorageTaskArgs &args, const std::vector<ArgDirection> &directions) :
        args_(args) {
        for (int32_t i = 0; i < args_.tensor_count(); ++i) {
            auto &tensor = args_.tensor(i);
            if (tensor.address_space != AddressSpace::HOST || tensor.transfer != TensorTransfer::NONE) continue;
            if (static_cast<size_t>(i) >= directions.size())
                throw std::invalid_argument("HOST/NONE snapshot requires callable direction");
            // Writable HOST arguments retain their externally visible destination.
            if (directions[i] != ArgDirection::IN) continue;
            if (tensor.ndims == 0 || tensor.ndims > MAX_TENSOR_DIMS)
                throw std::invalid_argument("HOST control snapshot has invalid rank");
            const auto element_size = get_element_size(tensor.dtype);
            if (element_size == 0) throw std::invalid_argument("HOST control snapshot has invalid dtype");
            for (uint32_t dim = 0; dim < tensor.ndims; ++dim) {
                if (tensor.shapes[dim] == 0 || tensor.strides[dim] == 0)
                    throw std::invalid_argument("HOST control snapshot requires nonempty positive-stride views");
            }
            const auto offset = tensor_mul_sat(tensor.start_offset, element_size);
            const auto extent = view_extent_bytes(tensor.shapes, tensor.strides, tensor.ndims, tensor.dtype);
            const auto bytes = tensor_add_sat(offset, extent);
            if (bytes == TENSOR_EXTENT_UNREPRESENTABLE || bytes > tensor.buffer.size ||
                bytes > std::numeric_limits<size_t>::max() - sizeof(uint64_t) || tensor.buffer.addr == 0 ||
                tensor.buffer.addr > std::numeric_limits<uintptr_t>::max() - bytes)
                throw std::invalid_argument("HOST control snapshot exceeds backing bounds");
            auto storage = std::make_shared<std::vector<uint64_t>>((bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t));
            std::memcpy(storage->data(), reinterpret_cast<const void *>(tensor.buffer.addr), bytes);
            tensor.buffer.addr = reinterpret_cast<uintptr_t>(storage->data());
            tensor.buffer.size = bytes;
            storage_.push_back(std::move(storage));
        }
    }

    const ChipStorageTaskArgs &args() const { return args_; }

private:
    ChipStorageTaskArgs args_;
    std::vector<std::shared_ptr<std::vector<uint64_t>>> storage_;
};
