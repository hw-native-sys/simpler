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
 * TaskArgs - Tensor + scalar argument storage
 *
 * Template: TaskArgs<T, S, MaxT, MaxS, TensorTag=void>
 *   - Static:  MaxT>0, MaxS>0 — fixed-size arrays
 *   - Dynamic: MaxT==0, MaxS==0 — std::vector backed
 *
 * Enforces tensor-before-scalar ordering: once add_scalar() is called,
 * add_tensor() is no longer allowed.
 *
 * Optional TensorTag (e.g. TensorArgType for INPUT/OUTPUT/INOUT):
 *   - void (default): no per-tensor tag — pure transport/storage
 *   - real type: adds tags_ storage + tag(i) accessor
 *
 * Type aliases:
 *   ChipStorageTaskArgs = TaskArgs<ContinuousTensor, uint64_t, 16, 128>
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "tensor_arg.h"  // NOLINT(build/include_subdir)

// ============================================================================
// TensorTagMixin — conditionally provides per-tensor tag storage
// ============================================================================

// Static array of tags (MaxT > 0, TensorTag != void)
template <typename TensorTag, size_t MaxT>
struct TensorTagMixin {
    TensorTag tags_[MaxT]{};

    const TensorTag& tag(int32_t i) const { return tags_[i]; }
    TensorTag& tag(int32_t i) { return tags_[i]; }
};

// Dynamic vector of tags (MaxT == 0, TensorTag != void)
template <typename TensorTag>
struct TensorTagMixin<TensorTag, 0> {
    std::vector<TensorTag> tags_;

    const TensorTag& tag(int32_t i) const { return tags_[static_cast<size_t>(i)]; }
    TensorTag& tag(int32_t i) { return tags_[static_cast<size_t>(i)]; }
};

// Empty: TensorTag == void, static (zero overhead)
template <size_t MaxT>
struct TensorTagMixin<void, MaxT> {};

// Empty: TensorTag == void, dynamic (resolves ambiguity)
template <>
struct TensorTagMixin<void, 0> {};

// ============================================================================
// TaskArgs — primary template (static / fixed-size)
// ============================================================================

template <typename T, typename S, size_t MaxT, size_t MaxS, typename TensorTag = void>
struct TaskArgs : TensorTagMixin<TensorTag, MaxT> {
    T tensors_[MaxT];
    S scalars_[MaxS];
    int32_t tensor_count_{0};
    int32_t scalar_count_{0};

    void add_tensor(const T& t) {
        if (scalar_count_ > 0) throw std::logic_error("TaskArgs: cannot add tensor after scalar");
        if (static_cast<size_t>(tensor_count_) >= MaxT) throw std::out_of_range("TaskArgs: tensor capacity exceeded");
        tensors_[tensor_count_++] = t;
    }

    void add_scalar(S s) {
        if (static_cast<size_t>(scalar_count_) >= MaxS) throw std::out_of_range("TaskArgs: scalar capacity exceeded");
        scalars_[scalar_count_++] = s;
    }

    const T& tensor(int32_t i) const { return tensors_[i]; }
    T& tensor(int32_t i) { return tensors_[i]; }

    S scalar(int32_t i) const { return scalars_[i]; }
    S& scalar(int32_t i) { return scalars_[i]; }

    const T* tensor_data() const { return tensors_; }
    const S* scalar_data() const { return scalars_; }

    int32_t tensor_count() const { return tensor_count_; }
    int32_t scalar_count() const { return scalar_count_; }

    void clear() {
        tensor_count_ = 0;
        scalar_count_ = 0;
    }
};

// ============================================================================
// TaskArgs — partial specialization (dynamic / vector-backed, MaxT==0, MaxS==0)
// ============================================================================

template <typename T, typename S, typename TensorTag>
struct TaskArgs<T, S, 0, 0, TensorTag> : TensorTagMixin<TensorTag, 0> {
    std::vector<T> tensors_;
    std::vector<S> scalars_;

    void add_tensor(const T& t) {
        if (!scalars_.empty()) throw std::logic_error("TaskArgs: cannot add tensor after scalar");
        tensors_.push_back(t);
        if constexpr (!std::is_void_v<TensorTag>) {
            this->tags_.push_back(TensorTag{});
        }
    }

    void add_scalar(S s) { scalars_.push_back(s); }

    const T& tensor(int32_t i) const { return tensors_[static_cast<size_t>(i)]; }
    T& tensor(int32_t i) { return tensors_[static_cast<size_t>(i)]; }

    S scalar(int32_t i) const { return scalars_[static_cast<size_t>(i)]; }
    S& scalar(int32_t i) { return scalars_[static_cast<size_t>(i)]; }

    const T* tensor_data() const { return tensors_.data(); }
    const S* scalar_data() const { return scalars_.data(); }

    int32_t tensor_count() const { return static_cast<int32_t>(tensors_.size()); }
    int32_t scalar_count() const { return static_cast<int32_t>(scalars_.size()); }

    void clear() {
        tensors_.clear();
        scalars_.clear();
        if constexpr (!std::is_void_v<TensorTag>) {
            this->tags_.clear();
        }
    }
};

// ============================================================================
// Type aliases
// ============================================================================

// Transport/storage: host → device, no per-tensor tags
using ChipStorageTaskArgs = TaskArgs<ContinuousTensor, uint64_t, 16, 128>;

// Dynamic variant (no capacity limit)
using DynamicTaskArgs = TaskArgs<ContinuousTensor, uint64_t, 0, 0>;

// Tagged variant with TensorArgType (for submit-time INPUT/OUTPUT/INOUT)
using TaggedTaskArgs = TaskArgs<ContinuousTensor, uint64_t, 16, 128, TensorArgType>;
