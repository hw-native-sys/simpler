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
 * TaskArgsTpl - tensor + scalar argument storage (template)
 *
 * Template: TaskArgsTpl<T, S, MaxT, MaxS, TensorTag=void>
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
 * The element type is the caller's: each layer instantiates the template over
 * whatever tensor form it holds. Only one instantiation lives here, the one whose
 * element crosses the host->device wire:
 *   - ChipStorageTaskArgs — fixed POD matching the runtime.so ABI byte-for-byte
 *
 * The L3+ form — `TaskArgs`, whose element is the self-describing `Tensor` from
 * buffer.h — is in task_args_wire.h, together with the blob codec and the
 * submit-time checks. Nothing here reaches buffer.h, which is what keeps that
 * header's global `Tensor` out of every kernel and orchestration translation unit;
 * they see this one.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "arg_direction.h"
#include "tensor.h"  // ChipTensor (device POD) + TensorArgType, the tag TaskArgs carries

// Opaque at the Python surface. The run id prevents a slot from being reused
// as a dependency after the parent-side Ring resets its monotonic slot ids.
struct TaskHandle {
    uint64_t run_id{0};
    int32_t task_slot{-1};
};

struct ExplicitTaskDependency {
    TaskHandle task;
    bool retain{false};
};

// ============================================================================
// TensorTagMixin — conditionally provides per-tensor tag storage
// ============================================================================

// Static array of tags (MaxT > 0, TensorTag != void)
template <typename TensorTag, size_t MaxT>
struct TensorTagMixin {
    TensorTag tags_[MaxT]{};

    const TensorTag &tag(int32_t i) const { return tags_[i]; }
    TensorTag &tag(int32_t i) { return tags_[i]; }
    const TensorTag *tag_data() const { return tags_; }
};

// Dynamic vector of tags (MaxT == 0, TensorTag != void)
template <typename TensorTag>
struct TensorTagMixin<TensorTag, 0> {
    std::vector<TensorTag> tags_;

    const TensorTag &tag(int32_t i) const { return tags_[static_cast<size_t>(i)]; }
    TensorTag &tag(int32_t i) { return tags_[static_cast<size_t>(i)]; }
    const TensorTag *tag_data() const { return tags_.data(); }
};

// Empty: TensorTag == void, static (zero overhead)
template <size_t MaxT>
struct TensorTagMixin<void, MaxT> {};

// Empty: TensorTag == void, dynamic (resolves ambiguity)
template <>
struct TensorTagMixin<void, 0> {};

// ============================================================================
// TaskArgsTpl — primary template (static / fixed-size)
// ============================================================================

template <typename T, typename S, size_t MaxT, size_t MaxS, typename TensorTag = void>
struct TaskArgsTpl : TensorTagMixin<TensorTag, MaxT> {
    T tensors_[MaxT];
    S scalars_[MaxS];
    int32_t tensor_count_{0};
    int32_t scalar_count_{0};

    void add_tensor(const T &t) {
        if (scalar_count_ > 0) throw std::logic_error("TaskArgs: cannot add tensor after scalar");
        if (static_cast<size_t>(tensor_count_) >= MaxT) throw std::out_of_range("TaskArgs: tensor capacity exceeded");
        tensors_[tensor_count_++] = t;
    }

    void add_scalar(S s) {
        if (static_cast<size_t>(scalar_count_) >= MaxS) throw std::out_of_range("TaskArgs: scalar capacity exceeded");
        scalars_[scalar_count_++] = s;
    }

    const T &tensor(int32_t i) const { return tensors_[i]; }
    T &tensor(int32_t i) { return tensors_[i]; }

    S scalar(int32_t i) const { return scalars_[i]; }
    S &scalar(int32_t i) { return scalars_[i]; }

    const S *scalars() const { return scalars_; }

    const T *tensor_data() const { return tensors_; }
    const S *scalar_data() const { return scalars_; }

    int32_t tensor_count() const { return tensor_count_; }
    int32_t scalar_count() const { return scalar_count_; }

    void clear() {
        tensor_count_ = 0;
        scalar_count_ = 0;
    }
};

// ============================================================================
// TaskArgsTpl — partial specialization (dynamic / vector-backed, MaxT==0, MaxS==0)
// ============================================================================

template <typename T, typename S, typename TensorTag>
struct TaskArgsTpl<T, S, 0, 0, TensorTag> : TensorTagMixin<TensorTag, 0> {
    std::vector<T> tensors_;
    std::vector<S> scalars_;
    std::vector<ExplicitTaskDependency> explicit_deps_;

    void add_tensor(const T &t) {
        if (!scalars_.empty()) throw std::logic_error("TaskArgs: cannot add tensor after scalar");
        tensors_.push_back(t);
        if constexpr (!std::is_void_v<TensorTag>) {
            this->tags_.push_back(TensorTag{});
        }
    }

    // Tagged overload: only enabled when TensorTag != void.
    template <typename Tag = TensorTag, typename = std::enable_if_t<!std::is_void_v<Tag>>>
    void add_tensor(const T &t, Tag tag) {
        if (!scalars_.empty()) throw std::logic_error("TaskArgs: cannot add tensor after scalar");
        tensors_.push_back(t);
        this->tags_.push_back(tag);
    }

    void add_scalar(S s) { scalars_.push_back(s); }

    void add_dep(const TaskHandle &task) { explicit_deps_.push_back(ExplicitTaskDependency{task, true}); }
    void add_dep_wait(const TaskHandle &task) { explicit_deps_.push_back(ExplicitTaskDependency{task, false}); }

    const T &tensor(int32_t i) const { return tensors_[static_cast<size_t>(i)]; }
    T &tensor(int32_t i) { return tensors_[static_cast<size_t>(i)]; }

    S scalar(int32_t i) const { return scalars_[static_cast<size_t>(i)]; }
    S &scalar(int32_t i) { return scalars_[static_cast<size_t>(i)]; }

    const T *tensor_data() const { return tensors_.data(); }
    const S *scalar_data() const { return scalars_.data(); }

    int32_t tensor_count() const { return static_cast<int32_t>(tensors_.size()); }
    int32_t scalar_count() const { return static_cast<int32_t>(scalars_.size()); }
    int32_t explicit_dep_count() const { return static_cast<int32_t>(explicit_deps_.size()); }
    const TaskHandle &explicit_dep(int32_t i) const { return explicit_deps_[static_cast<size_t>(i)].task; }
    bool explicit_dep_retain(int32_t i) const { return explicit_deps_[static_cast<size_t>(i)].retain; }

    void clear() {
        tensors_.clear();
        scalars_.clear();
        explicit_deps_.clear();
        if constexpr (!std::is_void_v<TensorTag>) {
            this->tags_.clear();
        }
    }
};

// ============================================================================
// Type aliases
// ============================================================================

// L2 runtime ABI: fixed POD matching runtime.so byte-for-byte, and the sole ChipTensor-typed args
// container — the materialized form a chip child decodes the L3->L2 Tensor blob into, just before
// simpler_run.
using ChipStorageTaskArgs = TaskArgsTpl<ChipTensor, uint64_t, CHIP_MAX_TENSOR_ARGS, CHIP_MAX_SCALAR_ARGS>;
