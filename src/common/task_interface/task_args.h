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
 * TaskArgsTpl - Tensor + scalar argument storage (template)
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
 * Concrete user-facing types (typedefs at the bottom):
 *   - TaskArgs            — vector-backed + TensorArgType tags (the unified
 *                           builder used by Orchestrator.submit_*)
 *   - ChipStorageTaskArgs — fixed POD matching the runtime.so ABI byte-for-byte
 *
 * Wire / dispatch helpers:
 *   - TaskArgsView        — zero-copy view into a {tensors, scalars} pair (no tags)
 *   - write_blob/read_blob — length-prefixed serialization for PROCESS-mode
 *                            mailbox transport (tags stripped on the wire)
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "arg_direction.h"
#include "tensor_arg.h"

// ============================================================================
// TensorTagMixin — conditionally provides per-tensor tag storage
// ============================================================================

// Static array of tags (MaxT > 0, TensorTag != void)
template <typename TensorTag, size_t MaxT>
struct TensorTagMixin {
    TensorTag tags_[MaxT]{};

    const TensorTag &tag(int32_t i) const { return tags_[i]; }
    TensorTag &tag(int32_t i) { return tags_[i]; }
};

// Dynamic vector of tags (MaxT == 0, TensorTag != void)
template <typename TensorTag>
struct TensorTagMixin<TensorTag, 0> {
    std::vector<TensorTag> tags_;

    const TensorTag &tag(int32_t i) const { return tags_[static_cast<size_t>(i)]; }
    TensorTag &tag(int32_t i) { return tags_[static_cast<size_t>(i)]; }
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

    const T &tensor(int32_t i) const { return tensors_[static_cast<size_t>(i)]; }
    T &tensor(int32_t i) { return tensors_[static_cast<size_t>(i)]; }

    S scalar(int32_t i) const { return scalars_[static_cast<size_t>(i)]; }
    S &scalar(int32_t i) { return scalars_[static_cast<size_t>(i)]; }

    const T *tensor_data() const { return tensors_.data(); }
    const S *scalar_data() const { return scalars_.data(); }

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

// Unified user-facing builder: vector-backed with TensorArgType tags.
// Used by Orchestrator.submit_*; tags drive dependency inference at submit
// time and are stripped before the args cross the dispatch boundary.
using TaskArgs = TaskArgsTpl<ContinuousTensor, uint64_t, 0, 0, TensorArgType>;

// L2 runtime ABI: fixed POD matching runtime.so byte-for-byte.
// Assembled from a TaskArgsView on the child side just before pto2_run_runtime.
using ChipStorageTaskArgs = TaskArgsTpl<ContinuousTensor, uint64_t, CHIP_MAX_TENSOR_ARGS, 128>;

// ============================================================================
// TaskArgsView — zero-copy view used by IWorker::run and the wire format
// ============================================================================
//
// View-only: refers to externally owned tensor + scalar arrays. No tags
// (tags are consumed by Orchestrator at submit time and never travel further).

struct TaskArgsView {
    int32_t tensor_count;
    int32_t scalar_count;
    const ContinuousTensor *tensors;
    const uint64_t *scalars;
};

// Build a view directly over a TaskArgs's vectors (THREAD-mode dispatch).
inline TaskArgsView make_view(const TaskArgs &a) {
    return TaskArgsView{a.tensor_count(), a.scalar_count(), a.tensor_data(), a.scalar_data()};
}

// ============================================================================
// Wire format — length-prefixed blob for PROCESS-mode mailbox transport
// ============================================================================
//
// Byte layout (tags stripped):
//   offset 0:                 int32 tensor_count = T
//   offset 4:                 int32 scalar_count = S
//   offset 8:                 ContinuousTensor tensors[T]   (40 B each)
//   offset 8 + 40T:           uint64_t scalars[S]           (8 B each)
// total bytes used:           8 + 40T + 8S

inline constexpr size_t TASK_ARGS_BLOB_HEADER_SIZE = 8;

inline size_t task_args_blob_size(const TaskArgs &a) {
    return TASK_ARGS_BLOB_HEADER_SIZE + static_cast<size_t>(a.tensor_count()) * sizeof(ContinuousTensor) +
           static_cast<size_t>(a.scalar_count()) * sizeof(uint64_t);
}

// Serialize a TaskArgs into `dst`. Caller must ensure `dst` has room for
// task_args_blob_size(a) bytes. Tags are not written.
inline void write_blob(uint8_t *dst, const TaskArgs &a) {
    int32_t T = a.tensor_count();
    int32_t S = a.scalar_count();
    std::memcpy(dst + 0, &T, sizeof(T));
    std::memcpy(dst + 4, &S, sizeof(S));
    if (T > 0) {
        std::memcpy(
            dst + TASK_ARGS_BLOB_HEADER_SIZE, a.tensor_data(), static_cast<size_t>(T) * sizeof(ContinuousTensor)
        );
    }
    if (S > 0) {
        std::memcpy(
            dst + TASK_ARGS_BLOB_HEADER_SIZE + static_cast<size_t>(T) * sizeof(ContinuousTensor), a.scalar_data(),
            static_cast<size_t>(S) * sizeof(uint64_t)
        );
    }
}

// Zero-copy view into a blob written by write_blob. The returned view is only
// valid as long as `src` stays alive in mapped/shm memory.
inline TaskArgsView read_blob(const uint8_t *src) {
    int32_t T;
    int32_t S;
    std::memcpy(&T, src + 0, sizeof(T));
    std::memcpy(&S, src + 4, sizeof(S));
    return TaskArgsView{
        T,
        S,
        reinterpret_cast<const ContinuousTensor *>(src + TASK_ARGS_BLOB_HEADER_SIZE),
        reinterpret_cast<const uint64_t *>(
            src + TASK_ARGS_BLOB_HEADER_SIZE + static_cast<size_t>(T) * sizeof(ContinuousTensor)
        ),
    };
}

// ============================================================================
// L2 ABI helper: build ChipStorageTaskArgs POD from a view (memcpy'd).
// Runs on the child side immediately before crossing into runtime.so.
// ============================================================================

inline ChipStorageTaskArgs view_to_chip_storage(TaskArgsView view) {
    ChipStorageTaskArgs out;
    if (static_cast<size_t>(view.tensor_count) > CHIP_MAX_TENSOR_ARGS) {
        throw std::out_of_range("view_to_chip_storage: tensor_count exceeds CHIP_MAX_TENSOR_ARGS");
    }
    if (view.scalar_count > 128) {
        throw std::out_of_range("view_to_chip_storage: scalar_count exceeds 128");
    }
    out.tensor_count_ = view.tensor_count;
    out.scalar_count_ = view.scalar_count;
    if (view.tensor_count > 0) {
        std::memcpy(out.tensors_, view.tensors, static_cast<size_t>(view.tensor_count) * sizeof(ContinuousTensor));
    }
    if (view.scalar_count > 0) {
        std::memcpy(out.scalars_, view.scalars, static_cast<size_t>(view.scalar_count) * sizeof(uint64_t));
    }
    return out;
}
