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
 * Concrete user-facing types (typedefs at the bottom):
 *   - TaskArgs            — vector-backed + TensorArgType tags (the unified
 *                           builder used by Orchestrator.submit_*)
 *   - ChipStorageTaskArgs — fixed POD matching the runtime.so ABI byte-for-byte
 *
 * Wire / dispatch helpers:
 *   - TaskArgsView        — zero-copy view over a wire blob (no tags)
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
#include "buffer.h"  // Tensor wire type + TENSOR_BLOB_MAGIC for the blob envelope
#include "tensor.h"  // ChipTensor (device POD) + TensorArgType, the tag TaskArgs carries

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
    // Host-side freshness provenance for materializing this TaskArgs directly
    // into ChipStorageTaskArgs. It is intentionally stripped with tags when a
    // task crosses a mailbox boundary; zero means the receiving producer has
    // supplied no freshness guarantee.
    std::vector<uint64_t> host_content_generations_;

    const TensorTag &tag(int32_t i) const { return tags_[static_cast<size_t>(i)]; }
    TensorTag &tag(int32_t i) { return tags_[static_cast<size_t>(i)]; }
    const TensorTag *tag_data() const { return tags_.data(); }
    uint64_t host_content_generation(int32_t i) const { return host_content_generations_[static_cast<size_t>(i)]; }
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
            this->host_content_generations_.push_back(0);
        }
    }

    // Tagged overload: only enabled when TensorTag != void.
    template <typename Tag = TensorTag, typename = std::enable_if_t<!std::is_void_v<Tag>>>
    void add_tensor(const T &t, Tag tag) {
        if (!scalars_.empty()) throw std::logic_error("TaskArgs: cannot add tensor after scalar");
        tensors_.push_back(t);
        this->tags_.push_back(tag);
        this->host_content_generations_.push_back(0);
    }

    // Tagged host-local overload with producer freshness provenance. The
    // generation is materialized into ChipTensor and is not part of the
    // Tensor mailbox wire format.
    template <typename Tag = TensorTag, typename = std::enable_if_t<!std::is_void_v<Tag>>>
    void add_tensor(const T &t, Tag tag, uint64_t host_content_generation) {
        if (!scalars_.empty()) throw std::logic_error("TaskArgs: cannot add tensor after scalar");
        tensors_.push_back(t);
        this->tags_.push_back(tag);
        this->host_content_generations_.push_back(host_content_generation);
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
            this->host_content_generations_.clear();
        }
    }
};

// ============================================================================
// Type aliases
// ============================================================================

// Unified user-facing builder: vector-backed with TensorArgType tags.
// Used by Orchestrator.submit_*; tags drive dependency inference at submit
// time and are stripped before the args cross the dispatch boundary. The element
// is Tensor (self-describing view; L3+ holds no C++ ChipTensor) — the L3→L2 wire
// carries Tensors, materialized to ChipStorageTaskArgs (ChipTensor) on the L2 child.
using TaskArgs = TaskArgsTpl<Tensor, uint64_t, 0, 0, TensorArgType>;

// L2 runtime ABI: fixed POD matching runtime.so byte-for-byte, and the sole ChipTensor-typed args
// container — the materialized form a chip child decodes the L3->L2 Tensor blob into, just before
// pto2_run_runtime.
using ChipStorageTaskArgs = TaskArgsTpl<ChipTensor, uint64_t, CHIP_MAX_TENSOR_ARGS, CHIP_MAX_SCALAR_ARGS>;

// ============================================================================
// TaskArgsView — zero-copy view over a wire blob
// ============================================================================
//
// View-only: refers to externally owned tensor + scalar arrays. No tags
// (tags are consumed by Orchestrator at submit time and never travel further).

struct TaskArgsView {
    int32_t tensor_count;
    int32_t scalar_count;
    // Raw bytes of the tensor array, NOT a `const Tensor *`. The blob's tensor region starts at the
    // 8-byte header boundary, so a `Tensor *` formed onto it would carry an alignment the type does
    // not promise. Copy a tensor out with tensors(i).
    const uint8_t *tensor_bytes;
    const uint64_t *scalars;  // 8-byte aligned by blob construction; safe to address as uint64_t*

    // Copy the i-th tensor into a properly-aligned local and gate it. Bounds-checked: a negative
    // index would otherwise wrap to a huge offset once cast to size_t. This is the ONLY validation
    // a blob element ever gets — nothing downstream re-checks magic, tag, body_len, the view's
    // containment in its backing, or the FORK_COW read-only rule.
    Tensor tensors(int32_t i) const {
        if (i < 0 || i >= tensor_count) {
            throw std::out_of_range("TaskArgsView::tensors: index out of range");
        }
        Tensor t;
        std::memcpy(&t, tensor_bytes + static_cast<size_t>(i) * sizeof(Tensor), sizeof(Tensor));
        validate_tensor(t);
        return t;
    }
};

// ============================================================================
// Wire format — length-prefixed blob for PROCESS-mode mailbox transport
// ============================================================================
//
// Byte layout (tags stripped):
//   offset 0:                 int32 tensor_count = T
//   offset 4:                 int32 scalar_count = S
//   offset 8:                 Tensor tensors[T]             (144 B each)
//   offset 8 + 144T:          uint64_t scalars[S]           (8 B each)
// total bytes used:           8 + 144T + 8S
//
// The element is the self-describing wire `Tensor`: it carries its backing's descriptor, so a
// consumer resolves it with no prior handshake. A chip child materializes each one to a
// `ChipTensor` (address-bearing) and assembles a `ChipStorageTaskArgs` for the runtime.so ABI.

inline constexpr size_t TASK_ARGS_BLOB_HEADER_SIZE = 8;

inline size_t task_args_blob_size(const TaskArgs &a) {
    return TASK_ARGS_BLOB_HEADER_SIZE + static_cast<size_t>(a.tensor_count()) * sizeof(Tensor) +
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
        std::memcpy(dst + TASK_ARGS_BLOB_HEADER_SIZE, a.tensor_data(), static_cast<size_t>(T) * sizeof(Tensor));
    }
    if (S > 0) {
        std::memcpy(
            dst + TASK_ARGS_BLOB_HEADER_SIZE + static_cast<size_t>(T) * sizeof(Tensor), a.scalar_data(),
            static_cast<size_t>(S) * sizeof(uint64_t)
        );
    }
}

// Zero-copy view into a blob written by write_blob. The returned view is only
// valid as long as `src` stays alive in mapped/shm memory.
//
// `capacity` is the maximum number of bytes the reader is allowed to consume
// from `src` (e.g. MAILBOX_ARGS_CAPACITY when reading from the IPC mailbox).
// Throws std::runtime_error if the header reports counts that would walk past
// `capacity` — defends against shared-memory corruption or a writer-side bug
// that slipped past the writer's own bounds check. This bounds the envelope
// only; each element is gated by TaskArgsView::tensors.
inline TaskArgsView read_blob(const uint8_t *src, size_t capacity) {
    if (capacity < TASK_ARGS_BLOB_HEADER_SIZE) {
        throw std::runtime_error(
            "read_blob: capacity " + std::to_string(capacity) + " < header size " +
            std::to_string(TASK_ARGS_BLOB_HEADER_SIZE)
        );
    }
    int32_t T;
    int32_t S;
    std::memcpy(&T, src + 0, sizeof(T));
    std::memcpy(&S, src + 4, sizeof(S));
    if (T < 0 || S < 0) {
        throw std::runtime_error(
            "read_blob: negative counts — tensors=" + std::to_string(T) + ", scalars=" + std::to_string(S)
        );
    }
    const size_t needed = TASK_ARGS_BLOB_HEADER_SIZE + static_cast<size_t>(T) * sizeof(Tensor) +
                          static_cast<size_t>(S) * sizeof(uint64_t);
    if (needed > capacity) {
        throw std::runtime_error(
            "read_blob: header reports " + std::to_string(needed) + " bytes (T=" + std::to_string(T) +
            ", S=" + std::to_string(S) + ") but capacity is " + std::to_string(capacity) +
            " — likely shm corruption or a writer-side bug"
        );
    }
    return TaskArgsView{
        T,
        S,
        src + TASK_ARGS_BLOB_HEADER_SIZE,
        reinterpret_cast<const uint64_t *>(src + TASK_ARGS_BLOB_HEADER_SIZE + static_cast<size_t>(T) * sizeof(Tensor)),
    };
}

// ============================================================================
// Submit-time argument validation
// ============================================================================

// access ⊆ granted: an arg's TensorArgType may only request what the backing grants.
//   INPUT -> READ, OUTPUT_EXISTING -> WRITE, INOUT -> READWRITE; READWRITE grants everything.
//   NO_DEP / OUTPUT are unconstrained.
// Catches e.g. a READ-only copy-on-write backing tagged OUTPUT_EXISTING, whose writes in a forked
// child would silently never reach the parent.
inline bool access_permits(uint8_t granted, TensorArgType tag) {
    auto granted_has = [&](AccessMode need) {
        return granted == static_cast<uint8_t>(AccessMode::READWRITE) || granted == static_cast<uint8_t>(need);
    };
    switch (tag) {
    case TensorArgType::INPUT:
        return granted_has(AccessMode::READ);
    case TensorArgType::OUTPUT_EXISTING:
        return granted_has(AccessMode::WRITE);
    case TensorArgType::INOUT:
        return granted == static_cast<uint8_t>(AccessMode::READWRITE);
    default:
        return true;
    }
}

// Does this tag declare a write? NO_DEP is excluded deliberately: it opts out of dependency
// tracking altogether, so its ordering is the caller's to arrange.
inline bool tag_writes(TensorArgType tag) {
    return tag == TensorArgType::OUTPUT || tag == TensorArgType::OUTPUT_EXISTING || tag == TensorArgType::INOUT;
}

/**
 * Validate one submit's whole argument set, at the point where the values are final.
 *
 * `access ⊆ granted` is re-checked here rather than trusted from add time because a tag is mutable
 * after its element is added — the pair that governs the dispatch is the one present now.
 *
 * Overlapping writes WITHIN one TaskArgs are rejected because no later layer can catch them: the two
 * args belong to one task node, so there is no order between them to express, and a device-staged
 * copy of a host backing does not even alias on the device for the L2 overlap map to notice.
 * Disjoint slices of one backing stay legal.
 *
 * Members of a group are NOT compared against each other. A group is one DAG node whose members
 * deliberately share their tags — naming one buffer as every member's OUTPUT is how a group
 * publishes a single completion token for a downstream task to depend on. Whether such a shared
 * write carries data or only ordering is not visible here, so the caller owns it.
 */
inline void validate_submit_args(const std::vector<TaskArgs> &args_list) {
    for (const TaskArgs &args : args_list) {
        for (int32_t i = 0; i < args.tensor_count(); ++i) {
            if (!access_permits(args.tensor(i).buffer.access, args.tag(i))) {
                throw std::invalid_argument(
                    "submit: an argument's TensorArgType requests access the backing does not grant"
                );
            }
        }
    }
    for (const TaskArgs &args : args_list) {
        for (int32_t i = 0; i < args.tensor_count(); ++i) {
            if (!tag_writes(args.tag(i))) continue;
            for (int32_t j = i + 1; j < args.tensor_count(); ++j) {
                if (!tensors_overlap(args.tensor(i), args.tensor(j))) continue;
                throw std::invalid_argument(
                    "submit: two arguments of one task write overlapping bytes of the same buffer; "
                    "give them disjoint ranges, or order them as separate tasks"
                );
            }
        }
    }
}
