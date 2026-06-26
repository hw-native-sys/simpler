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
 * Callable - Binary artifact with typed argument signature (host→device serialization format)
 *
 * Two concrete types, both using fixed-size arrays + flexible array member (FAM) storage:
 *
 *   CoreCallable = Callable<void, MaxSig, 0>              — leaf kernel binary
 *   ChipCallable = Callable<CoreCallable, MaxSig, MaxChildren> — orchestration + child kernels
 *
 * CoreCallable includes resolved_addr_ — a platform-resolved dispatch address
 * (binary code addr on onboard, func_ptr on sim) used by AICPU dispatch.
 * Binary data is placed at CALLABLE_ALIGN boundary within storage_ for
 * device-optimal alignment; binary_data() accounts for this automatically.
 *
 * Both types use placement-new via make_callable() factory functions.
 * The returned vector<uint8_t> owns the memory; reinterpret_cast to access.
 *
 * Higher-level callables (L3 HostCallable) are Python-only objects that
 * reference ChipCallable(s) by pointer. At L3+ submit, callables are passed as
 * an opaque uint64 Callable handle (see docs/task-flow.md); they never cross
 * the host-device boundary.
 *
 * Type aliases:
 *   CoreCallable = Callable<void, CORE_MAX_TENSOR_ARGS, 0>       — leaf kernel binary
 *   ChipCallable = Callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024> — orchestration + child kernels
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "arg_direction.h"

// ============================================================================
// Forward declaration
// ============================================================================

template <typename Child, int MaxSig, int MaxChildren>
struct Callable;

// ============================================================================
// Static leaf: Callable<void, MaxSig, 0> — FAM, no children
// ============================================================================

template <int MaxSig>
struct Callable<void, MaxSig, 0> {
    ArgDirection signature_[MaxSig];
    // Parallel to signature_ (both hold sig_count_ entries): signature_[i]
    // describes the payload tensor at absolute index arg_index_[i]. The tensor
    // dump uses it to map each declared tensor to an explicit payload slot.
    uint32_t arg_index_[MaxSig];
    int32_t sig_count_;
    uint32_t binary_size_;
    uint64_t resolved_addr_;
    char storage_[];

    ArgDirection sig(int32_t i) const {
        if (i < 0 || i >= sig_count_) throw std::out_of_range("Callable: sig index out of range");
        return signature_[i];
    }
    uint32_t arg_index(int32_t i) const {
        if (i < 0 || i >= sig_count_) throw std::out_of_range("Callable: arg_index out of range");
        return arg_index_[i];
    }
    int32_t sig_count() const { return sig_count_; }
    uint32_t binary_size() const { return binary_size_; }
    uint64_t resolved_addr() const { return resolved_addr_; }
    void set_resolved_addr(uint64_t addr) { resolved_addr_ = addr; }

    // Binary data is placed at the next CALLABLE_ALIGN boundary after the fixed fields.
    // storage_ sits between the fixed fields and the aligned binary; binary_data()
    // skips the padding automatically.
    const void *binary_data() const { return reinterpret_cast<const char *>(this) + binary_data_offset(); }

    static constexpr size_t binary_data_offset() {
        // sizeof(), not a manual member-size sum, so compiler padding can't overlap the binary.
        return (sizeof(Callable<void, MaxSig, 0>) + CALLABLE_ALIGN - 1) & ~(static_cast<size_t>(CALLABLE_ALIGN) - 1);
    }

private:
    Callable() = default;

    template <int MS>
    friend std::vector<uint8_t> make_callable(
        const ArgDirection *sig, const uint32_t *arg_index, int32_t sig_count, const void *binary, uint32_t binary_size
    );
};

// ============================================================================
// Static parent: Callable<Child, MaxSig, MaxChildren> — FAM + children
// ============================================================================

static constexpr int CALLABLE_FUNC_NAME_MAX = 64;

template <typename Child, int MaxSig, int MaxChildren>
struct Callable {
    ArgDirection signature_[MaxSig];
    int32_t sig_count_;
    uint32_t binary_size_;
    char func_name_[CALLABLE_FUNC_NAME_MAX];
    uint32_t func_name_len_;
    int32_t child_func_ids_[MaxChildren];
    uint32_t child_offsets_[MaxChildren];
    int32_t child_count_;
    char config_name_[CALLABLE_FUNC_NAME_MAX];
    uint32_t config_name_len_;
    // Children live in storage_ at CALLABLE_ALIGN-aligned offsets, but the
    // all-uint32 header above can leave offsetof(storage_) at 4-mod-8, which
    // would place an 8-byte-aligned Child (CoreCallable has a uint64) on a
    // misaligned address — UB on reference binding (caught by UBSan). It can
    // also leave it at 8-mod-16, which under-aligns a5 SIMT kernel binaries
    // (mscatter et al. require a 16-byte-aligned code address). Align storage_
    // to CALLABLE_CHILD_ALIGN (>= alignof(Child)) so every child — and the
    // kernel binary inside it — lands at the strictest required alignment.
    alignas(CALLABLE_CHILD_ALIGN) char storage_[];

    ArgDirection sig(int32_t i) const {
        if (i < 0 || i >= sig_count_) throw std::out_of_range("Callable: sig index out of range");
        return signature_[i];
    }
    int32_t sig_count() const { return sig_count_; }
    const void *binary_data() const { return storage_; }
    uint32_t binary_size() const { return binary_size_; }
    const char *func_name() const { return func_name_; }
    uint32_t func_name_len() const { return func_name_len_; }
    const char *config_name() const { return config_name_; }
    uint32_t config_name_len() const { return config_name_len_; }

    const Child &child(int32_t i) const {
        if (i < 0 || i >= child_count_) throw std::out_of_range("Callable: child index out of range");
        return *reinterpret_cast<const Child *>(storage_ + child_offsets_[i]);
    }
    int32_t child_func_id(int32_t i) const {
        if (i < 0 || i >= child_count_) throw std::out_of_range("Callable: child_func_id index out of range");
        return child_func_ids_[i];
    }
    int32_t child_count() const { return child_count_; }
    uint32_t child_offset(int32_t i) const {
        if (i < 0 || i >= child_count_) throw std::out_of_range("Callable: child_offset index out of range");
        return child_offsets_[i];
    }

private:
    Callable() = default;

    template <typename C, int MS, int MC>
    friend std::vector<uint8_t> make_callable(
        const ArgDirection *sig, int32_t sig_count, const char *func_name, const void *binary, uint32_t binary_size,
        const int32_t *child_func_ids, const std::vector<uint8_t> *child_buffers, int32_t child_count,
        const char *config_name
    );
};

// ============================================================================
// Type aliases
// ============================================================================

using CoreCallable = Callable<void, CORE_MAX_TENSOR_ARGS, 0>;
using ChipCallable = Callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>;

// storage_ holds CoreCallable children at CALLABLE_ALIGN-aligned offsets; the
// child kernel binary's device address is offsetof(storage_) + child_offset +
// CoreCallable::binary_data_offset(). The latter two are CALLABLE_ALIGN (64)
// multiples, so storage_ alignment alone decides the binary's alignment.
// CALLABLE_CHILD_ALIGN (16) must cover both: child() reference binding without
// UB (needs alignof(CoreCallable) = 8) and a5 SIMT kernel-binary fetch (needs
// 16). Assert both invariants so a future header tweak can't silently regress.
static_assert(
    CALLABLE_CHILD_ALIGN >= alignof(CoreCallable), "CALLABLE_CHILD_ALIGN must not under-align CoreCallable children"
);
static_assert(
    offsetof(ChipCallable, storage_) % CALLABLE_CHILD_ALIGN == 0,
    "ChipCallable.storage_ must be CALLABLE_CHILD_ALIGN-aligned for SIMT kernel binaries"
);

// ============================================================================
// Factory: make_callable for static leaf
// ============================================================================

template <int MaxSig>
std::vector<uint8_t> make_callable(
    const ArgDirection *sig, const uint32_t *arg_index, int32_t sig_count, const void *binary, uint32_t binary_size
) {
    if (sig_count > MaxSig) throw std::invalid_argument("make_callable: sig_count exceeds MaxSig");
    if (sig_count > 0 && (sig == nullptr || arg_index == nullptr))
        throw std::invalid_argument("make_callable: sig and arg_index are required when sig_count > 0");

    using T = Callable<void, MaxSig, 0>;
    size_t aligned_header = T::binary_data_offset();
    size_t total_size = aligned_header + binary_size;
    std::vector<uint8_t> buf(total_size, 0);

    T *obj = reinterpret_cast<T *>(buf.data());
    for (int32_t i = 0; i < sig_count; ++i) {
        obj->signature_[i] = sig[i];
        // arg_index_ is parallel to signature_: arg_index_[i] is the absolute
        // payload tensor slot that signature_[i] describes.
        obj->arg_index_[i] = arg_index[i];
    }
    obj->sig_count_ = sig_count;
    obj->binary_size_ = binary_size;
    obj->resolved_addr_ = 0;
    if (binary_size > 0) std::memcpy(buf.data() + aligned_header, binary, binary_size);

    return buf;
}

// ============================================================================
// Factory: make_callable for static parent
// ============================================================================

template <typename Child, int MaxSig, int MaxChildren>
std::vector<uint8_t> make_callable(
    const ArgDirection *sig, int32_t sig_count, const char *func_name, const void *binary, uint32_t binary_size,
    const int32_t *child_func_ids, const std::vector<uint8_t> *child_buffers, int32_t child_count,
    // No default arg here: the friend declaration above has none, so a default
    // on this definition is a "redeclaration may not have default arguments"
    // error once ChipCallable is instantiated (the static_assert below does
    // that). Both call sites pass config_name explicitly (the binding defaults
    // it to "" at the nb::arg level), so the default was dead anyway.
    const char *config_name
) {
    if (sig_count > MaxSig) throw std::invalid_argument("make_callable: sig_count exceeds MaxSig");
    if (child_count > MaxChildren) throw std::invalid_argument("make_callable: child_count exceeds MaxChildren");

    using T = Callable<Child, MaxSig, MaxChildren>;
    size_t header_size = offsetof(T, storage_);

    uint32_t offset = binary_size;
    uint32_t child_offsets[MaxChildren];
    for (int32_t i = 0; i < child_count; ++i) {
        offset = callable_align_up(offset);
        child_offsets[i] = offset;
        offset += static_cast<uint32_t>(child_buffers[i].size());
    }
    size_t total_size = header_size + offset;
    std::vector<uint8_t> buf(total_size, 0);

    T *obj = reinterpret_cast<T *>(buf.data());
    for (int32_t i = 0; i < sig_count; ++i)
        obj->signature_[i] = sig[i];
    obj->sig_count_ = sig_count;
    obj->binary_size_ = binary_size;

    // Store func_name (null-terminated, truncated to CALLABLE_FUNC_NAME_MAX-1)
    std::memset(obj->func_name_, 0, CALLABLE_FUNC_NAME_MAX);
    if (func_name != nullptr) {
        size_t name_len = std::strlen(func_name);
        if (name_len >= CALLABLE_FUNC_NAME_MAX) name_len = CALLABLE_FUNC_NAME_MAX - 1;
        std::memcpy(obj->func_name_, func_name, name_len);
        obj->func_name_len_ = static_cast<uint32_t>(name_len);
    } else {
        obj->func_name_len_ = 0;
    }

    // Store config_name (null-terminated, truncated to CALLABLE_FUNC_NAME_MAX-1)
    std::memset(obj->config_name_, 0, CALLABLE_FUNC_NAME_MAX);
    if (config_name != nullptr) {
        size_t name_len = std::strlen(config_name);
        if (name_len >= CALLABLE_FUNC_NAME_MAX) name_len = CALLABLE_FUNC_NAME_MAX - 1;
        std::memcpy(obj->config_name_, config_name, name_len);
        obj->config_name_len_ = static_cast<uint32_t>(name_len);
    } else {
        obj->config_name_len_ = 0;
    }

    if (binary_size > 0) std::memcpy(obj->storage_, binary, binary_size);

    for (int32_t i = 0; i < child_count; ++i) {
        obj->child_func_ids_[i] = child_func_ids[i];
        obj->child_offsets_[i] = child_offsets[i];
        if (!child_buffers[i].empty()) {
            std::memcpy(obj->storage_ + child_offsets[i], child_buffers[i].data(), child_buffers[i].size());
        }
    }
    obj->child_count_ = child_count;

    return buf;
}
