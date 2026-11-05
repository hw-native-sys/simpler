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
 *   ChipCallable = Callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, CHIP_MAX_CHILDREN> — orchestration + child kernels
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "arg_direction.h"
#include "async_capability.h"
#include "callable_limits.h"

constexpr uint32_t CALLABLE_IMAGE_MAGIC = 0x43414c4c;
constexpr uint16_t CALLABLE_IMAGE_VERSION = 1;
enum class CallableImageKind : uint16_t { CORE = 1, CHIP = 2 };

// This is an explicit new wire format. Legacy unversioned images must be
// rebuilt; alignment padding is never interpreted as a capability descriptor.
struct CallableImageHeader {
    uint32_t magic;
    uint16_t version;
    CallableImageKind kind;
    uint32_t header_size;
    uint32_t image_size;
};
static_assert(sizeof(CallableImageHeader) == 16, "Callable image prefix size changed");

inline bool callable_image_header_valid(const void *image, size_t size, CallableImageKind kind, size_t header_size) {
    if (image == nullptr || size < sizeof(CallableImageHeader)) return false;
    CallableImageHeader header{};
    std::memcpy(&header, image, sizeof(header));
    return header.magic == CALLABLE_IMAGE_MAGIC && header.version == CALLABLE_IMAGE_VERSION && header.kind == kind &&
           header.header_size == header_size && header.image_size == size;
}

// signature contains sig_count validated entries.
inline int32_t count_scalar_args(const ArgDirection *signature, int32_t sig_count) {
    int32_t scalar_count = 0;
    for (int32_t i = 0; i < sig_count; ++i) {
        if (signature[i] == ArgDirection::SCALAR) ++scalar_count;
    }
    return scalar_count;
}

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
    static constexpr int MAX_SIGNATURE_COUNT = MaxSig;
    CallableImageHeader image_;
    CoreAsyncCapability async_capability_;
    ArgDirection signature_[MaxSig];
    int32_t sig_count_;
    uint32_t binary_size_;
    uint64_t resolved_addr_;
    char storage_[];

    ArgDirection sig(int32_t i) const {
        if (i < 0 || i >= sig_count_) throw std::out_of_range("Callable: sig index out of range");
        return signature_[i];
    }
    int32_t sig_count() const { return sig_count_; }
    uint32_t binary_size() const { return binary_size_; }
    uint32_t image_size() const { return image_.image_size; }
    const CoreAsyncCapability &async_capability() const { return async_capability_; }
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
        const ArgDirection *sig, int32_t sig_count, const void *binary, uint32_t binary_size,
        const CoreAsyncCapability &capability
    );
};

// ============================================================================
// Static parent: Callable<Child, MaxSig, MaxChildren> — FAM + children
// ============================================================================

static constexpr int CALLABLE_FUNC_NAME_MAX = 64;

template <typename Child, int MaxSig, int MaxChildren>
struct Callable {
    static constexpr int MAX_CHILDREN = MaxChildren;

    CallableImageHeader image_;
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
    uint32_t image_size() const { return image_.image_size; }
    const char *func_name() const { return func_name_; }
    uint32_t func_name_len() const { return func_name_len_; }
    const char *config_name() const { return config_name_; }
    uint32_t config_name_len() const { return config_name_len_; }
    int32_t scalar_count() const {
        if (sig_count_ < 0 || sig_count_ > MaxSig) {
            throw std::invalid_argument(
                "Callable: signature count " + std::to_string(sig_count_) + " is outside the supported range [0, " +
                std::to_string(MaxSig) + "]"
            );
        }
        return count_scalar_args(signature_, sig_count_);
    }

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
using ChipCallable = Callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, CHIP_MAX_CHILDREN>;

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

// Callable bytes are shipped through L3/L4 IPC and the on-disk kernel cache,
// so the header layout is wire ABI. These explicit size and offset assertions
// pin the layout independently of the capacity constants.
static_assert(
    std::is_trivially_copyable_v<ChipCallable> && std::is_standard_layout_v<ChipCallable>,
    "ChipCallable wire ABI: must stay memcpy-able POD"
);
static_assert(
    std::is_trivially_copyable_v<CoreCallable> && std::is_standard_layout_v<CoreCallable>,
    "CoreCallable wire ABI: must stay memcpy-able POD"
);
static_assert(offsetof(ChipCallable, signature_) == 16, "ChipCallable wire ABI: signature offset changed");
static_assert(offsetof(ChipCallable, sig_count_) == 1040, "ChipCallable wire ABI: sig_count offset changed");
static_assert(offsetof(ChipCallable, binary_size_) == 1044, "ChipCallable wire ABI: binary_size offset changed");
static_assert(offsetof(ChipCallable, func_name_) == 1048, "ChipCallable wire ABI: func_name offset changed");
static_assert(offsetof(ChipCallable, func_name_len_) == 1112, "ChipCallable wire ABI: func_name_len offset changed");
static_assert(offsetof(ChipCallable, child_func_ids_) == 1116, "ChipCallable wire ABI: child_func_ids offset changed");
static_assert(offsetof(ChipCallable, child_offsets_) == 9308, "ChipCallable wire ABI: child_offsets offset changed");
static_assert(offsetof(ChipCallable, child_count_) == 17500, "ChipCallable wire ABI: child_count offset changed");
static_assert(offsetof(ChipCallable, config_name_) == 17504, "ChipCallable wire ABI: config_name offset changed");
static_assert(
    offsetof(ChipCallable, config_name_len_) == 17568, "ChipCallable wire ABI: config_name_len offset changed"
);
static_assert(offsetof(ChipCallable, storage_) == 17584, "ChipCallable wire ABI: storage offset changed");
static_assert(sizeof(ChipCallable) == 17584, "ChipCallable wire ABI: header size changed");
static_assert(offsetof(CoreCallable, signature_) == 32, "CoreCallable wire ABI: signature offset changed");
static_assert(offsetof(CoreCallable, sig_count_) == 160, "CoreCallable wire ABI: sig_count offset changed");
static_assert(offsetof(CoreCallable, binary_size_) == 164, "CoreCallable wire ABI: binary_size offset changed");
static_assert(offsetof(CoreCallable, resolved_addr_) == 168, "CoreCallable wire ABI: resolved_addr offset changed");
static_assert(offsetof(CoreCallable, storage_) == 176, "CoreCallable wire ABI: storage offset changed");
static_assert(sizeof(CoreCallable) == 176, "CoreCallable wire ABI: header size changed");
static_assert(CoreCallable::binary_data_offset() == 192, "CoreCallable wire ABI: binary data offset changed");
static_assert(offsetof(CoreCallable, image_) == 0 && offsetof(ChipCallable, image_) == 0);
static_assert(offsetof(CoreCallable, async_capability_) == 16, "CoreCallable capability offset changed");

// Validate untrusted bytes before casting/reading any signature or child.
// UNKNOWN is a well-formed capability, but execution must qualify it separately.
template <typename Core = CoreCallable>
inline bool core_callable_image_valid(const void *image, size_t size) {
    if (size < Core::binary_data_offset() || reinterpret_cast<uintptr_t>(image) % alignof(Core) != 0 ||
        !callable_image_header_valid(image, size, CallableImageKind::CORE, sizeof(Core))) {
        return false;
    }
    const auto *core = static_cast<const Core *>(image);
    return core->sig_count_ >= 0 && core->sig_count_ <= Core::MAX_SIGNATURE_COUNT &&
           core->binary_size_ == size - Core::binary_data_offset() &&
           core_async_capability_valid(core->async_capability_);
}

inline bool chip_callable_image_valid(const void *image, size_t size) {
    if (size < sizeof(ChipCallable) || reinterpret_cast<uintptr_t>(image) % alignof(ChipCallable) != 0 ||
        !callable_image_header_valid(image, size, CallableImageKind::CHIP, sizeof(ChipCallable))) {
        return false;
    }
    const auto *chip = static_cast<const ChipCallable *>(image);
    if (chip->sig_count_ < 0 || chip->sig_count_ > CHIP_MAX_TENSOR_ARGS || chip->child_count_ < 0 ||
        chip->child_count_ > ChipCallable::MAX_CHILDREN || chip->func_name_len_ >= CALLABLE_FUNC_NAME_MAX ||
        chip->config_name_len_ >= CALLABLE_FUNC_NAME_MAX || chip->func_name_[chip->func_name_len_] != '\0' ||
        chip->config_name_[chip->config_name_len_] != '\0') {
        return false;
    }
    const size_t storage_size = size - offsetof(ChipCallable, storage_);
    size_t used = chip->binary_size_;
    if (used > storage_size) return false;
    for (int32_t i = 0; i < chip->child_count_; ++i) {
        if (used > SIZE_MAX - (CALLABLE_ALIGN - 1)) return false;
        const size_t aligned = (used + CALLABLE_ALIGN - 1) & ~(static_cast<size_t>(CALLABLE_ALIGN) - 1);
        const size_t offset = chip->child_offsets_[i];
        if (offset != aligned || offset > storage_size || sizeof(CallableImageHeader) > storage_size - offset) {
            return false;
        }
        CallableImageHeader child_header{};
        std::memcpy(&child_header, chip->storage_ + offset, sizeof(child_header));
        if (child_header.image_size > storage_size - offset ||
            !core_callable_image_valid(chip->storage_ + offset, child_header.image_size)) {
            return false;
        }
        used = offset + child_header.image_size;
    }
    return used == storage_size;
}

// ============================================================================
// Factory: make_callable for static leaf
// ============================================================================

template <int MaxSig>
std::vector<uint8_t> make_callable(
    const ArgDirection *sig, int32_t sig_count, const void *binary, uint32_t binary_size,
    const CoreAsyncCapability &capability
) {
    if (sig_count < 0 || sig_count > MaxSig) {
        throw std::invalid_argument(
            "make_callable: requested tensor count " + std::to_string(sig_count) + " exceeds supported tensor count " +
            std::to_string(MaxSig)
        );
    }
    if (sig_count > 0 && sig == nullptr)
        throw std::invalid_argument("make_callable: sig is required when sig_count > 0");
    if (binary_size > 0 && binary == nullptr) throw std::invalid_argument("make_callable: binary is required");
    if (!core_async_capability_valid(capability))
        throw std::invalid_argument("make_callable: invalid async capability");

    using T = Callable<void, MaxSig, 0>;
    size_t aligned_header = T::binary_data_offset();
    size_t total_size = aligned_header + binary_size;
    if (total_size > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("make_callable: image too large");
    std::vector<uint8_t> buf(total_size, 0);

    T *obj = reinterpret_cast<T *>(buf.data());
    obj->image_ = {
        CALLABLE_IMAGE_MAGIC, CALLABLE_IMAGE_VERSION, CallableImageKind::CORE, static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(total_size)
    };
    obj->async_capability_ = capability;
    for (int32_t i = 0; i < sig_count; ++i) {
        obj->signature_[i] = sig[i];
    }
    obj->sig_count_ = sig_count;
    obj->binary_size_ = binary_size;
    obj->resolved_addr_ = 0;
    if (binary_size > 0) std::memcpy(buf.data() + aligned_header, binary, binary_size);

    return buf;
}

template <int MaxSig>
std::vector<uint8_t>
make_callable(const ArgDirection *sig, int32_t sig_count, const void *binary, uint32_t binary_size) {
    return make_callable<MaxSig>(sig, sig_count, binary, binary_size, CoreAsyncCapability{});
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
    if (sig_count < 0 || sig_count > MaxSig) {
        throw std::invalid_argument(
            "make_callable: requested tensor count " + std::to_string(sig_count) + " exceeds supported tensor count " +
            std::to_string(MaxSig)
        );
    }
    if (child_count < 0) throw std::invalid_argument("make_callable: negative child_count");
    if (child_count > MaxChildren) throw std::invalid_argument("make_callable: child_count exceeds MaxChildren");
    if (sig_count > 0 && sig == nullptr) throw std::invalid_argument("make_callable: signature is required");
    if (binary_size > 0 && binary == nullptr) throw std::invalid_argument("make_callable: binary is required");
    if (child_count > 0 && (child_func_ids == nullptr || child_buffers == nullptr)) {
        throw std::invalid_argument("make_callable: child buffers and IDs are required");
    }

    using T = Callable<Child, MaxSig, MaxChildren>;
    size_t header_size = offsetof(T, storage_);

    size_t offset = binary_size;
    uint32_t child_offsets[MaxChildren];
    for (int32_t i = 0; i < child_count; ++i) {
        if (!core_callable_image_valid<Child>(child_buffers[i].data(), child_buffers[i].size())) {
            throw std::invalid_argument("make_callable: incompatible or invalid child image; rebuild the callable");
        }
        offset = (offset + CALLABLE_ALIGN - 1) & ~(static_cast<size_t>(CALLABLE_ALIGN) - 1);
        if (offset > std::numeric_limits<uint32_t>::max() ||
            child_buffers[i].size() > std::numeric_limits<uint32_t>::max() - offset) {
            throw std::invalid_argument("make_callable: child image offset overflow");
        }
        child_offsets[i] = static_cast<uint32_t>(offset);
        offset += child_buffers[i].size();
    }
    size_t total_size = header_size + offset;
    if (total_size > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("make_callable: image too large");
    std::vector<uint8_t> buf(total_size, 0);

    T *obj = reinterpret_cast<T *>(buf.data());
    obj->image_ = {
        CALLABLE_IMAGE_MAGIC, CALLABLE_IMAGE_VERSION, CallableImageKind::CHIP, static_cast<uint32_t>(header_size),
        static_cast<uint32_t>(total_size)
    };
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

inline std::vector<uint8_t> make_chip_callable(
    const ArgDirection *sig, int32_t sig_count, const char *func_name, const void *binary, uint32_t binary_size,
    const int32_t *child_func_ids, const std::vector<uint8_t> *child_buffers, int32_t child_count,
    const char *config_name
) {
    return make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, ChipCallable::MAX_CHILDREN>(
        sig, sig_count, func_name, binary, binary_size, child_func_ids, child_buffers, child_count, config_name
    );
}
