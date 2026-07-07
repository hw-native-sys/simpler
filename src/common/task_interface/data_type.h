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
 * Data Type Definitions and Conversion Utilities
 *
 * Defines supported data types, element size helpers, and type-safe
 * packing/unpacking of values into uint64_t (the universal scalar storage
 * type in the orchestration framework).
 */

#ifndef SRC_COMMON_TASK_INTERFACE_DATA_TYPE_H_
#define SRC_COMMON_TASK_INTERFACE_DATA_TYPE_H_

#include <array>
#include <cstdint>
#include <type_traits>

// Kernel-callable qualifier: when compiling for AICore (ccec compiler defines
// __DAV_VEC__ or __DAV_CUBE__), PTO_DEVICE_FUNC adds the __aicore__ attribute.
// In orchestration / host builds, PTO_DEVICE_FUNC expands to nothing. Defined
// up here (before get_element_size / get_dtype_name) so those helpers can be
// marked device-callable for the distributed engine's AICore replay path.
#if defined(__DAV_VEC__) || defined(__DAV_CUBE__)
// Ensure __aicore__ is available (CCE attribute for bisheng compiler).
// Platform headers (inner_kernel.h) normally define this, but data_type.h
// may be included before them.
#ifndef __aicore__
#define __aicore__ [aicore]
#endif
#define PTO_DEVICE_FUNC __aicore__
// Dual host/aicore attribution: some shared Tensor methods (init_from / copy /
// view) are called both from __host__ orchestration code and from __aicore__
// engine code. Real CCEC (clang-based) accepts __host__ __aicore__; the sim
// build is g++ with __DAV_VEC__ pre-defined (so __aicore__ is a no-op) and has
// no __host__ attribute, so leave it empty there — sim runs everything as host
// threads, so no host/aicore split exists.
#if defined(__clang__)
#define PTO_HOST_DEVICE_FUNC __host__ __aicore__
#else
#define PTO_HOST_DEVICE_FUNC
#endif
#else
#define PTO_DEVICE_FUNC
#define PTO_HOST_DEVICE_FUNC
#endif

// __gm__ is a CCEC address-space keyword (global memory) on AICore. On host /
// sim / AICPU builds it has no meaning, so map it to nothing — lets the shared
// runtime headers (tensor.h, dist_engine.h, ...) declare __gm__-qualified free
// functions that compile unchanged on both sides. (CCEC treats __gm__ as a
// builtin keyword, not a macro, so do NOT define it on AICore.)
#if !defined(__DAV_VEC__) && !defined(__DAV_CUBE__) && !defined(__CCE_AICORE__)
#ifndef __gm__
#define __gm__
#endif
#endif

// AICore (CCEC) libc memcpy is __host__ (not callable from __aicore__) and
// __builtin_memcpy can't lower address-space-1 (GM) memory intrinsics. The
// runtime headers below (tensor.h, pto_types.h, pto_tensormap.h,
// tensor_create_info.h, dist_engine.cpp) memcpy 64B cache lines between local
// and/or __gm__ objects from __aicore__ context. Provide __aicore__ byte-loop
// overloads of ::memcpy in the global namespace covering the generic/generic,
// __gm__/generic, generic/__gm__, and __gm__/__gm__ pointer combinations (CCEC
// rejects cross-address-space pointer casts, so each combination needs its own
// overload with address-space-correct char pointers). Unqualified memcpy calls
// resolve to these overloads (the libc ::memcpy is __host__, hence not viable
// from __aicore__); qualified std::memcpy calls in host-only helpers are not
// instantiated on AICore. data_type.h is included before every other runtime
// header, so these overloads are visible at every call site.
#ifdef __CCE_AICORE__
__aicore__ inline void *memcpy(void *d, const void *s, unsigned long n) {
    auto *dp = (unsigned char *)d;
    const auto *sp = (const unsigned char *)s;
    for (unsigned long i = 0; i < n; ++i) dp[i] = sp[i];
    return d;
}
__aicore__ inline __gm__ void *memcpy(__gm__ void *d, const void *s, unsigned long n) {
    auto *dp = (__gm__ unsigned char *)d;
    const auto *sp = (const unsigned char *)s;
    for (unsigned long i = 0; i < n; ++i) dp[i] = sp[i];
    return d;
}
__aicore__ inline void *memcpy(void *d, __gm__ const void *s, unsigned long n) {
    auto *dp = (unsigned char *)d;
    const auto *sp = (const __gm__ unsigned char *)s;
    for (unsigned long i = 0; i < n; ++i) dp[i] = sp[i];
    return d;
}
__aicore__ inline __gm__ void *memcpy(__gm__ void *d, __gm__ const void *s, unsigned long n) {
    auto *dp = (__gm__ unsigned char *)d;
    const auto *sp = (const __gm__ unsigned char *)s;
    for (unsigned long i = 0; i < n; ++i) dp[i] = sp[i];
    return d;
}
#endif

template <typename T>
inline constexpr bool is_supported_scalar_arg_v = std::is_arithmetic_v<std::remove_cv_t<std::remove_reference_t<T>>> ||
                                                  std::is_enum_v<std::remove_cv_t<std::remove_reference_t<T>>>;

/**
 * Supported data types for tensor elements
 */
enum class DataType : uint8_t {
    FLOAT32,   // 4 bytes
    FLOAT16,   // 2 bytes
    INT32,     // 4 bytes
    INT16,     // 2 bytes
    INT8,      // 1 byte
    UINT8,     // 1 byte
    BFLOAT16,  // 2 bytes
    INT64,     // 8 bytes
    UINT64,    // 8 bytes
    UINT16,    // 2 bytes
    UINT32,    // 4 bytes
    BOOL,      // 1 byte (stored as 1/0 in uint64_t slot)
    DATA_TYPE_NUM,
};

static_assert(sizeof(DataType) == 1, "DataType must stay 1 byte");

/**
 * Get the size in bytes of a single element of the given data type
 *
 * @param dtype Data type
 * @return Size in bytes (0 for unknown types)
 */
PTO_DEVICE_FUNC inline uint64_t get_element_size(DataType dtype) {
    // Plain C array (not std::array) so indexing works on AICore (CCEC), where
    // std::array::operator[] is host-attributed and not callable from __aicore__.
    constexpr static uint64_t data_type_size[static_cast<int>(DataType::DATA_TYPE_NUM)] = {
        4,  // case DataType::FLOAT32
        2,  // DataType::FLOAT16
        4,  // DataType::INT32
        2,  // DataType::INT16
        1,  // DataType::INT8
        1,  // DataType::UINT8
        2,  // DataType::BFLOAT16
        8,  // DataType::INT64
        8,  // DataType::UINT64
        2,  // DataType::UINT16
        4,  // DataType::UINT32
        1,  // DataType::BOOL
    };
    return data_type_size[static_cast<int>(dtype)];
}

/**
 * Get the name of a data type as a string
 *
 * @param dtype Data type
 * @return String name of the data type
 */
PTO_DEVICE_FUNC inline const char *get_dtype_name(DataType dtype) {
#ifdef __CCE_AICORE__
    // AICore (CCEC): string literals live in __gm__, so returning them as a
    // generic `const char*` is an illegal address-space cast. get_dtype_name is
    // purely diagnostic (used in error/trace text, which compiles to no-ops on
    // AICore), so stub it here.
    (void)dtype;
    return nullptr;
#else
    switch (dtype) {
    case DataType::FLOAT32:
        return "FLOAT32";
    case DataType::FLOAT16:
        return "FLOAT16";
    case DataType::INT32:
        return "INT32";
    case DataType::INT16:
        return "INT16";
    case DataType::INT8:
        return "INT8";
    case DataType::UINT8:
        return "UINT8";
    case DataType::BFLOAT16:
        return "BFLOAT16";
    case DataType::INT64:
        return "INT64";
    case DataType::UINT64:
        return "UINT64";
    case DataType::UINT16:
        return "UINT16";
    case DataType::UINT32:
        return "UINT32";
    case DataType::BOOL:
        return "BOOL";
    default:
        return "UNKNOWN";
    }
#endif
}

// =============================================================================
// uint64_t Packing/Unpacking Utilities
// =============================================================================

// PTO_DEVICE_FUNC is defined near the top of this file (before
// get_element_size / get_dtype_name) so those helpers can use it.

// -----------------------------------------------------------------------------
// Unified template interface for all targets (AICore + CPU).
//
// ccec (Bisheng CCE compiler) does not support template *classes*, but does
// support template *functions* with __aicore__ — verified by existing kernel
// code (e.g. CeilAlign<T>, qk_matmul_impl<M,K,N>).  We use anonymous unions
// inside each function body to avoid any template class dependency.
//
// Named convenience functions (from_u64_f32 etc.) are removed — use the
// template form from_u64<T>() / to_u64() directly.

/**
 * Map a C++ type to its DataType enum value.
 *
 * Used to preserve original scalar type information through the
 * Arg -> PTO2TaskPayload -> tensor dump pipeline.
 */
template <typename T>
inline constexpr uint8_t dtype_of() {
    static_assert(is_supported_scalar_arg_v<T>, "dtype_of: type must be arithmetic or enum");
    if constexpr (std::is_same_v<T, float>) {
        return static_cast<uint8_t>(DataType::FLOAT32);
    } else if constexpr (std::is_same_v<T, double>) {
        return static_cast<uint8_t>(DataType::FLOAT32);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return static_cast<uint8_t>(DataType::INT32);
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return static_cast<uint8_t>(DataType::UINT32);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return static_cast<uint8_t>(DataType::INT64);
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        return static_cast<uint8_t>(DataType::UINT64);
    } else if constexpr (std::is_same_v<T, int16_t>) {
        return static_cast<uint8_t>(DataType::INT16);
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return static_cast<uint8_t>(DataType::UINT16);
    } else if constexpr (std::is_same_v<T, int8_t>) {
        return static_cast<uint8_t>(DataType::INT8);
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        return static_cast<uint8_t>(DataType::UINT8);
    } else if constexpr (std::is_same_v<T, bool>) {
        return static_cast<uint8_t>(DataType::BOOL);
    } else {
        return static_cast<uint8_t>(DataType::UINT64);
    }
}

/**
 * Pack a value into uint64_t storage (zero-extends smaller types).
 *
 *   uint64_t bits = to_u64(3.14f);        // float -> uint64_t
 *   uint64_t bits = to_u64(int32_t(42));  // int32 -> uint64_t
 */
template <typename T>
PTO_DEVICE_FUNC inline uint64_t to_u64(T value) {
    static_assert(sizeof(T) <= sizeof(uint64_t), "to_u64: type must fit in 8 bytes");
    static_assert(std::is_trivially_copyable_v<T>, "to_u64: type must be trivially copyable");
    union {
        uint64_t u;
        T v;
    } c;
    c.u = 0;
    c.v = value;
    return c.u;
}

/**
 * Unpack a value from uint64_t storage.
 *
 *   float f   = from_u64<float>(bits);
 *   int32_t i = from_u64<int32_t>(bits);
 */
template <typename T>
PTO_DEVICE_FUNC inline T from_u64(uint64_t bits) {
    static_assert(sizeof(T) <= sizeof(uint64_t), "from_u64: type must fit in 8 bytes");
    static_assert(std::is_trivially_copyable_v<T>, "from_u64: type must be trivially copyable");
    union {
        uint64_t u;
        T v;
    } c;
    c.u = bits;
    return c.v;
}

#endif  // SRC_COMMON_TASK_INTERFACE_DATA_TYPE_H_
