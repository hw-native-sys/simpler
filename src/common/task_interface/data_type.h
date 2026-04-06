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

#if __has_include(<type_traits>)
#include <type_traits>
#define PTO_HAS_TYPE_TRAITS 1
#else
#define PTO_HAS_TYPE_TRAITS 0
#endif

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
    DATA_TYPE_NUM,
};

static_assert(sizeof(DataType) == 1, "DataType must stay 1 byte");

/**
 * Get the size in bytes of a single element of the given data type
 *
 * @param dtype Data type
 * @return Size in bytes (0 for unknown types)
 */
inline uint64_t get_element_size(DataType dtype) {
    // Order must match the enum definition exactly
    constexpr static std::array<uint64_t, static_cast<int>(DataType::DATA_TYPE_NUM)> data_type_size = {
        4,  // case DataType::FLOAT32
        2,  // DataType::FLOAT16
        4,  // DataType::INT32
        2,  // DataType::INT16
        1,  // DataType::INT8
        1,  // DataType::UINT8
        2,  // DataType::BFLOAT16
        8,  // DataType::INT64
        8,  // DataType::UINT64
    };
    return data_type_size[static_cast<int>(dtype)];
}

/**
 * Get the name of a data type as a string
 *
 * @param dtype Data type
 * @return String name of the data type
 */
inline const char *get_dtype_name(DataType dtype) {
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
    default:
        return "UNKNOWN";
    }
}

// =============================================================================
// uint64_t Packing/Unpacking Utilities
// =============================================================================

// Kernel-callable qualifier: when compiling for AICore (ccec compiler defines
// __DAV_VEC__ or __DAV_CUBE__), PTO_DEVICE_FUNC adds the __aicore__ attribute.
// In orchestration / host builds, PTO_DEVICE_FUNC expands to nothing.
#if defined(__DAV_VEC__) || defined(__DAV_CUBE__)
// Ensure __aicore__ is available (CCE attribute for bisheng compiler).
// Platform headers (inner_kernel.h) normally define this, but data_type.h
// may be included before them.
#ifndef __aicore__
#define __aicore__ [aicore]  // NOLINT(whitespace/braces)
#endif
#define PTO_DEVICE_FUNC __aicore__
#else
#define PTO_DEVICE_FUNC
#endif

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
// -----------------------------------------------------------------------------

/**
 * Pack a value into uint64_t storage (zero-extends smaller types).
 *
 *   uint64_t bits = to_u64(3.14f);        // float -> uint64_t
 *   uint64_t bits = to_u64(int32_t(42));  // int32 -> uint64_t
 */
template <typename T>
PTO_DEVICE_FUNC inline uint64_t to_u64(T value) {
    static_assert(sizeof(T) <= sizeof(uint64_t), "to_u64: type must fit in 8 bytes");
#if PTO_HAS_TYPE_TRAITS
    static_assert(std::is_trivially_copyable_v<T>, "to_u64: type must be trivially copyable");
#endif
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
#if PTO_HAS_TYPE_TRAITS
    static_assert(std::is_trivially_copyable_v<T>, "from_u64: type must be trivially copyable");
#endif
    union {
        uint64_t u;
        T v;
    } c;
    c.u = bits;
    return c.v;
}

#endif  // SRC_COMMON_TASK_INTERFACE_DATA_TYPE_H_
