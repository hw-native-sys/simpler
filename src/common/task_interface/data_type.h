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
enum class DataType : uint32_t {
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

/**
 * Get the size in bytes of a single element of the given data type
 *
 * @param dtype Data type
 * @return Size in bytes (0 for unknown types)
 */
inline uint64_t get_element_size(DataType dtype) {
    // Order must match the enum definition exactly
    static uint64_t data_type_size[static_cast<int>(DataType::DATA_TYPE_NUM)] = {
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
inline const char* get_dtype_name(DataType dtype) {
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

// Kernel-callable qualifier: __may_used_by_aicore__ (from kernel_args.h)
// signals an incore build where __aicore__ is available (ccec or sim).
// In orchestration / host builds, PTO_DEVICE_FUNC expands to nothing.
#ifdef __may_used_by_aicore__
#define PTO_DEVICE_FUNC __aicore__
#else
#define PTO_DEVICE_FUNC
#endif

namespace detail {

template <typename T>
union U64Converter {
    uint64_t bits;
    T value;
};

}  // namespace detail

/**
 * Pack a value into uint64_t storage (zero-extends smaller types).
 * Uses union-based type punning — works on both host and AICore (no memcpy).
 *
 *   uint64_t bits = to_u64(3.14f);        // float → uint64_t
 *   uint64_t bits = to_u64(int32_t(42));  // int32 → uint64_t
 */
template <typename T>
PTO_DEVICE_FUNC inline uint64_t to_u64(T value) {
    static_assert(sizeof(T) <= sizeof(uint64_t), "to_u64: type must fit in 8 bytes");
#if PTO_HAS_TYPE_TRAITS
    static_assert(std::is_trivially_copyable<T>::value, "to_u64: type must be trivially copyable");
#endif
    detail::U64Converter<T> conv;
    conv.bits = 0;
    conv.value = value;
    return conv.bits;
}

/**
 * Unpack a value from uint64_t storage.
 * Uses union-based type punning — works on both host and AICore (no memcpy).
 *
 *   float f   = from_u64<float>(bits);
 *   int32_t i = from_u64<int32_t>(bits);
 */
template <typename T>
PTO_DEVICE_FUNC inline T from_u64(uint64_t bits) {
    static_assert(sizeof(T) <= sizeof(uint64_t), "from_u64: type must fit in 8 bytes");
#if PTO_HAS_TYPE_TRAITS
    static_assert(std::is_trivially_copyable<T>::value, "from_u64: type must be trivially copyable");
#endif
    detail::U64Converter<T> conv;
    conv.bits = bits;
    return conv.value;
}

#endif  // SRC_COMMON_TASK_INTERFACE_DATA_TYPE_H_
