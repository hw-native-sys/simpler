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

template <typename T>
inline constexpr bool is_supported_scalar_arg_v = std::is_arithmetic_v<std::remove_cv_t<std::remove_reference_t<T>>> ||
                                                  std::is_enum_v<std::remove_cv_t<std::remove_reference_t<T>>>;

// Maximum tensor rank a view can describe. Both the L3+ `Tensor` of buffer.h and the device
// `ChipTensor` of tensor.h size their shapes[] / strides[] by it, so it is shared here rather than
// owned by either.
constexpr int MAX_TENSOR_DIMS = 5;

// Memory space of a backing. Orthogonal to location (local/remote, derived) and to visibility.
// Both BufferDescriptor and ChipTensor store it, so it is shared here rather
// than owned by either.
enum class AddressSpace : uint8_t {
    HOST = 0,
    DEVICE = 1,
};

// Per-argument request, independent of the backing's physical address space. H2D requests
// Program-managed device storage; the callable direction decides copy-in and copy-back.
enum class TensorTransfer : uint8_t {
    NONE = 0,
    H2D = 1,
    D2H = 2,  // Reserved request; automatic D2H parameters are not supported.
};

inline constexpr TensorTransfer legacy_tensor_transfer(AddressSpace space) {
    return space == AddressSpace::HOST ? TensorTransfer::H2D : TensorTransfer::NONE;
}

inline const char *tensor_transfer_error(AddressSpace space, TensorTransfer transfer) {
    if (space != AddressSpace::HOST && space != AddressSpace::DEVICE) return "unknown address space";
    switch (transfer) {
    case TensorTransfer::NONE:
        return nullptr;
    case TensorTransfer::H2D:
        return space == AddressSpace::HOST ? nullptr : "H2D requires HOST backing";
    case TensorTransfer::D2H:
        return "automatic D2H parameters are unsupported; use explicit copy_from";
    default:
        return "unknown tensor transfer";
    }
}

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
    // MX block-scale dtypes — A5 only (consumed by the A5-only pto.tquant.mx /
    // tmatmul.mx ops). Host transfer itself is arch-agnostic (1 byte/element).
    // FP8E5M2 is intentionally omitted: MX paths use E4M3FN (+ E8M0 scale), not E5M2.
    FP8E4M3FN,  // 1 byte — MXFP8 E4M3FN (A5 only)
    FP8E8M0,    // 1 byte — MXFP8 E8M0 shared exponent (A5 only)
    FP4E2M1,    // 1 byte — MXFP4 E2M1, 2 values packed (torch float4_e2m1fn_x2) (A5 only)
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
        2,  // DataType::UINT16
        4,  // DataType::UINT32
        1,  // DataType::BOOL
        1,  // DataType::FP8E4M3FN (A5 only)
        1,  // DataType::FP8E8M0 (A5 only)
        1,  // DataType::FP4E2M1 (A5 only)
    };
    // Bounds-checked: `dtype` is a raw u8 on the wire, so a decoder can hand this any value.
    const auto index = static_cast<std::size_t>(dtype);
    return index < data_type_size.size() ? data_type_size[index] : 0;
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
    case DataType::UINT16:
        return "UINT16";
    case DataType::UINT32:
        return "UINT32";
    case DataType::BOOL:
        return "BOOL";
    case DataType::FP8E4M3FN:  // A5 only
        return "FP8E4M3FN";
    case DataType::FP8E8M0:  // A5 only
        return "FP8E8M0";
    case DataType::FP4E2M1:  // A5 only
        return "FP4E2M1";
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
#define __aicore__ [aicore]
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

/**
 * Map a C++ type to its DataType enum value.
 *
 * Used to preserve original scalar type information through the
 * Arg -> TaskPayload -> args dump pipeline.
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
