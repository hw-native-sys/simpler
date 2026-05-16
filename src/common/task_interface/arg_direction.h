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
 * ArgDirection - Argument direction enum for Callable signatures
 *
 * Defines whether each argument slot is a scalar, input tensor, output tensor,
 * or in-place (input+output) tensor.
 */

#pragma once

#include <cstdint>

enum class ArgDirection : int32_t {
    SCALAR = 0,
    IN = 1,
    OUT = 2,
    INOUT = 3,
};

inline constexpr int CORE_MAX_TENSOR_ARGS = 16;
// Chip-level entry-tensor cap. Sizes ChipCallable::signature_[] and
// ChipStorageTaskArgs::tensors_[], both of which cross the host->device wire
// as fixed POD — raising this is an additive ABI change (existing callers
// still fit; transient storage grows by 64 * sizeof(ContinuousTensor)).
inline constexpr int CHIP_MAX_TENSOR_ARGS = 128;
inline constexpr int CORE_MAX_SCALAR_ARGS = 32;
inline constexpr int CHIP_MAX_SCALAR_ARGS = 128;
inline constexpr uint32_t CALLABLE_ALIGN = 64;

static inline uint32_t callable_align_up(uint32_t size) { return (size + CALLABLE_ALIGN - 1) & ~(CALLABLE_ALIGN - 1); }

inline const char *arg_direction_name(ArgDirection d) {
    switch (d) {
    case ArgDirection::SCALAR:
        return "SCALAR";
    case ArgDirection::IN:
        return "IN";
    case ArgDirection::OUT:
        return "OUT";
    case ArgDirection::INOUT:
        return "INOUT";
    default:
        return "UNKNOWN";
    }
}
