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
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "../utils/fnv1a_64.h"
#include "arg_direction.h"

enum class CoreInputCacheMode : uint8_t { UNKNOWN = 0, NO_CACHED_READS = 1, SCALAR_RANGES = 2 };

// A cached read starts at buffer.addr + start_offset * start_offset_scale + byte_offset.
struct CoreInputCacheRange {
    uint8_t tensor_arg;
    uint8_t start_offset_scale;
    uint16_t byte_length;
    uint32_t byte_offset;
};

inline constexpr uint32_t CORE_INPUT_CACHE_PROOF_MAGIC = 0x31504348U;  // "HCP1", little endian
inline constexpr size_t CORE_INPUT_CACHE_MAX_RANGES = 3;
inline constexpr uint8_t CORE_INPUT_CACHE_AIC = 1;
inline constexpr uint8_t CORE_INPUT_CACHE_AIV = 2;
inline constexpr uint8_t CORE_INPUT_CACHE_BOTH = CORE_INPUT_CACHE_AIC | CORE_INPUT_CACHE_AIV;

// The compiler appends this footer to an extracted A5 image. CoreCallable packing moves
// it into the aligned child header, outside the executable image. Legacy headers contain zeroes.
struct CoreInputCacheProof {
    uint32_t magic;
    CoreInputCacheMode mode;
    uint8_t range_count;
    uint8_t min_tensor_count;  // Includes descriptor-only tensor reads in the kernel.
    uint8_t core_mask;         // Executor families for which this footprint was proven.
    uint64_t digest;
    CoreInputCacheRange ranges[CORE_INPUT_CACHE_MAX_RANGES];
};

static_assert(std::is_standard_layout_v<CoreInputCacheProof> && std::is_trivially_copyable_v<CoreInputCacheProof>);
static_assert(sizeof(CoreInputCacheRange) == 8 && offsetof(CoreInputCacheRange, byte_offset) == 4);
static_assert(sizeof(CoreInputCacheProof) == 40 && offsetof(CoreInputCacheProof, digest) == 8);
static_assert(offsetof(CoreInputCacheProof, ranges) == 16);

inline uint64_t core_input_cache_proof_digest(const CoreInputCacheProof &proof, const void *text, size_t size) {
    using simpler::common::utils::fnv1a_64;
    using simpler::common::utils::fnv1a_64_append;
    uint64_t hash = fnv1a_64(text, size);
    hash = fnv1a_64_append(hash, &proof, offsetof(CoreInputCacheProof, digest));
    return fnv1a_64_append(hash, proof.ranges, sizeof(proof.ranges));
}

inline bool core_input_cache_proof_matches_text(const CoreInputCacheProof &proof, const void *text, size_t size) {
    if (proof.magic != CORE_INPUT_CACHE_PROOF_MAGIC || proof.core_mask == 0 ||
        proof.core_mask > CORE_INPUT_CACHE_BOTH || proof.min_tensor_count > CORE_MAX_TENSOR_ARGS ||
        proof.range_count > CORE_INPUT_CACHE_MAX_RANGES || size == 0 || text == nullptr)
        return false;
    const auto expected_mode =
        proof.range_count == 0 ? CoreInputCacheMode::NO_CACHED_READS : CoreInputCacheMode::SCALAR_RANGES;
    if (proof.mode != expected_mode) return false;
    for (size_t i = 0; i < CORE_INPUT_CACHE_MAX_RANGES; ++i) {
        const auto &range = proof.ranges[i];
        if (i >= proof.range_count) {
            if (range.tensor_arg != 0 || range.start_offset_scale != 0 || range.byte_length != 0 ||
                range.byte_offset != 0)
                return false;
            continue;
        }
        const auto scale = range.start_offset_scale;
        if (range.tensor_arg >= proof.min_tensor_count || range.byte_length == 0 ||
            (scale != 0 && scale != 1 && scale != 2 && scale != 4 && scale != 8))
            return false;
    }
    return proof.digest == core_input_cache_proof_digest(proof, text, size);
}
