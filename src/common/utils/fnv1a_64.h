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

#ifndef SIMPLER_COMMON_UTILS_FNV1A_64_H_
#define SIMPLER_COMMON_UTILS_FNV1A_64_H_

#include <cstddef>
#include <cstdint>

namespace simpler::common::utils {

// FNV-1a 64-bit content hash. Deterministic, allocation-free, ~µs / MB.
// Used as a generic content-keyed dedup key (ChipCallable buffer hashing in
// DeviceRunner, ELF Build-ID fallback in elf_build_id.h, etc.).
inline uint64_t fnv1a_64(const void *data, std::size_t len) {
    constexpr uint64_t kPrime = 0x00000100000001b3ULL;
    uint64_t h = 0xcbf29ce484222325ULL;
    const auto *p = static_cast<const uint8_t *>(data);
    for (std::size_t i = 0; i < len; ++i) {
        h ^= p[i];
        h *= kPrime;
    }
    return h;
}

}  // namespace simpler::common::utils

#endif  // SIMPLER_COMMON_UTILS_FNV1A_64_H_
