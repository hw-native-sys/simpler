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

#include "device.h"

namespace async_case {
constexpr int64_t kSdmaRecords = 64;
constexpr int64_t kSdmaMetadataWords = 256;
constexpr int64_t kSdmaAddresses = 128;

inline __aicore__ uint64_t read_completion(__gm__ uint64_t *address) {
#ifdef PTO_CPUSTUB_HPP
    return __atomic_load_n(address, __ATOMIC_ACQUIRE);
#else
    auto *scratch = reinterpret_cast<__ubuf__ uint64_t *>(0);
    copy_gm_to_ubuf_align_v2(
        reinterpret_cast<__ubuf__ uint32_t *>(scratch), reinterpret_cast<__gm__ uint32_t *>(address), 0, 1,
        sizeof(uint64_t), 0, 0, 0, 0, 0, 0
    );
    pipe_barrier(PIPE_ALL);
    return *scratch;
#endif
}

inline __aicore__ bool sdma_ready(__gm__ uint64_t *meta) {
    const uint64_t count = observe(meta + 112);
    if (count > kSdmaRecords) return false;
    for (uint64_t i = 0; i < count; ++i) {
        // Addresses and thresholds come from the public completion-record API.
#ifndef PTO_CPUSTUB_HPP
        dcci(meta + kSdmaAddresses + 2 * i, SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
#endif
        auto *address = reinterpret_cast<__gm__ uint64_t *>(meta[kSdmaAddresses + 2 * i]);
        if (read_completion(address) < meta[kSdmaAddresses + 2 * i + 1]) return false;
    }
    return true;
}
}  // namespace async_case
