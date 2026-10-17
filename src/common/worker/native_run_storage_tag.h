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

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

/**
 * The tag a prepared run's storage carries while its carrier object is alive.
 *
 * The platform writes into worker-owned storage. Zero means the carrier has
 * been destroyed, not that all bindings were released. A pre-execution refusal
 * whose cleanup succeeded instead leaves a release marker and its run epoch.
 * Reading either is a byte read, never a use of a destroyed carrier object.
 */
inline constexpr uint64_t kNativeRunContextMagic = UINT64_C(0x534d504c52554e31);  // "SMPLRUN1"

// Host-only result in the destroyed carrier's storage; no device image changes.
inline constexpr uint64_t kNativeRunPrepareReleasedMagic = UINT64_C(0x534d504c5052454c);  // "SMPLPREL"
inline constexpr size_t kNativeRunPrepareReleaseBytes = 2 * sizeof(uint64_t);

inline void publish_released_preparation(void *storage, uint64_t run_epoch) noexcept {
    const std::array<uint64_t, 2> receipt{kNativeRunPrepareReleasedMagic, run_epoch};
    std::memcpy(storage, receipt.data(), sizeof(receipt));
}

// Only this prepare attempt's result can release its inputs. An old epoch,
// an ordinary destruction tag, or short storage supplies no such proof.
inline bool consume_released_preparation(void *storage, size_t bytes, uint64_t run_epoch) noexcept {
    if (storage == nullptr || bytes < kNativeRunPrepareReleaseBytes) return false;
    std::array<uint64_t, 2> receipt{};
    std::memcpy(receipt.data(), storage, sizeof(receipt));
    if (receipt[0] != kNativeRunPrepareReleasedMagic || receipt[1] != run_epoch) return false;
    std::memset(storage, 0, sizeof(receipt));
    return true;
}
