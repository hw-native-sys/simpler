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
 * PTO2TaskId — minimal standalone header.
 *
 * Factored out of pto_runtime2_types.h so that tensor.h can include it
 * without pulling in scheduler-internal constants (heap sizes, timeouts, etc.).
 */

#pragma once

#include <cstdint>

/**
 * TaskId: 64-bit encoding used across Runtime2.
 *
 * raw encoding: (ring_id << 32) | local_id
 *
 * ring_id:  which ring layer (0..PTO2_MAX_RING_DEPTH-1)
 * local_id: per-ring monotonic counter
 *
 * Invalid sentinel: raw == UINT64_MAX (no valid task has this encoding).
 */
struct PTO2TaskId {
    uint64_t raw;

    static constexpr PTO2TaskId make(uint8_t ring_id, uint32_t local_id) {
        return PTO2TaskId{(static_cast<uint64_t>(ring_id) << 32) | static_cast<uint64_t>(local_id)};
    }

    static constexpr PTO2TaskId invalid() { return PTO2TaskId{UINT64_MAX}; }

    constexpr uint8_t ring() const { return static_cast<uint8_t>(raw >> 32); }
    constexpr uint32_t local() const { return static_cast<uint32_t>(raw & 0xFFFFFFFFu); }
    constexpr bool is_valid() const { return raw != UINT64_MAX; }

    constexpr bool operator==(const PTO2TaskId& other) const { return raw == other.raw; }
    constexpr bool operator!=(const PTO2TaskId& other) const { return raw != other.raw; }
};

static_assert(sizeof(PTO2TaskId) == 8, "PTO2TaskId must stay 8 bytes (shared memory ABI)");
