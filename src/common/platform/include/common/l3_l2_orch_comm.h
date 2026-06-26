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

#ifndef SRC_COMMON_PLATFORM_INCLUDE_COMMON_L3_L2_ORCH_COMM_H_
#define SRC_COMMON_PLATFORM_INCLUDE_COMMON_L3_L2_ORCH_COMM_H_

#include <stddef.h>
#include <stdint.h>

static constexpr uint32_t L3L2_ORCH_COMM_MAGIC = 0x4C334C32u;  // "L3L2"
static constexpr uint16_t L3L2_ORCH_COMM_ABI_MAJOR = 2;
static constexpr uint16_t L3L2_ORCH_COMM_ABI_MINOR = 0;
static constexpr size_t L3L2_ORCH_REGION_DESC_SCALAR_COUNT = 6;
static constexpr uint64_t L3L2_ORCH_COMM_COUNTER_BYTES = sizeof(int32_t);
static constexpr uint64_t L3L2_ORCH_COMM_COUNTER_BASE_ALIGNMENT = 64;

struct L3L2OrchRegionDesc {
    uint64_t magic_version;
    uint64_t region_id;
    uint64_t payload_base;
    uint64_t payload_bytes;
    uint64_t counter_base;
    uint64_t counter_bytes;
};

enum class L3L2OrchCommCmd : uint32_t {
    ALLOC_REGION = 1,
    FREE_REGION = 2,
    PAYLOAD_WRITE = 3,
    PAYLOAD_READ = 4,
    SIGNAL_NOTIFY = 5,
    SIGNAL_WAIT = 6,
    SIGNAL_TEST = 7,
};

enum class L3L2OrchNotifyOp : uint32_t {
    Set = 0,
    Add = 1,
};

enum class L3L2OrchWaitCmp : uint32_t {
    EQ = 0,
    NE = 1,
    GT = 2,
    GE = 3,
    LT = 4,
    LE = 5,
};

struct L3L2OrchSignalTestResult {
    bool matched;
    int32_t observed;
};

enum class L3L2OrchCommValidationError : uint32_t {
    OK = 0,
    BAD_MAGIC_VERSION = 1,
    BAD_REGION_ID = 2,
    BAD_PAYLOAD_RANGE = 3,
    BAD_COUNTER_RANGE = 4,
    OUT_OF_BOUNDS = 5,
    BAD_SCALAR_COUNT = 6,
    NULL_POINTER = 7,
};

struct L3L2OrchCommRequest {
    uint32_t cmd;
    uint32_t op;
    uint64_t region_id;
    uint64_t payload_offset;
    uint64_t host_ptr;
    uint64_t payload_bytes;
    uint64_t counter_addr;
    uint64_t counter_bytes;
    int32_t counter_operand;
    uint32_t reserved0;
    uint64_t timeout_ns;
};

struct L3L2OrchCommResponse {
    int32_t status;
    uint32_t error_kind;
    uint64_t region_id;
    int32_t observed_counter;
    uint32_t matched;
    L3L2OrchRegionDesc desc;
    char message[256];
};

static inline uint64_t l3_l2_orch_comm_pack_magic_version(uint32_t magic, uint16_t major, uint16_t minor) {
    return (static_cast<uint64_t>(magic) << 32) | (static_cast<uint64_t>(major) << 16) | static_cast<uint64_t>(minor);
}

static inline uint64_t l3_l2_orch_comm_magic_version() {
    return l3_l2_orch_comm_pack_magic_version(L3L2_ORCH_COMM_MAGIC, L3L2_ORCH_COMM_ABI_MAJOR, L3L2_ORCH_COMM_ABI_MINOR);
}

static inline uint32_t l3_l2_orch_comm_magic(uint64_t magic_version) {
    return static_cast<uint32_t>(magic_version >> 32);
}

static inline uint16_t l3_l2_orch_comm_abi_major(uint64_t magic_version) {
    return static_cast<uint16_t>((magic_version >> 16) & 0xFFFFu);
}

static inline uint16_t l3_l2_orch_comm_abi_minor(uint64_t magic_version) {
    return static_cast<uint16_t>(magic_version & 0xFFFFu);
}

static inline bool l3_l2_orch_comm_add_overflows(uint64_t a, uint64_t b) { return a > UINT64_MAX - b; }

static inline bool l3_l2_orch_comm_is_aligned(uint64_t value, uint64_t align) {
    return align != 0 && (value % align) == 0;
}

static inline bool l3_l2_orch_comm_range_contains(uint64_t base, uint64_t size, uint64_t value) {
    if (size == 0 || l3_l2_orch_comm_add_overflows(base, size)) {
        return false;
    }
    return value >= base && value < base + size;
}

static inline bool
l3_l2_orch_comm_ranges_overlap(uint64_t first_base, uint64_t first_size, uint64_t second_base, uint64_t second_size) {
    if (first_size == 0 || second_size == 0 || l3_l2_orch_comm_add_overflows(first_base, first_size) ||
        l3_l2_orch_comm_add_overflows(second_base, second_size)) {
        return false;
    }
    return first_base < second_base + second_size && second_base < first_base + first_size;
}

static inline L3L2OrchCommValidationError
l3_l2_orch_comm_validate_payload_bounds(uint64_t offset, uint64_t nbytes, uint64_t payload_bytes) {
    if (nbytes == 0 || payload_bytes == 0 || l3_l2_orch_comm_add_overflows(offset, nbytes)) {
        return L3L2OrchCommValidationError::BAD_PAYLOAD_RANGE;
    }
    if (offset > payload_bytes || offset + nbytes > payload_bytes) {
        return L3L2OrchCommValidationError::OUT_OF_BOUNDS;
    }
    return L3L2OrchCommValidationError::OK;
}

static inline L3L2OrchCommValidationError l3_l2_orch_comm_validate_counter_range(const L3L2OrchRegionDesc &desc) {
    if (desc.counter_base == 0 || desc.counter_bytes == 0 ||
        !l3_l2_orch_comm_is_aligned(desc.counter_base, L3L2_ORCH_COMM_COUNTER_BASE_ALIGNMENT) ||
        (desc.counter_bytes % L3L2_ORCH_COMM_COUNTER_BYTES) != 0 ||
        l3_l2_orch_comm_add_overflows(desc.counter_base, desc.counter_bytes)) {
        return L3L2OrchCommValidationError::BAD_COUNTER_RANGE;
    }
    if (l3_l2_orch_comm_ranges_overlap(desc.payload_base, desc.payload_bytes, desc.counter_base, desc.counter_bytes)) {
        return L3L2OrchCommValidationError::BAD_COUNTER_RANGE;
    }
    return L3L2OrchCommValidationError::OK;
}

static inline L3L2OrchCommValidationError l3_l2_orch_comm_validate_desc(const L3L2OrchRegionDesc &desc) {
    if (l3_l2_orch_comm_magic(desc.magic_version) != L3L2_ORCH_COMM_MAGIC ||
        l3_l2_orch_comm_abi_major(desc.magic_version) != L3L2_ORCH_COMM_ABI_MAJOR) {
        return L3L2OrchCommValidationError::BAD_MAGIC_VERSION;
    }
    if (desc.region_id == 0) {
        return L3L2OrchCommValidationError::BAD_REGION_ID;
    }
    if (desc.payload_base == 0 || desc.payload_bytes == 0 ||
        l3_l2_orch_comm_add_overflows(desc.payload_base, desc.payload_bytes)) {
        return L3L2OrchCommValidationError::BAD_PAYLOAD_RANGE;
    }
    L3L2OrchCommValidationError counter_error = l3_l2_orch_comm_validate_counter_range(desc);
    if (counter_error != L3L2OrchCommValidationError::OK) {
        return counter_error;
    }
    return L3L2OrchCommValidationError::OK;
}

static inline bool l3_l2_orch_comm_encode_desc(const L3L2OrchRegionDesc &desc, uint64_t *scalars, size_t scalar_count) {
    if (scalars == nullptr || scalar_count < L3L2_ORCH_REGION_DESC_SCALAR_COUNT) {
        return false;
    }
    scalars[0] = desc.magic_version;
    scalars[1] = desc.region_id;
    scalars[2] = desc.payload_base;
    scalars[3] = desc.payload_bytes;
    scalars[4] = desc.counter_base;
    scalars[5] = desc.counter_bytes;
    return true;
}

static inline bool l3_l2_orch_comm_decode_desc(
    const uint64_t *scalars, size_t scalar_count, L3L2OrchRegionDesc *out_desc, L3L2OrchCommValidationError *out_error
) {
    if (out_error != nullptr) {
        *out_error = L3L2OrchCommValidationError::OK;
    }
    if (scalars == nullptr || out_desc == nullptr) {
        if (out_error != nullptr) {
            *out_error = L3L2OrchCommValidationError::NULL_POINTER;
        }
        return false;
    }
    if (scalar_count < L3L2_ORCH_REGION_DESC_SCALAR_COUNT) {
        if (out_error != nullptr) {
            *out_error = L3L2OrchCommValidationError::BAD_SCALAR_COUNT;
        }
        return false;
    }
    *out_desc = L3L2OrchRegionDesc{
        scalars[0], scalars[1], scalars[2], scalars[3], scalars[4], scalars[5],
    };
    L3L2OrchCommValidationError error = l3_l2_orch_comm_validate_desc(*out_desc);
    if (out_error != nullptr) {
        *out_error = error;
    }
    return error == L3L2OrchCommValidationError::OK;
}

static inline bool l3_l2_orch_comm_valid_notify_op(L3L2OrchNotifyOp op) {
    return op == L3L2OrchNotifyOp::Set || op == L3L2OrchNotifyOp::Add;
}

static inline bool l3_l2_orch_comm_valid_wait_cmp(L3L2OrchWaitCmp cmp) {
    return cmp == L3L2OrchWaitCmp::EQ || cmp == L3L2OrchWaitCmp::NE || cmp == L3L2OrchWaitCmp::GT ||
           cmp == L3L2OrchWaitCmp::GE || cmp == L3L2OrchWaitCmp::LT || cmp == L3L2OrchWaitCmp::LE;
}

static inline bool l3_l2_orch_comm_compare_counter(int32_t observed, int32_t cmp_value, L3L2OrchWaitCmp cmp) {
    switch (cmp) {
    case L3L2OrchWaitCmp::EQ:
        return observed == cmp_value;
    case L3L2OrchWaitCmp::NE:
        return observed != cmp_value;
    case L3L2OrchWaitCmp::GT:
        return observed > cmp_value;
    case L3L2OrchWaitCmp::GE:
        return observed >= cmp_value;
    case L3L2OrchWaitCmp::LT:
        return observed < cmp_value;
    case L3L2OrchWaitCmp::LE:
        return observed <= cmp_value;
    default:
        return false;
    }
}

static inline L3L2OrchCommValidationError
l3_l2_orch_comm_validate_counter_addr(const L3L2OrchRegionDesc &desc, uint64_t counter_addr) {
    // Address validity is 4-byte; wrapper protocols must keep different
    // counter writers off the same cache line.
    if (!l3_l2_orch_comm_is_aligned(counter_addr, L3L2_ORCH_COMM_COUNTER_BYTES)) {
        return L3L2OrchCommValidationError::BAD_COUNTER_RANGE;
    }
    if (l3_l2_orch_comm_validate_counter_range(desc) != L3L2OrchCommValidationError::OK ||
        l3_l2_orch_comm_add_overflows(counter_addr, L3L2_ORCH_COMM_COUNTER_BYTES) ||
        l3_l2_orch_comm_add_overflows(desc.counter_base, desc.counter_bytes)) {
        return L3L2OrchCommValidationError::BAD_COUNTER_RANGE;
    }
    if (counter_addr < desc.counter_base ||
        counter_addr + L3L2_ORCH_COMM_COUNTER_BYTES > desc.counter_base + desc.counter_bytes) {
        return L3L2OrchCommValidationError::OUT_OF_BOUNDS;
    }
    return L3L2OrchCommValidationError::OK;
}

#endif  // SRC_COMMON_PLATFORM_INCLUDE_COMMON_L3_L2_ORCH_COMM_H_
