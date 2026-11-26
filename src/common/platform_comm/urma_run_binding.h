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

#include <stdint.h>

// The AIV queue reader returns one of these values in its 32-bit result word.
enum class UrmaQueueState : uint32_t { PENDING = 0, READY = 1, FAILED = 2 };

// Independent queue claims occupy separate A5 GM contention units.
struct alignas(128) UrmaRowOwner {
    uint64_t value;
};
static_assert(sizeof(UrmaRowOwner) == 128);

// The Run retains this workspace and one UrmaRowOwner per physical queue row.
struct UrmaRunBinding {
    uint64_t workspace;
    uint64_t workspace_bytes;
    uint64_t row_owner;
    uint32_t local_rank;
    uint32_t rows_per_peer;
    uint32_t rank_count;
    uint32_t jetties_per_core;
};

// Device address of the contiguous bindings published with this scheduler image.
struct UrmaRunBindings {
    uint64_t address;
    uint32_t count;
};
