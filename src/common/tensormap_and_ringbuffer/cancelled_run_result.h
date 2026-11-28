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

// Written after every participating scheduler has stopped, before the run's
// terminal record is published. It describes device work, not RTS health.
struct KernelErrorCleanup {
    uint64_t run_epoch;
    uint64_t task_token;
    int32_t core_error;
    int32_t async_error;
    uint32_t source;
    int32_t core;
    uint32_t expected_cores;
    uint32_t released_cores;
    uint32_t drained_conditions;
    uint32_t complete;
};

template <size_t HeaderBytes>
struct TmrCancelledRunResult {
    uint8_t header[HeaderBytes];
    KernelErrorCleanup retirement;
};
