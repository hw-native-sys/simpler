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

inline constexpr int32_t MAX_COMPLETIONS_PER_TASK = 64;

#define COMPLETION_ENGINE_SDMA 0u
#define COMPLETION_ENGINE_ROCE 1u
#define COMPLETION_ENGINE_URMA 2u
#define COMPLETION_ENGINE_CCU 3u

#define COMPLETION_TYPE_COUNTER 0
#define COMPLETION_TYPE_SDMA_EVENT_RECORD 1
#define COMPLETION_TYPE_CCU 4

// Canonical completion condition shared by kernel writers and runtime readers.
struct DeferredCompletionEntry {
    uint64_t addr;
    uint64_t backend_cookie;
    uint32_t expected_value;
    uint32_t engine;
    int32_t completion_type;
    uint32_t _pad;
};

static_assert(sizeof(DeferredCompletionEntry) == 32, "DeferredCompletionEntry layout drift");
