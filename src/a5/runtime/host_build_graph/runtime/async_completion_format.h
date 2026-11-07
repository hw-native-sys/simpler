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

#include "intrinsic.h"
#include "aicore_completion_mailbox_types.h"

// Tokens and sealed entries share the same type/address/threshold contract.
// Format validation does not establish the address's lifetime or ownership.
template <typename Token>
inline __aicore__ bool async_completion_format_valid(const Token &token) {
    if (token.addr == 0 || token.engine != COMPLETION_ENGINE_SDMA) return false;
    if (token.completion_type == COMPLETION_TYPE_COUNTER)
        return token.addr % alignof(int32_t) == 0 && token.expected_value <= INT32_MAX && token.backend_cookie == 0;
    if (token.completion_type == COMPLETION_TYPE_SDMA_EVENT_RECORD)
        return token.addr % alignof(uint64_t) == 0 && token.backend_cookie != 0 && token.expected_value == 0;
    return false;
}
