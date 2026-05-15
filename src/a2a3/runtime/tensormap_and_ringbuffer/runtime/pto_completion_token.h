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

#ifndef SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_COMPLETION_TOKEN_H_
#define SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_COMPLETION_TOKEN_H_

#include <stdint.h>

#include "aicore_completion_mailbox.h"
#include "pto_runtime_status.h"

// CompletionToken is the runtime-internal POD that backend submit handlers
// produce and the generic register_completion_condition() consumes. It is the
// ABI contract for "this is one completion to wait on" — independent of which
// backend (SDMA, RoCE, notification counter, ...) generated it. Each backend's
// (poll, retire) pair is registered in pto_async_wait.h's ops table, keyed by
// completion_type.
struct CompletionToken {
    uint64_t addr;
    uint32_t expected_value;
    uint32_t engine;
    int32_t completion_type;
    uint64_t backend_cookie;
};

enum class CompletionPollState : uint8_t {
    PENDING = 0,
    READY = 1,
    FAILED = 2,
};

struct CompletionPollResult {
    CompletionPollState state{CompletionPollState::PENDING};
    int32_t error_code{PTO2_ERROR_NONE};
};

#endif  // SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_COMPLETION_TOKEN_H_
