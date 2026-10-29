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

#include "common/strace.h"
#include "host_log.h"

extern "C" __attribute__((visibility("default"))) int test_strace_bind(SimplerHostLogState *state) {
    return HostLogger::get_instance().bind_state(state);
}

extern "C" __attribute__((visibility("default"))) bool test_strace_touch() {
    STRACE_CONTEXT(71, 83, 2);
    STRACE("unloaded_consumer");
    auto *state = simpler::strace::strace_state();
    return state != nullptr && state->inv == 71 && state->hid == 83 && state->depth == 3;
}

extern "C" __attribute__((visibility("default"))) uint64_t test_strace_allocate_inv() { return STRACE_ALLOC_INV(); }

extern "C" __attribute__((visibility("default"))) bool test_strace_context(uint64_t inv, uint64_t hid, int depth) {
    auto *state = simpler::strace::strace_state();
    return state != nullptr && state->inv == inv && state->hid == hid && state->depth == depth;
}
