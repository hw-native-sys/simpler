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

#include "inner_kernel.h"
#include "runtime_types.h"
#include "scheduler/async_completion.h"

void exercise_unprofiled_memory_helpers() {
    uint32_t counter = 7;
    uint64_t event = 1;
    uint32_t (*volatile read)(volatile uint32_t &, int) = &percounter_gm_read;
    uint64_t (*volatile read_event)(uint64_t *) = &percounter_ccu_event_read;
    void (*volatile clear_event)(uint64_t *) = &percounter_ccu_event_clear;
    CompletionPollResult (*volatile poll)(const CompletionToken &) = &percounter_poll_valid_completion;
    (void)read(counter, __ATOMIC_ACQUIRE);
    (void)read_event(&event);
    clear_event(&event);
    const CompletionToken token{
        reinterpret_cast<uint64_t>(&counter), 7, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0
    };
    (void)poll(token);
}
