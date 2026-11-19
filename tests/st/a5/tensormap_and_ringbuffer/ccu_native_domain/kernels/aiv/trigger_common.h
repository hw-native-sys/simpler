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

#include <pto/pto-inst.hpp>
#include "async_kernel_api.h"
#include "platform_comm/comm_context.h"
#include "tensor.h"

inline __aicore__ void trigger_mission(__gm__ int64_t *args, uint32_t offset, uint32_t ccu_id) {
    auto *output = reinterpret_cast<__gm__ Tensor *>(args[offset]);
    auto *comm = reinterpret_cast<__gm__ CommContext *>(args[offset + 1]);
    const uint64_t value = args[offset + 2];
    AsyncCtx ctx = get_async_ctx(args);
    const uint64_t mode = args[offset + 3];
    if (mode == 2) {
        // Register the completion but deliberately leave the mission waiting for trigger.
        const auto *mission = simpler::ccu::ccu_control(simpler::ccu::ccu_workspace(comm), ccu_id);
        if (mission != nullptr) {
            const CompletionToken done{
                mission->done.address, 0, COMPLETION_ENGINE_CCU, COMPLETION_TYPE_CCU, mission->done.mask
            };
            (void)register_completion_condition(ctx, done);
            pto2::detail::defer_flush(ctx);
        }
        return;
    }
    const uint64_t output_index = output->start_offset + (mode == 3 ? 1 : 0);
    uint64_t parameters[]{output->buffer.addr + output_index * sizeof(uint64_t), value, mode != 0};
    (void)submit_ccu_mission(ctx, comm, ccu_id, parameters, sizeof(parameters));
}
