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

#include <cstdint>
#include <pto/pto-inst.hpp>
#include "common/platform_config.h"
#include "intrinsic.h"
#include "scheduler/gm.h"
#include "scheduler/wire_types.h"
#include "tensor.h"

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    // The cube participates in the predicated MIX cohort without issuing a transfer.
    const auto mode = static_cast<uint32_t>(args[5]);
    if (mode == 1) return;
    auto *gate = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *out = reinterpret_cast<__gm__ Tensor *>(args[2]);
    auto *signal = reinterpret_cast<__gm__ uint32_t *>(gate->buffer.addr) + gate->start_offset;
    const auto segment = static_cast<uint32_t>(args[4]) / static_cast<uint32_t>(args[6]);
    auto *sample = reinterpret_cast<__gm__ float *>(out->buffer.addr) + out->start_offset + segment - 32;
    const auto context = load_global_context(args);
    auto *control = reinterpret_cast<__gm__ simpler::percounter::PercounterControl *>(context.async_image_address);
    if (mode == 4) {
        // The AIC has already observed 128 pending groups. This physical AIV releases their counters.
        sample[16] = static_cast<float>(simpler::percounter::percounter_current_context()->urma_sender_id + 1);
        dcci(sample + 16, SINGLE_CACHE_LINE, CACHELINE_OUT);
        dsb(DSB_DDR);
        percounter_gm_store(*signal, uint32_t{1});
        return;
    }
    const uint64_t expected = static_cast<uint64_t>(args[6]);
    auto *image = reinterpret_cast<__gm__ uint8_t *>(control);
    auto *groups = reinterpret_cast<__gm__ simpler::percounter::PercounterAsyncCompletion *>(
        image + control->async_completions_offset
    );
    const uint64_t start = get_sys_cnt();
    const uint64_t timeout = static_cast<uint64_t>(PLATFORM_PROF_SYS_CNT_FREQ) * 2;
    uint64_t registered = 0;
    bool claimed = false;
    do {
        registered = percounter_gm_fetch_add(control->async_registered, uint64_t{0});
        if (registered == expected) {
            // A waiter has reached the live predecessor's completion list and still needs the notifier.
            for (uint32_t i = 0; i < expected && !claimed; ++i) {
                const auto owner = percounter_gm_fetch_add(groups[i].calendar_defer_owner, uint64_t{0});
                claimed = owner != 0 && owner != UINT64_MAX;
            }
        }
    } while (!claimed && get_sys_cnt() - start < timeout);
    sample[0] = static_cast<float>(registered);
    sample[1] = static_cast<float>(percounter_gm_fetch_add(control->async_settled, uint64_t{0}));
    sample[2] = claimed ? 1.0f : 0.0f;
    dcci(sample, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dsb(DSB_DDR);
    if (mode == 2) percounter_gm_store(*signal, uint32_t{1});
}
