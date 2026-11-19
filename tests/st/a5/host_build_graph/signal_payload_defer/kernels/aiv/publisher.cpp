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
#include "../comm.h"
extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    auto *comm = reinterpret_cast<__gm__ CommContext *>(args[5]);
    const uint32_t epoch = static_cast<uint32_t>(args[6]);
    const auto *tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    const uint32_t count = tensor->shapes[1];
    for (uint32_t peer = 0; peer < comm->rankNum; ++peer) {
        if (peer == comm->rankId) continue;
        auto *x = reinterpret_cast<__gm__ float *>(payload_case::remote(comm, payload_case::words(args, 1), peer));
        auto *scale = reinterpret_cast<__gm__ float *>(payload_case::remote(comm, payload_case::words(args, 2), peer));
        auto *weight = reinterpret_cast<__gm__ float *>(payload_case::remote(comm, payload_case::words(args, 3), peer));
        auto *route = payload_case::remote(comm, payload_case::words(args, 4), peer);
        for (uint32_t index = 0; index < count; ++index) {
            const uint32_t at = comm->rankId * count + index;
            x[at] = static_cast<float>((comm->rankId + 1) * 10000 + epoch * 100 + peer * 7 + index);
            scale[at] = static_cast<float>(comm->rankId + 1) / 2.0F + static_cast<float>(epoch) / 8.0F +
                        static_cast<float>(index % 4) / 4.0F;
            weight[at] =
                -static_cast<float>((comm->rankId + 1) * 100 + epoch + peer) - static_cast<float>(index) / 16.0F;
            route[at] = (comm->rankId + 1) * 100000 + epoch * 1000 + peer * 13 + index;
        }
        payload_case::publish(x);
        payload_case::notify(comm, payload_case::words(args, 0), peer);
    }
}
