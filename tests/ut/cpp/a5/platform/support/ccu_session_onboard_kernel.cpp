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
#include "platform_comm/ccu/ccu_device.h"

extern "C" __global__[aicore] void ccu_session_onboard(
    uint64_t context_address, uint64_t context_bytes, uint64_t window_address, uint64_t result_address, uint64_t rounds,
    uint64_t rank
) {
    auto *result = reinterpret_cast<__gm__ uint64_t *>(result_address);
    __gm__ CcuDeviceContext *ctx = nullptr;
    result[0] = 1;
    result[1] = 0;
    if (ccu_get_device_context(context_address, context_bytes, ctx) != CcuStatus::OK) {
        ccu_detail::Sync(result, 128);
        return;
    }
    auto *src = reinterpret_cast<__gm__ float *>(window_address);
    auto *dst = reinterpret_cast<__gm__ float *>(window_address + 8192);
    auto *scratch = reinterpret_cast<__gm__ float *>(window_address + 16384);
    for (uint64_t round = 0; round < rounds; ++round) {
        const uint64_t count = round % 2 ? 1024 : 64;
        const uint64_t offset = 16 + (round % 8) * 16;
        const uint32_t root = (round / 2) % 2;
        for (uint32_t i = 0; i < 2048; ++i) {
            src[i] = static_cast<float>(round * 4096 + rank * 2048 + i);
            dst[i] = -1;
            scratch[i] = -7;
        }
        ccu_detail::Sync(src, 24576);
        CcuPositionalArguments args{};
        args.tensor_count = 3;
        args.scalar_count = 1;
        args.tensors[0] = {window_address + offset * 4, 0, count * 4};
        args.tensors[1] = {window_address + 8192 + offset * 4, 0, count * 4};
        args.tensors[2] = {window_address + 16384, 0, 4096};
        args.scalars[0] = {count, CcuScalarKind::UNSIGNED, 64};
        CcuProgramRef program{};
        CcuRequestRef request{};
        auto status = ccu_resolve_program(ctx, root, program);
        if (status == CcuStatus::OK) status = ccu_reserve_request(ctx, program, request, args);
        if (status == CcuStatus::OK) status = ccu_submit_request(request);
        if (status != CcuStatus::OK) {
            result[0] = 100 + static_cast<uint32_t>(status);
            break;
        }
        CcuPollResult completion{};
        do {
            completion = ccu_poll_request(request);
        } while (completion.state == CcuRequestState::PENDING);
        if (completion.state != CcuRequestState::SUCCEEDED) {
            result[0] = 200 + static_cast<uint32_t>(completion.status);
            break;
        }
        ccu_retire_request(request);
        ccu_detail::Sync(src, 24576);
        bool correct = true;
        for (uint32_t i = 0; i < 2048; ++i) {
            const float expected =
                i >= offset && i < offset + count ? static_cast<float>(round * 4096 + root * 2048 + i) : -1.0f;
            if (dst[i] != expected) {
                result[0] = 300;
            } else if (src[i] != static_cast<float>(round * 4096 + rank * 2048 + i)) {
                result[0] = 301;
            } else if (scratch[i] != -7.0f) {
                result[0] = 302;
            } else {
                continue;
            }
            result[2] = i;
            correct = false;
            break;
        }
        if (!correct) break;
        result[1] = round + 1;
        if (round + 1 == rounds) result[0] = 0;
    }
    ccu_detail::Sync(result, 128);
}
