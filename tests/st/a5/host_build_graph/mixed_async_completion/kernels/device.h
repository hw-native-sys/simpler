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
#include <cstdint>
#include <pto/pto-inst.hpp>
#include "async_kernel_api.h"
#include "common/platform_config.h"
#include "platform_comm/comm_context.h"
#include "tensor.h"

namespace mixed_async {
constexpr uint32_t kLine = 32;
constexpr uint32_t kMeta = 16;

template <typename T>
inline __aicore__ __gm__ T *data(__gm__ int64_t *args, uint32_t argument) {
    const auto *tensor = reinterpret_cast<__gm__ Tensor *>(args[argument]);
    return reinterpret_cast<__gm__ T *>(tensor->buffer.addr) + tensor->start_offset;
}
inline __aicore__ uint32_t size(__gm__ int64_t *args, uint32_t argument) {
    return reinterpret_cast<__gm__ Tensor *>(args[argument])->shapes[0];
}
inline __aicore__ void publish(__gm__ void *address, uint32_t bytes) {
    auto *base = reinterpret_cast<__gm__ uint8_t *>(address);
    for (uint32_t offset = 0; offset < bytes; offset += 64)
        dcci(base + offset, cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    dsb((mem_dsb_t)0);
}
template <typename T>
inline __aicore__ __gm__ T *remote(__gm__ CommContext *comm, __gm__ T *local, uint32_t peer) {
    return reinterpret_cast<__gm__ T *>(
        comm->windowsIn[peer] + reinterpret_cast<uint64_t>(local) - comm->windowsIn[comm->rankId]
    );
}
inline __aicore__ bool active(uint32_t execution, uint32_t mask) { return (mask & (1u << (execution % 3))) != 0; }
inline __aicore__ int32_t notice(uint32_t tag, uint32_t peer, uint32_t execution, uint32_t index) {
    return static_cast<int32_t>(tag * 100000 + peer * 1000 + execution * 32 + index + 1);
}
inline __aicore__ uint64_t read_done(uint64_t address) {
    return __builtin_cce_ld_dev(reinterpret_cast<__gm__ uint64_t *>(address), 0);
}
}  // namespace mixed_async
