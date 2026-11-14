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

#include "async_kernel_api.h"
#include "tensor.h"

namespace {
inline __aicore__ int32_t read_word(__gm__ int32_t *address) {
#ifdef PTO_CPUSTUB_HPP
    return __atomic_load_n(address, __ATOMIC_ACQUIRE);
#else
    return static_cast<int32_t>(__builtin_cce_ld_dev(reinterpret_cast<__gm__ uint32_t *>(address), 0));
#endif
}
inline __aicore__ void write_word(__gm__ int32_t *address, int32_t value) {
#ifdef PTO_CPUSTUB_HPP
    __atomic_store_n(address, value, __ATOMIC_RELEASE);
#else
    *address = value;
    dcci(address, cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    dsb((mem_dsb_t)0);
#endif
}
}  // namespace
extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    const auto *state_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    const auto *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *state = reinterpret_cast<__gm__ int32_t *>(state_tensor->buffer.addr) + state_tensor->start_offset;
    auto *out = reinterpret_cast<__gm__ int32_t *>(out_tensor->buffer.addr) + out_tensor->start_offset;
    const int32_t role = static_cast<int32_t>(args[2]);
    const int32_t index = static_cast<int32_t>(args[3]);
    if (role == 0 || role == 2) write_word(state, role == 2 ? 1 : 0);
    else if (role == 1) {
        auto ctx = get_async_ctx(args);
        save_expected_notification_counter(ctx, state, 1);
        write_word(out + index * 32, 1);
    } else write_word(out + index * 32, read_word(state));
}
