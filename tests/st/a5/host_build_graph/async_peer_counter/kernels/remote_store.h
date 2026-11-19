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

#include "device.h"

namespace peer_counter {

template <typename Tile>
inline __aicore__ void remote_store(Tile &tile, __gm__ int32_t *source, __gm__ int32_t *destination) {
    using Global = pto::GlobalTensor<
        int32_t, pto::Shape<1, 1, 1, 1, kDataWords>, pto::Stride<kDataWords, kDataWords, kDataWords, kDataWords, 1>>;
    Global input(source), output(destination);
    pto::TLOAD(tile, input);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    pto::TSTORE(output, tile);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
#ifdef PTO_CPUSTUB_HPP
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#else
    dcci(destination, SINGLE_CACHE_LINE);
    dcci(destination + 16, SINGLE_CACHE_LINE);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
#endif
}
}  // namespace peer_counter
