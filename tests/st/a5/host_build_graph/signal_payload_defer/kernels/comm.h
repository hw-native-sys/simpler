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
#include "payload.h"
#include "platform_comm/comm_context.h"
namespace payload_case {
template <typename T>
inline __aicore__ __gm__ T *remote(__gm__ CommContext *comm, __gm__ T *local, uint32_t peer) {
    return reinterpret_cast<__gm__ T *>(
        comm->windowsIn[peer] + reinterpret_cast<uint64_t>(local) - comm->windowsIn[comm->rankId]
    );
}
inline __aicore__ void notify(__gm__ CommContext *comm, __gm__ uint32_t *signal, uint32_t peer) {
    send_notification(remote(comm, signal, peer) + comm->rankId * 32, 1, pto::comm::NotifyOp::AtomicAdd);
}
}  // namespace payload_case
