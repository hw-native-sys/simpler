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
#include "../device.h"
extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    using namespace mixed_async;
    auto *comm = reinterpret_cast<__gm__ CommContext *>(args[9]);
    const uint32_t mask = args[11], tag = args[12], role = args[13];
    const uint32_t executions = size(args, 7) / kLine;
    auto *ready = data<int32_t>(args, 3);
    if (role == 0 || role == 2) {
        auto ctx = get_async_ctx(args);
        if (role == 0) {
            for (uint32_t e = 0; e < executions; ++e)
                if (active(e, mask)) save_expected_notification_counter(ctx, data<int32_t>(args, 7) + e * kLine, 1);
        } else {
            for (uint32_t peer = 0; peer < comm->rankNum; ++peer)
                if (peer != comm->rankId) save_expected_notification_counter(ctx, ready + peer * kLine, 1);
        }
        return;
    }
    if (role == 1) {
        for (uint32_t peer = 0; peer < comm->rankNum; ++peer)
            if (peer != comm->rankId)
                send_notification(remote(comm, ready, peer) + comm->rankId * kLine, 1, pto::comm::NotifyOp::Set);
        return;
    }
    // All registrations precede readiness. The delay gives CHECK a pending
    // window without making publication depend on the waiting tasks' completion.
    const uint64_t start = get_sys_cnt();
    while (get_sys_cnt() - start < PLATFORM_PROF_SYS_CNT_FREQ / 50) {}
    for (uint32_t peer = 0; peer < comm->rankNum; ++peer) {
        if (peer == comm->rankId) continue;
        auto *payload = remote(comm, data<int32_t>(args, 2), peer);
        auto *signals = remote(comm, data<int32_t>(args, 1), peer);
        for (uint32_t e = 0; e < executions; ++e) {
            if (!active(e, mask)) continue;
            auto *row = payload + (e * comm->rankNum + comm->rankId) * kLine;
            for (uint32_t i = 0; i < kLine; ++i)
                row[i] = notice(tag, comm->rankId, e, i);
            publish(row, kLine * sizeof(int32_t));
            send_notification(signals + (e * comm->rankNum + comm->rankId) * kLine, 1, pto::comm::NotifyOp::Set);
        }
    }
}
