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
#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif
#ifdef MEMORY_BASE
#undef MEMORY_BASE
#endif
#ifndef REGISTER_BASE
#define REGISTER_BASE
#endif
#include <pto/pto-inst.hpp>
#include "backend/urma/urma_completion_kernel.h"
#include "platform_comm/comm_context.h"
#include "tensor.h"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *host = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *device = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *host_out = reinterpret_cast<__gm__ Tensor *>(args[2]);
    auto *device_out = reinterpret_cast<__gm__ Tensor *>(args[3]);
    auto *status = reinterpret_cast<__gm__ Tensor *>(args[4]);
    auto *domain = reinterpret_cast<__gm__ CommContext *>(args[5]);
    auto *code = reinterpret_cast<__gm__ int32_t *>(status->buffer.addr) + status->start_offset;
    auto *host_workspace = get_dma_workspace(args, DMA_WORKSPACE_URMA);
    auto *device_workspace =
        domain ? reinterpret_cast<__gm__ uint8_t *>(comm_workspace<CommWorkspaceKind::URMA>(*domain).addr) : nullptr;
    *code = 1;
    if (host_workspace && device_workspace && host_workspace != device_workspace && domain->rankNum == 2 &&
        domain->rankId < 2) {
        const uint32_t peer = 1 - domain->rankId;
        using HostRow = pto::GlobalTensor<uint8_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>>;
        using DeviceRow =
            pto::GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16384>, pto::Stride<16384, 16384, 16384, 16384, 1>>;
        const uint64_t host_offset = host->buffer.addr + host->start_offset * 2 -
                                     pto2::urma_backend::peer_mr_base_addr(host_workspace, domain->rankId);
        auto *host_src = pto2::urma_backend::peer_mr_ptr<uint8_t>(host_workspace, peer, host_offset);
        auto *host_dst = reinterpret_cast<__gm__ uint8_t *>(host_out->buffer.addr) + host_out->start_offset * 2;
        const uint64_t device_offset =
            device->buffer.addr + device->start_offset * sizeof(float) - domain->windowsIn[domain->rankId];
        auto *device_src = pto2::urma_backend::peer_mr_ptr<float>(device_workspace, peer, device_offset);
        auto *device_dst = reinterpret_cast<__gm__ float *>(device_out->buffer.addr) + device_out->start_offset;
        AsyncCtx context = get_async_ctx(args);
        bool ok = true;
        if (domain->rankId == 0) {
            ok = send_request_entry(context, UrmaTget(HostRow(host_dst), HostRow(host_src), host_workspace, peer));
            ok = send_request_entry(
                     context, UrmaTget(DeviceRow(device_dst), DeviceRow(device_src), device_workspace, peer)
                 ) &&
                 ok;
        } else {
            ok = send_request_entry(
                context, UrmaTget(DeviceRow(device_dst), DeviceRow(device_src), device_workspace, peer)
            );
            ok =
                send_request_entry(context, UrmaTget(HostRow(host_dst), HostRow(host_src), host_workspace, peer)) && ok;
        }
        *code = ok ? 0 : 2;
    }
    dcci(code, cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    dsb(DSB_DDR);
}
