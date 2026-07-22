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
/**
 * Async-DMA engine workspace kinds.
 *
 * The runtime provisions a per-device scratch workspace for each async-DMA
 * engine it supports and injects the device addresses into every core's
 * GlobalContext, so kernels obtain them via get_dma_workspace(args, kind)
 * without threading them as user args. One slot per engine, indexed by this
 * enum; DMA_WORKSPACE_KIND_COUNT sizes the injected array end to end
 * (InitArgs, the resident AICPU config, and GlobalContext).
 *
 * simpler-owned and deliberately independent of pto-isa's comm::DmaEngine —
 * the kernel maps this kind to the pto-isa engine tag at the call boundary.
 * Shared by host (provisioning + InitArgs) and device (scheduler + kernels),
 * so it carries no dependencies beyond the enum itself.
 *
 * Current support matrix: SDMA is available only on a2a3 onboard with the
 * tensormap_and_ringbuffer runtime. URMA is reserved for the future a5
 * per-domain provider. Host-build-graph, simulation, a5, and builds without
 * the a2a3 PTO-SDMA provider reject non-empty requirements at registration.
 */

#ifndef PLATFORM_COMMON_DMA_WORKSPACE_H_
#define PLATFORM_COMMON_DMA_WORKSPACE_H_

enum DmaWorkspaceKind {
    DMA_WORKSPACE_SDMA = 0,  // PTO-ISA async-SDMA (a2a3): TPREFETCH_ASYNC / TGET_ASYNC / TPUT_ASYNC
    DMA_WORKSPACE_URMA = 1,  // Reserved for the future a5 URMA async engine
    DMA_WORKSPACE_KIND_COUNT = 2,
};

#endif  // PLATFORM_COMMON_DMA_WORKSPACE_H_
