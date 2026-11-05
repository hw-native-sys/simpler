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
 * The runtime provisions a per-device scratch workspace and injects the device
 * addresses into every core's GlobalContext, so kernels obtain them via
 * get_dma_workspace(args, kind) without threading them as user args. Selecting
 * an engine is the kernel's decision, not the runtime's: every provisioned
 * address is injected, and a kind the runtime did not provision reads back 0.
 * One slot per engine, indexed by this enum; DMA_WORKSPACE_KIND_COUNT sizes the
 * injected array end to end (InitArgs, the resident AICPU config, and
 * GlobalContext).
 *
 * simpler-owned and deliberately independent of pto-isa's comm::DmaEngine —
 * the kernel maps this kind to the pto-isa engine tag at the call boundary.
 * Shared by host (provisioning + InitArgs) and device (scheduler + kernels),
 * so it carries no dependencies beyond the enum itself.
 *
 * A Worker provisions an engine only when it both requested that engine and the
 * device supports it (dma_workspace_supported_mask()), so an unrequested or
 * unsupported kind stays 0. SDMA is the only engine anyone declines today, and
 * the only one worth declining: provisioning it is inseparable from holding 48
 * CP-process STARS streams whose post-fault release CANN bounds neither with a
 * completion fence nor with a portable timeout — see
 * docs/investigations/2026-07-a2a3-sdma-fault-teardown.md. A Worker declines it
 * by leaving `enable_sdma` off, which is the default.
 *
 * Current support matrix: the a2a3 onboard PTO-SDMA provider is compiled into
 * every a2a3 onboard host_runtime, so both runtimes there provision SDMA on
 * request — the gate is the platform, not the runtime. Both schedulers copy the
 * provisioned addresses into every core's GlobalContext for get_dma_workspace.
 * A5 TMR publishes the URMA slot when L3 maps a shared Host backing; it is
 * separate from Device communication-domain workspaces and revoked before release.
 * A2/A3 simulation provisions inert scratch for the same address-injection contract;
 * it creates no SDMA streams and does not model prefetch performance. A5 onboard and
 * builds without the a2a3 PTO-SDMA provider reject Worker-level SDMA requests.
 */

#ifndef PLATFORM_COMMON_DMA_WORKSPACE_H_
#define PLATFORM_COMMON_DMA_WORKSPACE_H_

enum DmaWorkspaceKind {
    DMA_WORKSPACE_SDMA = 0,  // PTO-ISA async-SDMA (a2a3): TPREFETCH_ASYNC / TGET_ASYNC / TPUT_ASYNC
    DMA_WORKSPACE_URMA = 1,  // A5 TMR shared Host backing, prepared by L3 map_view
    DMA_WORKSPACE_KIND_COUNT = 2,
};

#endif  // PLATFORM_COMMON_DMA_WORKSPACE_H_
