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
 * injected array end to end (InitArgs, the Runtime device image, and
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
 * Current support matrix: a2a3 onboard TMR and HBG provision SDMA on request
 * in both program and kernel modes. Each Runtime snapshots its own addresses;
 * the scheduler publishes them through each core's GlobalContext.
 * URMA is reserved for the future a5 per-domain provider.
 * Program-mode simulation provisions inert scratch for no-op prefetch kernels.
 * Kernel-mode simulation, a5, and onboard builds without the PTO-SDMA provider
 * reject an explicit SDMA request during initialization.
 */

#pragma once

enum DmaWorkspaceKind {
    DMA_WORKSPACE_SDMA = 0,  // PTO-ISA async-SDMA (a2a3): TPREFETCH_ASYNC / TGET_ASYNC / TPUT_ASYNC
    DMA_WORKSPACE_URMA = 1,  // Reserved for the future a5 URMA async engine
    DMA_WORKSPACE_KIND_COUNT = 2,
};
