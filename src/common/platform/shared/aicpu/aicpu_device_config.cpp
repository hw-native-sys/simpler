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
#include "aicpu/aicpu_device_config.h"

#include "common/dma_workspace.h"

namespace {
// Latched by simpler_aicpu_init and updated after first-use DMA provisioning;
// survives per-task launches because the inner SO stays dlopen'd.
int g_orch_device_id = 0;
int g_scheduler_timeout_ms = 0;
unsigned long long g_dma_workspace_addr[DMA_WORKSPACE_KIND_COUNT] = {0};
}  // namespace

void set_orch_device_id(int device_id) { g_orch_device_id = device_id; }

int get_orch_device_id() { return g_orch_device_id; }

void set_scheduler_timeout_ms(int timeout_ms) { g_scheduler_timeout_ms = timeout_ms; }

int get_scheduler_timeout_ms() { return g_scheduler_timeout_ms; }

void set_dma_workspace_addr(int kind, unsigned long long addr) {
    if (kind < 0 || kind >= DMA_WORKSPACE_KIND_COUNT) return;
    g_dma_workspace_addr[kind] = addr;
}

unsigned long long get_dma_workspace_addr(int kind) {
    if (kind < 0 || kind >= DMA_WORKSPACE_KIND_COUNT) return 0;
    return g_dma_workspace_addr[kind];
}
