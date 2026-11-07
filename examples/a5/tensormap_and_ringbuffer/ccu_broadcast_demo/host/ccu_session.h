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

#include <cstddef>
#include <cstdint>
#include <type_traits>

struct CcuPeerInfo {
    uint64_t input;
    uint64_t output;
    uint64_t token;
    uint64_t trigger_status;
};

static_assert(std::is_trivially_copyable_v<CcuPeerInfo> && std::is_standard_layout_v<CcuPeerInfo>);
static_assert(sizeof(CcuPeerInfo) == 32);

extern "C" {
const char *ccu_demo_last_error();
size_t ccu_demo_root_info_size();
int ccu_demo_get_root_info(int device, void *root_info, size_t size);
int ccu_demo_open(
    int device, uint32_t rank, const void *root_info, size_t root_info_size, uint32_t count, uint32_t iteration,
    void **session, CcuPeerInfo *peer
);
int ccu_demo_prepare(
    void *session, const CcuPeerInfo *peers, size_t peer_count, uint32_t root, uint64_t *gate_va, uint32_t *gate_mask
);
int ccu_demo_wait_and_verify(void *session, uint32_t root, int timeout_ms);
int ccu_demo_close(void *session);
}
