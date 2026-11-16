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

#include "hccl/hccl_rank_graph.h"

extern "C" {
int simpler_ccu_test_manifest_variant = 0;
}

extern "C" HcclResult HcclGetCommName(HcclComm, char *name) {
    std::strcpy(name, "fixture_parent");
    return HCCL_SUCCESS;
}

extern "C" HcclResult HcclRankGraphGetLinks(HcclComm comm, uint32_t, uint32_t, uint32_t, CommLink **, uint32_t *) {
    assert(live_ccu_communicators.count(comm) == 1);
    return HCCL_E_INTERNAL;
}

static void test_ccu_channel_rollback(const char *module_path, bool destroy_failure) {
    std::ifstream input(module_path, std::ios::binary);
    assert(input.good());
    const std::vector<uint8_t> module((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    const auto package = pack_ccu_modules({module});
    CommHandle_ h{};
    h.rank = 0;
    h.nranks = 2;
    h.hccl_comm = reinterpret_cast<HcclComm>(1);
    uint8_t sdk[32], isa[32];
    std::memset(sdk, 0x11, sizeof(sdk));
    std::memset(isa, 0x22, sizeof(isa));
    const uint32_t ranks[]{0, 1};
    uint64_t registration = 0;
    fail_ccu_destroy = destroy_failure;
    const int result =
        comm_register_ccu_callable(&h, 8, 41, ranks, 2, 0, package.data(), package.size(), sdk, isa, &registration);
    assert(subcomm_creates == 1 && registration == 0);
    assert(ccu_communicator_name == "ccu_fixture_parent_41");
    if (destroy_failure) {
        assert(result == COMM_RECLAMATION_UNCONFIRMED);
        assert(h.cleanup_failed && h.ccu_registrations.size() == 1);
        auto &owner = h.ccu_registrations.begin()->second->owner;
        assert(owner && !owner->close());
        assert(live_ccu_communicators.size() == 1);
        // Recovery can still reach and release the communicator through its retained owner.
        fail_ccu_destroy = false;
        assert(owner->close());
        assert(live_ccu_communicators.empty());
        assert(owner->close());
    } else {
        assert(result == -1);
        assert(!h.cleanup_failed && h.ccu_registrations.empty());
        assert(live_ccu_communicators.empty());
    }
}
