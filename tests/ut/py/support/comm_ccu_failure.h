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

static void test_ccu_communicator_modes(const char *module_path) {
    using namespace simpler::ccu;
    std::ifstream input(module_path, std::ios::binary);
    assert(input.good());
    const std::vector<uint8_t> module((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    ExpectedBuildIdentity identity{};
    identity.sdk.fill(0x11);
    identity.pto_isa.fill(0x22);
    const auto base = reinterpret_cast<HcclComm>(1);
    HcommMissionStreamPool stream_pool;
    std::vector<std::shared_ptr<HcommDomainChannels>> domains;
    for (const auto mode : {COMM_CCU_MODE_SCHED, COMM_CCU_MODE_MS, COMM_CCU_MODE_SCHED}) {
        const uint32_t expansion = mode == COMM_CCU_MODE_MS ? 5 : 6;
        const std::string name = mode == COMM_CCU_MODE_MS ? "ms" : "sched";
        std::shared_ptr<HcommDomainChannels> channels;
        fail_ccu_create = true;
        try {
            create_hcomm_domain_channels(base, 41, {0, 1}, 0, channels, mode);
            assert(false);
        } catch (const std::runtime_error &error) {
            const std::string message = error.what();
            assert(message.find("mode=" + name) != std::string::npos);
            assert(message.find("failed with code " + std::to_string(HCCL_E_PARA)) != std::string::npos);
        }
        assert((ccu_communicator_modes.back() == std::make_pair(2u, expansion)));
        assert(channels && channels->mode == mode && channels->close());
        domains.push_back(channels);
    }
    // Registration uses its domain even when other modes were configured later.
    for (const auto &channels : domains) {
        std::unique_ptr<DomainCcuOwner> owner;
        HcommDomainSession *session{};
        try {
            prepare_hcomm_ccu_callable(
                base, 42, {0, 1}, 0, identity, {module}, "mode_test", owner, session, channels, 0, stream_pool
            );
            assert(false);
        } catch (const std::runtime_error &error) {
            const std::string message = error.what();
            const std::string mode = channels->mode == COMM_CCU_MODE_MS ? "ms" : "sched";
            assert(message.find("local CCU registration) mode=" + mode) != std::string::npos);
            assert(message.find("failed with code " + std::to_string(HCCL_E_PARA)) != std::string::npos);
        }
        const uint32_t expansion = channels->mode == COMM_CCU_MODE_MS ? 5 : 6;
        assert((ccu_communicator_modes.back() == std::make_pair(1u, expansion)));
        assert(!owner && session == nullptr);
        assert(channels->close());
    }
    std::unique_ptr<DomainCcuOwner> owner;
    HcommDomainSession *session{};
    try {
        prepare_hcomm_ccu_callable(
            base, 43, {0, 1}, 0, identity, {module}, "default_mode", owner, session, nullptr, 0, stream_pool
        );
        assert(false);
    } catch (const std::runtime_error &error) {
        const std::string message = error.what();
        assert(message.find("HcclCreateSubCommConfig(CCU) mode=sched") != std::string::npos);
        assert(message.find("failed with code " + std::to_string(HCCL_E_PARA)) != std::string::npos);
    }
    assert((ccu_communicator_modes.back() == std::make_pair(2u, 6u)));
    assert(!owner && session == nullptr);
    assert(live_ccu_communicators.empty());
    fail_ccu_create = false;
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
