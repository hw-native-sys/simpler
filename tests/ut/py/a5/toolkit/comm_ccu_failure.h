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
#include "ccu_domain_backend_hcomm.cpp"

struct CcuTestStream {
    aclrtStreamStatus status{ACL_STREAM_STATUS_COMPLETE};
    aclError abort_error{};
    aclError query_error{};
    int aborts{};
    int queries{};
    int not_ready_query{};
};
static std::unordered_map<aclrtStream, std::unique_ptr<CcuTestStream>> ccu_test_streams;

extern "C" aclError aclrtCreateStream(aclrtStream *out) {
    auto stream = std::make_unique<CcuTestStream>();
    *out = stream.get();
    ccu_test_streams.emplace(*out, std::move(stream));
    return ACL_SUCCESS;
}
extern "C" aclError aclrtDestroyStream(aclrtStream stream) {
    assert(ccu_test_streams.erase(stream) == 1);
    return ACL_SUCCESS;
}
extern "C" aclError aclrtStreamAbort(aclrtStream stream) {
    auto &state = *ccu_test_streams.at(stream);
    ++state.aborts;
    return state.abort_error;
}
extern "C" aclError aclrtStreamQuery(aclrtStream stream, aclrtStreamStatus *status) {
    auto &state = *ccu_test_streams.at(stream);
    ++state.queries;
    *status = state.queries == state.not_ready_query ? ACL_STREAM_STATUS_NOT_READY : state.status;
    return state.query_error;
}

extern "C" CcuResult HcommCcuGetMemToken(uint64_t, uint64_t, uint64_t *token) {
    *token = 1;
    return CCU_SUCCESS;
}
extern "C" HcclResult
HcclThreadAcquireWithStream(HcclComm, CommEngine engine, aclrtStream stream, uint32_t, ThreadHandle *thread) {
    assert(engine == COMM_ENGINE_CCU);
    *thread = reinterpret_cast<uint64_t>(stream);
    return HCCL_SUCCESS;
}
extern "C" CcuResult HcommCcuKernelLaunch(ThreadHandle thread, CcuKernelHandle, const void *, uint32_t) {
    assert(ccu_test_streams.count(reinterpret_cast<aclrtStream>(thread)));
    return CCU_SUCCESS;
}
extern "C" aclError aclrtLaunchKernelWithHostArgs(
    aclrtFuncHandle, uint32_t, aclrtStream, aclrtLaunchKernelCfg *, void *args, size_t, aclrtPlaceHolderInfo *, size_t
) {
    const auto *words = static_cast<const uint64_t *>(args);
    *reinterpret_cast<uint64_t *>(words[2]) = 0;
    return ACL_SUCCESS;
}
extern "C" aclError aclrtSynchronizeStreamWithTimeout(aclrtStream, int32_t) { return ACL_SUCCESS; }

static aclrtStream
add_ccu_abort_registration(CommHandle h, uint64_t id, uint32_t mission_count = 1, bool launch = false) {
    using namespace simpler::ccu;
    auto backend = std::make_unique<HcommDomainResourceBackend>();
    backend->stream_pool = &h->ccu_stream_pool;
    if (!launch) {
        for (uint32_t i = 0; i < mission_count; ++i)
            backend->streams.push_back(h->ccu_stream_pool.acquire());
    }
    assert(aclrtCreateStream(&backend->trigger_stream) == ACL_SUCCESS);
    std::vector<HcommPreparedMission> missions;
    if (launch) {
        for (uint32_t i = 0; i < mission_count; ++i) {
            HcommPreparedMission mission{};
            mission.ccu_id = i;
            mission.program_handle = 101 + i;
            mission.ready_gate_address = 0x1000 + i * 64;
            mission.trigger_gate_address = 0x2000 + i * 64;
            mission.done_gate_address = 0x3000 + i * 64;
            mission.parameter_size = 8;
            mission.parameter_alignment = 64;
            missions.push_back(mission);
        }
    }
    assert(backend->prepare(missions, 41) != 0);
    const auto trigger = backend->trigger_stream;
    auto registration = std::make_unique<CcuCallableRegistration>();
    registration->session = backend.get();
    registration->owner = std::make_unique<DomainCcuOwner>(std::move(backend));
    registration->prepared_allocations.emplace(41, 0);
    if (launch) {
        registration->owner->set_registered_kernels({{1, 0, 101}, {2, 1, 102}});
        assert(registration->session->begin(*registration->owner, missions, 41, 0) != 0);
    }
    h->ccu_registrations.emplace(id, std::move(registration));
    return trigger;
}

static void test_ccu_abort_retry(const std::string &failure) {
    CommHandle_ h{};
    char folder[] = "/tmp/simpler-ccu-retry-XXXXXX";
    assert(mkdtemp(folder));
    h.rootinfo_path = std::string(folder) + "/root";
    if (failure == "retry_pool_error" || failure == "retry_pool_pending") {
        const bool pending = failure == "retry_pool_pending";
        const auto trigger = add_ccu_abort_registration(&h, 1, 2, pending);
        auto &backend = *static_cast<simpler::ccu::HcommDomainResourceBackend *>(h.ccu_registrations.at(1)->session);
        const auto retained_stream = backend.streams[1].stream;
        auto &retained_state = *ccu_test_streams.at(retained_stream);
        if (pending) retained_state.not_ready_query = 2;
        else retained_state.query_error = ACL_ERROR_FAILURE;
        assert(comm_control_ccu_run(&h, 1, 4, 41) == (pending ? 1 : COMM_RECLAMATION_UNCONFIRMED));
        const auto reused = h.ccu_stream_pool.acquire();
        assert(reused.handle == 1);
        auto &reused_state = *ccu_test_streams.at(reused.stream);
        reused_state.status = ACL_STREAM_STATUS_NOT_READY;
        const int reused_queries = reused_state.queries;
        const int retained_queries = retained_state.queries;
        retained_state.query_error = ACL_SUCCESS;
        if (pending) {
            assert(retained_state.aborts == 1 && retained_queries == 2);
            assert(comm_control_ccu_run(&h, 1, 7, 41) == COMM_RECLAMATION_UNCONFIRMED);
            assert(mappings.size() == 5);
            assert(comm_control_ccu_run(&h, 1, 4, 41) == 0);
            assert(retained_state.aborts == 1 && retained_state.queries == 3);
            assert(ccu_test_streams.at(trigger)->aborts == 1 && ccu_test_streams.at(trigger)->queries == 1);
            assert(reused_state.aborts == 1 && reused_state.queries == reused_queries);
            assert(!h.cleanup_failed && backend.drained());
            reused_state.status = ACL_STREAM_STATUS_COMPLETE;
            assert(h.ccu_stream_pool.release(reused.handle) == simpler::ccu::MissionStreamReleaseResult::RELEASED);
            assert(comm_control_ccu_run(&h, 1, 7, 41) == 0);
            assert(comm_release_ccu_registration(&h, 1) == 0);
            assert(h.ccu_stream_pool.close());
            assert(mappings.empty() && ccu_test_streams.empty());
            std::filesystem::remove_all(folder);
            return;
        }
        assert(comm_control_ccu_run(&h, 1, 4, 41) == COMM_RECLAMATION_UNCONFIRMED);
        assert(comm_control_ccu_run(&h, 1, 7, 41) == COMM_RECLAMATION_UNCONFIRMED);
        assert(reused_state.aborts == 0 && reused_state.queries == reused_queries);
        assert(retained_state.queries == retained_queries);
        assert(h.cleanup_failed && !backend.drained());
        assert(mappings.size() == 3);
        std::filesystem::remove_all(folder);
        return;
    }
    const auto first = add_ccu_abort_registration(&h, 1);
    const auto second = add_ccu_abort_registration(&h, 2);
    auto &first_state = *ccu_test_streams.at(first);
    auto &second_state = *ccu_test_streams.at(second);
    h.ccu_run_sets[9].generation = 41;
    h.ccu_run_sets[9].registrations = {1, 2};
    if (failure == "retry_pending") {
        second_state.status = ACL_STREAM_STATUS_NOT_READY;
        assert(comm_control_ccu_run_set(&h, 9, 4) == 1);
        assert(!h.cleanup_failed && h.ccu_admission_closed);
        assert(comm_destroy(&h) == COMM_RECLAMATION_UNCONFIRMED);
        assert(!h.cleanup_failed);
        assert(comm_control_ccu_run_set(&h, 9, 7) == COMM_RECLAMATION_UNCONFIRMED);
        assert(comm_control_ccu_run(&h, 2, 7, 41) == COMM_RECLAMATION_UNCONFIRMED);
        assert(mappings.size() == 6 && !h.cleanup_failed);

        // A completed registration has already returned its pool lease.
        const auto reused = h.ccu_stream_pool.acquire();
        assert(reused.handle == 1);
        auto &reused_state = *ccu_test_streams.at(reused.stream);
        reused_state.status = ACL_STREAM_STATUS_NOT_READY;
        const int reused_queries = reused_state.queries;
        first_state.query_error = ACL_ERROR_FAILURE;
        second_state.status = ACL_STREAM_STATUS_COMPLETE;
        assert(comm_control_ccu_run_set(&h, 9, 4) == 0);
        assert(!h.cleanup_failed);
        assert(comm_control_ccu_run_set(&h, 9, 4) == 0);
        assert(first_state.aborts == 1 && first_state.queries == 1);
        assert(second_state.aborts == 1 && second_state.queries == 2);
        assert(reused_state.aborts == 0 && reused_state.queries == reused_queries);
        assert(h.ccu_registrations.at(1)->owner->drained());
        assert(comm_control_ccu_run(&h, 1, 1, 42) < 0);

        reused_state.status = ACL_STREAM_STATUS_COMPLETE;
        assert(h.ccu_stream_pool.release(reused.handle) == simpler::ccu::MissionStreamReleaseResult::RELEASED);
        assert(comm_control_ccu_run_set(&h, 9, 7) == 0);
        assert(comm_release_ccu_registration(&h, 1) == 0);
        assert(comm_release_ccu_registration(&h, 2) == 0);
        assert(h.ccu_stream_pool.close());
        assert(ccu_test_streams.empty() && mappings.empty());
        std::filesystem::remove_all(folder);
        return;
    }
    if (failure == "retry_abort_error") first_state.abort_error = ACL_ERROR_FAILURE;
    else if (failure == "retry_query_error") first_state.query_error = ACL_ERROR_FAILURE;
    else {
        assert(failure == "retry_unknown");
        first_state.status = ACL_STREAM_STATUS_RESERVED;
    }
    second_state.status = ACL_STREAM_STATUS_NOT_READY;
    assert(comm_control_ccu_run_set(&h, 9, 4) == COMM_RECLAMATION_UNCONFIRMED);
    assert(h.cleanup_failed && h.ccu_admission_closed);
    first_state.abort_error = first_state.query_error = ACL_SUCCESS;
    first_state.status = second_state.status = ACL_STREAM_STATUS_COMPLETE;
    assert(comm_control_ccu_run_set(&h, 9, 4) == COMM_RECLAMATION_UNCONFIRMED);
    assert(h.cleanup_failed);
    assert(first_state.aborts == 1 && first_state.queries == 1);
    assert(second_state.aborts == 1 && second_state.queries == 2);
    assert(!h.ccu_registrations.at(1)->owner->drained());
    assert(comm_destroy(&h) == COMM_RECLAMATION_UNCONFIRMED);
    assert(!h.ccu_stream_pool.close());
    assert(!ccu_test_streams.empty());
    std::filesystem::remove_all(folder);
}

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
