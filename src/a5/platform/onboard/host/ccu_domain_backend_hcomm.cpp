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

#include "ccu_domain_backend_hcomm.h"

#include <array>
#include <algorithm>
#include <dlfcn.h>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

#include "acl/acl.h"
#include "hccl/hccl_ccu_res.h"
#include "hccl/hccl_channel.h"
#include "hccl/hccl_rank_graph.h"
#include "hccl/hccl_res.h"
#include "hcomm/ccu/ccu_launch.h"
#include "hcomm/ccu/ccu_res.h"

#include "ccu_domain_registration_hcomm.h"
#include "ccu_run_lifecycle_hcomm.h"

extern "C" CcuResult CcuNotifyWait(ChannelHandle channel, uint32_t local_notify_index, uint16_t mask);

namespace simpler::ccu {

bool query_window_token(uint64_t address, uint64_t size, uint64_t &token) {
    return HcommCcuGetMemToken(address, size, &token) == HCCL_SUCCESS;
}
namespace {

[[noreturn]] void fail(const char *operation, int code) {
    throw std::runtime_error(std::string(operation) + " failed with code " + std::to_string(code));
}

void check(int code, const char *operation) {
    if (code != 0) fail(operation, code);
}

CcuResult probe_channel_die(const ChannelHandle *channel) { return CcuNotifyWait(*channel, 0, 1); }

std::string sibling_trigger_binary() {
    Dl_info info{};
    if (::dladdr(reinterpret_cast<const void *>(&prepare_hcomm_ccu_callable), &info) == 0 || info.dli_fname == nullptr)
        throw std::runtime_error("cannot locate the A5 host runtime for the CCU trigger binary");
    std::string path(info.dli_fname);
    const size_t slash = path.find_last_of('/');
    return (slash == std::string::npos ? std::string(".") : path.substr(0, slash)) + "/ccu_event_trigger.o";
}

std::array<uint64_t, SIMPLER_CCU_MAX_RANKS>
acquire_peer_channels(HcclComm communicator, uint32_t rank, uint32_t rank_count) {
    std::vector<HcclChannelDesc> descriptors;
    std::vector<uint32_t> peers;
    for (uint32_t peer = 0; peer < rank_count; ++peer) {
        if (peer == rank) continue;
        CommLink *links = nullptr;
        uint32_t link_count = 0;
        check(HcclRankGraphGetLinks(communicator, 0, rank, peer, &links, &link_count), "HcclRankGraphGetLinks");
        for (uint32_t i = 0; i < link_count; ++i) {
            if (links[i].linkAttr.linkProtocol != COMM_PROTOCOL_UBC_CTP) continue;
            HcclChannelDesc descriptor{};
            check(HcclChannelDescInit(&descriptor, 1), "HcclChannelDescInit");
            descriptor.remoteRank = peer;
            descriptor.notifyNum = 4;
            descriptor.channelProtocol = COMM_PROTOCOL_UBC_CTP;
            descriptor.localEndpoint = links[i].srcEndpointDesc;
            descriptor.remoteEndpoint = links[i].dstEndpointDesc;
            descriptors.push_back(descriptor);
            peers.push_back(peer);
            break;
        }
        if (peers.empty() || peers.back() != peer) throw std::runtime_error("CCU peer has no UBC_CTP link");
    }
    std::vector<ChannelHandle> handles(peers.size());
    check(
        HcclChannelAcquire(communicator, COMM_ENGINE_CCU, descriptors.data(), descriptors.size(), handles.data()),
        "HcclChannelAcquire(all peers)"
    );
    std::array<uint64_t, SIMPLER_CCU_MAX_RANKS> channels{};
    for (size_t i = 0; i < peers.size(); ++i) {
        if (handles[i] == 0) throw std::runtime_error("HcclChannelAcquire returned a null peer channel");
        channels[peers[i]] = handles[i];
    }
    return channels;
}

uint32_t find_channel_die(ChannelHandle channel) {
    int32_t selected = -1;
    const void *arguments[] = {&channel};
    for (uint32_t die = 0; die < 2; ++die) {
        HcommCcuResDescHandle descriptor{};
        check(HcommCcuInsResDescCreate(die, &descriptor), "HcommCcuInsResDescCreate(probe)");
        const int query =
            HcommCcuKernelQueryResReq(reinterpret_cast<const void *>(probe_channel_die), arguments, 1, descriptor);
        check(HcommCcuInsResDescDestroy(descriptor), "HcommCcuInsResDescDestroy(probe)");
        if (query == CCU_SUCCESS) {
            if (selected >= 0) throw std::runtime_error("CCU channel maps to multiple IO dies");
            selected = static_cast<int32_t>(die);
        } else if (query != CCU_E_PARA && query != CCU_E_UNAVAIL) {
            fail("HcommCcuKernelQueryResReq(channel probe)", query);
        }
    }
    if (selected < 0) throw std::runtime_error("CCU channel has no usable IO die");
    return static_cast<uint32_t>(selected);
}

SimplerCcuRegistrationBinding binding_for(
    uint32_t rank, uint32_t rank_count, const std::array<uint64_t, SIMPLER_CCU_MAX_RANKS> &channels, uint32_t ccu_id,
    const std::array<CcuEventHandle, 3> &events
) {
    SimplerCcuRegistrationBinding binding{
        sizeof(SimplerCcuRegistrationBinding), ccu_id, rank, rank_count, {}, events[0], events[1], events[2]
    };
    std::copy(channels.begin(), channels.end(), binding.channels);
    return binding;
}

class HcommDomainResourceBackend final : public HcommDomainSession {
public:
    HcclComm communicator{};
    uint64_t domain_id{};
    CcuInsHandle unassigned_instance{};
    CcuInsHandle staging_instance{};
    bool instance_assigned{false};
    std::array<uint64_t, SIMPLER_CCU_MAX_RANKS> channels{};
    std::vector<HcommMissionStream> streams;
    aclrtBinHandle trigger_binary{};
    aclrtFuncHandle trigger_function{};
    aclrtStream trigger_stream{};

    uint64_t prepare(const std::vector<HcommPreparedMission> &missions, uint64_t generation) override {
        if (generation == 0 || prepared_.count(generation) != 0 || closed_)
            throw std::runtime_error("invalid or duplicate CCU preparation generation");
        allocation_generation_ = generation;
        auto &prepared = prepared_[generation];
        try {
            const uint64_t event_result = allocate(sizeof(uint64_t), 64);
            prepared.event_result = event_result;
            auto &resources = prepared.resources;
            std::vector<SimplerCcuControl> descriptors;
            for (const auto &mission : missions) {
                const auto parameter_address =
                    allocate(std::max<uint64_t>(mission.parameter_size, 1), mission.parameter_alignment);
                const auto parameter_token = token(parameter_address, std::max<uint64_t>(mission.parameter_size, 1));
                SimplerCcuLaunchArguments arguments{parameter_address, parameter_token};
                resources.push_back(
                    {mission.ccu_id,
                     mission.program_handle,
                     mission.stream_handle,
                     arguments,
                     {mission.ready_gate_address, SIMPLER_CCU_EVENT_MASK}}
                );
                SimplerCcuControl descriptor{};
                descriptor.struct_size = sizeof(descriptor);
                descriptor.run_generation = generation;
                descriptor.ccu_id = mission.ccu_id;
                descriptor.parameter_alignment = mission.parameter_alignment;
                descriptor.parameter_addr = parameter_address;
                descriptor.parameter_size = mission.parameter_size;
                descriptor.parameter_token = parameter_token;
                descriptor.ready = {mission.ready_gate_address, SIMPLER_CCU_EVENT_MASK};
                descriptor.trigger = {mission.trigger_gate_address, SIMPLER_CCU_EVENT_MASK};
                descriptor.done = {mission.done_gate_address, SIMPLER_CCU_EVENT_MASK};
                descriptors.push_back(descriptor);
            }
            const uint64_t table = allocate(descriptors.size() * sizeof(descriptors.front()), 64);
            copy_to_device(table, descriptors.data(), descriptors.size() * sizeof(descriptors.front()));
            SimplerCcuWorkspace workspace{};
            workspace.struct_size = sizeof(workspace);
            workspace.ccu_count = descriptors.size();
            workspace.run_generation = generation;
            workspace.controls_addr = table;
            workspace.control_size = sizeof(descriptors.front());
            const uint64_t workspace_address = allocate(sizeof(workspace), 64);
            copy_to_device(workspace_address, &workspace, sizeof(workspace));
            prepared.workspace = workspace_address;
            return workspace_address;
        } catch (...) {
            discard(generation);
            throw;
        }
    }

    void discard(uint64_t generation) override {
        if (generation == active_generation_ && run_ && !run_->confirmed_stopped())
            throw std::runtime_error("cannot discard an executing CCU preparation");
        release_memory(generation);
        prepared_.erase(generation);
    }

    uint64_t begin(
        DomainCcuOwner &owner, const std::vector<HcommPreparedMission> &missions, uint64_t generation, uint64_t
    ) override {
        if (run_ && !run_->confirmed_stopped()) throw std::runtime_error("CCU prior run has no confirmed stop");
        if (generation == active_generation_) throw std::runtime_error("CCU generation was already activated");
        auto found = prepared_.find(generation);
        if (found == prepared_.end()) throw std::runtime_error("CCU run was not prepared");
        run_.reset();
        sdk_.reset();
        if (active_generation_ != 0) discard(active_generation_);
        active_generation_ = generation;
        const auto &prepared = found->second;
        auto event_action = [this,
                             event_result = prepared.event_result](uint64_t address, uint64_t value, uint64_t *result) {
            uint64_t arguments[]{address, value, event_result};
            aclrtLaunchKernelAttr engine{};
            engine.id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
            engine.value.engineType = ACL_RT_ENGINE_TYPE_AIV;
            aclrtLaunchKernelCfg configuration{&engine, 1};
            int rc = aclrtLaunchKernelWithHostArgs(
                trigger_function, 1, trigger_stream, &configuration, arguments, sizeof(arguments), nullptr, 0
            );
            if (rc == 0) rc = aclrtSynchronizeStreamWithTimeout(trigger_stream, 10000);
            if (rc == 0)
                rc = aclrtMemcpy(
                    result, sizeof(*result), reinterpret_cast<void *>(event_result), sizeof(*result),
                    ACL_MEMCPY_DEVICE_TO_HOST
                );
            return rc;
        };
        for (const auto &mission : missions) {
            for (const auto address :
                 {mission.ready_gate_address, mission.trigger_gate_address, mission.done_gate_address}) {
                uint64_t value{};
                check(event_action(address, 0, &value), "reset CCU Event at execution boundary");
                if (value != 0) throw std::runtime_error("CCU Event reset is unconfirmed");
            }
        }
        sdk_ = std::make_unique<HcommRunLifecycleSdk>(streams, [](uint64_t, uint64_t, bool *) {
            return -1;
        });
        run_ = owner.begin_run(generation, prepared.resources, *sdk_);
        run_->launch_all();
        return prepared.workspace;
    }

    bool stop() override {
        if (!run_) return true;
        const bool complete = run_->poll_terminal();
        if (const auto *failure = run_->primary_failure())
            throw std::runtime_error(failure->operation + ": " + std::to_string(failure->native_error));
        return complete;
    }

    void abort() override {
        if (run_) run_->abort();
    }

    bool drained() const override {
        if (closed_) return false;
        if (run_ && !run_->confirmed_stopped()) return false;
        for (const auto &mission : streams) {
            if (mission.stream == nullptr) continue;
            aclrtStreamStatus state{};
            if (aclrtStreamQuery(mission.stream, &state) != ACL_SUCCESS || state != ACL_STREAM_STATUS_COMPLETE)
                return false;
        }
        if (trigger_stream != nullptr) {
            aclrtStreamStatus state{};
            if (aclrtStreamQuery(trigger_stream, &state) != ACL_SUCCESS || state != ACL_STREAM_STATUS_COMPLETE)
                return false;
        }
        return true;
    }

    bool close() override {
        if (closed_) return true;
        if (!drained()) return false;
        run_.reset();
        sdk_.reset();
        for (const auto &entry : prepared_)
            release_memory(entry.first);
        prepared_.clear();
        if (staging_instance != 0) {
            if (HcommCcuInsDestroy(staging_instance) != CCU_SUCCESS) return false;
            staging_instance = 0;
        }
        if (!instance_assigned && unassigned_instance != 0) {
            if (HcommCcuInsDestroy(unassigned_instance) != CCU_SUCCESS) return false;
            unassigned_instance = 0;
        }
        // The communicator owns an assigned instance, the channel and acquired CCU threads.
        if (communicator != nullptr) {
            if (HcclCommDestroy(communicator) != HCCL_SUCCESS) return false;
            communicator = nullptr;
            instance_assigned = false;
            unassigned_instance = 0;
            channels.fill(0);
        }
        for (auto it = streams.rbegin(); it != streams.rend(); ++it) {
            if (it->stream != nullptr && aclrtDestroyStream(it->stream) != ACL_SUCCESS) return false;
            it->stream = nullptr;
            it->thread = 0;
        }
        streams.clear();
        if (trigger_stream != nullptr && aclrtDestroyStream(trigger_stream) != ACL_SUCCESS) return false;
        trigger_stream = nullptr;
        trigger_function = nullptr;
        if (trigger_binary != nullptr && aclrtBinaryUnLoad(trigger_binary) != ACL_SUCCESS) return false;
        trigger_binary = nullptr;
        closed_ = true;
        return true;
    }

private:
    static void copy_to_device(uint64_t address, const void *source, size_t bytes) {
        check(
            aclrtMemcpy(reinterpret_cast<void *>(address), bytes, source, bytes, ACL_MEMCPY_HOST_TO_DEVICE),
            "CCU initialize GM"
        );
    }
    uint64_t allocate(uint64_t bytes, uint32_t alignment) {
        alignment = std::max<uint32_t>(alignment, 64);
        if (bytes > std::numeric_limits<size_t>::max() - alignment)
            throw std::overflow_error("CCU GM allocation overflow");
        void *base{};
        check(aclrtMalloc(&base, bytes + alignment - 1, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(CCU run)");
        try {
            allocations_[allocation_generation_].push_back(base);
        } catch (...) {
            (void)aclrtFree(base);
            throw;
        }
        const uint64_t address = reinterpret_cast<uint64_t>(base);
        return (address + alignment - 1) & ~(uint64_t(alignment) - 1);
    }
    static uint64_t token(uint64_t address, uint64_t bytes) {
        uint64_t result{};
        check(HcommCcuGetMemToken(address, bytes, &result), "HcommCcuGetMemToken");
        return result;
    }
    void release_memory(uint64_t generation) {
        for (auto &allocation : allocations_[generation]) {
            if (allocation) check(aclrtFree(allocation), "aclrtFree(CCU run)");
            allocation = nullptr;
        }
        allocations_.erase(generation);
    }
    struct PreparedRun {
        uint64_t workspace{};
        uint64_t event_result{};
        std::vector<MissionRunResources> resources;
    };
    std::map<uint64_t, std::vector<void *>> allocations_;
    std::map<uint64_t, PreparedRun> prepared_;
    uint64_t allocation_generation_{};
    uint64_t active_generation_{};
    std::unique_ptr<HcommRunLifecycleSdk> sdk_;
    std::unique_ptr<DomainRun> run_;
    bool closed_{false};
};

}  // namespace

std::vector<HcommPreparedMission> prepare_hcomm_ccu_callable(
    HcclComm base_communicator, uint64_t domain_id, const std::vector<uint32_t> &rank_ids, uint32_t domain_rank,
    const ExpectedBuildIdentity &expected_identity, const std::vector<std::vector<uint8_t>> &module_bytes,
    const std::string &program_namespace, std::unique_ptr<DomainCcuOwner> &owner, HcommDomainSession *&session
) {
    if (owner) throw std::invalid_argument("CCU domain already has an owner");
    if (base_communicator == nullptr || rank_ids.size() < 2 || domain_rank >= rank_ids.size() || module_bytes.empty() ||
        program_namespace.empty())
        throw std::invalid_argument("invalid A5 CCU domain preparation request");

    static NativeModuleCache module_cache;
    std::vector<std::shared_ptr<NativeModule>> modules;
    std::vector<NativeKernelDefinition> kernels;
    std::vector<uint32_t> kernel_modules;
    for (const auto &bytes : module_bytes) {
        auto module = module_cache.load(bytes, expected_identity);
        for (const auto &kernel : module->kernels()) {
            kernels.push_back(kernel);
            kernel_modules.push_back(static_cast<uint32_t>(modules.size()));
        }
        modules.push_back(std::move(module));
    }
    if (kernels.size() > 1024) throw std::invalid_argument("too many CCU kernels");
    if (kernels.empty()) throw std::invalid_argument("the callable has no CCU kernels");

    auto owned_backend = std::make_unique<HcommDomainResourceBackend>();
    auto *backend = owned_backend.get();
    backend->domain_id = domain_id;
    owner = std::make_unique<DomainCcuOwner>(std::move(owned_backend));
    session = backend;
    for (auto &module : modules)
        owner->retain_module(std::move(module));
    try {
        HcclCommConfig config{};
        HcclCommConfigInit(&config);
        // HcclCommConfig uses HcclAccelerator values, not HcclOpExpansionMode enum values.
        constexpr uint32_t ccu_ms_config_mode = 5;
        config.hcclOpExpansionMode = ccu_ms_config_mode;
        auto mutable_rank_ids = rank_ids;
        check(
            HcclCreateSubCommConfig(
                &base_communicator, static_cast<uint32_t>(mutable_rank_ids.size()), mutable_rank_ids.data(), domain_id,
                domain_rank, &config, &backend->communicator
            ),
            "HcclCreateSubCommConfig(CCU)"
        );
        backend->channels = acquire_peer_channels(backend->communicator, domain_rank, rank_ids.size());
        uint32_t die = UINT32_MAX;
        for (const auto channel : backend->channels) {
            if (channel == 0) continue;
            const uint32_t peer_die = find_channel_die(channel);
            if (die != UINT32_MAX && die != peer_die)
                throw std::runtime_error(
                    "CCU callable peer channels span IO dies; a single-die program cannot bind them"
                );
            die = peer_die;
        }

        auto &staging = backend->staging_instance;
        HcommCcuResDescHandle staging_descriptor{};
        check(HcommCcuInsResDescCreate(die, &staging_descriptor), "create staging resource descriptor");
        try {
            // Query executes definitions without reserving their program resources; only bound Events are live.
            check(
                HcommCcuInsResDescSetNum(staging_descriptor, HCOMM_CCU_RES_TYPE_EVENT, kernels.size() * 3),
                "size staging Events"
            );
            const HcommCcuResDescHandle descriptors[]{staging_descriptor};
            check(HcommCcuInsCreate(descriptors, 1, &staging), "HcommCcuInsCreate(staging Events)");
        } catch (...) {
            (void)HcommCcuInsResDescDestroy(staging_descriptor);
            throw;
        }
        check(HcommCcuInsResDescDestroy(staging_descriptor), "destroy staging resource descriptor");
        try {
            HcommDomainRegistrationSdk sizing(staging, die);
            for (uint32_t i = 0; i < kernels.size(); ++i) {
                std::array<CcuEventHandle, 3> events{};
                for (auto &event : events)
                    check(HcommCcuEventAlloc(staging, die, 1, &event), "allocate sizing Event");
                const auto binding =
                    binding_for(domain_rank, static_cast<uint32_t>(rank_ids.size()), backend->channels, i, events);
                const void *arguments[] = {&binding};
                check(
                    sizing.query_resources(kernels[i].entry, arguments, SIMPLER_CCU_REGISTRATION_ARGUMENT_COUNT),
                    "HcommCcuKernelQueryResReq(sizing)"
                );
            }

            HcommCcuResDescHandle descriptor{};
            check(HcommCcuInsResDescCreate(die, &descriptor), "HcommCcuInsResDescCreate(sized)");
            try {
                for (size_t type = 0; type < sizing.required_resources().size(); ++type) {
                    uint64_t count = sizing.required_resources()[type];
                    if (type == HCOMM_CCU_RES_TYPE_EVENT) count += kernels.size() * 3;
                    if (count > std::numeric_limits<uint32_t>::max())
                        throw std::overflow_error("CCU resource requirement exceeds SDK width");
                    check(
                        HcommCcuInsResDescSetNum(
                            descriptor, static_cast<HcommCcuResType>(type), static_cast<uint32_t>(count)
                        ),
                        "HcommCcuInsResDescSetNum"
                    );
                }
                check(HcommCcuInsDestroy(staging), "HcommCcuInsDestroy(staging)");
                staging = 0;
                const HcommCcuResDescHandle descriptors[] = {descriptor};
                check(HcommCcuInsCreate(descriptors, 1, &backend->unassigned_instance), "HcommCcuInsCreate(sized)");
            } catch (...) {
                (void)HcommCcuInsResDescDestroy(descriptor);
                throw;
            }
            check(HcommCcuInsResDescDestroy(descriptor), "HcommCcuInsResDescDestroy(sized)");
        } catch (...) {
            throw;
        }

        DomainRegistrationRequest request{};
        request.domain_id = domain_id;
        request.rank_id = domain_rank;
        request.rank_count = static_cast<uint32_t>(rank_ids.size());
        request.channels = backend->channels;
        request.program_namespace = program_namespace;
        std::vector<HcommPreparedMission> missions;
        missions.reserve(kernels.size());
        for (uint32_t i = 0; i < kernels.size(); ++i) {
            std::array<CcuEventHandle, 3> events{};
            std::array<uint64_t, 3> addresses{};
            for (uint32_t j = 0; j < 3; ++j) {
                check(HcommCcuEventAlloc(backend->unassigned_instance, die, 1, &events[j]), "allocate CCU Event");
                check(HcommCcuEventGetAddr(events[j], 0, &addresses[j]), "map CCU Event");
            }
            const auto &kernel = kernels[i];
            request.missions.push_back({events[0], events[1], events[2]});
            HcommPreparedMission mission{};
            mission.ccu_id = i;
            mission.kernel_id = kernel.record.kernel_id;
            mission.module_index = kernel_modules[i];
            mission.ready_event_handle = events[0];
            mission.trigger_event_handle = events[1];
            mission.done_event_handle = events[2];
            mission.ready_gate_address = addresses[0];
            mission.trigger_gate_address = addresses[1];
            mission.done_gate_address = addresses[2];
            mission.parameter_size = kernel.record.parameter_size;
            mission.parameter_alignment = kernel.record.parameter_alignment;
            missions.push_back(mission);
        }

        HcommDomainRegistrationSdk registration(backend->unassigned_instance, die);
        const auto registered = register_domain_kernels(kernels, request, registration);
        backend->streams.reserve(missions.size());
        for (auto &mission : missions) {
            mission.program_handle = registered.at(mission.ccu_id).program_handle;
            if (mission.program_handle == 0) throw std::runtime_error("missing registered CCU mission program");
            HcommMissionStream stream{};
            stream.handle = static_cast<uint64_t>(backend->streams.size() + 1);
            check(aclrtCreateStream(&stream.stream), "aclrtCreateStream(CCU mission)");
            backend->streams.push_back(stream);
            check(
                HcclThreadAcquireWithStream(
                    backend->communicator, COMM_ENGINE_CCU, backend->streams.back().stream, 1,
                    &backend->streams.back().thread
                ),
                "HcclThreadAcquireWithStream(CCU mission)"
            );
            mission.stream_handle = stream.handle;
        }

        check(aclrtCreateStream(&backend->trigger_stream), "aclrtCreateStream(CCU trigger)");
        const std::string binary_path = sibling_trigger_binary();
        check(
            aclrtBinaryLoadFromFile(binary_path.c_str(), nullptr, &backend->trigger_binary),
            "aclrtBinaryLoadFromFile(CCU trigger)"
        );
        check(
            aclrtBinaryGetFunction(backend->trigger_binary, "simpler_ccu_event_trigger", &backend->trigger_function),
            "aclrtBinaryGetFunction(CCU trigger)"
        );
        check(HcclCommAssignCcuIns(backend->communicator, backend->unassigned_instance), "HcclCommAssignCcuIns");
        backend->instance_assigned = true;
        CcuInsHandle assigned{};
        uint32_t assigned_count{};
        check(
            HcclCommQueryAssignedCcuIns(backend->communicator, &assigned, &assigned_count),
            "HcclCommQueryAssignedCcuIns"
        );
        if (assigned_count != 1 || assigned != backend->unassigned_instance)
            throw std::runtime_error("assigned CCU instance does not match the prepared instance");

        owner->set_registered_kernels(registered);
        return missions;
    } catch (...) {
        // Keep the owner attached to the allocation when SDK cleanup cannot be confirmed.
        if (owner->close()) {
            owner.reset();
            session = nullptr;
        }
        throw;
    }
}

}  // namespace simpler::ccu
