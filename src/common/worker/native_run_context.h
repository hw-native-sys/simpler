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

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include "call_config.h"
#include "common/host_api.h"
#include "native_run_execution.h"
#include "native_run_phase.h"
#include "native_run_storage_tag.h"
#include "runtime_c_api.h"
#include "runtime.h"
#include "../platform_comm/comm_domain_use_wire.h"

/**
 * The single placement-owned carrier for one progressable native lifecycle.
 * It captures immutable selection, identity, acceptance, configuration, and
 * HostApi binding during prepare; later phases never recover them from a
 * runner field or thread-local. The object is address-stable through finalize.
 */
template <typename Runner>
struct NativeRunContext {
    static constexpr uint64_t kMagic = kNativeRunContextMagic;

    NativeRunContext(
        Runner *runner_in, const CallConfig &config_in, uint64_t trace_hid_in, const NativeRunDescriptor &descriptor_in,
        const HostApiOps *host_api_ops, const NativeRunDomainProvider *domain_provider = nullptr,
        const NativeRunDomainProviderV2 *domain_provider_v2 = nullptr
    ) :
        runner(runner_in),
        config(config_in),
        descriptor(descriptor_in),
        host_api(
            runner_in, descriptor_in.pipeline_slot, descriptor_in.arena_bank, descriptor_in.run_epoch, host_api_ops,
            HostApi::DiagnosticConfig{static_cast<uint32_t>(config_in.enable_chip_swimlane), config_in.enable_pmu > 0},
            this
        ),
        trace_hid(trace_hid_in) {
        const NativeRunDomainProvider empty_provider{1, 0, nullptr, nullptr, 0};
        const auto &provider = domain_provider != nullptr ? *domain_provider : empty_provider;
        if (provider.reserved != 0 ||
            (domain_provider_v2 == nullptr ? provider.version != 1 :
                                             (domain_provider != &domain_provider_v2->base || provider.version != 2)))
            throw std::invalid_argument("unsupported native communication-domain provider version");
        if (provider.count != 0) {
            if (provider.owner == nullptr || provider.query == nullptr)
                throw std::invalid_argument("native communication-domain provider is missing");
            domain_snapshots.resize(provider.count);
            domain_execution.resize(provider.count);
            domain_sdma_owners.resize(provider.count);
            std::vector<CommDomainUse> uses;
            uses.reserve(provider.count);
            for (size_t i = 0; i < provider.count; ++i) {
                uint32_t qualification = 0;
                if (provider.query(
                        provider.owner, descriptor.run_epoch, descriptor.pipeline_slot, i, &domain_snapshots[i],
                        &qualification
                    ) != 0 ||
                    qualification > static_cast<uint32_t>(CommDomainExecution::EXCLUSIVE_RUN) ||
                    !comm_domain_snapshot_valid(domain_snapshots[i]))
                    throw std::invalid_argument("native communication-domain owner query failed");
                if (domain_provider_v2 != nullptr && domain_provider_v2->query_sdma_owner != nullptr) {
                    auto &info = domain_sdma_owners[i];
                    if (domain_provider_v2->query_sdma_owner(
                            provider.owner, descriptor.run_epoch, descriptor.pipeline_slot, i, &info, sizeof(info)
                        ) != 0 ||
                        !comm_sdma_owner_info_valid(info))
                        throw std::invalid_argument("native SDMA workspace owner query failed");
                    const auto &workspace = comm_workspace<CommWorkspaceKind::SDMA>(domain_snapshots[i].context);
                    if (info.access != CommSdmaAccess::UNKNOWN &&
                        (info.workspace_addr != workspace.addr || info.workspace_bytes != workspace.size))
                        throw std::invalid_argument("native SDMA workspace owner does not match selected domain");
                }
                domain_execution[i] = static_cast<CommDomainExecution>(qualification);
                uses.push_back(domain_snapshots[i].use);
            }
            validate_comm_domain_uses(uses);
        }
        // The borrowed provider is consumed here and never stored. Later
        // phases retain values, never the submitting Host object's address.
        // Publish the storage tag only after every potentially-throwing member
        // has been constructed. A failed placement construction must leave the
        // caller-owned slot reusable rather than looking like a prepared run.
        magic = kMagic;
    }

    NativeRunContext(const NativeRunContext &) = delete;
    NativeRunContext &operator=(const NativeRunContext &) = delete;
    NativeRunContext(NativeRunContext &&) = delete;
    NativeRunContext &operator=(NativeRunContext &&) = delete;

    /** Publish acceptance only from this run's completed launch receipt. */
    bool publish_acceptance(const LaunchReceipt &receipt) const noexcept {
        if (!receipt.matches(identity())) return false;
        if (descriptor.accepted_state != nullptr) {
            __atomic_store_n(descriptor.accepted_state, descriptor.accepted_value, __ATOMIC_RELEASE);
        }
        return true;
    }

    uint64_t magic{0};
    Runner *runner{nullptr};
    CallConfig config{};
    NativeRunDescriptor descriptor{};
    HostApi host_api;
    std::vector<CommDomainSnapshot> domain_snapshots;
    std::vector<CommDomainExecution> domain_execution;
    std::vector<CommSdmaOwnerInfoV1> domain_sdma_owners;
    std::unique_ptr<Runtime> runtime{std::make_unique<Runtime>()};
    uint64_t trace_hid{0};
    uint64_t trace_inv{0};
    long long trace_start_ns{0};
    long long runner_trace_start_ns{0};
    uint64_t clock_log_offset{0};
    int completion_rc{-1};
    bool result_observed{false};
    bool physical_complete{false};
    bool execution_retired{false};
    // This call's finished host-phase export requirement, retained across slot reuse.
    bool clock_alignment_log_required{false};
    // Detached unmanaged HOST staging, owned until output delivery/release.
    void *retired_host_staging{nullptr};
    std::atomic<NativeRunPhase> phase{NativeRunPhase::Prepared};
    std::unique_ptr<typename Runner::PreparedExecution> prepared_execution{};
    std::unique_ptr<typename Runner::ActiveExecution> active_execution{};
    LaunchPermit launch_permit{};
    char trace_attrs[192]{};
    // True only for a launch entry that has not attempted execution-visible
    // submission, or whose returned transaction explicitly reports NotStarted.
    // Backend exceptions return Partial with an active owner, never this proof.
    bool launch_not_started_proven{false};
    // A returned finalize may leave a Canonical carrier for cleanup retry.
    // The synchronous output reader is not repeated after it has returned.
    bool recording{false};
    bool copyback_returned{false};
    bool failure_reported{false};
    int copyback_rc{0};
    bool bindings_released{false};
    bool runner_resources_owned{false};
    bool runner_reserved{false};
    bool runner_claimed{false};

    NativeRunIdentity identity() const {
        return NativeRunIdentity{
            descriptor.run_epoch, descriptor.generation, descriptor.dispatch_id, descriptor.pipeline_slot
        };
    }

    /**
     * This run's whole published identity, by value.
     *
     * Separate from `identity()` because that one carries what later phase
     * calls address the run by, and has no field for `run_id` — which a reader
     * joining an artifact to the submission that produced it needs. The
     * acceptance sink and its value are deliberately absent: they are the
     * caller's pointer and sentinel, and belong to `publish_acceptance` alone.
     */
    NativeRunArtifactIdentity artifact_identity() const {
        NativeRunArtifactIdentity snapshot{};
        snapshot.run_id = descriptor.run_id;
        snapshot.dispatch_id = descriptor.dispatch_id;
        snapshot.generation = descriptor.generation;
        std::memcpy(
            snapshot.owner_instance_id, descriptor.provenance.owner_instance_id, sizeof(snapshot.owner_instance_id)
        );
        snapshot.pipeline_slot = descriptor.pipeline_slot;
        snapshot.chip_rank = descriptor.provenance.chip_rank;
        snapshot.capture_index = descriptor.provenance.capture_index;
        snapshot.task_slot = descriptor.provenance.task_slot;
        snapshot.group_index = descriptor.provenance.group_index;
        snapshot.group_size = descriptor.provenance.group_size;
        return snapshot;
    }
};

/** End object lifetime, then mark the caller-owned storage reusable. */
template <typename Runner>
void destroy_native_run_context(NativeRunContext<Runner> *context) {
    static_assert(sizeof(NativeRunContext<Runner>) >= kNativeRunPrepareReleaseBytes);
    void *storage = context;
    context->~NativeRunContext<Runner>();
    constexpr uint64_t kEmpty = 0;
    std::memcpy(storage, &kEmpty, sizeof(kEmpty));
}
