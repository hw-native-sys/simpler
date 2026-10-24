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
#include "device_runner_base.h"

#include <acl/acl.h>
#include <cstring>
#include <limits>
#include <atomic>
#include <optional>

#include "common/host_api.h"
#include "host/kernel_launch_binder.h"
#include "host_build_graph/host/host_tensor_access.h"
#include "host_build_graph/host_graph_build.h"
#include "host_build_graph/kernel_argument_snapshot.h"
#include "host_build_graph/kernel_callable_registration.h"
#include "host_build_graph/kernel_resource_requirements.h"
#include "host_build_graph/kernel_graph_template.h"
#include "task_interface/kernel_invocation_validation.h"

namespace {
std::mutex claim_mutex;
std::unordered_set<int> claimed_devices;
std::atomic<uint64_t> next_graph_lease{1};

uint64_t allocate_graph_lease() noexcept {
    auto next = next_graph_lease.load(std::memory_order_relaxed);
    while (next != std::numeric_limits<uint64_t>::max()) {
        if (next_graph_lease.compare_exchange_weak(next, next + 1, std::memory_order_relaxed)) return next;
    }
    return 0;
}

int drain_owned_streams(const KernelExecutionState &state, int timeout_ms) {
    for (auto kind : {KernelStreamKind::Aicpu, KernelStreamKind::Aicore}) {
        const int rc = aclrtSynchronizeStreamWithTimeout(state.hidden_stream(kind), timeout_ms);
        if (rc != 0) return rc;
    }
    return 0;
}

int publish_control(
    DeviceRunnerBase &runner, void *stream, void *payload, size_t bytes, const char *symbol, int timeout_ms,
    bool *published = nullptr
) try {
    const int rc = runner.launch_aicpu_payload(stream, payload, bytes, symbol, 1);
    if (rc != 0) return rc;
    // Submission acceptance survives a failed completion wait. Cleanup must
    // revoke accepted work, but must not revoke a record never submitted.
    if (published != nullptr) *published = true;
    return aclrtSynchronizeStreamWithTimeout(stream, timeout_ms);
} catch (...) {
    return PTO_RUNTIME_ERR_INTERNAL;
}
}  // namespace

namespace hbg {
struct KernelContextLaunchState {
    int device{-1};
    uint64_t context_generation{0};
    uint64_t runtime_binary_id{0};
    GraphSlotRegistration registration{};
    RuntimeArenaLayout layout{};
    uintptr_t previous_caller_identity{0};
    // Zero means unbound. A successful Host launch binds this stream while
    // graph resource tokens are held; the last successful release clears it.
    uintptr_t bound_submit_stream{0};
    std::unordered_set<uint64_t> graph_leases;
    bool diagnostics_requested{false};
    bool registered{false};
    bool submitted{false};
    bool closing{false};
    bool uncertain{false};
    ~KernelContextLaunchState() {
        if (device >= 0) {
            std::scoped_lock lock(claim_mutex);
            claimed_devices.erase(device);
        }
    }
};
struct KernelCallableLaunchState {
    Runtime runtime;
    PersistentKernelArgs arguments;
    simpler::kernel::HbgCallableBinding binding{};
    bool published{false};
};
}  // namespace hbg

extern "C" int runtime_hbg_kernel_supported_impl() { return 1; }

int DeviceRunnerBase::admit_hbg_native_run() {
    if (hbg_launch_context_ == nullptr) return 0;
    if (hbg_launch_context_->closing || hbg_launch_context_->uncertain || !hbg_launch_context_->graph_leases.empty())
        return PTO_RUNTIME_ERR_INVALID_STATE;
    if (!hbg_launch_context_->submitted) return 0;
    aclrtEventRecordedStatus status{};
    const int rc = aclrtQueryEventStatus(kernel_exec_state_.event(KernelEventKind::SerialTail), &status);
    if (rc != 0) return rc;
    return status == ACL_EVENT_RECORDED_STATUS_COMPLETE ? 0 : PTO_RUNTIME_ERR_INVALID_STATE;
}

int DeviceRunnerBase::configure_hbg_kernel_events() {
    if (runtime_hbg_kernel_supported_impl() == 0) return 0;
    uint8_t mode = 0;
    int rc = rtEventWorkModeGet(&mode);
    if (rc == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) return 0;
    if (rc != 0 || mode == 1) return rc;
    rc = rtEventWorkModeSet(1);
    if (rc == 0) return 0;
    if (rtEventWorkModeGet(&mode) == 0 && mode == 1) return 0;
    return rc;
}

int DeviceRunnerBase::prepare_hbg_kernel_runtime(
    const CallConfig &config, uint64_t generation, uint64_t runtime_binary_id
) {
    if (hbg_launch_context_ != nullptr) return PTO_RUNTIME_ERR_INVALID_STATE;
    auto state = std::make_shared<hbg::KernelContextLaunchState>();
    {
        std::scoped_lock lock(claim_mutex);
        if (!claimed_devices.insert(device_id_).second) return PTO_RUNTIME_ERR_INVALID_STATE;
        state->device = device_id_;
    }
    state->context_generation = generation;
    state->diagnostics_requested = config.enable_chip_swimlane || config.enable_dump_args || config.enable_pmu ||
                                   config.enable_dep_gen || config.enable_scope_stats;
    state->runtime_binary_id = runtime_binary_id;
    if (state->runtime_binary_id == 0) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    hbg_launch_context_ = state;
    if (aicore_bin_handle_ == nullptr) {
        rtDevBinary_t binary{};
        binary.magic = RT_DEV_BINARY_MAGIC_ELF;
        binary.data = aicore_kernel_binary_.data();
        binary.length = aicore_kernel_binary_.size();
        const int rc = rtRegisterAllKernel(&binary, &aicore_bin_handle_);
        if (rc != 0) return rc;
    }
    const uint64_t capacity =
        config.runtime_env.ring_task_window[0] == 0 ? CHIP_DEFAULT_GRAPH_TASKS : config.runtime_env.ring_task_window[0];
    int rc = hbg::make_kernel_graph_layout(capacity, state->layout);
    if (rc != 0) return rc;
    const uint64_t budget =
        config.runtime_env.ring_heap[0] == 0 ? UINT64_C(256) * 1024 * 1024 : config.runtime_env.ring_heap[0];
    hbg::GraphResourceRequirements requirements{
        {hbg::RuntimeArchitecture::A2A3, capacity, state->layout.arena_size, state->layout.off_copied_begin,
         state->layout.off_copied_end},
        budget,
        state->layout.off_copied_end + SharedMemoryHandle::calculate_size(capacity),
        budget,
        0
    };
    hbg::KernelResourceCapacity plan;
    rc = hbg::KernelResourceCapacity::create(&requirements, 1, plan);
    if (rc == 0) rc = plan.prepare(kernel_exec_state_, KernelResourceOps::from_allocator(mem_alloc_));
    if (rc == 0) rc = kernel_exec_state_.freeze_resources();
    if (rc == 0)
        rc = plan.make_registration(
            kernel_exec_state_, device_id_, generation, state->runtime_binary_id, state->registration
        );
    if (rc != 0) return rc;
    const auto &binding = state->registration;
    kernel_runtime_.set_gm_sm_ptr(
        reinterpret_cast<void *>(binding.destinations[1].address + state->layout.off_copied_end)
    );
    kernel_runtime_.set_prebuilt_arena(
        reinterpret_cast<void *>(binding.destinations[1].address), state->layout.off_runtime
    );
    auto stream = kernel_exec_state_.hidden_stream(KernelStreamKind::Aicpu);
    rc = publish_control(
        *this, stream, &state->registration, sizeof(state->registration),
        "simpler_aicpu_l1_hbg_register_execution_slot", timeout_config_.stream_sync_timeout_ms, &state->registered
    );
    if (rc != 0) kernel_exec_state_.poison(rc);
    return rc;
}

int DeviceRunnerBase::prepare_hbg_kernel_callable(int32_t id) {
    const auto it = callables_.find(id);
    if (hbg_launch_context_ == nullptr || !hbg_launch_context_->registered || hbg_launch_context_->closing ||
        it == callables_.end() || it->second.host_orch_func_ptr == nullptr)
        return PTO_RUNTIME_ERR_INVALID_STATE;
    if (it->second.hbg_launch_state != nullptr) return PTO_RUNTIME_ERR_INVALID_STATE;
    if (next_hbg_callable_generation_ == std::numeric_limits<uint64_t>::max()) return PTO_RUNTIME_ERR_INVALID_STATE;
    auto state = std::make_shared<hbg::KernelCallableLaunchState>();
    it->second.hbg_launch_state = state;
    std::memcpy(&state->runtime.dev, &kernel_runtime_.dev, sizeof(kernel_runtime_.dev));
    int rc = bind_registered_callable_tables(state->runtime, id);
    if (rc != 0) return rc;
    auto &binding = state->binding;
    binding.callable_id = id;
    binding.context_generation = hbg_launch_context_->context_generation;
    binding.callable_generation = next_hbg_callable_generation_++;
    if (simpler::kernel::derive_invocation_counts(
            it->second.signature.data(), it->second.signature.size(), &binding.tensor_count, &binding.scalar_count
        ) != simpler::kernel::InvocationStatus::Ok)
        return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    rc = state->arguments.prepare_once(state->runtime, persistent_args_ops(), device_id_);
    if (rc != 0) return rc;
    binding.runtime_address = reinterpret_cast<uint64_t>(state->arguments.args().runtime_args);
    auto stream = kernel_exec_state_.hidden_stream(KernelStreamKind::Aicpu);
    rc = publish_control(
        *this, stream, &binding, sizeof(binding), "simpler_aicpu_l1_hbg_register_callable",
        timeout_config_.stream_sync_timeout_ms, &state->published
    );
    if (rc != 0) {
        kernel_exec_state_.poison(rc);
        return rc;
    }
    return kernel_exec_state_.mark_ready_enqueued();
}

namespace kl = simpler::kernel_launch;

struct DeviceRunnerBase::HbgKernelLaunchHelpers {
    struct HbgGraphBuildWorkspace {
        DeviceArena arena;
        RuntimeContext *runtime{nullptr};  // Borrowed from arena.
        std::optional<ChipTaskArgs> orchestration;
        std::optional<HostTensorAccessor> access;
        std::optional<hbg::GraphBuild> build;
    };

    struct HbgGraphPacket {
        std::optional<hbg::GraphLaunchTemplate> graph;
        std::optional<hbg::GraphHostArgs> packet;
    };

    struct HbgLaunchPayload {
        std::optional<std::vector<uint8_t>> bytes;
    };

    // Owners survive submission; deferred construction and member order retain
    // the original construction and reverse-destruction sequence on every exit.
    struct HbgLaunchPreparation {
        HbgGraphBuildWorkspace workspace;
        HbgGraphPacket packet;
        HbgLaunchPayload payload;
    };

    // The caller owns this context and its payload until the binder returns.
    struct Submission {
        DeviceRunnerBase *runner;
        hbg::KernelContextLaunchState *context;
        hbg::KernelCallableLaunchState *callable;
        void *caller;
        std::vector<uint8_t> *launch;
    };

    static int resolve_hbg_callable_for_launch(
        DeviceRunnerBase &runner, int32_t id, void *caller, const HostApi *api, CallableState *&registered_callable
    ) {
        const auto callable_it = runner.callables_.find(id);
        // Retention protects captured work through graph destruction. This
        // binding constrains Host launches, including those recorded during
        // capture; it does not detect capture or intercept external replay.
        // The caller must serialize graph replays on the bound stream.
        if (runner.hbg_launch_context_ != nullptr && runner.hbg_launch_context_->bound_submit_stream != 0 &&
            runner.hbg_launch_context_->bound_submit_stream != reinterpret_cast<uintptr_t>(caller))
            return PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE;
        if (runner.hbg_launch_context_ == nullptr || runner.hbg_launch_context_->closing ||
            runner.hbg_launch_context_->uncertain || !runner.kernel_exec_state_.accepts_launch() ||
            callable_it == runner.callables_.end() || callable_it->second.hbg_launch_state == nullptr ||
            !callable_it->second.hbg_launch_state->published || api == nullptr)
            return PTO_RUNTIME_ERR_INVALID_STATE;
        if (runner.hbg_launch_context_->diagnostics_requested) return PTO_RUNTIME_ERR_UNSUPPORTED;
        registered_callable = &callable_it->second;
        return 0;
    }

    static int validate_hbg_launch_arguments(
        const ChipStorageTaskArgs &args, const simpler::kernel::HbgCallableBinding &binding, uint64_t &argument_hash
    ) {
        if (args.tensor_count() != binding.tensor_count || args.scalar_count() != binding.scalar_count)
            return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        for (int i = 0; i < args.tensor_count(); ++i) {
            if (args.tensor(i).address_space != AddressSpace::DEVICE || args.tensor(i).transfer != TensorTransfer::NONE)
                return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        }
        argument_hash = hbg::kernel_argument_snapshot_hash(args);
        if (argument_hash == 0) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        return 0;
    }

    static int build_hbg_task_graph(
        DeviceRunnerBase &runner, const ChipStorageTaskArgs &args, const HostApi *api,
        CallableState &registered_callable, HbgGraphBuildWorkspace &workspace
    ) {
        auto &state = *registered_callable.hbg_launch_state;
        // The Host workspace uses the task capacity frozen at context initialization.
        auto &arena = workspace.arena;
        const uint64_t capacity = runner.hbg_launch_context_->layout.task_capacity;
        const auto layout = runtime_reserve_layout(arena, capacity);
        if (arena.commit(DeviceArena::kDefaultBaseAlign) == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
        const uint64_t sm_bytes = SharedMemoryHandle::calculate_size(capacity);
        auto *rt = runtime_init_data_from_layout(arena, layout, MODE_EXECUTE, nullptr, sm_bytes);
        if (rt == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
        runtime_wire_arena_pointers(arena, layout, rt);
        rt->prebuilt_layout = layout;
        workspace.runtime = rt;

        // Host orchestration builds a fresh graph for this invocation.
        state.runtime.set_orch_args(args);
        auto &orchestration = workspace.orchestration.emplace();
        orchestration.create_from_entry_storage(state.runtime.get_orch_args());
        auto &access = workspace.access.emplace();
        auto &build = workspace.build.emplace();
        int rc = hbg::build_graph(
            &state.runtime, api, access, rt, sm_bytes, capacity, registered_callable.host_orch_func_ptr, orchestration,
            build
        );
        if (rc < 0) return rc;
        return 0;
    }

    // Requires a successful graph build; the workspace remains owned by preparation.
    static int encode_hbg_graph_packet(
        DeviceRunnerBase &runner, int32_t id, const ChipStorageTaskArgs &args, const CallableState &registered_callable,
        uint64_t argument_hash, HbgGraphBuildWorkspace &workspace, HbgGraphPacket &output
    ) {
        if (!workspace.build.has_value() || workspace.runtime == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
        const auto &state = *registered_callable.hbg_launch_state;
        // The packet carries the callable generation, code identity and argument snapshot.
        const hbg::GraphInvocationIdentity identity{
            id,
            args.tensor_count(),
            args.scalar_count(),
            state.binding.callable_generation,
            registered_callable.chip_buffer_hash,
            argument_hash,
            registered_callable.aicore_image_hash
        };
        auto &graph = output.graph.emplace();
        int rc = hbg::make_graph_launch_template(
            *workspace.build, *workspace.runtime, runner.hbg_launch_context_->registration, identity, graph
        );
        if (rc != 0) return rc;
        auto &packet = output.packet.emplace();
        rc = hbg::make_graph_host_args(graph, packet);
        if (rc != 0 || packet.bytes > UINT32_MAX - LAUNCH_ENVELOPE_HEADER_BYTES)
            return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        return 0;
    }

    // Encoding has validated packet size; only the resulting bytes are submitted.
    static int
    pack_hbg_launch_payload(hbg::KernelCallableLaunchState &state, HbgGraphPacket &encoded, HbgLaunchPayload &payload) {
        if (!encoded.packet.has_value()) return PTO_RUNTIME_ERR_INTERNAL;
        auto &packet = *encoded.packet;
        // The launch envelope selects packet validation and fixed-slot restoration on AICPU.
        auto &launch = payload.bytes.emplace(LAUNCH_ENVELOPE_HEADER_BYTES + packet.bytes, 0);
        auto header = state.arguments.args();
        header.entry_args_offset = LAUNCH_ENVELOPE_HEADER_BYTES;
        header.entry_args_source = static_cast<uint32_t>(EntryArgsSource::HbgKernelInvocation);
        header.graph_section_offset = LAUNCH_ENVELOPE_HEADER_BYTES;
        header.graph_section_bytes = packet.bytes;
        header.graph_section_source = static_cast<uint32_t>(GraphSectionSource::RestoredSlot);
        std::memcpy(launch.data(), &header, sizeof(header));
        std::memcpy(launch.data() + LAUNCH_ENVELOPE_HEADER_BYTES, packet.storage.data(), packet.bytes);
        return 0;
    }

    static kl::KernelLaunchGateOps make_hbg_launch_gate(Submission &submission) {
        kl::KernelLaunchGateOps gate;
        gate.context = &submission;
        gate.acquire = [](void *p, const kl::KernelInvocationBinding &, void *,
                          kl::KernelLaunchAdmission *out) noexcept -> int {
            auto &s = *static_cast<Submission *>(p);
            auto &context = s.runner->kernel_exec_state_;
            if (!context.accepts_launch()) return PTO_RUNTIME_ERR_INVALID_STATE;
            out->handles = {
                nullptr,
                context.hidden_stream(KernelStreamKind::Aicpu),
                context.hidden_stream(KernelStreamKind::Aicore),
                context.event(KernelEventKind::Start),
                context.event(KernelEventKind::AicoreStart),
                context.event(KernelEventKind::AicoreDone),
                context.event(KernelEventKind::AicpuDone),
                context.event(KernelEventKind::SerialTail)
            };
            out->previous_caller_identity = s.context->previous_caller_identity;
            return 0;
        };
        gate.finish = [](void *p, const kl::KernelLaunchResult &result) noexcept {
            auto &s = *static_cast<Submission *>(p);
            if (result.status == 0) {
                s.context->previous_caller_identity = reinterpret_cast<uintptr_t>(s.caller);
                s.context->submitted = true;
                if (!s.context->graph_leases.empty())
                    s.context->bound_submit_stream = reinterpret_cast<uintptr_t>(s.caller);
            } else if (result.enqueue_started) {
                s.context->uncertain = true;
                s.runner->kernel_exec_state_.poison(result.status);
            }
        };
        gate.query_tail = [](void *, void *event, bool *complete) noexcept {
            aclrtEventRecordedStatus status{};
            const int rc = aclrtQueryEventStatus(event, &status);
            *complete = rc == 0 && status == ACL_EVENT_RECORDED_STATUS_COMPLETE;
            return rc;
        };
        return gate;
    }

    static kl::KernelLaunchOps make_hbg_launch_ops(Submission &submission) {
        kl::KernelLaunchOps ops;
        ops.context = &submission;
        ops.wait_event = [](void *, void *stream, void *event) noexcept {
            return aclrtStreamWaitEvent(stream, event);
        };
        ops.record_event = [](void *, void *event, void *stream) noexcept {
            return aclrtRecordEvent(event, stream);
        };
        ops.memset_handshake = [](void *p, void *stream) noexcept {
            auto &s = *static_cast<Submission *>(p);
            const auto offset = reinterpret_cast<uintptr_t>(s.callable->runtime.get_workers()) -
                                reinterpret_cast<uintptr_t>(&s.callable->runtime);
            const size_t bytes = sizeof(Handshake) * s.callable->runtime.get_worker_count();
            auto *reports = reinterpret_cast<uint8_t *>(s.callable->arguments.args().runtime_args) + offset;
            return aclrtMemsetAsync(reports, bytes, 0, bytes, stream);
        };
        ops.launch_aicore = [](void *p, void *stream) noexcept -> int {
            auto &s = *static_cast<Submission *>(p);
            try {
                s.runner->activate_launch_shape(s.callable->runtime);
                return s.runner->launch_aicore_kernel(stream, s.callable->arguments.args());
            } catch (...) {
                return PTO_RUNTIME_ERR_INTERNAL;
            }
        };
        ops.launch_aicpu = [](void *p, void *stream) noexcept -> int {
            auto &s = *static_cast<Submission *>(p);
            try {
                return s.runner->launch_aicpu_payload(
                    stream, s.launch->data(), s.launch->size(), host::KernelNames::RunName,
                    s.callable->runtime.get_aicpu_launch_count()
                );
            } catch (...) {
                return PTO_RUNTIME_ERR_INTERNAL;
            }
        };
        return ops;
    }
};

int DeviceRunnerBase::launch_hbg_kernel_callable(
    int32_t id, const ChipStorageTaskArgs &args, void *caller, const HostApi *api
) try {
    using Launch = HbgKernelLaunchHelpers;
    CallableState *registered_callable = nullptr;
    int rc = Launch::resolve_hbg_callable_for_launch(*this, id, caller, api, registered_callable);
    if (rc != 0) return rc;

    auto &callable_state = *registered_callable->hbg_launch_state;
    uint64_t argument_hash = 0;
    rc = Launch::validate_hbg_launch_arguments(args, callable_state.binding, argument_hash);
    if (rc != 0) return rc;

    Launch::HbgLaunchPreparation preparation;
    rc = Launch::build_hbg_task_graph(*this, args, api, *registered_callable, preparation.workspace);
    if (rc != 0) return rc;
    rc = Launch::encode_hbg_graph_packet(
        *this, id, args, *registered_callable, argument_hash, preparation.workspace, preparation.packet
    );
    if (rc != 0) return rc;
    rc = Launch::pack_hbg_launch_payload(callable_state, preparation.packet, preparation.payload);
    if (rc != 0) return rc;

    if (!preparation.payload.bytes.has_value()) return PTO_RUNTIME_ERR_INTERNAL;
    auto &launch = *preparation.payload.bytes;
    Launch::Submission submission{this, hbg_launch_context_.get(), &callable_state, caller, &launch};
    auto gate = Launch::make_hbg_launch_gate(submission);
    auto ops = Launch::make_hbg_launch_ops(submission);

    // Success means submission, not device completion.
    return kl::launch_bound_kernel({launch.data(), launch.size(), nullptr, 0}, caller, gate, ops).status;
} catch (...) {
    return PTO_RUNTIME_ERR_INTERNAL;
}

int DeviceRunnerBase::revoke_hbg_kernel_callable(int32_t id) {
    const auto live = callables_.find(id);
    if (live == callables_.end() || live->second.hbg_launch_state == nullptr) return 0;
    auto &state = *live->second.hbg_launch_state;
    if (hbg_launch_context_ == nullptr || !hbg_launch_context_->graph_leases.empty() || hbg_launch_context_->uncertain)
        return PTO_RUNTIME_ERR_INVALID_STATE;
    if (hbg_launch_context_->submitted) {
        aclrtEventRecordedStatus status{};
        const int rc = aclrtQueryEventStatus(kernel_exec_state_.event(KernelEventKind::SerialTail), &status);
        if (rc != 0) return rc;
        if (status != ACL_EVENT_RECORDED_STATUS_COMPLETE) return PTO_RUNTIME_ERR_INVALID_STATE;
    }
    auto stream = kernel_exec_state_.hidden_stream(KernelStreamKind::Aicpu);
    if (state.published) {
        int rc = publish_control(
            *this, stream, &state.binding, sizeof(state.binding), "simpler_aicpu_l1_hbg_revoke_callable",
            timeout_config_.stream_sync_timeout_ms
        );
        if (rc != 0) {
            kernel_exec_state_.poison(rc);
            return rc;
        }
        state.published = false;
    }
    const int rc = state.arguments.finalize_once();
    if (rc == 0) live->second.hbg_launch_state.reset();
    return rc;
}

int DeviceRunnerBase::close_hbg_kernel_runtime() {
    if (hbg_launch_context_ == nullptr) return 0;
    if (!hbg_launch_context_->graph_leases.empty()) return PTO_RUNTIME_ERR_INVALID_STATE;
    const int closing_rc = kernel_exec_state_.begin_close();
    if (closing_rc != 0) return closing_rc;
    hbg_launch_context_->closing = true;
    if (hbg_launch_context_->uncertain) {
        const int rc = drain_owned_streams(kernel_exec_state_, timeout_config_.stream_sync_timeout_ms);
        if (rc != 0) return rc;
        hbg_launch_context_->uncertain = false;
        hbg_launch_context_->submitted = false;
    }
    for (auto &entry : callables_) {
        const int rc = revoke_hbg_kernel_callable(entry.first);
        if (rc != 0) return rc;
    }
    if (hbg_launch_context_->registered) {
        auto stream = kernel_exec_state_.hidden_stream(KernelStreamKind::Aicpu);
        hbg::GraphSlotDetachRequest detach{
            hbg_launch_context_->registration.registry.address, hbg_launch_context_->context_generation,
            hbg_launch_context_->runtime_binary_id, device_id_, 0
        };
        int rc = publish_control(
            *this, stream, &detach, sizeof(detach), "simpler_aicpu_l1_hbg_detach_execution_slot",
            timeout_config_.stream_sync_timeout_ms
        );
        if (rc != 0) {
            kernel_exec_state_.poison(rc);
            return rc;
        }
        hbg_launch_context_->registered = false;
    }
    return 0;
}

int DeviceRunnerBase::retain_hbg_kernel_graph(uint64_t *lease) {
    if (lease == nullptr) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    if (hbg_launch_context_ == nullptr || hbg_launch_context_->closing || !kernel_exec_state_.accepts_launch())
        return PTO_RUNTIME_ERR_INVALID_STATE;
    const uint64_t next = allocate_graph_lease();
    if (next == 0) return PTO_RUNTIME_ERR_INVALID_STATE;
    hbg_launch_context_->graph_leases.insert(next);
    *lease = next;
    return 0;
}
int DeviceRunnerBase::release_hbg_kernel_graph(uint64_t lease) {
    if (hbg_launch_context_ == nullptr || hbg_launch_context_->graph_leases.count(lease) == 0)
        return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    if (hbg_launch_context_->graph_leases.size() == 1) {
        // A destroyed graph may never have replayed: its captured tail is not a
        // completion fence. Drain only owned streams after the caller retires
        // the graph, keeping the token live if completion cannot be established.
        const int rc = drain_owned_streams(kernel_exec_state_, timeout_config_.stream_sync_timeout_ms);
        if (rc != 0) return rc;
        hbg_launch_context_->submitted = false;
        hbg_launch_context_->previous_caller_identity = 0;
        hbg_launch_context_->bound_submit_stream = 0;
    }
    hbg_launch_context_->graph_leases.erase(lease);
    return 0;
}
