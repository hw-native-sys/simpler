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
 * Device Runner Implementation
 *
 * This file implements the device execution utilities for launching and
 * managing AICPU and AICore kernels on Ascend devices.
 */

#include "device_runner.h"

#include "acl/acl.h"
#include "host/acl_error_log.h"
#include "host_log.h"
#include "platform_comm/comm.h"
#include "pto_runtime_c_api.h"

#include <dlfcn.h>

#include "aicpu_loader/host/load_aicpu_op.h"

#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "aicpu_topology_probe.h"
#include "callable.h"
#include "callable_protocol.h"
#include "call_config.h"
#include "utils/elf_build_id.h"
#include "utils/fnv1a_64.h"
#include "host/host_regs.h"  // Register address retrieval
#include "host/raii_scope_guard.h"
#include "utils/fatal_shutdown_latch.h"

namespace {
constexpr const char *kAicpuTopologyQueryName = "simpler_aicpu_query_topology";
}

// dep_gen_replay_emit_deps_json: strong symbol provided by
// runtime/tensormap_and_ringbuffer/host/dep_gen_replay.cpp when that runtime is
// linked into host_runtime.so. host_build_graph has no replay implementation
// today, so its host_runtime.so falls through to this weak stub. visibility=
// hidden keeps the stub off the global dynamic symbol table so it can't
// accidentally shadow the strong symbol via RTLD_GLOBAL.
// LOG_DEBUG (not WARN): runtimes that don't link dep_gen never enable it in
// practice, so this path is unreachable for end users — the symbol exists
// purely to keep the .so loadable.
extern "C" __attribute__((weak, visibility("hidden"))) int dep_gen_replay_emit_deps_json(
    const struct DepGenRecord * /*records*/, size_t /*num_records*/, const char * /*deps_json_path*/
) {
    LOG_DEBUG("dep_gen replay not implemented for this runtime — deps.json skipped");
    return -1;
}

// =============================================================================
// DeviceRunner Implementation
// =============================================================================

DeviceRunner::~DeviceRunner() { finalize(); }

// `setup_static_arena`, `create_thread`, `attach_current_thread`,
// `configure_aicore_op_timeout`, `ensure_device_initialized`,
// `ensure_binaries_loaded`, `query_max_block_dim`, and `validate_block_dim`
// live on `DeviceRunnerBase` — see
// `src/common/platform/onboard/host/device_runner_base.cpp`.

// Comm/ACL lifecycle methods are arch-specific (HCCL backend), so they
// stay on DeviceRunner rather than DeviceRunnerBase. Mirrors a2a3 onboard.

int DeviceRunner::ensure_acl_ready(int device_id) {
    if (device_id < 0) {
        LOG_ERROR("ensure_acl_ready: invalid device_id %d", device_id);
        return -1;
    }

    // aclInit is process-wide; CANN returns 100002 if it has already been
    // initialized (possibly by another owner), which we treat as success.
    constexpr int kAclRepeatInit = 100002;
    aclError aRet = aclInit(nullptr);
    if (aRet != ACL_SUCCESS && static_cast<int>(aRet) != kAclRepeatInit) {
        LOG_ERROR("aclInit failed: %d", static_cast<int>(aRet));
        ACL_LOG_ERROR_DETAIL(aRet);
        return static_cast<int>(aRet);
    }

    // ACL device binding is per-thread; every caller must still hit it.
    aRet = aclrtSetDevice(device_id);
    if (aRet != ACL_SUCCESS) {
        LOG_ERROR("aclrtSetDevice(%d) failed: %d", device_id, static_cast<int>(aRet));
        ACL_LOG_ERROR_DETAIL(aRet);
        return static_cast<int>(aRet);
    }

    // Record that we are responsible for aclFinalize at teardown.
    acl_ready_ = true;
    if (device_id_ < 0) device_id_ = device_id;
    return 0;
}

void *DeviceRunner::create_comm_stream() {
    aclrtStream stream = nullptr;
    aclError aRet = aclrtCreateStream(&stream);
    if (aRet != ACL_SUCCESS) {
        LOG_ERROR("aclrtCreateStream failed: %d", static_cast<int>(aRet));
        ACL_LOG_ERROR_DETAIL(aRet);
        return nullptr;
    }
    return stream;
}

int DeviceRunner::destroy_comm_stream(void *stream) {
    if (stream == nullptr) return 0;

    // Best-effort teardown.  HcclBarrier submits async work on the stream;
    // if the caller never blocked for completion (or hit the HCCL 507018
    // barrier regression), aclrtDestroyStream will refuse with 507901
    // ("stream still has pending tasks").  We try to drain first, then
    // destroy anyway, and log failures without propagating them — leaking
    // a stream at teardown is strictly better than failing the teardown
    // itself, which would block device finalization.  This matches the
    // cleanup behavior of the HCCL C++ hardware UT.
    aclError sync_rc = aclrtSynchronizeStream(static_cast<aclrtStream>(stream));
    if (sync_rc != ACL_SUCCESS) {
        LOG_ERROR("aclrtSynchronizeStream during stream teardown failed: %d", static_cast<int>(sync_rc));
    }
    aclError destroy_rc = aclrtDestroyStream(static_cast<aclrtStream>(stream));
    if (destroy_rc != ACL_SUCCESS) {
        LOG_ERROR("aclrtDestroyStream failed (leaking stream): %d", static_cast<int>(destroy_rc));
    }
    return 0;
}

int DeviceRunner::query_aicpu_device_occupancy(pto::a5::AicpuDeviceOccupancy &out) {
    if (aicpu_device_occupancy_cached_) {
        out = aicpu_device_occupancy_;
        return 0;
    }

    void *device_result = mem_alloc_.alloc(sizeof(AicpuTopologyQueryResult));
    if (device_result == nullptr) {
        LOG_ERROR("AICPU topology query result allocation failed");
        return -1;
    }
    auto result_cleanup = RAIIScopeGuard([&]() {
        mem_alloc_.free(device_result);
    });
    AicpuTopologyQueryResult zero{};
    int rc = rtMemcpy(device_result, sizeof(zero), &zero, sizeof(zero), RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        LOG_ERROR("AICPU topology query result initialization failed: %d", rc);
        return rc;
    }
    AicpuTopologyQueryArgs args{};
    args.result_addr = reinterpret_cast<uint64_t>(device_result);
    rc = launch_aicpu_payload(stream_aicpu_, &args, sizeof(args), kAicpuTopologyQueryName, /*aicpu_num=*/1);
    if (rc != 0) {
        LOG_ERROR("AICPU device occupancy query launch failed: %d", rc);
        recover_device_or_mark_unusable(rc);
        return rc;
    }
    rc = aclrtSynchronizeStreamWithTimeout(stream_aicpu_, PLATFORM_STREAM_SYNC_TIMEOUT_MS);
    if (rc != 0) {
        LOG_ERROR("AICPU device occupancy query sync failed: %d", rc);
        recover_device_or_mark_unusable(rc);
        return rc;
    }
    AicpuTopologyQueryResult result{};
    rc = rtMemcpy(&result, sizeof(result), device_result, sizeof(result), RT_MEMCPY_DEVICE_TO_HOST);
    if (rc != 0) {
        LOG_ERROR("AICPU device occupancy query copy failed: %d", rc);
        return rc;
    }
    if (result.occupy_rc != 0 || result.occupy == 0) {
        LOG_ERROR(
            "device-side AICPU OCCUPY query failed: rc=%d mask=0x%llx", result.occupy_rc,
            static_cast<unsigned long long>(result.occupy)
        );
        return -1;
    }
    aicpu_device_occupancy_.occupy = result.occupy;
    aicpu_device_occupancy_.pf_occupy = result.pf_occupy;
    aicpu_device_occupancy_.os_sched = result.os_sched;
    aicpu_device_occupancy_.occupy_valid = result.occupy_rc == 0;
    aicpu_device_occupancy_.pf_occupy_valid = result.pf_occupy_rc == 0;
    aicpu_device_occupancy_.os_sched_valid = result.os_sched_rc == 0;
    aicpu_device_occupancy_cached_ = true;
    out = aicpu_device_occupancy_;
    return 0;
}

int DeviceRunner::query_aicpu_topology(pto::a5::AicpuTopology &out) {
    if (aicpu_topology_cached_) {
        out = aicpu_topology_;
        return 0;
    }

    pto::a5::AicpuDeviceOccupancy occupancy;
    int rc = query_aicpu_device_occupancy(occupancy);
    if (rc != 0) return rc;

    pto::a5::AicpuTopology topology;
    if (!pto::a5::probe_aicpu_topology(static_cast<uint32_t>(device_id_), occupancy, topology)) return -1;

    aicpu_topology_ = std::move(topology);
    aicpu_topology_cached_ = true;
    out = aicpu_topology_;
    return 0;
}

void DeviceRunner::clear_aicpu_topology_cache() {
    aicpu_device_occupancy_cached_ = false;
    aicpu_device_occupancy_ = {};
    aicpu_topology_cached_ = false;
    aicpu_topology_ = {};
}

int DeviceRunner::prepare_execution(
    Runtime &runtime, const CallConfig &config, uint32_t pipeline_slot, const NativeRunIdentity &identity,
    std::unique_ptr<PreparedExecution> *prepared
) {
    if (prepared == nullptr || *prepared != nullptr) return -1;
    auto execution = std::make_unique<PreparedExecution>(identity, runtime, config, pipeline_slot);
    execution->resources_owned = true;
    auto prepare_rollback = RAIIScopeGuard([this, &execution]() {
        cleanup_execution(*execution, /*launched=*/false);
    });
    const int block_dim = runtime.get_worker_count() / cores_per_blockdim_;
    int requested_aicpu_num = config.aicpu_thread_num;
    const bool automatic_aicpu_num = requested_aicpu_num == 0;
    // A prior AICore launch/sync error poisoned the device context and the
    // in-place drain could not clear it. Refuse to run rather than cascade
    // into halResMap rc=62 (init_aicore_register_addresses) or rtMalloc
    // 507899. A soft close()+reset does NOT clear the poison on a5, but
    // finalize() force-resets the card on this path so the next Worker re-inits
    // clean in the same process (see force_reset_device()). Failing fast here
    // turns the rest of an xdist worker session's tests from a slow, confusing
    // failure cascade into a single fast, self-explanatory error; the runner is
    // then recovered at finalize.
    if (device_unusable_.load(std::memory_order_acquire)) {
        LOG_ERROR(
            "DeviceRunner marked unusable by a prior AICore failure; refusing to enqueue. "
            "A soft reset does not clear the poison on a5; finalize() will force-reset "
            "the card so the next Worker on it inits clean."
        );
        return -1;
    }
    if (validate_launch_aicpu_num(requested_aicpu_num) != 0) return -1;
    int active_aicpu_num = automatic_aicpu_num ? PLATFORM_DEFAULT_AICPU_THREAD_NUM : requested_aicpu_num;
    runtime.set_aicpu_thread_num(active_aicpu_num);

    int rc = ensure_device_initialized();
    if (rc != 0) {
        LOG_ERROR("ensure_device_initialized failed: %d", rc);
        return rc;
    }

    ensure_device_wall_buffer(execution->kernel_args);

    if (block_dim < 1) {
        LOG_ERROR("prepare_execution computed block_dim < 1 from worker_count=%d", runtime.get_worker_count());
        return -1;
    }
    int num_aicore = block_dim * cores_per_blockdim_;

    rc = init_aicore_register_addresses(
        &execution->kernel_args.args.regs, static_cast<uint64_t>(device_id_), mem_alloc_
    );
    if (rc != 0) {
        LOG_ERROR("init_aicore_register_addresses failed: %d", rc);
        return rc;
    }

    // Build the profiling-flag bitfield.
    uint32_t enable_profiling_flag = SIMPLER_DFX_FLAG_NONE;
    if (enable_dump_args_) SIMPLER_SET_DFX_FLAG(enable_profiling_flag, SIMPLER_DFX_FLAG_DUMP_ARGS);
    if (enable_chip_swimlane_) SIMPLER_SET_DFX_FLAG(enable_profiling_flag, SIMPLER_DFX_FLAG_CHIP_SWIMLANE);
    if (enable_pmu_) SIMPLER_SET_DFX_FLAG(enable_profiling_flag, SIMPLER_DFX_FLAG_PMU);
    if (enable_dep_gen_) SIMPLER_SET_DFX_FLAG(enable_profiling_flag, SIMPLER_DFX_FLAG_DEP_GEN);
    if (enable_scope_stats_ && !scope_stats_uses_host_capture()) {
        SIMPLER_SET_DFX_FLAG(enable_profiling_flag, SIMPLER_DFX_FLAG_SCOPE_STATS);
    }
    execution->kernel_args.args.enable_profiling_flag = enable_profiling_flag;

    resolve_task_binary_addrs(runtime);

    // a5-specific: probe the AICPU topology + compute ALLOWED_CPUS for the
    // filter-style gate (see src/common/platform/onboard/aicpu/
    // platform_aicpu_affinity.cpp::platform_aicpu_affinity_gate_filter).
    // Convention: indices 0..active-2 are scheduler slots and the last slot
    // is the orchestrator. In auto mode only, unknown shapes may reduce the
    // active count to the available pool, but execution keeps at least one of
    // each role.
    {
        pto::a5::AicpuTopology topology;
        runtime.set_aicpu_allowed_cpu_count(0);
        if (query_aicpu_topology(topology) != 0) {
            LOG_ERROR("AICPU topology probe failed; affinity gate will not launch");
            return -1;
        }
        pto::a5::AicpuLaunchPlan launch_plan;
        std::string plan_error;
        if (!pto::a5::build_aicpu_launch_plan(topology, requested_aicpu_num, launch_plan, plan_error)) {
            LOG_ERROR(
                "cannot build AICPU launch plan: soc=%s scenario=%s occupy=0x%llx reason=%s",
                topology.soc_name.empty() ? "(unknown)" : topology.soc_name.c_str(),
                pto::a5::aicpu_scenario_name(topology.scenario_type),
                static_cast<unsigned long long>(topology.device_occupancy.occupy), plan_error.c_str()
            );
            return -1;
        }
        const auto &allowed = launch_plan.allowed_cpus;
        active_aicpu_num = launch_plan.effective_active_count;
        runtime.set_aicpu_thread_num(active_aicpu_num);
        {
            const size_t cap = runtime.aicpu_allowed_cpus_capacity();
            if (allowed.size() > cap) {
                LOG_ERROR("AICPU selection returned %zu > cap %zu", allowed.size(), cap);
                return -1;
            }
            int32_t *allowed_cpus = runtime.get_aicpu_allowed_cpus();
            for (size_t i = 0; i < allowed.size(); ++i)
                allowed_cpus[i] = allowed[i];
            runtime.set_aicpu_allowed_cpu_count(static_cast<int32_t>(allowed.size()));
            runtime.set_aicpu_launch_count(launch_plan.launch_count);
            std::string dump;
            for (size_t i = 0; i < allowed.size(); ++i) {
                if (i) dump += ", ";
                dump += std::to_string(allowed[i]);
                if (i + 1 == allowed.size()) dump += "(orch)";
            }
            if (launch_plan.warn_cpu_topology_unavailable) {
                LOG_WARN(
                    "AICPU CPU_TOPO unavailable; using %s: soc=%s occupy=0x%llx "
                    "stable_reachable=%d requested=%d effective=%d affinity=[%s]%s",
                    pto::a5::aicpu_topology_source_name(topology.source),
                    topology.soc_name.empty() ? "(unknown)" : topology.soc_name.c_str(),
                    static_cast<unsigned long long>(topology.device_occupancy.occupy),
                    launch_plan.stable_reachable_count, requested_aicpu_num, active_aicpu_num, dump.c_str(),
                    topology.source == pto::a5::AicpuTopologySource::kOccupyFallback ?
                        "; physical/SMT/cluster/die placement is unknown" :
                        ""
                );
            }
            if (launch_plan.warn_stable_reachable_below_default) {
                LOG_WARN(
                    "AICPU stable reachable CPUs below active capacity: soc=%s scenario=%s occupy=0x%llx "
                    "stable_reachable=%d capacity=%d requested=%d effective=%d affinity=[%s]",
                    topology.soc_name.empty() ? "(unknown)" : topology.soc_name.c_str(),
                    pto::a5::aicpu_scenario_name(topology.scenario_type),
                    static_cast<unsigned long long>(topology.device_occupancy.occupy),
                    launch_plan.stable_reachable_count, PLATFORM_DEFAULT_AICPU_THREAD_NUM, requested_aicpu_num,
                    active_aicpu_num, dump.c_str()
                );
            }
            LOG_INFO(
                "AICPU ALLOWED_CPUS = [%s] (scenario=%s active=%d launch=%d user_cpus=%zu)", dump.c_str(),
                pto::a5::aicpu_scenario_name(topology.scenario_type), active_aicpu_num, launch_plan.launch_count,
                topology.os_schedulable_cpus.size()
            );
        }
    }

    // Initialize per-subsystem shared memory.
    if (enable_chip_swimlane_) {
        rc = init_chip_swimlane(num_aicore, runtime.get_aicpu_thread_num(), device_id_, execution->kernel_args);
        if (rc != 0) {
            LOG_ERROR("init_chip_swimlane failed: %d", rc);
            return rc;
        }
    }

    if (enable_dump_args_) {
        rc = init_args_dump(runtime, device_id_, execution->kernel_args);
        if (rc != 0) {
            LOG_ERROR("init_args_dump failed: %d", rc);
            return rc;
        }
    }

    if (enable_pmu_) {
        rc = init_pmu(
            num_aicore, active_aicpu_num, make_pmu_csv_path(output_prefix_), pmu_event_type_, device_id_,
            execution->kernel_args
        );
        if (rc != 0) {
            LOG_ERROR("PMU init failed: %d, disabling PMU for this run", rc);
            execution->kernel_args.args.pmu_data_base = 0;
            enable_pmu_ = false;
        }
    }

    if (enable_dep_gen_) {
        rc = init_dep_gen(active_aicpu_num, device_id_, execution->kernel_args);
        if (rc != 0) {
            LOG_ERROR("init_dep_gen failed: %d", rc);
            return rc;
        }
    }

    if (enable_scope_stats_ && !scope_stats_uses_host_capture()) {
        rc = init_scope_stats(active_aicpu_num, device_id_, execution->kernel_args);
        if (rc != 0) {
            LOG_ERROR("init_scope_stats failed: %d", rc);
            return rc;
        }
    }

    rc = prepare_orch_so(runtime);
    if (rc != 0) {
        LOG_ERROR("prepare_orch_so failed: %d", rc);
        return rc;
    }
    rc = init_runtime_args_with_metadata(runtime, execution->kernel_args);
    if (rc != 0) return rc;

    rc = execution->kernel_args.init_device_kernel_args(mem_alloc_);
    if (rc != 0) {
        LOG_ERROR("init_device_kernel_args failed: %d", rc);
        return rc;
    }
    execution->num_aicore = num_aicore;
    execution->launch_aicpu_num = active_aicpu_num;
    prepare_rollback.dismiss();
    *prepared = std::move(execution);
    return 0;
}

DeviceRunnerBase::LaunchOutcome
DeviceRunner::launch_execution(std::unique_ptr<PreparedExecution> prepared, LaunchPermit permit) {
    LaunchOutcome outcome;
    if (prepared == nullptr) return outcome;

    Runtime &runtime = *prepared->runtime;
    const int num_aicore = prepared->num_aicore;
    const int launch_aicpu_num = prepared->launch_aicpu_num;

    LaunchTransactionResult transaction = exact_launch_transaction(
        prepared->identity, std::move(permit),
        [&]() {
            // Arming precedes any execution-visible submission, so its failures —
            // including a thread-spawn or allocation throw — are reported as an rc
            // and leave the run safely rollback-able.
            try {
                activate_launch_shape(runtime);
                (void)arm_device_wall_buffer(prepared->kernel_args);
                start_shared_collectors_for_run();
                if (enable_dep_gen_) {
                    auto thread_factory = [this](std::function<void()> fn) {
                        return create_thread(std::move(fn));
                    };
                    dep_gen_collector_.start(thread_factory);
                }
                if (enable_chip_swimlane_ && chip_swimlane_collector_.is_initialized()) {
                    std::vector<CoreType> core_types(num_aicore);
                    for (int i = 0; i < num_aicore; i++)
                        core_types[i] = runtime.get_workers()[i].core_type;
                    chip_swimlane_collector_.set_core_types(core_types.data(), num_aicore);
                }

                // Launch the AICore worker BEFORE the AICPU Run task. This is a first-launch
                // latency optimization, not a correctness requirement (the handshake is
                // launch-order-independent). When the AICPU Run task is launched first it
                // immediately occupies the device (spinning in handshake_all_cores), and the
                // first AICore launch — which lazily loads the kernel binary onto the device
                // inside rtKernelLaunchWithHandleV2 — then takes ~1.4 s instead of ~0.4 ms
                // (measured a5; the exact device-side contention is not pinned, see the
                // investigation doc). Submitting the AICore first does that load on an idle
                // device, then the AICPU spins and finds the AICore already up.
                //
                // Defense-in-depth for the op-timeout family (#1019): that ~1.4 s slow launch
                // is what trips the op-execute timeout when it is tight. #1035 widened the
                // timeout 1 s -> 3 s so the slow launch no longer wedges, but this ordering
                // removes the slow launch itself, so the wedge cannot return if the timeout
                // is ever tightened or a slower device pushes the launch past it. See
                // docs/investigations/2026-06-pa-unroll-207001-optimeout-window.md.
                // The AICore publishes aicore_done on launch (gated by nothing), and the
                // workers region persists across runs in the pooled arena. Clearing each
                // worker's aicore_done before the AICore kernel launches keeps the AICPU's
                // handshake sweep from reading a prior run's report — which would open a
                // window on that run's physical_core_id. Only aicore_done needs clearing; the
                // AICore overwrites physical_core_id/core_type in the same report.
                Handshake *workers = runtime.get_workers();
                for (int i = 0; i < num_aicore; i++)
                    workers[i].aicore_done = 0;
            } catch (...) {
                LOG_ERROR("launch_execution: arming failed before any stream submission");
                return -1;
            }

            run_poll_slot_.store(prepared->pipeline_slot, std::memory_order_relaxed);
            run_poll_state_.store(RunPollState::Enqueuing, std::memory_order_release);
            LOG_INFO("=== launch_aicore_kernel ===");
            run_poll_state_.store(RunPollState::Submitted, std::memory_order_release);
            int launch_rc = launch_aicore_kernel(stream_aicore_, prepared->kernel_args.device_k_args_);
            if (launch_rc != 0) {
                LOG_ERROR("launch_aicore_kernel failed: %d", launch_rc);
                recover_device_or_mark_unusable(launch_rc);
            }
            return launch_rc;
        },
        [&]() {
            LOG_INFO("=== launch_aicpu_kernel %s ===", host::KernelNames::RunName);
            // launch_count = popcount(OCCUPY) from the topology probe — one thread
            // per user-schedulable cpu_id. The filter gate barriers exactly this
            // many threads (runtime.aicpu_launch_count is read on the device side
            // by kernel.cpp). The fallback is defensive; normal runs always publish
            // the topology-derived launch count above.
            int aicpu_launch_n =
                (runtime.get_aicpu_launch_count() > 0) ? runtime.get_aicpu_launch_count() : launch_aicpu_num;
            int launch_rc = launch_aicpu_kernel(
                stream_aicpu_, &prepared->kernel_args.args, host::KernelNames::RunName, aicpu_launch_n
            );
            if (launch_rc != 0) {
                LOG_ERROR("launch_aicpu_kernel (main) failed: %d", launch_rc);
            }
            return launch_rc;
        }
    );

    if (transaction.poisoned()) recover_device_or_mark_unusable(transaction.rc);
    outcome.rc = transaction.rc;
    outcome.progress = transaction.progress;
    outcome.receipt = std::move(transaction.receipt);
    if (transaction.progress == LaunchProgress::NotStarted) {
        outcome.prepared = std::move(prepared);
    } else {
        outcome.active = std::make_unique<ActiveExecution>(std::move(prepared), transaction.progress);
    }
    return outcome;
}

int DeviceRunner::poll_execution(const ActiveExecution &active) {
    if (active.prepared == nullptr) return SIMPLER_NATIVE_RUN_POLL_ERROR;
    const uint32_t pipeline_slot = active.prepared->pipeline_slot;
    const RunPollState state = run_poll_state_.load(std::memory_order_acquire);
    if (run_poll_slot_.load(std::memory_order_relaxed) != pipeline_slot) {
        return SIMPLER_NATIVE_RUN_POLL_ERROR;
    }
    if (state == RunPollState::DeviceComplete || state == RunPollState::Drained) {
        return SIMPLER_NATIVE_RUN_POLL_COMPLETE;
    }
    if (state != RunPollState::Submitted) return SIMPLER_NATIVE_RUN_POLL_ERROR;

    const int rc = query_stream_pair_nonblocking(stream_aicpu_, stream_aicore_);
    if (rc == SIMPLER_NATIVE_RUN_POLL_COMPLETE) {
        RunPollState expected = RunPollState::Submitted;
        (void)run_poll_state_.compare_exchange_strong(
            expected, RunPollState::DeviceComplete, std::memory_order_acq_rel, std::memory_order_acquire
        );
    }
    return rc;
}

int DeviceRunner::drain_execution(ActiveExecution &active) {
    if (active.prepared == nullptr) return -1;
    PreparedExecution &prepared = *active.prepared;
    const uint32_t pipeline_slot = prepared.pipeline_slot;
    if (!prepared.resources_owned || run_poll_slot_.load(std::memory_order_relaxed) != pipeline_slot) {
        LOG_ERROR(
            "drain_execution slot mismatch: requested=%u active=%u owns=%d", pipeline_slot,
            run_poll_slot_.load(std::memory_order_relaxed), static_cast<int>(prepared.resources_owned)
        );
        return -1;
    }
    auto drain_cleanup = RAIIScopeGuard([this, &prepared]() {
        cleanup_execution(prepared, /*launched=*/true);
    });

    int rc = sync_run_streams();
    if (rc != 0) {
        // sync_run_streams surfaces the AICore op-timeout (STARS-reaped op ->
        // 507000/507018/507046 at AICPU/AICore stream sync). The op-timeout
        // leaves the context poisoned, so recovery remains the drain owner's
        // responsibility and its error remains authoritative over cleanup.
        recover_device_or_mark_unusable(rc);
        // Emergency shutdown may already have flushed diagnostics. Export the
        // manifest on the error path exactly once.
        teardown_shared_collectors_after_run(false);
        return rc;
    }

    read_device_wall_ns();
    teardown_shared_collectors_after_run(true);

    // a5-specific dep_gen teardown: stop + reconcile + replay emit.
    if (enable_dep_gen_) {
        dep_gen_collector_.stop();
        if (dep_gen_collector_.reconcile_counters()) {
            const auto &records = dep_gen_collector_.records();
            const std::string deps = make_deps_json_path(output_prefix_);
            int replay_rc = dep_gen_replay_emit_deps_json(records.data(), records.size(), deps.c_str());
            if (replay_rc != 0) {
                LOG_ERROR("dep_gen replay failed (%d) — deps.json not produced", replay_rc);
            }
        }
    }

    // Reads device memory, so it must precede KernelArgs/runtime cleanup.
    print_handshake_results(prepared.kernel_args);
    return 0;
}

void DeviceRunner::cleanup_execution(PreparedExecution &prepared, bool launched) noexcept {
    if (!prepared.resources_owned) return;

    // A poisoned card cannot retire per-resource frees, and issuing them can
    // block in the driver. Drop host-side ownership instead; finalize()'s force
    // reset invalidates the whole device generation.
    const bool abandon = device_unusable_.load(std::memory_order_acquire);

    // Collectors stop before device/runtime arguments and register buffers.
    finalize_collectors(abandon);
    if (abandon) {
        prepared.kernel_args.abandon_after_device_failure();
    } else {
        (void)prepared.kernel_args.finalize_device_kernel_args();
        (void)prepared.kernel_args.finalize_runtime_args();
    }
    if (prepared.kernel_args.args.regs != 0) {
        if (!abandon) {
            (void)mem_alloc_.free(reinterpret_cast<void *>(prepared.kernel_args.args.regs));
        }
        prepared.kernel_args.args.regs = 0;
    }
    prepared.resources_owned = false;
    if (launched) run_poll_state_.store(RunPollState::Drained, std::memory_order_release);
}

void DeviceRunner::abandon_prepared_execution(PreparedExecution &prepared) noexcept {
    cleanup_execution(prepared, /*launched=*/false);
}

void DeviceRunner::recover_device_or_mark_unusable(int aicore_rc) {
    // An AICore launch failure (207001) or an op-timeout reaped by STARS
    // (surfaced as 507000/507018/507046 at stream sync) leaves the device
    // context in a sticky-error state: the streams stay poisoned and the
    // SAME DeviceRunner's next enqueue fails early — observed on a5 as
    // `halResMap failed (rc=62)` in init_aicore_register_addresses, and on
    // a2a3 as `rtMalloc failed: 507899`. Reused across a session (the L2
    // st_worker pool hands one ChipWorker to every test class on a device),
    // that one error poisons every later test in the xdist worker process.
    //
    // Best-effort BOUNDED drain (aclrtSynchronizeDeviceWithTimeout, NOT an
    // unbounded aclrtSynchronizeStream* on the error-state stream — the latter
    // wedges subsequent tests, see DeviceRunnerBase::finalize_common). But DO
    // NOT gate recovery on the drain's rc: the bounded drain can return success
    // on a still-poisoned card — a false-negative that leaves force_reset_device()
    // untriggered (observed cascading into skipped L2 cases on a2a3 CI run
    // 27742754024; the same gate exists here). The op-timeout sticky-error is
    // only cleared by a force reset (a soft reset/drain does not), so always mark
    // the runner unusable here: admission fails fast and finalize() force-resets
    // the card, so the next Worker.init lands clean regardless of the drain result.
    int sync_rc = aclrtSynchronizeDeviceWithTimeout(timeout_config_.stream_sync_timeout_ms);
    if (sync_rc != ACL_SUCCESS) {
        LOG_ERROR(
            "AICore error %d: bounded device drain failed: %d (force reset will follow in finalize)", aicore_rc, sync_rc
        );
    } else {
        LOG_WARN(
            "AICore error %d: device drained, but force-resetting in finalize regardless "
            "(drain success does not prove the card is clean)",
            aicore_rc
        );
    }
    clear_aicpu_topology_cache();
    device_unusable_.store(true, std::memory_order_release);
}

namespace {

// RAII: bring ACL up if needed, and finalize on scope exit ONLY if this guard
// is the one that initialized it. On the L2 poison path the rt-layer runner
// never brought ACL up (acl_ready_ is false, so finalize() did a bare
// rtDeviceReset and no aclFinalize), so aclInit here genuinely initializes ACL
// and we own its teardown. If some other owner already init'd ACL, aclInit
// returns 100002 and we leave it alone — finalizing it would tear ACL down for
// the rest of the process.
class AclInitGuard {
public:
    AclInitGuard() {
        constexpr int kAclRepeatInit = 100002;
        aclError rc = aclInit(nullptr);
        if (rc == ACL_SUCCESS) {
            owns_ = true;
            ok_ = true;
        } else if (static_cast<int>(rc) == kAclRepeatInit) {
            ok_ = true;
        } else {
            LOG_ERROR("force_reset_device: aclInit failed: %d", static_cast<int>(rc));
        }
    }
    ~AclInitGuard() {
        if (owns_) {
            (void)aclFinalize();
        }
    }
    AclInitGuard(const AclInitGuard &) = delete;
    AclInitGuard &operator=(const AclInitGuard &) = delete;
    bool ok() const { return ok_; }

private:
    bool owns_{false};
    bool ok_{false};
};

// RAII: bind the device to this thread for the force reset, and unbind it on
// scope exit so the per-thread device reference does not leak into the next
// Worker on this card.
class DeviceBindGuard {
public:
    explicit DeviceBindGuard(int device_id) :
        device_id_(device_id) {
        aclError rc = aclrtSetDevice(device_id_);
        if (rc == ACL_SUCCESS) {
            bound_ = true;
        } else {
            LOG_ERROR("force_reset_device: aclrtSetDevice(%d) failed: %d", device_id_, static_cast<int>(rc));
        }
    }
    ~DeviceBindGuard() {
        if (bound_) {
            (void)aclrtResetDevice(device_id_);
        }
    }
    DeviceBindGuard(const DeviceBindGuard &) = delete;
    DeviceBindGuard &operator=(const DeviceBindGuard &) = delete;
    bool bound() const { return bound_; }

private:
    int device_id_;
    bool bound_{false};
};

}  // namespace

int DeviceRunner::force_reset_device() {
    clear_aicpu_topology_cache();
    if (device_id_ < 0) {
        return -1;
    }
    // aclrtResetDeviceForce is an ACL API; bring ACL up for the whole sequence,
    // released on scope exit so a repeated poison-then-reset cycle in a
    // long-lived process leaks no ACL state.
    AclInitGuard acl_guard;
    if (!acl_guard.ok()) {
        LOG_ERROR("force_reset_device: ACL init failed; cannot reset device %d", device_id_);
        return -1;
    }
    {
        // Reset phase. Bind the device, best-effort drain (the op-timeout
        // sticky-error sometimes settles with a drain first) *inside* this valid
        // ACL/bound context — finalize() may have already torn ACL down via
        // aclFinalize/rtDeviceReset, so the drain must live here, not in the
        // caller — then force-reset. The bind is released at the end of this
        // block so the probe below holds the only active device reference.
        DeviceBindGuard bind_guard(device_id_);
        if (!bind_guard.bound()) {
            LOG_ERROR("force_reset_device: could not bind device %d; reset skipped", device_id_);
            return -1;
        }
        (void)aclrtSynchronizeDeviceWithTimeout(timeout_config_.stream_sync_timeout_ms);
        aclError rc = aclrtResetDeviceForce(device_id_);
        if (rc != ACL_SUCCESS) {
            LOG_ERROR("force_reset_device: aclrtResetDeviceForce(%d) failed: %d", device_id_, static_cast<int>(rc));
            return static_cast<int>(rc);
        }
    }
    // Post-reset self-check: a 0 rc from aclrtResetDeviceForce does not by itself
    // prove the card is usable. Re-bind (fresh guard, balanced on exit) and
    // exercise both poison surfaces — a trivial stream create/destroy (a5 poison:
    // rtStreamCreate / halResMap during Worker.init) and an HBM alloc/free (a2a3
    // poison: rtMalloc 507899) — checking every rc so any failure (incl. the
    // frees) returns non-zero and finalize() keeps the card flagged for the layer
    // above (st_worker poison-skip + #1110 dispatcher retry). The stream probe is
    // the one that exercises a5's actual failing path; this mirror is unvalidated
    // locally (no a5 silicon) — st-onboard-a5 CI is the a5 channel.
    DeviceBindGuard probe_bind(device_id_);
    if (!probe_bind.bound()) {
        LOG_ERROR("force_reset_device: post-reset DeviceBindGuard failed for device %d", device_id_);
        return -1;
    }
    aclrtStream probe_stream = nullptr;
    aclError stream_rc = aclrtCreateStream(&probe_stream);
    if (stream_rc != ACL_SUCCESS) {
        LOG_ERROR(
            "force_reset_device: post-reset probe aclrtCreateStream on device %d failed: %d (card still poisoned)",
            device_id_, static_cast<int>(stream_rc)
        );
        return static_cast<int>(stream_rc);
    }
    aclError destroy_rc = aclrtDestroyStream(probe_stream);
    if (destroy_rc != ACL_SUCCESS) {
        LOG_ERROR(
            "force_reset_device: post-reset probe aclrtDestroyStream on device %d failed: %d", device_id_,
            static_cast<int>(destroy_rc)
        );
        return static_cast<int>(destroy_rc);
    }
    void *probe_ptr = nullptr;
    int probe_rc = rtMalloc(&probe_ptr, 64, RT_MEMORY_HBM, 0);
    if (probe_rc != 0) {
        LOG_ERROR(
            "force_reset_device: post-reset probe rtMalloc on device %d failed: %d (card still poisoned)", device_id_,
            probe_rc
        );
        return probe_rc;
    }
    int free_rc = rtFree(probe_ptr);
    if (free_rc != 0) {
        LOG_ERROR("force_reset_device: post-reset probe rtFree on device %d failed: %d", device_id_, free_rc);
        return free_rc;
    }
    LOG_WARN(
        "force_reset_device: aclrtResetDeviceForce(%d) cleared the poisoned card (probe confirmed clean)", device_id_
    );
    return 0;
}

// `print_handshake_results`, `prepare_orch_so`, `register_callable`,
// `record_host_orch_callable`, `unregister_callable`, `has_callable`,
// `bind_callable_to_runtime`, and `upload_chip_callable_buffer` live on
// `DeviceRunnerBase`.

int DeviceRunner::finalize() {
    if (device_id_ == -1) {
        return 0;
    }

    // Fatal cleanup must not walk poisoned streams, mappings, or allocations.
    // Stop collector threads locally, drain and force-reset the card, then
    // forget the old generation's handles.
    if (device_unusable_.load(std::memory_order_acquire)) {
        finalize_collectors(true);

        // force_reset_device() drains before it resets and returns 0 only when
        // its post-reset probe confirms the card, so a second pass runs against
        // a settled card and can recover a poison the first pass could not. A
        // Worker holding CP-process SDMA streams gets a single attempt: there a
        // non-confirming reset already blocks on the driver's remote-event
        // timeout, which a retry only multiplies. Read dma_workspace_handle_
        // before abandon_common_after_device_failure() clears it.
        constexpr int kFatalResetAttempts = 3;
        const bool sdma_provisioned = dma_workspace_handle_ != nullptr;
        int reset_rc = attempt_fatal_reset(
            [this]() {
                return force_reset_device();
            },
            sdma_provisioned ? 1 : kFatalResetAttempts
        );
        const bool reset_confirmed = reset_rc == 0;
        if (!reset_confirmed) {
            LOG_ERROR(
                "Fatal teardown: force reset of device %d did not confirm clean (rc=%d); "
                "quarantining old handles without per-resource RTS calls",
                device_id_, reset_rc
            );
        }

        int abandon_rc = abandon_common_after_device_failure();
        if (acl_ready_) {
            if (reset_confirmed) {
                int finalize_rc = aclFinalize();
                if (finalize_rc != 0) {
                    LOG_ERROR("aclFinalize failed during fatal finalize: %d", finalize_rc);
                    if (abandon_rc == 0) abandon_rc = finalize_rc;
                }
            } else {
                LOG_WARN("Fatal teardown: skipping aclFinalize because device reset was not confirmed");
            }
            acl_ready_ = false;
        }

        device_id_ = -1;
        if (reset_confirmed) {
            device_unusable_.store(false, std::memory_order_release);
        }
        LOG_WARN("DeviceRunner finalized after fatal device failure");
        return abandon_rc != 0 ? abandon_rc : reset_rc;
    }

    int rc = attach_current_thread(device_id_);
    if (rc != 0) {
        LOG_ERROR("Failed to attach finalize thread to device %d: %d", device_id_, rc);
        return rc;
    }

    // Cleanup all profiling subsystems (free shm + per-buffer dev/host
    // shadows). Normally already done by drain or enqueue rollback; this is the
    // backstop for the no-run-since-init case.
    finalize_collectors();

    // Shared cleanup body — streams, kernel_args, callable/orch maps,
    // chip-callable buffer pool, the three arenas, device_wall,
    // mem_alloc_.finalize(), and cached arena sizes.
    rc = finalize_common();

    // Reset device and finalize ACL AFTER all device memory is freed. When the
    // ACL layer was brought up (comm path), aclrtResetDevice supersedes
    // rtDeviceReset and additionally releases ACL's per-thread ref-count;
    // calling raw rtDeviceReset in that state would leave ACL with stale
    // bookkeeping. Pure rt-layer runtimes that never asked for ACL still get
    // the bare rtDeviceReset.
    if (acl_ready_ && device_id_ >= 0) {
        int reset_rc = aclrtResetDevice(device_id_);
        if (reset_rc != 0) {
            LOG_ERROR("aclrtResetDevice(%d) failed during finalize: %d", device_id_, reset_rc);
            if (rc == 0) rc = reset_rc;
        }
        int finalize_rc = aclFinalize();
        if (finalize_rc != 0) {
            LOG_ERROR("aclFinalize failed during finalize: %d", finalize_rc);
            if (rc == 0) rc = finalize_rc;
        }
        acl_ready_ = false;
    } else {
        int reset_rc = rtDeviceReset(device_id_);
        if (reset_rc != 0) {
            LOG_ERROR("rtDeviceReset(%d) failed during finalize: %d", device_id_, reset_rc);
            if (rc == 0) rc = reset_rc;
        }
    }

    // Only the healthy path reaches here: a poisoned card returned from the
    // fatal branch at the top of finalize(), which owns the force reset.
    clear_aicpu_topology_cache();
    device_id_ = -1;
    device_unusable_.store(false, std::memory_order_release);
    return rc;
}

// `launch_aicpu_kernel` and `launch_aicore_kernel` live on `DeviceRunnerBase`.

void DeviceRunner::finalize_collectors(bool abandon_device_resources) {
    // On drain or enqueue rollback, release the diagnostics collectors' shared
    // memory. They are only re-initialized per run, so a
    // Worker reused across runs (e.g. a pytest session-scoped worker pool) would
    // otherwise re-enter init_chip_swimlane() with stale state still allocated.
    // Matches a2a3's finalize_collectors().
    auto free_cb = [this, abandon_device_resources](void *dev_ptr) -> int {
        if (abandon_device_resources) return 0;
        return mem_alloc_.free(dev_ptr);
    };
    if (chip_swimlane_collector_.is_initialized()) {
        chip_swimlane_collector_.finalize(/*unregister_cb=*/nullptr, free_cb);
    }
    if (dump_collector_.is_initialized()) {
        dump_collector_.finalize(/*unregister_cb=*/nullptr, free_cb);
    }
    if (pmu_collector_.is_initialized()) {
        pmu_collector_.finalize(/*unregister_cb=*/nullptr, free_cb);
    }
    if (dep_gen_collector_.is_initialized()) {
        dep_gen_collector_.finalize(/*unregister_cb=*/nullptr, free_cb);
    }
    if (scope_stats_collector_.is_initialized()) {
        scope_stats_collector_.finalize(/*unregister_cb=*/nullptr, free_cb);
    }
}

int DeviceRunner::init_chip_swimlane(
    int num_aicore, int aicpu_thread_num, int device_id, KernelArgsHelper &kernel_args
) {
    auto alloc_cb = [this](size_t size) -> void * {
        return mem_alloc_.alloc(size);
    };
    auto free_cb = [this](void *dev_ptr) -> int {
        return mem_alloc_.free(dev_ptr);
    };
    int rc = chip_swimlane_collector_.initialize(
        num_aicore, aicpu_thread_num, device_id, chip_swimlane_level_, alloc_cb,
        /*register_cb=*/nullptr, free_cb, output_prefix_
    );
    if (rc == 0) {
        kernel_args.args.chip_swimlane_data_base =
            reinterpret_cast<uint64_t>(chip_swimlane_collector_.get_chip_swimlane_setup_device_ptr());
        kernel_args.args.chip_swimlane_aicore_rotation_table =
            reinterpret_cast<uint64_t>(chip_swimlane_collector_.get_aicore_ring_addr_table_device_ptr());
    }
    return rc;
}

int DeviceRunner::init_args_dump(Runtime &runtime, int device_id, KernelArgsHelper &kernel_args) {
    int num_dump_threads = runtime.get_aicpu_thread_num();

    auto alloc_cb = [this](size_t size) -> void * {
        return mem_alloc_.alloc(size);
    };
    auto free_cb = [this](void *dev_ptr) -> int {
        return mem_alloc_.free(dev_ptr);
    };
    int rc = dump_collector_.initialize(
        num_dump_threads, device_id, alloc_cb, /*register_cb=*/nullptr, free_cb, output_prefix_, dump_args_level_
    );
    if (rc != 0) {
        return rc;
    }

    kernel_args.args.dump_data_base = reinterpret_cast<uint64_t>(dump_collector_.get_dump_shm_device_ptr());
    return 0;
}

int DeviceRunner::init_pmu(
    int num_cores, int num_threads, const std::string &csv_path, PmuEventType event_type, int device_id,
    KernelArgsHelper &kernel_args
) {
    auto alloc_cb = [this](size_t size) -> void * {
        return mem_alloc_.alloc(size);
    };
    auto free_cb = [this](void *dev_ptr) -> int {
        return mem_alloc_.free(dev_ptr);
    };
    int rc = pmu_collector_.init(
        num_cores, num_threads, csv_path, event_type, alloc_cb, /*register_cb=*/nullptr, free_cb, device_id
    );
    if (rc == 0) {
        kernel_args.args.pmu_data_base = reinterpret_cast<uint64_t>(pmu_collector_.get_pmu_shm_device_ptr());
        kernel_args.args.aicore_pmu_ring_addrs =
            reinterpret_cast<uint64_t>(pmu_collector_.get_aicore_ring_addrs_device_ptr());
    }
    return rc;
}

int DeviceRunner::init_scope_stats(int num_threads, int device_id, KernelArgsHelper &kernel_args) {
    // a5: register_cb=nullptr, so the collector mallocs a host shadow per
    // device buffer + rtMemcpy's the zeroed shadow to device (see
    // ProfilerBase::alloc_paired_buffer). No halHostRegister on a5.
    auto alloc_cb = [this](size_t size) -> void * {
        return mem_alloc_.alloc(size);
    };
    auto free_cb = [this](void *dev_ptr) -> int {
        return mem_alloc_.free(dev_ptr);
    };
    int rc = scope_stats_collector_.init(num_threads, alloc_cb, /*register_cb=*/nullptr, free_cb, device_id);
    if (rc != 0) {
        return rc;
    }
    kernel_args.args.scope_stats_data_base =
        reinterpret_cast<uint64_t>(scope_stats_collector_.get_scope_stats_shm_device_ptr());
    return 0;
}

int DeviceRunner::init_dep_gen(int num_threads, int device_id, KernelArgsHelper &kernel_args) {
    // a5: register_cb=nullptr, so the collector mallocs a host shadow per
    // device buffer + rtMemcpy's the zeroed shadow to device. No
    // halHostRegister on a5 (matches PMU / chip swimlane / dump collectors).
    auto alloc_cb = [this](size_t size) -> void * {
        return mem_alloc_.alloc(size);
    };
    auto free_cb = [this](void *dev_ptr) -> int {
        return mem_alloc_.free(dev_ptr);
    };
    int rc = dep_gen_collector_.init(num_threads, alloc_cb, /*register_cb=*/nullptr, free_cb, device_id);
    if (rc != 0) {
        return rc;
    }
    kernel_args.args.dep_gen_data_base = reinterpret_cast<uint64_t>(dep_gen_collector_.get_dep_gen_shm_device_ptr());
    return 0;
}
