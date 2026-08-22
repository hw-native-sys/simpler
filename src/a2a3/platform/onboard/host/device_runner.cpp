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

#include "host_log.h"
#include "aicpu_loader/host/load_aicpu_op.h"

#include <dlfcn.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include "acl/acl.h"
#include "host/acl_error_log.h"
#include "platform_comm/comm.h"
#include "pto_runtime_c_api.h"

// Include HAL constants from CANN (header only, library loaded dynamically)
#include "ascend_hal.h"
#include "aicpu_topology_probe.h"
#include "callable.h"
#include "callable_protocol.h"
#include "call_config.h"
#include "chip_callable_layout.h"
#include "utils/elf_build_id.h"
#include "host/host_regs.h"  // Register address retrieval
#include "host/raii_scope_guard.h"
#include "utils/fatal_shutdown_latch.h"

// dep_gen has two shapes, one per orchestration site, and each runtime provides
// the strong symbols for the one it uses:
//   - device orchestration (tensormap_and_ringbuffer): the AICPU writes a ring
//     of captured submits, the host collector drains it, and
//     `dep_gen_replay_emit_deps_json` (runtime/.../host/dep_gen_replay.cpp)
//     replays them into deps.json.
//   - host orchestration (host_build_graph): the graph is captured from the
//     orchestrator's own dependency path as it runs on the host, and
//     `dep_gen_host_graph_*` (runtime/.../host/dep_gen_host_graph.cpp) writes it
//     out directly — no ring, no collector, nothing to reconcile.
// A runtime links only its own half, so each half needs a weak fallback here.
// visibility=hidden keeps the stubs off the global dynamic symbol table so they
// can't accidentally shadow a strong symbol via RTLD_GLOBAL.
// LOG_DEBUG (not WARN): the runner picks the shape via
// `dep_gen_host_graph_active()`, so neither stub is reachable when dep_gen is on
// — they exist purely to keep the .so loadable.
extern "C" __attribute__((weak, visibility("hidden"))) int dep_gen_replay_emit_deps_json(
    const struct DepGenRecord * /*records*/, size_t /*num_records*/, const char * /*deps_json_path*/
) {
    LOG_DEBUG("dep_gen replay not implemented for this runtime — deps.json skipped");
    return -1;
}

extern "C" __attribute__((weak, visibility("hidden"))) bool dep_gen_host_graph_active() { return false; }
extern "C" __attribute__((weak, visibility("hidden"))) void dep_gen_host_graph_set_enabled(bool /*enable*/) {}
extern "C" __attribute__((weak, visibility("hidden"))) int dep_gen_host_graph_emit(const char * /*deps_json_path*/) {
    LOG_DEBUG("dep_gen host graph not implemented for this runtime — deps.json skipped");
    return -1;
}

// =============================================================================
// Lazy-loaded HAL (ascend_hal) for halHostRegister / halHostUnregister
// =============================================================================

namespace {
void *g_hal_handle = nullptr;

using HalHostRegisterFn = int (*)(void *dev_ptr, size_t size, unsigned int flags, int device_id, void **host_ptr);
using HalHostUnregisterFn = int (*)(void *host_ptr, int device_id);

int load_hal_if_needed() {
    if (g_hal_handle != nullptr) {
        return 0;
    }
    g_hal_handle = dlopen("libascend_hal.so", RTLD_NOW | RTLD_LOCAL);
    if (g_hal_handle == nullptr) {
        return -1;
    }
    return 0;
}

HalHostRegisterFn get_halHostRegister() {
    if (g_hal_handle == nullptr) {
        return nullptr;
    }
    return reinterpret_cast<HalHostRegisterFn>(dlsym(g_hal_handle, "halHostRegister"));
}

HalHostUnregisterFn get_halHostUnregister() {
    if (g_hal_handle == nullptr) {
        return nullptr;
    }
    return reinterpret_cast<HalHostUnregisterFn>(dlsym(g_hal_handle, "halHostUnregister"));
}

}  // namespace

// =============================================================================
// a2a3-only KernelArgsHelper extension
// =============================================================================

int kernel_args_init_ffts_base_addr(KernelArgsHelper &helper) {
    uint64_t ffts_base_addr{0};
    uint32_t ffts_len{0};
    int rc = rtGetC2cCtrlAddr(&ffts_base_addr, &ffts_len);
    if (rc != 0) {
        LOG_ERROR("rtGetC2cCtrlAddr failed: %d", rc);
        return rc;
    }
    helper.args.ffts_base_addr = ffts_base_addr;
    return 0;
}

// =============================================================================
// DeviceRunner Implementation
// =============================================================================

DeviceRunner::~DeviceRunner() { finalize(); }

// `setup_static_arena`, `create_thread`, `attach_current_thread`,
// `configure_aicore_op_timeout`, and `ensure_device_initialized` live on
// `DeviceRunnerBase` — see
// `src/common/platform/onboard/host/device_runner_base.cpp`.

int DeviceRunner::ensure_acl_ready(int device_id) {
    if (device_id < 0) {
        LOG_ERROR("ensure_acl_ready: invalid device_id %d", device_id);
        return -1;
    }

    // aclInit is process-wide; CANN returns ACL_ERROR_REPEAT_INITIALIZE if it
    // has already been initialized (possibly by another owner), which we
    // treat as success.
    aclError aRet = aclInit(nullptr);
    if (aRet != ACL_SUCCESS && static_cast<int>(aRet) != ACL_ERROR_REPEAT_INITIALIZE) {
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

// `ensure_binaries_loaded`, `query_max_block_dim`, and `validate_block_dim`
// live on `DeviceRunnerBase` — see
// `src/common/platform/onboard/host/device_runner_base.cpp`.

void DeviceRunner::set_dep_gen_enabled(bool enable) {
    enable_dep_gen_ = enable;
    // Arms host-side capture for a host-orch runtime (no-op weak stub for the
    // device-orch one). The c_api latches the CallConfig before bind, and the
    // orchestration entry resets the graph before recording it.
    dep_gen_host_graph_set_enabled(enable);
}

int DeviceRunner::prepare_execution(
    Runtime &runtime, const CallConfig &config, uint32_t pipeline_slot, const NativeRunIdentity &identity,
    std::unique_ptr<PreparedExecution> *prepared
) {
    if (prepared == nullptr || *prepared != nullptr) return -1;
    auto execution = std::make_unique<PreparedExecution>(identity, runtime, config, pipeline_slot);
    execution->resources_owned = true;
    auto prepare_rollback = RAIIScopeGuard([this, &execution]() {
        cleanup_execution(*execution, /*retire_aicore=*/false);
    });
    const int block_dim = runtime.get_worker_count() / cores_per_blockdim_;
    int launch_aicpu_num = config.aicpu_thread_num;
    // A prior AICore launch/sync error poisoned the device context and the
    // in-place drain could not clear it. Refuse to run rather than cascade into
    // rtMalloc 507899 / halResMap rc=62 (init_aicore_register_addresses). A soft
    // close()+reset does NOT clear the poison, but finalize() force-resets the
    // card on this path so the next Worker re-inits clean in the same process
    // (see force_reset_device()). Failing fast here turns the rest of an xdist
    // worker session's tests from a slow, confusing failure cascade into a
    // single fast, self-explanatory error; the runner is then recovered at
    // finalize.
    if (device_unusable_.load(std::memory_order_acquire)) {
        LOG_ERROR(
            "DeviceRunner marked unusable by a prior AICore failure; refusing to enqueue. "
            "A soft reset does not clear the poison; finalize() will force-reset "
            "the card so the next Worker on it inits clean."
        );
        return -1;
    }
    if (validate_launch_aicpu_num(launch_aicpu_num) != 0) return -1;

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

    // Get AICore register addresses for register-based task dispatch
    rc = init_aicore_register_addresses(
        &execution->kernel_args.args.regs, static_cast<uint64_t>(device_id_), mem_alloc_, AicoreRegKind::Ctrl
    );
    if (rc != 0) {
        LOG_ERROR("init_aicore_register_addresses(Ctrl) failed: %d", rc);
        return rc;
    }

    // Get AICore PMU register addresses (distinct MMIO page from AIC_CTRL).
    if (enable_pmu_) {
        rc = init_aicore_register_addresses(
            &execution->kernel_args.args.pmu_reg_addrs, static_cast<uint64_t>(device_id_), mem_alloc_,
            AicoreRegKind::Pmu
        );
        if (rc != 0) {
            LOG_ERROR("init_aicore_register_addresses(Pmu) failed: %d", rc);
            return rc;
        }
    }

    // Build the profiling-flag bitfield (a2a3 carries an extra dep_gen bit).
    uint32_t enable_profiling_flag = SIMPLER_DFX_FLAG_NONE;
    if (enable_dump_args_) SIMPLER_SET_DFX_FLAG(enable_profiling_flag, SIMPLER_DFX_FLAG_DUMP_ARGS);
    if (enable_chip_swimlane_) SIMPLER_SET_DFX_FLAG(enable_profiling_flag, SIMPLER_DFX_FLAG_CHIP_SWIMLANE);
    if (enable_pmu_) SIMPLER_SET_DFX_FLAG(enable_profiling_flag, SIMPLER_DFX_FLAG_PMU);
    // The device flag drives the AICPU writer only; a host-orch runtime has no
    // device-side dep_gen to switch on.
    if (enable_dep_gen_ && !dep_gen_host_graph_active()) {
        SIMPLER_SET_DFX_FLAG(enable_profiling_flag, SIMPLER_DFX_FLAG_DEP_GEN);
    }
    if (enable_scope_stats_ && !scope_stats_uses_host_capture()) {
        SIMPLER_SET_DFX_FLAG(enable_profiling_flag, SIMPLER_DFX_FLAG_SCOPE_STATS);
    }
    execution->kernel_args.args.enable_profiling_flag = enable_profiling_flag;

    resolve_task_binary_addrs(runtime);

    // a2a3 onboard now uses the same host-computed, device-filtered affinity
    // shape as a5. Host probes the AICPU user pool once, chooses the active
    // cpu_ids deterministically, writes them into Runtime, and the AICPU-side
    // gate only matches sched_getcpu() against this table.
    {
        std::vector<pto::a2a3::AicpuLogicalCpu> user_cpus;
        std::vector<int32_t> allowed;
        runtime.set_aicpu_allowed_cpu_count(0);
        runtime.set_aicpu_launch_count(0);
        if (!pto::a2a3::probe_aicpu_topology(static_cast<uint32_t>(device_id_), user_cpus)) {
            LOG_ERROR("A2A3 AICPU topology probe failed; cannot configure affinity gate");
            return -1;
        }
        int resolved_aicpu = resolve_aicpu_thread_num(
            runtime.get_aicpu_thread_num(), static_cast<int>(user_cpus.size()), PLATFORM_DEFAULT_AICPU_THREAD_NUM
        );
        if (resolved_aicpu < 0) return -1;
        if (!pto::a2a3::compute_allowed_cpus(user_cpus, resolved_aicpu, allowed)) {
            LOG_ERROR(
                "A2A3 AICPU topology has %zu user cpus, cannot fit %d active threads", user_cpus.size(), resolved_aicpu
            );
            return -1;
        }
        runtime.set_aicpu_thread_num(resolved_aicpu);
        launch_aicpu_num = resolved_aicpu;
        const size_t cap = runtime.aicpu_allowed_cpus_capacity();
        if (allowed.size() > cap) {
            LOG_ERROR("A2A3 compute_allowed_cpus returned %zu > cap %zu", allowed.size(), cap);
            return -1;
        }
        int32_t *allowed_cpus = runtime.get_aicpu_allowed_cpus();
        for (size_t i = 0; i < allowed.size(); ++i)
            allowed_cpus[i] = allowed[i];
        runtime.set_aicpu_allowed_cpu_count(static_cast<int32_t>(allowed.size()));
        int32_t launch_n = static_cast<int32_t>(user_cpus.size());
        if (launch_n > PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH) {
            launch_n = PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH;
        }
        runtime.set_aicpu_launch_count(launch_n);

        std::string dump;
        for (size_t i = 0; i < allowed.size(); ++i) {
            if (i) dump += ", ";
            dump += std::to_string(allowed[i]);
            if (i + 1 == allowed.size()) dump += "(last)";
        }
        LOG_INFO(
            "A2A3 AICPU ALLOWED_CPUS = [%s] (active=%d, launch=%d, user_cpus=%zu)", dump.c_str(),
            runtime.get_aicpu_thread_num(), runtime.get_aicpu_launch_count(), user_cpus.size()
        );
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
        // Initialize args dump (independent from profiling)
        rc = init_args_dump(runtime, device_id_, execution->kernel_args);
        if (rc != 0) {
            LOG_ERROR("init_args_dump failed: %d", rc);
            return rc;
        }
    }

    if (enable_pmu_) {
        rc = init_pmu(
            num_aicore, launch_aicpu_num, make_pmu_csv_path(output_prefix_), pmu_event_type_, device_id_,
            execution->kernel_args
        );
        if (rc != 0) {
            LOG_ERROR("init_pmu failed: %d", rc);
            return rc;
        }
    }

    // A host-orch runtime already holds the graph in host memory; standing up
    // the device ring and its collector would allocate shared memory and a
    // drain thread for a stream that never produces a record.
    if (enable_dep_gen_ && !dep_gen_host_graph_active()) {
        rc = init_dep_gen(launch_aicpu_num, device_id_, execution->kernel_args);
        if (rc != 0) {
            LOG_ERROR("init_dep_gen failed: %d", rc);
            return rc;
        }
    }

    if (enable_scope_stats_ && !scope_stats_uses_host_capture()) {
        rc = init_scope_stats(launch_aicpu_num, device_id_, execution->kernel_args);
        if (rc != 0) {
            LOG_ERROR("init_scope_stats failed: %d", rc);
            return rc;
        }
    }

    // Resolve the orchestration SO into a device-resident buffer and refresh
    // runtime metadata before the Runtime struct is uploaded to device.
    rc = prepare_orch_so(runtime);
    if (rc != 0) {
        LOG_ERROR("prepare_orch_so failed: %d", rc);
        return rc;
    }

    rc = init_runtime_args_with_metadata(runtime, execution->kernel_args);
    if (rc != 0) return rc;

    rc = kernel_args_init_ffts_base_addr(execution->kernel_args);
    if (rc != 0) {
        LOG_ERROR("init_ffts_base_addr failed: %d", rc);
        return rc;
    }

    // Copy KernelArgs to device memory for AICore
    rc = execution->kernel_args.init_device_kernel_args(mem_alloc_);
    if (rc != 0) {
        LOG_ERROR("init_device_kernel_args failed: %d", rc);
        return rc;
    }

    execution->num_aicore = num_aicore;
    execution->launch_aicpu_num = launch_aicpu_num;
    prepare_rollback.dismiss();
    *prepared = std::move(execution);
    return 0;
}

int DeviceRunner::poll_execution(const ActiveExecution &active) {
    if (active.prepared == nullptr) return SIMPLER_NATIVE_RUN_POLL_ERROR;
    return run_streams_.poll([](void *aicpu, void *aicore) {
        return query_stream_pair_nonblocking(static_cast<rtStream_t>(aicpu), static_cast<rtStream_t>(aicore));
    });
}

int DeviceRunner::drain_execution(ActiveExecution &active) {
    if (active.prepared == nullptr || !active.prepared->resources_owned) return -1;
    PreparedExecution &prepared = *active.prepared;
    auto drain_cleanup = RAIIScopeGuard([this, &prepared]() {
        cleanup_execution(prepared, /*retire_aicore=*/true);
    });

    int rc = reap_run();
    if (rc != 0) {
        // The device/sync error remains authoritative over teardown errors.
        return rc;
    }

    // A proven-complete stream is reusable until a code publication marks it
    // stale. Publish retirement so cleanup does not replace it with an
    // unproven state after the device result has already been established.
    prepared.aicore_retirement_attempted = true;
    rc = retire_run_aicore_stream(&prepared, RunStreamPair::CompletionStatus::Complete);
    if (rc != 0) return rc;

    // Reads device memory, so it must precede KernelArgs/runtime cleanup.
    print_handshake_results(prepared.kernel_args);
    return 0;
}

void DeviceRunner::cleanup_execution(PreparedExecution &prepared, bool retire_aicore) noexcept {
    if (!prepared.resources_owned) return;

    // A poisoned card cannot retire per-resource frees or stream destroys, and
    // issuing them can block in the driver. Drop host-side ownership instead;
    // finalize()'s force reset invalidates the whole device generation.
    const bool abandon = device_unusable_.load(std::memory_order_acquire);

    // Collectors must stop before their backing arguments are released; the
    // per-run stream retires last. Each cleanup operation is idempotent.
    finalize_collectors(abandon);
    if (abandon) {
        prepared.kernel_args.abandon_after_device_failure();
    } else {
        (void)prepared.kernel_args.finalize_device_kernel_args();
        (void)prepared.kernel_args.finalize_runtime_args();
    }
    if (prepared.kernel_args.args.pmu_reg_addrs != 0) {
        if (!abandon) {
            (void)mem_alloc_.free(reinterpret_cast<void *>(prepared.kernel_args.args.pmu_reg_addrs));
        }
        prepared.kernel_args.args.pmu_reg_addrs = 0;
    }
    if (prepared.kernel_args.args.regs != 0) {
        if (!abandon) {
            (void)mem_alloc_.free(reinterpret_cast<void *>(prepared.kernel_args.args.regs));
        }
        prepared.kernel_args.args.regs = 0;
    }
    if (retire_aicore && !abandon && !prepared.aicore_retirement_attempted) {
        prepared.aicore_retirement_attempted = true;
        (void)retire_run_aicore_stream(&prepared, RunStreamPair::CompletionStatus::Unproven);
    }
    prepared.resources_owned = false;
}

void DeviceRunner::abandon_prepared_execution(PreparedExecution &prepared) noexcept {
    cleanup_execution(prepared, /*retire_aicore=*/true);
}

int DeviceRunner::create_run_stream(void **out) {
    if (out == nullptr) return -1;
    *out = nullptr;

    rtStream_t stream = nullptr;
    int rc = rtStreamCreate(&stream, 0);
    if (rc != 0) {
        LOG_ERROR("rtStreamCreate (run stream) failed: %d", rc);
        ACL_LOG_ERROR_DETAIL(rc);
        return rc;
    }

    // CANN 9.0.0 + driver 26.0.rc1 needs the host to observe an SDMA-generation
    // AICore fault before the device reaches DEV_RUNNING_DOWN; otherwise reset
    // walks the 48 CP-process streams and blocks in a 300-second remote event.
    // Keep this workaround strictly on Workers that actually provisioned SDMA.
    // Ordinary a2a3 and every a5 stream retain their normal error/diagnostic
    // contract (for example 507046 rather than the early 507015).
    if (dma_workspace_handle_ != nullptr) {
        aclError acl_rc = aclrtSetStreamFailureMode(stream, ACL_STOP_ON_FAILURE);
        if (acl_rc != ACL_SUCCESS) {
            LOG_ERROR("aclrtSetStreamFailureMode (SDMA run stream) failed: %d", static_cast<int>(acl_rc));
            (void)rtStreamDestroy(stream);
            return static_cast<int>(acl_rc);
        }
    }

    *out = stream;
    return 0;
}

int DeviceRunner::ensure_run_streams() {
    int rc = run_streams_.ensure();
    if (rc != 0) {
        LOG_ERROR("ensure_run_streams failed: %d", rc);
        ACL_LOG_ERROR_DETAIL(rc);
    }
    return rc;
}

int DeviceRunner::retire_run_aicore_stream(const void *owner, RunStreamPair::CompletionStatus completion_status) {
    int rc = run_streams_.retire(completion_status, owner);
    if (rc != 0) {
        LOG_ERROR("rtStreamDestroy (run AICore stream) failed: %d, the handle is retained for teardown", rc);
    }
    return rc;
}

int DeviceRunner::destroy_run_streams() {
    // No pre-destroy sync, for the reason finalize_common() documents for the
    // bootstrap pair: rtStreamDestroy is the supported teardown for a stream
    // left in the error state by an op-timeout.
    int rc = run_streams_.destroy();
    if (rc != 0) {
        LOG_ERROR("destroy_run_streams: a stream survived teardown: %d", rc);
    }
    return rc;
}

DeviceRunnerBase::LaunchOutcome
DeviceRunner::launch_execution(std::unique_ptr<PreparedExecution> prepared, LaunchPermit permit) {
    LaunchOutcome outcome;
    if (prepared == nullptr) return outcome;

    LaunchTransactionResult transaction = launch_run(*prepared, std::move(permit));
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

LaunchTransactionResult DeviceRunner::launch_run(PreparedExecution &prepared, LaunchPermit permit) {
    Runtime &runtime = *prepared.runtime;
    const int num_aicore = prepared.num_aicore;
    const int launch_aicpu_num = prepared.launch_aicpu_num;
    // KernelLaunch is the pipeline boundary: this method clears the handshake
    // consumed by the launch and submits exactly the AICore and AICPU kernels.
    // It intentionally performs no stream synchronization or per-run cleanup.
    //
    // The pair is readied here rather than at prepare because this is the first
    // point the caller holds the execution claim: a prepared successor overlaps
    // its predecessor's execution, so replacing a stale AICore stream during
    // preparation would destroy a stream the predecessor is still running on.
    if (ensure_run_streams() != 0) {
        return LaunchTransactionResult{};
    }
    RunStreamSet streams{static_cast<rtStream_t>(run_streams_.aicpu()), static_cast<rtStream_t>(run_streams_.aicore())};
    LaunchTransactionResult result = exact_launch_transaction(
        prepared.identity, std::move(permit),
        [&]() {
            // Arming precedes any execution-visible submission, so its failures —
            // including a thread-spawn or allocation throw — are reported as an rc
            // and leave the run safely rollback-able.
            try {
                activate_launch_shape(runtime);
                (void)arm_device_wall_buffer(prepared.kernel_args);
                start_shared_collectors_for_run();
                if (enable_dep_gen_ && !dep_gen_host_graph_active()) {
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

                // Launch the AICore worker BEFORE the AICPU Run task — mirrors the a5 path
                // so the two arches stay symmetric. First-launch latency optimization +
                // op-timeout-family defense-in-depth: with the AICPU Run task launched first
                // it occupies the device (spinning in the handshake), and the first AICore
                // launch — which lazily loads the kernel binary onto the device inside
                // rtKernelLaunchWithHandleV2 — is slow; launching AICore first does that load
                // on an idle device. The handshake is launch-order-independent. The ~1.4 s
                // slow-launch / 207001 wedge was measured on a5; this mirror is UNVERIFIED on
                // a2a3 silicon (the dev box is a5-only), relying on CI. See
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
                LOG_ERROR("launch_run: arming failed before any stream submission");
                return -1;
            }

            // Publishing query/drain ownership is a state change, so it sits past
            // the arming block: everything the block covers is rollback-free.
            int rc = run_streams_.mark_submitted(&prepared);
            if (rc != 0) {
                LOG_ERROR("launch_run: failed to publish the run stream pair: %d", rc);
                return rc;
            }

            LOG_INFO("=== launch_aicore_kernel ===");
            int launch_rc = launch_aicore_kernel(streams.aicore, prepared.kernel_args.device_k_args_);
            if (launch_rc != 0) {
                LOG_ERROR("launch_aicore_kernel failed: %d", launch_rc);
                recover_device_or_mark_unusable(launch_rc);
            }
            return launch_rc;
        },
        [&]() {
            LOG_INFO("=== launch_aicpu_kernel %s ===", host::KernelNames::RunName);
            int aicpu_launch_n =
                (runtime.get_aicpu_launch_count() > 0) ? runtime.get_aicpu_launch_count() : launch_aicpu_num;
            int launch_rc = launch_aicpu_kernel(
                streams.aicpu, &prepared.kernel_args.args, host::KernelNames::RunName, aicpu_launch_n
            );
            if (launch_rc != 0) {
                LOG_ERROR("launch_aicpu_kernel (main) failed: %d", launch_rc);
            }
            return launch_rc;
        }
    );
    return result;
}

int DeviceRunner::reap_run() {
    if (!run_streams_.ready()) {
        LOG_ERROR("reap_run: the run stream pair is not ready");
        return -1;
    }
    int rc = sync_stream_pair(run_streams_.aicpu(), run_streams_.aicore());
    if (rc != 0) {
        // The pair wait surfaces the AICore op-timeout (STARS-reaped op ->
        // 507000/507018/507046 at AICPU/AICore stream sync). The op-timeout
        // leaves the device context poisoned for the SAME DeviceRunner's next
        // run, so attempt recovery / mark-unusable here too, not only on the
        // launch-error path above.
        recover_device_or_mark_unusable(rc);
        // On an AICPU-detected scheduler hang the device flushed its diagnostic
        // buffers during emergency_shutdown before returning the timeout rc.
        // Export them here too — otherwise the success-only teardown below is
        // skipped and the dumped tensors (the stuck task's inputs plus every
        // completed task's in/out) are streamed to .bin but left without the
        // JSON manifest, i.e. unusable for triage. reconcile/export are not
        // idempotent, so this runs only on the error return; the success path
        // still exports exactly once below.
        teardown_shared_collectors_after_run(false);
        return rc;
    }

    read_device_wall_ns();

    // Tear down collectors. stop() joins mgmt then collector in the only safe
    // order (mgmt's final-drain pass into L2 has poll as its consumer).
    teardown_shared_collectors_after_run(true);

    // a2a3-only dep_gen teardown: host-orch emits the graph its orchestration
    // built on this same thread; device-orch stops the collector, reconciles the
    // ring, and replays the records.
    if (enable_dep_gen_) {
        const std::string deps = make_deps_json_path(output_prefix_);
        if (dep_gen_host_graph_active()) {
            int rc = dep_gen_host_graph_emit(deps.c_str());
            if (rc != 0) {
                LOG_ERROR("dep_gen host graph emit failed (%d) — deps.json not produced", rc);
            }
        } else {
            dep_gen_collector_.stop();
            if (dep_gen_collector_.reconcile_counters()) {
                const auto &records = dep_gen_collector_.records();
                int rc = dep_gen_replay_emit_deps_json(records.data(), records.size(), deps.c_str());
                if (rc != 0) {
                    LOG_ERROR("dep_gen replay failed (%d) — deps.json not produced", rc);
                }
            }
        }
    }

    return 0;
}

// `print_handshake_results`, `prepare_orch_so`, `register_callable`,
// `record_host_orch_callable`, `unregister_callable`, `has_callable`,
// `bind_callable_to_runtime`, and `upload_chip_callable_buffer` live on
// `DeviceRunnerBase`.

void DeviceRunner::recover_device_or_mark_unusable(int aicore_rc) {
    // An AICore launch failure (207001) or an op-timeout reaped by STARS
    // (surfaced as 507000/507018/507046 at stream sync) leaves the device
    // context in a sticky-error state: the streams stay poisoned and the
    // SAME DeviceRunner's next enqueue fails early — observed on a2a3 as
    // `rtMalloc failed: 507899`, and on a5 as `halResMap failed (rc=62)` in
    // init_aicore_register_addresses. Reused across a session (the L2
    // st_worker pool hands one ChipWorker to every test class on a device),
    // that one error poisons every later test in the xdist worker process.
    //
    // Best-effort BOUNDED drain (aclrtSynchronizeDeviceWithTimeout, NOT an
    // unbounded aclrtSynchronizeStream* on the error-state stream — the latter
    // wedges subsequent tests, see DeviceRunnerBase::finalize_common). But DO
    // NOT gate recovery on the drain's rc: the bounded drain can return success
    // on a still-poisoned card — a false-negative that left force_reset_device()
    // untriggered and cascaded into 16 skipped L2 cases on CI run 27742754024
    // (st-onboard-a2a3 gw3). The op-timeout sticky-error is only cleared by a
    // force reset (a soft reset/drain does not), so always mark the runner
    // unusable here: admission fails fast and finalize() force-resets the card,
    // so the next Worker.init lands clean regardless of what the drain reported.
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
        aclError rc = aclInit(nullptr);
        if (rc == ACL_SUCCESS) {
            owns_ = true;
            ok_ = true;
        } else if (static_cast<int>(rc) == ACL_ERROR_REPEAT_INITIALIZE) {
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

    // A successful force reset has already destroyed the bound default
    // context. Suppress the ordinary reset that otherwise balances a live
    // binding; issuing it after force reset reports 507007 and pollutes ACL's
    // recent-error state even though the requested teardown succeeded.
    void dismiss_after_force_reset() { bound_ = false; }

private:
    int device_id_;
    bool bound_{false};
};

}  // namespace

int DeviceRunner::force_reset_device() {
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
        bind_guard.dismiss_after_force_reset();
    }
    // Post-reset self-check: a 0 rc from aclrtResetDeviceForce does not by itself
    // prove the card is usable. Re-bind (fresh guard, balanced on exit) and
    // exercise both poison surfaces — a trivial stream create/destroy (a5 poison:
    // rtStreamCreate / halResMap) and an HBM alloc/free (a2a3 poison: rtMalloc
    // 507899) — checking every rc so any failure (incl. the frees) returns
    // non-zero and finalize() keeps the card flagged for the layer above
    // (st_worker poison-skip + #1110 dispatcher retry).
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
        "force_reset_device: aclrtResetDeviceForce(%d) established a usable device generation (probe confirmed)",
        device_id_
    );
    return 0;
}

void *DeviceRunner::register_device_memory_to_host(void *dev_ptr, std::size_t bytes) {
    if (dev_ptr == nullptr || bytes == 0) {
        return nullptr;
    }
    if (device_id_ < 0) {
        LOG_ERROR("register_device_memory_to_host: invalid device_id %d", device_id_);
        return nullptr;
    }
    if (load_hal_if_needed() != 0) {
        LOG_ERROR("register_device_memory_to_host: failed to load ascend_hal: %s", dlerror());
        return nullptr;
    }
    HalHostRegisterFn fn = get_halHostRegister();
    if (fn == nullptr) {
        LOG_ERROR("register_device_memory_to_host: halHostRegister symbol not found: %s", dlerror());
        return nullptr;
    }
    void *host_va = nullptr;
    int rc = fn(dev_ptr, bytes, DEV_SVM_MAP_HOST, device_id_, &host_va);
    if (rc != 0) {
        LOG_ERROR("register_device_memory_to_host: halHostRegister failed for dev_ptr %p (rc=%d)", dev_ptr, rc);
        return nullptr;
    }
    return host_va;
}

void DeviceRunner::unregister_device_memory_from_host(void *dev_ptr) {
    if (dev_ptr == nullptr) {
        return;
    }
    if (device_id_ < 0) {
        LOG_ERROR("unregister_device_memory_from_host: invalid device_id %d", device_id_);
        return;
    }
    if (load_hal_if_needed() != 0) {
        LOG_ERROR("unregister_device_memory_from_host: failed to load ascend_hal: %s", dlerror());
        return;
    }
    // halHostUnregister is keyed by the device pointer; the HAL maps it back to
    // the host VA internally.
    HalHostUnregisterFn fn = get_halHostUnregister();
    if (fn == nullptr) {
        LOG_ERROR("unregister_device_memory_from_host: halHostUnregister symbol not found: %s", dlerror());
        return;
    }
    int rc = fn(dev_ptr, device_id_);
    if (rc != 0) {
        LOG_ERROR("unregister_device_memory_from_host: halHostUnregister failed for dev_ptr %p (rc=%d)", dev_ptr, rc);
    }
}

int DeviceRunner::finalize() {
    if (device_id_ == -1) {
        return 0;
    }

    // Fatal path: the ordinary stream completion/error boundary has already
    // reaped the submitted run. Stop host collector threads locally, drain and
    // force-reset the card, then forget old handles without per-resource
    // RTS/HAL calls.
    if (device_unusable_.load(std::memory_order_acquire)) {
        // A Worker that provisioned SDMA holds 48 CP-process streams, which is
        // what makes both the handoff delay and the single reset attempt below
        // necessary. Read before abandon_common_after_device_failure() clears
        // the handle.
        const bool sdma_provisioned = dma_workspace_handle_ != nullptr;

        // CANN exposes no retirement fence for its CP-process SDMA streams.
        // On the package named above, repeated hardware A/B runs established
        // this handoff delay as a containment workaround, not a portable
        // runtime guarantee. Do not delay ordinary (non-SDMA) failures.
        if (sdma_provisioned) {
            constexpr uint64_t kCANN900FatalHandoffMs = 10000;
            constexpr uint64_t kDeviceDownSettleMarginMs = 5000;
            const uint64_t op_timeout_ms = (timeout_config_.op_execute_timeout_us + 999) / 1000;
            const uint64_t configured_handoff_ms = op_timeout_ms + kDeviceDownSettleMarginMs;
            const uint64_t reset_delay_ms = std::max(kCANN900FatalHandoffMs, configured_handoff_ms);
            LOG_WARN(
                "SDMA fatal teardown workaround (CANN 9.0.0/driver 26.0.rc1): "
                "waiting %llu ms before force reset; no runtime retirement fence is available",
                static_cast<unsigned long long>(reset_delay_ms)
            );
            std::this_thread::sleep_for(std::chrono::milliseconds(reset_delay_ms));
        }

        finalize_collectors(true);

        // force_reset_device() drains before it resets and returns 0 only when
        // its post-reset probe confirms the card, so a second pass runs against
        // a settled card and can recover a poison the first pass could not
        // (verified on a2a3). An SDMA-provisioned card gets a single attempt:
        // there a non-confirming reset already blocks on the driver's
        // remote-event timeout, which a retry only multiplies.
        constexpr int kFatalResetAttempts = 3;
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

        run_streams_.abandon();
        int abandon_rc = abandon_common_after_device_failure();

        // Only finalize the ACL owner after force reset established a clean
        // generation. On reset failure, aclFinalize may itself walk poisoned
        // stream resources and enter the same remote-event timeout.
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

    // Cleanup performance profiling (including a2a3's dep_gen). Normally
    // already done by drain or enqueue rollback; this is the backstop
    // for the no-run-since-init case.
    finalize_collectors();

    // The run stream pair is this subclass's own RTS-owning member, so it is
    // released here, while RTS is live and before the device reset below — the
    // same window finalize_common() uses for the bootstrap pair.
    int stream_rc = destroy_run_streams();

    // Shared cleanup body — streams, kernel_args, callable/orch maps,
    // chip-callable buffer pool, the three arenas, device_wall,
    // mem_alloc_.finalize(), and cached arena sizes.
    rc = finalize_common();
    if (rc == 0) rc = stream_rc;

    // Reset device AFTER all device memory is freed. Two paths:
    //
    // - acl_ready_=true (HCCL / comm path):  aclrtResetDevice + aclFinalize
    //   tear down both the device runtime and the ACL bring-up that
    //   ensure_acl_ready() did.
    //
    // - acl_ready_=false (pure rt path, the common case for non-HCCL tests):
    //   rtDeviceReset is still needed to clear any per-device runtime state
    //   the test left behind — without this, an AICPU exception / stuck
    //   kernel on this DeviceRunner wedges the device for the next
    //   ChipWorker that initializes on the same id (rtStreamCreate then
    //   returns 507899). a5 has been doing this unconditionally; a2a3 was
    //   missing the rt-path reset, which is the root cause of the chronic
    //   "test_dedup_shared_so_independent_unregister → 507899 cascade"
    //   pattern seen across PR CI all session.
    if (device_id_ >= 0) {
        if (acl_ready_) {
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
    }

    // Only the healthy path reaches here: a poisoned card returned from the
    // fatal branch at the top of finalize(), which owns the force reset.
    device_id_ = -1;
    device_unusable_.store(false, std::memory_order_release);
    return rc;
}

// `launch_aicpu_kernel` and `launch_aicore_kernel` live on `DeviceRunnerBase`.

int DeviceRunner::init_chip_swimlane(
    int num_aicore, int aicpu_thread_num, int device_id, KernelArgsHelper &kernel_args
) {
    auto alloc_cb = [this](size_t size) -> void * {
        return mem_alloc_.alloc(size);
    };

    auto register_cb = [](void *dev_ptr, size_t size, int device_id, void **host_ptr) -> int {
        if (load_hal_if_needed() != 0) {
            LOG_ERROR("Failed to load ascend_hal for profiling: %s", dlerror());
            return -1;
        }
        HalHostRegisterFn fn = get_halHostRegister();
        if (fn == nullptr) {
            LOG_ERROR("halHostRegister symbol not found: %s", dlerror());
            return -1;
        }
        return fn(dev_ptr, size, DEV_SVM_MAP_HOST, device_id, host_ptr);
    };

    auto free_cb = [this](void *dev_ptr) -> int {
        return mem_alloc_.free(dev_ptr);
    };

    int rc = chip_swimlane_collector_.initialize(
        num_aicore, aicpu_thread_num, device_id, chip_swimlane_level_, alloc_cb, register_cb, free_cb, output_prefix_
    );
    if (rc != 0) {
        return rc;
    }

    kernel_args.args.chip_swimlane_data_base =
        reinterpret_cast<uint64_t>(chip_swimlane_collector_.get_chip_swimlane_setup_device_ptr());
    kernel_args.args.chip_swimlane_aicore_rotation_table =
        reinterpret_cast<uint64_t>(chip_swimlane_collector_.get_aicore_ring_addr_table_device_ptr());
    return 0;
}

int DeviceRunner::init_args_dump(Runtime &runtime, int device_id, KernelArgsHelper &kernel_args) {
    int num_dump_threads = runtime.get_aicpu_thread_num();

    auto alloc_cb = [this](size_t size) -> void * {
        return mem_alloc_.alloc(size);
    };

    auto register_cb = [](void *dev_ptr, size_t size, int device_id, void **host_ptr) -> int {
        if (load_hal_if_needed() != 0) {
            LOG_ERROR("Failed to load ascend_hal for args dump: %s", dlerror());
            return -1;
        }
        HalHostRegisterFn fn = get_halHostRegister();
        if (fn == nullptr) {
            LOG_ERROR("halHostRegister symbol not found: %s", dlerror());
            return -1;
        }
        return fn(dev_ptr, size, DEV_SVM_MAP_HOST, device_id, host_ptr);
    };

    auto free_cb = [this](void *dev_ptr) -> int {
        return mem_alloc_.free(dev_ptr);
    };

    int rc = dump_collector_.initialize(
        num_dump_threads, device_id, alloc_cb, register_cb, free_cb, output_prefix_, dump_args_level_
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

    auto register_cb = [](void *dev_ptr, size_t size, int device_id, void **host_ptr) -> int {
        if (load_hal_if_needed() != 0) {
            LOG_ERROR("Failed to load ascend_hal for PMU: %s", dlerror());
            return -1;
        }
        HalHostRegisterFn fn = get_halHostRegister();
        if (fn == nullptr) {
            LOG_ERROR("halHostRegister symbol not found: %s", dlerror());
            return -1;
        }
        return fn(dev_ptr, size, DEV_SVM_MAP_HOST, device_id, host_ptr);
    };

    auto free_cb = [this](void *dev_ptr) -> int {
        return mem_alloc_.free(dev_ptr);
    };

    int rc =
        pmu_collector_.init(num_cores, num_threads, csv_path, event_type, alloc_cb, register_cb, free_cb, device_id);
    if (rc != 0) {
        return rc;
    }

    kernel_args.args.pmu_data_base = reinterpret_cast<uint64_t>(pmu_collector_.get_pmu_shm_device_ptr());
    return 0;
}

int DeviceRunner::init_dep_gen(int num_threads, int device_id, KernelArgsHelper &kernel_args) {
    auto alloc_cb = [this](size_t size) -> void * {
        return mem_alloc_.alloc(size);
    };

    auto register_cb = [](void *dev_ptr, size_t size, int device_id, void **host_ptr) -> int {
        if (load_hal_if_needed() != 0) {
            LOG_ERROR("Failed to load ascend_hal for dep_gen: %s", dlerror());
            return -1;
        }
        HalHostRegisterFn fn = get_halHostRegister();
        if (fn == nullptr) {
            LOG_ERROR("halHostRegister symbol not found: %s", dlerror());
            return -1;
        }
        return fn(dev_ptr, size, DEV_SVM_MAP_HOST, device_id, host_ptr);
    };

    auto free_cb = [this](void *dev_ptr) -> int {
        return mem_alloc_.free(dev_ptr);
    };

    int rc = dep_gen_collector_.init(num_threads, alloc_cb, register_cb, free_cb, device_id);
    if (rc != 0) {
        return rc;
    }

    kernel_args.args.dep_gen_data_base = reinterpret_cast<uint64_t>(dep_gen_collector_.get_dep_gen_shm_device_ptr());
    return 0;
}

int DeviceRunner::init_scope_stats(int num_threads, int device_id, KernelArgsHelper &kernel_args) {
    auto alloc_cb = [this](size_t size) -> void * {
        return mem_alloc_.alloc(size);
    };

    auto register_cb = +[](void *dev_ptr, size_t size, int device_id, void **host_ptr) -> int {
        if (load_hal_if_needed() != 0) {
            LOG_ERROR("Failed to load ascend_hal for scope_stats: %s", dlerror());
            return -1;
        }
        HalHostRegisterFn fn = get_halHostRegister();
        if (fn == nullptr) {
            LOG_ERROR("halHostRegister symbol not found: %s", dlerror());
            return -1;
        }
        return fn(dev_ptr, size, DEV_SVM_MAP_HOST, device_id, host_ptr);
    };

    auto free_cb = [this](void *dev_ptr) -> int {
        return mem_alloc_.free(dev_ptr);
    };

    int rc = scope_stats_collector_.init(num_threads, alloc_cb, register_cb, free_cb, device_id);
    if (rc != 0) {
        return rc;
    }

    kernel_args.args.scope_stats_data_base =
        reinterpret_cast<uint64_t>(scope_stats_collector_.get_scope_stats_shm_device_ptr());
    return 0;
}

void DeviceRunner::finalize_collectors(bool abandon_device_resources) {
    auto healthy_unregister_cb = [](void *dev_ptr, int device_id) -> int {
        HalHostUnregisterFn fn = get_halHostUnregister();
        if (fn != nullptr) {
            return fn(dev_ptr, device_id);
        }
        return 0;
    };
    auto no_op_unregister_cb = [](void *, int) -> int {
        return 0;
    };
    auto unregister_cb = abandon_device_resources ? no_op_unregister_cb : healthy_unregister_cb;
    auto free_cb = [this, abandon_device_resources](void *dev_ptr) -> int {
        if (abandon_device_resources) return 0;
        return mem_alloc_.free(dev_ptr);
    };

    if (chip_swimlane_collector_.is_initialized()) {
        chip_swimlane_collector_.finalize(unregister_cb, free_cb);
    }
    if (dump_collector_.is_initialized()) {
        dump_collector_.finalize(unregister_cb, free_cb);
    }
    if (pmu_collector_.is_initialized()) {
        pmu_collector_.finalize(unregister_cb, free_cb);
    }
    if (dep_gen_collector_.is_initialized()) {
        dep_gen_collector_.finalize(unregister_cb, free_cb);
    }
    if (scope_stats_collector_.is_initialized()) {
        scope_stats_collector_.finalize(unregister_cb, free_cb);
    }
}
