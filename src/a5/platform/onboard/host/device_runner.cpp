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
#include "host_log.h"

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
#include "utils/elf_build_id.h"
#include "utils/fnv1a_64.h"
#include "host/host_regs.h"  // Register address retrieval
#include "host/raii_scope_guard.h"

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

// rtMalloc / rtFree wrappers shared by all three profiling subsystems.
// a5 onboard goes directly through CANN runtime — no per-allocation tracking,
// so the framework's std::function alloc / free shapes match plain function
// pointers here.
static void *prof_alloc_cb(size_t size) {
    void *ptr = nullptr;
    int rc = rtMalloc(&ptr, size, RT_MEMORY_HBM, 0);
    return (rc == 0) ? ptr : nullptr;
}

static int prof_free_cb(void *dev_ptr) { return rtFree(dev_ptr); }

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
        return static_cast<int>(aRet);
    }

    // ACL device binding is per-thread; every caller must still hit it.
    aRet = aclrtSetDevice(device_id);
    if (aRet != ACL_SUCCESS) {
        LOG_ERROR("aclrtSetDevice(%d) failed: %d", device_id, static_cast<int>(aRet));
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

int DeviceRunner::run(Runtime &runtime, int block_dim, int launch_aicpu_num) {
    // A prior AICore launch/sync error poisoned the device context and the
    // in-place drain could not clear it. Refuse to run rather than cascade
    // into halResMap rc=62 (init_aicore_register_addresses) or rtMalloc
    // 507899. A soft close()+reset does NOT clear the poison on a5, but
    // finalize() force-resets the card on this path so the next Worker re-inits
    // clean in the same process (see force_reset_device()). Failing fast here
    // turns the rest of an xdist worker session's tests from a slow, confusing
    // failure cascade into a single fast, self-explanatory error; the runner is
    // then recovered at finalize.
    if (device_unusable_) {
        LOG_ERROR(
            "DeviceRunner marked unusable by a prior AICore failure; refusing to run. "
            "A soft reset does not clear the poison on a5; finalize() will force-reset "
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

    ensure_device_wall_buffer();

    block_dim = resolve_block_dim(block_dim);
    if (block_dim < 0) return -1;
    int num_aicore = block_dim * cores_per_blockdim_;

    rc = init_aicore_register_addresses(&kernel_args_.args.regs, static_cast<uint64_t>(device_id_), mem_alloc_);
    if (rc != 0) {
        LOG_ERROR("init_aicore_register_addresses failed: %d", rc);
        return rc;
    }

    // Build the profiling-flag bitfield.
    uint32_t enable_profiling_flag = PROFILING_FLAG_NONE;
    if (enable_dump_tensor_) SET_PROFILING_FLAG(enable_profiling_flag, PROFILING_FLAG_DUMP_TENSOR);
    if (enable_l2_swimlane_) SET_PROFILING_FLAG(enable_profiling_flag, PROFILING_FLAG_L2_SWIMLANE);
    if (enable_pmu_) SET_PROFILING_FLAG(enable_profiling_flag, PROFILING_FLAG_PMU);
    if (enable_dep_gen_) SET_PROFILING_FLAG(enable_profiling_flag, PROFILING_FLAG_DEP_GEN);
    if (enable_scope_stats_) SET_PROFILING_FLAG(enable_profiling_flag, PROFILING_FLAG_SCOPE_STATS);
    kernel_args_.args.enable_profiling_flag = enable_profiling_flag;

    if (prepare_runtime_for_launch(runtime, block_dim, launch_aicpu_num) != 0) return -1;

    // a5-specific: probe the AICPU topology + compute ALLOWED_CPUS for the
    // filter-style gate (see src/common/platform/onboard/aicpu/
    // platform_aicpu_affinity.cpp::platform_aicpu_affinity_gate_filter).
    // Convention: indices 0..n_sched-1 = sched slots, last = orch slot.
    // n_sched = launch_aicpu_num - 1 (one orch + the rest sched). When
    // launch_aicpu_num == 1 (init-only path) we leave allowed_cpus empty —
    // the gate is a no-op for a single thread.
    {
        std::vector<pto::a5::AicpuLogicalCpu> user_cpus;
        std::vector<int32_t> allowed;
        const int32_t n_orch = 1;
        const int32_t n_sched = (launch_aicpu_num > 1) ? (launch_aicpu_num - n_orch) : 0;
        runtime.aicpu_allowed_cpu_count = 0;
        if (n_sched > 0) {
            // The affinity gate is a performance optimization (ready-queue
            // locality), not a correctness requirement. On a5 SKUs/drivers
            // whose CPU_TOPO query is unsupported the probe fails
            // (halGetDeviceInfoByBuff rc=65534 / dsmi_get_device_info rc=27).
            // Degrade to the pre-#963 behavior — leave aicpu_allowed_cpu_count
            // and aicpu_launch_count at 0, which the device kernel reads as
            // "gate disabled": no CPU pinning, launch the caller-requested
            // thread count, roles assigned by arrival order. Do NOT fail the
            // run — only probe-incapable hardware reaches here, and CANN still
            // schedules the AICPU threads correctly without the gate.
            if (!pto::a5::probe_aicpu_topology(static_cast<uint32_t>(device_id_), user_cpus)) {
                LOG_WARN(
                    "AICPU topology probe failed; disabling affinity gate "
                    "(no CPU pinning, launching %d AICPU threads unpinned)", launch_aicpu_num
                );
            } else if (!pto::a5::compute_allowed_cpus(user_cpus, n_sched, n_orch, allowed)) {
                LOG_ERROR(
                    "AICPU topology has %zu user cpus, cannot fit %d sched + %d orch", user_cpus.size(), n_sched, n_orch
                );
                return -1;
            } else {
                const size_t cap = sizeof(runtime.aicpu_allowed_cpus) / sizeof(runtime.aicpu_allowed_cpus[0]);
                if (allowed.size() > cap) {
                    LOG_ERROR("compute_allowed_cpus returned %zu > cap %zu", allowed.size(), cap);
                    return -1;
                }
                for (size_t i = 0; i < allowed.size(); ++i)
                    runtime.aicpu_allowed_cpus[i] = allowed[i];
                runtime.aicpu_allowed_cpu_count = static_cast<int32_t>(allowed.size());
                // Launch one AICPU thread per OCCUPY-visible user cpu so CANN
                // spreads exactly across the user pool — over-subscription on a
                // SKU with fewer user cpus than the compile-time bound deadlocks
                // the production AICPU kernel. Capped by the compile-time array
                // sizing in case the SKU exceeds expectation.
                int32_t launch_n = static_cast<int32_t>(user_cpus.size());
                if (launch_n > PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH) {
                    launch_n = PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH;
                }
                runtime.aicpu_launch_count = launch_n;
                std::string dump;
                for (size_t i = 0; i < allowed.size(); ++i) {
                    if (i) dump += ", ";
                    dump += std::to_string(allowed[i]);
                    if (i + 1 == allowed.size()) dump += "(orch)";
                }
                LOG_INFO_V0(
                    "AICPU ALLOWED_CPUS = [%s] (n_sched=%d, n_orch=%d, launch=%d, user_cpus=%zu)", dump.c_str(), n_sched,
                    n_orch, launch_n, user_cpus.size()
                );
            }
        }
    }

    // Scope guards for cleanup on all exit paths
    auto regs_cleanup = RAIIScopeGuard([this]() {
        if (kernel_args_.args.regs != 0) {
            mem_alloc_.free(reinterpret_cast<void *>(kernel_args_.args.regs));
            kernel_args_.args.regs = 0;
        }
    });

    auto runtime_args_cleanup = RAIIScopeGuard([this]() {
        kernel_args_.finalize_device_kernel_args();
        kernel_args_.finalize_runtime_args();
    });

    // Initialize per-subsystem shared memory.
    if (enable_l2_swimlane_) {
        rc = init_l2_swimlane(num_aicore, runtime.aicpu_thread_num, device_id_);
        if (rc != 0) {
            LOG_ERROR("init_l2_swimlane failed: %d", rc);
            return rc;
        }
    }

    if (enable_dump_tensor_) {
        rc = init_tensor_dump(runtime, device_id_);
        if (rc != 0) {
            LOG_ERROR("init_tensor_dump failed: %d", rc);
            return rc;
        }
    }

    if (enable_pmu_) {
        rc = init_pmu(num_aicore, launch_aicpu_num, make_pmu_csv_path(output_prefix_), pmu_event_type_, device_id_);
        if (rc != 0) {
            LOG_ERROR("PMU init failed: %d, disabling PMU for this run", rc);
            kernel_args_.args.pmu_data_base = 0;
            enable_pmu_ = false;
        }
    }

    if (enable_dep_gen_) {
        rc = init_dep_gen(launch_aicpu_num, device_id_);
        if (rc != 0) {
            LOG_ERROR("init_dep_gen failed: %d", rc);
            return rc;
        }
    }

    if (enable_scope_stats_) {
        rc = init_scope_stats(launch_aicpu_num, device_id_);
        if (rc != 0) {
            LOG_ERROR("init_scope_stats failed: %d", rc);
            return rc;
        }
    }

    // Cleanup guard for early returns: stops all started collectors so
    // their mgmt + poll threads exit cleanly. stop() is idempotent and a
    // no-op on collectors that never started.
    auto perf_cleanup = RAIIScopeGuard([this]() {
        finalize_collectors();
    });

    LOG_INFO_V0("=== Initialize runtime args ===");
    rc = prepare_orch_so(runtime);
    if (rc != 0) {
        LOG_ERROR("prepare_orch_so failed: %d", rc);
        return rc;
    }
    rc = init_runtime_args_with_metadata(runtime);
    if (rc != 0) return rc;

    start_shared_collectors_for_run();
    // a5-specific dep_gen collector — share the same thread_factory shape as base.
    if (enable_dep_gen_) {
        auto thread_factory = [this](std::function<void()> fn) {
            return create_thread(std::move(fn));
        };
        dep_gen_collector_.start(thread_factory);
    }

    LOG_INFO_V0("=== launch_aicpu_kernel %s ===", host::KernelNames::InitName);
    rc = launch_aicpu_kernel(stream_aicpu_, &kernel_args_.args, host::KernelNames::InitName, 1);
    if (rc != 0) {
        LOG_ERROR("launch_aicpu_kernel (init) failed: %d", rc);
        return rc;
    }

    // Init kernel populated runtime.workers[i].core_type via the AICore
    // handshake. Publish the table to the L2 swimlane collector so the
    // AICORE_TIMING (level=1) host emit path can label lanes ("aic"/"aiv")
    // without consulting an AICPU record.
    if (enable_l2_swimlane_ && l2_swimlane_collector_.is_initialized()) {
        std::vector<CoreType> core_types(num_aicore);
        for (int i = 0; i < num_aicore; i++) {
            core_types[i] = runtime.workers[i].core_type;
        }
        l2_swimlane_collector_.set_core_types(core_types.data(), num_aicore);
    }

    LOG_INFO_V0("=== launch_aicpu_kernel %s ===", host::KernelNames::RunName);
    // launch_count = popcount(OCCUPY) from the topology probe — one thread
    // per user-schedulable cpu_id. The filter gate barriers exactly this
    // many threads (runtime.aicpu_launch_count is read on the device side
    // by kernel.cpp). Fall back to the caller-requested active count when
    // the probe was skipped (single-thread init / launch_aicpu_num == 1)
    // — over-launching to the compile-time bound on that path would
    // start 14 threads to do a 1-thread job and deadlock the device.
    int aicpu_launch_n = (runtime.aicpu_launch_count > 0) ? runtime.aicpu_launch_count : launch_aicpu_num;
    rc = launch_aicpu_kernel(stream_aicpu_, &kernel_args_.args, host::KernelNames::RunName, aicpu_launch_n);
    if (rc != 0) {
        LOG_ERROR("launch_aicpu_kernel (main) failed: %d", rc);
        return rc;
    }

    LOG_INFO_V0("=== launch_aicore_kernel ===");
    rc = kernel_args_.init_device_kernel_args(mem_alloc_);
    if (rc != 0) {
        LOG_ERROR("init_device_kernel_args failed: %d", rc);
        return rc;
    }
    rc = launch_aicore_kernel(stream_aicore_, kernel_args_.device_k_args_);
    if (rc != 0) {
        LOG_ERROR("launch_aicore_kernel failed: %d", rc);
        recover_device_or_mark_unusable(rc);
        return rc;
    }

    rc = sync_run_streams();
    if (rc != 0) {
        // sync_run_streams surfaces the AICore op-timeout (STARS-reaped op ->
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
        teardown_shared_collectors_after_run();
        return rc;
    }

    read_device_wall_ns();

    teardown_shared_collectors_after_run();

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

    // Print handshake results (reads from device memory, must be before free)
    print_handshake_results();

    return 0;
}

void DeviceRunner::recover_device_or_mark_unusable(int aicore_rc) {
    // An AICore launch failure (207001) or an op-timeout reaped by STARS
    // (surfaced as 507000/507018/507046 at stream sync) leaves the device
    // context in a sticky-error state: the streams stay poisoned and the
    // SAME DeviceRunner's next run() fails early — observed on a5 as
    // `halResMap failed (rc=62)` in init_aicore_register_addresses, and on
    // a2a3 as `rtMalloc failed: 507899`. Reused across a session (the L2
    // st_worker pool hands one ChipWorker to every test class on a device),
    // that one error poisons every later test in the xdist worker process.
    //
    // Try to drain the device so a subsequent run can recover in place. Use
    // the BOUNDED aclrtSynchronizeDeviceWithTimeout, NOT an unbounded
    // aclrtSynchronizeStream* on the error-state stream — the latter wedges
    // subsequent tests (see DeviceRunnerBase::finalize_common's comment on
    // why finalize deliberately skips a pre-destroy sync). If the bounded
    // device drain itself errors, the context is unrecoverable without a full
    // device reset, so mark the runner unusable: run() then fails fast with a
    // clear message instead of cascading, and the fixture rebuilds the Worker.
    int sync_rc = aclrtSynchronizeDeviceWithTimeout(PLATFORM_STREAM_SYNC_TIMEOUT_MS);
    if (sync_rc != ACL_SUCCESS) {
        LOG_ERROR(
            "Device unrecoverable after AICore error %d: aclrtSynchronizeDeviceWithTimeout failed: %d. "
            "Marking DeviceRunner unusable; run() will fail fast until a fresh worker process runs "
            "(in-process reset does not clear the poison on a5).",
            aicore_rc, sync_rc
        );
        device_unusable_ = true;
    } else {
        LOG_WARN("AICore error %d: device drained via aclrtSynchronizeDeviceWithTimeout", aicore_rc);
    }
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

void DeviceRunner::force_reset_device() {
    if (device_id_ < 0) {
        return;
    }
    // aclrtResetDeviceForce is an ACL API; bring ACL up and bind the device,
    // both released on scope exit (LIFO) so a repeated poison-then-reset cycle
    // in a long-lived process leaks neither ACL state nor a device reference.
    AclInitGuard acl_guard;
    if (!acl_guard.ok()) {
        return;
    }
    DeviceBindGuard bind_guard(device_id_);
    if (!bind_guard.bound()) {
        return;
    }
    aclError rc = aclrtResetDeviceForce(device_id_);
    if (rc != ACL_SUCCESS) {
        LOG_ERROR("force_reset_device: aclrtResetDeviceForce(%d) failed: %d", device_id_, static_cast<int>(rc));
    } else {
        LOG_WARN("force_reset_device: aclrtResetDeviceForce(%d) cleared the poisoned card", device_id_);
    }
}

// `print_handshake_results`, `prepare_orch_so`, `register_callable`,
// `register_callable_host_orch`, `unregister_callable`, `has_callable`,
// `bind_callable_to_runtime`, and `upload_chip_callable_buffer` live on
// `DeviceRunnerBase`.

int DeviceRunner::finalize() {
    if (device_id_ == -1) {
        return 0;
    }

    int rc = attach_current_thread(device_id_);
    if (rc != 0) {
        LOG_ERROR("Failed to attach finalize thread to device %d: %d", device_id_, rc);
        return rc;
    }

    // Cleanup all profiling subsystems (free shm + per-buffer dev/host
    // shadows). All four shared collectors use the same alloc/free shape
    // on a5: no unregister callback (a5 doesn't use halHostRegister) +
    // prof_free_cb (rtFree directly).
    if (l2_swimlane_collector_.is_initialized()) {
        l2_swimlane_collector_.finalize(/*unregister_cb=*/nullptr, prof_free_cb);
    }
    if (dump_collector_.is_initialized()) {
        dump_collector_.finalize(/*unregister_cb=*/nullptr, prof_free_cb);
    }
    if (pmu_collector_.is_initialized()) {
        pmu_collector_.finalize(/*unregister_cb=*/nullptr, prof_free_cb);
    }
    if (dep_gen_collector_.is_initialized()) {
        dep_gen_collector_.finalize(/*unregister_cb=*/nullptr, prof_free_cb);
    }
    if (scope_stats_collector_.is_initialized()) {
        scope_stats_collector_.finalize(/*unregister_cb=*/nullptr, prof_free_cb);
        kernel_args_.args.scope_stats_data_base = 0;
    }

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

    // On the poison path the soft reset above does NOT clear the op-timeout
    // sticky-error — a fresh in-process Worker.init then fails at rtStreamCreate
    // 507899. A FORCE reset clears it, so the next Worker on this card inits
    // clean in the SAME process and the remaining tests run instead of cascading
    // / being skipped. Only reached on the (rare) device-poison path; onboard
    // work always holds an exclusive task-submit lock on the card (enforced by
    // .claude/rules/running-onboard.md), and the reset is verified to scope to
    // this card alone, so it cannot disturb other devices/users.
    if (device_unusable_) {
        force_reset_device();
    }

    device_id_ = -1;
    // A finalized runner has reset the device; clear the poison flag so a
    // reused DeviceRunner object (re-init after close) starts clean.
    device_unusable_ = false;
    LOG_INFO_V0("DeviceRunner finalized");
    return rc;
}

// `launch_aicpu_kernel` and `launch_aicore_kernel` live on `DeviceRunnerBase`.

void DeviceRunner::finalize_collectors() {
    if (l2_swimlane_collector_.is_initialized()) {
        l2_swimlane_collector_.stop();
    }
    if (dump_collector_.is_initialized()) {
        dump_collector_.stop();
    }
    if (pmu_collector_.is_initialized()) {
        pmu_collector_.stop();
    }
    if (dep_gen_collector_.is_initialized()) {
        dep_gen_collector_.stop();
    }
}

int DeviceRunner::init_l2_swimlane(int num_aicore, int aicpu_thread_num, int device_id) {
    int rc = l2_swimlane_collector_.initialize(
        num_aicore, aicpu_thread_num, device_id, l2_swimlane_level_, prof_alloc_cb, /*register_cb=*/nullptr,
        prof_free_cb, output_prefix_
    );
    if (rc == 0) {
        kernel_args_.args.l2_swimlane_data_base =
            reinterpret_cast<uint64_t>(l2_swimlane_collector_.get_l2_swimlane_setup_device_ptr());
        kernel_args_.args.l2_swimlane_aicore_rotation_table =
            reinterpret_cast<uint64_t>(l2_swimlane_collector_.get_aicore_ring_addr_table_device_ptr());
    }
    return rc;
}

int DeviceRunner::init_tensor_dump(Runtime &runtime, int device_id) {
    int num_dump_threads = runtime.aicpu_thread_num;

    int rc = dump_collector_.initialize(
        num_dump_threads, device_id, prof_alloc_cb, /*register_cb=*/nullptr, prof_free_cb, output_prefix_,
        dump_tensor_level_
    );
    if (rc != 0) {
        return rc;
    }

    kernel_args_.args.dump_data_base = reinterpret_cast<uint64_t>(dump_collector_.get_dump_shm_device_ptr());
    return 0;
}

int DeviceRunner::init_pmu(
    int num_cores, int num_threads, const std::string &csv_path, PmuEventType event_type, int device_id
) {
    int rc = pmu_collector_.init(
        num_cores, num_threads, csv_path, event_type, prof_alloc_cb, /*register_cb=*/nullptr, prof_free_cb, device_id
    );
    if (rc == 0) {
        kernel_args_.args.pmu_data_base = reinterpret_cast<uint64_t>(pmu_collector_.get_pmu_shm_device_ptr());
        kernel_args_.args.aicore_pmu_ring_addrs =
            reinterpret_cast<uint64_t>(pmu_collector_.get_aicore_ring_addrs_device_ptr());
    }
    return rc;
}

int DeviceRunner::init_scope_stats(int num_threads, int device_id) {
    // a5: register_cb=nullptr, so the collector mallocs a host shadow per
    // device buffer + rtMemcpy's the zeroed shadow to device (see
    // ProfilerBase::alloc_paired_buffer). No halHostRegister on a5.
    int rc = scope_stats_collector_.init(num_threads, prof_alloc_cb, /*register_cb=*/nullptr, prof_free_cb, device_id);
    if (rc != 0) {
        return rc;
    }
    kernel_args_.args.scope_stats_data_base =
        reinterpret_cast<uint64_t>(scope_stats_collector_.get_scope_stats_shm_device_ptr());
    return 0;
}

int DeviceRunner::init_dep_gen(int num_threads, int device_id) {
    // a5: register_cb=nullptr, so the collector mallocs a host shadow per
    // device buffer + rtMemcpy's the zeroed shadow to device. No
    // halHostRegister on a5 (matches PMU / L2 swimlane / dump collectors).
    int rc = dep_gen_collector_.init(num_threads, prof_alloc_cb, /*register_cb=*/nullptr, prof_free_cb, device_id);
    if (rc != 0) {
        return rc;
    }
    kernel_args_.args.dep_gen_data_base = reinterpret_cast<uint64_t>(dep_gen_collector_.get_dep_gen_shm_device_ptr());
    return 0;
}
