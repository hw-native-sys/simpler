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

#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include "acl/acl.h"

// Include HAL constants from CANN (header only, library loaded dynamically)
#include "ascend_hal.h"
#include "callable.h"
#include "callable_protocol.h"
#include "chip_callable_layout.h"
#include "utils/elf_build_id.h"
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
// Lazy-loaded HAL (ascend_hal) for profiling host-register only
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

// `ensure_binaries_loaded`, `query_max_block_dim`, and `validate_block_dim`
// live on `DeviceRunnerBase` — see
// `src/common/platform/onboard/host/device_runner_base.cpp`.

int DeviceRunner::run(Runtime &runtime, int block_dim, int launch_aicpu_num) {
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

    // Scope guards for register-address cleanup on all exit paths. Declared
    // before the allocs so that an alloc-failure early-return still triggers
    // cleanup of previously-allocated buffers (the predicates no-op on 0).
    auto regs_cleanup = RAIIScopeGuard([this]() {
        if (kernel_args_.args.regs != 0) {
            mem_alloc_.free(reinterpret_cast<void *>(kernel_args_.args.regs));
            kernel_args_.args.regs = 0;
        }
    });

    auto pmu_regs_cleanup = RAIIScopeGuard([this]() {
        if (kernel_args_.args.pmu_reg_addrs != 0) {
            mem_alloc_.free(reinterpret_cast<void *>(kernel_args_.args.pmu_reg_addrs));
            kernel_args_.args.pmu_reg_addrs = 0;
        }
    });

    // Get AICore register addresses for register-based task dispatch
    rc = init_aicore_register_addresses(
        &kernel_args_.args.regs, static_cast<uint64_t>(device_id_), mem_alloc_, AicoreRegKind::Ctrl
    );
    if (rc != 0) {
        LOG_ERROR("init_aicore_register_addresses(Ctrl) failed: %d", rc);
        return rc;
    }

    // Get AICore PMU register addresses (distinct MMIO page from AIC_CTRL).
    if (enable_pmu_) {
        rc = init_aicore_register_addresses(
            &kernel_args_.args.pmu_reg_addrs, static_cast<uint64_t>(device_id_), mem_alloc_, AicoreRegKind::Pmu
        );
        if (rc != 0) {
            LOG_ERROR("init_aicore_register_addresses(Pmu) failed: %d", rc);
            return rc;
        }
    }

    // Build the profiling-flag bitfield (a2a3 carries an extra dep_gen bit).
    uint32_t enable_profiling_flag = PROFILING_FLAG_NONE;
    if (enable_dump_tensor_) SET_PROFILING_FLAG(enable_profiling_flag, PROFILING_FLAG_DUMP_TENSOR);
    if (enable_l2_swimlane_) SET_PROFILING_FLAG(enable_profiling_flag, PROFILING_FLAG_L2_SWIMLANE);
    if (enable_pmu_) SET_PROFILING_FLAG(enable_profiling_flag, PROFILING_FLAG_PMU);
    if (enable_dep_gen_) SET_PROFILING_FLAG(enable_profiling_flag, PROFILING_FLAG_DEP_GEN);
    if (enable_scope_stats_) SET_PROFILING_FLAG(enable_profiling_flag, PROFILING_FLAG_SCOPE_STATS);
    kernel_args_.args.enable_profiling_flag = enable_profiling_flag;

    if (prepare_runtime_for_launch(runtime, block_dim, launch_aicpu_num) != 0) return -1;

    auto runtime_args_cleanup = RAIIScopeGuard([this]() {
        kernel_args_.finalize_device_kernel_args();
        kernel_args_.finalize_runtime_args();
    });

    // Initialize per-subsystem shared memory.
    if (enable_l2_swimlane_) {
        rc = init_l2_swimlane(num_aicore, device_id_);
        if (rc != 0) {
            LOG_ERROR("init_l2_swimlane failed: %d", rc);
            return rc;
        }
    }

    if (enable_dump_tensor_) {
        // Initialize tensor dump (independent from profiling)
        rc = init_tensor_dump(runtime, device_id_);
        if (rc != 0) {
            LOG_ERROR("init_tensor_dump failed: %d", rc);
            return rc;
        }
    }

    if (enable_pmu_) {
        rc = init_pmu(num_aicore, launch_aicpu_num, make_pmu_csv_path(output_prefix_), pmu_event_type_, device_id_);
        if (rc != 0) {
            LOG_ERROR("init_pmu failed: %d", rc);
            return rc;
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

    // On any exit from run() — success or early error — release the diagnostics
    // collectors' shared memory. They are only re-initialized per run(), so a
    // Worker reused across runs (e.g. a pytest session-scoped worker pool) would
    // otherwise re-enter init_l2_swimlane() with stale state still allocated.
    auto perf_cleanup = RAIIScopeGuard([this]() {
        finalize_collectors();
    });

    LOG_INFO_V0("=== Initialize runtime args ===");
    // Resolve the orchestration SO into a device-resident buffer and refresh
    // runtime metadata before the Runtime struct is uploaded to device.
    rc = prepare_orch_so(runtime);
    if (rc != 0) {
        LOG_ERROR("prepare_orch_so failed: %d", rc);
        return rc;
    }

    rc = init_runtime_args_with_metadata(runtime);
    if (rc != 0) return rc;

    rc = kernel_args_init_ffts_base_addr(kernel_args_);
    if (rc != 0) {
        LOG_ERROR("init_ffts_base_addr failed: %d", rc);
        return rc;
    }

    // Copy KernelArgs to device memory for AICore
    rc = kernel_args_.init_device_kernel_args(mem_alloc_);
    if (rc != 0) {
        LOG_ERROR("init_device_kernel_args failed: %d", rc);
        return rc;
    }

    start_shared_collectors_for_run();
    // a2a3-only dep_gen collector — share the same thread_factory shape as base.
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

    LOG_INFO_V0("=== launch_aicpu_kernel %s ===", host::KernelNames::RunName);
    rc = launch_aicpu_kernel(
        stream_aicpu_, &kernel_args_.args, host::KernelNames::RunName, PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH
    );
    if (rc != 0) {
        LOG_ERROR("launch_aicpu_kernel (main) failed: %d", rc);
        return rc;
    }

    LOG_INFO_V0("=== launch_aicore_kernel ===");
    // Launch AICore kernel (pass device copy of KernelArgs)
    rc = launch_aicore_kernel(stream_aicore_, kernel_args_.device_k_args_);
    if (rc != 0) {
        LOG_ERROR("launch_aicore_kernel failed: %d", rc);
        return rc;
    }

    rc = sync_run_streams();
    if (rc != 0) return rc;

    read_device_wall_ns();

    // Tear down collectors. stop() joins mgmt then collector in the only safe
    // order (mgmt's final-drain pass into L2 has poll as its consumer).
    teardown_shared_collectors_after_run();

    // a2a3-only dep_gen teardown: stop + reconcile + replay emit.
    if (enable_dep_gen_) {
        dep_gen_collector_.stop();
        if (dep_gen_collector_.reconcile_counters()) {
            const auto &records = dep_gen_collector_.records();
            const std::string deps = make_deps_json_path(output_prefix_);
            int rc = dep_gen_replay_emit_deps_json(records.data(), records.size(), deps.c_str());
            if (rc != 0) {
                LOG_ERROR("dep_gen replay failed (%d) — deps.json not produced", rc);
            }
        }
    }

    // Print handshake results (reads from device memory, must be before free)
    print_handshake_results();

    return 0;
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

    // Cleanup performance profiling (including a2a3's dep_gen). Normally
    // already done by run()'s perf_cleanup guard; this is the backstop
    // for the no-run-since-init case.
    finalize_collectors();

    // Shared cleanup body — streams, kernel_args, callable/orch maps,
    // chip-callable buffer pool, the three arenas, device_wall,
    // mem_alloc_.finalize(), and cached arena sizes.
    rc = finalize_common();

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

    device_id_ = -1;
    LOG_INFO_V0("DeviceRunner finalized");
    return rc;
}

// `launch_aicpu_kernel` and `launch_aicore_kernel` live on `DeviceRunnerBase`.

int DeviceRunner::init_l2_swimlane(int num_aicore, int device_id) {
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

    int rc = l2_swimlane_collector_.initialize(
        num_aicore, device_id, l2_swimlane_level_, alloc_cb, register_cb, free_cb, output_prefix_
    );
    if (rc != 0) {
        return rc;
    }

    kernel_args_.args.l2_swimlane_data_base =
        reinterpret_cast<uint64_t>(l2_swimlane_collector_.get_l2_swimlane_setup_device_ptr());
    kernel_args_.args.l2_swimlane_aicore_rotation_table =
        reinterpret_cast<uint64_t>(l2_swimlane_collector_.get_aicore_ring_addr_table_device_ptr());
    return 0;
}

int DeviceRunner::init_tensor_dump(Runtime &runtime, int device_id) {
    int num_dump_threads = runtime.aicpu_thread_num;

    auto alloc_cb = [this](size_t size) -> void * {
        return mem_alloc_.alloc(size);
    };

    auto register_cb = [](void *dev_ptr, size_t size, int device_id, void **host_ptr) -> int {
        if (load_hal_if_needed() != 0) {
            LOG_ERROR("Failed to load ascend_hal for tensor dump: %s", dlerror());
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

    int rc = dump_collector_.initialize(num_dump_threads, device_id, alloc_cb, register_cb, free_cb, output_prefix_);
    if (rc != 0) {
        return rc;
    }

    kernel_args_.args.dump_data_base = reinterpret_cast<uint64_t>(dump_collector_.get_dump_shm_device_ptr());
    return 0;
}

int DeviceRunner::init_pmu(
    int num_cores, int num_threads, const std::string &csv_path, PmuEventType event_type, int device_id
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

    kernel_args_.args.pmu_data_base = reinterpret_cast<uint64_t>(pmu_collector_.get_pmu_shm_device_ptr());
    return 0;
}

int DeviceRunner::init_dep_gen(int num_threads, int device_id) {
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

    kernel_args_.args.dep_gen_data_base = reinterpret_cast<uint64_t>(dep_gen_collector_.get_dep_gen_shm_device_ptr());
    return 0;
}

int DeviceRunner::init_scope_stats(int num_threads, int device_id) {
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

    kernel_args_.args.scope_stats_data_base =
        reinterpret_cast<uint64_t>(scope_stats_collector_.get_scope_stats_shm_device_ptr());
    return 0;
}

void DeviceRunner::finalize_collectors() {
    auto unregister_cb = [](void *dev_ptr, int device_id) -> int {
        HalHostUnregisterFn fn = get_halHostUnregister();
        if (fn != nullptr) {
            return fn(dev_ptr, device_id);
        }
        return 0;
    };
    auto free_cb = [this](void *dev_ptr) -> int {
        return mem_alloc_.free(dev_ptr);
    };

    if (l2_swimlane_collector_.is_initialized()) {
        l2_swimlane_collector_.finalize(unregister_cb, free_cb);
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
        kernel_args_.args.scope_stats_data_base = 0;
    }
}
