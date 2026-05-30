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
 * `DeviceRunnerBase` — onboard host lifecycle shared by a2a3 and a5.
 *
 * Constructor wires the three arenas to call back into `mem_alloc_` via
 * the static trampolines declared in the header. Per-region commit is
 * still driven by the subclass's `setup_static_arena`.
 *
 * Each lifecycle method is a verbatim move of code that was identical
 * between `src/{a2a3,a5}/platform/onboard/host/device_runner.cpp` —
 * the implementations have already been validated by the production CI
 * for both arches. No behavioral changes here; this is a pure
 * deduplication pass.
 */

#include "device_runner_base.h"

#include <runtime/rt.h>
#include <acl/acl.h>

#include <algorithm>
#include <cstdint>

#include "common/platform_config.h"
#include "common/unified_log.h"
#include "host_log.h"
// `runtime.h` (pulled in via `device_runner_helpers.h` in the base header)
// supplies the per-arch `Handshake` definition used by
// `print_handshake_results`.

DeviceRunnerBase::DeviceRunnerBase() :
    gm_heap_arena_(&arena_alloc_trampoline, &arena_free_trampoline, &mem_alloc_),
    gm_sm_arena_(&arena_alloc_trampoline, &arena_free_trampoline, &mem_alloc_),
    runtime_arena_pool_(&arena_alloc_trampoline, &arena_free_trampoline, &mem_alloc_) {}

void *DeviceRunnerBase::allocate_tensor(std::size_t bytes) { return mem_alloc_.alloc(bytes); }

void DeviceRunnerBase::free_tensor(void *dev_ptr) {
    if (dev_ptr != nullptr) {
        mem_alloc_.free(dev_ptr);
    }
}

int DeviceRunnerBase::copy_to_device(void *dev_ptr, const void *host_ptr, std::size_t bytes) {
    return rtMemcpy(dev_ptr, bytes, host_ptr, bytes, RT_MEMCPY_HOST_TO_DEVICE);
}

int DeviceRunnerBase::copy_from_device(void *host_ptr, const void *dev_ptr, std::size_t bytes) {
    return rtMemcpy(host_ptr, bytes, dev_ptr, bytes, RT_MEMCPY_DEVICE_TO_HOST);
}

void *DeviceRunnerBase::acquire_pooled_gm_heap() {
    if (!gm_heap_arena_.is_committed()) return nullptr;
    return gm_heap_arena_.base();
}

void *DeviceRunnerBase::acquire_pooled_gm_sm() {
    if (!gm_sm_arena_.is_committed()) return nullptr;
    return gm_sm_arena_.base();
}

void *DeviceRunnerBase::acquire_pooled_runtime_arena() {
    // hbg calls setup_static_arena(...,0) and leaves runtime_arena_pool_
    // uncommitted — fail loudly if a caller asks for it anyway.
    if (!runtime_arena_pool_.is_committed()) return nullptr;
    return runtime_arena_pool_.base();
}

std::thread DeviceRunnerBase::create_thread(std::function<void()> fn) {
    int dev_id = device_id_;
    return std::thread([dev_id, fn = std::move(fn)]() {
        rtSetDevice(dev_id);
        fn();
    });
}

int DeviceRunnerBase::attach_current_thread(int device_id) {
    if (device_id < 0) {
        LOG_ERROR("Invalid device_id: %d", device_id);
        return -1;
    }
    if (device_id_ != -1 && device_id_ != device_id) {
        LOG_ERROR(
            "DeviceRunner already initialized on device %d; reset/finalize before switching to device %d", device_id_,
            device_id
        );
        return -1;
    }

    // CANN device context is per-thread, so every caller must attach explicitly.
    int rc = rtSetDevice(device_id);
    if (rc != 0) {
        LOG_ERROR("rtSetDevice(%d) failed: %d", device_id, rc);
        return rc;
    }

    if (device_id_ == -1) {
        configure_aicore_op_timeout();
    }

    device_id_ = device_id;
    return 0;
}

void DeviceRunnerBase::configure_aicore_op_timeout() {
    uint64_t actual_timeout = 0;
    int rc = aclrtSetOpExecuteTimeOutV2(PLATFORM_OP_EXECUTE_TIMEOUT_US, &actual_timeout);
    if (rc != 0) {
        LOG_ERROR(
            "aclrtSetOpExecuteTimeOutV2(%llu us) failed: %d", (unsigned long long)PLATFORM_OP_EXECUTE_TIMEOUT_US, rc
        );
    } else {
        LOG_INFO_V0(
            "aclrtSetOpExecuteTimeOutV2: requested=%llu us, actual=%llu us",
            (unsigned long long)PLATFORM_OP_EXECUTE_TIMEOUT_US, (unsigned long long)actual_timeout
        );
    }
}

int DeviceRunnerBase::ensure_device_initialized() {
    // Attach the current thread to the device (device_id_ was set in
    // attach_current_thread() during simpler_init) and create the persistent
    // AICPU/AICore streams. Streams live for the DeviceRunner's lifetime and
    // are destroyed in finalize().
    int rc = attach_current_thread(device_id_);
    if (rc != 0) {
        return rc;
    }

    bool aicpu_created_here = false;
    bool aicore_created_here = false;
    if (stream_aicpu_ == nullptr) {
        rc = rtStreamCreate(&stream_aicpu_, 0);
        if (rc != 0) {
            LOG_ERROR("rtStreamCreate (AICPU) failed: %d", rc);
            return rc;
        }
        aicpu_created_here = true;
    }
    if (stream_aicore_ == nullptr) {
        rc = rtStreamCreate(&stream_aicore_, 0);
        if (rc != 0) {
            LOG_ERROR("rtStreamCreate (AICore) failed: %d", rc);
            // Roll back only the AICPU stream we just created, not a
            // pre-existing persistent one.
            if (aicpu_created_here) {
                rtStreamDestroy(stream_aicpu_);
                stream_aicpu_ = nullptr;
            }
            return rc;
        }
        aicore_created_here = true;
    }
    if (aicpu_created_here || aicore_created_here) {
        LOG_INFO_V0("DeviceRunner: device=%d set, streams created", device_id_);
    }

    return ensure_binaries_loaded();
}

int DeviceRunnerBase::ensure_binaries_loaded() {
    // Check if already loaded (binaries are owned by the runner via
    // set_executors and live for the runner's lifetime).
    if (binaries_loaded_) {
        return 0;
    }

    // Device must be set first
    if (stream_aicpu_ == nullptr) {
        LOG_ERROR("Device not set before loading binaries");
        return -1;
    }

    if (dispatcher_so_binary_.empty()) {
        LOG_ERROR(
            "DeviceRunner: dispatcher SO bytes not provided; pass dispatcher_path through ChipWorker.init "
            "(RuntimeBinaries.dispatcher_path)"
        );
        return -1;
    }

    // One-shot bootstrap: libaicpu_extend_kernels invokes our dispatcher,
    // which writes the runtime AICPU SO bytes to
    // simpler_inner_<fp>_<device_id>.so in the device-side preinstall path.
    // The dispatcher SO itself is never persisted to disk — only the
    // transient libaicpu_extend_kernels dlopen. Subsequent per-task AICPU
    // launches resolve symbols via rtsBinaryLoadFromFile + rtsFuncGetByName +
    // rtsLaunchCpuKernel directly against the preinstall file.
    int rc = load_aicpu_op_.BootstrapDispatcher(
        dispatcher_so_binary_.data(), dispatcher_so_binary_.size(), aicpu_so_binary_.data(), aicpu_so_binary_.size(),
        stream_aicpu_, device_id_
    );
    if (rc != 0) {
        LOG_ERROR("LoadAicpuOp::BootstrapDispatcher failed: %d", rc);
        return rc;
    }
    LOG_INFO_V2("DeviceRunner: inner SO uploaded to preinstall via dispatcher bootstrap");

    // JSON-register the inner SO and resolve simpler_aicpu_init / _exec handles.
    rc = load_aicpu_op_.Init();
    if (rc != 0) {
        LOG_ERROR("LoadAicpuOp::Init failed: %d", rc);
        return rc;
    }
    LOG_INFO_V2("DeviceRunner: inner SO registered (simpler_aicpu_init/exec handles ready)");

    // H2D the per-task DeviceArgs struct itself. device_args_.aicpu_so_bin/len
    // stay zero — our own per-task AICPU code (launched via rtsLaunchCpuKernel
    // against the cached rtFuncHandle on LoadAicpuOp) never reads them, and
    // the dispatcher-bootstrap KernelArgs (KERNEL_TYPE_AICPU_KFC) builds its
    // own DeviceArgs view inside BootstrapDispatcher rather than reading
    // ours. The "load-bearing on a5" finding documented prior to #864/#870
    // no longer reproduces against current HEAD — see PR removing
    // AicpuSoInfo (CI on both archs green).
    rc = kernel_args_.init_device_args(device_args_, mem_alloc_);
    if (rc != 0) {
        LOG_ERROR("init_device_args failed: %d", rc);
        return rc;
    }

    // Release host bytes — bootstrap is done. Per-task launches go through
    // the cached rtFuncHandle owned by LoadAicpuOp; dispatcher SO bytes are
    // never referenced again; the aicpu kernel SO's host buffer is no longer
    // needed either (we used to H2D it through AicpuSoInfo as a CANN-internal
    // bookkeeping workaround; that's gone).
    dispatcher_so_binary_.clear();
    dispatcher_so_binary_.shrink_to_fit();
    aicpu_so_binary_.clear();
    aicpu_so_binary_.shrink_to_fit();

    binaries_loaded_ = true;
    LOG_INFO_V0("DeviceRunner: binaries loaded");
    return 0;
}

int DeviceRunnerBase::query_max_block_dim(rtStream_t stream, uint32_t *out_cube, uint32_t *out_vector) {
    uint32_t cube_limit = 0, vector_limit = 0;
    bool got_limits = (aclrtGetStreamResLimit(stream, ACL_RT_DEV_RES_CUBE_CORE, &cube_limit) == ACL_ERROR_NONE) &&
                      (aclrtGetStreamResLimit(stream, ACL_RT_DEV_RES_VECTOR_CORE, &vector_limit) == ACL_ERROR_NONE) &&
                      cube_limit > 0 && vector_limit > 0;
    if (out_cube != nullptr) *out_cube = got_limits ? cube_limit : 0;
    if (out_vector != nullptr) *out_vector = got_limits ? vector_limit : 0;
    if (got_limits) {
        // Cap by PLATFORM_MAX_BLOCKDIM as well: runtime handshake/scheduler
        // arrays are statically sized to RUNTIME_MAX_WORKER (= PLATFORM_MAX_BLOCKDIM
        // * PLATFORM_CORES_PER_BLOCKDIM), so even if ACL reports more cores
        // than the platform cap we must not exceed it.
        int from_stream = static_cast<int>(
            std::min(cube_limit / PLATFORM_AIC_CORES_PER_BLOCKDIM, vector_limit / PLATFORM_AIV_CORES_PER_BLOCKDIM)
        );
        return std::min(from_stream, PLATFORM_MAX_BLOCKDIM);
    }
    return PLATFORM_MAX_BLOCKDIM;
}

int DeviceRunnerBase::validate_block_dim(rtStream_t stream, int block_dim) {
    if (block_dim < 1) {
        LOG_ERROR("block_dim (%d) must be >= 1", block_dim);
        return -1;
    }
    uint32_t cube_limit = 0, vector_limit = 0;
    int max_bd = query_max_block_dim(stream, &cube_limit, &vector_limit);
    if (block_dim > max_bd) {
        if (cube_limit > 0 && vector_limit > 0) {
            LOG_ERROR(
                "block_dim (%d) exceeds available cores (max_block_dim=%d, cube=%u, vector=%u)", block_dim, max_bd,
                cube_limit, vector_limit
            );
        } else {
            LOG_ERROR(
                "aclrtGetStreamResLimit unavailable; block_dim (%d) exceeds static cap PLATFORM_MAX_BLOCKDIM (%d)",
                block_dim, PLATFORM_MAX_BLOCKDIM
            );
        }
        return -1;
    }
    return 0;
}

void DeviceRunnerBase::print_handshake_results() {
    if (stream_aicpu_ == nullptr || worker_count_ == 0 || kernel_args_.args.runtime_args == nullptr) {
        return;
    }

    // Allocate temporary buffer to read handshake data from device
    std::vector<Handshake> workers(worker_count_);
    size_t total_size = sizeof(Handshake) * worker_count_;
    rtMemcpy(workers.data(), total_size, kernel_args_.args.runtime_args->workers, total_size, RT_MEMCPY_DEVICE_TO_HOST);

    LOG_DEBUG("Handshake results for %d cores:", worker_count_);
    for (int i = 0; i < worker_count_; i++) {
        LOG_DEBUG(
            "  Core %d: aicore_done=%d aicpu_ready=%d task=%d", i, workers[i].aicore_done, workers[i].aicpu_ready,
            workers[i].task
        );
    }
}
