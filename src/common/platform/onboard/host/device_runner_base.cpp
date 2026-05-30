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
#include <dlfcn.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>

#include "callable.h"
#include "callable_protocol.h"
#include "chip_callable_layout.h"
#include "common/core_type.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "host_log.h"
#include "utils/elf_build_id.h"
// `runtime.h` (pulled in via `device_runner_helpers.h` in the base header)
// supplies the per-arch `Handshake` + `Runtime` types used by
// `print_handshake_results` / `bind_callable_to_runtime` /
// `prepare_orch_so`.

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

int DeviceRunnerBase::setup_static_arena(size_t gm_heap_size, size_t gm_sm_size, size_t runtime_arena_size) {
    // Three independent device_malloc'd buffers: GM heap, PTO2 SM, prebuilt
    // runtime arena. Split out from a single large allocation because the
    // combined size can exceed the device allocator's largest contiguous
    // block. Each arena commits exactly one region, so its base() is the
    // pooled pointer the caller wants.
    //
    // Idempotent for the production case (sizes do not change across a
    // worker's lifetime). If a caller asks for a larger layout on any
    // region, redo just that region — already-committed peers stay alive
    // so their callers don't have to re-acquire.
    auto commit_region = [](DeviceArena &arena, size_t &cached_size, size_t requested_size) -> int {
        if (requested_size == 0) {
            // hbg's runtime_arena path: caller passed 0 and never reserved
            // a region. Leave the arena uncommitted; acquire_pooled_* will
            // return nullptr.
            if (arena.is_committed() && cached_size != 0) {
                arena.release();
                cached_size = 0;
            }
            return 0;
        }
        if (arena.is_committed() && requested_size <= cached_size) {
            return 0;
        }
        arena.release();
        cached_size = 0;
        arena.reserve(requested_size, DeviceArena::kDefaultBaseAlign);
        if (arena.commit(DeviceArena::kDefaultBaseAlign) == nullptr) {
            // commit() failure leaves committed_=false, so the next entry's
            // is_committed() guard skips the release branch. release() is
            // idempotent on a never-committed arena (zeroes cursor_).
            arena.release();
            return -1;
        }
        cached_size = requested_size;
        return 0;
    };
    // Try to commit all three regions; on any failure, fully roll back —
    // including any earlier-committed peers from a PRIOR successful call.
    // The simpler "only roll back peers from this call" pattern would
    // leave stale committed regions when a re-init (e.g., later worker
    // asking for a larger layout) fails midway, defeating the
    // "failure means failure" guarantee. Reset everything to the
    // post-construction state so the caller can retry with a new layout.
    bool ok = commit_region(gm_heap_arena_, cached_gm_heap_size_, gm_heap_size) == 0;
    ok = ok && commit_region(gm_sm_arena_, cached_gm_sm_size_, gm_sm_size) == 0;
    ok = ok && commit_region(runtime_arena_pool_, cached_runtime_arena_size_, runtime_arena_size) == 0;
    if (!ok) {
        gm_heap_arena_.release();
        gm_sm_arena_.release();
        runtime_arena_pool_.release();
        cached_gm_heap_size_ = 0;
        cached_gm_sm_size_ = 0;
        cached_runtime_arena_size_ = 0;
        return -1;
    }
    return 0;
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

// =============================================================================
// Group D — chip-callable upload + per-callable_id registration
// =============================================================================

uint64_t DeviceRunnerBase::upload_chip_callable_buffer(const ChipCallable *callable) {
    if (callable == nullptr || callable->child_count() == 0) {
        return 0;
    }
    if (stream_aicpu_ == nullptr) {
        LOG_ERROR("Run context not prepared before upload_chip_callable_buffer()");
        return 0;
    }

    const ChipCallableLayout layout = compute_chip_callable_layout(callable);

    // Content-hash dedup: identical bytes → return cached chip_dev.
    auto it = chip_callable_buffers_.find(layout.content_hash);
    if (it != chip_callable_buffers_.end()) {
        LOG_DEBUG(
            "Chip callable dedup hit: chip_dev=0x%lx, size=%zu, hash=0x%lx", it->second.chip_dev, it->second.total_size,
            layout.content_hash
        );
        return it->second.chip_dev;
    }

    void *gm_addr = mem_alloc_.alloc(layout.total_size);
    if (gm_addr == nullptr) {
        LOG_ERROR("Failed to allocate device GM for ChipCallable buffer (size=%zu)", layout.total_size);
        return 0;
    }
    const uint64_t chip_dev = reinterpret_cast<uint64_t>(gm_addr);
    assert((chip_dev & (CALLABLE_ALIGN - 1)) == 0 && "device alloc must be CALLABLE_ALIGN-byte aligned");

    // Build a host scratch with each child's resolved_addr_ fixed up to the
    // device-side address of that child's binary code (so the AICPU dispatch
    // path's `reinterpret_cast<CoreCallable*>(addr)->resolved_addr()` lands
    // on the right device offset).
    std::vector<uint8_t> scratch(layout.total_size);
    std::memcpy(scratch.data(), callable, layout.total_size);
    patch_chip_callable_scratch_for_device(callable, layout, chip_dev, scratch.data());

    int rc = rtMemcpy(gm_addr, layout.total_size, scratch.data(), layout.total_size, RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        LOG_ERROR("rtMemcpy chip callable H2D failed: %d", rc);
        mem_alloc_.free(gm_addr);
        return 0;
    }

    chip_callable_buffers_.emplace(layout.content_hash, ChipCallableBuffer{chip_dev, layout.total_size});
    LOG_DEBUG(
        "Uploaded chip callable: chip_dev=0x%lx, size=%zu, child_count=%d, hash=0x%lx", chip_dev, layout.total_size,
        callable->child_count(), layout.content_hash
    );
    return chip_dev;
}

int DeviceRunnerBase::prepare_orch_so(Runtime &runtime) {
    // Registered-callable flow only: the SO bytes were already H2D'd at
    // register_callable time. Stamp dev_orch_so on the runtime and mark
    // `is_new` based on whether the AICPU has seen this cid since
    // registration.
    const int32_t cid = runtime.get_active_callable_id();
    if (cid < 0) {
        LOG_ERROR("prepare_orch_so: no active callable_id; registered-callable flow required");
        return -1;
    }
    auto it = callables_.find(cid);
    if (it == callables_.end()) {
        LOG_ERROR("prepare_orch_so: callable_id=%d not registered", cid);
        return -1;
    }
    const auto &state = it->second;
    // hbg variant: orch SO never crosses the host/device boundary, so the
    // AICPU does no per-cid dlopen. Skip the orch_so_table_ bookkeeping
    // (and the AICPU dlopen counter) and clear the device-orch metadata.
    if (state.host_dlopen_handle != nullptr) {
        runtime.set_dev_orch_so(0, 0);
        runtime.set_active_callable_id(cid, /*is_new=*/false);
        return 0;
    }
    const bool first_sighting = aicpu_seen_callable_ids_.insert(cid).second;
    if (first_sighting) {
        ++aicpu_dlopen_total_;
    }
    runtime.set_dev_orch_so(state.dev_orch_so_addr, state.dev_orch_so_size);
    // The c_api caller passed is_new=false; refresh with the authoritative
    // first_sighting flag before AICPU consumes register_new_callable_id_.
    runtime.set_active_callable_id(cid, first_sighting);
    LOG_INFO_V0(
        "Orch SO prepared cid=%d hash=0x%lx %zu bytes (is_new=%d)", cid, state.hash, state.dev_orch_so_size,
        first_sighting ? 1 : 0
    );
    return 0;
}

int DeviceRunnerBase::register_callable(
    int32_t callable_id, const void *orch_so_data, size_t orch_so_size, const char *func_name, const char *config_name,
    std::vector<std::pair<int, uint64_t>> kernel_addrs, std::vector<ArgDirection> signature
) {
    // The AICPU executor reserves `orch_so_table_[MAX_REGISTERED_CALLABLE_IDS]`
    // (declared in src/common/task_interface/callable_protocol.h) and indexes
    // it by callable_id; rejecting an out-of-range id here keeps the host and
    // AICPU sides in sync and avoids an OOB access at run time.
    if (callable_id < 0 || callable_id >= MAX_REGISTERED_CALLABLE_IDS) {
        LOG_ERROR("register_callable: callable_id=%d out of range [0, %d)", callable_id, MAX_REGISTERED_CALLABLE_IDS);
        return -1;
    }
    if (orch_so_data == nullptr || orch_so_size == 0) {
        LOG_ERROR("register_callable: empty orch SO for callable_id=%d", callable_id);
        return -1;
    }
    if (callables_.count(callable_id) != 0) {
        LOG_ERROR("register_callable: callable_id=%d already registered", callable_id);
        return -1;
    }

    const uint64_t hash = simpler::common::utils::elf_build_id_64(orch_so_data, orch_so_size);

    // Hash dedup: share device buffer across callable_ids that carry the same
    // SO bytes. Refcount drops in unregister_callable; we only free when the
    // count hits zero.
    auto buf_it = orch_so_dedup_.find(hash);
    uint64_t dev_addr = 0;
    if (buf_it == orch_so_dedup_.end()) {
        void *buf = mem_alloc_.alloc(orch_so_size);
        if (buf == nullptr) {
            LOG_ERROR("register_callable: alloc %zu bytes failed", orch_so_size);
            return -1;
        }
        int rc = rtMemcpy(buf, orch_so_size, orch_so_data, orch_so_size, RT_MEMCPY_HOST_TO_DEVICE);
        if (rc != 0) {
            LOG_ERROR("register_callable: rtMemcpy failed: %d", rc);
            mem_alloc_.free(buf);
            return rc;
        }
        OrchSoBuffer entry;
        entry.dev_addr = buf;
        entry.capacity = orch_so_size;
        entry.refcount = 1;
        orch_so_dedup_.emplace(hash, entry);
        dev_addr = reinterpret_cast<uint64_t>(buf);
        LOG_INFO_V0("register_callable: hash=0x%lx new buffer %zu bytes", hash, orch_so_size);
    } else {
        buf_it->second.refcount++;
        dev_addr = reinterpret_cast<uint64_t>(buf_it->second.dev_addr);
        LOG_INFO_V0("register_callable: hash=0x%lx shared buffer (refcount=%d)", hash, buf_it->second.refcount);
    }

    CallableState state;
    state.hash = hash;
    state.dev_orch_so_addr = dev_addr;
    state.dev_orch_so_size = orch_so_size;
    state.func_name = (func_name != nullptr) ? func_name : "";
    state.config_name = (config_name != nullptr) ? config_name : "";
    state.kernel_addrs = std::move(kernel_addrs);
    state.signature = std::move(signature);
    callables_.emplace(callable_id, std::move(state));
    return 0;
}

int DeviceRunnerBase::register_callable_host_orch(
    int32_t callable_id, void *host_dlopen_handle, void *host_orch_func_ptr,
    std::vector<std::pair<int, uint64_t>> kernel_addrs, std::vector<ArgDirection> signature
) {
    if (callable_id < 0 || callable_id >= MAX_REGISTERED_CALLABLE_IDS) {
        LOG_ERROR(
            "register_callable_host_orch: callable_id=%d out of range [0, %d)", callable_id, MAX_REGISTERED_CALLABLE_IDS
        );
        return -1;
    }
    if (host_dlopen_handle == nullptr || host_orch_func_ptr == nullptr) {
        LOG_ERROR("register_callable_host_orch: null handle/fn for callable_id=%d", callable_id);
        return -1;
    }
    if (callables_.count(callable_id) != 0) {
        LOG_ERROR("register_callable_host_orch: callable_id=%d already registered", callable_id);
        return -1;
    }

    CallableState state;
    state.host_dlopen_handle = host_dlopen_handle;
    state.host_orch_func_ptr = host_orch_func_ptr;
    state.kernel_addrs = std::move(kernel_addrs);
    state.signature = std::move(signature);
    callables_.emplace(callable_id, std::move(state));
    ++host_dlopen_total_;
    LOG_INFO_V0("register_callable_host_orch: cid=%d (host dlopen #%zu)", callable_id, host_dlopen_total_);
    return 0;
}

int DeviceRunnerBase::unregister_callable(int32_t callable_id) {
    auto it = callables_.find(callable_id);
    if (it == callables_.end()) {
        return 0;
    }
    CallableState state = std::move(it->second);
    callables_.erase(it);
    aicpu_seen_callable_ids_.erase(callable_id);

    if (state.host_dlopen_handle != nullptr) {
        // hbg path: no orch SO refcount, just dlclose the host handle.
        dlclose(state.host_dlopen_handle);
        return 0;
    }

    auto buf_it = orch_so_dedup_.find(state.hash);
    if (buf_it != orch_so_dedup_.end()) {
        if (--buf_it->second.refcount <= 0) {
            mem_alloc_.free(buf_it->second.dev_addr);
            orch_so_dedup_.erase(buf_it);
        }
    }
    return 0;
}

bool DeviceRunnerBase::has_callable(int32_t callable_id) const { return callables_.count(callable_id) != 0; }

BindCallableResult DeviceRunnerBase::bind_callable_to_runtime(Runtime &runtime, int32_t callable_id) {
    auto it = callables_.find(callable_id);
    if (it == callables_.end()) {
        LOG_ERROR("bind_callable_to_runtime: callable_id=%d not registered", callable_id);
        return {-1, nullptr, nullptr, 0};
    }
    const auto &state = it->second;

    // Replay kernel addresses directly into runtime.func_id_to_addr_ without
    // going through set_function_bin_addr. The latter records func_ids in
    // registered_kernel_func_ids_, which validate_runtime_impl iterates to
    // free kernel binaries — but registered kernels must survive across runs
    // and are only freed by `finalize()` / `unregister_callable`.
    for (const auto &kv : state.kernel_addrs) {
        if (kv.first < 0 || kv.first >= RUNTIME_MAX_FUNC_ID) {
            LOG_ERROR("bind_callable_to_runtime: func_id=%d out of range", kv.first);
            return {-1, nullptr, nullptr, 0};
        }
        runtime.replay_function_bin_addr(kv.first, kv.second);
    }
    runtime.set_device_orch_func_name(state.func_name.c_str());
    runtime.set_device_orch_config_name(state.config_name.c_str());
    // Stamp callable_id with is_new=false; prepare_orch_so refreshes the flag
    // with the authoritative first_sighting answer right before launch.
    runtime.set_active_callable_id(callable_id, /*is_new=*/false);
    // hbg path: host_orch_func_ptr travels back to the c_api caller, which
    // hands it to bind_callable_to_runtime_impl. trb path: stays null and
    // the device-side orch SO is resolved from the symbol names above.
    return {
        0, state.host_orch_func_ptr, state.signature.empty() ? nullptr : state.signature.data(),
        static_cast<int>(state.signature.size())
    };
}

// =============================================================================
// Group E (minimal) — shared AICPU launch helper
// =============================================================================

int DeviceRunnerBase::launch_aicpu_kernel(
    rtStream_t stream, KernelArgs *k_args, const char *kernel_name, int aicpu_num
) {
    // kernel_name is host::KernelNames::InitName / RunName — the runtime SO's
    // actual exported symbol (simpler_aicpu_init / simpler_aicpu_exec).
    // LaunchBuiltInOp dispatches via rtsLaunchCpuKernel on the cached
    // rtFuncHandle resolved by LoadAicpuOp::Init at first-time bootstrap.
    return load_aicpu_op_.LaunchBuiltInOp(stream, k_args, aicpu_num, kernel_name);
}

int DeviceRunnerBase::finalize_common() {
    int rc = 0;
    auto capture = [&rc](int err) {
        if (err != 0 && rc == 0) rc = err;
    };

    // Streams are persistent for the DeviceRunner's lifetime; destroy them here.
    // Intentionally no pre-destroy sync: when a run hits the AICore op-timeout
    // chain (PR #718), the AICPU stream surfaces ACL_ERROR_RT_AICPU_EXCEPTION
    // (507018) at run-path sync; calling aclrtSynchronizeStream* again on the
    // error-state stream at finalize wedges subsequent tests (observed: 507018
    // / 507899 / 507901 cascade across the whole st-onboard-a2a3 suite).
    // rtStreamDestroy on an error-state stream is the supported teardown path.
    if (stream_aicpu_ != nullptr) {
        capture(rtStreamDestroy(stream_aicpu_));
        stream_aicpu_ = nullptr;
    }
    if (stream_aicore_ != nullptr) {
        capture(rtStreamDestroy(stream_aicore_));
        stream_aicore_ = nullptr;
    }

    // Cleanup kernel args (deviceArgs); device-side KernelArgs + runtime args
    // are released by runtime_args_cleanup RAII so they also unwind on errors.
    capture(kernel_args_.finalize_device_args());

    // load_aicpu_op_ has no per-task host-side state to release —
    // rtsLaunchCpuKernel does not hand back any per-launch handle, and the
    // dispatcher itself was a transient libaicpu_extend_kernels dlopen.
    // aicore_bin_handle_ was registered once via rtRegisterAllKernel; CANN
    // releases its device-side state when the device context tears down.
    aicore_bin_handle_ = nullptr;
    binaries_loaded_ = false;

    // Release any chip callable buffers uploaded via upload_chip_callable_buffer.
    // Pool semantics mirror per-fid binaries: never freed until finalize.
    for (auto &kv : chip_callable_buffers_) {
        mem_alloc_.free(reinterpret_cast<void *>(kv.second.chip_dev));
        LOG_DEBUG(
            "Freed chip callable buffer: chip_dev=0x%lx, size=%zu, hash=0x%lx", kv.second.chip_dev,
            kv.second.total_size, kv.first
        );
    }
    chip_callable_buffers_.clear();

    // Release any registered-callable orch SO buffers that callers forgot to
    // unregister. Refcounts no longer matter at this point — the device is
    // about to be reset.
    for (auto &kv : orch_so_dedup_) {
        if (kv.second.dev_addr != nullptr) {
            mem_alloc_.free(kv.second.dev_addr);
        }
    }
    orch_so_dedup_.clear();
    // hbg path: dlclose any host orch handles callers forgot to unregister.
    // finalize() is the last chance; Worker.close() does not auto-unregister
    // each callable_id, so without this loop the host process leaks one
    // dlopen handle per (re)created Worker — observable in long-running
    // pytest sessions.
    for (auto &kv : callables_) {
        if (kv.second.host_dlopen_handle != nullptr) {
            dlclose(kv.second.host_dlopen_handle);
        }
    }
    callables_.clear();
    aicpu_seen_callable_ids_.clear();
    aicpu_dlopen_total_ = 0;

    // Release the three per-Worker pooled arenas (GM heap, PTO2 SM, optional
    // trb prebuilt runtime arena — each its own device_malloc). Must precede
    // mem_alloc_.finalize() so the arenas free through the still-live
    // allocator, not after it.
    gm_heap_arena_.release();
    gm_sm_arena_.release();
    runtime_arena_pool_.release();

    // Free the 8-byte device_wall buffer (allocated lazily in run()) while
    // mem_alloc_ and the device context are still live. free_tensor() routes
    // through mem_alloc_.free(), so it must run before mem_alloc_.finalize()
    // and before the subclass's `rtDeviceReset()` tears down the device runtime.
    if (device_wall_dev_ptr_ != nullptr) {
        free_tensor(device_wall_dev_ptr_);
        device_wall_dev_ptr_ = nullptr;
    }

    // Free all remaining allocations (including handshake buffer and binGmAddr)
    mem_alloc_.finalize();

    block_dim_ = 0;
    worker_count_ = 0;
    aicore_kernel_binary_.clear();
    return rc;
}

int DeviceRunnerBase::launch_aicore_kernel(rtStream_t stream, KernelArgs *k_args) {
    // Lazy-register the AICore binary on first call; reuse cached handle
    // thereafter. CANN has no public rtUnregisterAllKernel, so re-registering
    // every run would pin another device-side copy of the ELF and quickly
    // exhaust HBM — surfaced in CI as 207001 at rtKernelLaunchWithHandleV2
    // with a 507899 cascade at rtStreamCreate.
    if (aicore_bin_handle_ == nullptr) {
        if (aicore_kernel_binary_.empty()) {
            LOG_ERROR("AICore kernel binary is empty");
            return -1;
        }
        rtDevBinary_t binary;
        std::memset(&binary, 0, sizeof(binary));
        binary.magic = RT_DEV_BINARY_MAGIC_ELF;
        binary.version = 0;
        binary.data = aicore_kernel_binary_.data();
        binary.length = aicore_kernel_binary_.size();
        int rc = rtRegisterAllKernel(&binary, &aicore_bin_handle_);
        if (rc != RT_ERROR_NONE) {
            LOG_ERROR("rtRegisterAllKernel failed: %d", rc);
            aicore_bin_handle_ = nullptr;
            return rc;
        }
    }

    struct Args {
        KernelArgs *k_args;
    };
    Args args = {k_args};
    rtArgsEx_t rt_args;
    std::memset(&rt_args, 0, sizeof(rt_args));
    rt_args.args = &args;
    rt_args.argsSize = sizeof(args);

    rtTaskCfgInfo_t cfg = {};
    cfg.schemMode = RT_SCHEM_MODE_BATCH;

    int rc = rtKernelLaunchWithHandleV2(aicore_bin_handle_, 0, block_dim_, &rt_args, nullptr, stream, &cfg);
    if (rc != RT_ERROR_NONE) {
        LOG_ERROR("rtKernelLaunchWithHandleV2 failed: %d", rc);
        return rc;
    }

    return rc;
}

// =============================================================================
// run() sub-sequence helpers — head + tail chunks shared by both arches
// =============================================================================

int DeviceRunnerBase::validate_launch_aicpu_num(int launch_aicpu_num) {
    if (launch_aicpu_num < 1 || launch_aicpu_num > PLATFORM_MAX_AICPU_THREADS) {
        LOG_ERROR("launch_aicpu_num (%d) must be in range [1, %d]", launch_aicpu_num, PLATFORM_MAX_AICPU_THREADS);
        return -1;
    }
    return 0;
}

void DeviceRunnerBase::ensure_device_wall_buffer() {
    if (device_wall_dev_ptr_ != nullptr) return;
    device_wall_dev_ptr_ = allocate_tensor(sizeof(uint64_t));
    if (device_wall_dev_ptr_ != nullptr) {
        kernel_args_.args.device_wall_data_base = reinterpret_cast<uint64_t>(device_wall_dev_ptr_);
        uint64_t zero = 0;
        (void)copy_to_device(device_wall_dev_ptr_, &zero, sizeof(uint64_t));
    }
}

int DeviceRunnerBase::resolve_block_dim(int requested_block_dim) {
    // Auto sentinel (block_dim == 0) is resolved directly from
    // query_max_block_dim; explicit values still go through validate. The
    // auto branch skips validate so we don't pay the ACL syscalls twice.
    int resolved = requested_block_dim;
    if (resolved == 0) {
        resolved = query_max_block_dim(stream_aicore_);
        LOG_INFO_V0("block_dim auto-resolved to %d", resolved);
        if (resolved < 1) {
            LOG_ERROR("block_dim auto-resolved to invalid value %d", resolved);
            return -1;
        }
    } else {
        int rc = validate_block_dim(stream_aicore_, resolved);
        if (rc != 0) {
            return -1;
        }
    }
    block_dim_ = resolved;
    return resolved;
}

int DeviceRunnerBase::prepare_runtime_for_launch(Runtime &runtime, int block_dim, int launch_aicpu_num) {
    int num_aicore = block_dim * cores_per_blockdim_;
    if (num_aicore > RUNTIME_MAX_WORKER) {
        LOG_ERROR("block_dim (%d) exceeds RUNTIME_MAX_WORKER (%d)", block_dim, RUNTIME_MAX_WORKER);
        return -1;
    }

    runtime.worker_count = num_aicore;
    worker_count_ = num_aicore;  // Stored for print_handshake_results in destructor
    runtime.aicpu_thread_num = launch_aicpu_num;

    // First `block_dim` cores are AIC; remaining ~2/3 are AIV.
    int num_aic = block_dim;
    for (int i = 0; i < num_aicore; i++) {
        runtime.workers[i].aicpu_ready = 0;
        runtime.workers[i].aicore_done = 0;
        runtime.workers[i].task = 0;
        runtime.workers[i].core_type = (i < num_aic) ? CoreType::AIC : CoreType::AIV;
    }

    // Set function_bin_addr for all tasks: Runtime::func_id_to_addr_[] stores
    // a CoreCallable device address; the binary code address is one
    // compile-time offset further in. The dispatch path then reads
    // resolved_addr_ from the on-device CoreCallable header.
    LOG_DEBUG("Setting function_bin_addr for Tasks");
    for (int i = 0; i < runtime.get_task_count(); i++) {
        Task *task = runtime.get_task(i);
        if (task != nullptr) {
            uint64_t callable_addr = runtime.get_function_bin_addr(task->func_id);
            task->function_bin_addr = callable_addr + CoreCallable::binary_data_offset();
            LOG_DEBUG("Task %d (func_id=%d) -> function_bin_addr=0x%lx", i, task->func_id, task->function_bin_addr);
        }
    }
    LOG_DEBUG("");
    return 0;
}

int DeviceRunnerBase::sync_run_streams() {
    LOG_INFO_V0("=== aclrtSynchronizeStreamWithTimeout stream_aicpu_ ===");
    int rc = aclrtSynchronizeStreamWithTimeout(stream_aicpu_, PLATFORM_STREAM_SYNC_TIMEOUT_MS);
    if (rc == ACL_ERROR_RT_STREAM_SYNC_TIMEOUT) {
        LOG_ERROR(
            "Stream sync timeout: stream=AICPU timeout_ms=%d device_id=%d block_dim=%d",
            PLATFORM_STREAM_SYNC_TIMEOUT_MS, device_id_, block_dim_
        );
        return rc;
    }
    if (rc != 0) {
        LOG_ERROR("aclrtSynchronizeStreamWithTimeout (AICPU) failed: %d", rc);
        return rc;
    }

    LOG_INFO_V0("=== aclrtSynchronizeStreamWithTimeout stream_aicore_ ===");
    rc = aclrtSynchronizeStreamWithTimeout(stream_aicore_, PLATFORM_STREAM_SYNC_TIMEOUT_MS);
    if (rc == ACL_ERROR_RT_STREAM_SYNC_TIMEOUT) {
        LOG_ERROR(
            "Stream sync timeout: stream=AICore timeout_ms=%d device_id=%d block_dim=%d",
            PLATFORM_STREAM_SYNC_TIMEOUT_MS, device_id_, block_dim_
        );
        return rc;
    }
    if (rc != 0) {
        LOG_ERROR("aclrtSynchronizeStreamWithTimeout (AICore) failed: %d", rc);
        return rc;
    }
    return 0;
}

void DeviceRunnerBase::read_device_wall_ns() {
    // Pull the platform-level device wall (ns) back from the 8-byte
    // device buffer that AICPU writes through via
    // KernelArgs::device_wall_data_base. (We can't use the
    // device_k_args_ shadow here — CANN's rtAicpuKernelLaunchExWithArgs
    // copies KernelArgs into AICPU-private memory at launch, so AICPU's
    // writes to its local copy don't propagate to device_k_args_.)
    // Failure path is a soft warn — wall stays zero.
    device_wall_ns_ = 0;
    if (device_wall_dev_ptr_ != nullptr) {
        int wall_rc = rtMemcpy(
            &device_wall_ns_, sizeof(uint64_t), device_wall_dev_ptr_, sizeof(uint64_t), RT_MEMCPY_DEVICE_TO_HOST
        );
        if (wall_rc != 0) {
            LOG_WARN("rtMemcpy(device_wall_ns) D2H failed: %d", wall_rc);
            device_wall_ns_ = 0;
        }
    }
}
