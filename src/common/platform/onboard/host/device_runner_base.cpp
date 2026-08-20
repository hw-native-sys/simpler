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
 * Shared lifecycle methods own runner-level resources; architecture-specific
 * launch, completion, and reset behavior remains in each DeviceRunner.
 */

#include "device_runner_base.h"

#include <runtime/rt.h>
#include <acl/acl.h>
#include <dlfcn.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "callable.h"
#include "callable_protocol.h"
#include "call_config.h"
#include "chip_callable_layout.h"
#include "common/core_type.h"
#include "common/host_api.h"
#include "common/platform_config.h"
#include "common/sdma_warmup_layout.h"
#include "common/unified_log.h"
#include "host/acl_error_log.h"
#include "host/host_phase_records_artifact.h"
#include "host/raii_scope_guard.h"
#include "host_log.h"
#include "platform_comm/comm.h"
#include "pto_runtime_c_api.h"
#include "task_args.h"
#include "utils/elf_build_id.h"
// `runtime.h` (pulled in via `device_runner_helpers.h` in the base header)
// supplies the per-arch `Handshake` + `Runtime` types used by
// `print_handshake_results` / `bind_callable_to_runtime` /
// `prepare_orch_so`.

// Implemented by each runtime's host part (runtime_maker.cpp). Reports the
// AICPU entry symbols this runtime exports beyond the base {exec, init} set, so
// the common AICPU loader carries no runtime-specific symbol knowledge. TMARB
// returns simpler_aicpu_register_callable; host_build_graph returns none.
extern "C" const char *const *runtime_extra_aicpu_symbols(size_t *count);

namespace {

HostRuntimeTimeoutConfig resolve_onboard_timeout_config() {
    RuntimeTimeoutConfig order_defaults{
        PLATFORM_OP_EXECUTE_TIMEOUT_US, PLATFORM_STREAM_SYNC_TIMEOUT_MS, PLATFORM_ONBOARD_SCHEDULER_TIMEOUT_MS
    };
    RuntimeTimeoutParseStatus parse_status;
    RuntimeTimeoutConfig cfg = resolve_runtime_timeout_config(order_defaults, &parse_status);

    if (parse_status.op_execute_env_set && !parse_status.op_execute_valid) {
        const char *op_env = std::getenv(SIMPLER_OP_EXECUTE_TIMEOUT_US_ENV);
        LOG_WARN(
            "%s=%s invalid, using default %llu", SIMPLER_OP_EXECUTE_TIMEOUT_US_ENV, op_env,
            (unsigned long long)order_defaults.op_execute_timeout_us
        );
    }

    if (parse_status.stream_sync_env_set && !parse_status.stream_sync_valid) {
        const char *sync_env = std::getenv(SIMPLER_STREAM_SYNC_TIMEOUT_MS_ENV);
        LOG_WARN(
            "%s=%s invalid, using default %d", SIMPLER_STREAM_SYNC_TIMEOUT_MS_ENV, sync_env,
            order_defaults.stream_sync_timeout_ms
        );
    }

    if (parse_status.scheduler_env_set && !parse_status.scheduler_valid) {
        const char *sched_env = std::getenv(SIMPLER_SCHEDULER_TIMEOUT_MS_ENV);
        LOG_WARN(
            "%s=%s invalid, using default %d", SIMPLER_SCHEDULER_TIMEOUT_MS_ENV, sched_env,
            order_defaults.scheduler_timeout_ms
        );
    }

    bool host_timeout_env_set =
        parse_status.op_execute_env_set || parse_status.stream_sync_env_set || parse_status.scheduler_env_set;
    RuntimeTimeoutOrderStatus order_status = validate_runtime_timeout_order(cfg);
    // The scheduler override is forwarded to the device (via InitArgs at init)
    // only when explicitly set, valid, and consistent with the op/stream
    // ordering. 0 means "no override" — the AICPU scheduler then keeps its
    // compile-time default. op/stream remain host-side acl knobs.
    int32_t scheduler_override = (parse_status.scheduler_env_set && parse_status.scheduler_valid &&
                                  order_status == RuntimeTimeoutOrderStatus::OK) ?
                                     cfg.scheduler_timeout_ms :
                                     0;
    if (host_timeout_env_set && order_status != RuntimeTimeoutOrderStatus::OK) {
        LOG_WARN(
            "Ignoring PTO2 timeout env overrides: %s (scheduler=%d ms, op_execute=%llu us, stream_sync=%d ms)",
            runtime_timeout_order_status_name(order_status), cfg.scheduler_timeout_ms,
            (unsigned long long)cfg.op_execute_timeout_us, cfg.stream_sync_timeout_ms
        );
        return HostRuntimeTimeoutConfig{
            order_defaults.op_execute_timeout_us, order_defaults.stream_sync_timeout_ms, scheduler_override
        };
    }
    return HostRuntimeTimeoutConfig{cfg.op_execute_timeout_us, cfg.stream_sync_timeout_ms, scheduler_override};
}

}  // namespace

DeviceRunnerBase::DeviceRunnerBase() {
    for (auto &bank : arena_banks_) {
        bank = std::make_unique<ArenaBank>(&arena_alloc_trampoline, &arena_free_trampoline, &mem_alloc_);
    }
}

uint64_t DeviceRunnerBase::arena_bank_gm_heap_base(uint32_t bank_id) const {
    if (bank_id >= arena_banks_.size()) return 0;
    const ArenaBank &bank = *arena_banks_[bank_id];
    return bank.gm_heap.is_committed() ? reinterpret_cast<uint64_t>(bank.gm_heap.base()) : 0;
}

uint64_t DeviceRunnerBase::retained_temp_addr(uint32_t slot_id) const {
    if (slot_id >= retained_temp_addrs_.size()) return 0;
    return reinterpret_cast<uint64_t>(retained_temp_addrs_[slot_id]);
}

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

int DeviceRunnerBase::device_memset(void *dev_ptr, int value, std::size_t bytes) {
    return aclrtMemset(dev_ptr, bytes, value, bytes);
}

void DeviceRunnerBase::get_retained_temp_buffer(uint32_t pipeline_slot, void **addr, size_t *size) {
    if (pipeline_slot >= retained_temp_addrs_.size()) {
        if (addr != nullptr) *addr = nullptr;
        if (size != nullptr) *size = 0;
        return;
    }
    if (addr != nullptr) *addr = retained_temp_addrs_[pipeline_slot];
    if (size != nullptr) *size = retained_temp_sizes_[pipeline_slot];
}

void DeviceRunnerBase::set_retained_temp_buffer(uint32_t pipeline_slot, void *addr, size_t size) {
    if (pipeline_slot >= retained_temp_addrs_.size()) return;
    retained_temp_addrs_[pipeline_slot] = addr;
    retained_temp_sizes_[pipeline_slot] = size;
}

void DeviceRunnerBase::clear_temporary_buffer() {
    for (size_t slot = 0; slot < retained_temp_addrs_.size(); ++slot) {
        if (retained_temp_addrs_[slot] == nullptr) continue;
        mem_alloc_.free(retained_temp_addrs_[slot]);
        retained_temp_addrs_[slot] = nullptr;
        retained_temp_sizes_[slot] = 0;
    }
}

void *DeviceRunnerBase::acquire_pooled_gm_heap(uint32_t arena_bank) {
    if (arena_bank >= arena_banks_.size()) return nullptr;
    DeviceArena &arena = this->arena_bank(arena_bank).gm_heap;
    if (!arena.is_committed()) return nullptr;
    return arena.base();
}

void *DeviceRunnerBase::acquire_pooled_gm_sm(uint32_t arena_bank) {
    if (arena_bank >= arena_banks_.size()) return nullptr;
    DeviceArena &arena = this->arena_bank(arena_bank).gm_sm;
    if (!arena.is_committed()) return nullptr;
    return arena.base();
}

void *DeviceRunnerBase::acquire_pooled_runtime_arena(uint32_t arena_bank) {
    if (arena_bank >= arena_banks_.size()) return nullptr;
    // hbg calls setup_static_arena(...,0) and leaves the runtime pool
    // uncommitted — fail loudly if a caller asks for it anyway.
    DeviceArena &arena = this->arena_bank(arena_bank).runtime_pool;
    if (!arena.is_committed()) return nullptr;
    return arena.base();
}

bool DeviceRunnerBase::lookup_prebuilt_runtime_arena_cache(
    uint32_t arena_bank, uint64_t hash, const void *key_data, size_t key_size, void **gm_heap_base, void **sm_base,
    void **runtime_arena_base, size_t *runtime_off, const void **image_data, size_t *image_size
) const {
    // The cache holds one entry and its bases point into bank 0, so any other
    // bank must rebuild rather than be handed a region it does not own.
    if (arena_bank != 0) return false;
    if (!prebuilt_runtime_arena_cache_valid_ || prebuilt_runtime_arena_cache_hash_ != hash ||
        prebuilt_runtime_arena_cache_key_.size() != key_size || key_data == nullptr || gm_heap_base == nullptr ||
        sm_base == nullptr || runtime_arena_base == nullptr || runtime_off == nullptr || image_data == nullptr ||
        image_size == nullptr) {
        return false;
    }
    if (std::memcmp(prebuilt_runtime_arena_cache_key_.data(), key_data, key_size) != 0) {
        return false;
    }
    *gm_heap_base = prebuilt_runtime_arena_cache_gm_heap_base_;
    *sm_base = prebuilt_runtime_arena_cache_sm_base_;
    *runtime_arena_base = prebuilt_runtime_arena_cache_runtime_arena_base_;
    *runtime_off = prebuilt_runtime_arena_cache_runtime_off_;
    *image_data = prebuilt_runtime_arena_cache_image_.data();
    *image_size = prebuilt_runtime_arena_cache_image_.size();
    return true;
}

void DeviceRunnerBase::mark_prebuilt_runtime_arena_cached(
    uint32_t arena_bank, uint64_t hash, const void *key_data, size_t key_size, void *gm_heap_base, void *sm_base,
    void *runtime_arena_base, size_t runtime_off, const void *image_data, size_t image_size
) {
    // Single-entry cache owned by bank 0; see lookup_prebuilt_runtime_arena_cache.
    if (arena_bank != 0) return;
    prebuilt_runtime_arena_cache_valid_ = false;
    prebuilt_runtime_arena_cache_hash_ = hash;
    prebuilt_runtime_arena_cache_key_.assign(
        static_cast<const uint8_t *>(key_data), static_cast<const uint8_t *>(key_data) + key_size
    );
    prebuilt_runtime_arena_cache_gm_heap_base_ = gm_heap_base;
    prebuilt_runtime_arena_cache_sm_base_ = sm_base;
    prebuilt_runtime_arena_cache_runtime_arena_base_ = runtime_arena_base;
    prebuilt_runtime_arena_cache_runtime_off_ = runtime_off;
    prebuilt_runtime_arena_cache_image_.assign(
        static_cast<const uint8_t *>(image_data), static_cast<const uint8_t *>(image_data) + image_size
    );
    prebuilt_runtime_arena_cache_valid_ = true;
}

int DeviceRunnerBase::setup_static_arena(
    uint32_t arena_bank, size_t gm_heap_size, size_t gm_sm_size, size_t runtime_arena_size
) {
    if (arena_bank >= arena_banks_.size()) {
        LOG_ERROR("arena bank %u is outside [0, %zu)", arena_bank, arena_banks_.size());
        return -1;
    }
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
    ArenaBank &bank = this->arena_bank(arena_bank);

    bool arena_changed = false;
    auto commit_region = [&arena_changed](DeviceArena &arena, size_t &cached_size, size_t requested_size) -> int {
        if (requested_size == 0) {
            // hbg's runtime_arena path: caller passed 0 and never reserved
            // a region. Leave the arena uncommitted; acquire_pooled_* will
            // return nullptr.
            if (arena.is_committed() && cached_size != 0) {
                arena.release();
                cached_size = 0;
                arena_changed = true;
            }
            return 0;
        }
        if (arena.is_committed() && requested_size <= cached_size) {
            return 0;
        }
        arena.release();
        cached_size = 0;
        arena_changed = true;
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
    bool ok = commit_region(bank.gm_heap, bank.cached_gm_heap_size, gm_heap_size) == 0;
    ok = ok && commit_region(bank.gm_sm, bank.cached_gm_sm_size, gm_sm_size) == 0;
    ok = ok && commit_region(bank.runtime_pool, bank.cached_runtime_arena_size, runtime_arena_size) == 0;
    if (!ok) {
        bank.gm_heap.release();
        bank.gm_sm.release();
        bank.runtime_pool.release();
        bank.cached_gm_heap_size = 0;
        bank.cached_gm_sm_size = 0;
        bank.cached_runtime_arena_size = 0;
        if (arena_bank == 0) {
            prebuilt_runtime_arena_cache_valid_ = false;
            prebuilt_runtime_arena_cache_key_.clear();
            prebuilt_runtime_arena_cache_gm_heap_base_ = nullptr;
            prebuilt_runtime_arena_cache_sm_base_ = nullptr;
            prebuilt_runtime_arena_cache_runtime_arena_base_ = nullptr;
            prebuilt_runtime_arena_cache_image_.clear();
        }
        return -1;
    }
    if (arena_changed && arena_bank == 0) {
        prebuilt_runtime_arena_cache_valid_ = false;
        prebuilt_runtime_arena_cache_key_.clear();
        prebuilt_runtime_arena_cache_gm_heap_base_ = nullptr;
        prebuilt_runtime_arena_cache_sm_base_ = nullptr;
        prebuilt_runtime_arena_cache_runtime_arena_base_ = nullptr;
        prebuilt_runtime_arena_cache_image_.clear();
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
        ACL_LOG_ERROR_DETAIL(rc);
        return rc;
    }

    // simpler_init performs the only lifetime write. Prepared-run admission
    // and execution subsequently attach different host threads, so repeated
    // same-value writes here would still be a C++ data race.
    if (device_id_ == -1) {
        timeout_config_ = resolve_onboard_timeout_config();
        configure_aicore_op_timeout();
        device_id_ = device_id;
    }
    return 0;
}

void DeviceRunnerBase::configure_aicore_op_timeout() {
    uint64_t actual_timeout = 0;
    int rc = aclrtSetOpExecuteTimeOutV2(timeout_config_.op_execute_timeout_us, &actual_timeout);
    if (rc != 0) {
        LOG_ERROR(
            "aclrtSetOpExecuteTimeOutV2(%llu us) failed: %d", (unsigned long long)timeout_config_.op_execute_timeout_us,
            rc
        );
    } else {
        LOG_INFO(
            "aclrtSetOpExecuteTimeOutV2: requested=%llu us, actual=%llu us",
            (unsigned long long)timeout_config_.op_execute_timeout_us, (unsigned long long)actual_timeout
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
            ACL_LOG_ERROR_DETAIL(rc);
            return rc;
        }
        aicpu_created_here = true;
    }
    if (stream_aicore_ == nullptr) {
        rc = rtStreamCreate(&stream_aicore_, 0);
        if (rc != 0) {
            LOG_ERROR("rtStreamCreate (AICore) failed: %d", rc);
            ACL_LOG_ERROR_DETAIL(rc);
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
        LOG_INFO("DeviceRunner: device=%d set, streams created", device_id_);
    }

    // Latch the AICore stream's block_dim ceiling. resolve_block_dim() is then
    // pure arithmetic and can run before any per-run stream work.
    if (max_block_dim_ == 0) {
        max_block_dim_ = query_max_block_dim(stream_aicore_, &max_cube_cores_, &max_vector_cores_);
        LOG_INFO(
            "DeviceRunner: device=%d max_block_dim=%d (cube=%u, vector=%u)", device_id_, max_block_dim_,
            max_cube_cores_, max_vector_cores_
        );
    }

    rc = ensure_binaries_loaded();
    if (rc != 0) return rc;

    return ensure_aicpu_init_launched();
}

int DeviceRunnerBase::ensure_aicpu_init_launched() {
    if (aicpu_init_launched_) {
        return 0;
    }

    InitArgs init_args{};
    init_args.device_id = static_cast<uint32_t>(device_id_);
    init_args.log_level = static_cast<uint32_t>(HostLogger::get_instance().level());
    // Per-device scheduler watchdog override, resolved once at attach into
    // timeout_config_. 0 -> the AICPU scheduler keeps its compile-time default.
    init_args.scheduler_timeout_ms = timeout_config_.scheduler_timeout_ms;
    // Publish the provisioned async-DMA workspace addresses (all-zero until a
    // Worker opts into SDMA). provision_dma_workspace() re-launches this entry to
    // re-latch them; the AICPU SO stays resident, so the latest values survive
    // every subsequent per-task launch.
    for (int kind = 0; kind < DMA_WORKSPACE_KIND_COUNT; ++kind) {
        init_args.dma_workspace_addr[kind] = dma_workspace_addr_[kind];
    }

    LOG_INFO("=== launch_aicpu_payload %s ===", host::KernelNames::InitName);
    int rc = launch_aicpu_payload(
        stream_aicpu_, &init_args, sizeof(init_args), host::KernelNames::InitName, /*aicpu_num=*/1
    );
    if (rc != 0) {
        LOG_ERROR("ensure_aicpu_init_launched: launch_aicpu_payload failed: %d", rc);
        return rc;
    }

    rc = aclrtSynchronizeStreamWithTimeout(stream_aicpu_, PLATFORM_STREAM_SYNC_TIMEOUT_MS);
    if (rc != 0) {
        LOG_ERROR("ensure_aicpu_init_launched: stream sync failed: %d (device_id=%d)", rc, device_id_);
        return rc;
    }
    aicpu_init_launched_ = true;
    return 0;
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
    LOG_INFO("DeviceRunner: inner SO uploaded to preinstall via dispatcher bootstrap");

    // JSON-register the inner SO and resolve its runtime entry handles. The
    // runtime reports any AICPU entries it exports beyond the base set so the
    // loader stays runtime-agnostic.
    std::vector<std::string> extra_symbols;
    size_t extra_count = 0;
    const char *const *extra = runtime_extra_aicpu_symbols(&extra_count);
    for (size_t i = 0; i < extra_count && extra != nullptr; ++i) {
        if (extra[i] != nullptr) extra_symbols.emplace_back(extra[i]);
    }
    rc = load_aicpu_op_.Init(extra_symbols);
    if (rc != 0) {
        LOG_ERROR("LoadAicpuOp::Init failed: %d", rc);
        return rc;
    }
    LOG_INFO("DeviceRunner: inner SO registered (runtime entry handles ready)");

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
    LOG_INFO("DeviceRunner: binaries loaded");
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

void DeviceRunnerBase::print_handshake_results(const KernelArgsHelper &kernel_args) {
    if (stream_aicpu_ == nullptr || worker_count_ == 0 || kernel_args.args.runtime_args == nullptr) {
        return;
    }

    // Allocate temporary buffer to read handshake data from device
    std::vector<Handshake> workers(worker_count_);
    size_t total_size = sizeof(Handshake) * worker_count_;
    rtMemcpy(
        workers.data(), total_size, kernel_args.args.runtime_args->get_workers(), total_size, RT_MEMCPY_DEVICE_TO_HOST
    );

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
    if (callable == nullptr) {
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
        it->second.refcount++;
        LOG_DEBUG(
            "Chip callable dedup hit: chip_dev=0x%lx, size=%zu, hash=0x%lx, refcount=%d", it->second.chip_dev,
            it->second.total_size, layout.content_hash, it->second.refcount
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
        ACL_LOG_ERROR_DETAIL(rc);
        mem_alloc_.free(gm_addr);
        return 0;
    }
    mark_run_streams_stale();

    chip_callable_buffers_.emplace(layout.content_hash, ChipCallableBuffer{chip_dev, layout.total_size, 1});
    LOG_DEBUG(
        "Uploaded chip callable: chip_dev=0x%lx, size=%zu, child_count=%d, hash=0x%lx", chip_dev, layout.total_size,
        callable->child_count(), layout.content_hash
    );
    return chip_dev;
}

int DeviceRunnerBase::release_chip_callable_buffer(uint64_t hash) {
    if (hash == 0) {
        return 0;
    }
    auto it = chip_callable_buffers_.find(hash);
    if (it == chip_callable_buffers_.end()) {
        LOG_WARN("release_chip_callable_buffer: hash=0x%lx not found", hash);
        return 0;
    }
    if (--it->second.refcount <= 0) {
        mem_alloc_.free(reinterpret_cast<void *>(it->second.chip_dev));
        LOG_DEBUG(
            "Freed chip callable buffer: chip_dev=0x%lx, size=%zu, hash=0x%lx", it->second.chip_dev,
            it->second.total_size, hash
        );
        chip_callable_buffers_.erase(it);
    }
    return 0;
}

int DeviceRunnerBase::stamp_orch_so(Runtime &runtime, int32_t cid) {
    // Registered-callable flow only: the orch SO was already H2D'd and
    // dlopen'd device-side at record_device_orch_callable / launch_device_register
    // time. All that remains for a run is to tell the AICPU which orch_so_table_
    // slot to dispatch — the active callable_id.
    if (cid < 0) {
        LOG_ERROR("stamp_orch_so: invalid callable_id=%d", cid);
        return -1;
    }
    auto it = callables_.find(cid);
    if (it == callables_.end()) {
        LOG_ERROR("stamp_orch_so: callable_id=%d not registered", cid);
        return -1;
    }
    runtime.set_active_callable_id(cid);
    return 0;
}

int DeviceRunnerBase::prepare_orch_so(Runtime &runtime) {
    const int32_t cid = runtime.get_active_callable_id();
    if (cid < 0) {
        LOG_ERROR("prepare_orch_so: no active callable_id; registered-callable flow required");
        return -1;
    }
    return stamp_orch_so(runtime, cid);
}

int DeviceRunnerBase::commit_device_register(int32_t cid) {
    auto it = callables_.find(cid);
    if (it == callables_.end()) {
        LOG_ERROR("commit_device_register: callable_id=%d not registered", cid);
        return -1;
    }
    const auto &state = it->second;
    if (state.host_dlopen_handle != nullptr) {
        return 0;
    }
    const bool inserted = aicpu_seen_callable_ids_.insert(cid).second;
    if (inserted) {
        ++aicpu_dlopen_total_;
        LOG_INFO("AICPU callable load committed cid=%d (count=%zu)", cid, aicpu_dlopen_total_);
    }
    return 0;
}

int DeviceRunnerBase::launch_device_register(int32_t callable_id) {
    auto it = callables_.find(callable_id);
    if (it == callables_.end()) {
        LOG_ERROR("launch_device_register: callable_id=%d not registered", callable_id);
        return -1;
    }
    if (it->second.host_dlopen_handle != nullptr) {
        return 0;
    }

    int rc = ensure_device_initialized();
    if (rc != 0) {
        LOG_ERROR("launch_device_register: ensure_device_initialized failed: %d", rc);
        return rc;
    }

    // Build the orch-SO descriptor straight from CallableState — no full
    // Runtime H2D as the old prewarm path did. Registration always (re)dlopens
    // the SO device-side, so there is no per-callable "new?" bit to carry.
    const CallableState &state = it->second;
    RegisterCallableArgs reg_args{};
    reg_args.active_callable_id = callable_id;
    reg_args.dev_orch_so_addr = state.dev_orch_so_addr;
    reg_args.dev_orch_so_size = state.dev_orch_so_size;
    snprintf(reg_args.device_orch_func_name, sizeof(reg_args.device_orch_func_name), "%s", state.func_name.c_str());
    snprintf(
        reg_args.device_orch_config_name, sizeof(reg_args.device_orch_config_name), "%s", state.config_name.c_str()
    );

    LOG_INFO("=== launch_aicpu_payload %s ===", host::KernelNames::RegisterCallableName);
    rc = launch_aicpu_payload(
        stream_aicpu_, &reg_args, sizeof(reg_args), host::KernelNames::RegisterCallableName, /*aicpu_num=*/1
    );
    if (rc != 0) {
        LOG_ERROR("launch_device_register: launch_aicpu_payload failed: %d", rc);
        return rc;
    }

    rc = aclrtSynchronizeStreamWithTimeout(stream_aicpu_, PLATFORM_STREAM_SYNC_TIMEOUT_MS);
    if (rc == ACL_ERROR_RT_STREAM_SYNC_TIMEOUT) {
        LOG_ERROR(
            "launch_device_register: stream sync timeout timeout_ms=%d device_id=%d", PLATFORM_STREAM_SYNC_TIMEOUT_MS,
            device_id_
        );
        return rc;
    }
    if (rc != 0) {
        LOG_ERROR("launch_device_register: aclrtSynchronizeStreamWithTimeout failed: %d", rc);
        ACL_LOG_ERROR_DETAIL(rc);
        return rc;
    }

    return commit_device_register(callable_id);
}

int DeviceRunnerBase::record_device_orch_callable(
    int32_t callable_id, uint64_t chip_buffer_hash, uint64_t aicore_image_hash, uint64_t chip_dev,
    const void *orch_so_data, size_t orch_so_size, const char *func_name, const char *config_name,
    std::vector<std::pair<int, uint64_t>> kernel_addrs, std::vector<ArgDirection> signature
) {
    // The AICPU executor reserves `orch_so_table_[MAX_REGISTERED_CALLABLE_IDS]`
    // (declared in src/common/task_interface/callable_protocol.h) and indexes
    // it by callable_id; rejecting an out-of-range id here keeps the host and
    // AICPU sides in sync and avoids an OOB access at run time.
    if (callable_id < 0 || callable_id >= MAX_REGISTERED_CALLABLE_IDS) {
        LOG_ERROR(
            "record_device_orch_callable: callable_id=%d out of range [0, %d)", callable_id, MAX_REGISTERED_CALLABLE_IDS
        );
        return -1;
    }
    if (orch_so_data == nullptr || orch_so_size == 0) {
        LOG_ERROR("record_device_orch_callable: empty orch SO for callable_id=%d", callable_id);
        return -1;
    }
    if (chip_buffer_hash == 0 || chip_dev == 0) {
        LOG_ERROR("record_device_orch_callable: missing chip buffer for callable_id=%d", callable_id);
        return -1;
    }
    if (callables_.count(callable_id) != 0) {
        LOG_ERROR("record_device_orch_callable: callable_id=%d already registered", callable_id);
        return -1;
    }

    const uint64_t hash = simpler::common::utils::elf_build_id_64(orch_so_data, orch_so_size);

    CallableState state;
    state.hash = hash;
    state.chip_buffer_hash = chip_buffer_hash;
    state.aicore_image_hash = aicore_image_hash;
    state.dev_orch_so_addr = chip_dev + offsetof(ChipCallable, storage_);
    state.dev_orch_so_size = orch_so_size;
    state.func_name = (func_name != nullptr) ? func_name : "";
    state.config_name = (config_name != nullptr) ? config_name : "";
    state.kernel_addrs = std::move(kernel_addrs);
    state.signature = std::move(signature);
    callables_.emplace(callable_id, std::move(state));
    LOG_INFO(
        "record_device_orch_callable: cid=%d orch_hash=0x%lx chip_hash=0x%lx %zu bytes", callable_id, hash,
        chip_buffer_hash, orch_so_size
    );
    return 0;
}

int DeviceRunnerBase::record_host_orch_callable(
    int32_t callable_id, uint64_t chip_buffer_hash, uint64_t aicore_image_hash, void *host_dlopen_handle,
    void *host_orch_func_ptr, std::vector<std::pair<int, uint64_t>> kernel_addrs, std::vector<ArgDirection> signature
) {
    if (callable_id < 0 || callable_id >= MAX_REGISTERED_CALLABLE_IDS) {
        LOG_ERROR(
            "record_host_orch_callable: callable_id=%d out of range [0, %d)", callable_id, MAX_REGISTERED_CALLABLE_IDS
        );
        return -1;
    }
    if (host_dlopen_handle == nullptr || host_orch_func_ptr == nullptr) {
        LOG_ERROR("record_host_orch_callable: null handle/fn for callable_id=%d", callable_id);
        return -1;
    }
    if (chip_buffer_hash == 0) {
        LOG_ERROR("record_host_orch_callable: missing chip buffer for callable_id=%d", callable_id);
        return -1;
    }
    if (callables_.count(callable_id) != 0) {
        LOG_ERROR("record_host_orch_callable: callable_id=%d already registered", callable_id);
        return -1;
    }

    CallableState state;
    state.chip_buffer_hash = chip_buffer_hash;
    state.aicore_image_hash = aicore_image_hash;
    state.host_dlopen_handle = host_dlopen_handle;
    state.host_orch_func_ptr = host_orch_func_ptr;
    state.kernel_addrs = std::move(kernel_addrs);
    state.signature = std::move(signature);
    callables_.emplace(callable_id, std::move(state));
    ++host_dlopen_total_;
    LOG_INFO("record_host_orch_callable: cid=%d (host dlopen #%zu)", callable_id, host_dlopen_total_);
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
    release_chip_callable_buffer(state.chip_buffer_hash);

    if (state.host_dlopen_handle != nullptr) {
        // hbg path: no device-side orch SO handle, just dlclose the host handle.
        dlclose(state.host_dlopen_handle);
        return 0;
    }
    return 0;
}

bool DeviceRunnerBase::has_callable(int32_t callable_id) const { return callables_.count(callable_id) != 0; }

int DeviceRunnerBase::provision_dma_workspace(
    uint32_t required_mask, const void *sdma_warmup_binary, size_t sdma_warmup_size
) {
    const uint32_t supported = dma_workspace_supported_mask();
    if ((required_mask & ~supported) != 0) {
        LOG_ERROR("provision_dma_workspace: unsupported mask=0x%x (supported=0x%x)", required_mask, supported);
        return -1;
    }
    if (dma_workspace_handle_ != nullptr) {
        LOG_ERROR("provision_dma_workspace: workspace already provisioned");
        return -1;
    }

    for (int kind = 0; kind < DMA_WORKSPACE_KIND_COUNT; ++kind)
        dma_workspace_addr_[kind] = 0;

    // The provisioned addresses are stable for the Worker's life.
    int rc =
        dma_workspace_provision(required_mask, dma_workspace_addr_, DMA_WORKSPACE_KIND_COUNT, &dma_workspace_handle_);
    if (rc != 0) {
        LOG_ERROR("provision_dma_workspace: mask=0x%x failed: %d", required_mask, rc);
        for (int kind = 0; kind < DMA_WORKSPACE_KIND_COUNT; ++kind)
            dma_workspace_addr_[kind] = 0;
        dma_workspace_handle_ = nullptr;
        return rc;
    }

    // Re-latch the resident AICPU globals: simpler_aicpu_init publishes the
    // provisioned addresses into g_dma_workspace_addr, which the scheduler
    // prefills into every core's GlobalContext (get_dma_workspace). The AICPU SO
    // stays dlopen'd, so the values survive every subsequent per-task launch.
    aicpu_init_launched_ = false;
    rc = ensure_aicpu_init_launched();
    if (rc != 0) {
        LOG_ERROR("provision_dma_workspace: re-latch of simpler_aicpu_init failed: %d", rc);
        dma_workspace_release(dma_workspace_handle_);
        dma_workspace_handle_ = nullptr;
        for (int kind = 0; kind < DMA_WORKSPACE_KIND_COUNT; ++kind)
            dma_workspace_addr_[kind] = 0;
        return rc;
    }

    // Deliberately after the re-latch: warming needs the live workspace. An
    // unavailable warmup leaves provisioning successful, because the only cost is
    // first-call latency. A device error does not: the card the warmup just faulted
    // on would otherwise reach the first run. No dma_workspace_release() on that
    // path — launch_sdma_warmup_kernel() has marked the runner unusable, and
    // per-resource release on a faulted card is exactly what finalize()'s fatal
    // path exists to avoid. The workspace handle stays set so that path still sees
    // an SDMA generation and applies its handoff delay.
    rc = launch_sdma_warmup_kernel(sdma_warmup_binary, sdma_warmup_size);
    if (rc != 0) {
        LOG_ERROR("provision_dma_workspace: sdma warmup left the device unusable: %d", rc);
        return rc;
    }
    return 0;
}

int DeviceRunnerBase::launch_sdma_warmup_kernel(const void *binary, size_t size) {
    // Reaching here means the workspace was provisioned, so this platform does
    // support SDMA and a missing ELF is a build/staging regression rather than
    // the expected state — worth a warning even though it is not fatal.
    if (binary == nullptr || size == 0) {
        LOG_WARN("sdma warmup: no warmup ELF supplied; the first TPREFETCH_ASYNC will pay the cold control path");
        return 0;
    }
    const uint32_t channel_count = dma_workspace_channel_count();
    const uint64_t workspace = dma_workspace_addr_[DMA_WORKSPACE_SDMA];
    if (channel_count == 0 || workspace == 0) {
        LOG_INFO("sdma warmup: no channels or no workspace address; skipping");
        return 0;
    }

    if (sdma_warmup_bin_handle_ == nullptr) {
        rtDevBinary_t warmup_binary;
        std::memset(&warmup_binary, 0, sizeof(warmup_binary));
        // AIVEC, not the executor's ELF magic: this binary has no cube half.
        warmup_binary.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
        warmup_binary.version = 0;
        warmup_binary.data = binary;
        warmup_binary.length = size;
        int reg_rc = rtRegisterAllKernel(&warmup_binary, &sdma_warmup_bin_handle_);
        if (reg_rc != 0 || sdma_warmup_bin_handle_ == nullptr) {
            LOG_WARN("sdma warmup: rtRegisterAllKernel failed: %d; skipping warmup", reg_rc);
            sdma_warmup_bin_handle_ = nullptr;
            return 0;
        }
    }

    // One cache line per channel, see sdma_warmup_layout.h. Transient: the launch
    // is synchronized here, so nothing outlives this call.
    const size_t status_bytes = static_cast<size_t>(channel_count) * kSdmaWarmupStatusStrideBytes;
    void *status_dev = allocate_tensor(status_bytes);
    if (status_dev == nullptr) {
        LOG_WARN("sdma warmup: could not allocate %zu status bytes; skipping warmup", status_bytes);
        return 0;
    }
    // Zero means "no core reached this channel", so the buffer must start clean
    // for the readback below to be meaningful.
    int rc = device_memset(status_dev, 0, status_bytes);
    if (rc != 0) {
        LOG_WARN("sdma warmup: status zero-fill failed: %d; skipping warmup", rc);
        free_tensor(status_dev);
        return 0;
    }

    struct Args {
        uint64_t workspace;
        uint64_t status;
        uint64_t channel_count;
    };
    Args args = {workspace, reinterpret_cast<uint64_t>(status_dev), channel_count};
    rtArgsEx_t rt_args;
    std::memset(&rt_args, 0, sizeof(rt_args));
    rt_args.args = &args;
    rt_args.argsSize = sizeof(args);

    rtTaskCfgInfo_t cfg = {};
    cfg.schemMode = RT_SCHEM_MODE_BATCH;

    // block_dim is kSdmaWarmupBlockDim, NOT channel_count: the cold cost is
    // per-channel and serializes in the engine regardless of how many cores push
    // on it, so extra blocks buy nothing (measured) and tying block_dim to the
    // channel count would break on any chip with fewer AIVs than channels. The
    // kernel walks channels grid-stride to cover all of them from 8 blocks.
    const auto started = std::chrono::steady_clock::now();
    rc = rtKernelLaunchWithHandleV2(
        sdma_warmup_bin_handle_, 0, kSdmaWarmupBlockDim, &rt_args, nullptr, stream_aicore_, &cfg
    );
    if (rc == 0) {
        rc = rtStreamSynchronize(stream_aicore_);
    }
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    if (rc != 0) {
        LOG_ERROR("sdma warmup: launch/sync failed: %d; the card is left poisoned", rc);
        // No free_tensor(status_dev): rtFree on a card that just failed an AICore
        // operation can block in DEV_RUNNING_DOWN. The allocation stays tracked by
        // mem_alloc_ and is forgotten wholesale by the fatal teardown the mark
        // below routes finalize() into.
        recover_device_or_mark_unusable(rc);
        return rc;
    }

    report_sdma_warmup_status(status_dev, channel_count, elapsed_ms);
    return 0;
}

void DeviceRunnerBase::report_sdma_warmup_status(void *status_dev, uint32_t channel_count, double elapsed_ms) {
    const size_t status_bytes = static_cast<size_t>(channel_count) * kSdmaWarmupStatusStrideBytes;
    std::vector<uint8_t> status_host(status_bytes, 0);
    const int rc = copy_from_device(status_host.data(), status_dev, status_bytes);
    free_tensor(status_dev);
    if (rc != 0) {
        LOG_WARN("sdma warmup: status D2H failed: %d; warmup ran but is unverified", rc);
        return;
    }

    uint32_t warmed = 0;
    uint32_t declined = 0;
    for (uint32_t channel = 0; channel < channel_count; ++channel) {
        uint32_t slot = 0;
        std::memcpy(
            &slot, status_host.data() + static_cast<size_t>(channel) * kSdmaWarmupStatusStrideBytes, sizeof(slot)
        );
        if (slot == kSdmaWarmupStatusOk) {
            ++warmed;
        } else if (slot == kSdmaWarmupStatusFailed) {
            ++declined;
        }
    }
    // TIMING, not INFO: this is a one-off multi-millisecond init cost paid to
    // remove the same cost from the first run, so it belongs with the other
    // performance markers that stay visible at the default threshold.
    if (warmed == channel_count) {
        LOG_TIMING("sdma warmup: %u/%u channels warmed in %.2f ms", warmed, channel_count, elapsed_ms);
    } else {
        // The two shortfalls have different causes: a declined channel was reached
        // but failed the warmup's preconditions (unpopulated SQ, non-empty queue),
        // while an unreached one means the walk itself did not cover the channel.
        LOG_WARN(
            "sdma warmup: only %u/%u channels warmed in %.2f ms (%u declined, %u unreached); "
            "the rest keep their cold-start cost",
            warmed, channel_count, elapsed_ms, declined, channel_count - warmed - declined
        );
    }
}

uint64_t DeviceRunnerBase::callable_hash(int32_t callable_id) const {
    auto it = callables_.find(callable_id);
    return it == callables_.end() ? 0 : it->second.hash;
}

// Per-run binding half, defined in each runtime's runtime_maker.cpp and linked
// into this same host_runtime.so. Declared here (rather than only in
// c_api_shared.cpp) so bind_callable_to_runtime can call it directly, keeping
// the CallableState-derived host_orch_func_ptr / signature internal to the
// runner instead of returning them across the c_api boundary.
extern "C" int bind_callable_to_runtime_impl(
    Runtime *runtime, const HostApi *api, const ChipStorageTaskArgs *orch_args, void *host_orch_func_ptr,
    const ArgDirection *signature, int sig_count, const uint64_t *ring_task_window, const uint64_t *ring_heap,
    const uint64_t *ring_dep_pool
);

int DeviceRunnerBase::bind_callable_to_runtime(
    Runtime &runtime, int32_t callable_id, const HostApi *api, const void *orch_args, const uint64_t *ring_task_window,
    const uint64_t *ring_heap, const uint64_t *ring_dep_pool
) {
    auto it = callables_.find(callable_id);
    if (it == callables_.end()) {
        LOG_ERROR("bind_callable_to_runtime: callable_id=%d not registered", callable_id);
        return -1;
    }
    const auto &state = it->second;

    // runtime.func_id_to_addr_ holds exactly the active callable's mappings.
    //
    // Each entry is an address inside that callable's retained ChipCallable
    // buffer, which `unregister_callable` frees once its last handle goes away.
    // The table is zeroed only in Runtime's constructor and a Runtime outlives
    // many register/unregister cycles, so an entry left behind by a previous
    // callable dangles into GM the allocator can hand out again. The scheduler
    // dereferences that address to read CoreCallable::resolved_addr() and the
    // AICore calls the result, so a stale entry is executed as code. Replaying
    // only this callable's func_ids would leave every other entry standing.
    runtime.clear_function_bin_addrs();
    for (const auto &kv : state.kernel_addrs) {
        if (kv.first < 0 || kv.first >= RUNTIME_MAX_FUNC_ID) {
            LOG_ERROR("bind_callable_to_runtime: func_id=%d out of range", kv.first);
            return -1;
        }
        runtime.replay_function_bin_addr(kv.first, kv.second);
    }
    // Tell the AICPU which orch_so_table_ slot this run dispatches. The orch SO
    // descriptor itself was delivered at register time via RegisterCallableArgs.
    runtime.set_active_callable_id(callable_id);

    // Per-run binding (tensor args, GM heap, SM alloc). host_orch_func_ptr is
    // non-null only on the hbg path; signature is the cached ChipCallable
    // signature_[], plumbed end-to-end for per-tensor H2D/D2H direction
    // decisions in runtime_maker (trb consumes it, hbg ignores it). Both stay
    // internal to the runner now — they are no longer returned to the c_api.
    return bind_callable_to_runtime_impl(
        &runtime, api, reinterpret_cast<const ChipStorageTaskArgs *>(orch_args), state.host_orch_func_ptr,
        state.signature.empty() ? nullptr : state.signature.data(), static_cast<int>(state.signature.size()),
        ring_task_window, ring_heap, ring_dep_pool
    );
}

// Eager prebuilt-arena warm-up. A runtime that has a prebuilt runtime arena
// (tensormap_and_ringbuffer) provides a strong prewarm_config_impl in its
// runtime_maker.cpp that overrides this weak no-op default. Runtimes without one
// (host_build_graph, or an arch that has not implemented it yet) link this weak
// default and treat prewarm as a no-op. simpler_init calls it directly for the
// fork-constant ring sizing once the device is up.
extern "C" __attribute__((weak)) int prewarm_config_impl(
    const HostApi * /*api*/, const uint64_t * /*ring_task_window*/, const uint64_t * /*ring_heap*/,
    const uint64_t * /*ring_dep_pool*/
) {
    return 0;
}

void DeviceRunnerBase::apply_call_config(const CallConfig &config) {
    set_chip_swimlane_enabled(config.enable_chip_swimlane);
    set_dump_args_enabled(config.enable_dump_args);
    set_pmu_enabled(config.enable_pmu);
    // Virtual: a2a3 and a5 wire through to their enable_dep_gen_; an arch
    // without dep_gen falls through to the base no-op.
    set_dep_gen_enabled(config.enable_dep_gen != 0);
    set_scope_stats_enabled(config.enable_scope_stats != 0);
    set_output_prefix(config.output_prefix);
}

HostPhaseRecordPool *DeviceRunnerBase::host_phase_pool_arm(bool producer_wants_records) noexcept {
    if (clock_correlation_provider_ != nullptr) {
        clock_correlation_provider_->release(false);
        clock_correlation_provider_.reset();
    }
    const bool swimlane_wants_records = chip_swimlane_level_ == ChipSwimlaneLevel::ORCH_PHASES;
    const bool artifact_wants_records = producer_wants_records && !output_prefix_.empty();
    chip_swimlane_collector_.set_host_orchestrated(swimlane_wants_records);
    // arm() allocates the pool's buffers, so it can throw; this path is noexcept,
    // where an escaping exception is std::terminate. A pass that cannot get its
    // storage collects no records and says so by handing back nullptr.
    HostPhaseRecordPool *pool = nullptr;
    try {
        pool = host_phase_records_.arm(artifact_wants_records || swimlane_wants_records);
    } catch (...) {
        LOG_WARN("Host phase pool could not be armed; this pass collects no per-event records");
    }
    if (!swimlane_wants_records) return pool;

    // Only the chip-swimlane reader places these records against device
    // timestamps, so only it needs the two clocks anchored.
    try {
        clock_correlation_provider_ = simpler::dfx::make_clock_correlation_provider();
        chip_swimlane_collector_.begin_clock_correlation_session(
            clock_correlation_provider_->name(), clock_correlation_provider_->raw_device_timestamp_unit()
        );
        chip_swimlane_collector_.record_clock_anchor_samples(
            simpler::dfx::capture_clock_anchor_group(
                *clock_correlation_provider_, simpler::dfx::ClockAnchorPosition::HostOrchestrationBegin
            )
        );
    } catch (...) {
        clock_correlation_provider_.reset();
        try {
            chip_swimlane_collector_.begin_clock_correlation_session("unavailable", "unknown");
        } catch (...) {
            // begin() stores diagnostic strings and can still fail under
            // allocation pressure. Keep this noexcept path fail-closed.
            chip_swimlane_collector_.finish_clock_correlation_session();
        }
    }
    return pool;
}

void DeviceRunnerBase::publish_host_phase_records_to_swimlane() {
    if (!host_phase_records_.finished()) return;
    chip_swimlane_collector_.set_host_phase_records(
        host_phase_records_.submit_records(), host_phase_records_.device_upload_records(),
        host_phase_records_.submitted_tasks(), host_phase_records_.total_records(),
        host_phase_records_.dropped_records()
    );
}

void DeviceRunnerBase::finish_clock_correlation_session(
    bool capture_device_complete, bool abandon_device_resources
) noexcept {
    if (!chip_swimlane_collector_.clock_correlation_active()) {
        if (clock_correlation_provider_ != nullptr) {
            clock_correlation_provider_->release(abandon_device_resources);
            clock_correlation_provider_.reset();
        }
        return;
    }

    if (capture_device_complete && clock_correlation_provider_ != nullptr) {
        try {
            chip_swimlane_collector_.record_clock_anchor_samples(
                simpler::dfx::capture_clock_anchor_group(
                    *clock_correlation_provider_, simpler::dfx::ClockAnchorPosition::DeviceExecutionComplete
                )
            );
        } catch (...) {}
    }
    chip_swimlane_collector_.finish_clock_correlation_session();
    if (clock_correlation_provider_ != nullptr) {
        clock_correlation_provider_->release(abandon_device_resources);
        clock_correlation_provider_.reset();
    }
}

// =============================================================================
// Group E (minimal) — shared AICPU launch helper
// =============================================================================

int DeviceRunnerBase::launch_aicpu_kernel(
    rtStream_t stream, KernelArgs *k_args, const char *kernel_name, int aicpu_num
) {
    // kernel_name is host::KernelNames::RunName — the runtime SO's actual
    // exported symbol (simpler_aicpu_exec). LaunchBuiltInOp dispatches via
    // rtsLaunchCpuKernel on the cached rtFuncHandle resolved by
    // LoadAicpuOp::Init at first-time bootstrap.
    return load_aicpu_op_.LaunchBuiltInOp(stream, k_args, sizeof(KernelArgs), aicpu_num, kernel_name);
}

int DeviceRunnerBase::launch_aicpu_payload(
    rtStream_t stream, void *args, size_t args_size, const char *kernel_name, int aicpu_num
) {
    return load_aicpu_op_.LaunchBuiltInOp(stream, args, args_size, aicpu_num, kernel_name);
}

int DeviceRunnerBase::finalize_common() { return finalize_common_impl(false); }

int DeviceRunnerBase::abandon_common_after_device_failure() { return finalize_common_impl(true); }

int DeviceRunnerBase::finalize_common_impl(bool abandon_device_resources) {
    int rc = 0;
    auto capture = [&rc](int err) {
        if (err != 0 && rc == 0) rc = err;
    };

    finish_clock_correlation_session(false, abandon_device_resources);

    // Teardown invariant: finalize_common() is the single place that releases
    // every RTS/device-owning resource, and the subclass runs it BEFORE its
    // device reset / aclFinalize. Several base-class members have destructors
    // that themselves call an RTS API -- LoadAicpuOp::~ -> rtsBinaryUnload,
    // MemoryAllocator::~ -> finalize -> rtFree, DeviceArena::~ -> release ->
    // rtFree. A member destructor runs (per C++ rules) only AFTER finalize()
    // returns, i.e. AFTER aclFinalize has torn down the RTS context, and
    // touching an RTS interface on a dead context segfaults on a5 (a2a3 happens
    // to tolerate it). So each such member is released explicitly here while RTS
    // is live; every release is idempotent (guarded on a handle / committed_ /
    // raw_base_ flag) so the eventual destructor no-ops. Any new member owning
    // an RTS/device resource must be released here, with an idempotent
    // destructor as the backstop. See issue #1197.
    // Streams are persistent for the DeviceRunner's lifetime; destroy them here.
    // Intentionally no pre-destroy sync: when a run hits the AICore op-timeout
    // chain (PR #718), the AICPU stream surfaces ACL_ERROR_RT_AICPU_EXCEPTION
    // (507018) at run-path sync; calling aclrtSynchronizeStream* again on the
    // error-state stream at finalize wedges subsequent tests (observed: 507018
    // / 507899 / 507901 cascade across the whole st-onboard-a2a3 suite).
    // rtStreamDestroy on an error-state stream is the supported teardown path.
    if (abandon_device_resources) {
        LOG_WARN("Fatal teardown: force reset/quarantine finished; skipping per-resource RTS destroy/free calls");
    }
    if (stream_aicpu_ != nullptr) {
        if (!abandon_device_resources) {
            capture(rtStreamDestroy(stream_aicpu_));
        }
        stream_aicpu_ = nullptr;
    }
    if (stream_aicore_ != nullptr) {
        if (!abandon_device_resources) {
            capture(rtStreamDestroy(stream_aicore_));
        }
        stream_aicore_ = nullptr;
    }

    // Release the async-DMA provider (SDMA STARS streams + workspace) only on
    // healthy teardown. A fatal reset invalidates its device resources as a
    // group, so running its per-stream destructor afterwards is unsafe.
    if (dma_workspace_handle_ != nullptr) {
        if (!abandon_device_resources) {
            dma_workspace_release(dma_workspace_handle_);
        }
        dma_workspace_handle_ = nullptr;
    }
    for (int kind = 0; kind < DMA_WORKSPACE_KIND_COUNT; ++kind)
        dma_workspace_addr_[kind] = 0;

    // LoadAicpuOp holds a binary_handle_ from rtsBinaryLoadFromFile; unload it
    // here while RTS is live so ~LoadAicpuOp's idempotent Finalize() no-ops
    // instead of unloading after aclFinalize (see the invariant above).
    if (abandon_device_resources) {
        load_aicpu_op_.AbandonAfterDeviceFailure();
        // A force reset invalidates every device allocation at once. If the
        // reset failed, the device is quarantined and per-allocation rtFree is
        // still unsafe. Forget allocator ownership before the shared host-side
        // cleanup below, so arena/free backstops become local no-ops. Per-run
        // kernel arguments live on PreparedExecution and are abandoned by
        // cleanup_execution() before finalize is reached.
        mem_alloc_.abandon_after_device_failure();
    } else {
        load_aicpu_op_.Finalize();
    }

    // aicore_bin_handle_ was registered once via rtRegisterAllKernel; CANN
    // releases its device-side state when the device context tears down. Same for
    // the SDMA warmup ELF's separate handle.
    aicore_bin_handle_ = nullptr;
    sdma_warmup_bin_handle_ = nullptr;
    binaries_loaded_ = false;
    // The inner AICPU SO is unloaded with the binaries above, so its latched
    // globals are gone too — clear the one-shot guard so a reused runner
    // re-launches simpler_aicpu_init after the next ensure_binaries_loaded().
    aicpu_init_launched_ = false;

    // Release any chip callable buffers callers forgot to unregister.
    if (!abandon_device_resources) {
        for (auto &kv : chip_callable_buffers_) {
            mem_alloc_.free(reinterpret_cast<void *>(kv.second.chip_dev));
            LOG_DEBUG(
                "Freed chip callable buffer: chip_dev=0x%lx, size=%zu, hash=0x%lx", kv.second.chip_dev,
                kv.second.total_size, kv.first
            );
        }
    }
    chip_callable_buffers_.clear();

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
    for (auto &bank : arena_banks_) {
        if (abandon_device_resources) {
            bank->gm_heap.abandon_after_device_failure();
            bank->gm_sm.abandon_after_device_failure();
            bank->runtime_pool.abandon_after_device_failure();
        } else {
            bank->gm_heap.release();
            bank->gm_sm.release();
            bank->runtime_pool.release();
        }
    }
    prebuilt_runtime_arena_cache_valid_ = false;
    prebuilt_runtime_arena_cache_key_.clear();
    prebuilt_runtime_arena_cache_gm_heap_base_ = nullptr;
    prebuilt_runtime_arena_cache_sm_base_ = nullptr;
    prebuilt_runtime_arena_cache_runtime_arena_base_ = nullptr;
    prebuilt_runtime_arena_cache_image_.clear();

    if (abandon_device_resources) {
        retained_temp_addrs_.fill(nullptr);
        retained_temp_sizes_.fill(0);
    } else {
        clear_temporary_buffer();
    }

    // Free the device-phase/task-timing buffer (allocated lazily in run()) while
    // mem_alloc_ and the device context are still live. free_tensor() routes
    // through mem_alloc_.free(), so it must run before mem_alloc_.finalize()
    // and before the subclass's `rtDeviceReset()` tears down the device runtime.
    if (device_wall_dev_ptr_ != nullptr) {
        if (!abandon_device_resources) {
            free_tensor(device_wall_dev_ptr_);
        }
        device_wall_dev_ptr_ = nullptr;
    }

    // Free all remaining allocations (including handshake buffer and binGmAddr)
    if (!abandon_device_resources) {
        mem_alloc_.finalize();
    }

    block_dim_ = 0;
    worker_count_ = 0;
    // Tied to stream_aicore_, destroyed above: a re-provisioned runner
    // re-queries rather than trusting the previous stream's limits.
    max_block_dim_ = 0;
    max_cube_cores_ = 0;
    max_vector_cores_ = 0;
    aicore_kernel_binary_.clear();
    for (auto &bank : arena_banks_) {
        bank->cached_gm_heap_size = 0;
        bank->cached_gm_sm_size = 0;
        bank->cached_runtime_arena_size = 0;
    }
    if (abandon_device_resources) {
        LOG_WARN("Fatal teardown: host-side ownership cleared without further device calls");
    }
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
            ACL_LOG_ERROR_DETAIL(rc);
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
        ACL_LOG_ERROR_DETAIL(rc);
        return rc;
    }

    return rc;
}

// =============================================================================
// run() sub-sequence helpers — head + tail chunks shared by both arches
// =============================================================================

int DeviceRunnerBase::validate_launch_aicpu_num(int launch_aicpu_num) {
    if (launch_aicpu_num == 1 || launch_aicpu_num < 0 || launch_aicpu_num > PLATFORM_MAX_AICPU_THREADS) {
        LOG_ERROR(
            "launch_aicpu_num (%d) must be 0 (auto) or in range [2, %d]", launch_aicpu_num, PLATFORM_MAX_AICPU_THREADS
        );
        return -1;
    }
    return 0;
}

int DeviceRunnerBase::resolve_aicpu_thread_num(int requested, int usable, int arch_default) {
    if (usable < 2) {
        LOG_ERROR("AICPU usable count %d < 2 (need >=1 orchestrator + >=1 scheduler)", usable);
        return -1;
    }
    int desired = (requested > 0) ? requested : arch_default;
    int total = std::min(desired, usable);
    if (total < desired) {
        LOG_WARN(
            "AICPU: requested %d active threads, only %d usable on this die — running 1 orch + %d sched", desired,
            usable, total - 1
        );
    }
    return total;
}

void DeviceRunnerBase::ensure_device_wall_buffer(KernelArgsHelper &kernel_args) {
    if (!device_phase_capture_enabled()) {
        // A null base makes the AICPU stamping helpers no-op.
        kernel_args.args.device_wall_data_base = 0;
        return;
    }
    // Fixed header followed by per-thread AICPU phase records (thread-major:
    // AicpuPhaseRecord[NUM_AICPU_PHASES] per launched AICPU thread). Slot
    // AicpuPhase::RunWall keeps the original whole-run wall; the rest subdivide
    // the on-NPU portion. Each surviving AICPU thread writes its own records
    // (plain stores, no atomics); read_device_phases() reduces RunWall as
    // max(end) - min(start) and surfaces the other phases as trace markers. The
    // buffer is allocated once (lazy) but RESET every run so a stale prior run
    // cannot leak into the reduction.
    constexpr int kThreads = PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH;
    using BufferImage = DevicePhaseBufferStorage<kThreads>;
    constexpr size_t kBytes = device_phase_buffer_bytes(kThreads);
    static_assert(sizeof(BufferImage) == kBytes, "device-phase buffer layout drift");
    if (device_wall_dev_ptr_ == nullptr) {
        device_wall_dev_ptr_ = allocate_tensor(kBytes);
    }
    if (device_wall_dev_ptr_ != nullptr) {
        kernel_args.args.device_wall_data_base = reinterpret_cast<uint64_t>(device_wall_dev_ptr_);
    }
}

int DeviceRunnerBase::arm_device_wall_buffer(KernelArgsHelper &kernel_args) {
    if (device_wall_dev_ptr_ == nullptr || kernel_args.args.device_wall_data_base == 0) return 0;
    constexpr int kThreads = PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH;
    using BufferImage = DevicePhaseBufferStorage<kThreads>;
    static const BufferImage init = [] {
        BufferImage image{};
        reset_device_phase_buffer(&image, kThreads);
        return image;
    }();
    if (copy_to_device(device_wall_dev_ptr_, &init, sizeof(init)) != 0) {
        // Reset failed — disable capture for this run so stale slot data
        // can't leak into the reduction. Keep the shared allocation alive:
        // an earlier run may still reference it, and the next run retries reset.
        LOG_WARN("device_phase reset H2D failed; disabling phase capture this run");
        kernel_args.args.device_wall_data_base = 0;
        return 0;
    }
    return 0;
}

int DeviceRunnerBase::resolve_block_dim() {
    if (max_block_dim_ < 1) {
        LOG_ERROR(
            "block_dim ceiling not resolved (cube=%u, vector=%u); ensure_device_initialized must run first",
            max_cube_cores_, max_vector_cores_
        );
        return -1;
    }
    LOG_INFO("block_dim resolved to %d (cube=%u, vector=%u)", max_block_dim_, max_cube_cores_, max_vector_cores_);
    return max_block_dim_;
}

int DeviceRunnerBase::prepare_launch_shape(Runtime &runtime, const CallConfig &config) {
    if (validate_launch_aicpu_num(config.aicpu_thread_num) != 0) {
        return -1;
    }
    int block_dim = resolve_block_dim();
    if (block_dim < 0) {
        return -1;
    }

    int num_aicore = block_dim * cores_per_blockdim_;
    if (num_aicore > RUNTIME_MAX_WORKER) {
        LOG_ERROR("block_dim (%d) exceeds RUNTIME_MAX_WORKER (%d)", block_dim, RUNTIME_MAX_WORKER);
        return -1;
    }

    runtime.set_worker_count(num_aicore);
    runtime.set_aicpu_thread_num(config.aicpu_thread_num);

    // First `block_dim` cores are AIC; remaining ~2/3 are AIV.
    int num_aic = block_dim;
    Handshake *workers = runtime.get_workers();
    for (int i = 0; i < num_aicore; i++) {
        workers[i].aicpu_ready = 0;
        workers[i].aicore_done = 0;
        workers[i].task = 0;
        workers[i].core_type = (i < num_aic) ? CoreType::AIC : CoreType::AIV;
    }
    return 0;
}

void DeviceRunnerBase::activate_launch_shape(const Runtime &runtime) {
    worker_count_ = runtime.get_worker_count();
    block_dim_ = worker_count_ / cores_per_blockdim_;
}

void DeviceRunnerBase::resolve_task_binary_addrs(Runtime &runtime) {
    // Runtime::func_id_to_addr_[] stores a CoreCallable device address; the
    // binary code address is one compile-time offset further in. The dispatch
    // path then reads resolved_addr_ from the on-device CoreCallable header.
    for (int i = 0; i < runtime.get_task_count(); i++) {
        Task *task = runtime.get_task(i);
        if (task != nullptr) {
            uint64_t callable_addr = runtime.get_function_bin_addr(task->func_id);
            task->function_bin_addr = callable_addr + CoreCallable::binary_data_offset();
            LOG_DEBUG("Task %d (func_id=%d) -> function_bin_addr=0x%lx", i, task->func_id, task->function_bin_addr);
        }
    }
}

int DeviceRunnerBase::sync_run_streams() { return sync_stream_pair(stream_aicpu_, stream_aicore_); }

int DeviceRunnerBase::sync_stream_pair(rtStream_t aicpu_stream, rtStream_t aicore_stream) {
    LOG_INFO("=== aclrtSynchronizeStreamWithTimeout AICPU stream ===");
    int rc = aclrtSynchronizeStreamWithTimeout(aicpu_stream, timeout_config_.stream_sync_timeout_ms);
    if (rc == ACL_ERROR_RT_STREAM_SYNC_TIMEOUT) {
        LOG_ERROR(
            "Stream sync timeout: stream=AICPU timeout_ms=%d device_id=%d block_dim=%d",
            timeout_config_.stream_sync_timeout_ms, device_id_, block_dim_
        );
        ACL_LOG_ERROR_DETAIL(rc);
        return rc;
    }
    if (rc != 0) {
        LOG_ERROR("aclrtSynchronizeStreamWithTimeout (AICPU) failed: %d", rc);
        ACL_LOG_ERROR_DETAIL(rc);
        return rc;
    }

    LOG_INFO("=== aclrtSynchronizeStreamWithTimeout AICore stream ===");
    rc = aclrtSynchronizeStreamWithTimeout(aicore_stream, timeout_config_.stream_sync_timeout_ms);
    if (rc == ACL_ERROR_RT_STREAM_SYNC_TIMEOUT) {
        LOG_ERROR(
            "Stream sync timeout: stream=AICore timeout_ms=%d device_id=%d block_dim=%d",
            timeout_config_.stream_sync_timeout_ms, device_id_, block_dim_
        );
        ACL_LOG_ERROR_DETAIL(rc);
        return rc;
    }
    if (rc != 0) {
        LOG_ERROR("aclrtSynchronizeStreamWithTimeout (AICore) failed: %d", rc);
        ACL_LOG_ERROR_DETAIL(rc);
        return rc;
    }
    return 0;
}

void DeviceRunnerBase::read_device_wall_ns() {
    // Pull the per-thread AICPU phase records back from the device buffer that
    // AICPU writes through via KernelArgs::device_wall_data_base. (We can't use
    // the device_k_args_ shadow here — CANN's rtAicpuKernelLaunchExWithArgs
    // copies KernelArgs into AICPU-private memory at launch, so AICPU's writes
    // to its local copy don't propagate to device_k_args_.) Failure path is a
    // soft warn — wall + phases stay zero.
    device_wall_ns_ = 0;
    for (int p = 0; p < NUM_AICPU_PHASES; ++p) {
        device_phase_ns_[p] = 0;
        device_phase_start_ns_[p] = 0;
    }
    for (int s = 0; s < NUM_TASK_TIMING_SLOTS; ++s) {
        task_slot_dispatch_ns_[s] = 0;
        task_slot_finish_ns_[s] = 0;
    }
    if (!device_phase_capture_enabled()) return;
    if (device_wall_dev_ptr_ == nullptr) return;

    constexpr int kThreads = PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH;
    using BufferPrefix = DevicePhaseBufferPrefixStorage<kThreads>;
    static_assert(sizeof(BufferPrefix) == task_timing_tail_offset(kThreads), "device-phase prefix layout drift");
    BufferPrefix buf{};
    int wall_rc = rtMemcpy(&buf, sizeof(buf), device_wall_dev_ptr_, sizeof(buf), RT_MEMCPY_DEVICE_TO_HOST);
    if (wall_rc != 0) {
        LOG_WARN("rtMemcpy(device_phase) D2H failed: %d", wall_rc);
        return;
    }

    // Reduce across threads: per phase, min(start) + span = max(end) - min(start)
    // in cycles. RunWall (slot 0) is published as device_wall_ns_ for backward
    // compatibility; its duration is the whole-run wall.
    uint64_t start_cycles[NUM_AICPU_PHASES];
    uint64_t span_cycles[NUM_AICPU_PHASES];
    reduce_aicpu_phase_windows(buf.phases, kThreads, start_cycles, span_cycles);

    // Origin = earliest sub-phase start (Preamble..SchedWindow share the device
    // clock; RunWall is the bracket at offset 0). Sub-phase start offsets from
    // this origin give a common device-clock timeline so the orchestrator and
    // scheduler windows are comparable (their union is the "Effective" window).
    uint64_t origin = kPhaseUnset;
    for (int p = static_cast<int>(AicpuPhase::Preamble); p < NUM_AICPU_PHASES; ++p) {
        if (start_cycles[p] != kPhaseUnset && start_cycles[p] < origin) origin = start_cycles[p];
    }

    for (int p = 0; p < NUM_AICPU_PHASES; ++p) {
        device_phase_ns_[p] = span_cycles[p] > 0 ? static_cast<uint64_t>(cycles_to_us(span_cycles[p]) * 1000.0) : 0;
        if (p != static_cast<int>(AicpuPhase::RunWall) && start_cycles[p] != kPhaseUnset && origin != kPhaseUnset &&
            start_cycles[p] >= origin) {
            device_phase_start_ns_[p] = static_cast<uint64_t>(cycles_to_us(start_cycles[p] - origin) * 1000.0);
        }
    }
    device_wall_ns_ = device_phase_ns_[static_cast<int>(AicpuPhase::RunWall)];

    // A nonzero header means the last AICPU thread found at least one
    // dispatched timing slot. The conditional callback D2Hs the optional tail
    // and resolves it on the phase `origin` timeline.
    constexpr int kTailRecords = task_timing_buffer_slots(kThreads);
    int tail_rc = read_task_timing_tail_if_used(buf.header, [&]() {
        TaskTimingRecord tail[kTailRecords] = {};
        const void *tail_src =
            reinterpret_cast<const uint8_t *>(device_wall_dev_ptr_) + task_timing_tail_offset(kThreads);
        int rc = rtMemcpy(tail, sizeof(tail), tail_src, sizeof(tail), RT_MEMCPY_DEVICE_TO_HOST);
        if (rc != 0) return rc;
        resolve_task_timing_slots_ns(
            tail, kThreads, origin,
            [](uint64_t cyc) {
                return static_cast<uint64_t>(cycles_to_us(cyc) * 1000.0);
            },
            task_slot_dispatch_ns_, task_slot_finish_ns_
        );
        return 0;
    });
    if (tail_rc != 0) {
        LOG_WARN("rtMemcpy(task_timing) D2H failed: %d", tail_rc);
    }
}

int DeviceRunnerBase::init_runtime_args_with_metadata(Runtime &runtime, KernelArgsHelper &kernel_args) {
    int rc = kernel_args.init_runtime_args(runtime, mem_alloc_);
    if (rc != 0) {
        LOG_ERROR("init_runtime_args failed: %d", rc);
        return rc;
    }
    // Log config and device ordinal are no longer published per-run on
    // KernelArgs — they were latched once into the AICPU SO globals by
    // simpler_aicpu_init (ensure_aicpu_init_launched) at device init.
    return 0;
}

void DeviceRunnerBase::start_shared_collectors_for_run() {
    // Start collector mgmt + poll threads now, just before kernels launch.
    // Starting earlier wastes CPU on empty queues and risks tripping
    // ProfilerBase's poll-loop idle-timeout if device-side init is slow.
    auto thread_factory = [this](std::function<void()> fn) {
        return create_thread(std::move(fn));
    };
    if (enable_chip_swimlane_) {
        chip_swimlane_collector_.start(thread_factory);
    }
    if (enable_dump_args_) {
        dump_collector_.start(thread_factory);
    }
    if (enable_pmu_) {
        pmu_collector_.start(thread_factory);
    }
    if (enable_scope_stats_) {
        scope_stats_collector_.start(thread_factory);
    }
}

void DeviceRunnerBase::write_host_phase_records_artifact() {
    // Per-event view of the prepare path. Every phase it records is produced on
    // the host during bind, and the store is finished before launch, so this
    // touches no device state and is callable from any point after bind --
    // including a path that never launches. Keyed on output_prefix_ because it is
    // non-empty exactly when this run produces diagnostic artifacts, and on the
    // store having finished a pass, which is what a host-orchestrating runtime
    // leaves behind. The store writes a pass at most once, so every path that can
    // end a run calls this unconditionally.
    if (!output_prefix_.empty() && host_phase_records_.finished()) {
        (void)host_phase_records_.write_records_jsonl(make_host_phase_records_path(output_prefix_));
    }
}

void DeviceRunnerBase::teardown_shared_collectors_after_run(bool device_execution_complete) {
    // Tear down collectors. stop() joins mgmt then collector in the only safe
    // order (mgmt's final-drain pass into L2 has poll as its consumer).
    // Diagnostic exports use the per-task `output_prefix_` directory the user
    // set on CallConfig (CallConfig::validate() enforces non-empty upstream).
    finish_clock_correlation_session(device_execution_complete, !can_accept_run());
    if (enable_chip_swimlane_) {
        chip_swimlane_collector_.stop();
        chip_swimlane_collector_.read_phase_header_metadata();
        chip_swimlane_collector_.reconcile_counters();
        publish_host_phase_records_to_swimlane();
        chip_swimlane_collector_.export_swimlane_json();
    }

    write_host_phase_records_artifact();

    if (enable_dump_args_) {
        dump_collector_.stop();
        dump_collector_.reconcile_counters();
        dump_collector_.export_dump_files();
    }

    if (enable_pmu_) {
        pmu_collector_.stop();
        pmu_collector_.reconcile_counters();
    }

    if (enable_scope_stats_) {
        scope_stats_collector_.stop();
        scope_stats_collector_.reconcile_counters();
        scope_stats_collector_.write_jsonl(output_prefix_);
    }
}

bool DeviceRunnerBase::try_acquire_native_run(
    const void *owner, const NativeRunIdentity &identity, LaunchPermit *permit
) {
    if (owner == nullptr || permit == nullptr) return false;
    std::lock_guard<std::mutex> lk(native_run_mu_);
    bool reserved = false;
    for (const NativeRunReservation &reservation : native_run_reservations_) {
        if (reservation.owner == owner) {
            reserved = true;
            break;
        }
    }
    if (!reserved) return false;
    const void *expected = nullptr;
    if (!active_native_run_.compare_exchange_strong(
            expected, owner, std::memory_order_acq_rel, std::memory_order_acquire
        )) {
        return false;
    }
    *permit = LaunchPermit(identity);
    return true;
}

void DeviceRunnerBase::release_native_run(const void *owner) {
    std::lock_guard<std::mutex> lk(native_run_mu_);
    if (active_native_run_.load(std::memory_order_acquire) != owner) return;
    const void *expected = owner;
    (void)active_native_run_.compare_exchange_strong(
        expected, nullptr, std::memory_order_release, std::memory_order_relaxed
    );
}

bool DeviceRunnerBase::native_run_active() const {
    return active_native_run_.load(std::memory_order_acquire) != nullptr;
}

bool DeviceRunnerBase::native_run_owned_by(const void *owner) const {
    return owner != nullptr && active_native_run_.load(std::memory_order_acquire) == owner;
}

bool DeviceRunnerBase::try_reserve_native_run(
    const void *owner, uint32_t pipeline_slot, uint32_t arena_bank, bool allow_prepared_successor
) {
    if (owner == nullptr || pipeline_slot >= PTO_PIPELINE_MAX_DEPTH || arena_bank >= PTO_PIPELINE_MAX_DEPTH) {
        return false;
    }
    std::lock_guard<std::mutex> lk(native_run_mu_);

    size_t occupied = 0;
    const NativeRunReservation *existing = nullptr;
    for (const NativeRunReservation &reservation : native_run_reservations_) {
        if (reservation.owner == nullptr) continue;
        if (reservation.owner == owner || reservation.pipeline_slot == pipeline_slot) {
            return false;
        }
        ++occupied;
        existing = &reservation;
    }
    if (occupied != 0) {
        const void *active = active_native_run_.load(std::memory_order_acquire);
        if (!allow_prepared_successor || occupied != 1 || existing == nullptr ||
            !existing->permits_prepared_successor || active != existing->owner) {
            return false;
        }
    }

    for (NativeRunReservation &reservation : native_run_reservations_) {
        if (reservation.owner == nullptr) {
            reservation = NativeRunReservation{owner, pipeline_slot, arena_bank, allow_prepared_successor};
            return true;
        }
    }
    return false;
}

void DeviceRunnerBase::release_native_run_reservation(const void *owner) {
    if (owner == nullptr) return;
    std::lock_guard<std::mutex> lk(native_run_mu_);
    for (NativeRunReservation &reservation : native_run_reservations_) {
        if (reservation.owner == owner) {
            reservation = NativeRunReservation{};
            return;
        }
    }
}

bool DeviceRunnerBase::native_runs_outstanding() const {
    std::lock_guard<std::mutex> lk(native_run_mu_);
    for (const NativeRunReservation &reservation : native_run_reservations_) {
        if (reservation.owner != nullptr) return true;
    }
    return false;
}
