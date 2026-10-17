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

#include <sys/stat.h>
#include <stdlib.h>

#if defined(__linux__)
#include <pthread.h>
#endif

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>

#include "callable.h"
#include "callable_protocol.h"
#include "call_config.h"
#include "chip_callable_layout.h"
#include "common/host_api.h"
#include "cpu_sim_context.h"
#include "host/host_phase_records_artifact.h"
#include "host/raii_scope_guard.h"
#include "host/run_boundary.h"
#include "task_args_wire.h"
#include "utils/elf_build_id.h"

// The host-orchestrating runtime's graph hand-off, and a weak fallback for a
// runtime that captures on the device: its `dep_gen_host_graph_active()` is
// false, so the runner never asks it for a graph. File scope, like the runner
// method that takes its address and like the arch runners that define it:
// `extern "C"` names the symbol, but lookup is still scoped, so a declaration
// inside this file's own namespace is invisible to `SimDeviceRunnerBase` below.
extern "C" int dep_gen_host_graph_take(simpler::dfx::host_graph::HostGraphExport *out);

namespace simpler::common::sim_host {

namespace {

bool write_all_bytes(int fd, const uint8_t *data, size_t size) {
    size_t total_written = 0;
    while (total_written < size) {
        ssize_t written = write(fd, data + total_written, size - total_written);
        if (written <= 0) return false;
        total_written += static_cast<size_t>(written);
    }
    return true;
}

}  // namespace

bool create_temp_so_file(const std::string &path_template, const uint8_t *data, size_t size, std::string *out_path) {
    std::vector<char> path_buf(path_template.begin(), path_template.end());
    path_buf.push_back('\0');

    int fd = mkstemp(path_buf.data());
    if (fd < 0) {
        return false;
    }

    // dlopen requires the file to be executable; mkstemp creates 0600 (no exec bit)
    if (fchmod(fd, 0755) != 0) {
        close(fd);
        unlink(path_buf.data());
        return false;
    }

    bool ok = write_all_bytes(fd, data, size);
    if (close(fd) != 0) {
        ok = false;
    }
    if (!ok) {
        unlink(path_buf.data());
        return false;
    }

    *out_path = path_buf.data();
    return true;
}

}  // namespace simpler::common::sim_host

// =============================================================================
// SimDeviceRunnerBase Implementation
// =============================================================================

bool SimDeviceRunnerBase::try_acquire_native_run(
    const void *owner, const NativeRunIdentity &identity, LaunchPermit *permit
) {
    if (owner == nullptr || permit == nullptr) return false;
    const void *expected = nullptr;
    if (!active_native_run_.compare_exchange_strong(
            expected, owner, std::memory_order_acq_rel, std::memory_order_acquire
        )) {
        return false;
    }
    *permit = LaunchPermit(identity);
    return true;
}

bool SimDeviceRunnerBase::reissue_launch_permit(
    const void *owner, const NativeRunIdentity &identity, LaunchPermit *permit
) {
    if (owner == nullptr || permit == nullptr) return false;
    if (active_native_run_.load(std::memory_order_acquire) != owner) return false;
    *permit = LaunchPermit(identity);
    return true;
}

void SimDeviceRunnerBase::release_native_run(const void *owner) {
    if (active_native_run_.load(std::memory_order_acquire) != owner) return;
    const void *expected = owner;
    (void)active_native_run_.compare_exchange_strong(
        expected, nullptr, std::memory_order_release, std::memory_order_relaxed
    );
}

bool SimDeviceRunnerBase::native_run_active() const {
    return active_native_run_.load(std::memory_order_acquire) != nullptr;
}

bool SimDeviceRunnerBase::native_run_owned_by(const void *owner) const {
    return owner != nullptr && active_native_run_.load(std::memory_order_acquire) == owner;
}

/**
 * Build every per-set resource for `granted`, publish none.
 *
 * Phase one of the grant, for the same reason the onboard runner splits it:
 * resizing members one at a time leaves the earlier tables at the new count and
 * the later ones at the old, and leaves the pointer-held bank table holding
 * nulls that every reader and the whole cleanup path dereference.
 */
SimDeviceRunnerBase::RunResourceStaging SimDeviceRunnerBase::build_run_resources(size_t granted) {
    RunResourceStaging staged;
    staged.retained_temp_addrs = RunResourceSlots<void *>::build(granted);
    staged.retained_temp_sizes = RunResourceSlots<size_t>::build(granted);
    staged.graph_definition_staging = RunResourceSlots<RetainedGraphStaging>::build(granted);
    staged.scheduler_state_storage = RunResourceSlots<RetainedSchedulerStorage>::build(granted);
    staged.orchestration_scratch = RunResourceSlots<std::shared_ptr<void>>::build(granted);
    staged.sm_mirrors = RunResourceSlots<RetainedSmMirror>::build(granted);
    staged.run_image_stagings = RunResourceSlots<RetainedSmMirror>::build(granted);
    staged.host_phase_runs = RunResourceSlots<HostPhaseRunState>::build(granted);
    staged.device_run_results = RunResourceSlots<DeviceRunResultRegion>::build(granted);
    // The bank carries constructor arguments. Built here rather than filled in
    // after a resize, so the table is never visible holding a null entry.
    staged.arena_banks = RunResourceSlots<std::unique_ptr<ArenaBank>>::build(granted);
    for (std::unique_ptr<ArenaBank> &bank : staged.arena_banks) {
        bank = std::make_unique<ArenaBank>(&arena_alloc_trampoline, &arena_free_trampoline, &mem_alloc_);
    }
    return staged;
}

/** Adopt a built grant by swapping every container. Never throws. */
void SimDeviceRunnerBase::commit_run_resources(size_t granted, RunResourceStaging &&staged) noexcept {
    retained_temp_addrs_.commit(std::move(staged.retained_temp_addrs));
    retained_temp_sizes_.commit(std::move(staged.retained_temp_sizes));
    graph_definition_staging_.commit(std::move(staged.graph_definition_staging));
    scheduler_state_storage_.commit(std::move(staged.scheduler_state_storage));
    orchestration_scratch_.commit(std::move(staged.orchestration_scratch));
    sm_mirrors_.commit(std::move(staged.sm_mirrors));
    run_image_stagings_.commit(std::move(staged.run_image_stagings));
    arena_banks_.commit(std::move(staged.arena_banks));
    host_phase_runs_.commit(std::move(staged.host_phase_runs));
    device_run_results_.commit(std::move(staged.device_run_results));
    // Last: until this is published, no run may name a set in the new tables.
    granted_run_resource_sets_ = granted;
}

bool SimDeviceRunnerBase::set_run_resource_grant(size_t granted) {
    if (granted == 0) return false;
    if (granted == granted_run_resource_sets_) return true;
    // Anything already committed belongs to a set at its current index, and a
    // run's lease names its set by that index.
    if (active_native_run_.load(std::memory_order_acquire) != nullptr) return false;
    for (const std::unique_ptr<ArenaBank> &bank : arena_banks_) {
        if (bank != nullptr &&
            (bank->gm_heap.is_committed() || bank->gm_sm.is_committed() || bank->runtime_pool.is_committed())) {
            return false;
        }
    }
    // Build first, publish second: a failure here leaves the previous grant
    // whole and usable rather than a corrupted runner that cleanup then walks.
    try {
        RunResourceStaging staged = build_run_resources(granted);
        commit_run_resources(granted, std::move(staged));
    } catch (...) {
        LOG_ERROR(
            "set_run_resource_grant: could not build %zu run-resource sets; keeping the previous grant of %zu", granted,
            granted_run_resource_sets_
        );
        return false;
    }
    return true;
}

uint64_t SimDeviceRunnerBase::arena_bank_gm_heap_base(uint32_t bank_id) const {
    if (bank_id >= arena_banks_.size()) return 0;
    const ArenaBank &bank = *arena_banks_[bank_id];
    return bank.gm_heap.is_committed() ? reinterpret_cast<uint64_t>(bank.gm_heap.base()) : 0;
}

uint64_t SimDeviceRunnerBase::retained_temp_addr(uint32_t slot_id) const {
    if (slot_id >= retained_temp_addrs_.size()) return 0;
    return reinterpret_cast<uint64_t>(retained_temp_addrs_[slot_id]);
}

int SimDeviceRunnerBase::setup_static_arena(
    uint32_t arena_bank, size_t gm_heap_size, size_t gm_sm_size, size_t runtime_arena_size
) {
    if (arena_bank >= arena_banks_.size()) return PTO_RUNTIME_ERR_INTERNAL;
    ArenaBank &bank = this->arena_bank(arena_bank);
    // Three independent device_malloc'd buffers: GM heap, shared memory, prebuilt
    // runtime arena. Split out from a single large allocation because the
    // combined size can exceed the device allocator's largest contiguous
    // block. Each arena commits exactly one region, so its base() is the
    // pooled pointer the caller wants.
    //
    // Idempotent for the production case (sizes do not change across a
    // worker's lifetime). If a caller asks for a larger layout on any
    // region, redo just that region.
    bool arena_changed = false;
    auto commit_region = [&arena_changed](DeviceArena &arena, size_t &cached_size, size_t requested_size) -> int {
        if (requested_size == 0) {
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
            arena.release();
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        cached_size = requested_size;
        return 0;
    };
    // Failure of any region releases all peers — mirrors the onboard "rollback
    // all on any failure" semantic (PR #922). Pooled pointers from a prior
    // successful call stay valid; a failed resize attempt does not leave a
    // partial layout behind.
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
        prebuilt_runtime_arena_cache_valid_ = false;
        prebuilt_runtime_arena_cache_key_.clear();
        prebuilt_runtime_arena_cache_gm_heap_base_ = nullptr;
        prebuilt_runtime_arena_cache_sm_base_ = nullptr;
        prebuilt_runtime_arena_cache_runtime_arena_base_ = nullptr;
        prebuilt_runtime_arena_cache_image_.clear();
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (arena_changed) {
        prebuilt_runtime_arena_cache_valid_ = false;
        prebuilt_runtime_arena_cache_key_.clear();
        prebuilt_runtime_arena_cache_gm_heap_base_ = nullptr;
        prebuilt_runtime_arena_cache_sm_base_ = nullptr;
        prebuilt_runtime_arena_cache_runtime_arena_base_ = nullptr;
        prebuilt_runtime_arena_cache_image_.clear();
    }
    return 0;
}

bool SimDeviceRunnerBase::lookup_prebuilt_runtime_arena_cache(
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

void SimDeviceRunnerBase::mark_prebuilt_runtime_arena_cached(
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

void *SimDeviceRunnerBase::acquire_pooled_gm_heap(uint32_t arena_bank) {
    if (arena_bank >= arena_banks_.size()) return nullptr;
    DeviceArena &arena = this->arena_bank(arena_bank).gm_heap;
    if (!arena.is_committed()) return nullptr;
    return arena.base();
}

void *SimDeviceRunnerBase::acquire_pooled_gm_sm(uint32_t arena_bank) {
    if (arena_bank >= arena_banks_.size()) return nullptr;
    DeviceArena &arena = this->arena_bank(arena_bank).gm_sm;
    if (!arena.is_committed()) return nullptr;
    return arena.base();
}

void *SimDeviceRunnerBase::acquire_pooled_runtime_arena(uint32_t arena_bank) {
    if (arena_bank >= arena_banks_.size()) return nullptr;
    DeviceArena &arena = this->arena_bank(arena_bank).runtime_pool;
    if (!arena.is_committed()) return nullptr;
    return arena.base();
}

std::thread SimDeviceRunnerBase::create_thread(std::function<void()> fn, std::string name) {
    int dev_id = device_id_;
    return std::thread([dev_id, fn = std::move(fn), name = std::move(name)]() {
#if defined(__linux__)
        // Linux caps a thread name at 15 characters plus NUL and fails the call
        // outright on a longer one, which would leave the thread unnamed.
        if (!name.empty()) {
            pthread_setname_np(pthread_self(), name.substr(0, 15).c_str());
        }
#endif
        pto_cpu_sim_bind_device(dev_id);
        fn();
        pto_cpu_sim_bind_device(-1);
    });
}

int SimDeviceRunnerBase::attach_current_thread(int device_id) {
    if (device_id < 0) {
        LOG_ERROR("Invalid device_id: %d", device_id);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (device_id_ != -1 && device_id_ != device_id) {
        LOG_ERROR(
            "DeviceRunner already initialized on device %d; finalize before switching to device %d", device_id_,
            device_id
        );
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    // Per-thread bind so sim hooks (TPUSH/TPOP, identity helpers) route through
    // the correct context. acquire is process-wide and idempotent (no-op after
    // first call for a given device_id), so it is safe to fold in here.
    pto_cpu_sim_bind_device(device_id);
    pto_cpu_sim_acquire_device(device_id);
    device_id_ = device_id;
    return 0;
}

int SimDeviceRunnerBase::ensure_device_initialized() {
    // device_id_ was set in attach_current_thread() during simpler_init.
    int rc = attach_current_thread(device_id_);
    if (rc != 0) return rc;
    return ensure_binaries_loaded();
}

int SimDeviceRunnerBase::ensure_dma_workspace_provisioned() {
    if (dma_workspace_handle_ != nullptr) {
        return 0;
    }
    const uint32_t supported = dma_workspace_supported_mask();
    constexpr uint32_t kSdmaBit = uint32_t{1} << DMA_WORKSPACE_SDMA;
    if (sdma_requested_ && (supported & kSdmaBit) == 0) {
        LOG_ERROR("dma workspace: SDMA requested where unsupported (supported=0x%x)", supported);
        return PTO_RUNTIME_ERR_UNSUPPORTED;
    }

    const uint32_t required_mask = sdma_requested_ ? supported : (supported & ~kSdmaBit);
    if (required_mask == 0) {
        return 0;
    }
    if ((required_mask & (required_mask - 1)) != 0) {
        LOG_ERROR(
            "dma workspace: mask=0x%x names %d engines; one handle owns one provider", required_mask,
            __builtin_popcount(required_mask)
        );
        return PTO_RUNTIME_ERR_UNSUPPORTED;
    }

    for (int kind = 0; kind < DMA_WORKSPACE_KIND_COUNT; ++kind) {
        dma_workspace_addr_[kind] = 0;
    }

    int rc =
        dma_workspace_provision(required_mask, dma_workspace_addr_, DMA_WORKSPACE_KIND_COUNT, &dma_workspace_handle_);
    if (rc != 0) {
        LOG_ERROR("dma workspace: mask=0x%x failed: %d", required_mask, rc);
        for (int kind = 0; kind < DMA_WORKSPACE_KIND_COUNT; ++kind) {
            dma_workspace_addr_[kind] = 0;
        }
        dma_workspace_handle_ = nullptr;
        return rc;
    }
    return 0;
}

int SimDeviceRunnerBase::prepare_launch_shape(Runtime &runtime, const CallConfig &config) {
    if (config.aicpu_thread_num == 1 || config.aicpu_thread_num < 0 ||
        config.aicpu_thread_num > PLATFORM_MAX_AICPU_THREADS) {
        LOG_ERROR(
            "launch_aicpu_num (%d) must be 0 (auto) or in range [2, %d]", config.aicpu_thread_num,
            PLATFORM_MAX_AICPU_THREADS
        );
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    // A run always takes the whole simulated device; orchestration sizes its
    // cohorts from rt_available_cluster_count() rather than a per-call width.
    const int block_dim = SIM_AUTO_BLOCKDIM;
    LOG_INFO("block_dim resolved to %d", block_dim);

    int num_aicore = block_dim * cores_per_blockdim_;
    if (num_aicore > RUNTIME_MAX_WORKER) {
        LOG_ERROR("num_aicore (%d) exceeds RUNTIME_MAX_WORKER (%d)", num_aicore, RUNTIME_MAX_WORKER);
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    block_dim_ = block_dim;
    runtime.set_worker_count(num_aicore);
    worker_count_ = num_aicore;
    runtime.set_aicpu_thread_num(config.aicpu_thread_num);

    // First `block_dim` cores are AIC; remaining ~2/3 are AIV. The rule is host
    // state read by the sim's per-core thread start and by the DFX collector.
    runtime.set_core_type_rule(num_aicore, block_dim);

    // The sim has no upload: its threads read this host object in place, so
    // `workers[]` here is the live handshake region and this loop is its
    // per-run initialization, not a staged copy.
    Handshake *workers = runtime.get_workers();
    for (int i = 0; i < num_aicore; i++) {
        workers[i].aicpu_ready = 0;
        workers[i].aicore_done = 0;
        workers[i].task = 0;
        workers[i].core_type = (i < block_dim) ? CoreType::AIC : CoreType::AIV;
        // Cleared with the rest of the report so a run's own epoch is the only
        // value that can ever satisfy its sweep.
        workers[i].report_epoch = 0;
    }
    return 0;
}

void *SimDeviceRunnerBase::allocate_tensor(size_t bytes) { return mem_alloc_.alloc(bytes); }

void *SimDeviceRunnerBase::allocate_caller_buffer(size_t bytes) {
    void *ptr = allocate_tensor(bytes);
    if (ptr == nullptr) return nullptr;
    try {
        caller_device_buffers_.record(ptr, bytes);
    } catch (...) {
        // Rolled back for the reason the onboard twin gives: the record is what makes the
        // allocation the caller's, so without it there is nothing to hand back.
        free_tensor(ptr);
        LOG_ERROR("allocate_caller_buffer: could not record %zu bytes as a caller allocation; rolled it back", bytes);
        return nullptr;
    }
    return ptr;
}

int SimDeviceRunnerBase::free_caller_buffer(void *dev_ptr) {
    if (dev_ptr == nullptr) return 0;
    // Same one-step check-and-forget as the onboard runner; see its comment for why the two are
    // one operation and why an unrecorded address passes through unchanged.
    if (!caller_device_buffers_.forget_if_unborrowed(dev_ptr)) {
        LOG_ERROR(
            "free_caller_buffer: %p is still held by %zu borrowing run(s) and %zu retained "
            "reference(s); the caller's release is refused rather than performed",
            dev_ptr, caller_device_buffers_.borrow_count(), caller_device_buffers_.retained_count()
        );
        return PTO_RUNTIME_ERR_INVALID_STATE;
    }
    free_tensor(dev_ptr);
    return 0;
}

bool SimDeviceRunnerBase::borrow_caller_buffers(
    uint64_t identity, const CallerDeviceBuffers::Span *spans, size_t count
) {
    return caller_device_buffers_.borrow(identity, spans, count);
}

void SimDeviceRunnerBase::release_caller_buffers(uint64_t identity, bool keep) {
    caller_device_buffers_.release(identity, keep);
}

void SimDeviceRunnerBase::free_tensor(void *dev_ptr) {
    if (dev_ptr != nullptr) {
        mem_alloc_.free(dev_ptr);
    }
}

void *SimDeviceRunnerBase::acquire_child_memory_host_view(void *dev_ptr, size_t bytes) {
    if (dev_ptr == nullptr || bytes == 0) return nullptr;
    void *alloc_base = nullptr;
    size_t alloc_size = 0;
    if (!mem_alloc_.owning_allocation(dev_ptr, &alloc_base, &alloc_size)) {
        LOG_ERROR("acquire_child_memory_host_view: %p is not inside a tracked device allocation", dev_ptr);
        return nullptr;
    }
    const auto *end = static_cast<const unsigned char *>(dev_ptr) + bytes;
    if (end > static_cast<const unsigned char *>(alloc_base) + alloc_size) {
        LOG_ERROR(
            "acquire_child_memory_host_view: [%p, +%zu) overruns its allocation [%p, +%zu)", dev_ptr, bytes, alloc_base,
            alloc_size
        );
        return nullptr;
    }
    return dev_ptr;
}

int SimDeviceRunnerBase::copy_to_device(void *dev_ptr, const void *host_ptr, size_t bytes) {
    std::memcpy(dev_ptr, host_ptr, bytes);
    return 0;
}

int SimDeviceRunnerBase::copy_from_device(void *host_ptr, const void *dev_ptr, size_t bytes) {
    std::memcpy(host_ptr, dev_ptr, bytes);
    return 0;
}

int SimDeviceRunnerBase::device_memset(void *dev_ptr, int value, size_t bytes) {
    std::memset(dev_ptr, value, bytes);
    return 0;
}

uint64_t SimDeviceRunnerBase::run_result_region_base(uint32_t pipeline_slot) {
    if (pipeline_slot >= device_run_results_.size()) return 0;
    return reinterpret_cast<uint64_t>(&device_run_results_[pipeline_slot]);
}

const uint8_t *
SimDeviceRunnerBase::device_run_result(uint32_t pipeline_slot, uint64_t run_epoch, size_t *bytes_out) const {
    if (bytes_out != nullptr) *bytes_out = 0;
    if (pipeline_slot >= device_run_results_.size()) return nullptr;
    const DeviceRunResultRegion &region = device_run_results_[pipeline_slot];
    if (!device_run_result_published(region, run_epoch)) return nullptr;
    if (bytes_out != nullptr) *bytes_out = region.payload_bytes;
    return region.payload;
}

void SimDeviceRunnerBase::get_retained_temp_buffer(uint32_t pipeline_slot, void **addr, size_t *size) {
    if (pipeline_slot >= retained_temp_addrs_.size()) {
        if (addr != nullptr) *addr = nullptr;
        if (size != nullptr) *size = 0;
        return;
    }
    if (addr != nullptr) *addr = retained_temp_addrs_[pipeline_slot];
    if (size != nullptr) *size = retained_temp_sizes_[pipeline_slot];
}

void SimDeviceRunnerBase::set_retained_temp_buffer(uint32_t pipeline_slot, void *addr, size_t size) {
    if (pipeline_slot >= retained_temp_addrs_.size()) return;
    retained_temp_addrs_[pipeline_slot] = addr;
    retained_temp_sizes_[pipeline_slot] = size;
}

int SimDeviceRunnerBase::acquire_graph_staging(
    uint32_t pipeline_slot, size_t bytes, size_t alignment, void **staging_out
) {
    if (staging_out == nullptr) return -1;
    *staging_out = nullptr;
    if (pipeline_slot >= graph_definition_staging_.size() || bytes == 0 || alignment == 0 ||
        (alignment & (alignment - 1)) != 0) {
        return -1;
    }
    // A byte vector's storage is aligned for every type with fundamental
    // alignment, which is what a Definition section asks for.
    if (alignment > alignof(std::max_align_t)) return -1;
    std::vector<std::byte> &staging = graph_definition_staging_[pipeline_slot].bytes;
    // Grow-only and content-preserving: the recorders' objects are already in
    // this block's prefix at the offsets they claimed.
    if (staging.size() < bytes) staging.resize(bytes);
    *staging_out = staging.data();
    return 0;
}

int SimDeviceRunnerBase::publish_graph_section(
    uint32_t pipeline_slot, const void *bytes, size_t length, uint32_t *source_out, uint64_t *base_out
) {
    if (source_out != nullptr) *source_out = static_cast<uint32_t>(GraphSectionSource::None);
    if (base_out != nullptr) *base_out = 0;
    if (bytes == nullptr || length == 0) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    if (pipeline_slot >= graph_definition_staging_.size()) return PTO_RUNTIME_ERR_INTERNAL;
    if (length > UINT32_MAX) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    std::vector<std::byte> &snapshot = graph_definition_staging_[pipeline_slot].snapshot;
    if (snapshot.size() < length) snapshot.resize(length);
    std::memcpy(snapshot.data(), bytes, length);
    if (source_out != nullptr) *source_out = static_cast<uint32_t>(GraphSectionSource::HostSnapshot);
    if (base_out != nullptr) *base_out = reinterpret_cast<uint64_t>(snapshot.data());
    return 0;
}

void SimDeviceRunnerBase::get_graph_definition_staging(uint32_t pipeline_slot, void **addr, size_t *size) {
    if (addr != nullptr) *addr = nullptr;
    if (size != nullptr) *size = 0;
    if (pipeline_slot >= graph_definition_staging_.size()) return;
    std::vector<std::byte> &staging = graph_definition_staging_[pipeline_slot].bytes;
    if (staging.empty()) return;
    if (addr != nullptr) *addr = staging.data();
    if (size != nullptr) *size = staging.size();
}

int SimDeviceRunnerBase::acquire_scheduler_state_storage(
    uint32_t pipeline_slot, size_t bytes, size_t alignment, void **device_out, void **host_out
) {
    if (device_out != nullptr) *device_out = nullptr;
    if (host_out != nullptr) *host_out = nullptr;
    if (pipeline_slot >= scheduler_state_storage_.size()) return -1;
    if (!can_accept_run()) {
        LOG_ERROR("scheduler-state storage: refusing slot %u on a runner that cannot accept a run", pipeline_slot);
        return -1;
    }
    const RetainedSchedulerStorage::Status status = scheduler_state_storage_[pipeline_slot].acquire(
        bytes, alignment,
        [this](size_t raw_bytes) {
            return mem_alloc_.alloc(raw_bytes);
        },
        [this](void *ptr) {
            return mem_alloc_.free(ptr);
        },
        device_out, host_out
    );
    if (status == RetainedSchedulerStorage::Status::Ok) return 0;
    LOG_ERROR(
        "scheduler-state storage: slot %u cannot serve %zu bytes (alignment %zu): status %u", pipeline_slot, bytes,
        alignment, static_cast<uint32_t>(status)
    );
    return -1;
}

int SimDeviceRunnerBase::release_scheduler_state_storage() {
    int first_error = 0;
    for (RetainedSchedulerStorage &entry : scheduler_state_storage_) {
        const int rc = entry.release([this](void *ptr) {
            return mem_alloc_.free(ptr);
        });
        if (rc != 0 && first_error == 0) first_error = rc;
    }
    return first_error;
}

int SimDeviceRunnerBase::acquire_sm_mirror(uint32_t pipeline_slot, size_t bytes, size_t alignment, void **addr_out) {
    if (addr_out == nullptr) return -1;
    *addr_out = nullptr;
    if (pipeline_slot >= sm_mirrors_.size() || bytes == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0 ||
        bytes > SIZE_MAX - (alignment - 1)) {
        return -1;
    }
    RetainedSmMirror &mirror = sm_mirrors_[pipeline_slot];
    // Grow-only and never shrunk: the task capacity is fixed for a given run
    // configuration, so past the first bind the image is written into host pages
    // that are already mapped, and each page of it faults once per process rather
    // than once per bind.
    const size_t needed = bytes + alignment - 1;
    if (mirror.capacity < needed) {
        // `new[]` on a trivially-typed array default-initializes, so the block
        // costs no page until a bind writes one; make_unique would zero it. The
        // outgoing block's bytes are not carried over, because nothing reads a byte
        // this bind did not write.
        std::unique_ptr<std::byte[]> storage(new (std::nothrow) std::byte[needed]);
        if (storage == nullptr) return -1;
        mirror.storage = std::move(storage);
        mirror.capacity = needed;
    }
    const uintptr_t raw = reinterpret_cast<uintptr_t>(mirror.storage.get());
    *addr_out = reinterpret_cast<void *>((raw + alignment - 1) & ~static_cast<uintptr_t>(alignment - 1));
    return 0;
}

int SimDeviceRunnerBase::acquire_run_image_staging(
    uint32_t pipeline_slot, size_t bytes, size_t alignment, void **addr_out
) {
    if (addr_out == nullptr) return -1;
    *addr_out = nullptr;
    if (pipeline_slot >= run_image_stagings_.size() || bytes == 0 || alignment == 0 ||
        (alignment & (alignment - 1)) != 0 || bytes > SIZE_MAX - (alignment - 1)) {
        return -1;
    }
    RetainedSmMirror &staging = run_image_stagings_[pipeline_slot];
    // Grow-only, like the mirror: an image's size follows the graph a run builds,
    // so a repeated workload writes host pages that are already mapped.
    const size_t needed = bytes + alignment - 1;
    if (staging.capacity < needed) {
        // `new[]` default-initializes a trivially-typed array, so the block costs
        // no page until the bind writes one. The outgoing block's bytes are not
        // carried over: a publication ships what its own bind assembled.
        std::unique_ptr<std::byte[]> storage(new (std::nothrow) std::byte[needed]);
        if (storage == nullptr) return -1;
        staging.storage = std::move(storage);
        staging.capacity = needed;
    }
    const uintptr_t raw = reinterpret_cast<uintptr_t>(staging.storage.get());
    *addr_out = reinterpret_cast<void *>((raw + alignment - 1) & ~static_cast<uintptr_t>(alignment - 1));
    return 0;
}

void SimDeviceRunnerBase::release_run_image_stagings() {
    for (RetainedSmMirror &staging : run_image_stagings_) {
        staging.storage.reset();
        staging.capacity = 0;
    }
}

void SimDeviceRunnerBase::release_sm_mirrors() {
    for (auto &scratch : orchestration_scratch_)
        scratch.reset();
    for (RetainedSmMirror &mirror : sm_mirrors_) {
        mirror.storage.reset();
        mirror.capacity = 0;
    }
}

void SimDeviceRunnerBase::release_graph_definition_staging() {
    for (RetainedGraphStaging &staging : graph_definition_staging_) {
        staging.bytes.clear();
        staging.bytes.shrink_to_fit();
        staging.snapshot.clear();
        staging.snapshot.shrink_to_fit();
    }
}

void SimDeviceRunnerBase::clear_temporary_buffer() {
    for (size_t slot = 0; slot < retained_temp_addrs_.size(); ++slot) {
        if (retained_temp_addrs_[slot] == nullptr) continue;
        mem_alloc_.free(retained_temp_addrs_[slot]);
        retained_temp_addrs_[slot] = nullptr;
        retained_temp_sizes_[slot] = 0;
    }
}

int SimDeviceRunnerBase::stamp_orch_so(Runtime &runtime, int32_t cid) {
    // Registered-callable flow only: the orch SO was already delivered to the
    // sim AICPU at launch_device_register time. A run just needs the active
    // callable_id so the AICPU dispatches the right orch_so_table_ slot.
    if (cid < 0) {
        LOG_ERROR("stamp_orch_so: invalid callable_id=%d", cid);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    auto it = callables_.find(cid);
    if (it == callables_.end()) {
        LOG_ERROR("stamp_orch_so: callable_id=%d not registered", cid);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    runtime.set_active_callable_id(cid);
    return 0;
}

int SimDeviceRunnerBase::prepare_orch_so(Runtime &runtime) {
    const int32_t cid = runtime.get_active_callable_id();
    if (cid < 0) {
        LOG_ERROR("prepare_orch_so: no active callable_id; prepared-callable flow required");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    return stamp_orch_so(runtime, cid);
}

int SimDeviceRunnerBase::commit_device_register(int32_t cid) {
    auto it = callables_.find(cid);
    if (it == callables_.end()) {
        LOG_ERROR("commit_device_register: callable_id=%d not registered", cid);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (it->second.host_dlopen_handle != nullptr) {
        return 0;
    }
    const bool inserted = aicpu_seen_callable_ids_.insert(cid).second;
    if (inserted) {
        ++aicpu_dlopen_total_;
        LOG_INFO("AICPU callable load committed cid=%d (count=%zu)", cid, aicpu_dlopen_total_);
    }
    return 0;
}

int SimDeviceRunnerBase::launch_device_register(int32_t callable_id) {
    auto it = callables_.find(callable_id);
    if (it == callables_.end()) {
        LOG_ERROR("launch_device_register: callable_id=%d not registered", callable_id);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (it->second.host_dlopen_handle != nullptr) {
        return 0;
    }

    int rc = ensure_device_initialized();
    if (rc != 0) {
        LOG_ERROR("launch_device_register: ensure_device_initialized failed: %d", rc);
        return rc;
    }

    // Build the orch-SO descriptor straight from CallableState — no throwaway
    // Runtime + stamp round-trip. Mirrors the onboard launch_device_register.
    const CallableState &state = it->second;
    RegisterCallableArgs reg_args{};
    reg_args.active_callable_id = callable_id;
    reg_args.dev_orch_so_addr = state.dev_orch_so_addr;
    reg_args.dev_orch_so_size = state.dev_orch_so_size;
    snprintf(reg_args.device_orch_func_name, sizeof(reg_args.device_orch_func_name), "%s", state.func_name.c_str());
    snprintf(
        reg_args.device_orch_config_name, sizeof(reg_args.device_orch_config_name), "%s", state.config_name.c_str()
    );

    rc = invoke_device_register(reg_args);
    if (rc != 0) {
        LOG_ERROR("launch_device_register: invoke_device_register failed: %d", rc);
        return rc;
    }

    return commit_device_register(callable_id);
}

int SimDeviceRunnerBase::record_device_orch_callable(
    int32_t callable_id, uint64_t chip_buffer_hash, uint64_t chip_dev, const void *orch_so_data, size_t orch_so_size,
    const char *func_name, const char *config_name, std::vector<ArgDirection> signature
) {
    // The AICPU executor reserves `orch_so_table_[MAX_REGISTERED_CALLABLE_IDS]`
    // (declared in src/common/task_interface/callable_protocol.h) and indexes
    // it by callable_id; rejecting an out-of-range id here keeps the host and
    // AICPU sides in sync and avoids an OOB access at run time.
    if (callable_id < 0 || callable_id >= MAX_REGISTERED_CALLABLE_IDS) {
        LOG_ERROR(
            "record_device_orch_callable: callable_id=%d out of range [0, %d)", callable_id, MAX_REGISTERED_CALLABLE_IDS
        );
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (orch_so_data == nullptr || orch_so_size == 0) {
        LOG_ERROR("record_device_orch_callable: empty orch SO for callable_id=%d", callable_id);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (chip_buffer_hash == 0 || chip_dev == 0) {
        LOG_ERROR("record_device_orch_callable: missing chip buffer for callable_id=%d", callable_id);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (callables_.count(callable_id) != 0) {
        LOG_ERROR("record_device_orch_callable: callable_id=%d already registered", callable_id);
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    const uint64_t hash = simpler::common::utils::elf_build_id_64(orch_so_data, orch_so_size);

    CallableState state;
    state.hash = hash;
    state.chip_buffer_hash = chip_buffer_hash;
    state.dev_orch_so_addr = chip_dev + offsetof(ChipCallable, storage_);
    state.dev_orch_so_size = orch_so_size;
    state.func_name = (func_name != nullptr) ? func_name : "";
    state.config_name = (config_name != nullptr) ? config_name : "";
    state.signature = std::move(signature);
    callables_.emplace(callable_id, std::move(state));
    LOG_INFO(
        "record_device_orch_callable: cid=%d orch_hash=0x%lx chip_hash=0x%lx %zu bytes", callable_id, hash,
        chip_buffer_hash, orch_so_size
    );
    return 0;
}

int SimDeviceRunnerBase::record_host_orch_callable(
    int32_t callable_id, uint64_t chip_buffer_hash, void *host_dlopen_handle, void *host_orch_func_ptr,
    std::vector<ArgDirection> signature
) {
    if (callable_id < 0 || callable_id >= MAX_REGISTERED_CALLABLE_IDS) {
        LOG_ERROR(
            "record_host_orch_callable: callable_id=%d out of range [0, %d)", callable_id, MAX_REGISTERED_CALLABLE_IDS
        );
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (host_dlopen_handle == nullptr || host_orch_func_ptr == nullptr) {
        LOG_ERROR("record_host_orch_callable: null handle/fn for callable_id=%d", callable_id);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (chip_buffer_hash == 0) {
        LOG_ERROR("record_host_orch_callable: missing chip buffer for callable_id=%d", callable_id);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (callables_.count(callable_id) != 0) {
        LOG_ERROR("record_host_orch_callable: callable_id=%d already registered", callable_id);
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    CallableState state;
    state.chip_buffer_hash = chip_buffer_hash;
    state.host_dlopen_handle = host_dlopen_handle;
    state.host_orch_func_ptr = host_orch_func_ptr;
    state.signature = std::move(signature);
    callables_.emplace(callable_id, std::move(state));
    ++host_dlopen_total_;
    LOG_INFO("record_host_orch_callable: cid=%d (host dlopen #%zu)", callable_id, host_dlopen_total_);
    return 0;
}

int SimDeviceRunnerBase::unregister_callable(int32_t callable_id) {
    auto it = callables_.find(callable_id);
    if (it == callables_.end()) {
        return 0;
    }
    CallableState state = std::move(it->second);
    callables_.erase(it);
    aicpu_seen_callable_ids_.erase(callable_id);
    release_chip_callable_buffer(state.chip_buffer_hash);

    if (state.host_dlopen_handle != nullptr) {
        // hbg: dlclose the host handle; no device-side orch SO handle.
        dlclose(state.host_dlopen_handle);
        return 0;
    }
    return 0;
}

bool SimDeviceRunnerBase::has_callable(int32_t callable_id) const { return callables_.count(callable_id) != 0; }

// Per-run binding half, defined in each runtime's runtime_maker.cpp and linked
// into this same sim runtime .so. Declared here (rather than only in
// c_api_shared.cpp) so bind_callable_to_runtime can call it directly, keeping
// the CallableState-derived host_orch_func_ptr / signature internal to the
// runner instead of returning them across the c_api boundary.
extern "C" int bind_callable_to_runtime_impl(
    Runtime *runtime, const HostApi *api, const ChipStorageTaskArgs *orch_args, void *host_orch_func_ptr,
    const ArgDirection *signature, int sig_count, const uint64_t *ring_task_window, const uint64_t *ring_heap,
    const uint64_t *ring_dep_pool
);

int SimDeviceRunnerBase::bind_callable_to_runtime(
    Runtime &runtime, int32_t callable_id, const HostApi *api, const void *orch_args, const uint64_t *ring_task_window,
    const uint64_t *ring_heap, const uint64_t *ring_dep_pool
) {
    // Clear before anything else, including before the registry lookup: an id
    // that is gone is exactly the case where the reference still standing names
    // a scratch `unregister_callable` already freed. So no bind — refused,
    // failed, or successful — may leave a predecessor's.
    runtime.clear_callable_tables();
    auto it = callables_.find(callable_id);
    if (it == callables_.end()) {
        LOG_ERROR("bind_callable_to_runtime: callable_id=%d not registered", callable_id);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    const auto &state = it->second;

    auto block = chip_callable_buffers_.find(state.chip_buffer_hash);
    if (block == chip_callable_buffers_.end()) {
        LOG_ERROR("bind_callable_to_runtime: callable_id=%d has no retained registration block", callable_id);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    // Sim holds the tables in host memory the registration built, so the
    // reference is the same on both sides and nothing is copied per run.
    const ChipCallableBuffer &tables = block->second;
    runtime.set_callable_tables(
        tables.object_table.empty() ? nullptr : tables.object_table.data(), tables.object_table_dev,
        tables.entry_table_dev, tables.table_len
    );
    // The AICPU dispatches the orch SO via this callable_id; the SO descriptor
    // was already delivered at launch_device_register time.
    runtime.set_active_callable_id(callable_id);

    // Per-run binding. host_orch_func_ptr + signature come from CallableState and
    // stay inside the runner — no longer returned to the c_api.
    return bind_callable_to_runtime_impl(
        &runtime, api, reinterpret_cast<const ChipStorageTaskArgs *>(orch_args), state.host_orch_func_ptr,
        state.signature.empty() ? nullptr : state.signature.data(), static_cast<int>(state.signature.size()),
        ring_task_window, ring_heap, ring_dep_pool
    );
}

// Eager prebuilt-arena warm-up. A runtime with a prebuilt runtime arena
// (tensormap_and_ringbuffer) provides a strong prewarm_config_impl in its
// runtime_maker.cpp that overrides this weak no-op default; runtimes without one
// link the weak default and treat prewarm as a no-op. simpler_init calls it
// directly for the fork-constant ring sizing once the runner is attached.
extern "C" __attribute__((weak)) int prewarm_config_impl(
    const HostApi * /*api*/, const uint64_t * /*ring_task_window*/, const uint64_t * /*ring_heap*/,
    const uint64_t * /*ring_dep_pool*/
) {
    return 0;
}

void SimDeviceRunnerBase::apply_call_config(const CallConfig &config) {
    set_chip_swimlane_enabled(config.enable_chip_swimlane);
    set_pmu_enabled(config.enable_pmu > 0);
    set_output_prefix(config.output_prefix);
}

void SimDeviceRunnerBase::begin_host_phase_run(uint32_t pipeline_slot, const DfxRunConfig &dfx) {
    if (pipeline_slot >= host_phase_runs_.size()) return;
    host_phase_runs_[pipeline_slot].begin(dfx);
}

HostPhaseRecordPool *
SimDeviceRunnerBase::host_phase_pool_arm(uint32_t pipeline_slot, bool producer_wants_records) noexcept {
    if (pipeline_slot >= host_phase_runs_.size()) return nullptr;
    HostPhaseRunState &run = host_phase_runs_[pipeline_slot];
    run.host_orchestrated = run.chip_swimlane_level == ChipSwimlaneLevel::ORCH_PHASES;
    // arm() allocates the pool's buffers, so it can throw; this path is noexcept,
    // where an escaping exception is std::terminate. A pass that cannot get its
    // storage collects no records and says so by handing back nullptr.
    HostPhaseRecordPool *pool = nullptr;
    try {
        pool = run.records.arm(run.wants_records(producer_wants_records));
    } catch (...) {
        LOG_WARN("Host phase pool could not be armed; this pass collects no per-event records");
    }
    if (!run.host_orchestrated) return pool;

    return pool;
}

void SimDeviceRunnerBase::publish_host_phase_run_to_collector(uint32_t pipeline_slot) noexcept {
    if (pipeline_slot >= host_phase_runs_.size()) return;
    const HostPhaseRunState &run = host_phase_runs_[pipeline_slot];
    chip_swimlane_collector_.set_host_orchestrated(run.host_orchestrated);
}

void SimDeviceRunnerBase::publish_host_phase_records_to_swimlane(uint32_t pipeline_slot) {
    if (pipeline_slot >= host_phase_runs_.size()) return;
    const simpler::dfx::HostPhaseRecordStore &records = host_phase_runs_[pipeline_slot].records;
    if (!records.finished()) return;
    chip_swimlane_collector_.set_host_phase_records(
        records.submit_records(), records.device_upload_records(), records.submitted_tasks(), records.total_records(),
        records.dropped_records()
    );
}

int SimDeviceRunnerBase::start_shared_collectors_for_run(
    const DfxRunConfig &dfx, uint64_t run_epoch, const RunLocalDfxMetadata &run_metadata, DiagnosticAdmission *admission
) {
    // Read before the first owner is asked. A release that lands between this
    // read and a caller's wait has already advanced past it, so the caller
    // cannot park on a generation that is already stale — the whole of the
    // lost-wakeup argument.
    simpler::dfx::runs::AdmissionRefusal why = simpler::dfx::runs::AdmissionRefusal::None;
    if (admission != nullptr) {
        *admission = DiagnosticAdmission{};
        admission->credit_generation = credit_signal_.generation();
    }
    auto decline = [&](const char *owner) {
        if (admission != nullptr) {
            admission->refusal =
                why == simpler::dfx::runs::AdmissionRefusal::None ? simpler::dfx::runs::AdmissionRefusal::Fatal : why;
            admission->owner = owner;
        }
    };
    // All or nothing: an owner that declines after earlier ones admitted gives
    // their admissions back, and so does an exception unwinding through here.
    // Withdrawal is by run identity and is a no-op for an owner this run never
    // reached, so sweeping every enabled owner is exact rather than merely
    // convenient.
    bool admitted_all = false;
    struct RollbackGuard {
        SimDeviceRunnerBase *self;
        const DfxRunConfig &dfx;
        uint64_t run_epoch;
        const bool *done;
        ~RollbackGuard() {
            if (!*done) self->undo_admission_attempt(dfx, run_epoch);
        }
    } rollback{this, dfx, run_epoch, &admitted_all};

    // Opening a resident collector's window drops the previous run's records and
    // republishes the device level, so it belongs with the start, at launch.
    auto thread_factory = [this](std::function<void()> fn) {
        return create_thread(std::move(fn));
    };
    if (dfx.chip_swimlane_enabled()) {
        // Same rule as the onboard base: configuration picks the path, and a
        // run the retaining collector will not admit fails here rather than
        // falling into the single-run reset, which would drop a predecessor's
        // records. The reader shards start first because admission waits for
        // their acknowledgement of the run table.
        if (chip_swimlane_collector_.retains_runs()) {
            chip_swimlane_collector_.start(thread_factory);
            if (!chip_swimlane_collector_.run_begin(
                    run_epoch, dfx.output_prefix, dfx.chip_swimlane_level, run_metadata.core_types,
                    run_metadata.core_type_count, run_metadata.host_orchestrated, run_metadata.artifact_identity, &why
                )) {
                LOG_ERROR(
                    "ChipSwimlane: run %llu was not admitted for retained collection",
                    static_cast<unsigned long long>(run_epoch)
                );
                decline("chip_swimlane");
                return PTO_RUNTIME_ERR_INTERNAL;
            }
        } else {
            chip_swimlane_collector_.begin_run(dfx.output_prefix, dfx.chip_swimlane_level);
            chip_swimlane_collector_.start(thread_factory);
        }
    }
    if (dfx.dump_args_enabled()) {
        // Configuration picks the path, as it does for swimlane and PMU: a run
        // a retaining collector will not admit fails here, before any kernel is
        // submitted, rather than falling into the single-run reset that would
        // zero counters a predecessor's writer is still acknowledging against.
        if (dump_collector_.retains_runs()) {
            dump_collector_.start(thread_factory);
            if (!dump_collector_.run_begin(run_epoch, dfx.output_prefix, dfx.dump_args_level, &why)) {
                LOG_ERROR(
                    "ArgsDump: run %llu was not admitted for retained collection",
                    static_cast<unsigned long long>(run_epoch)
                );
                decline("args_dump");
                return PTO_RUNTIME_ERR_INTERNAL;
            }
        } else {
            dump_collector_.begin_run(dfx.output_prefix, dfx.dump_args_level);
            dump_collector_.start(thread_factory);
        }
    }
    if (dfx.pmu_enabled) {
        // Configuration picks the path, exactly as it does for swimlane: a run
        // a retaining collector will not admit fails here, before any kernel is
        // submitted, rather than falling into the single-run reset that would
        // drop a predecessor's rows.
        if (pmu_collector_.retains_runs()) {
            // Reader shards before admission, and only here: admitting a run
            // waits for every shard to acknowledge the new epoch table, and a
            // shard that has not been spawned cannot acknowledge anything.
            pmu_collector_.start(thread_factory);
            if (!pmu_collector_.run_begin(run_epoch, make_pmu_csv_path(dfx.output_prefix), dfx.pmu_event_type, &why)) {
                LOG_ERROR(
                    "PmuCollector: run %llu was not admitted for retained collection",
                    static_cast<unsigned long long>(run_epoch)
                );
                decline("pmu");
                return PTO_RUNTIME_ERR_INTERNAL;
            }
        } else {
            // The default path keeps its order: the window is opened, then the
            // threads that serve it start.
            pmu_collector_.begin_run(make_pmu_csv_path(dfx.output_prefix), dfx.pmu_event_type);
            pmu_collector_.start(thread_factory);
        }
    }
    if (dfx.scope_stats_enabled) {
        // Configuration picks the path, as it does for the other retaining
        // collectors: a run a retaining ScopeStats will not admit fails here,
        // before any kernel is submitted. The refusal that matters is the
        // quarantine one — `begin_run` would clear the very records a run
        // whose completion could not be proved is still holding.
        if (scope_stats_collector_.retains_runs()) {
            scope_stats_collector_.start(thread_factory);
            if (!scope_stats_collector_.run_begin(
                    run_epoch, dfx.output_prefix, &why, run_metadata.scope_stats_capacity
                )) {
                LOG_ERROR(
                    "ScopeStats: run %llu was not admitted for retained collection",
                    static_cast<unsigned long long>(run_epoch)
                );
                decline("scope_stats");
                return PTO_RUNTIME_ERR_INTERNAL;
            }
        } else {
            // The default path writes one file per boundary rather than a
            // per-run artifact, and on failure it keeps the behaviour it has
            // without the re-stamp: a run is never refused here. The helper has
            // already logged which buffer and which run it concerns.
            (void)scope_stats_collector_.begin_run(run_epoch);
            scope_stats_collector_.start(thread_factory);
        }
    }
    // Every enabled owner admitted. From here the run owns their slots and the
    // guard must not take them back; the boundary that retires them is this
    // run's own close, or `withdraw_unlaunched_collectors_for_run` if the
    // launch turns out to have submitted nothing.
    admitted_all = true;
    return 0;
}

// The terminal bank a run publishes into is the one its epoch acquired, not the
// one its run-resource set is numbered. The banks are a fixed device resource
// and the sets are as many as the grant allows, so the two counts need not agree
// for either to be correct.

uint64_t SimDeviceRunnerBase::arm_chip_swimlane_run_terminal_bank(uint64_t run_epoch) {
    // Zero means "publish no snapshot". Every path that cannot resolve a bank —
    // swimlane off, collector not initialized, no free bank, no run identity —
    // returns it rather than letting the device derive an address.
    const int bank = chip_swimlane_collector_.acquire_run_terminal_bank(run_epoch);
    if (bank < 0) return 0;
    void *armed = chip_swimlane_collector_.arm_run_terminal_bank(static_cast<uint32_t>(bank), run_epoch);
    if (armed == nullptr) {
        // The acquire above took the bank and nothing was armed into it, so no
        // producer can reach that address and the credit is proved free. Keeping
        // it would retire one of the three banks per failed arm, and the run
        // publishing no snapshot is already the reported outcome.
        chip_swimlane_collector_.release_run_terminal_bank(run_epoch);
        return 0;
    }
    return reinterpret_cast<uint64_t>(armed);
}

void SimDeviceRunnerBase::undo_admission_attempt(const DfxRunConfig &dfx, uint64_t run_epoch) noexcept {
    bool restored = false;
    {
        // Giving an owner's slot back restores the state every waiter already
        // saw before this attempt, so it is progress for nobody — and notifying
        // there would wake this very thread onto its own bump, letting a
        // repeated attempt manufacture its own wake every round while the
        // blocking owner frees nothing. Thread-local, because the thread
        // undoing an attempt is the thread that made it; a release on any other
        // thread still advances past this attempt's snapshot.
        simpler::dfx::credit::DiagnosticCreditSignal::SuppressSelfNotify quiet;
        restored = withdraw_unlaunched_collectors_for_run(dfx, run_epoch);
    }
    // An owner that could not give its slot back restored nothing: it
    // quarantined this run or is fatal, and the capacity every waiter last
    // observed is gone for good. That is a real event — the opposite of the
    // absence of one — so it is published here, outside the suppression, and a
    // waiter learns it by re-attempting rather than by spending its whole
    // deadline on a wake nobody will raise.
    if (!restored) credit_signal_.advance();
}

bool SimDeviceRunnerBase::withdraw_unlaunched_collectors_for_run(const DfxRunConfig &dfx, uint64_t run_epoch) noexcept {
    // Same rule as the onboard base: a launch that submitted nothing leaves an
    // admitted, targetless slot that neither the writer nor a flush can see,
    // and `kMaxOpenEpochs` of them exhaust the capacity the next admission
    // waits on. Nothing may escape, for the reason the onboard base gives.
    //
    // The result is each owner's answer to "is this slot back", conjoined: an
    // owner that never admitted this run answers yes, and one that quarantined
    // it or is fatal answers no. The caller needs the distinction because only
    // the first is a restoration of what every waiter already saw.
    bool restored = true;
    if (dfx.pmu_enabled && pmu_collector_.retains_runs()) {
        // Nothing may escape a rollback path, for the reason given below. A
        // throw is not a proof of release, so it settles the way a refusal does.
        try {
            restored = pmu_collector_.abandon_run(run_epoch) && restored;
        } catch (...) {
            restored = false;
        }
    }
    if (dfx.dump_args_enabled() && dump_collector_.retains_runs()) {
        try {
            restored = dump_collector_.abandon_run(run_epoch) && restored;
        } catch (...) {
            restored = false;
        }
    }
    if (dfx.scope_stats_enabled && scope_stats_collector_.retains_runs()) {
        // Void by contract: this owner holds its own receipts and frees the
        // slot under its own lock, so it has no acknowledgement to fail and no
        // quarantine to report from here.
        try {
            scope_stats_collector_.abandon_run(run_epoch);
        } catch (...) {
            restored = false;
        }
    }
    // DepGen's admission lives on the arch runner, so it is given back through
    // the hook. This sits with the other per-collector withdrawals and ahead of
    // the swimlane gates below: those return early, and a run with dep_gen on
    // and swimlane off would otherwise keep an export slot no writer and no
    // flush can see — two of them exhaust admission for the runner's life.
    if (dfx.dep_gen_enabled && dep_gen_retains_runs()) {
        withdraw_dep_gen_run(run_epoch);
    }
    if (!dfx.chip_swimlane_enabled()) return restored;
    try {
        if (chip_swimlane_collector_.retains_runs()) {
            restored = chip_swimlane_collector_.abandon_run(run_epoch) && restored;
        } else {
            // A terminal bank is taken per run whether or not this collector
            // retains output, so its credit is owed back on the same terms.
            // Retention decides only whether there is also a retained admission
            // to withdraw; an ordinary-output run that stopped at NotStarted
            // submitted nothing, so no producer can hold its bank and the
            // release is proved rather than assumed.
            chip_swimlane_collector_.release_run_terminal_bank(run_epoch);
        }
    } catch (...) {
        restored = false;
    }
    return restored;
}

int SimDeviceRunnerBase::flush_diagnostics(int timeout_ms, std::string *error) {
    // Both retaining collectors, inside the one call and the one budget: each
    // is serviced with the time left rather than re-acquiring the caller's
    // whole timeout, so the bound never doubles. Both are always attempted —
    // the first one's failure must not hide the second's — and the call fails
    // if either does, naming each.
    //
    // Deliberately not gated on `retains_runs()`: PMU keeps a sticky error
    // record for the runner's whole life, so a failure recorded before a
    // collector rebuild turned retention off must still be reported here.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    auto remaining_ms = [&deadline]() -> int {
        const auto left =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
        if (left <= 0) return 0;
        return static_cast<int>(left);
    };
    std::string swimlane_error;
    std::string pmu_error;
    std::string dump_error;
    std::string scope_stats_error;
    std::string dep_gen_error;
    std::string host_graph_error;
    bool ok = true;
    // Gated on whether this collector has ever retained, not on whether it
    // retains now — the same boundary the onboard runner uses, and the same
    // reason PMU's arm below is ungated. A failure recorded under retention has
    // to survive the rebuild or reconfiguration that turned retention off, and
    // a collector that never retained has nothing here its own runs did not
    // already report.
    if (chip_swimlane_collector_.ever_retained() &&
        !chip_swimlane_collector_.flush_retained_runs(remaining_ms(), &swimlane_error)) {
        ok = false;
    }
    if (!dump_collector_.flush_retained_runs(remaining_ms(), &dump_error)) {
        ok = false;
    }
    if (!pmu_collector_.flush_retained_runs(remaining_ms(), &pmu_error)) {
        ok = false;
    }
    // Deliberately not gated on `retains_runs()`, for the same reason PMU's arm
    // is not: a sticky failure recorded before a collector rebuild reconfigured
    // retention off must still be reported here. With retention never
    // configured the call has nothing to wait for and nothing to report.
    if (!scope_stats_collector_.flush_retained_runs(remaining_ms(), &scope_stats_error)) {
        ok = false;
    }
    // DepGen lives on the arch runner, so it is reached through the hook. The
    // arm is deliberately ungated, for the same reason PMU's is: a failure
    // recorded before a collector rebuild reconfigured retention off must
    // still be reported here, and a collector that retains nothing has
    // nothing to wait for and nothing to report.
    if (!dep_gen_flush_retained(remaining_ms(), &dep_gen_error)) {
        ok = false;
    }
    // The host-built graph's own owner. Ungated for the same reason as the arms
    // above: a failure recorded before retention was reconfigured off must still
    // be reported, and an exporter holding nothing has nothing to wait for.
    if (!host_graph_exporter_.flush_retained_runs(remaining_ms(), &host_graph_error)) {
        ok = false;
    }
    if (ok) return 0;
    if (error != nullptr) {
        *error = swimlane_error;
        for (const std::string &part : {dump_error, pmu_error, scope_stats_error, dep_gen_error, host_graph_error}) {
            if (part.empty()) continue;
            if (!error->empty()) *error += "; ";
            *error += part;
        }
    }
    return PTO_RUNTIME_ERR_INTERNAL;
}

void SimDeviceRunnerBase::finish_retained_runs() {
    chip_swimlane_collector_.finish_retained_runs();
    dump_collector_.finish_retained_runs();
    pmu_collector_.finish_retained_runs();
    scope_stats_collector_.finish_retained_runs();
    dep_gen_finish_retained();
    host_graph_exporter_.finish_retained_runs();
}

bool SimDeviceRunnerBase::seal_host_dep_gen_graph(const DfxRunConfig &dfx, uint64_t run_epoch) noexcept {
    if (!dfx.dep_gen_enabled || !host_graph_exporter_.retains_runs()) return true;
    // The hand-off is passed in rather than called: this base names no runtime
    // symbol, and the weak `take` a device-capturing runtime links reports that
    // there is nothing here to publish.
    return host_graph_exporter_.seal(run_epoch, dfx.output_prefix, &dep_gen_host_graph_take);
}

void SimDeviceRunnerBase::write_host_phase_records_artifact(const std::string &output_prefix, uint32_t pipeline_slot) {
    if (pipeline_slot >= host_phase_runs_.size()) return;
    simpler::dfx::HostPhaseRecordStore &records = host_phase_runs_[pipeline_slot].records;
    // Every phase this records is produced on the host during bind and the store
    // is finished before launch, so it touches no device state and is callable
    // from any point after bind — including a path that never launched. The run's
    // output prefix is non-empty exactly when it produces diagnostic artifacts,
    // and the store writes a pass at most once.
    if (!output_prefix.empty() && records.finished()) {
        (void)records.write_records_jsonl(make_host_phase_records_path(output_prefix));
    }
}

void SimDeviceRunnerBase::close_pmu_run_boundary(
    const DfxRunConfig &dfx, uint64_t run_epoch, bool device_execution_complete, bool device_producers_stopped
) {
    if (!pmu_collector_.retains_runs()) {
        pmu_collector_.quiesce();
        pmu_collector_.reconcile_counters();
        return;
    }
    // A retained run keeps only the run boundary's device-side reads — the
    // per-core counters and the buffers the cores still hold — and hands the
    // rest to the writer. No quiesce: the pipeline is shared with the
    // successor, and draining it here is what the per-queue cut replaces.
    //
    // `device_execution_complete` is what the caller observed of this run's
    // fence, and it is the whole of PMU's "this run finished" proof: the
    // recovery path clears it, and a producer may then never have reached its
    // own close.
    (void)dfx;
    pmu_collector_.run_close(run_epoch, device_execution_complete);
    simpler::dfx::runs::settle_failed_run_terminal_bank(
        pmu_collector_, run_epoch, device_execution_complete, device_producers_stopped
    );
}

int SimDeviceRunnerBase::close_args_dump_run_boundary(
    const DfxRunConfig &dfx, uint64_t run_epoch, bool device_execution_complete, bool device_producers_stopped
) {
    if (!dump_collector_.retains_runs()) {
        dump_collector_.quiesce();
        dump_collector_.reconcile_counters();
        dump_collector_.export_dump_files();
        return 0;
    }
    // A retained run keeps only the run boundary's device-side reads — this
    // run's terminal lane state and, for a lane whose handover its receipt
    // ledger can decide, an unpublished buffer's records and payload — and
    // hands the rest to the writer. No quiesce: the pipeline is shared with the
    // successor, and draining it here is what the per-queue cut replaces.
    (void)dfx;
    // Non-zero when the close could not prove this run's published buffers were
    // processed into host-owned storage inside its execution claim. That is an
    // ownership failure, not a file annotation, so it travels out of here.
    const int rc = dump_collector_.run_close(run_epoch, device_execution_complete);
    simpler::dfx::runs::settle_failed_run_terminal_bank(
        dump_collector_, run_epoch, device_execution_complete, device_producers_stopped
    );
    return rc;
}

void SimDeviceRunnerBase::close_scope_stats_run_boundary(
    const DfxRunConfig &dfx, uint64_t run_epoch, bool device_execution_complete, bool device_producers_stopped
) {
    if (!scope_stats_collector_.retains_runs()) {
        scope_stats_collector_.quiesce();
        scope_stats_collector_.reconcile_counters();
        scope_stats_collector_.write_jsonl(dfx.output_prefix);
        return;
    }
    // A retained run keeps the boundary's ownership steps — the receive drain
    // and the terminal read — and hands only the rendering and the file write
    // to the writer. `device_execution_complete` is what the caller observed of
    // this run's fence and is the whole of this collector's completion proof:
    // without it nothing shared is read and the run produces no artifact.
    (void)dfx;
    scope_stats_collector_.run_close(run_epoch, device_execution_complete);
    simpler::dfx::runs::settle_failed_run_terminal_bank(
        scope_stats_collector_, run_epoch, device_execution_complete, device_producers_stopped
    );
}

int SimDeviceRunnerBase::teardown_shared_collectors_after_run(
    const DfxRunConfig &dfx, uint32_t pipeline_slot, uint64_t run_epoch, bool device_execution_complete,
    bool device_producers_stopped
) {
    // The order is fixed by two couplings, not by preference: the host phase
    // records reach the collector before the swimlane export serializes them,
    // and each collector drains before it reconciles before it exports.
    // Diagnostic exports use the per-task output prefix the user set on
    // CallConfig (CallConfig::validate() enforces non-empty upstream).
    //
    // Only ArgsDump's close reports anything here: a retained run whose
    // published buffers it could not prove processed inside the execution claim
    // is an ownership failure the caller has to see, because releasing the
    // claim past one would let the successor reuse an arena whose bytes nobody
    // copied. Every other collector's close is void, as it was.
    int args_dump_rc = 0;
    if (dfx.chip_swimlane_enabled() && chip_swimlane_collector_.retains_runs()) {
        // Both publications precede the epoch's metadata snapshot, which is
        // what copies them. The extensions belong here for the same reason the
        // host phase records do: a5's host_build_graph supplies a real
        // publisher for them at every enabled level, and the collector holds
        // one copy of the sections for every run it serves.
        // No bank index is passed: a bank is held by the epoch that acquired it,
        // so the close resolves its own and settles it there. See the onboard
        // base for the two proofs that let one be handed on, and
        // `device_producers_stopped` for the third, which only a simulated
        // device can supply.
        simpler::dfx::runs::close_run_boundary(
            chip_swimlane_collector_, run_epoch, device_execution_complete, device_producers_stopped,
            [this, pipeline_slot] {
                publish_host_phase_records_to_swimlane(pipeline_slot);
                publish_chip_swimlane_runtime_extensions();
            }
        );
        write_host_phase_records_artifact(dfx.output_prefix, pipeline_slot);
        if (dfx.dump_args_enabled()) {
            args_dump_rc =
                close_args_dump_run_boundary(dfx, run_epoch, device_execution_complete, device_producers_stopped);
        }
        if (dfx.pmu_enabled) {
            close_pmu_run_boundary(dfx, run_epoch, device_execution_complete, device_producers_stopped);
        }
        if (dfx.scope_stats_enabled) {
            close_scope_stats_run_boundary(dfx, run_epoch, device_execution_complete, device_producers_stopped);
        }
        return args_dump_rc;
    }
    if (dfx.chip_swimlane_enabled()) {
        chip_swimlane_collector_.quiesce();
        chip_swimlane_collector_.read_phase_header_metadata();
        chip_swimlane_collector_.reconcile_counters();
        // Only the completion path reads the bank; see the onboard base for why
        // an incomplete run's snapshot says nothing about that run. Whether the
        // bank itself comes back is a separate question, and on a simulated
        // device it has an answer: the producers are this process's AICPU and
        // AICore threads, and the drain joined every one of them before it got
        // here.
        const int bank = chip_swimlane_collector_.run_terminal_bank_of(run_epoch);
        if (device_execution_complete) {
            if (bank >= 0) {
                chip_swimlane_collector_.report_run_terminal_snapshot(static_cast<uint32_t>(bank), run_epoch);
                chip_swimlane_collector_.release_run_terminal_bank(run_epoch);
            }
        } else if (bank >= 0) {
            if (device_producers_stopped) {
                // Nothing reads what this run left there — `quiesce` above
                // stopped the host's readers, and the entries carry their own
                // epoch, so a survivor is counted as foreign by the next
                // occupant rather than attributed to it.
                chip_swimlane_collector_.discard_run_terminal_bank(run_epoch);
            } else {
                chip_swimlane_collector_.quarantine_run_terminal_bank(run_epoch);
            }
        }
        publish_host_phase_records_to_swimlane(pipeline_slot);
        publish_chip_swimlane_runtime_extensions();
        chip_swimlane_collector_.export_swimlane_json();
    }

    write_host_phase_records_artifact(dfx.output_prefix, pipeline_slot);

    if (dfx.dump_args_enabled()) {
        args_dump_rc =
            close_args_dump_run_boundary(dfx, run_epoch, device_execution_complete, device_producers_stopped);
    }

    if (dfx.pmu_enabled) {
        close_pmu_run_boundary(dfx, run_epoch, device_execution_complete, device_producers_stopped);
    }

    if (dfx.scope_stats_enabled) {
        close_scope_stats_run_boundary(dfx, run_epoch, device_execution_complete, device_producers_stopped);
    }
    return args_dump_rc;
}

// Whether this runtime's device scheduler dispatches from resolved kernel-entry
// addresses rather than resolving each entry out of the CoreCallable object it
// is handed. Defined by every runtime's runtime_maker.cpp, so adding a runtime
// cannot leave the answer implicit, and true only where a device consumer reads
// the entry view.
extern "C" bool runtime_uses_callable_entry_table_impl();

uint64_t SimDeviceRunnerBase::upload_chip_callable_buffer(const ChipCallable *callable) {
    if (callable == nullptr) {
        return 0;
    }

    const ChipCallableLayout layout = compute_chip_callable_layout(callable);

    auto it = chip_callable_buffers_.find(layout.content_hash);
    if (it != chip_callable_buffers_.end()) {
        it->second.refcount++;
        LOG_DEBUG(
            "Chip callable dedup hit (sim): chip_dev=0x%lx, size=%zu, hash=0x%lx, refcount=%d", it->second.chip_dev,
            it->second.total_size, layout.content_hash, it->second.refcount
        );
        return it->second.chip_dev;
    }

    // Sized before anything is allocated, so an unaddressable func_id refuses
    // the registration rather than leaving a scratch behind.
    uint32_t table_len = 0;
    int32_t bad_func_id = 0;
    if (!chip_callable_table_length(callable, CHIP_MAX_FUNC_ID, &table_len, &bad_func_id)) {
        LOG_ERROR("Chip callable declares func_id=%d outside [0, %d)", bad_func_id, CHIP_MAX_FUNC_ID);
        return 0;
    }
    const bool want_entry_table = table_len != 0 && runtime_uses_callable_entry_table_impl();

    // Every allocation this upload makes is taken before the scratch, so the
    // only raw resource it owns is created with nothing fallible left between
    // it and the guard that owns it. Filling these below allocates nothing.
    std::vector<uint64_t> object_table(table_len, 0);
    std::vector<uint64_t> entry_table;
    if (want_entry_table) entry_table.assign(table_len, 0);
    std::vector<void *> dlopen_handles;
    dlopen_handles.reserve(callable->child_count());
    bool published = false;

    // Allocate host scratch (host == device in sim). Plain new[] keeps
    // ChipCallableBuffer::host_scratch ownership symmetric with finalize().
    auto *scratch = new uint8_t[layout.total_size];

    // The guard owns the scratch and every handle until the map entry does,
    // and it takes that ownership on the statement after the allocation:
    // `RAIIScopeGuard` stores the by-reference lambda in place, so nothing
    // between the two can throw. It decides by `published` rather than by
    // being dismissed, because the handles move into the published entry only
    // after the insertion returns and a guard consulting a moved-from vector
    // could not close them. Every early return and every throw before that
    // point unwinds through here.
    auto cleanup = RAIIScopeGuard([&]() {
        if (published) return;
        for (void *h : dlopen_handles)
            dlclose(h);
        delete[] scratch;
    });
    std::memcpy(scratch, callable, layout.total_size);

    // Per-child dlopen + dlsym kernel_entry + register pto-sim hooks, then
    // patch the child's resolved_addr_ to the function pointer.
    for (int32_t i = 0; i < callable->child_count(); ++i) {
        const uint32_t off = callable->child_offset(i);
        auto *child_in_scratch = reinterpret_cast<CoreCallable *>(scratch + layout.header_size + off);
        const void *kernel_binary = child_in_scratch->binary_data();
        size_t kernel_size = static_cast<size_t>(child_in_scratch->binary_size());

        std::string tmpfile;
        if (!simpler::common::sim_host::create_temp_so_file(
                "/tmp/kernel_" + std::to_string(callable->child_func_id(i)) + "_XXXXXX",
                reinterpret_cast<const uint8_t *>(kernel_binary), kernel_size, &tmpfile
            )) {
            LOG_ERROR("Failed to create temp file for child kernel #%d", i);
            return 0;
        }

        void *handle = dlopen(tmpfile.c_str(), RTLD_NOW | RTLD_LOCAL);
        std::remove(tmpfile.c_str());
        if (!handle) {
            LOG_ERROR("dlopen failed for child kernel #%d: %s", i, dlerror());
            return 0;
        }
        dlopen_handles.push_back(handle);

        void *func = dlsym(handle, "kernel_entry");
        if (!func) {
            LOG_ERROR("dlsym failed for child kernel #%d 'kernel_entry': %s", i, dlerror());
            return 0;
        }

        auto register_hooks = reinterpret_cast<void (*)(void *, void *)>(dlsym(handle, "pto_sim_register_hooks"));
        if (register_hooks != nullptr) {
            register_hooks(
                reinterpret_cast<void *>(pto_sim_get_subblock_id),
                reinterpret_cast<void *>(pto_sim_get_pipe_shared_state)
            );
        }

        child_in_scratch->set_resolved_addr(reinterpret_cast<uint64_t>(func));
    }

    const uint64_t chip_dev = reinterpret_cast<uint64_t>(scratch);

    // The tables are filled from the same scratch the dlopen loop just patched,
    // by the same helper the onboard path uses: sim's own host addresses stand
    // in for the device base, so the object view addresses each CoreCallable
    // inside the scratch and the entry view carries the host function pointer
    // that child's resolved_addr_ now holds.
    chip_callable_fill_tables(
        callable, layout, scratch, chip_dev, table_len, object_table.data(),
        want_entry_table ? entry_table.data() : nullptr
    );

    // One publication point. The entry is built with no handles and inserted
    // first: the insertion is the last thing that can fail, and until it
    // returns the guard above is still the only owner of the scratch and the
    // handles. The moves that follow are `std::vector`'s noexcept move
    // assignment, so the entry cannot end up published and incomplete.
    ChipCallableBuffer retained{chip_dev, scratch, layout.total_size, 1, {}};
    retained.table_len = table_len;
    if (table_len != 0) {
        retained.object_table_dev = reinterpret_cast<uint64_t>(object_table.data());
        if (want_entry_table) retained.entry_table_dev = reinterpret_cast<uint64_t>(entry_table.data());
    }
    auto inserted = chip_callable_buffers_.emplace(layout.content_hash, std::move(retained));
    ChipCallableBuffer &owner = inserted.first->second;
    owner.object_table = std::move(object_table);
    owner.entry_table = std::move(entry_table);
    owner.dlopen_handles = std::move(dlopen_handles);
    published = true;
    LOG_DEBUG(
        "Uploaded chip callable (sim): chip_dev=0x%lx, size=%zu, child_count=%d, hash=0x%lx", chip_dev,
        layout.total_size, callable->child_count(), layout.content_hash
    );
    return chip_dev;
}

int SimDeviceRunnerBase::release_chip_callable_buffer(uint64_t hash) {
    if (hash == 0) {
        return 0;
    }
    auto it = chip_callable_buffers_.find(hash);
    if (it == chip_callable_buffers_.end()) {
        LOG_WARN("release_chip_callable_buffer: hash=0x%lx not found", hash);
        return 0;
    }
    if (--it->second.refcount <= 0) {
        for (void *h : it->second.dlopen_handles) {
            if (h != nullptr) dlclose(h);
        }
        delete[] it->second.host_scratch;
        LOG_DEBUG(
            "Freed chip callable buffer (sim): chip_dev=0x%lx, size=%zu, hash=0x%lx", it->second.chip_dev,
            it->second.total_size, hash
        );
        chip_callable_buffers_.erase(it);
    }
    return 0;
}

void SimDeviceRunnerBase::print_handshake_results() {
    if (worker_count_ == 0 || last_runtime_ == nullptr) {
        return;
    }

    LOG_DEBUG("Handshake results for %d cores:", worker_count_);
    Handshake *workers = last_runtime_->get_workers();
    for (int i = 0; i < worker_count_; i++) {
        LOG_DEBUG(
            "  Core %d: aicore_done=%d aicpu_ready=%d task=%d", i, workers[i].aicore_done, workers[i].aicpu_ready,
            workers[i].task
        );
    }
}

void SimDeviceRunnerBase::release_callable_state() {
    // Release any chip callable buffers callers forgot to unregister.
    for (auto &kv : chip_callable_buffers_) {
        for (void *h : kv.second.dlopen_handles) {
            if (h != nullptr) dlclose(h);
        }
        delete[] kv.second.host_scratch;
        LOG_DEBUG(
            "Freed chip callable buffer (sim): chip_dev=0x%lx, size=%zu, hash=0x%lx", kv.second.chip_dev,
            kv.second.total_size, kv.first
        );
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
}

bool SimDeviceRunnerBase::every_enabled_owner_can_admit(const DfxRunConfig &dfx, uint64_t run_epoch) const {
    // The onboard base's reasoning, on the sim runner's own members: an owner
    // with no mapped bank cannot tell two open runs apart whatever its
    // capacity, and an owner with none free declines this round only.
    if (dfx.chip_swimlane_enabled()) {
        if (!chip_swimlane_collector_.can_admit_retained_run()) return false;
        if (!chip_swimlane_collector_.can_take_run_terminal_bank(run_epoch)) return false;
    }
    if (dfx.dump_args_enabled()) {
        if (!dump_collector_.device_publishes_run_terminal()) return false;
        if (!dump_collector_.can_admit_retained_run()) return false;
        if (!dump_collector_.can_take_run_terminal_bank(run_epoch)) return false;
    }
    if (dfx.pmu_enabled) {
        if (!pmu_collector_.device_publishes_run_terminal()) return false;
        if (!pmu_collector_.can_admit_retained_run()) return false;
        if (!pmu_collector_.can_take_run_terminal_bank(run_epoch)) return false;
    }
    // dep_gen's device side exists only where the graph is built on the device,
    // and the collector itself lives on the arch runner — so both the
    // host-orch question and the capacity one are answered through the hook.
    if (dfx.dep_gen_enabled && !dep_gen_can_admit_joined_run(run_epoch)) return false;
    if (dfx.scope_stats_enabled) {
        if (!scope_stats_collector_.device_publishes_run_terminal()) return false;
        if (!scope_stats_collector_.can_admit_retained_run()) return false;
        if (!scope_stats_collector_.can_take_run_terminal_bank(run_epoch)) return false;
    }
    return true;
}

void SimDeviceRunnerBase::resolve_run_terminal_banks(
    const DfxRunConfig &dfx, uint64_t run_epoch, KernelArgs &kernel_args
) const {
    // Written unconditionally, including the zeroes: the fields mean "this
    // run's bank", and argument storage is reused across runs, so a field left
    // alone would hand a successor its predecessor's address.
    kernel_args.args_dump_run_terminal_bank =
        dfx.dump_args_enabled() ? dump_collector_.run_terminal_bank_address(run_epoch) : 0;
    kernel_args.pmu_run_terminal_bank = dfx.pmu_enabled ? pmu_collector_.run_terminal_bank_address(run_epoch) : 0;
    kernel_args.dep_gen_run_terminal_bank = dfx.dep_gen_enabled ? dep_gen_run_terminal_bank(run_epoch) : 0;
    kernel_args.scope_stats_run_terminal_bank =
        dfx.scope_stats_enabled ? scope_stats_collector_.run_terminal_bank_address(run_epoch) : 0;
}

void SimDeviceRunnerBase::install_credit_release_notifiers() {
    // Exactly once per runner, and before any collector thread exists: the
    // first call runs during the first run's arming, which precedes every
    // `collector.start(thread_factory)`. A second assignment would write a
    // `std::function` a background writer may be calling at that moment —
    // releases run outside the collector locks — and assigning the same
    // lambda is no defence, because the race is on the object and not on its
    // value. Installed once, the callbacks are immutable for the runner's
    // life, which is what makes them safe to call from any writer.
    if (credit_notifiers_installed_) return;
    credit_notifiers_installed_ = true;
    // Every collector, not only the ones this run enabled: a collector whose
    // owner is off raises nothing anyway, and one enabled by a later run must
    // already be wired when its first writer publishes.
    auto notify = [this] {
        credit_signal_.advance();
    };
    chip_swimlane_collector_.set_credit_release_notifier(notify);
    dump_collector_.set_credit_release_notifier(notify);
    pmu_collector_.set_credit_release_notifier(notify);
    scope_stats_collector_.set_credit_release_notifier(notify);
    install_dep_gen_credit_release_notifier(notify);
}

int SimDeviceRunnerBase::open_run_terminal_banks_for_enabled_owners(const DfxRunConfig &dfx) {
    install_credit_release_notifiers();
    // Terminal storage must be initialized before admission. An enabled
    // retained owner cannot silently fall back to shared live counters.
    auto report = [](const char *owner, bool opened) -> int {
        if (opened) return 0;
        LOG_ERROR("%s: per-run terminal backing could not be initialized", owner);
        return PTO_RUNTIME_ERR_INTERNAL;
    };
    if (dfx.dump_args_enabled() && dump_collector_.retains_runs() && !dump_collector_.device_publishes_run_terminal()) {
        if (int rc =
                report("ArgsDump", dump_collector_.open_run_terminal_banks(dump_collector_.terminal_producer_count()));
            rc != 0)
            return rc;
    }
    if (dfx.pmu_enabled && pmu_collector_.retains_runs() && !pmu_collector_.device_publishes_run_terminal()) {
        if (int rc = report(
                "PmuCollector", pmu_collector_.open_run_terminal_banks(pmu_collector_.terminal_producer_count())
            );
            rc != 0)
            return rc;
    }
    if (dfx.scope_stats_enabled && scope_stats_collector_.retains_runs() &&
        !scope_stats_collector_.device_publishes_run_terminal()) {
        if (int rc =
                report("ScopeStats", scope_stats_collector_.open_run_terminal_banks(SCOPE_STATS_TERMINAL_PRODUCERS));
            rc != 0)
            return rc;
    }
    return open_dep_gen_run_terminal_banks(dfx);
}

bool SimDeviceRunnerBase::retained_consumers_outstanding() const {
    // An unpublished slot on any owner means a writer may still read that
    // owner's pools. Asked of every collector rather than only the ones this
    // run enabled: a rebuild frees what a *previous* run's owner allocated,
    // and this run's configuration says nothing about that.
    if (!chip_swimlane_collector_.retained_runs_idle()) return true;
    if (!dump_collector_.retained_runs_idle()) return true;
    if (!pmu_collector_.retained_runs_idle()) return true;
    if (!scope_stats_collector_.retained_runs_idle()) return true;
    return dep_gen_retained_runs_outstanding();
}
