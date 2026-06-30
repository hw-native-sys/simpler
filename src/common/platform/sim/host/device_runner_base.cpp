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

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>

#include "callable.h"
#include "callable_protocol.h"
#include "chip_callable_layout.h"
#include "cpu_sim_context.h"
#include "host/raii_scope_guard.h"
#include "utils/elf_build_id.h"

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

int SimDeviceRunnerBase::setup_static_arena(size_t gm_heap_size, size_t gm_sm_size, size_t runtime_arena_size) {
    // Three independent device_malloc'd buffers: GM heap, PTO2 SM, prebuilt
    // runtime arena. Split out from a single large allocation because the
    // combined size can exceed the device allocator's largest contiguous
    // block. Each arena commits exactly one region, so its base() is the
    // pooled pointer the caller wants.
    //
    // Idempotent for the production case (sizes do not change across a
    // worker's lifetime). If a caller asks for a larger layout on any
    // region, redo just that region.
    auto commit_region = [](DeviceArena &arena, size_t &cached_size, size_t requested_size) -> int {
        if (requested_size == 0) {
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
            arena.release();
            return -1;
        }
        cached_size = requested_size;
        return 0;
    };
    // Failure of any region releases all peers — mirrors the onboard "rollback
    // all on any failure" semantic (PR #922). Pooled pointers from a prior
    // successful call stay valid; a failed resize attempt does not leave a
    // partial layout behind.
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

void *SimDeviceRunnerBase::acquire_pooled_gm_heap() {
    if (!gm_heap_arena_.is_committed()) return nullptr;
    return gm_heap_arena_.base();
}

void *SimDeviceRunnerBase::acquire_pooled_gm_sm() {
    if (!gm_sm_arena_.is_committed()) return nullptr;
    return gm_sm_arena_.base();
}

void *SimDeviceRunnerBase::acquire_pooled_runtime_arena() {
    if (!runtime_arena_pool_.is_committed()) return nullptr;
    return runtime_arena_pool_.base();
}

std::thread SimDeviceRunnerBase::create_thread(std::function<void()> fn) {
    int dev_id = device_id_;
    return std::thread([dev_id, fn = std::move(fn)]() {
        pto_cpu_sim_bind_device(dev_id);
        fn();
        pto_cpu_sim_bind_device(-1);
    });
}

int SimDeviceRunnerBase::attach_current_thread(int device_id) {
    if (device_id < 0) {
        LOG_ERROR("Invalid device_id: %d", device_id);
        return -1;
    }
    if (device_id_ != -1 && device_id_ != device_id) {
        LOG_ERROR(
            "DeviceRunner already initialized on device %d; finalize before switching to device %d", device_id_,
            device_id
        );
        return -1;
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

void *SimDeviceRunnerBase::allocate_tensor(size_t bytes) { return mem_alloc_.alloc(bytes); }

void SimDeviceRunnerBase::free_tensor(void *dev_ptr) {
    if (dev_ptr != nullptr) {
        mem_alloc_.free(dev_ptr);
    }
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

int SimDeviceRunnerBase::l3_l2_orch_comm_init(void *control_block, size_t control_block_size) {
    return l3_l2_orch_comm_service_.start(this, control_block, control_block_size);
}

int SimDeviceRunnerBase::l3_l2_orch_comm_shutdown() { return l3_l2_orch_comm_service_.stop(); }

void *SimDeviceRunnerBase::l3_l2_allocate_region_bytes(uint64_t bytes) {
    if (bytes == 0 || bytes > std::numeric_limits<size_t>::max()) {
        return nullptr;
    }
    void *ptr = nullptr;
    if (posix_memalign(&ptr, L3L2_ORCH_COMM_COUNTER_BASE_ALIGNMENT, static_cast<size_t>(bytes)) != 0) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lk(l3_l2_alloc_mu_);
    l3_l2_allocations_.insert(ptr);
    return ptr;
}

void SimDeviceRunnerBase::l3_l2_free_region_bytes(void *ptr) {
    if (ptr == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lk(l3_l2_alloc_mu_);
    auto it = l3_l2_allocations_.find(ptr);
    if (it == l3_l2_allocations_.end()) {
        return;
    }
    std::free(ptr);
    l3_l2_allocations_.erase(it);
}

int SimDeviceRunnerBase::l3_l2_copy_to_device(void *dev_ptr, const void *host_ptr, uint64_t bytes) {
    if (bytes > std::numeric_limits<size_t>::max()) {
        return -1;
    }
    return copy_to_device(dev_ptr, host_ptr, static_cast<size_t>(bytes));
}

int SimDeviceRunnerBase::l3_l2_copy_from_device(void *host_ptr, const void *dev_ptr, uint64_t bytes) {
    if (bytes > std::numeric_limits<size_t>::max()) {
        return -1;
    }
    return copy_from_device(host_ptr, dev_ptr, static_cast<size_t>(bytes));
}

std::thread SimDeviceRunnerBase::l3_l2_create_service_thread(std::function<void()> fn) {
    return create_thread(std::move(fn));
}

int SimDeviceRunnerBase::stamp_orch_so(Runtime &runtime, int32_t cid, bool force_reload) {
    if (cid < 0) {
        LOG_ERROR("stamp_orch_so: invalid callable_id=%d", cid);
        return -1;
    }
    auto it = callables_.find(cid);
    if (it == callables_.end()) {
        LOG_ERROR("stamp_orch_so: callable_id=%d not registered", cid);
        return -1;
    }
    const auto &state = it->second;
    // hbg: orch SO never crosses host/device — clear device-orch metadata
    // and skip AICPU bookkeeping.
    if (state.host_dlopen_handle != nullptr) {
        runtime.set_dev_orch_so(0, 0);
        runtime.set_active_callable_id(cid, /*is_new=*/false);
        return 0;
    }
    const bool needs_load = force_reload || (aicpu_seen_callable_ids_.count(cid) == 0);
    runtime.set_dev_orch_so(state.dev_orch_so_addr, state.dev_orch_so_size);
    runtime.set_device_orch_func_name(state.func_name.c_str());
    runtime.set_device_orch_config_name(state.config_name.c_str());
    runtime.set_active_callable_id(cid, needs_load);
    LOG_INFO_V0(
        "Orch SO stamped cid=%d hash=0x%lx %zu bytes (needs_load=%d)", cid, state.hash, state.dev_orch_so_size,
        needs_load ? 1 : 0
    );
    return 0;
}

int SimDeviceRunnerBase::prepare_orch_so(Runtime &runtime) {
    const int32_t cid = runtime.get_active_callable_id();
    if (cid < 0) {
        LOG_ERROR("prepare_orch_so: no active callable_id; prepared-callable flow required");
        return -1;
    }
    return stamp_orch_so(runtime, cid, /*force_reload=*/false);
}

int SimDeviceRunnerBase::commit_device_register(int32_t cid) {
    auto it = callables_.find(cid);
    if (it == callables_.end()) {
        LOG_ERROR("commit_device_register: callable_id=%d not registered", cid);
        return -1;
    }
    if (it->second.host_dlopen_handle != nullptr) {
        return 0;
    }
    const bool inserted = aicpu_seen_callable_ids_.insert(cid).second;
    if (inserted) {
        ++aicpu_dlopen_total_;
        LOG_INFO_V0("AICPU callable load committed cid=%d (count=%zu)", cid, aicpu_dlopen_total_);
    }
    return 0;
}

int SimDeviceRunnerBase::launch_device_register(int32_t callable_id) {
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

    Runtime runtime;
    rc = stamp_orch_so(runtime, callable_id, /*force_reload=*/true);
    if (rc != 0) return rc;

    rc = invoke_device_register(runtime);
    if (rc != 0) {
        LOG_ERROR("launch_device_register: invoke_device_register failed: %d", rc);
        return rc;
    }

    return commit_device_register(callable_id);
}

int SimDeviceRunnerBase::record_device_orch_callable(
    int32_t callable_id, const void *orch_so_data, size_t orch_so_size, const char *func_name, const char *config_name,
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
    if (callables_.count(callable_id) != 0) {
        LOG_ERROR("record_device_orch_callable: callable_id=%d already registered", callable_id);
        return -1;
    }

    const uint64_t hash = simpler::common::utils::elf_build_id_64(orch_so_data, orch_so_size);

    auto buf_it = orch_so_dedup_.find(hash);
    uint64_t dev_addr = 0;
    if (buf_it == orch_so_dedup_.end()) {
        void *buf = mem_alloc_.alloc(orch_so_size);
        if (buf == nullptr) {
            LOG_ERROR("record_device_orch_callable: alloc %zu bytes failed", orch_so_size);
            return -1;
        }
        // Sim shares an address space with the simulated AICPU thread, so a
        // plain memcpy is the moral equivalent of rtMemcpy on hardware.
        std::memcpy(buf, orch_so_data, orch_so_size);
        OrchSoBuffer entry;
        entry.dev_addr = buf;
        entry.capacity = orch_so_size;
        entry.refcount = 1;
        orch_so_dedup_.emplace(hash, entry);
        dev_addr = reinterpret_cast<uint64_t>(buf);
        LOG_INFO_V0("record_device_orch_callable: hash=0x%lx new buffer %zu bytes", hash, orch_so_size);
    } else {
        buf_it->second.refcount++;
        dev_addr = reinterpret_cast<uint64_t>(buf_it->second.dev_addr);
        LOG_INFO_V0(
            "record_device_orch_callable: hash=0x%lx shared buffer (refcount=%d)", hash, buf_it->second.refcount
        );
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

int SimDeviceRunnerBase::record_host_orch_callable(
    int32_t callable_id, void *host_dlopen_handle, void *host_orch_func_ptr,
    std::vector<std::pair<int, uint64_t>> kernel_addrs, std::vector<ArgDirection> signature
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
    if (callables_.count(callable_id) != 0) {
        LOG_ERROR("record_host_orch_callable: callable_id=%d already registered", callable_id);
        return -1;
    }

    CallableState state;
    state.host_dlopen_handle = host_dlopen_handle;
    state.host_orch_func_ptr = host_orch_func_ptr;
    state.kernel_addrs = std::move(kernel_addrs);
    state.signature = std::move(signature);
    callables_.emplace(callable_id, std::move(state));
    ++host_dlopen_total_;
    LOG_INFO_V0("record_host_orch_callable: cid=%d (host dlopen #%zu)", callable_id, host_dlopen_total_);
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

    if (state.host_dlopen_handle != nullptr) {
        // hbg: dlclose the host handle; no orch SO refcount to decrement.
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

bool SimDeviceRunnerBase::has_callable(int32_t callable_id) const { return callables_.count(callable_id) != 0; }

BindCallableResult SimDeviceRunnerBase::bind_callable_to_runtime(Runtime &runtime, int32_t callable_id) {
    auto it = callables_.find(callable_id);
    if (it == callables_.end()) {
        LOG_ERROR("bind_callable_to_runtime: callable_id=%d not registered", callable_id);
        return {-1, nullptr, nullptr, 0};
    }
    const auto &state = it->second;
    for (const auto &kv : state.kernel_addrs) {
        if (kv.first < 0 || kv.first >= RUNTIME_MAX_FUNC_ID) {
            LOG_ERROR("bind_callable_to_runtime: func_id=%d out of range", kv.first);
            return {-1, nullptr, nullptr, 0};
        }
        runtime.replay_function_bin_addr(kv.first, kv.second);
    }
    runtime.set_device_orch_func_name(state.func_name.c_str());
    runtime.set_device_orch_config_name(state.config_name.c_str());
    runtime.set_active_callable_id(callable_id, /*is_new=*/false);
    return {
        0, state.host_orch_func_ptr, state.signature.empty() ? nullptr : state.signature.data(),
        static_cast<int>(state.signature.size())
    };
}

uint64_t SimDeviceRunnerBase::upload_chip_callable_buffer(const ChipCallable *callable) {
    if (callable == nullptr || callable->child_count() == 0) {
        return 0;
    }

    const ChipCallableLayout layout = compute_chip_callable_layout(callable);

    auto it = chip_callable_buffers_.find(layout.content_hash);
    if (it != chip_callable_buffers_.end()) {
        LOG_DEBUG(
            "Chip callable dedup hit (sim): chip_dev=0x%lx, size=%zu, hash=0x%lx", it->second.chip_dev,
            it->second.total_size, layout.content_hash
        );
        return it->second.chip_dev;
    }

    // Allocate host scratch (host == device in sim). Plain new[] keeps
    // ChipCallableBuffer::host_scratch ownership symmetric with finalize().
    auto *scratch = new uint8_t[layout.total_size];
    std::memcpy(scratch, callable, layout.total_size);

    // Per-child dlopen + dlsym kernel_entry + register pto-sim hooks, then
    // patch the child's resolved_addr_ to the function pointer. A scope guard
    // owns scratch and any dlopen'd handles until the success path dismisses
    // it; every early return unwinds cleanly.
    std::vector<void *> dlopen_handles;
    dlopen_handles.reserve(callable->child_count());
    auto cleanup = RAIIScopeGuard([&]() {
        for (void *h : dlopen_handles)
            dlclose(h);
        delete[] scratch;
    });

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

    cleanup.dismiss();
    const uint64_t chip_dev = reinterpret_cast<uint64_t>(scratch);
    chip_callable_buffers_.emplace(
        layout.content_hash, ChipCallableBuffer{chip_dev, scratch, layout.total_size, std::move(dlopen_handles)}
    );
    LOG_DEBUG(
        "Uploaded chip callable (sim): chip_dev=0x%lx, size=%zu, child_count=%d, hash=0x%lx", chip_dev,
        layout.total_size, callable->child_count(), layout.content_hash
    );
    return chip_dev;
}

void SimDeviceRunnerBase::print_handshake_results() {
    if (worker_count_ == 0 || last_runtime_ == nullptr) {
        return;
    }

    LOG_DEBUG("Handshake results for %d cores:", worker_count_);
    for (int i = 0; i < worker_count_; i++) {
        LOG_DEBUG(
            "  Core %d: aicore_done=%d aicpu_ready=%d task=%d", i, last_runtime_->workers[i].aicore_done,
            last_runtime_->workers[i].aicpu_ready, last_runtime_->workers[i].task
        );
    }
}

void SimDeviceRunnerBase::release_callable_state() {
    // Release any chip callable buffers uploaded via upload_chip_callable_buffer.
    // Pool semantics mirror per-fid binaries: never freed until finalize.
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

    // Release any prepared-callable orch SO buffers callers forgot to drop.
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
}
