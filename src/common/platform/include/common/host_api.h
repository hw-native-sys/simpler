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
#pragma once

#include <cstddef>
#include <cstdint>

/**
 * Host API function pointers for device memory operations.
 * Allows a runtime to use pluggable device-memory backends.
 *
 * The platform layer owns one immutable function table per backend. A HostApi
 * value binds that table to one runner and one run's slot/bank selection, so
 * callbacks never recover mutable context from process or thread globals.
 */
struct HostApiOps {
    void *(*device_malloc)(void *runner_ctx, size_t size);
    void (*device_free)(void *runner_ctx, void *dev_ptr);
    int (*copy_to_device)(void *runner_ctx, void *dev_ptr, const void *host_ptr, size_t size);
    int (*copy_from_device)(void *runner_ctx, void *host_ptr, const void *dev_ptr, size_t size);
    // Map a device buffer into host address space and return a host-readable VA
    // (nullptr on failure); the paired unregister releases it. The returned VA
    // may differ from dev_ptr, so callers must use it, not dev_ptr, for host
    // access, and pair every register with an unregister before free. Used by a
    // host-side orchestrator (host_build_graph) to read control tensors whose
    // buffer.addr is a device address. a2a3 onboard wraps
    // halHostRegister(DEV_SVM_MAP_HOST); sim is identity; a5 onboard and any
    // backend without a host-map path return nullptr / no-op.
    void *(*register_device_memory_to_host)(void *runner_ctx, void *dev_ptr, size_t bytes);
    void (*unregister_device_memory_from_host)(void *runner_ctx, void *dev_ptr);
    // Set a device buffer to a byte value (device-side, no PCIe). Used to
    // zero-init pure OUTPUT buffers in lieu of an H2D copy-in.
    int (*device_memset)(void *runner_ctx, void *dev_ptr, int value, size_t size);
    // Runner-scoped retained temporary buffer for TRB device-arg staging.
    // This is NOT an allocator — it is a single {addr, size} slot that lives
    // across runs on the DeviceRunner. trb bind reads the slot, and if the
    // retained buffer is too small for this run's packed temporary size it
    // device_free's the old one, device_malloc's a bigger one, and writes the
    // new {addr, size} back. The grow/pack/slice logic lives in trb bind
    // (runtime_maker); the platform only remembers the slot so it can be reused
    // by later runs and freed at finalize. The slot is per pipeline slot, so
    // two runs in different slots never share a staging buffer. `get` returns
    // {nullptr, 0} when nothing is retained yet.
    void (*get_retained_temp_buffer)(void *runner_ctx, uint32_t pipeline_slot, void **addr, size_t *size);
    void (*set_retained_temp_buffer)(void *runner_ctx, uint32_t pipeline_slot, void *addr, size_t size);
    // Commit the three pooled regions (GM heap, runtime shared memory, and
    // prebuilt runtime arena) of the arena bank selected by this run, as three
    // independent device allocations. `runtime_arena_size == 0` skips the
    // third region. Idempotent on identical sizes; returns 0 on success, -1 on
    // allocation failure.
    int (*setup_static_arena)(
        void *runner_ctx, uint32_t arena_bank, size_t gm_heap_size, size_t gm_sm_size, size_t runtime_arena_size
    );
    // Return the per-Worker pooled pointer for the GM heap / runtime shared
    // memory / prebuilt runtime arena. setup_static_arena must have already
    // committed the relevant region; the returned pointer is owned by the
    // DeviceRunner and freed in `DeviceRunner::finalize()` — do NOT pass it
    // to device_free or record it as an owned tensor lease.
    //
    // The runtime-arena region exists only when setup_static_arena was invoked
    // with runtime_arena_size > 0; otherwise acquire_pooled_runtime_arena
    // returns nullptr.
    void *(*acquire_pooled_gm_heap)(void *runner_ctx, uint32_t arena_bank);
    void *(*acquire_pooled_gm_sm)(void *runner_ctx, uint32_t arena_bank);
    void *(*acquire_pooled_runtime_arena)(void *runner_ctx, uint32_t arena_bank);
    // Prebuilt runtime-arena image cache (trb): look up a previously built
    // image by content hash, returning its pooled device bases + image bytes on
    // a hit; and record a freshly built image so a later run with the same key
    // can skip the rebuild. Populated on the trb path; unused by hbg.
    bool (*lookup_prebuilt_runtime_arena_cache)(
        void *runner_ctx, uint32_t arena_bank, uint64_t hash, const void *key_data, size_t key_size,
        void **gm_heap_base, void **sm_base, void **runtime_arena_base, size_t *runtime_off, const void **image_data,
        size_t *image_size
    );
    void (*mark_prebuilt_runtime_arena_cached)(
        void *runner_ctx, uint32_t arena_bank, uint64_t hash, const void *key_data, size_t key_size, void *gm_heap_base,
        void *sm_base, void *runtime_arena_base, size_t runtime_off, const void *image_data, size_t image_size
    );
    // Single-shot upload of the entire ChipCallable buffer. `callable` is a
    // `const ChipCallable *` (declared void* to avoid pulling task_interface
    // headers into this header). DeviceRunner walks child_offsets_ to compute
    // total byte size, allocates device GM once, fixes up each child's
    // resolved_addr_ in an internal host scratch (onboard: device addr; sim:
    // dlopen function pointer), H2D's once, and returns the device-side
    // address of the ChipCallable header. Pool-managed: identical buffer
    // contents (FNV-1a 64-bit) hit the dedup cache; all chip buffers are
    // bulk-freed in DeviceRunner::finalize(). Returns 0 on error or when
    // child_count() == 0. Caller computes child addrs as
    //     chip_dev + offsetof(ChipCallable, storage_) + child_offset(i)
    // and records them in the CallableArtifacts kernel_addrs table, which
    // DeviceRunner::bind_callable_to_runtime replays onto the runtime's
    // func_id_to_addr_ before each run.
    uint64_t (*upload_chip_callable_buffer)(void *runner_ctx, const void *callable);
    // Host phase records. The pool is platform-allocated but written directly by
    // the runtime through the inline path in host/host_phase_records.h, so these
    // two run once per prepare pass rather than once per record. Arming is the
    // union of the two enabling conditions: the runner contributes the
    // chip-swimlane level, `producer_wants_records` carries the producer's own
    // (a runtime knob the platform does not read).
    uint32_t (*get_chip_swimlane_level)(void *runner_ctx);
    void *(*host_phase_pool_arm)(void *runner_ctx, int producer_wants_records);
    void (*host_phase_pool_finish)(void *runner_ctx, uint64_t submitted_tasks, uint64_t invocation_id);
};

/**
 * One run's binding of the immutable function table to a runner and that run's
 * slot/bank selection. Constructed per run and passed by const pointer into the
 * runtime impls; a callback reaches its runner and resources through the bound
 * members instead of a thread-local.
 */
struct HostApi {
public:
    HostApi(void *runner_ctx, uint32_t pipeline_slot, uint32_t arena_bank, const HostApiOps *ops) noexcept :
        runner_ctx_(runner_ctx),
        pipeline_slot_(pipeline_slot),
        arena_bank_(arena_bank),
        ops_(ops) {}

    void *device_malloc(size_t size) const { return ops_->device_malloc(runner_ctx_, size); }
    void device_free(void *dev_ptr) const { ops_->device_free(runner_ctx_, dev_ptr); }
    int copy_to_device(void *dev_ptr, const void *host_ptr, size_t size) const {
        return ops_->copy_to_device(runner_ctx_, dev_ptr, host_ptr, size);
    }
    int copy_from_device(void *host_ptr, const void *dev_ptr, size_t size) const {
        return ops_->copy_from_device(runner_ctx_, host_ptr, dev_ptr, size);
    }
    void *register_device_memory_to_host(void *dev_ptr, size_t bytes) const {
        return ops_->register_device_memory_to_host(runner_ctx_, dev_ptr, bytes);
    }
    void unregister_device_memory_from_host(void *dev_ptr) const {
        ops_->unregister_device_memory_from_host(runner_ctx_, dev_ptr);
    }
    int device_memset(void *dev_ptr, int value, size_t size) const {
        return ops_->device_memset(runner_ctx_, dev_ptr, value, size);
    }
    void get_retained_temp_buffer(void **addr, size_t *size) const {
        ops_->get_retained_temp_buffer(runner_ctx_, pipeline_slot_, addr, size);
    }
    void set_retained_temp_buffer(void *addr, size_t size) const {
        ops_->set_retained_temp_buffer(runner_ctx_, pipeline_slot_, addr, size);
    }
    int setup_static_arena(size_t gm_heap_size, size_t gm_sm_size, size_t runtime_arena_size) const {
        return ops_->setup_static_arena(runner_ctx_, arena_bank_, gm_heap_size, gm_sm_size, runtime_arena_size);
    }
    void *acquire_pooled_gm_heap() const { return ops_->acquire_pooled_gm_heap(runner_ctx_, arena_bank_); }
    void *acquire_pooled_gm_sm() const { return ops_->acquire_pooled_gm_sm(runner_ctx_, arena_bank_); }
    void *acquire_pooled_runtime_arena() const { return ops_->acquire_pooled_runtime_arena(runner_ctx_, arena_bank_); }
    bool lookup_prebuilt_runtime_arena_cache(
        uint64_t hash, const void *key_data, size_t key_size, void **gm_heap_base, void **sm_base,
        void **runtime_arena_base, size_t *runtime_off, const void **image_data, size_t *image_size
    ) const {
        return ops_->lookup_prebuilt_runtime_arena_cache(
            runner_ctx_, arena_bank_, hash, key_data, key_size, gm_heap_base, sm_base, runtime_arena_base, runtime_off,
            image_data, image_size
        );
    }
    void mark_prebuilt_runtime_arena_cached(
        uint64_t hash, const void *key_data, size_t key_size, void *gm_heap_base, void *sm_base,
        void *runtime_arena_base, size_t runtime_off, const void *image_data, size_t image_size
    ) const {
        ops_->mark_prebuilt_runtime_arena_cached(
            runner_ctx_, arena_bank_, hash, key_data, key_size, gm_heap_base, sm_base, runtime_arena_base, runtime_off,
            image_data, image_size
        );
    }
    uint64_t upload_chip_callable_buffer(const void *callable) const {
        return ops_->upload_chip_callable_buffer(runner_ctx_, callable);
    }
    uint32_t chip_swimlane_level() const {
        return ops_->get_chip_swimlane_level != nullptr ? ops_->get_chip_swimlane_level(runner_ctx_) : 0;
    }
    /**
     * Arm this pass's host phase pool.
     *
     * @param producer_wants_records  the producer's own enabling condition; the
     *                                runner ORs it with the chip-swimlane level
     * @return HostPhaseRecordPool* to record into, or nullptr when this pass
     *         collects no records (typed void* to keep the profiling headers out
     *         of this one)
     */
    void *host_phase_pool_arm(bool producer_wants_records) const noexcept {
        if (ops_->host_phase_pool_arm == nullptr) return nullptr;
        return ops_->host_phase_pool_arm(runner_ctx_, producer_wants_records ? 1 : 0);
    }
    void host_phase_pool_finish(uint64_t submitted_tasks, uint64_t invocation_id) const noexcept {
        if (ops_->host_phase_pool_finish != nullptr) {
            ops_->host_phase_pool_finish(runner_ctx_, submitted_tasks, invocation_id);
        }
    }

private:
    void *runner_ctx_{nullptr};
    uint32_t pipeline_slot_{0};
    uint32_t arena_bank_{0};
    const HostApiOps *ops_{nullptr};
};
