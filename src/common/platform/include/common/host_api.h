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
#ifndef COMMON_HOST_API_H_
#define COMMON_HOST_API_H_

#include <cstddef>
#include <cstdint>

/**
 * Host API function pointers for device memory operations.
 * Allows a runtime to use pluggable device-memory backends.
 *
 * This is a platform capability, not runtime state: the platform layer
 * (onboard / sim c_api_shared.cpp) builds one shared, const table of
 * context-free function pointers at load time and passes it by address into
 * bind_callable_to_runtime_impl / validate_runtime_impl. Each pointer recovers
 * its runner from a thread-local, so the single table is valid for every runner
 * and every run — it is not rebuilt per simpler_run. Shared by every runtime
 * variant (tensormap_and_ringbuffer / host_build_graph) and arch (a2a3 / a5) so
 * the field set stays defined in exactly one place.
 */
struct HostApi {
    // Async Host orchestration must inherit both the runner TLS and the
    // backend's per-thread device binding before using callbacks below.
    void *(*capture_thread_context)();
    int (*bind_thread_context)(void *context);
    void (*unbind_thread_context)();
    void *(*device_malloc)(size_t size);
    void (*device_free)(void *dev_ptr);
    int (*copy_to_device)(void *dev_ptr, const void *host_ptr, size_t size);
    int (*copy_from_device)(void *host_ptr, const void *dev_ptr, size_t size);
    // Publish and observe 64-bit device control words with release/acquire
    // semantics. Onboard backends use synchronous device copies. Sim backends
    // must use atomic operations because Host and Device are concurrent threads
    // sharing one address space; a plain memcpy does not form a publication
    // edge on weakly ordered hosts.
    int (*store_u64_release_to_device)(void *dev_ptr, uint64_t value);
    int (*load_u64_acquire_from_device)(uint64_t *value, const void *dev_ptr);
    // Map a device buffer into host address space and return a host-readable VA
    // (nullptr on failure); the paired unregister releases it. The returned VA
    // may differ from dev_ptr, so callers must use it, not dev_ptr, for host
    // access, and pair every register with an unregister before free. Used by a
    // host-side orchestrator (host_build_graph) to read control tensors whose
    // buffer.addr is a device address. a2a3 onboard wraps
    // halHostRegister(DEV_SVM_MAP_HOST); sim is identity; a5 onboard and any
    // backend without a host-map path return nullptr / no-op.
    void *(*register_device_memory_to_host)(void *dev_ptr, size_t bytes);
    void (*unregister_device_memory_from_host)(void *dev_ptr);
    // Set a device buffer to a byte value (device-side, no PCIe). Used to
    // zero-init pure OUTPUT buffers in lieu of an H2D copy-in. May be
    // null on backends that don't wire it; callers must fall back to
    // copy_to_device.
    int (*device_memset)(void *dev_ptr, int value, size_t size);
    // Runner-scoped retained temporary buffer for TRB device-arg staging.
    // This is NOT an allocator — it is a single {addr, size} slot that lives
    // across runs on the DeviceRunner. trb bind reads the slot, and if the
    // retained buffer is too small for this run's packed temporary size it
    // device_free's the old one, device_malloc's a bigger one, and writes the
    // new {addr, size} back. The grow/pack/slice logic lives in trb bind
    // (runtime_maker); the platform only remembers the slot so it can be reused
    // next run and freed at finalize. hbg leaves these null and stages tensors
    // through per-tensor device_malloc. `get` returns {nullptr, 0} when nothing
    // is retained yet.
    void (*get_retained_temp_buffer)(void **addr, size_t *size);
    void (*set_retained_temp_buffer)(void *addr, size_t size);
    // Commit the three per-Worker pooled regions (PTO2 GM heap, PTO2 shared
    // memory, trb prebuilt runtime arena) as three independent device
    // allocations. `runtime_arena_size == 0` skips the third region (hbg
    // path: hbg has no prebuilt runtime arena). Idempotent on identical
    // sizes; returns 0 on success, -1 on allocation failure.
    int (*setup_static_arena)(size_t gm_heap_size, size_t gm_sm_size, size_t runtime_arena_size);
    // Return the per-Worker pooled pointer for the PTO2 GM heap / shared
    // memory / prebuilt runtime arena. setup_static_arena must have already
    // committed the relevant region; the returned pointer is owned by the
    // DeviceRunner and freed in `DeviceRunner::finalize()` — do NOT pass it
    // to device_free or record it as an owned tensor lease.
    //
    // acquire_pooled_runtime_arena is trb-only — the runtime-arena region is
    // only committed when setup_static_arena was invoked with
    // runtime_arena_size > 0. Calling it on the hbg path
    // (setup_static_arena(...,0)) returns nullptr (not undefined).
    void *(*acquire_pooled_gm_heap)();
    void *(*acquire_pooled_gm_sm)();
    void *(*acquire_pooled_runtime_arena)();
    // Prebuilt runtime-arena image cache (trb): look up a previously built
    // image by content hash, returning its pooled device bases + image bytes on
    // a hit; and record a freshly built image so a later run with the same key
    // can skip the rebuild. Populated on the trb path; unused by hbg.
    bool (*lookup_prebuilt_runtime_arena_cache)(
        uint64_t hash, const void *key_data, size_t key_size, void **gm_heap_base, void **sm_base,
        void **runtime_arena_base, size_t *runtime_off, const void **image_data, size_t *image_size
    );
    void (*mark_prebuilt_runtime_arena_cached)(
        uint64_t hash, const void *key_data, size_t key_size, void *gm_heap_base, void *sm_base,
        void *runtime_arena_base, size_t runtime_off, const void *image_data, size_t image_size
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
    uint64_t (*upload_chip_callable_buffer)(const void *callable);
};

#endif  // COMMON_HOST_API_H_
