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
 * SimDeviceRunnerBase — shared base class for sim DeviceRunners (a2a3 + a5).
 *
 * Mirrors the onboard DeviceRunnerBase pattern: shared lifecycle / callable
 * registry / arena / tensor-copy methods live here once; per-arch DeviceRunner
 * subclasses (in src/{a2a3,a5}/platform/sim/host/) implement the arch-specific
 * run() / finalize() / init_* / ensure_binaries_loaded path with their own
 * dlsym'd function-pointer table.
 *
 * Polymorphism keeps the c_api shared glue (c_api_shared.cpp) arch-agnostic —
 * it works through SimDeviceRunnerBase* and dispatches run() / finalize() /
 * set_dep_gen_enabled() through virtuals.
 */

#ifndef SRC_COMMON_PLATFORM_SIM_HOST_DEVICE_RUNNER_BASE_H_
#define SRC_COMMON_PLATFORM_SIM_HOST_DEVICE_RUNNER_BASE_H_

#include <dlfcn.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "callable.h"
#include "prepare_callable_common.h"
#include "utils/device_arena.h"
#include "common/kernel_args.h"
#include "common/device_phase.h"
#include "common/l2_swimlane_profiling.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "host/memory_allocator.h"
#include "host/l2_swimlane_collector.h"
#include "host/args_dump_collector.h"
#include "host/pmu_collector.h"
#include "host/scope_stats_collector.h"
#include "runtime.h"

struct HostApi;     // common/host_api.h — fwd-declared to keep task_interface headers out
struct CallConfig;  // task_interface/call_config.h — per-run config threaded into run()

class SimDeviceRunnerBase {
public:
    SimDeviceRunnerBase() :
        gm_heap_arena_(&arena_alloc_trampoline, &arena_free_trampoline, &mem_alloc_),
        gm_sm_arena_(&arena_alloc_trampoline, &arena_free_trampoline, &mem_alloc_),
        runtime_arena_pool_(&arena_alloc_trampoline, &arena_free_trampoline, &mem_alloc_) {}

    // Public virtual dtor so c_api_shared can `delete` a SimDeviceRunnerBase *
    // (destroy_device_context entrypoint).
    virtual ~SimDeviceRunnerBase() = default;

    // --- Pure / no-op virtuals dispatched from the shared c_api glue ----
    virtual int run(Runtime &runtime, const CallConfig &config) = 0;
    virtual int finalize() = 0;
    // a2a3 and a5 both override; an arch without dep_gen leaves the no-op.
    virtual void set_dep_gen_enabled(bool /*enable*/) {}

    // --- Shared methods --------------------------------------------------

    int setup_static_arena(size_t gm_heap_size, size_t gm_sm_size, size_t runtime_arena_size);

    void *acquire_pooled_gm_heap();
    void *acquire_pooled_gm_sm();
    void *acquire_pooled_runtime_arena();
    bool lookup_prebuilt_runtime_arena_cache(
        uint64_t hash, const void *key_data, size_t key_size, void **gm_heap_base, void **sm_base,
        void **runtime_arena_base, size_t *runtime_off, const void **image_data, size_t *image_size
    ) const;
    void mark_prebuilt_runtime_arena_cached(
        uint64_t hash, const void *key_data, size_t key_size, void *gm_heap_base, void *sm_base,
        void *runtime_arena_base, size_t runtime_off, const void *image_data, size_t image_size
    );

    std::thread create_thread(std::function<void()> fn);
    int attach_current_thread(int device_id);

    void *allocate_tensor(size_t bytes);
    void free_tensor(void *dev_ptr);
    int copy_to_device(void *dev_ptr, const void *host_ptr, size_t bytes);
    int copy_from_device(void *host_ptr, const void *dev_ptr, size_t bytes);
    int device_memset(void *dev_ptr, int value, size_t bytes);
    void get_retained_temp_buffer(void **addr, size_t *size);
    void set_retained_temp_buffer(void *addr, size_t size);
    void clear_temporary_buffer();

    // On sim, allocate_tensor returns a plain host pointer, so the "device"
    // address is already host-readable — register is identity, unregister a
    // no-op. Mirrors the onboard DeviceRunnerBase API (separate class trees).
    void *register_device_memory_to_host(void *dev_ptr, size_t bytes) {
        (void)bytes;
        return dev_ptr;
    }
    void unregister_device_memory_from_host(void *dev_ptr) { (void)dev_ptr; }

    int record_device_orch_callable(
        int32_t callable_id, uint64_t chip_buffer_hash, uint64_t chip_dev, const void *orch_so_data,
        size_t orch_so_size, const char *func_name, const char *config_name,
        std::vector<std::pair<int, uint64_t>> kernel_addrs, std::vector<ArgDirection> signature
    );
    int record_host_orch_callable(
        int32_t callable_id, uint64_t chip_buffer_hash, void *host_dlopen_handle, void *host_orch_func_ptr,
        std::vector<std::pair<int, uint64_t>> kernel_addrs, std::vector<ArgDirection> signature
    );
    int unregister_callable(int32_t callable_id);
    bool has_callable(int32_t callable_id) const;
    // One-step bind: replay CallableState (kernel addrs + active_callable_id)
    // then run the per-run bind_callable_to_runtime_impl with the state's
    // host_orch_func_ptr + signature. `api` is g_host_api; `orch_args` is a
    // const ChipStorageTaskArgs* (void* keeps task_interface headers out of this
    // header). Returns 0 on success, non-zero on failure.
    int bind_callable_to_runtime(
        Runtime &runtime, int32_t callable_id, const HostApi *api, const void *orch_args,
        const uint64_t *ring_task_window, const uint64_t *ring_heap, const uint64_t *ring_dep_pool
    );
    uint64_t upload_chip_callable_buffer(const ChipCallable *callable);
    int release_chip_callable_buffer(uint64_t hash);
    int launch_device_register(int32_t callable_id);
    int commit_device_register(int32_t callable_id);

    void print_handshake_results();

    void set_executors(std::vector<uint8_t> aicpu_so_binary, std::vector<uint8_t> aicore_kernel_binary) {
        aicpu_so_binary_ = std::move(aicpu_so_binary);
        aicore_kernel_binary_ = std::move(aicore_kernel_binary);
    }
    int device_id() const { return device_id_; }
    uint64_t last_device_wall_ns() const { return device_wall_ns_; }
    // Per-phase AICPU wall (ns) from the most recent run; RunWall aliases
    // last_device_wall_ns(). 0 for a phase that was never stamped. Used to emit
    // device-phase trace markers from the sim c_api, mirroring onboard.
    uint64_t last_device_phase_ns(AicpuPhase phase) const { return device_phase_ns_[static_cast<int>(phase)]; }
    // Per-phase start offset (ns) on a common device-clock timeline (origin =
    // earliest sub-phase start), so device spans carry a device-domain `ts` and
    // the orch∪sched "Effective" window is computable. 0 for RunWall / unstamped.
    uint64_t last_device_phase_start_ns(AicpuPhase phase) const {
        return device_phase_start_ns_[static_cast<int>(phase)];
    }
    // Per-slot task-timing dispatch/finish (ns) on the same device-clock timeline
    // as the phases. Both 0 for an untagged or incomplete slot. `slot` is 0..15.
    uint64_t last_task_slot_dispatch_ns(int slot) const { return task_slot_dispatch_ns_[slot]; }
    uint64_t last_task_slot_finish_ns(int slot) const { return task_slot_finish_ns_[slot]; }

    void set_l2_swimlane_enabled(int level) {
        l2_swimlane_level_ = static_cast<L2SwimlaneLevel>(level);
        enable_l2_swimlane_ = (l2_swimlane_level_ != L2SwimlaneLevel::DISABLED);
    }
    void set_dump_args_enabled(int level) {
        dump_args_level_ = static_cast<DumpArgsLevel>(level);
        enable_dump_args_ = (dump_args_level_ != DumpArgsLevel::OFF);
    }
    void set_pmu_enabled(int enable_pmu) {
        enable_pmu_ = (enable_pmu > 0);
        pmu_event_type_ = resolve_pmu_event_type(enable_pmu);
    }
    void set_scope_stats_enabled(bool enable) { enable_scope_stats_ = enable; }
    // Diagnostic artifact root directory (CallConfig::validate() enforces non-empty
    // upstream when any diagnostic is enabled).
    void set_output_prefix(const char *prefix) { output_prefix_ = (prefix != nullptr) ? prefix : ""; }
    const std::string &output_prefix() const { return output_prefix_; }

    // Latch this run's per-run diagnostic config onto the runner's enable_*_
    // members before run() uses them. Each arch's run() calls this at entry; the
    // c_api threads the CallConfig through instead of calling set_*_enabled.
    // Defined in the .cpp so this header does not need the full CallConfig.
    void apply_call_config(const CallConfig &config);

    size_t aicpu_dlopen_count() const { return aicpu_dlopen_total_; }
    size_t host_dlopen_count() const { return host_dlopen_total_; }

protected:
    // --- Helpers usable by subclass run() / finalize() -------------------
    int ensure_device_initialized();
    virtual int ensure_binaries_loaded() = 0;
    // Hand the orch-SO descriptor to the sim AICPU register entry. Built
    // directly from CallableState by launch_device_register — no Runtime
    // round-trip.
    virtual int invoke_device_register(const RegisterCallableArgs &reg_args) = 0;
    int prepare_orch_so(Runtime &runtime);
    int stamp_orch_so(Runtime &runtime, int32_t callable_id);

    // Bulk-free the shared callable / chip-callable / orch-SO state. Subclass
    // finalize() calls this before mem_alloc_.finalize(). Idempotent.
    void release_callable_state();

    // --- Shared state (protected so subclass run() / init_* / finalize()
    // can read or write directly) ----------------------------------------

    // Configuration. device_id_ is set once in attach_current_thread() during
    // simpler_init and read afterwards; the user's call sequence is single-
    // threaded with respect to it so plain int is sufficient.
    int device_id_{-1};
    int block_dim_{0};
    int cores_per_blockdim_{PLATFORM_CORES_PER_BLOCKDIM};
    int worker_count_{0};

    // Executor binaries — populated once via set_executors() during simpler_init,
    // owned for the rest of the runner's lifetime.
    std::vector<uint8_t> aicpu_so_binary_;
    std::vector<uint8_t> aicore_kernel_binary_;

    MemoryAllocator mem_alloc_;
    void *retained_temp_addr_ = nullptr;
    size_t retained_temp_size_ = 0;

    // Three independent per-Worker arenas, each backing a single pooled
    // region (PTO2 GM heap / PTO2 shared memory / trb prebuilt runtime
    // arena). Split out from a single backing allocation because the
    // combined size can exceed the device allocator's largest contiguous
    // block. Released explicitly in finalize() before mem_alloc_.finalize().
    //
    // runtime_arena_pool_ stays unreserved when setup_static_arena was
    // invoked with runtime_arena_size == 0 (hbg path).
    static void *arena_alloc_trampoline(void *ctx, size_t size) {
        return static_cast<MemoryAllocator *>(ctx)->alloc(size);
    }
    static void arena_free_trampoline(void *ctx, void *p) { static_cast<MemoryAllocator *>(ctx)->free(p); }
    DeviceArena gm_heap_arena_;
    DeviceArena gm_sm_arena_;
    DeviceArena runtime_arena_pool_;
    // Cached sizes for setup_static_arena's "fits" check — avoids re-allocating
    // a buffer when a later worker init asks for an equal-or-smaller layout.
    size_t cached_gm_heap_size_{0};
    size_t cached_gm_sm_size_{0};
    size_t cached_runtime_arena_size_{0};
    bool prebuilt_runtime_arena_cache_valid_{false};
    uint64_t prebuilt_runtime_arena_cache_hash_{0};
    std::vector<uint8_t> prebuilt_runtime_arena_cache_key_;
    void *prebuilt_runtime_arena_cache_gm_heap_base_{nullptr};
    void *prebuilt_runtime_arena_cache_sm_base_{nullptr};
    void *prebuilt_runtime_arena_cache_runtime_arena_base_{nullptr};
    size_t prebuilt_runtime_arena_cache_runtime_off_{0};
    std::vector<uint8_t> prebuilt_runtime_arena_cache_image_;

    // Simulation state — written by run() / init_* and read by the AICPU /
    // AICore execute functions via the platform-regs setter functions.
    KernelArgs kernel_args_;

    // Platform-level device wall buffer: 8-byte device-resident slot whose
    // address rides on KernelArgs.device_wall_data_base. AICPU writes the
    // run wall (ns) through that pointer; this DeviceRunner pulls it back
    // via copy_from_device after stream sync and caches it for
    // last_device_wall_ns(). Allocated lazily in run(), freed in finalize().
    void *device_wall_dev_ptr_{nullptr};
    uint64_t device_wall_ns_{0};
    uint64_t device_phase_ns_[NUM_AICPU_PHASES] = {0};
    // Per-phase start offset (ns) from the earliest sub-phase start; see
    // last_device_phase_start_ns().
    uint64_t device_phase_start_ns_[NUM_AICPU_PHASES] = {0};
    // Per-slot task-timing dispatch/finish (ns), offset from the same origin as
    // the phases; see last_task_slot_dispatch_ns() / last_task_slot_finish_ns().
    uint64_t task_slot_dispatch_ns_[NUM_TASK_TIMING_SLOTS] = {0};
    uint64_t task_slot_finish_ns_[NUM_TASK_TIMING_SLOTS] = {0};

    // Chip-callable buffer pool (sim path). Keyed by FNV-1a 64-bit content
    // hash. Each entry owns a host scratch holding the ChipCallable with each
    // child's resolved_addr_ fixed up to the dlopen'd function pointer;
    // chip_dev == (uint64_t)host_scratch. The dlopen handles in
    // dlopen_handles are bulk-dlclose'd in finalize().
    struct ChipCallableBuffer {
        uint64_t chip_dev{0};  // (uint64_t)host_scratch
        uint8_t *host_scratch{nullptr};
        size_t total_size{0};
        int refcount{0};
        std::vector<void *> dlopen_handles;
    };
    std::unordered_map<uint64_t, ChipCallableBuffer> chip_callable_buffers_;

    // Per-callable_id prepared state. Mirrors onboard.
    struct CallableState {
        // trb path
        uint64_t hash{0};
        uint64_t chip_buffer_hash{0};
        uint64_t dev_orch_so_addr{0};
        size_t dev_orch_so_size{0};
        std::string func_name;
        std::string config_name;
        // common
        std::vector<std::pair<int, uint64_t>> kernel_addrs;
        std::vector<ArgDirection> signature;
        // hbg path
        void *host_dlopen_handle{nullptr};
        void *host_orch_func_ptr{nullptr};
    };
    std::unordered_map<int32_t, CallableState> callables_;
    std::unordered_set<int32_t> aicpu_seen_callable_ids_;
    size_t aicpu_dlopen_total_{0};
    size_t host_dlopen_total_{0};
    // AICPU executor SO: load-once, matching onboard's binaries_loaded_ pattern.
    // The aicpu_executor g_aicpu_executor static lives inside the dlopen'd DSO;
    // reloading it destroys orch_so_handle_ and breaks the orch-SO cache-hit path.
    bool aicpu_so_loaded_{false};

    Runtime *last_runtime_{nullptr};

    // Dynamically loaded executor libraries (shared infra; the dlsym'd function-
    // pointer table itself lives on the subclass since signatures diverge
    // per-arch — a2a3 vs a5 differ on aicore_execute and several setters).
    void *aicpu_so_handle_{nullptr};
    void *aicore_so_handle_{nullptr};
    std::string aicpu_so_path_;
    std::string aicore_so_path_;

    // Performance / diagnostics collectors shared across arches.
    L2SwimlaneCollector l2_swimlane_collector_;
    ArgsDumpCollector dump_collector_;
    PmuCollector pmu_collector_;
    ScopeStatsCollector scope_stats_collector_;

    // Enablement flags. Written via setters before run(); read inside run().
    bool enable_l2_swimlane_{false};
    bool enable_dump_args_{false};
    DumpArgsLevel dump_args_level_{DumpArgsLevel::OFF};  // resolved from set_dump_args_enabled()
    bool enable_pmu_{false};
    bool enable_scope_stats_{false};
    L2SwimlaneLevel l2_swimlane_level_{L2SwimlaneLevel::DISABLED};  // resolved from set_l2_swimlane_enabled()
    PmuEventType pmu_event_type_{PmuEventType::PIPE_UTILIZATION};   // resolved from set_pmu_enabled()
    std::string output_prefix_{};                                   // diagnostic artifact root directory
};

namespace simpler::common::sim_host {

// Shared utility used by ensure_binaries_loaded() / upload_chip_callable_buffer()
// to materialize an in-memory DSO into /tmp so dlopen can pick it up. mkstemp +
// fchmod 0755 + write_all + close; on success out_path receives the path.
bool create_temp_so_file(const std::string &path_template, const uint8_t *data, size_t size, std::string *out_path);

}  // namespace simpler::common::sim_host

#endif  // SRC_COMMON_PLATFORM_SIM_HOST_DEVICE_RUNNER_BASE_H_
