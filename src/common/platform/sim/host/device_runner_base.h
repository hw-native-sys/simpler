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
#include "common/l2_swimlane_profiling.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "host/memory_allocator.h"
#include "host/l3_l2_orch_comm_service.h"
#include "host/l2_swimlane_collector.h"
#include "host/tensor_dump_collector.h"
#include "host/pmu_collector.h"
#include "host/scope_stats_collector.h"
#include "runtime.h"

class SimDeviceRunnerBase : public L3L2OrchCommBackend {
public:
    SimDeviceRunnerBase() :
        gm_heap_arena_(&arena_alloc_trampoline, &arena_free_trampoline, &mem_alloc_),
        gm_sm_arena_(&arena_alloc_trampoline, &arena_free_trampoline, &mem_alloc_),
        runtime_arena_pool_(&arena_alloc_trampoline, &arena_free_trampoline, &mem_alloc_) {}

    // Public virtual dtor so c_api_shared can `delete` a SimDeviceRunnerBase *
    // (destroy_device_context entrypoint).
    virtual ~SimDeviceRunnerBase() = default;

    // --- Pure / no-op virtuals dispatched from the shared c_api glue ----
    virtual int run(Runtime &runtime, int block_dim, int launch_aicpu_num = 1) = 0;
    virtual int finalize() = 0;
    // a2a3 and a5 both override; an arch without dep_gen leaves the no-op.
    virtual void set_dep_gen_enabled(bool /*enable*/) {}

    // --- Shared methods --------------------------------------------------

    int setup_static_arena(size_t gm_heap_size, size_t gm_sm_size, size_t runtime_arena_size);

    void *acquire_pooled_gm_heap();
    void *acquire_pooled_gm_sm();
    void *acquire_pooled_runtime_arena();

    std::thread create_thread(std::function<void()> fn);
    int attach_current_thread(int device_id);

    void *allocate_tensor(size_t bytes);
    void free_tensor(void *dev_ptr);
    int copy_to_device(void *dev_ptr, const void *host_ptr, size_t bytes);
    int copy_from_device(void *host_ptr, const void *dev_ptr, size_t bytes);

    int l3_l2_orch_comm_init(void *control_block, size_t control_block_size);
    int l3_l2_orch_comm_shutdown();

    int register_callable(
        int32_t callable_id, const void *orch_so_data, size_t orch_so_size, const char *func_name,
        const char *config_name, std::vector<std::pair<int, uint64_t>> kernel_addrs, std::vector<ArgDirection> signature
    );
    int register_callable_host_orch(
        int32_t callable_id, void *host_dlopen_handle, void *host_orch_func_ptr,
        std::vector<std::pair<int, uint64_t>> kernel_addrs, std::vector<ArgDirection> signature
    );
    int unregister_callable(int32_t callable_id);
    bool has_callable(int32_t callable_id) const;
    BindCallableResult bind_callable_to_runtime(Runtime &runtime, int32_t callable_id);
    uint64_t upload_chip_callable_buffer(const ChipCallable *callable);

    void print_handshake_results();

    void set_executors(std::vector<uint8_t> aicpu_so_binary, std::vector<uint8_t> aicore_kernel_binary) {
        aicpu_so_binary_ = std::move(aicpu_so_binary);
        aicore_kernel_binary_ = std::move(aicore_kernel_binary);
    }
    int device_id() const { return device_id_; }
    uint64_t last_device_wall_ns() const { return device_wall_ns_; }

    void set_l2_swimlane_enabled(int level) {
        l2_swimlane_level_ = static_cast<L2SwimlaneLevel>(level);
        enable_l2_swimlane_ = (l2_swimlane_level_ != L2SwimlaneLevel::DISABLED);
    }
    void set_dump_tensor_enabled(int level) {
        dump_tensor_level_ = static_cast<DumpTensorLevel>(level);
        enable_dump_tensor_ = (dump_tensor_level_ != DumpTensorLevel::OFF);
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

    size_t aicpu_dlopen_count() const { return aicpu_dlopen_total_; }
    size_t host_dlopen_count() const { return host_dlopen_total_; }

protected:
    // --- Helpers usable by subclass run() / finalize() -------------------
    int ensure_device_initialized();
    virtual int ensure_binaries_loaded() = 0;
    int prepare_orch_so(Runtime &runtime);

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

    // Chip-callable buffer pool (sim path). Keyed by FNV-1a 64-bit content
    // hash. Each entry owns a host scratch holding the ChipCallable with each
    // child's resolved_addr_ fixed up to the dlopen'd function pointer;
    // chip_dev == (uint64_t)host_scratch. The dlopen handles in
    // dlopen_handles are bulk-dlclose'd in finalize().
    struct ChipCallableBuffer {
        uint64_t chip_dev{0};  // (uint64_t)host_scratch
        uint8_t *host_scratch{nullptr};
        size_t total_size{0};
        std::vector<void *> dlopen_handles;
    };
    std::unordered_map<uint64_t, ChipCallableBuffer> chip_callable_buffers_;

    // Per-callable_id prepared state. Mirrors onboard.
    struct CallableState {
        // trb path
        uint64_t hash{0};
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
    struct OrchSoBuffer {
        void *dev_addr{nullptr};
        size_t capacity{0};
        int refcount{0};
    };
    std::unordered_map<int32_t, CallableState> callables_;
    std::unordered_map<uint64_t, OrchSoBuffer> orch_so_dedup_;
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
    TensorDumpCollector dump_collector_;
    PmuCollector pmu_collector_;
    ScopeStatsCollector scope_stats_collector_;

    // Enablement flags. Written via setters before run(); read inside run().
    bool enable_l2_swimlane_{false};
    bool enable_dump_tensor_{false};
    DumpTensorLevel dump_tensor_level_{DumpTensorLevel::OFF};  // resolved from set_dump_tensor_enabled()
    bool enable_pmu_{false};
    bool enable_scope_stats_{false};
    L2SwimlaneLevel l2_swimlane_level_{L2SwimlaneLevel::DISABLED};  // resolved from set_l2_swimlane_enabled()
    PmuEventType pmu_event_type_{PmuEventType::PIPE_UTILIZATION};   // resolved from set_pmu_enabled()
    std::string output_prefix_{};                                   // diagnostic artifact root directory

private:
    void *l3_l2_allocate_region_bytes(uint64_t bytes) override;
    void l3_l2_free_region_bytes(void *ptr) override;
    int l3_l2_copy_to_device(void *dev_ptr, const void *host_ptr, uint64_t bytes) override;
    int l3_l2_copy_from_device(void *host_ptr, const void *dev_ptr, uint64_t bytes) override;
    std::thread l3_l2_create_service_thread(std::function<void()> fn) override;

    L3L2OrchCommService l3_l2_orch_comm_service_;
    std::mutex l3_l2_alloc_mu_;
    std::unordered_set<void *> l3_l2_allocations_;
};

namespace simpler::common::sim_host {

// Shared utility used by ensure_binaries_loaded() / upload_chip_callable_buffer()
// to materialize an in-memory DSO into /tmp so dlopen can pick it up. mkstemp +
// fchmod 0755 + write_all + close; on success out_path receives the path.
bool create_temp_so_file(const std::string &path_template, const uint8_t *data, size_t size, std::string *out_path);

}  // namespace simpler::common::sim_host

#endif  // SRC_COMMON_PLATFORM_SIM_HOST_DEVICE_RUNNER_BASE_H_
