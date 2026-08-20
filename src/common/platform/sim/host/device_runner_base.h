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
 * enqueue/poll/drain / finalize / init_* / ensure_binaries_loaded path with
 * their own dlsym'd function-pointer table.
 *
 * Polymorphism keeps the c_api shared glue (c_api_shared.cpp) arch-agnostic —
 * it works through SimDeviceRunnerBase* and dispatches the execution lifecycle,
 * finalize(), and set_dep_gen_enabled() through virtuals.
 */

#pragma once

#include <dlfcn.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pto_runtime_c_api.h"
#include "native_run_execution.h"

#include "callable.h"
#include "call_config.h"
#include "prepare_callable_common.h"
#include "utils/device_arena.h"
#include "common/kernel_args.h"
#include "common/device_phase.h"
#include "common/chip_swimlane_profiling.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "host/memory_allocator.h"
#include "host/chip_swimlane_collector.h"
#include "host/host_phase_records.h"
#include "host/args_dump_collector.h"
#include "host/pmu_collector.h"
#include "host/scope_stats_collector.h"
#include "runtime.h"

struct HostApi;  // common/host_api.h — fwd-declared to keep task_interface headers out

// Width sim resolves the CallConfig "auto" sentinel to, deliberately below
// PLATFORM_MAX_BLOCKDIM (24 on a2a3, 36 on a5). The simulator runs one OS
// thread per AICore, so taking the whole modelled chip would be 72-108 threads
// per case; under xdist that is several hundred threads for no added coverage.
// CallConfig exposes no block_dim knob, so this is what a sim run takes.
// Onboard is unaffected — it auto-resolves from the real per-stream core
// limits.
constexpr int SIM_AUTO_BLOCKDIM = 8;

class SimDeviceRunnerBase {
public:
    SimDeviceRunnerBase() {
        for (auto &bank : arena_banks_) {
            bank = std::make_unique<ArenaBank>(&arena_alloc_trampoline, &arena_free_trampoline, &mem_alloc_);
        }
    }

    uint64_t arena_bank_gm_heap_base(uint32_t bank_id) const;

    /**
     * Retained temporary-buffer address held for one pipeline slot, or 0 while
     * that slot holds none. Two slots that have both staged arguments hold
     * distinct buffers; tests read this to prove the split is real.
     */
    uint64_t retained_temp_addr(uint32_t slot_id) const;

    // Public virtual dtor so c_api_shared can `delete` a SimDeviceRunnerBase *
    // (destroy_device_context entrypoint).
    virtual ~SimDeviceRunnerBase() = default;

    // --- Pure / no-op virtuals dispatched from the shared c_api glue ----
    struct PreparedExecution {
        PreparedExecution(
            const NativeRunIdentity &identity_in, Runtime &runtime_in, const CallConfig &config_in,
            uint32_t pipeline_slot_in
        ) :
            identity(identity_in),
            runtime(&runtime_in),
            config(config_in),
            pipeline_slot(pipeline_slot_in) {}
        virtual ~PreparedExecution() = default;
        PreparedExecution(const PreparedExecution &) = delete;
        PreparedExecution &operator=(const PreparedExecution &) = delete;
        PreparedExecution(PreparedExecution &&other) noexcept :
            identity(other.identity),
            runtime(std::exchange(other.runtime, nullptr)),
            config(other.config),
            pipeline_slot(other.pipeline_slot),
            num_aicore(other.num_aicore),
            launch_aicpu_num(other.launch_aicpu_num) {}
        PreparedExecution &operator=(PreparedExecution &&) = delete;

        NativeRunIdentity identity{};
        Runtime *runtime{nullptr};
        CallConfig config{};
        uint32_t pipeline_slot{PTO_PIPELINE_MAX_DEPTH};
        int num_aicore{0};
        int launch_aicpu_num{0};
    };

    struct ActiveExecution {
        explicit ActiveExecution(std::unique_ptr<PreparedExecution> prepared_in, LaunchProgress progress_in) :
            prepared(std::move(prepared_in)),
            progress(progress_in) {}
        ActiveExecution(const ActiveExecution &) = delete;
        ActiveExecution &operator=(const ActiveExecution &) = delete;
        ActiveExecution(ActiveExecution &&) noexcept = default;
        ActiveExecution &operator=(ActiveExecution &&) noexcept = default;
        std::unique_ptr<PreparedExecution> prepared;
        LaunchProgress progress{LaunchProgress::NotStarted};
    };

    struct LaunchOutcome {
        int rc{-1};
        LaunchProgress progress{LaunchProgress::NotStarted};
        std::unique_ptr<PreparedExecution> prepared{};
        std::unique_ptr<ActiveExecution> active{};
        LaunchReceipt receipt{};

        bool poisoned() const { return progress == LaunchProgress::Partial; }
    };
    /** Submit a Runtime and retain all state needed to query and drain it. */
    virtual int prepare_execution(
        Runtime &runtime, const CallConfig &config, uint32_t pipeline_slot, const NativeRunIdentity &identity,
        std::unique_ptr<PreparedExecution> *prepared
    ) = 0;
    virtual LaunchOutcome launch_execution(std::unique_ptr<PreparedExecution> prepared, LaunchPermit permit) = 0;
    virtual void abandon_prepared_execution(PreparedExecution &prepared) noexcept = 0;
    /** Return one of the SIMPLER_NATIVE_RUN_POLL_* values without waiting. */
    virtual int poll_execution(const ActiveExecution &active) = 0;
    /** Wait for completion, publish DFX, and release per-run resources. */
    virtual int drain_execution(ActiveExecution &active) = 0;
    virtual int finalize() = 0;
    // a2a3 and a5 both override; an arch without dep_gen leaves the no-op.
    virtual void set_dep_gen_enabled(bool /*enable*/) {}

    /** Reserve the runner's single active native execution through finalize. */
    bool try_acquire_native_run(const void *owner, const NativeRunIdentity &identity, LaunchPermit *permit);
    void release_native_run(const void *owner);
    bool native_run_active() const;
    bool native_run_owned_by(const void *owner) const;
    bool can_accept_run() const { return !launch_poisoned_.load(std::memory_order_acquire); }
    void poison_launch() { launch_poisoned_.store(true, std::memory_order_release); }

    // --- Shared methods --------------------------------------------------

    int setup_static_arena(uint32_t arena_bank, size_t gm_heap_size, size_t gm_sm_size, size_t runtime_arena_size);

    void *acquire_pooled_gm_heap(uint32_t arena_bank);
    void *acquire_pooled_gm_sm(uint32_t arena_bank);
    void *acquire_pooled_runtime_arena(uint32_t arena_bank);
    bool lookup_prebuilt_runtime_arena_cache(
        uint32_t arena_bank, uint64_t hash, const void *key_data, size_t key_size, void **gm_heap_base, void **sm_base,
        void **runtime_arena_base, size_t *runtime_off, const void **image_data, size_t *image_size
    ) const;
    void mark_prebuilt_runtime_arena_cached(
        uint32_t arena_bank, uint64_t hash, const void *key_data, size_t key_size, void *gm_heap_base, void *sm_base,
        void *runtime_arena_base, size_t runtime_off, const void *image_data, size_t image_size
    );

    std::thread create_thread(std::function<void()> fn);
    int attach_current_thread(int device_id);

    void *allocate_tensor(size_t bytes);
    /** Total device memory (bytes) currently committed by this runner's MemoryAllocator. */
    size_t committed_device_memory() const { return mem_alloc_.committed_bytes(); }
    void free_tensor(void *dev_ptr);
    int copy_to_device(void *dev_ptr, const void *host_ptr, size_t bytes);
    int copy_from_device(void *host_ptr, const void *dev_ptr, size_t bytes);
    int device_memset(void *dev_ptr, int value, size_t bytes);
    void get_retained_temp_buffer(uint32_t pipeline_slot, void **addr, size_t *size);
    void set_retained_temp_buffer(uint32_t pipeline_slot, void *addr, size_t size);
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
    // host_orch_func_ptr + signature. `api` is bound to this run; `orch_args` is a
    // const ChipStorageTaskArgs* (void* keeps task_interface headers out of this
    // header). Returns 0 on success, non-zero on failure.
    int bind_callable_to_runtime(
        Runtime &runtime, int32_t callable_id, const HostApi *api, const void *orch_args,
        const uint64_t *ring_task_window, const uint64_t *ring_heap, const uint64_t *ring_dep_pool
    );

    // Publish this run's core geometry onto `Runtime` before the graph is
    // built: resolves block_dim (SIM_AUTO_BLOCKDIM on auto), derives num_aicore,
    // and publishes worker_count / aicpu_thread_num plus the AIC/AIV-typed
    // handshake array. Callers run this before bind_callable_to_runtime so a
    // host-side orchestrator sees the real core count while it submits, rather
    // than the zeros a freshly constructed Runtime carries. Returns 0 on
    // success, -1 on a bad block_dim / aicpu_thread_num.
    int prepare_launch_shape(Runtime &runtime, const CallConfig &config);
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

    void set_chip_swimlane_enabled(int level) {
        chip_swimlane_level_ = static_cast<ChipSwimlaneLevel>(level);
        enable_chip_swimlane_ = (chip_swimlane_level_ != ChipSwimlaneLevel::DISABLED);
    }
    uint32_t chip_swimlane_level() const { return static_cast<uint32_t>(chip_swimlane_level_); }
    HostPhaseRecordPool *host_phase_pool_arm(bool producer_wants_records) noexcept;
    void host_phase_pool_finish(uint64_t submitted_tasks, uint64_t invocation_id) noexcept {
        host_phase_records_.finish(submitted_tasks, invocation_id);
    }
    const simpler::dfx::HostPhaseRecordStore &host_phase_records() const { return host_phase_records_; }
    /** Hand this pass's records to the swimlane reader, just before its export. */
    void publish_host_phase_records_to_swimlane();
    void finish_clock_correlation_session(bool capture_device_complete) noexcept;
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
    // members before prepare_execution() uses them. Each arch calls this at prepare;
    // the c_api threads the CallConfig through instead of calling setters.
    // Defined in the .cpp so this header does not need the full CallConfig.
    void apply_call_config(const CallConfig &config);

    size_t aicpu_dlopen_count() const { return aicpu_dlopen_total_; }
    size_t host_dlopen_count() const { return host_dlopen_total_; }

protected:
    // --- Helpers usable by subclass execution / finalize -----------------
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

    // --- Shared state (protected so subclass execution / init_* / finalize()
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
    std::array<void *, PTO_PIPELINE_MAX_DEPTH> retained_temp_addrs_{};
    std::array<size_t, PTO_PIPELINE_MAX_DEPTH> retained_temp_sizes_{};
    // Each arena bank backs the pooled regions (PTO2 GM heap / PTO2 shared
    // memory / prebuilt runtime arena) for one pipeline slot. They
    // are separate allocations because the combined size can exceed the device
    // allocator's largest contiguous block. Released explicitly in finalize()
    // before mem_alloc_.finalize().
    //
    // A bank's runtime pool stays unreserved when setup_static_arena was
    // invoked with runtime_arena_size == 0 (hbg path).
    static void *arena_alloc_trampoline(void *ctx, size_t size) {
        return static_cast<MemoryAllocator *>(ctx)->alloc(size);
    }
    static void arena_free_trampoline(void *ctx, void *p) { static_cast<MemoryAllocator *>(ctx)->free(p); }
    // One independently committed set of the three pooled regions per pipeline
    // slot, so preparing one bank never mutates a region the active run is
    // executing out of. `cached_*` back setup_static_arena's "fits" check —
    // avoids re-allocating when a later worker init asks for an equal-or-
    // smaller layout. Held by pointer because DeviceArena is neither copyable
    // nor movable, so the array cannot be brace-initialised without naming
    // every bank.
    struct ArenaBank {
        ArenaBank(DeviceArena::AllocFn alloc, DeviceArena::FreeFn free_fn, void *ctx) :
            gm_heap(alloc, free_fn, ctx),
            gm_sm(alloc, free_fn, ctx),
            runtime_pool(alloc, free_fn, ctx) {}

        DeviceArena gm_heap;
        DeviceArena gm_sm;
        DeviceArena runtime_pool;
        size_t cached_gm_heap_size{0};
        size_t cached_gm_sm_size{0};
        size_t cached_runtime_arena_size{0};
    };
    std::array<std::unique_ptr<ArenaBank>, PTO_PIPELINE_MAX_DEPTH> arena_banks_;
    ArenaBank &arena_bank(uint32_t bank_id) { return *arena_banks_[bank_id]; }
    bool prebuilt_runtime_arena_cache_valid_{false};
    uint64_t prebuilt_runtime_arena_cache_hash_{0};
    std::vector<uint8_t> prebuilt_runtime_arena_cache_key_;
    void *prebuilt_runtime_arena_cache_gm_heap_base_{nullptr};
    void *prebuilt_runtime_arena_cache_sm_base_{nullptr};
    void *prebuilt_runtime_arena_cache_runtime_arena_base_{nullptr};
    size_t prebuilt_runtime_arena_cache_runtime_off_{0};
    std::vector<uint8_t> prebuilt_runtime_arena_cache_image_;

    // Simulation state — written by enqueue/init and read by the AICPU /
    // AICore execute functions via the platform-regs setter functions.
    KernelArgs kernel_args_;

    // Platform-level device wall buffer: 8-byte device-resident slot whose
    // address rides on KernelArgs.device_wall_data_base. AICPU writes the
    // run wall (ns) through that pointer; this DeviceRunner pulls it back
    // via copy_from_device after stream sync and caches it for
    // last_device_wall_ns(). Allocated lazily at enqueue, freed in finalize().
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

    std::atomic<const void *> active_native_run_{nullptr};
    std::atomic<bool> launch_poisoned_{false};

    // Dynamically loaded executor libraries (shared infra; the dlsym'd function-
    // pointer table itself lives on the subclass since signatures diverge
    // per-arch — a2a3 vs a5 differ on aicore_execute and several setters).
    void *aicpu_so_handle_{nullptr};
    void *aicore_so_handle_{nullptr};
    std::string aicpu_so_path_;
    std::string aicore_so_path_;

    // Performance / diagnostics collectors shared across arches.
    ChipSwimlaneCollector chip_swimlane_collector_;
    // Not a collector: the pool the runtime's prepare path writes into, read by
    // whichever per-event views the run enabled. Its two readers are gated
    // independently, so it belongs to neither.
    simpler::dfx::HostPhaseRecordStore host_phase_records_;
    std::unique_ptr<simpler::dfx::ClockCorrelationProvider> clock_correlation_provider_{};
    ArgsDumpCollector dump_collector_;
    PmuCollector pmu_collector_;
    ScopeStatsCollector scope_stats_collector_;

    // Enablement flags. Written before enqueue and read by execution helpers.
    bool enable_chip_swimlane_{false};
    bool enable_dump_args_{false};
    DumpArgsLevel dump_args_level_{DumpArgsLevel::OFF};  // resolved from set_dump_args_enabled()
    bool enable_pmu_{false};
    bool enable_scope_stats_{false};
    ChipSwimlaneLevel chip_swimlane_level_{ChipSwimlaneLevel::DISABLED};  // resolved from set_chip_swimlane_enabled()
    PmuEventType pmu_event_type_{PmuEventType::PIPE_UTILIZATION};         // resolved from set_pmu_enabled()
    std::string output_prefix_{};                                         // diagnostic artifact root directory
};

namespace simpler::common::sim_host {

// Shared utility used by ensure_binaries_loaded() / upload_chip_callable_buffer()
// to materialize an in-memory DSO into /tmp so dlopen can pick it up. mkstemp +
// fchmod 0755 + write_all + close; on success out_path receives the path.
bool create_temp_so_file(const std::string &path_template, const uint8_t *data, size_t size, std::string *out_path);

}  // namespace simpler::common::sim_host
