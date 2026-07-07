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
 * Runtime Class - Device Execution and Handshake Control
 *
 * This class manages device-side execution through AICPU-AICore handshake
 * protocol. Task graph construction is handled by PTO2Runtime; this class
 * only handles:
 * - Handshake buffers for AICPU-AICore communication
 * - Execution parameters (block_dim, aicpu_thread_num)
 * - Tensor pair management for host-device memory tracking
 * - Device orchestration state (gm_sm_ptr_, orch_args_)
 * - Function address mapping (func_id_to_addr_)
 *
 * Task dispatch uses a per-core PTO2DispatchPayload written by the scheduler.
 * At dispatch time, build_payload() copies tensor pointers and scalars from
 * the task payload into the per-core args[], populates SPMD context, then
 * signals AICore via DATA_MAIN_BASE.
 */

#ifndef SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_RUNTIME_H_
#define SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_RUNTIME_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>   // for fprintf, printf
#include <string.h>  // for memset

#include <vector>

#include "common/core_type.h"
#include "common/platform_config.h"
#include "pto2_dispatch_payload.h"
#include "task_args.h"

// =============================================================================
// Configuration Macros
// =============================================================================

#define RUNTIME_MAX_ARGS 128
#define RUNTIME_MAX_WORKER 108  // 36 AIC + 72 AIV cores
// DEBUG crumb slot stride: each worker's aicore_progress slot gets its OWN
// 64-byte cache line (16 uint32). A dense array false-shares — the DIST_CRUMB
// dcci write-back flushes the whole line, so core A's flush clobbers core B's
// freshly-written crumb (observed as garbage progress like [27,23,27,11,...]).
#define AICORE_PROGRESS_STRIDE 16
#define RUNTIME_MAX_FUNC_ID 1024
#define RUNTIME_MAX_ORCH_SO_SIZE (4 * 1024 * 1024)  // 4MB max for orchestration SO
#define RUNTIME_MAX_ORCH_SYMBOL_NAME 64

// Default ready queue shards: one shard per worker thread (total minus orchestrator)
constexpr int RUNTIME_DEFAULT_READY_QUEUE_SHARDS = PLATFORM_MAX_AICPU_THREADS - 1;

// =============================================================================
// Data Structures
// =============================================================================

/**
 * Handshake Structure - Shared between Host, AICPU, and AICore
 *
 * This structure facilitates communication and synchronization between
 * AICPU and AICore during task execution.
 *
 * Protocol State Machine:
 * 1. Initialization: AICPU sets aicpu_ready=1
 * 2. Acknowledgment: AICore sets aicore_done=core_id+1
 * 3. Task Dispatch: AICPU writes DATA_MAIN_BASE after updating the per-core payload
 * 4. Task Execution: AICore reads the cached PTO2DispatchPayload and executes
 * 5. Task Completion: AICore writes FIN to COND; AICPU observes completion
 * 6. Shutdown: AICPU sets control=1, AICore exits
 *
 * Each AICore instance has its own handshake buffer to enable concurrent
 * task execution across multiple cores.
 */

/**
 * Handshake buffer for AICPU-AICore communication
 *
 * Each AICore has its own handshake buffer for synchronization with AICPU.
 * The structure is cache-line aligned (64 bytes) to prevent false sharing
 * between cores and optimize cache coherency operations.
 *
 * Profiling state lives outside this struct: enablement bits and per-core
 * ring/reg addresses travel through `KernelArgs::enable_profiling_flag` +
 * `KernelArgs::aicore_* per-core address arrays`, which the AICore kernel entry
 * forwards into platform-owned per-core slots
 * (`aicore/aicore_profiling_state.h`). Adding a profiling sub-feature does
 * not require touching this struct anymore.
 *
 * Field Access Patterns:
 * - aicpu_ready: Written by AICPU, read by AICore
 * - aicore_done: Written by AICore, read by AICPU
 * - task: Written by AICPU, read by AICore (Init: PTO2DispatchPayload*; runtime: unused)
 * - core_type: Written by AICPU, read by AICore (CoreType::AIC or CoreType::AIV)
 * - physical_core_id: Written by AICore (Phase 2), read by AICPU
 * - aicpu_regs_ready / aicore_regs_ready: handshake sequence flags
 */
struct Handshake {
    volatile uint32_t aicpu_ready;        // AICPU ready signal: 0=not ready, 1=ready
    volatile uint32_t aicore_done;        // AICore ready signal: 0=not ready, core_id+1=ready
    volatile uint64_t task;               // Init: PTO2DispatchPayload* (set before aicpu_ready); runtime: unused
    volatile CoreType core_type;          // Core type: CoreType::AIC or CoreType::AIV
    volatile uint32_t physical_core_id;   // Physical core ID
    volatile uint32_t aicpu_regs_ready;   // AICPU register init done: 0=pending, 1=done
    volatile uint32_t aicore_regs_ready;  // AICore ID reported: 0=pending, 1=done
    // dist_go lives in the FIRST 64-byte cache line (offset 32). The AICPU
    // publishes it with a plain store + cache_flush_range (write-back) of this
    // line — the AICPU->AICore path is NOT coherent on A5, so the flush is
    // REQUIRED to push the store to GM where the AICore's dcci-invalidate poll
    // can see it (plain store + barrier alone only reached 2/9 cores). The
    // AICore polls with dcci(my_hank, SINGLE_CACHE_LINE) — the same primitive
    // the Phase 1 aicpu_ready poll uses. The AICPU sets dist_go AFTER
    // handshake_all_cores has confirmed every worker completed Phase 3, so the
    // Phase 3 dcci write-back (which carries stale dist_go=0) cannot race with
    // the AICPU's dist_go=1 store.
    volatile uint32_t dist_go;            // offset 32: 1 once the distributed engine is wired
    char _pad0[28];                       // pad first line to 64 bytes
    // dist_done lives in the SECOND 64-byte cache line (offset 64), ISOLATED
    // from dist_go's line. The AICPU flushes the first line to publish dist_go;
    // if dist_done shared that line, the AICPU's write-back would clobber the
    // AICore's dist_done=1 with the AICPU's stale dist_done=0. The AICore sets
    // dist_done + dcci write-back of this second line after dist_core_main
    // returns; the AICPU reads it via cache_invalidate_range (invalidate-only,
    // never write-back) — replacing the shared atomicAdd done_count, whose
    // result was not visible to the AICPU on A5.
    volatile uint32_t dist_done;          // offset 64: 1 once this worker's dist_core_main returned
} __attribute__((aligned(64)));

/**
 * Tensor pair for tracking host-device memory mappings.
 * Used for copy-back during finalize.
 */
struct TensorPair {
    void *host_ptr;
    void *dev_ptr;
    size_t size;
    // false for read-only INPUT tensors: they are never written by the kernel,
    // so the end-of-run D2H copy-back is skipped. OUTPUT/INOUT/unknown
    // keep the safe default of copying back.
    bool needs_copy_back = true;
};

/**
 * Host API function pointers for device memory operations.
 * Allows runtime to use pluggable device memory backends.
 */
struct HostApi {
    void *(*device_malloc)(size_t size);
    void (*device_free)(void *dev_ptr);
    int (*copy_to_device)(void *dev_ptr, const void *host_ptr, size_t size);
    int (*copy_from_device)(void *host_ptr, const void *dev_ptr, size_t size);
    // Set a device buffer to a byte value (device-side, no PCIe). Used to
    // zero-init pure OUTPUT buffers in lieu of an H2D copy-in. May be
    // null on backends that don't wire it; callers must fall back to
    // copy_to_device.
    int (*device_memset)(void *dev_ptr, int value, size_t size);
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
    // to device_free or record it in `tensor_pairs_`.
    //
    // acquire_pooled_runtime_arena is trb-only — the runtime-arena region is
    // only committed when setup_static_arena was invoked with
    // runtime_arena_size > 0. Calling it on the hbg path
    // (setup_static_arena(...,0)) returns nullptr (not undefined).
    void *(*acquire_pooled_gm_heap)();
    void *(*acquire_pooled_gm_sm)();
    void *(*acquire_pooled_runtime_arena)();
    // Single-shot upload of the entire ChipCallable buffer. `callable` is a
    // `const ChipCallable *` (declared void* to avoid pulling task_interface
    // headers into runtime.h). DeviceRunner walks child_offsets_ to compute
    // total byte size, allocates device GM once, fixes up each child's
    // resolved_addr_ in an internal host scratch (onboard: device addr; sim:
    // dlopen function pointer), H2D's once, and returns the device-side
    // address of the ChipCallable header. Pool-managed: identical buffer
    // contents (FNV-1a 64-bit) hit the dedup cache; all chip buffers are
    // bulk-freed in DeviceRunner::finalize(). Returns 0 on error or when
    // child_count() == 0. Caller computes child addrs as
    //     chip_dev + offsetof(ChipCallable, storage_) + child_offset(i)
    // and stores them via runtime->set_function_bin_addr(fid, child_dev).
    uint64_t (*upload_chip_callable_buffer)(const void *callable);
};

/**
 * Task structure - Compatibility stub for platform layer
 *
 * RT2 uses PTO2DispatchPayload instead of Task for task dispatch.
 * This stub exists only for API compatibility with device_runner.cpp.
 * Since get_task_count() returns 0, this struct is never actually used.
 */
struct Task {
    int func_id;
    uint64_t function_bin_addr;
};

// Per-core entry point of the fully_distributed_within_core engine. Implemented
// in runtime/dist_engine.cpp (compiled into the AICPU .so), invoked by each
// AICore worker thread via Runtime::dist.core_main_fn. `runtime` is Runtime*,
// `core_type` is CoreType (cast to int to keep this typedef header-light).
// See docs/fully_distributed_within_core.md.
typedef void (*DistCoreMainFn)(void *runtime, int core_idx, int core_type);

// =============================================================================
// Runtime Class
// =============================================================================

/**
 * Runtime class for device execution and handshake control
 *
 * This class manages AICPU-AICore communication through handshake buffers.
 * Task graph construction is handled by PTO2Runtime; this class only handles
 * execution control and device orchestration state.
 */
class Runtime {
public:
    // Handshake buffers for AICPU-AICore communication
    Handshake workers[RUNTIME_MAX_WORKER];  // Worker (AICore) handshake buffers
    int worker_count;                       // Number of active workers

    // Execution parameters for AICPU scheduling.
    //
    // aicpu_thread_num is the *total* AICPU thread count launched on this run
    // (= orch + schedulers). AicpuExecutor splits this into one orchestrator
    // thread (highest idx, runs aicpu_orchestration_entry) and the remaining
    // aicpu_thread_num-1 scheduler threads that dispatch tasks to AICore.
    // The orch thread also dispatches when env PTO2_ORCH_TO_SCHED is set.
    int aicpu_thread_num;
    int ready_queue_shards;  // Number of ready queue shards (1..MAX_AICPU_THREADS, default MAX-1)

    // Filter-style affinity gate input (a5 onboard). Host fills before
    // launch from device-side OCCUPY + DSMI CPU_TOPO via
    // pto::a5::compute_allowed_cpus. The on-device gate keeps threads whose
    // sched_getcpu() lands on one of these cpu_ids; exec_idx = position in
    // this array drives sched/orch role assignment. Indices 0..count-2 are
    // scheduler slots, index count-1 is the orchestrator slot. Sized to
    // PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH for headroom — current
    // policy is 4 sched + 1 orch = 5 active.
    int32_t aicpu_allowed_cpus[16];
    int32_t aicpu_allowed_cpu_count;
    // Actual AICPU thread launch count for this run. Host sets from
    // popcount(OCCUPY) via the topology probe. See the matching field in
    // src/a5/runtime/host_build_graph/runtime/runtime.h for rationale.
    int32_t aicpu_launch_count;

    // PTO2 integration: kernel_id -> GM function_bin_addr mapping
    // NOTE: Made public for direct access from aicore code
    uint64_t func_id_to_addr_[RUNTIME_MAX_FUNC_ID];

    // Sim-only trace-driven replay (CallConfig::use_example_exec_time). Filled by
    // the host from CallConfig at bind time; read by execute_slot in dist_engine:
    // when use_example_exec_time_ is set, a func whose example_exec_time_ns_[fid]
    // is > 0 is "executed" by busy-waiting that many nanoseconds instead of
    // calling the real kernel (funcs left at 0 still run for real). Public for
    // direct AICore-side access, mirroring func_id_to_addr_.
    bool use_example_exec_time_;
    int32_t example_exec_time_ns_[RUNTIME_MAX_FUNC_ID];

    // Orchestrator-to-scheduler transition control
    // When true, orchestrator threads convert to scheduler threads after orchestration completes.
    // When false (default), orchestrator threads exit after orchestration without dispatching tasks.
    // Controlled via PTO2_ORCH_TO_SCHED environment variable.
    bool orch_to_sched;

    // ---- fully_distributed_within_core handoff (SPMD-on-core) ----
    // The AICPU orchestrator thread does dlopen/arena setup, then hands the
    // resolved orchestration entry + per-core engine off to the AICore worker
    // threads through these fields instead of running orchestration/scheduling
    // itself. Each AICore worker invokes core_main_fn(runtime, idx, core_type)
    // once `go` is set, then increments `done_count` when finished. See
    // runtime/dist_engine.* and docs/fully_distributed_within_core.md.
    // Feature flag: this runtime carries the fully_distributed_within_core
    // handoff (Runtime::dist). The shared a5 device_runner.cpp keys off it to
    // compile the fdwc-only shared-segment reserve wiring, since the other a5
    // runtimes (tensormap_and_ringbuffer, host_build_graph) share the same
    // device_runner.cpp but have no Runtime::dist member.
#define RUNTIME_HAS_FDWC_DIST 1
    struct DistHandoff {
        volatile uint64_t core_main_fn;      // DistCoreMainFn (in AICPU .so)
        volatile uint64_t global_data_base;  // base of the shared DistGlobal segment
                                             // (docs §13): allocated + initialized by
                                             // dist_engine_register on the AICPU; each
                                             // AICore worker binds its per-core gd to
                                             // this so all shared state is base+offset
                                             // addressed with NO process-global symbol.
        volatile uint32_t go;                // 1 once engine wired and cores may start
        volatile int32_t num_workers;        // number of AICore workers participating
        volatile int32_t done_count;         // workers atomically increment when done
        // DEBUG: per-worker AICore progress trace. AICore writes crumb i at
        // aicore_progress[i * AICORE_PROGRESS_STRIDE]; AICPU orchestrator reads
        // + logs to localize stalls. Each slot sits on its OWN 64-byte cache
        // line (stride 16 uint32) and the array is 64-byte aligned so a core's
        // dcci write-back never clobbers a neighbor's crumb (false sharing gave
        // garbage values like [27,23,27,11,...]).
        // Each worker owns a 64-byte line (stride 16 u32): slot 0 is the crumb,
        // slots 1..15 are diagnostic scratch (DIST_DBG stashes a pointer's low
        // 32 bits in slot 1, high 32 in slot 2) — same proven-writable line, so
        // no separate array (a separate array write faulted on-core: the device
        // Runtime mapping does not cover fields added past aicore_progress).
        alignas(64) volatile uint32_t aicore_progress[RUNTIME_MAX_WORKER * AICORE_PROGRESS_STRIDE];
        // Host-preallocated shared segment (dist_engine.h DIST_SEG_RESERVE_BYTES).
        // Allocated via host rtMalloc(RT_MEMORY_HBM) so the base is an
        // AICore-MPU-visible device VA (~0x1000…); dist_engine_register carves
        // DistGlobal + heap from it instead of AICPU-side halMemAlloc (which
        // returns an AICore-invisible ~0x3ffa… aperture). 0 => legacy allocator.
        volatile uint64_t seg_base;
        volatile uint64_t seg_size;
    } dist;

private:
    // Kernel binary tracking for cleanup
    int registered_kernel_func_ids_[RUNTIME_MAX_FUNC_ID];
    int registered_kernel_count_;

    void *gm_sm_ptr_;                        // GM pointer to PTO2 shared memory (device)
    void *gm_heap_ptr_;                      // GM heap for orchestrator output buffers (device)
    void *slot_states_ptr_;                  // Pointer to PTO2TaskSlotState array (scheduler-private, for profiling)
    ChipStorageTaskArgs orch_args_storage_;  // Copy of args for device

    // Prebuilt-arena fast path (trb only). Set by the host before rtMemcpy'ing
    // Runtime to device; AICPU reads them in the boot path to skip
    // runtime_create_from_sm and reuse the pooled, prebuilt arena buffer
    // (already populated by runtime_init_data_from_layout + wire on host).
    void *prebuilt_arena_base_;
    size_t prebuilt_runtime_offset_;

    // Device orchestration SO (for dlopen on AICPU thread 3).
    // The SO bytes themselves live in a separately-allocated device buffer
    // owned by DeviceRunner; only the metadata below travels inside Runtime.
    uint64_t dev_orch_so_addr_;
    uint64_t dev_orch_so_size_;
    // Per-callable_id dispatch. AICPU dispatches via
    // `orch_so_table_[active_callable_id_]`; `register_new_callable_id_`
    // signals whether the host is delivering a freshly-registered
    // callable_id (write+dlopen) or reusing an already-loaded one.
    int32_t active_callable_id_;
    bool register_new_callable_id_;
    char device_orch_func_name_[RUNTIME_MAX_ORCH_SYMBOL_NAME];
    char device_orch_config_name_[RUNTIME_MAX_ORCH_SYMBOL_NAME];

public:
    /**
     * Constructor - zero-initialize all arrays
     */
    Runtime();

    // =========================================================================
    // Performance Profiling
    // =========================================================================

    // =========================================================================
    // Device orchestration (for AICPU thread 3)
    // =========================================================================

    void *get_gm_sm_ptr() const;
    void *get_gm_heap_ptr() const;
    const ChipStorageTaskArgs &get_orch_args() const;
    void set_gm_sm_ptr(void *p);
    void set_gm_heap(void *p);
    void set_slot_states_ptr(void *p);
    void set_orch_args(const ChipStorageTaskArgs &args);

    // Prebuilt-arena fast path (trb only). Set by host's
    // bind_callable_to_runtime_impl; consumed by AICPU at boot to attach a
    // DeviceArena to `prebuilt_arena_base_` and pick up the PTO2Runtime at
    // `prebuilt_arena_base_ + prebuilt_runtime_offset_`. Both stay zero on
    // first construction (Runtime() ctor zeros them) so a non-prebuilt boot
    // path can still detect "no prebuilt image set" via nullptr.
    void set_prebuilt_arena(void *arena_base, size_t runtime_off);
    void *get_prebuilt_arena_base() const;
    size_t get_prebuilt_runtime_offset() const;

    // Device orchestration SO binary (for dlopen on AICPU thread 3)
    void set_dev_orch_so(uint64_t dev_addr, uint64_t size);
    uint64_t get_dev_orch_so_addr() const;
    uint64_t get_dev_orch_so_size() const;
    // Per-callable_id dispatch. callable_id must be in
    // [0, MAX_REGISTERED_CALLABLE_IDS); register_new_callable_id_ tells AICPU
    // whether to (re)load the orch SO into orch_so_table_[callable_id] or
    // reuse the cached entry.
    void set_active_callable_id(int32_t callable_id, bool is_new);
    int32_t get_active_callable_id() const;
    bool register_new_callable_id() const;
    void set_device_orch_func_name(const char *name);
    const char *get_device_orch_func_name() const;
    void set_device_orch_config_name(const char *name);
    const char *get_device_orch_config_name() const;

    uint64_t get_function_bin_addr(int func_id) const;
    void set_function_bin_addr(int func_id, uint64_t addr);
    /**
     * Replay a previously-uploaded kernel address onto a fresh Runtime
     * without recording it in registered_kernel_func_ids_. Used by
     * DeviceRunner::bind_callable_to_runtime so prepared kernel
     * binaries are not freed by validate_runtime_impl across runs.
     */
    void replay_function_bin_addr(int func_id, uint64_t addr);

    int get_registered_kernel_count() const;
    int get_registered_kernel_func_id(int index) const;
    void clear_registered_kernels();

    // =========================================================================
    // Deprecated API (for platform compatibility, always returns 0/nullptr)
    // Task graph is now managed by PTO2Runtime, not Runtime
    // =========================================================================

    /** @deprecated Task count is now in PTO2 shared memory */
    int get_task_count() const { return 0; }

    /** @deprecated RT2 uses PTO2DispatchPayload, not Task. Always returns nullptr. */
    Task *get_task(int) { return nullptr; }

    // =========================================================================
    // Host API (host-only, not copied to device)
    // =========================================================================

    // Host API function pointers for device memory operations
    // NOTE: Placed at end of class to avoid affecting device memory layout
    HostApi host_api;

    // Host-side tensor ledger for D2H copy-back at finalize. Populated by
    // runtime_maker.cpp from orch_args at bind time, then iterated in
    // validate_runtime_impl. Not read by AICPU/AICore — the device-side
    // Runtime image carries the std::vector control block as harmless
    // garbage, identical to host_api above. No fixed cap — grows with the
    // chip-level entry-tensor count.
    std::vector<TensorPair> tensor_pairs_;
};

#endif  // SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_RUNTIME_H_
