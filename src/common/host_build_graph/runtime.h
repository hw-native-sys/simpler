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
 * protocol. Task graph construction is handled by RuntimeContext; this class
 * only handles:
 * - Handshake buffers for AICPU-AICore communication
 * - Execution parameters (block_dim, aicpu_thread_num)
 * - simpler::hbg::Tensor lease management for host-device memory tracking
 * - Device orchestration state (gm_sm_ptr_, orch_args_)
 * - Function address mapping (func_id_to_addr_)
 *
 * Task dispatch uses a per-core DispatchPayload written by the scheduler.
 * At dispatch time, build_payload() copies tensor pointers and scalars from
 * the task payload into the per-core args[], populates SPMD context, then
 * signals AICore via DATA_MAIN_BASE.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "common/core_type.h"
#include "common/host_phase_kind.h"
#include "common/platform_config.h"
#include "aicpu/platform_aicpu_affinity.h"  // MAX_GATE_THREADS (aicpu_allowed_cpus bound)
#include "task_args.h"
#include "aicore_teardown.h"
#include "host_build_graph/entry_args.h"  // EntryArgsStorage
#include "utils/tensor_lease.h"

// =============================================================================
// Configuration Macros
// =============================================================================

#define RUNTIME_MAX_WORKER PLATFORM_MAX_CORES
#define RUNTIME_MAX_FUNC_ID 1024

// Default number of ready-queue shards.
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
 * 1. AICore publishes physical_core_id, core_type, and aicore_done on launch
 * 2. AICPU publishes the task pointer and opens the register window with DATA_MAIN_BASE=IDLE
 * 3. AICore observes window-open, reports initial idle state, and reads the task pointer
 * 4. Task Dispatch: AICPU writes DATA_MAIN_BASE after updating the per-core payload
 * 5. Task Execution: AICore reads the cached DispatchPayload and executes
 * 6. Task Completion: AICore writes FIN to COND; AICPU observes completion
 * 7. Shutdown (A2/A3): EXIT -> EXITED -> window-close -> GM release -> AICore returns
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
 * Field Access Patterns:
 * - aicpu_ready: Reserved legacy field; the current handshake does not use it
 * - aicore_done: Written by AICore, read by AICPU (final report; physical_core_id
 *   and core_type are published alongside it in the same write)
 * - task: Written by AICPU before window-open, read by AICore after window-open
 * - core_type: Written by AICore (with aicore_done), read by AICPU (CoreType::AIC or CoreType::AIV)
 * - physical_core_id: Written by AICore (with aicore_done), read by AICPU
 */
struct Handshake {
    volatile uint32_t aicpu_ready;  // Legacy layout field; unused by the current handshake
    volatile uint32_t aicore_done;  // AICore ready signal: 0=not ready, core_id+1=ready
    volatile uint64_t task;         // DispatchPayload* published before register window-open
    volatile CoreType core_type;    // Core type: CoreType::AIC or CoreType::AIV (reported by AICore with aicore_done)
    volatile uint32_t physical_core_id;  // Physical core ID (reported by AICore with aicore_done)
} __attribute__((aligned(64)));

// The AICore owns this line's writeback: it flushes the whole line with
// dcci(..., CACHELINE_OUT) on its report and again on exit. A word the AICPU
// must publish independently cannot live here — a stale line writeback would
// overwrite it. The A2/A3 post-close return gates live in
// Runtime::teardown_gates, one isolated line each; A5 leaves them unused.
static_assert(sizeof(Handshake) == 64);
static_assert(std::is_standard_layout_v<Handshake> && std::is_trivially_copyable_v<Handshake>);

/**
 * Task structure - Compatibility stub for platform layer
 *
 * RT2 uses DispatchPayload instead of Task for task dispatch.
 * This stub exists only for API compatibility with device_runner.cpp.
 * Since get_task_count() returns 0, this struct is never actually used.
 */
struct Task {
    int func_id;
    uint64_t function_bin_addr;
};

// =============================================================================
// Runtime Class
// =============================================================================

/**
 * Runtime class for device execution and handshake control
 *
 * This class manages AICPU-AICore communication through handshake buffers.
 * Task graph construction is handled by RuntimeContext; this class only handles
 * execution control and device orchestration state.
 */
/**
 * DeviceRuntimeLaunchDesc - the device-copied half of Runtime, named.
 *
 * This is the ONLY part of Runtime that crosses the host->device boundary: the
 * host fills it, `device_runner_helpers.cpp` rtMemcpy's exactly
 * `sizeof(DeviceRuntimeLaunchDesc)` bytes from offset 0 of the Runtime image,
 * and the AICPU/AICore read these fields back. It is the first member of
 * Runtime (offsetof == 0), so the narrowed copy needs no offset arithmetic.
 *
 * The boundary was already load-bearing as an offset — everything host-only
 * lives in `Runtime::HostOnlyState`, and the image ended where that member
 * began. Naming the other side makes the image a type rather than a distance:
 * its size is `sizeof`, not `offsetof` on a class that is not standard-layout,
 * and a field lands on the device because of where it is declared rather than
 * because of what it happens to precede. Same split, checked by construction.
 *
 * Keep it standard-layout and trivially copyable (the static_asserts below) so
 * the rtMemcpy is well-defined — Runtime itself is neither, because
 * HostOnlyState holds a std::vector. alignas(64) plus the whole-cache-line size
 * assertion keep cache_invalidate_range(runtime, sizeof(dev)) from rounding into
 * a neighbouring line.
 *
 * Membership here says a field is device-visible. It says nothing about
 * lifetime, and the fields inside do not share one: launch parameters the host
 * rewrites per run sit beside a dispatch table that is constant per callable and
 * beside a handshake region the device itself writes and the host must not
 * overwrite while a run is live. So this being one struct is not licence to
 * overwrite it early, to reset it as a unit, or to conclude that a run must
 * re-upload all of it. Splitting it further belongs to whoever establishes the
 * real read/write relationships, field by field.
 */
struct alignas(64) DeviceRuntimeLaunchDesc {
    // Handshake buffers for AICPU-AICore communication
    Handshake workers[RUNTIME_MAX_WORKER];  // Worker (AICore) handshake buffers
    // A2/A3 post-close return gates, one isolated cache line per worker. The
    // AICPU stores here only after that worker's register window is closed;
    // the AICore bypass-loads its own entry and returns once it reads RELEASE.
    // Separate from workers[] because the AICore flushes its whole Handshake
    // line, which would overwrite a gate sharing it. Unused on A5.
    AicoreTeardownControl teardown_gates[RUNTIME_MAX_WORKER];
    int worker_count;  // Number of active workers

    // Execution parameters for AICPU scheduling.
    //
    // aicpu_thread_num is the total AICPU thread count launched on this run.
    // host_build_graph builds the task graph on the host, so there is no
    // on-device orchestrator: every thread is a scheduler that dispatches tasks
    // to AICore. The highest-index thread additionally performs the one-time
    // host-orch boot (attach SM, latch task count) before it starts dispatching.
    int aicpu_thread_num;
    int ready_queue_shards;  // Number of ready queue shards (1..MAX_AICPU_THREADS, default MAX-1)

    // Filter-style affinity gate input (a2a3 onboard). Host fills these
    // before launch from AICPU OCCUPY, and the device gate keeps threads whose
    // sched_getcpu() lands on one of the cpu_ids. The array position is the
    // deterministic exec_idx used by AicpuExecutor for scheduler-thread
    // assignment; the highest active index additionally runs the host-orch boot.
    int32_t aicpu_allowed_cpus[MAX_GATE_THREADS];
    int32_t aicpu_allowed_cpu_count;
    int32_t aicpu_launch_count;

    // kernel binary resolution: kernel_id -> GM function_bin_addr mapping
    // NOTE: Made public for direct access from aicore code
    uint64_t func_id_to_addr_[RUNTIME_MAX_FUNC_ID];

    // Total tasks the host orchestrator submitted, handed to the scheduler by
    // SchedulerContext::on_graph_attached. host_build_graph builds the whole
    // graph on the host, so this scalar is the count's only carrier: the shared
    // memory header holds no task counter for the boot thread to read.
    int32_t host_total_tasks;

    // Size of the shipped shared-memory image, argument pools included. Set by the
    // host before the image is copied; the AICPU cannot recompute it because the pool
    // extents are the bind's cursors, which only the host saw. It bounds the region at
    // attach and checks the int32 delta reach — it places no segment, since a payload
    // names its argument regions by delta.
    uint64_t sm_image_bytes;

    void *gm_sm_ptr_;  // GM pointer to shared memory (device)

    // Prebuilt-arena fast path. Set by the host before rtMemcpy'ing Runtime to
    // device; AICPU reads them in the boot path to skip runtime_create_from_sm
    // and reuse the pooled, prebuilt arena buffer (already populated by
    // runtime_init_data_from_layout + wire on host).
    void *prebuilt_arena_base_;
    size_t prebuilt_runtime_offset_;
};

static_assert(
    std::is_standard_layout_v<DeviceRuntimeLaunchDesc>,
    "DeviceRuntimeLaunchDesc must be standard-layout: it is rtMemcpy'd to device"
);
static_assert(
    std::is_trivially_copyable_v<DeviceRuntimeLaunchDesc>,
    "DeviceRuntimeLaunchDesc must be trivially copyable: it is rtMemcpy'd to device"
);
static_assert(
    sizeof(DeviceRuntimeLaunchDesc) % 64 == 0,
    "DeviceRuntimeLaunchDesc size must be a multiple of 64 so cache_invalidate_range(sizeof(dev)) "
    "stays cache-line aligned"
);

// =============================================================================
// Runtime Class
// =============================================================================

/**
 * Runtime class for device execution and handshake control
 *
 * This class manages AICPU-AICore communication through handshake buffers.
 * Task graph construction is handled by RuntimeContext; this class only handles
 * execution control and device orchestration state.
 */
class Runtime {
public:
    // The device-copied half. First member, so the copy starts at offset 0.
    DeviceRuntimeLaunchDesc dev;

private:
    // Everything the host keeps to itself. Outside `dev`, so it cannot travel:
    // a new host-only field belongs in here, which is what keeps the boundary
    // from drifting.
    struct HostOnlyState {
        // Entry args, adopted on the host. The host orchestrator is the only
        // reader (runtime_maker.cpp builds its ChipTaskArgs from
        // get_orch_args()); unlike tensormap_and_ringbuffer, no AICPU entry
        // touches them, which is why they can stay off the device entirely.
        simpler::hbg::EntryArgsStorage orch_args_storage_;

        // The callable this runtime is stamped with, and the only orchestration
        // metadata it holds: host_build_graph resolves the orchestration .so and
        // its entry symbols on the host, where the platform's
        // `CallableArtifacts` owns them for the callable's lifetime, so neither
        // the SO bytes nor the symbol names have a home in here.
        int32_t active_callable_id_;

        // Host-side tensor ledger for the run's H2D and D2H transfers.
        // Populated by runtime_maker.cpp from orch_args at bind time, iterated
        // by copy_back_run_outputs_impl and released by
        // release_run_bindings_impl. No fixed cap — grows with the chip-level
        // entry-tensor count, and its std::vector control block holds host heap
        // addresses, which is a second reason this struct cannot travel.
        std::vector<TensorLease> tensor_leases_;

        // Sources for one synchronous publication. Definition and execution-image
        // staging belong to the exclusive slot; scheduler staging is owned here.
        // Destinations remain owned by their existing slot/bank or scheduler owner.
        struct MetadataRegion {
            void *device_target;
            const void *source;
            size_t bytes;
            HostPhaseKind phase;
            std::string attributes;
            std::vector<uint8_t> storage;
        };
        struct RunImagePublication {
            void *device_target;
            const void *source;
            uint64_t bytes;
            uint64_t fanin_elems;
            uint64_t tensor_elems;
            uint64_t scalar_elems;
            std::vector<MetadataRegion> prerequisites;
        };
        RunImagePublication pending_publication_;
    };
    HostOnlyState host_;

public:
    /**
     * Bytes of this object that cross to the device: the device descriptor's
     * size. The AICPU addresses fields inside it directly, so it is also the
     * only length that may be cache-invalidated — reaching past it would touch
     * bytes the host never uploaded.
     */
    static size_t device_image_bytes();

    /**
     * One bind's metadata. A nonzero image byte count seals the record after
     * its prerequisites are prepared. Consumption empties it on success or
     * failure; a partial bind is cleared by the bind's failure guard.
     */
    const HostOnlyState::RunImagePublication &pending_publication() const { return host_.pending_publication_; }
    void set_pending_publication(
        void *device_target, const void *source, uint64_t bytes, uint64_t fanin_elems, uint64_t tensor_elems,
        uint64_t scalar_elems
    ) {
        auto &pending = host_.pending_publication_;
        pending.device_target = device_target;
        pending.source = source;
        pending.bytes = bytes;
        pending.fanin_elems = fanin_elems;
        pending.tensor_elems = tensor_elems;
        pending.scalar_elems = scalar_elems;
    }
    void add_pending_metadata(
        void *target, const void *source, size_t bytes, HostPhaseKind phase, std::string attributes,
        std::vector<uint8_t> storage = {}
    ) {
        host_.pending_publication_.prerequisites.push_back(
            {target, source, bytes, phase, std::move(attributes), std::move(storage)}
        );
    }
    auto take_pending_publication() { return std::exchange(host_.pending_publication_, {}); }
    void clear_pending_publication() { host_.pending_publication_ = {}; }

    /**
     * Constructor - zero-initialize all arrays
     */
    Runtime();

    // =========================================================================
    // Accessors for the execution-parameter fields
    //
    // These exist with identical signatures on the tensormap_and_ringbuffer
    // Runtime so the shared platform layer (device_runner*.cpp, kernel.cpp) can
    // compile against either variant. Both runtimes keep the fields in a `dev`
    // sub-struct; the accessors keep that out of the callers.
    // =========================================================================

    int get_worker_count() const { return dev.worker_count; }
    void set_worker_count(int n) { dev.worker_count = n; }
    int get_aicpu_thread_num() const { return dev.aicpu_thread_num; }
    void set_aicpu_thread_num(int n) { dev.aicpu_thread_num = n; }
    Handshake *get_workers() { return dev.workers; }
    const Handshake *get_workers() const { return dev.workers; }
    AicoreTeardownControl *get_teardown_gates() { return dev.teardown_gates; }
    int32_t get_aicpu_allowed_cpu_count() const { return dev.aicpu_allowed_cpu_count; }
    void set_aicpu_allowed_cpu_count(int32_t n) { dev.aicpu_allowed_cpu_count = n; }
    int32_t get_aicpu_launch_count() const { return dev.aicpu_launch_count; }
    void set_aicpu_launch_count(int32_t n) { dev.aicpu_launch_count = n; }
    int32_t *get_aicpu_allowed_cpus() { return dev.aicpu_allowed_cpus; }
    size_t aicpu_allowed_cpus_capacity() const {
        return sizeof(dev.aicpu_allowed_cpus) / sizeof(dev.aicpu_allowed_cpus[0]);
    }

    // =========================================================================
    // Performance Profiling
    // =========================================================================

    // =========================================================================
    // Shared-memory / orchestration argument plumbing
    // =========================================================================

    void *get_gm_sm_ptr() const;
    const simpler::hbg::EntryArgsStorage &get_orch_args() const;
    void set_gm_sm_ptr(void *p);
    void set_orch_args(const ChipStorageTaskArgs &args);

    // Prebuilt-arena fast path (trb only). Set by host's
    // bind_callable_to_runtime_impl; consumed by AICPU at boot to attach a
    // DeviceArena to `prebuilt_arena_base_` and pick up the RuntimeContext at
    // `prebuilt_arena_base_ + prebuilt_runtime_offset_`. Both stay zero on
    // first construction (Runtime() ctor zeros them) so a non-prebuilt boot
    // path can still detect "no prebuilt image set" via nullptr.
    void set_prebuilt_arena(void *arena_base, size_t runtime_off);
    void *get_prebuilt_arena_base() const;
    size_t get_prebuilt_runtime_offset() const;

    // The callable this runtime is stamped with. The platform stores it once the
    // id is known to be registered, and reads it back on the prepare path to
    // resolve that callable again.
    void set_active_callable_id(int32_t callable_id);
    int32_t get_active_callable_id() const;

    uint64_t get_function_bin_addr(int func_id) const;
    /**
     * Map a func_id onto the device address of its CoreCallable. Used by
     * DeviceRunner::bind_callable_to_runtime for each of the active callable's
     * child kernels, after clear_function_bin_addrs() has emptied the table.
     */
    void replay_function_bin_addr(int func_id, uint64_t addr);

    /**
     * Drop every func_id -> CoreCallable address mapping.
     *
     * Each mapping points into one callable's retained ChipCallable buffer,
     * which unregistering that callable frees. `bind_callable_to_runtime` calls
     * this before replaying the active callable's addresses so no entry outlives
     * the buffer it points into: the scheduler dereferences these addresses and
     * the AICore calls what it finds there.
     */
    void clear_function_bin_addrs();

    // Host-side tensor ledger for the run's H2D and D2H transfers. Populated by
    // runtime_maker.cpp from orch_args at bind time, iterated by
    // copy_back_run_outputs_impl and released by release_run_bindings_impl.
    std::vector<TensorLease> &tensor_leases() { return host_.tensor_leases_; }
    const std::vector<TensorLease> &tensor_leases() const { return host_.tensor_leases_; }

    // =========================================================================
    // Deprecated API (for platform compatibility, always returns 0/nullptr)
    // Task graph is now managed by RuntimeContext, not Runtime
    // =========================================================================

    /** @deprecated Task count is now in shared memory */
    int get_task_count() const { return 0; }

    /** @deprecated RT2 uses DispatchPayload, not Task. Always returns nullptr. */
    Task *get_task(int) { return nullptr; }
};

// `dev` must be the first member so the narrowed H2D copy starts at offset 0,
// and the host-only tail must begin no earlier than the descriptor ends — the
// two together are what keep a field from crossing because of where it was
// declared. Runtime is not standard-layout (std::vector member + mixed access),
// so guard the offsetof against -Winvalid-offsetof; offsetof on such a class is
// conditionally-supported, and both GCC and Clang document that they support it.
// Mirrors the guard tensormap_and_ringbuffer uses for the same assertion.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
#endif
inline size_t Runtime::device_image_bytes() {
    // The image is the descriptor, so its length is a type's size. The two
    // assertions are what keep that true of the object as well: `dev` first, and
    // the host-only tail beginning no earlier than the descriptor ends. Between
    // them a member cannot start travelling, or stop, because of where it was
    // declared relative to something else.
    static_assert(offsetof(Runtime, dev) == 0, "DeviceRuntimeLaunchDesc must be the first member of Runtime");
    static_assert(
        offsetof(Runtime, host_) >= sizeof(DeviceRuntimeLaunchDesc),
        "the host-only tail must start at or after the end of the device image"
    );
    return sizeof(DeviceRuntimeLaunchDesc);
}
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

// Number of bytes of the Runtime image that must be copied to the device. Both
// runtimes return sizeof(DeviceRuntimeLaunchDesc) — their own, which differ in
// content. Defined per-runtime so the shared device_runner_helpers.cpp copy path
// stays runtime-agnostic.
size_t runtime_device_copy_size(const Runtime &rt);
