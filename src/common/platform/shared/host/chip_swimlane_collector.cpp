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
 * @file chip_swimlane_collector.cpp
 * @brief Performance data collector implementation. The mgmt-thread + buffer-pool
 *        machinery lives in profiling_common::BufferPoolManager parameterized by
 *        ChipSwimlaneModule (host/chip_swimlane_collector.h); the poll loop lives in
 *        profiling_common::ProfilerBase. This file owns the per-buffer
 *        on_buffer_collected callback and the export logic.
 */

#include "host/chip_swimlane_collector.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

#include "common/memory_barrier.h"
#include "common/unified_log.h"
#include "host/profiling_copy.h"
#include "host/scheduler_profiling_json.h"
#include "../../../worker/runtime_c_api.h"

#ifndef SIMPLER_RUNTIME_NAME
#error "SIMPLER_RUNTIME_NAME must be defined by RuntimeBuilder"
#endif

// =============================================================================
// ChipSwimlaneCollector Implementation
// =============================================================================

// Sched / orch phase records route through separate BufferKinds; no
// parse-time discriminator function is needed (the device-side type tag is
// the source of truth).

namespace {

std::string linux_boot_clock_domain_id() {
    std::ifstream boot_id_file("/proc/sys/kernel/random/boot_id");
    std::string boot_id;
    if (!(boot_id_file >> boot_id) || boot_id.empty()) return {};
    for (unsigned char ch : boot_id) {
        if (!std::isalnum(ch) && ch != '-') return {};
    }
    return "linux-boot-id:" + boot_id;
}

int owner_recycled_shard_for_core(int core_index, int thread_count) {
    int cluster_index = core_index / PLATFORM_CORES_PER_BLOCKDIM;
    return cluster_index % thread_count;
}

bool recycled_seed_capacity_is_sufficient(
    const char *label, int num_cores, int thread_count, int surplus_per_core, size_t capacity
) {
    if (surplus_per_core <= 0) return true;
    std::array<int, PLATFORM_MAX_AICPU_THREADS> per_shard{};
    for (int core = 0; core < num_cores; core++) {
        per_shard[static_cast<size_t>(owner_recycled_shard_for_core(core, thread_count))] += surplus_per_core;
    }

    bool ok = true;
    for (int shard = 0; shard < thread_count; shard++) {
        if (static_cast<size_t>(per_shard[static_cast<size_t>(shard)]) <= capacity) continue;
        LOG_ERROR(
            "%s recycled seed exceeds lane capacity: shard=%d need=%d capacity=%zu "
            "(num_cores=%d thread_count=%d)",
            label, shard, per_shard[static_cast<size_t>(shard)], capacity, num_cores, thread_count
        );
        ok = false;
    }
    return ok;
}

}  // namespace

ChipSwimlaneCollector::~ChipSwimlaneCollector() {
    stop();
    if (shm_host_ != nullptr) {
        LOG_WARN("ChipSwimlaneCollector destroyed without finalize()");
    }
}

bool ChipSwimlaneCollector::set_json_extension(ChipSwimlaneExtensionSection section, const std::string &json_value) {
    if (!chip_swimlane_extension_has_expected_root(section, json_value)) return false;
    std::string &slot = json_extensions_[static_cast<size_t>(section)];
    if (!slot.empty()) return false;
    slot = json_value;
    return true;
}

int ChipSwimlaneCollector::initialize(
    int num_aicore, int aicpu_thread_num, int device_id, ChipSwimlaneLevel chip_swimlane_level,
    const ChipSwimlaneAllocCallback &alloc_cb, ChipSwimlaneRegisterCallback register_cb,
    const ChipSwimlaneFreeCallback &free_cb
) {
    if (shm_host_ != nullptr) {
        // Already holding this run's device resources. They are not per-run,
        // with one exception: the level decides whether a device orch-phase
        // pool exists, and begin_run() re-publishes the level every run, so a
        // run may ask for a level the pools were not built for.
        return ensure_device_orch_pool(chip_swimlane_level);
    }
    chip_swimlane_level_ = chip_swimlane_level;
    if (num_aicore <= 0 || num_aicore > PLATFORM_MAX_CORES) {
        LOG_ERROR("Invalid number of AICores: %d (max=%d)", num_aicore, PLATFORM_MAX_CORES);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (aicpu_thread_num <= 0 || aicpu_thread_num > PLATFORM_MAX_AICPU_THREADS) {
        LOG_ERROR(
            "Invalid number of AICPU threads: %d (valid range: 1-%d)", aicpu_thread_num, PLATFORM_MAX_AICPU_THREADS
        );
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    // register_cb may legitimately be null on simulation / non-SVM platforms;
    // alloc and free callbacks are mandatory. Matches dep_gen / pmu / scope_stats.
    if (alloc_cb == nullptr || free_cb == nullptr) {
        LOG_ERROR("ChipSwimlaneCollector::initialize: alloc_cb/free_cb must be non-null");
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    LOG_INFO("Initializing performance profiling");

    // Must precede the recycled-lane seeding below: push_recycled() folds its
    // shard argument modulo the manager's shard count.
    set_aicpu_thread_num(aicpu_thread_num);

    num_aicore_ = num_aicore;
    aicpu_thread_num_ = aicpu_thread_num;
    total_perf_collected_ = 0;
    total_sched_phase_collected_ = 0;
    total_orch_phase_collected_ = 0;
    total_aicore_collected_ = 0;
    aicore_skipped_unwritten_ = 0;
    aicore_skipped_overflow_ = 0;
    aicore_skipped_bad_core_ = 0;
    aicore_foreign_identity_ = 0;
    has_phase_data_ = false;
    merged_slot_ = -1;
    json_extensions_.fill({});

    // Stash the memory context on the base up-front so alloc_paired_buffer
    // sees consistent values during init. shm_host_ stays nullptr until the
    // shm allocation succeeds — the nullptr guard makes a post-failure
    // start(tf) a no-op.
    set_memory_context(
        alloc_cb, register_cb, free_cb, profiling_copy_to_device_or_null(), profiling_copy_from_device_or_null(),
        /*shm_dev=*/nullptr, /*shm_host=*/nullptr, /*shm_size=*/0, device_id
    );

    // RAII rollback: shm_host_ is only set at the end of init, so finalize()
    // (which early-returns on shm_host_ == nullptr) cannot clean up a partial
    // allocation. Any early return after this point therefore releases every
    // manager-tracked device buffer + non-SVM host shadow allocated so far via
    // the guard's destructor; guard.commit() disarms it on the success path.
    // Matches dep_gen / pmu.
    profiling_common::InitRollbackGuard<decltype(manager_)> guard(manager_, free_cb);

    // Step 1: Calculate shared memory size (slot arrays only, no actual
    // buffers). Host over-allocates phase pool slots to the platform max for
    // both sched and orch — AICPU picks the actual counts at init_phase time
    // and writes them into the header.
    int num_phase_threads = PLATFORM_MAX_AICPU_THREADS;
    size_t total_size = calc_perf_data_size_with_phases();

    LOG_DEBUG("Shared memory allocation plan:");
    LOG_DEBUG("  Number of cores:      %d", num_aicore);
    LOG_DEBUG("  Header size:          %zu bytes", sizeof(ChipSwimlaneDataHeader));
    LOG_DEBUG("  ChipSwimlaneAicpuTaskPool size: %zu bytes each", sizeof(ChipSwimlaneAicpuTaskPool));
    LOG_DEBUG("  ChipSwimlaneAicpuSchedPhasePool size: %zu bytes each", sizeof(ChipSwimlaneAicpuSchedPhasePool));
    LOG_DEBUG("  ChipSwimlaneAicpuOrchPhasePool size:  %zu bytes each", sizeof(ChipSwimlaneAicpuOrchPhasePool));
    LOG_DEBUG("  Total shared memory:  %zu bytes (%zu KB)", total_size, total_size / 1024);

    // Step 2: Allocate the shared-memory region (header + SPSC slot arrays)
    // via the base allocator. Non-SVM platforms do not expose device HBM as
    // host-addressable memory, so alloc_paired_buffer mallocs a host shadow and
    // seeds the device copy (the shadow path is selected by the copy_to_device
    // callback installed in set_memory_context above). The host initializes the
    // region through perf_host_ptr below, and a single profiling_copy_to_device
    // at the end of init pushes the primed state to the device. Writing
    // perf_host_ptr directly to the raw device pointer there would SIGSEGV —
    // see set_memory_context above.
    void *perf_host_ptr = nullptr;
    void *perf_dev_ptr = alloc_paired_buffer(total_size, &perf_host_ptr);
    if (perf_dev_ptr == nullptr) {
        LOG_ERROR("Failed to allocate shared memory (%zu bytes)", total_size);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    LOG_DEBUG("Allocated shared memory: dev=%p host=%p", perf_dev_ptr, perf_host_ptr);

    // Zero the whole host shadow before initializing individual fields. Don't
    // assume the allocator hands back zeroed memory: the malloc'd-shadow path
    // of alloc_paired_buffer does memset, but the halHostRegister and
    // identity-map paths do not, and neither guarantees the inter-field
    // padding/gaps are clean. A single up-front memset makes the whole region
    // (header, pool states, and all padding) well-defined regardless of which
    // path ran; the explicit field inits below then set the meaningful values,
    // and the end-of-init profiling_copy_to_device pushes the clean region to
    // the device.
    memset(perf_host_ptr, 0, total_size);

    // Step 4: Initialize header
    ChipSwimlaneDataHeader *header = get_chip_swimlane_header(perf_host_ptr);

    for (int t = 0; t < PLATFORM_MAX_AICPU_THREADS; t++) {
        memset(header->queues[t], 0, sizeof(header->queues[t]));
        header->queue_heads[t] = 0;
        header->queue_tails[t] = 0;
    }

    header->num_cores = num_aicore;
    header->chip_swimlane_level = static_cast<uint32_t>(chip_swimlane_level_);
    // Phase metadata: must be zero-initialized here. alloc_cb returns
    // uninitialized device memory; AICPU only writes these fields when
    // phase init runs (level >= SCHED_PHASES). Without zeroing, lower
    // levels (TASK_TIMING / SCHEDULE_TIMING) leave garbage that
    // for_each_instance iterates as `num_sched_phase_threads` /
    // `num_orch_phase_threads`, walking off the end of the allocated pool
    // array → segfault. The host-side reader (read_phase_header_metadata)
    // and BufferPoolManager replenish loop both gate on these counts being
    // sane values.
    header->num_sched_phase_threads = 0;
    header->num_orch_phase_threads = 0;
    header->num_phase_cores = 0;
    memset(header->core_to_thread, -1, sizeof(header->core_to_thread));

    LOG_DEBUG("Initialized ChipSwimlaneDataHeader:");
    LOG_DEBUG("  num_cores:              %d", header->num_cores);
    LOG_DEBUG("  chip_swimlane_level: %u", header->chip_swimlane_level);
    LOG_DEBUG("  buffer_capacity:        %d", PLATFORM_PROF_BUFFER_SIZE);
    LOG_DEBUG("  queue capacity:         %d", PLATFORM_PROF_READYQUEUE_SIZE);

    // Step 5: Initialize ChipSwimlaneAicpuTaskPools. Seed as many buffers as
    // the device-side free_queue can hold; any remaining buffers stay in the
    // host recycled pool.
    constexpr int kAicpuInitialFreeCount = (PLATFORM_PROF_BUFFERS_PER_CORE < PLATFORM_PROF_SLOT_COUNT) ?
                                               PLATFORM_PROF_BUFFERS_PER_CORE :
                                               PLATFORM_PROF_SLOT_COUNT;
    constexpr int kAicpuSurplusPerCore = PLATFORM_PROF_BUFFERS_PER_CORE - kAicpuInitialFreeCount;
    if (!recycled_seed_capacity_is_sufficient(
            "ChipSwimlaneAicpuTask", num_aicore, aicpu_thread_num, kAicpuSurplusPerCore,
            decltype(manager_)::kRecycledQueueCapacity
        )) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    for (int i = 0; i < num_aicore; i++) {
        ChipSwimlaneAicpuTaskPool *state = get_perf_buffer_state(perf_host_ptr, i);
        memset(state, 0, sizeof(ChipSwimlaneAicpuTaskPool));

        state->free_queue.head = 0;
        state->free_queue.tail = 0;
        state->head.current_buf_ptr = 0;
        state->head.current_buf_seq = 0;

        const int initial_free_count = kAicpuInitialFreeCount;
        for (int s = 0; s < PLATFORM_PROF_BUFFERS_PER_CORE; s++) {
            void *host_buf_ptr = nullptr;
            void *dev_buf_ptr = alloc_paired_buffer(sizeof(ChipSwimlaneAicpuTaskBuffer), &host_buf_ptr);
            if (dev_buf_ptr == nullptr) {
                LOG_ERROR("Failed to allocate ChipSwimlaneAicpuTaskBuffer for core %d, buffer %d", i, s);
                return PTO_RUNTIME_ERR_INTERNAL;
            }
            ChipSwimlaneAicpuTaskBuffer *buf = reinterpret_cast<ChipSwimlaneAicpuTaskBuffer *>(host_buf_ptr);
            memset(buf, 0, sizeof(ChipSwimlaneAicpuTaskBuffer));
            buf->count = 0;
            // Part of this kind's paired seed, which is what a session's cap is
            // twice of.
            manager_.note_paired_allocation(
                static_cast<int>(ProfBufferType::AICPU_TASK), sizeof(ChipSwimlaneAicpuTaskBuffer)
            );

            if (s < initial_free_count) {
                state->free_queue.buffer_ptrs[s] = reinterpret_cast<uint64_t>(dev_buf_ptr);
            } else {
                int shard = owner_recycled_shard_for_core(i, aicpu_thread_num);
                int kind = static_cast<int>(ProfBufferType::AICPU_TASK);
                if (!manager_.push_recycled(kind, dev_buf_ptr, shard)) {
                    (void)manager_.retire_unqueued_buffer(kind, dev_buf_ptr, shard);
                }
            }
        }
        wmb();
        state->free_queue.tail = static_cast<uint32_t>(initial_free_count);
        wmb();
    }

    // Step 5b: Initialize ChipSwimlaneAicoreTaskPools — per-core AICore rotation
    // channel + buffer pool. Same SPSC pattern as the AICPU pool above.
    constexpr int kAicoreInitialFreeCount = (PLATFORM_AICORE_BUFFERS_PER_CORE < PLATFORM_PROF_SLOT_COUNT) ?
                                                PLATFORM_AICORE_BUFFERS_PER_CORE :
                                                PLATFORM_PROF_SLOT_COUNT;
    constexpr int kAicoreSurplusPerCore = PLATFORM_AICORE_BUFFERS_PER_CORE - kAicoreInitialFreeCount;
    if (!recycled_seed_capacity_is_sufficient(
            "ChipSwimlaneAicoreTask", num_aicore, aicpu_thread_num, kAicoreSurplusPerCore,
            decltype(manager_)::kRecycledQueueCapacity
        )) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    for (int i = 0; i < num_aicore; i++) {
        ChipSwimlaneAicoreTaskPool *ac_state = get_aicore_buffer_state(perf_host_ptr, i);
        memset(ac_state, 0, sizeof(ChipSwimlaneAicoreTaskPool));

        const int initial_free_count = kAicoreInitialFreeCount;
        for (int s = 0; s < PLATFORM_AICORE_BUFFERS_PER_CORE; s++) {
            void *host_buf_ptr = nullptr;
            void *dev_buf_ptr = alloc_paired_buffer(sizeof(ChipSwimlaneAicoreTaskBuffer), &host_buf_ptr);
            if (dev_buf_ptr == nullptr) {
                LOG_ERROR("Failed to allocate ChipSwimlaneAicoreTaskBuffer for core %d, buffer %d", i, s);
                return PTO_RUNTIME_ERR_INTERNAL;
            }
            ChipSwimlaneAicoreTaskBuffer *buf = reinterpret_cast<ChipSwimlaneAicoreTaskBuffer *>(host_buf_ptr);
            memset(buf, 0, sizeof(ChipSwimlaneAicoreTaskBuffer));
            buf->count = 0;
            manager_.note_paired_allocation(
                static_cast<int>(ProfBufferType::AICORE_TASK), sizeof(ChipSwimlaneAicoreTaskBuffer)
            );

            if (s < initial_free_count) {
                ac_state->free_queue.buffer_ptrs[s] = reinterpret_cast<uint64_t>(dev_buf_ptr);
            } else {
                int shard = owner_recycled_shard_for_core(i, aicpu_thread_num);
                int kind = static_cast<int>(ProfBufferType::AICORE_TASK);
                if (!manager_.push_recycled(kind, dev_buf_ptr, shard)) {
                    (void)manager_.retire_unqueued_buffer(kind, dev_buf_ptr, shard);
                }
            }
        }
        wmb();
        ac_state->free_queue.tail = static_cast<uint32_t>(initial_free_count);
        wmb();
    }
    LOG_DEBUG(
        "Initialized buffer pools: %d ChipSwimlaneAicpuTaskBuffers/core + %d ChipSwimlaneAicoreTaskBuffers/core "
        "(seeded up to PLATFORM_PROF_SLOT_COUNT free_queue slots, rest in recycled pool)",
        PLATFORM_PROF_BUFFERS_PER_CORE, PLATFORM_AICORE_BUFFERS_PER_CORE
    );

    // Step 5c: Standalone uint64_t[num_aicore] table that will hold per-core
    // ChipSwimlaneActiveHead device addresses. Host only allocates the bytes and
    // hands the device pointer to AICPU via KernelArgs::chip_swimlane_aicore_rotation_table;
    // AICPU itself fills the entries inside `chip_swimlane_aicpu_init` (it has
    // direct access to `&ac_state->head` device addresses, no
    // host-to-device translation needed). AICore reads
    // rotation_table[block_idx] at kernel entry.
    // Held in a local and published to aicore_ring_addr_table_dev_ only after
    // guard.commit() (see end of this function). The alloc registers the buffer
    // in the rollback guard, so a later init failure frees it via
    // release_all_owned; assigning the member here would leave it dangling.
    void *rotation_table_dev = nullptr;
    {
        size_t table_bytes = static_cast<size_t>(num_aicore) * sizeof(uint64_t);
        void *rotation_table_host = nullptr;
        rotation_table_dev = alloc_paired_buffer(table_bytes, &rotation_table_host);
        if (rotation_table_dev == nullptr) {
            LOG_ERROR(
                "Failed to allocate chip_swimlane_aicore_rotation_table (rotation) table (%zu bytes)", table_bytes
            );
            return PTO_RUNTIME_ERR_INTERNAL;
        }
    }

    // Step 6: Initialize per-thread phase pools — both sched and orch. Each
    // pool is sized to its own PLATFORM_PROF_{SCHED,ORCH}_BUFFERS_PER_THREAD
    // (up to PLATFORM_PROF_SLOT_COUNT in free_queue, rest in the recycled pool
    // tagged by kind). Templated on the
    // concrete TypedBuffer so the `count` zero-store uses the matching layout
    // — sched and orch buffers have DIFFERENT sizes (64B vs 32B records),
    // so a single cast type for both would land the count store past the end
    // of the orch allocation and corrupt the heap.
    // state_count pool states are zeroed (so the host's [0, PLATFORM_MAX)
    // reconcile/iteration reads count=0 for unused slots); buffers are
    // allocated only for the first buffer_count pools. For sched the two are
    // equal; orch is a single instance (pool 0), so it zeroes all slots but
    // allocates buffers for just pool 0 — no buffers wasted on unused slots.
    auto init_phase_pools = [&](auto *buffer_tag, ChipSwimlaneAicpuTaskPool *(*get_state)(void *, int), int state_count,
                                int buffer_count, int buffers_per_thread, ProfBufferType recycle_kind,
                                const char *kind_label) -> int {
        using Buffer = std::remove_pointer_t<decltype(buffer_tag)>;
        constexpr size_t buffer_bytes = sizeof(Buffer);
        for (int t = 0; t < state_count; t++) {
            auto *state = get_state(perf_host_ptr, t);
            memset(state, 0, sizeof(ChipSwimlaneAicpuTaskPool));
            if (t >= buffer_count) continue;  // zeroed state only; no buffers (unused slot)
            const int initial_free_count =
                (buffers_per_thread < PLATFORM_PROF_SLOT_COUNT) ? buffers_per_thread : PLATFORM_PROF_SLOT_COUNT;
            for (int s = 0; s < buffers_per_thread; s++) {
                void *host_buf_ptr = nullptr;
                void *dev_buf_ptr = alloc_paired_buffer(buffer_bytes, &host_buf_ptr);
                if (dev_buf_ptr == nullptr) {
                    LOG_ERROR("Failed to allocate %s phase buffer for thread %d, slot %d", kind_label, t, s);
                    return PTO_RUNTIME_ERR_INTERNAL;
                }
                // Zero only the `count` word at the buffer's tail, using the
                // matching Buffer type. The records payload is overwritten by
                // AICPU on first use.
                reinterpret_cast<Buffer *>(host_buf_ptr)->count = 0;
                manager_.note_paired_allocation(static_cast<int>(recycle_kind), buffer_bytes);
                if (s < initial_free_count) {
                    state->free_queue.buffer_ptrs[s] = reinterpret_cast<uint64_t>(dev_buf_ptr);
                } else {
                    int shard = t;
                    if (recycle_kind == ProfBufferType::AICPU_ORCH_PHASE) {
                        shard = (aicpu_thread_num > 0) ? (aicpu_thread_num - 1) : 0;
                    }
                    int kind = static_cast<int>(recycle_kind);
                    if (!manager_.push_recycled(kind, dev_buf_ptr, shard)) {
                        (void)manager_.retire_unqueued_buffer(kind, dev_buf_ptr, shard);
                    }
                }
            }
            wmb();
            state->free_queue.tail = static_cast<uint32_t>(initial_free_count);
            wmb();
        }
        return 0;
    };

    // The shm layout spans PLATFORM_MAX_AICPU_THREADS pool states (state_count)
    // because AICPU's pool-array offsets are fixed at that stride, but only the
    // first `aicpu_thread_num` of them ever get a producer — so buffers are
    // allocated for those alone. Seeding a pool at t >= aicpu_thread_num would
    // also push its surplus into recycled lane `t`, which no drain thread owns.
    // Device-orchestrated level 4 uses one orch instance (pool 0). HBG starts
    // its host capture before collector initialize(), so it needs no device
    // orch buffers at all; the fixed pool-state layout is still zeroed.
    if (init_phase_pools(
            static_cast<ChipSwimlaneAicpuSchedPhaseBuffer *>(nullptr), get_sched_phase_buffer_state,
            /*state_count=*/num_phase_threads, /*buffer_count=*/aicpu_thread_num,
            /*buffers_per_thread=*/PLATFORM_PROF_SCHED_BUFFERS_PER_THREAD, ProfBufferType::AICPU_SCHED_PHASE, "sched"
        ) != 0) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    auto orch_get_state = [](void *base, int t) {
        return get_orch_phase_buffer_state(base, t);
    };
    const int orch_buffer_count = chip_swimlane_level_ >= ChipSwimlaneLevel::ORCH_PHASES && !host_orchestrated_ ? 1 : 0;
    if (init_phase_pools(
            static_cast<ChipSwimlaneAicpuOrchPhaseBuffer *>(nullptr), orch_get_state,
            /*state_count=*/num_phase_threads, /*buffer_count=*/orch_buffer_count,
            /*buffers_per_thread=*/PLATFORM_PROF_ORCH_BUFFERS_PER_THREAD, ProfBufferType::AICPU_ORCH_PHASE, "orch"
        ) != 0) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    LOG_DEBUG(
        "Initialized %d sched (%d buf/thread) + %d orch (%d buf/thread) PhaseBufferStates", num_phase_threads,
        PLATFORM_PROF_SCHED_BUFFERS_PER_THREAD, orch_buffer_count, PLATFORM_PROF_ORCH_BUFFERS_PER_THREAD
    );

    wmb();

    // Push the host-initialized region (header + every pool's primed
    // free_queue tail/buffer_ptrs[]) down to the device. perf_host_ptr is a
    // malloc'd shadow distinct from the device HBM region, so without this the
    // device never sees the primed free queues and AICPU/AICore read zeros.
    // The mgmt-loop mirror is read-only (device→host) and never re-pushes this
    // initial state — it must land here, before start(tf) launches mgmt.
    profiling_copy_to_device(perf_dev_ptr, perf_host_ptr, total_size);

    // Step 7: Stash device pointer for the caller to publish via
    // kernel_args.chip_swimlane_data_base (read back via get_chip_swimlane_setup_device_ptr()).
    LOG_DEBUG("chip swimlane device base = 0x%lx", reinterpret_cast<uint64_t>(perf_dev_ptr));

    // Reserve the per-core / per-thread record vectors while the rollback guard
    // is still armed, so a std::bad_alloc here unwinds through the guard and
    // frees every buffer. Publication of the device pointers and the memory
    // context is deferred to after commit (below): otherwise a throw here would
    // leave perf_shared_mem_dev_ dangling and shm_host_ non-null, which would
    // make is_initialized() report true and finalize() double-free.
    reset_collector_shards();

    LOG_INFO("Performance profiling initialized (dynamic buffer mode)");
    guard.commit();
    // Publish device-buffer members + memory context only after the rollback
    // guard is disarmed: on a failed init they stay nullptr / shm_host_ stays
    // null, so is_initialized() is false and finalize() never frees buffers the
    // guard already freed. set_memory_context publishes shm_host_; start(tf)
    // gates on it, so this is the moment the collector becomes startable.
    perf_shared_mem_dev_ = perf_dev_ptr;
    aicore_ring_addr_table_dev_ = rotation_table_dev;
    set_memory_context(
        alloc_cb, register_cb, free_cb, profiling_copy_to_device_or_null(), profiling_copy_from_device_or_null(),
        perf_dev_ptr, perf_host_ptr, total_size, device_id
    );
    return 0;
}

// ---------------------------------------------------------------------------
// ProfilerBase callbacks
// ---------------------------------------------------------------------------

size_t ChipSwimlaneCollector::normalize_collector_shard(int collector_shard) const {
    const size_t shard_count = epoch_stores_[0].counters.size();
    const bool valid_shard = collector_shard >= 0 && static_cast<size_t>(collector_shard) < shard_count;
    if (!valid_shard) {
        assert(false && "collector_shard out of range");
        return shard_count;
    }
    return static_cast<size_t>(collector_shard);
}

int ChipSwimlaneCollector::ensure_device_orch_pool(ChipSwimlaneLevel chip_swimlane_level) {
    if (shm_host_ == nullptr) return 0;
    if (chip_swimlane_level < ChipSwimlaneLevel::ORCH_PHASES || host_orchestrated_) return 0;

    // Device-orchestrated level 4 uses one orch instance (pool 0). A non-zero
    // tail is the host's own seeding mark, and nothing lowers it, so it reads
    // "an earlier run already built this pool".
    ChipSwimlaneAicpuTaskPool *state = get_orch_phase_buffer_state(shm_host_, 0);
    if (state->free_queue.tail != 0) return 0;

    constexpr size_t buffer_bytes = sizeof(ChipSwimlaneAicpuOrchPhaseBuffer);
    constexpr int initial_free_count = (PLATFORM_PROF_ORCH_BUFFERS_PER_THREAD < PLATFORM_PROF_SLOT_COUNT) ?
                                           PLATFORM_PROF_ORCH_BUFFERS_PER_THREAD :
                                           PLATFORM_PROF_SLOT_COUNT;
    // The surplus goes to the lane the orch pool's drain shard owns, matching
    // where initialize() puts it when it builds this pool up front.
    const int shard = (aicpu_thread_num_ > 0) ? (aicpu_thread_num_ - 1) : 0;
    const int kind = static_cast<int>(ProfBufferType::AICPU_ORCH_PHASE);

    for (int s = 0; s < PLATFORM_PROF_ORCH_BUFFERS_PER_THREAD; s++) {
        void *host_buf_ptr = nullptr;
        void *dev_buf_ptr = alloc_paired_buffer(buffer_bytes, &host_buf_ptr);
        if (dev_buf_ptr == nullptr) {
            LOG_ERROR(
                "Failed to allocate orch phase buffer %d while raising the level to %d", s,
                static_cast<int>(chip_swimlane_level)
            );
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        reinterpret_cast<ChipSwimlaneAicpuOrchPhaseBuffer *>(host_buf_ptr)->count = 0;
        if (s < initial_free_count) {
            state->free_queue.buffer_ptrs[s] = reinterpret_cast<uint64_t>(dev_buf_ptr);
        } else if (!manager_.push_recycled(kind, dev_buf_ptr, shard)) {
            (void)manager_.retire_unqueued_buffer(kind, dev_buf_ptr, shard);
        }
    }

    // Slots before tail, and each published on its own: the region-wide push in
    // initialize() is not available here, because on a later run it would also
    // roll back every device-written field the mirror has since advanced.
    wmb();
    publish_field(
        &state->free_queue.buffer_ptrs[0], static_cast<size_t>(initial_free_count) * sizeof(uint64_t),
        "orch free_queue slots"
    );
    state->free_queue.tail = static_cast<uint32_t>(initial_free_count);
    wmb();
    publish_field(&state->free_queue.tail, sizeof(state->free_queue.tail), "orch free_queue tail");

    LOG_INFO("Built the device orch-phase pool on demand for level %d", static_cast<int>(chip_swimlane_level));
    return 0;
}

void ChipSwimlaneCollector::reset_collector_shards() { reset_epoch_store(0, /*reset_merged_view=*/true); }

void ChipSwimlaneCollector::reset_epoch_store(size_t slot, bool reset_merged_view) {
    const size_t shard_count = static_cast<size_t>(manager_.shard_count());

    if (reset_merged_view) {
        collected_perf_records_.assign(num_aicore_, {});
        collected_aicore_records_.assign(num_aicore_, {});
        collected_sched_phase_records_.assign(PLATFORM_MAX_AICPU_THREADS, {});
        collected_orch_phase_records_.assign(PLATFORM_MAX_AICPU_THREADS, {});
    }

    store(slot).perf.assign(shard_count, {});
    store(slot).aicore.assign(shard_count, {});
    store(slot).sched_phase.assign(shard_count, {});
    store(slot).orch_phase.assign(shard_count, {});
    for (size_t shard = 0; shard < shard_count; shard++) {
        store(slot).perf[shard].assign(num_aicore_, {});
        store(slot).aicore[shard].assign(num_aicore_, {});
        store(slot).sched_phase[shard].assign(PLATFORM_MAX_AICPU_THREADS, {});
        store(slot).orch_phase[shard].assign(PLATFORM_MAX_AICPU_THREADS, {});
    }
    store(slot).counters.assign(shard_count, {});
    if (!reset_merged_view) return;
    total_perf_collected_ = 0;
    total_sched_phase_collected_ = 0;
    total_orch_phase_collected_ = 0;
    total_aicore_collected_ = 0;
    aicore_skipped_unwritten_ = 0;
    aicore_skipped_overflow_ = 0;
    aicore_skipped_bad_core_ = 0;
    aicore_foreign_identity_ = 0;
    has_phase_data_ = false;
    for (auto &r : merged_receipt_)
        r = HandoffReceipt{};
    merged_slot_ = -1;
}

template <typename T>
static void merge_record_shards(
    const std::vector<std::vector<std::vector<T>>> &by_collector, std::vector<std::vector<T>> &merged,
    size_t instance_count
) {
    merged.assign(instance_count, {});
    for (size_t instance = 0; instance < instance_count; instance++) {
        size_t total = 0;
        for (const auto &collector_records : by_collector) {
            if (instance < collector_records.size()) {
                total += collector_records[instance].size();
            }
        }
        merged[instance].reserve(total);
        for (const auto &collector_records : by_collector) {
            if (instance < collector_records.size()) {
                const auto &records = collector_records[instance];
                merged[instance].insert(merged[instance].end(), records.begin(), records.end());
            }
        }
    }
}

void ChipSwimlaneCollector::merge_collector_shards() { merge_epoch_store(0); }

void ChipSwimlaneCollector::merge_epoch_store(size_t slot) {
    if (merged_slot_ == static_cast<int>(slot)) {
        return;
    }

    merge_record_shards(store(slot).perf, collected_perf_records_, static_cast<size_t>(num_aicore_));
    merge_record_shards(store(slot).aicore, collected_aicore_records_, static_cast<size_t>(num_aicore_));
    merge_record_shards(
        store(slot).sched_phase, collected_sched_phase_records_, static_cast<size_t>(PLATFORM_MAX_AICPU_THREADS)
    );
    merge_record_shards(
        store(slot).orch_phase, collected_orch_phase_records_, static_cast<size_t>(PLATFORM_MAX_AICPU_THREADS)
    );

    total_perf_collected_ = 0;
    total_sched_phase_collected_ = 0;
    total_orch_phase_collected_ = 0;
    total_aicore_collected_ = 0;
    aicore_skipped_unwritten_ = 0;
    aicore_skipped_overflow_ = 0;
    aicore_skipped_bad_core_ = 0;
    aicore_foreign_identity_ = 0;
    has_phase_data_ = false;
    for (auto &r : merged_receipt_)
        r = HandoffReceipt{};
    merged_presented_buffers_ = 0;
    merged_unroutable_buffers_ = 0;
    for (const auto &counter : store(slot).counters) {
        total_perf_collected_ += counter.total_perf_collected;
        total_sched_phase_collected_ += counter.total_sched_phase_collected;
        total_orch_phase_collected_ += counter.total_orch_phase_collected;
        total_aicore_collected_ += counter.total_aicore_collected;
        aicore_skipped_unwritten_ += counter.aicore_skipped_unwritten;
        aicore_skipped_overflow_ += counter.aicore_skipped_overflow;
        aicore_skipped_bad_core_ += counter.aicore_skipped_bad_core;
        aicore_foreign_identity_ += counter.aicore_foreign_identity;
        has_phase_data_ = has_phase_data_ || counter.has_phase_data;
        merged_presented_buffers_ += counter.buffers_presented;
        merged_unroutable_buffers_ += counter.unroutable_buffers;
        for (size_t k = 0; k < kProducerClasses; k++) {
            HandoffReceipt &dst = merged_receipt_[k];
            const HandoffReceipt &src = counter.receipt[k];
            dst.observed_buffers += src.observed_buffers;
            dst.invalid_index_buffers += src.invalid_index_buffers;
            dst.foreign_epoch_buffers += src.foreign_epoch_buffers;
            dst.malformed_count_buffers += src.malformed_count_buffers;
            dst.received_buffers += src.received_buffers;
            dst.received_records += src.received_records;
        }
    }
    merged_slot_ = static_cast<int>(slot);
}

void ChipSwimlaneCollector::copy_perf_buffer(const ReadyBufferInfo &info, int collector_shard, size_t slot) {
    ChipSwimlaneAicpuTaskBuffer *buf = reinterpret_cast<ChipSwimlaneAicpuTaskBuffer *>(info.host_buffer_ptr);
    rmb();
    uint32_t count = buf->count;
    if (count > PLATFORM_PROF_BUFFER_SIZE) {
        count = PLATFORM_PROF_BUFFER_SIZE;
    }
    uint32_t core_index = info.index;
    size_t shard = normalize_collector_shard(collector_shard);
    if (core_index < static_cast<uint32_t>(num_aicore_) && shard < store(slot).perf.size()) {
        auto &dst = store(slot).perf[shard][core_index];
        if (!session_reserve_records(dst, count, slot)) return;
        for (uint32_t i = 0; i < count; i++) {
            dst.push_back({buf->records[i], buf->run_epoch, buf->local_seq, 0});
        }
        if (!simpler::dfx::session::checked_increment(store(slot).counters[shard].total_perf_collected, count)) {
            session_set_fatal("aicpu-task record counter is out of headroom");
        }
    }
}

void ChipSwimlaneCollector::copy_sched_phase_buffer(const ReadyBufferInfo &info, int collector_shard, size_t slot) {
    auto *buf = reinterpret_cast<ChipSwimlaneAicpuSchedPhaseBuffer *>(info.host_buffer_ptr);
    rmb();
    uint32_t count = buf->count;
    if (count > static_cast<uint32_t>(PLATFORM_PHASE_RECORDS_PER_THREAD)) {
        count = PLATFORM_PHASE_RECORDS_PER_THREAD;
    }
    uint32_t tidx = info.index;
    size_t shard = normalize_collector_shard(collector_shard);
    if (shard < store(slot).sched_phase.size() && tidx < store(slot).sched_phase[shard].size()) {
        auto &dst = store(slot).sched_phase[shard][tidx];
        if (!session_reserve_records(dst, count, slot)) return;
        for (uint32_t i = 0; i < count; i++) {
            dst.push_back({buf->records[i], buf->run_epoch, buf->local_seq, 0});
        }
        if (!simpler::dfx::session::checked_increment(store(slot).counters[shard].total_sched_phase_collected, count)) {
            session_set_fatal("sched-phase record counter is out of headroom");
        }
        if (count > 0) {
            store(slot).counters[shard].has_phase_data = true;
        }
    }
}

void ChipSwimlaneCollector::copy_orch_phase_buffer(const ReadyBufferInfo &info, int collector_shard, size_t slot) {
    auto *buf = reinterpret_cast<ChipSwimlaneAicpuOrchPhaseBuffer *>(info.host_buffer_ptr);
    rmb();
    uint32_t count = buf->count;
    if (count > static_cast<uint32_t>(PLATFORM_PHASE_RECORDS_PER_THREAD)) {
        count = PLATFORM_PHASE_RECORDS_PER_THREAD;
    }
    uint32_t tidx = info.index;
    size_t shard = normalize_collector_shard(collector_shard);
    if (shard < store(slot).orch_phase.size() && tidx < store(slot).orch_phase[shard].size()) {
        auto &dst = store(slot).orch_phase[shard][tidx];
        if (!session_reserve_records(dst, count, slot)) return;
        for (uint32_t i = 0; i < count; i++) {
            dst.push_back({buf->records[i], buf->run_epoch, buf->local_seq, 0});
        }
        if (!simpler::dfx::session::checked_increment(store(slot).counters[shard].total_orch_phase_collected, count)) {
            session_set_fatal("orch-phase record counter is out of headroom");
        }
        if (count > 0) {
            store(slot).counters[shard].has_phase_data = true;
        }
    }
}

// AICore record buffers arrive on the ready queue in per-core rotation order
// (AICPU enqueues them at PLATFORM_AICORE_BUFFER_SIZE dispatch boundaries +
// once at flush). Within a single buffer, AICore wrote records[0..buf->count)
// in the order tasks ran on that core (completion-before-dispatch invariant
// + AICPU stamps buf->count just before enqueue). Records are stored in the
// current collector shard and later merged; downstream consumers join by
// reg_task_id / timestamp and do not require cross-shard arrival order.
//
// Defensive filter: skip records whose `start_time == 0`. AICore writes
// `get_sys_cnt_aicore()` (a free-running cycle counter, always non-zero in
// practice) at task end, so a zero start_time means the slot was never
// written by AICore for this session. This handles two edge cases without
// special-casing them:
//   - Recycled buffer where AICore wrote fewer records than the count stamp
//     (e.g., the rare dispatch-boundary race for sub-microsecond kernels
//     where AICore's next record_task fires before AICPU's rotation has
//     propagated). The "missing" slot's previous contents are zero because
//     allocate_single_buffer memsets at allocation.
//   - Flush-path partial buffer whose tail wasn't reached.
void ChipSwimlaneCollector::copy_aicore_buffer(
    const ReadyBufferInfo &info, int collector_shard, size_t slot, uint64_t expected_epoch
) {
    ChipSwimlaneAicoreTaskBuffer *buf = reinterpret_cast<ChipSwimlaneAicoreTaskBuffer *>(info.host_buffer_ptr);
    rmb();
    uint32_t core_index = info.index;
    // A shard index outside `store(slot).counters` is a contract violation, not
    // an accountable loss class: the counters are per-shard and non-atomic, so
    // charging the reduction to any other shard would race that shard's own
    // live collector thread. Report and return without touching one.
    size_t shard = normalize_collector_shard(collector_shard);
    if (shard >= store(slot).counters.size() || shard >= store(slot).aicore.size()) {
        LOG_ERROR(
            "ChipSwimlane: AICore buffer delivered on collector shard %d, outside [0, %zu) — "
            "precondition violated, buffer not accounted",
            collector_shard, store(slot).counters.size()
        );
        return;
    }
    if (core_index >= static_cast<uint32_t>(num_aicore_)) {
        store(slot).counters[shard].aicore_skipped_bad_core += buf->count;
        return;
    }
    uint32_t count = buf->count;
    uint32_t overflow = 0;
    if (count > static_cast<uint32_t>(PLATFORM_AICORE_BUFFER_SIZE)) {
        overflow = count - static_cast<uint32_t>(PLATFORM_AICORE_BUFFER_SIZE);
        count = PLATFORM_AICORE_BUFFER_SIZE;
    }

    // A buffer's records all carry the stamp the producer wrote when it
    // acquired the buffer, so the identity decision is per buffer, and it is
    // made against the epoch that owns the slot this buffer resolved to.
    // `armed_run_epoch_` is a different thing: the newest armed run, which
    // with a session open is the successor of every late predecessor buffer.
    // The caller passes that field itself when no session is open, so the
    // default path compares against the only identity it has.
    const uint64_t buffer_epoch = buf->run_epoch;
    const bool identity_matches = expected_epoch != 0 && buffer_epoch == expected_epoch;

    uint32_t skipped = 0;
    uint32_t accepted = 0;
    auto &dst = store(slot).aicore[shard][core_index];
    if (!session_reserve_records(dst, count, slot)) {
        // The budget declined this epoch's records. The receipt for this buffer
        // is already taken, so the loss is accounted; the epoch's verdict says
        // its content is incomplete.
        return;
    }
    for (uint32_t i = 0; i < count; i++) {
        const ChipSwimlaneAicoreTaskRecord &r = buf->records[i];
        if (r.start_time == 0) {
            skipped++;
            continue;
        }
        // Collected either way, with its own stamp: the artifact keeps every
        // record the device produced, and the export reads these vectors.
        dst.push_back({r, buffer_epoch, buf->local_seq, 0});
        accepted++;
    }

    if (identity_matches) {
        const bool ok =
            simpler::dfx::session::checked_increment(store(slot).counters[shard].total_aicore_collected, accepted) &&
            simpler::dfx::session::checked_increment(store(slot).counters[shard].aicore_skipped_unwritten, skipped) &&
            simpler::dfx::session::checked_increment(store(slot).counters[shard].aicore_skipped_overflow, overflow);
        if (!ok) session_set_fatal("aicore record counter is out of headroom");
    } else {
        // Counted apart from both sides of the conservation check. A record
        // another run produced is not this run's `collected`, and it is not a
        // device drop either — folding it into `host_skipped` would let it
        // balance away as though this run had accounted for it.
        store(slot).counters[shard].aicore_foreign_identity += accepted + skipped + overflow;
    }

    if (skipped > 0) {
        LOG_WARN(
            "Core %u: skipped %u AICore record slot(s) with start_time=0 (race-window write or "
            "recycled-buffer tail). buf seq=%u count=%u",
            core_index, skipped, info.buffer_seq, count
        );
    }
}

// Whether a ready entry's index addresses a producer this run owns. Checked
// before the index reaches any per-producer array.
bool ChipSwimlaneCollector::producer_index_in_range(ProfBufferType type, uint32_t index) const {
    switch (type) {
    case ProfBufferType::AICPU_TASK:
    case ProfBufferType::AICORE_TASK:
        return index < static_cast<uint32_t>(num_aicore_);
    case ProfBufferType::AICPU_SCHED_PHASE:
    case ProfBufferType::AICPU_ORCH_PHASE:
        return index < static_cast<uint32_t>(PLATFORM_MAX_AICPU_THREADS);
    }
    return false;
}

// The stamped identity and raw count, read through the buffer type that
// matches the kind. `count` sits after records[] in TypedBuffer, so a fixed
// cast would read the wrong offset for three of the four kinds.
bool ChipSwimlaneCollector::read_buffer_identity(
    const ReadyBufferInfo &info, uint64_t *epoch_out, uint32_t *count_out, uint32_t *capacity_out
) const {
    if (info.host_buffer_ptr == nullptr) return false;
    rmb();
    switch (info.type) {
    case ProfBufferType::AICPU_TASK: {
        auto *b = reinterpret_cast<ChipSwimlaneAicpuTaskBuffer *>(info.host_buffer_ptr);
        *epoch_out = b->run_epoch;
        *count_out = b->count;
        *capacity_out = PLATFORM_PROF_BUFFER_SIZE;
        return true;
    }
    case ProfBufferType::AICORE_TASK: {
        auto *b = reinterpret_cast<ChipSwimlaneAicoreTaskBuffer *>(info.host_buffer_ptr);
        *epoch_out = b->run_epoch;
        *count_out = b->count;
        *capacity_out = PLATFORM_AICORE_BUFFER_SIZE;
        return true;
    }
    case ProfBufferType::AICPU_SCHED_PHASE: {
        auto *b = reinterpret_cast<ChipSwimlaneAicpuSchedPhaseBuffer *>(info.host_buffer_ptr);
        *epoch_out = b->run_epoch;
        *count_out = b->count;
        *capacity_out = PLATFORM_PHASE_RECORDS_PER_THREAD;
        return true;
    }
    case ProfBufferType::AICPU_ORCH_PHASE: {
        auto *b = reinterpret_cast<ChipSwimlaneAicpuOrchPhaseBuffer *>(info.host_buffer_ptr);
        *epoch_out = b->run_epoch;
        *count_out = b->count;
        *capacity_out = PLATFORM_PHASE_RECORDS_PER_THREAD;
        return true;
    }
    }
    return false;
}

// Transport observation and run-owned receipt, before any copy decision.
//
// Order matters twice. Every buffer the poll loop hands over is counted
// globally before its kind selects a class, so a kind outside the four is
// still a buffer this collector was handed rather than a silent return. Within
// a class, `observed` is bumped before anything is checked, so a buffer the
// host declines for any reason is still accounted somewhere. Validation then
// decides whether this run's producer may also count it as received — an
// invalid index must never be used to address a per-producer figure, and a
// foreign epoch must never discharge this run's handoff.
//
// This is not the whole transport boundary and does not claim to be.
// `ChipSwimlaneModule::resolve_entry` validates each ready entry's kind and
// index before delivery and retires the ones that fail, so those never arrive
// here; `drain_dropped_buffers()` is their tally and reconcile captures it into
// the report as `transport_retired_buffers`. What is counted here is what was
// presented to the collector.
void ChipSwimlaneCollector::note_buffer_observed(
    const ReadyBufferInfo &info, int collector_shard, size_t slot, uint64_t expected_epoch
) {
    const size_t shard = normalize_collector_shard(collector_shard);
    if (shard >= store(slot).counters.size()) return;
    CollectorShardCounters &counters = store(slot).counters[shard];
    counters.buffers_presented++;

    const size_t klass = static_cast<size_t>(info.type);
    if (klass >= kProducerClasses) {
        counters.unroutable_buffers++;
        return;
    }
    HandoffReceipt &r = counters.receipt[klass];
    r.observed_buffers++;

    if (!producer_index_in_range(info.type, info.index)) {
        r.invalid_index_buffers++;
        return;
    }
    uint64_t buffer_epoch = 0;
    uint32_t raw_count = 0;
    uint32_t capacity = 0;
    if (!read_buffer_identity(info, &buffer_epoch, &raw_count, &capacity)) {
        r.invalid_index_buffers++;
        return;
    }
    if (expected_epoch == 0 || buffer_epoch != expected_epoch) {
        r.foreign_epoch_buffers++;
        return;
    }
    r.received_buffers++;
    if (raw_count > capacity) {
        r.malformed_count_buffers++;
        return;
    }
    r.received_records += raw_count;
}

void ChipSwimlaneCollector::on_buffer_collected(const ReadyBufferInfo &info, int collector_shard) {
    size_t slot = 0;
    uint64_t expected_epoch = armed_run_epoch_;
    bool retain = true;
    if (session_active_.load(std::memory_order_acquire)) {
        // Route by the identity the producer stamped on the buffer, using only
        // this shard's private view of the epoch table. A buffer whose epoch is
        // not in that view belongs to a sealed run or to none, and either way it
        // is counted and dropped — never appended, and never able to reopen a
        // bucket the session has already moved.
        uint64_t buffer_epoch = 0;
        uint32_t raw_count = 0;
        uint32_t capacity = 0;
        const size_t shard = normalize_collector_shard(collector_shard);
        if (shard >= shard_views_.size() || !read_buffer_identity(info, &buffer_epoch, &raw_count, &capacity)) {
            session_no_bucket_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        const int resolved = shard_views_[shard].slot_for(buffer_epoch, &retain);
        if (resolved < 0) {
            if (session_epoch_is_tombstoned(buffer_epoch)) {
                session_late_after_seal_.fetch_add(1, std::memory_order_relaxed);
            } else {
                session_unknown_epoch_.fetch_add(1, std::memory_order_relaxed);
            }
            return;
        }
        slot = static_cast<size_t>(resolved);
        expected_epoch = buffer_epoch;
    }

    note_buffer_observed(info, collector_shard, slot, expected_epoch);
    // A budget refusal declines the records, never the transport: the buffer is
    // still received, still counted and still recycled by the caller.
    if (!retain) return;
    switch (info.type) {
    case ProfBufferType::AICPU_TASK:
        copy_perf_buffer(info, collector_shard, slot);
        break;
    case ProfBufferType::AICPU_SCHED_PHASE:
        copy_sched_phase_buffer(info, collector_shard, slot);
        break;
    case ProfBufferType::AICPU_ORCH_PHASE:
        copy_orch_phase_buffer(info, collector_shard, slot);
        break;
    case ProfBufferType::AICORE_TASK:
        copy_aicore_buffer(info, collector_shard, slot, expected_epoch);
        break;
    }
}

// ---------------------------------------------------------------------------
// reconcile_counters / read_phase_header_metadata
// ---------------------------------------------------------------------------
//
// Host never recovers records from device-side current_buf_ptr. Device flush is
// the only data path: a flush failure bumps dropped_record_count and zeroes the
// buffer's count, but the buffer stays the pool's — AICPU consumes the free
// queue and never produces into it, so reuse by the next run's init is its only
// return. Host's job here is purely accounting + sanity check.

void ChipSwimlaneCollector::reconcile_counters() {
    if (shm_host_ == nullptr) {
        return;
    }
    // Captured before the report, which consumes the counter. A buffer the
    // drain path retired never reached `on_buffer_collected`, so it is in
    // neither the per-class receipts nor the presented tally, and the handoff
    // report carries it as its own layer.
    transport_retired_buffers_ = drain_dropped_buffers();
    report_drain_drops();
    merge_collector_shards();

    // Refresh the pool states (current_buf_ptr + total/dropped counters) from
    // device before the sanity loop so leftovers reflect post-stop() device
    // state. Per-buffer contents are pulled individually inside reconcile_one —
    // an un-flushed active buffer was never enqueued, so the mgmt loop's
    // process_entry never copied its contents into the shadow.
    //
    // `mirror_ok` records whether these bytes actually arrived. The existing
    // reconcile logging is unchanged by a failure — it reports whatever the
    // shadow holds, as it always has — but the terminal-snapshot comparison
    // refuses to call a stale mirror agreement.
    live_counters_.mirror_ok = true;
    if (manager_.shared_mem_dev() != nullptr && shm_size_ > 0) {
        int mirror_rc = profiling_copy_from_device(shm_host_, manager_.shared_mem_dev(), shm_size_);
        if (mirror_rc != 0) {
            live_counters_.mirror_ok = false;
            LOG_WARN("ChipSwimlane reconcile: shared-memory mirror refresh failed (rc=%d)", mirror_rc);
        }
    }
    rmb();

    // Two-bucket invariant (post-AICore-as-producer): every commit attempt
    // bumps total_record_count; capacity-driven drops (no free buffer /
    // queue full / flush failure) bump dropped_record_count.
    //   silent_loss = device_total - (collected + dropped)
    // and any non-zero silent loss flags an unaccounted gap on top of the
    // already-classified dropped losses.
    //
    // Sanity sub-check: after stop(), a retained buffer must hold no records.
    // Two outcomes leave `current_buf_ptr` set, and both are legitimate: a run
    // with nothing to publish, and one whose enqueue failed (which charges
    // `dropped` and zeroes `count` first). Either way the count is 0 and the
    // next run's init reuses the buffer in place. A non-zero pointer with a
    // non-zero count is the bug: those records were neither delivered nor
    // charged to `dropped`.
    //
    // This check covers the PERF and PHASE pools. The AICore task pool is not
    // reconciled here at all, so it says nothing about that pool either way.
    // `live_out` is non-null only for the PERF class, whose device sums the
    // terminal-snapshot comparison reads back. Set at the point the sums are
    // complete, before the early return below can skip the logging.
    auto reconcile_one = [&](const char *kind, const char *unit_name, int unit_count, auto get_state,
                             auto read_buf_count, size_t buf_size, uint64_t collected, bool optional,
                             LiveTaskCounters *live_out) {
        int leftover_active = 0;
        for (int i = 0; i < unit_count; i++) {
            ChipSwimlaneAicpuTaskPool *state = get_state(i);
            uint64_t buf_ptr = state->head.current_buf_ptr;
            if (buf_ptr == 0) continue;
            void *host_ptr = manager_.resolve_host_ptr(reinterpret_cast<void *>(buf_ptr));
            if (host_ptr == nullptr) continue;
            // This buffer was never enqueued (it's the still-active head), so
            // process_entry never pulled its contents into the shadow. Refresh
            // it from device before reading count.
            profiling_copy_from_device(host_ptr, reinterpret_cast<void *>(buf_ptr), buf_size);
            uint32_t count = read_buf_count(host_ptr);
            if (count == 0) continue;
            LOG_ERROR(
                "ChipSwimlane reconcile: %s %d has un-flushed %s buffer (current_buf_ptr=0x%lx, count=%u) "
                "after stop() — device flush failed",
                unit_name, i, kind, static_cast<unsigned long>(buf_ptr), count
            );
            leftover_active++;
        }

        uint64_t total_device = 0;
        uint64_t dropped_device = 0;
        for (int i = 0; i < unit_count; i++) {
            ChipSwimlaneAicpuTaskPool *state = get_state(i);
            total_device += state->head.total_record_count;
            dropped_device += state->head.dropped_record_count;
        }
        if (live_out != nullptr) {
            live_out->aicpu_task_total = total_device;
            live_out->aicpu_task_dropped = dropped_device;
            live_out->live_ok = true;
        }

        // PHASE counters are populated only by runtimes that actually emit
        // phase records; skip the comparison entirely when nothing happened.
        if (optional && total_device == 0 && collected == 0 && dropped_device == 0) {
            return;
        }

        if (dropped_device > 0) {
            LOG_WARN(
                "ChipSwimlane reconcile: %lu %s records dropped on device side.",
                static_cast<unsigned long>(dropped_device), kind
            );
        }
        uint64_t accounted = collected + dropped_device;
        if (accounted != total_device) {
            LOG_WARN(
                "ChipSwimlane reconcile: %s count mismatch (collected=%lu + dropped=%lu != "
                "device_total=%lu, silent_loss=%ld)",
                kind, static_cast<unsigned long>(collected), static_cast<unsigned long>(dropped_device),
                static_cast<unsigned long>(total_device), static_cast<long>(total_device) - static_cast<long>(accounted)
            );
        } else {
            LOG_INFO(
                "ChipSwimlane reconcile: %s counts match (collected=%lu, dropped=%lu, device_total=%lu)", kind,
                static_cast<unsigned long>(collected), static_cast<unsigned long>(dropped_device),
                static_cast<unsigned long>(total_device)
            );
        }

        if (leftover_active > 0) {
            LOG_ERROR(
                "ChipSwimlane reconcile: %d %s(s) had un-cleared %s current_buf_ptr — see prior errors",
                leftover_active, unit_name, kind
            );
        }
    };

    reconcile_one(
        "PERF", "core", num_aicore_,
        [this](int core_index) {
            return get_perf_buffer_state(shm_host_, core_index);
        },
        [](void *host_ptr) {
            return reinterpret_cast<ChipSwimlaneAicpuTaskBuffer *>(host_ptr)->count;
        },
        sizeof(ChipSwimlaneAicpuTaskBuffer), total_perf_collected_, /*optional=*/false, &live_counters_
    );

    reconcile_one(
        "SCHED_PHASE", "thread", PLATFORM_MAX_AICPU_THREADS,
        [this](int thread_index) {
            return get_sched_phase_buffer_state(shm_host_, thread_index);
        },
        [](void *host_ptr) {
            return reinterpret_cast<ChipSwimlaneAicpuSchedPhaseBuffer *>(host_ptr)->count;
        },
        sizeof(ChipSwimlaneAicpuSchedPhaseBuffer), total_sched_phase_collected_, /*optional=*/true, nullptr
    );

    reconcile_one(
        "ORCH_PHASE", "thread", PLATFORM_MAX_AICPU_THREADS,
        [this](int thread_index) {
            return get_orch_phase_buffer_state(shm_host_, thread_index);
        },
        [](void *host_ptr) {
            return reinterpret_cast<ChipSwimlaneAicpuOrchPhaseBuffer *>(host_ptr)->count;
        },
        sizeof(ChipSwimlaneAicpuOrchPhaseBuffer), total_orch_phase_collected_, /*optional=*/true, nullptr
    );

    reconcile_aicore_counters();
}

// The AICore pool has no `reconcile_one` form: that helper's leftover check
// reads the active buffer's settled `count` through a per-buffer
// `profiling_copy_from_device`, and the AICore active buffer has no such count
// — `live_record_count` is the device's running tally, settled into `dropped`
// or `published` by `take_aicore_live_count` at rotation or flush. This
// reconciles from the mirror the caller already refreshed and adds no transfer.
//
//   device_total   — the pool's `total_record_count`, one per dispatch
//   device_dropped — the pool's `dropped_record_count`
//   host_collected — records carrying THIS run's identity that the host accepted
//
// `host_collected` is not "records the device handed over". Slots the host
// declines (unwritten, over-capacity, unowned core, unowned shard) are counted
// separately, and records carrying another run's identity are counted apart
// from both: folding either into `device_dropped` would relabel it as a
// device-side loss, and folding a foreign record into `host_skipped` would let
// it balance away as though this run had accounted for it.
void ChipSwimlaneCollector::reconcile_aicore_counters() {
    if (shm_host_ == nullptr) return;

    // Without a trustworthy mirror the device figures are last run's or
    // uninitialised, and a comparison against them establishes nothing.
    if (!live_counters_.mirror_ok) {
        LOG_INFO("ChipSwimlane reconcile: AICORE accounting unknown — the device mirror did not refresh this run");
        aicore_accounting_.known = false;
        return;
    }

    uint64_t total_device = 0;
    uint64_t dropped_device = 0;
    for (int i = 0; i < num_aicore_; i++) {
        const ChipSwimlaneAicoreTaskPool *state = get_aicore_buffer_state(shm_host_, i);
        total_device += state->head.total_record_count;
        dropped_device += state->head.dropped_record_count;
    }

    const uint64_t host_skipped = aicore_skipped_unwritten_ + aicore_skipped_overflow_ + aicore_skipped_bad_core_;
    aicore_accounting_.known = true;
    // An expected identity has to exist before a population can be said to
    // match it. With none, `identity_ok` is false however few records
    // arrived — zero records under no expected identity is unknown, not a
    // clean run.
    aicore_accounting_.identity_ok = armed_run_epoch_ != 0 && aicore_foreign_identity_ == 0;
    aicore_accounting_.device_total = total_device;
    aicore_accounting_.device_dropped = dropped_device;
    aicore_accounting_.host_collected = total_aicore_collected_;
    aicore_accounting_.host_skipped = host_skipped;
    aicore_accounting_.foreign_identity = aicore_foreign_identity_;

    if (armed_run_epoch_ == 0) {
        // No expected identity: nothing here can be attributed to a run, so the
        // figures are recorded but carry no verdict.
        LOG_WARN(
            "ChipSwimlane reconcile: AICORE accounting has no expected run identity — "
            "not evidence about any run (collected=%lu, device_total=%lu)",
            static_cast<unsigned long>(total_aicore_collected_), static_cast<unsigned long>(total_device)
        );
        return;
    }

    if (total_device == 0 && total_aicore_collected_ == 0 && dropped_device == 0 && host_skipped == 0 &&
        aicore_foreign_identity_ == 0) {
        return;  // nothing happened on this pool; say nothing
    }

    if (aicore_foreign_identity_ > 0) {
        // A record produced under another run reached this host. The
        // conservation figures cover only records carrying this run's identity,
        // so they cannot establish that this run balanced — whatever they sum to.
        LOG_ERROR(
            "ChipSwimlane reconcile: %lu AICORE record(s) carried an identity other than this run's "
            "(expected epoch=%lu) — AICORE accounting is not evidence about this run",
            static_cast<unsigned long>(aicore_foreign_identity_), static_cast<unsigned long>(armed_run_epoch_)
        );
    }

    if (dropped_device > 0) {
        LOG_WARN(
            "ChipSwimlane reconcile: %lu AICORE records dropped on device side.",
            static_cast<unsigned long>(dropped_device)
        );
    }
    if (host_skipped > 0) {
        // Host-side, and reported as such: these were delivered and then
        // declined here, so they are not device drops.
        LOG_WARN(
            "ChipSwimlane reconcile: host declined %lu AICORE record slot(s) "
            "(unwritten=%lu over-capacity=%lu bad-core=%lu)",
            static_cast<unsigned long>(host_skipped), static_cast<unsigned long>(aicore_skipped_unwritten_),
            static_cast<unsigned long>(aicore_skipped_overflow_), static_cast<unsigned long>(aicore_skipped_bad_core_)
        );
    }

    const uint64_t accounted = total_aicore_collected_ + dropped_device + host_skipped;
    if (!aicore_accounting_.identity_ok) {
        // Neither a match nor a mismatch: with part of the population excluded
        // on identity grounds, neither verdict would be about this run.
        LOG_WARN(
            "ChipSwimlane reconcile: AICORE conservation not evaluated (collected=%lu, dropped=%lu, "
            "host_skipped=%lu, foreign=%lu, device_total=%lu)",
            static_cast<unsigned long>(total_aicore_collected_), static_cast<unsigned long>(dropped_device),
            static_cast<unsigned long>(host_skipped), static_cast<unsigned long>(aicore_foreign_identity_),
            static_cast<unsigned long>(total_device)
        );
    } else if (accounted != total_device) {
        LOG_WARN(
            "ChipSwimlane reconcile: AICORE count mismatch (collected=%lu + dropped=%lu + host_skipped=%lu != "
            "device_total=%lu, silent_loss=%ld)",
            static_cast<unsigned long>(total_aicore_collected_), static_cast<unsigned long>(dropped_device),
            static_cast<unsigned long>(host_skipped), static_cast<unsigned long>(total_device),
            static_cast<long>(total_device) - static_cast<long>(accounted)
        );
    } else {
        LOG_INFO(
            "ChipSwimlane reconcile: AICORE counts match (collected=%lu, dropped=%lu, host_skipped=%lu, "
            "device_total=%lu)",
            static_cast<unsigned long>(total_aicore_collected_), static_cast<unsigned long>(dropped_device),
            static_cast<unsigned long>(host_skipped), static_cast<unsigned long>(total_device)
        );
    }
}

// ---------------------------------------------------------------------------
// Retained per-run terminal snapshots
// ---------------------------------------------------------------------------
//
// publish_run_config below zeroes every pool head at each begin_run, so a run's
// record totals do not survive its successor. A producer's last flush copies its
// settled total/dropped into this run's bank, which the host arms per run from
// the run's actual pipeline slot and reads back only after that run's completion
// has been established. The bank is retained storage: nothing clears it between
// runs, so the previous occupant's snapshot stays readable until this run's
// producers overwrite their own entries.

void *ChipSwimlaneCollector::arm_run_terminal_bank(uint32_t bank_index, uint64_t run_epoch) {
    // Any outcome other than a successful arm leaves this collector with no
    // expected identity. Keeping the previous run's would make the next run's
    // records look like they matched, and would make an unarmed collector claim
    // an identity it was never given.
    armed_run_epoch_ = 0;
    if (shm_host_ == nullptr || perf_shared_mem_dev_ == nullptr) return nullptr;
    if (bank_index >= static_cast<uint32_t>(PLATFORM_RUN_TERMINAL_BANKS)) {
        LOG_ERROR(
            "ChipSwimlane terminal: pipeline slot %u exceeds the %d retained banks — no snapshot armed", bank_index,
            PLATFORM_RUN_TERMINAL_BANKS
        );
        return nullptr;
    }
    // Zero is the entries' "no snapshot" state, so it cannot also be a run's
    // identity; a run without one publishes no bank rather than claiming this
    // one.
    if (run_epoch == 0) return nullptr;

    // The run this collector is arming for. On the default path it is also the
    // identity each arriving record's stamp is compared against, so a record
    // produced under another run cannot count toward this one's accounting.
    // Arming precedes `begin_run` in every runner, so this is the only point in
    // the existing host flow where the epoch is in hand before records arrive.
    // With a session open the comparison uses the epoch that owns the buffer's
    // slot instead: this field names the newest armed run, which is not the
    // owner of a predecessor's late buffer.
    armed_run_epoch_ = run_epoch;

    return get_run_terminal_bank(perf_shared_mem_dev_, static_cast<int>(bank_index));
}

ChipSwimlaneCollector::RunTerminalSnapshot
ChipSwimlaneCollector::read_run_terminal_snapshot(uint32_t bank_index, uint64_t run_epoch) {
    RunTerminalSnapshot snapshot;
    snapshot.run_epoch = run_epoch;

    if (shm_host_ == nullptr) return snapshot;
    if (bank_index >= static_cast<uint32_t>(PLATFORM_RUN_TERMINAL_BANKS)) return snapshot;
    if (run_epoch == 0) return snapshot;

    ChipSwimlaneRunTerminal *host_bank = get_run_terminal_bank(shm_host_, static_cast<int>(bank_index));
    // Narrow and checked, rather than relying on the bulk mirror reconcile does:
    // this is the only read of these bytes, and an unchecked copy would turn a
    // failed transfer into a snapshot of whatever the shadow happened to hold.
    //
    // A platform whose host and device share the region installs no copy hook,
    // so `perf_shared_mem_dev_` aliases `shm_host_` and there is nothing to
    // transfer. That is a successful read of bytes already in place, not a
    // skipped one: `transport_ok` is true either way, and it stays false only
    // when a copy was attempted and failed.
    if (perf_shared_mem_dev_ != nullptr && perf_shared_mem_dev_ != shm_host_) {
        ChipSwimlaneRunTerminal *dev_bank = get_run_terminal_bank(perf_shared_mem_dev_, static_cast<int>(bank_index));
        int rc = profiling_copy_from_device(host_bank, dev_bank, calc_run_terminal_bank_size());
        if (rc != 0) {
            LOG_WARN(
                "ChipSwimlane terminal: bank %u copy-from-device failed (rc=%d) — snapshot unknown for epoch %lu",
                bank_index, rc, static_cast<unsigned long>(run_epoch)
            );
            return snapshot;
        }
    }
    snapshot.transport_ok = true;
    rmb();

    auto accumulate = [&](RunTerminalClassSnapshot &cls, int base, int count) {
        for (int i = 0; i < count; i++) {
            const ChipSwimlaneRunTerminal *entry = get_run_terminal(host_bank, base + i);
            uint64_t entry_epoch = entry->run_epoch;
            if (entry_epoch == 0) continue;  // producer never closed into this entry
            if (entry_epoch != run_epoch) {
                snapshot.foreign_entries++;
                continue;
            }
            cls.producers++;
            // Which index reported, not merely how many: a class whose count
            // matches can still have an unexpected index standing in for a
            // missing expected one.
            cls.reported_indices.push_back(i);
            cls.total += entry->total;
            cls.dropped += entry->dropped;
            cls.published_records += entry->published_records;
            cls.published_buffers += entry->published_buffers;
            cls.live_at_close += entry->live_at_close;
            // UINT32_MAX is the device's saturation sentinel: past that point
            // the producer stopped counting rather than wrapping, so no exact
            // figure can be derived from this entry.
            if (entry->total == UINT32_MAX || entry->dropped == UINT32_MAX || entry->published_records == UINT32_MAX ||
                entry->published_buffers == UINT32_MAX) {
                cls.saturated = true;
            }
        }
    };
    accumulate(snapshot.aicpu_task, PLATFORM_RUN_TERMINAL_AICPU_TASK_BASE, PLATFORM_MAX_CORES);
    accumulate(snapshot.aicore_task, PLATFORM_RUN_TERMINAL_AICORE_TASK_BASE, PLATFORM_MAX_CORES);
    accumulate(snapshot.sched_phase, PLATFORM_RUN_TERMINAL_SCHED_PHASE_BASE, PLATFORM_MAX_AICPU_THREADS);
    accumulate(snapshot.orch_phase, PLATFORM_RUN_TERMINAL_ORCH_PHASE_BASE, PLATFORM_MAX_AICPU_THREADS);

    snapshot.valid = snapshot.aicpu_task.producers > 0 || snapshot.aicore_task.producers > 0 ||
                     snapshot.sched_phase.producers > 0 || snapshot.orch_phase.producers > 0;
    return snapshot;
}

namespace {

const char *handoff_verdict_name(ChipSwimlaneCollector::HandoffVerdict v) {
    switch (v) {
    case ChipSwimlaneCollector::HandoffVerdict::Unknown:
        return "unknown";
    case ChipSwimlaneCollector::HandoffVerdict::NotApplicable:
        return "n/a";
    case ChipSwimlaneCollector::HandoffVerdict::Incomplete:
        return "incomplete";
    case ChipSwimlaneCollector::HandoffVerdict::Saturated:
        return "saturated";
    case ChipSwimlaneCollector::HandoffVerdict::Unsettled:
        return "unsettled";
    case ChipSwimlaneCollector::HandoffVerdict::Inconsistent:
        return "inconsistent";
    case ChipSwimlaneCollector::HandoffVerdict::RecordsUntrusted:
        return "records_untrusted";
    case ChipSwimlaneCollector::HandoffVerdict::Shortfall:
        return "shortfall";
    case ChipSwimlaneCollector::HandoffVerdict::Overrun:
        return "overrun";
    case ChipSwimlaneCollector::HandoffVerdict::RecordMismatch:
        return "record_mismatch";
    case ChipSwimlaneCollector::HandoffVerdict::Match:
        return "match";
    }
    return "unknown";
}

const char *handoff_coverage_name(ChipSwimlaneCollector::HandoffCoverage c) {
    switch (c) {
    case ChipSwimlaneCollector::HandoffCoverage::Unknown:
        return "unknown";
    case ChipSwimlaneCollector::HandoffCoverage::NotApplicable:
        return "n/a";
    case ChipSwimlaneCollector::HandoffCoverage::Incomplete:
        return "incomplete";
    case ChipSwimlaneCollector::HandoffCoverage::Complete:
        return "complete";
    }
    return "unknown";
}

const char *run_terminal_verdict_name(ChipSwimlaneCollector::RunTerminalVerdict v) {
    switch (v) {
    case ChipSwimlaneCollector::RunTerminalVerdict::Unknown:
        return "unknown";
    case ChipSwimlaneCollector::RunTerminalVerdict::NotApplicable:
        return "n/a";
    case ChipSwimlaneCollector::RunTerminalVerdict::Unexpected:
        return "unexpected";
    case ChipSwimlaneCollector::RunTerminalVerdict::Partial:
        return "partial";
    case ChipSwimlaneCollector::RunTerminalVerdict::Disagree:
        return "disagree";
    case ChipSwimlaneCollector::RunTerminalVerdict::Agree:
        return "agree";
    }
    return "unknown";
}

}  // namespace

// The expected-index comparison, in one place. `expected` is always a
// host-known denominator; a class without one does not call this.
ChipSwimlaneCollector::TerminalIndexCoverage
ChipSwimlaneCollector::terminal_index_coverage(const RunTerminalClassSnapshot &cls, int expected) {
    TerminalIndexCoverage cov;
    cov.expected = expected;
    cov.reported = cls.producers;
    for (int index : cls.reported_indices) {
        if (index >= expected) cov.unexpected++;
    }
    cov.missing = expected - (cls.producers - cov.unexpected);

    if (cov.unexpected > 0 || cov.missing > 0) {
        cov.state = HandoffCoverage::Incomplete;
    } else if (expected == 0) {
        cov.state = HandoffCoverage::NotApplicable;
    } else {
        cov.state = HandoffCoverage::Complete;
    }
    return cov;
}

ChipSwimlaneCollector::RunTerminalConsistency
ChipSwimlaneCollector::run_terminal_consistency(const RunTerminalSnapshot &snapshot) const {
    RunTerminalConsistency result;

    // The phase classes stay Unknown, which is their default. Their producer
    // counts live in the shared header as untagged device observations that no
    // per-run reset clears, so a successful read cannot tell this run's counts
    // from a previous run's — there is no independent denominator to compare
    // against, and claiming coverage from that would be claiming more than the
    // data supports.

    // A failed transfer means the bytes examined below are not this run's.
    if (!snapshot.transport_ok) return result;

    // The expected index set for both task classes is [0, num_aicore_): the
    // host passed that count to initialize(), so it does not depend on anything
    // the device reports back.
    const int expected = num_aicore_;

    auto classify = [&](const RunTerminalClassSnapshot &cls, bool have_live, uint64_t live_total,
                        uint64_t live_dropped) {
        const TerminalIndexCoverage cov = terminal_index_coverage(cls, expected);
        RunTerminalClassConsistency out;
        out.expected_count = cov.expected;
        out.reported_count = cov.reported;
        out.unexpected_count = cov.unexpected;
        out.missing_count = cov.missing;

        if (out.unexpected_count > 0) {
            // Ranked above a gap and above a sum difference, and reported even
            // when the expected set is empty or the cardinality happens to
            // match: an entry at an index no producer of this run owns says the
            // identity mapping is wrong, which subsumes either of the others.
            out.verdict = RunTerminalVerdict::Unexpected;
            return out;
        }
        if (expected == 0) {
            out.verdict = RunTerminalVerdict::NotApplicable;
            return out;
        }
        if (out.missing_count > 0) {
            out.verdict = RunTerminalVerdict::Partial;
            return out;
        }
        if (!have_live) {
            // Every expected producer reported, but there is nothing sound to
            // compare the sums against.
            out.verdict = RunTerminalVerdict::Unknown;
            return out;
        }
        out.verdict = (cls.total == live_total && cls.dropped == live_dropped) ? RunTerminalVerdict::Agree :
                                                                                 RunTerminalVerdict::Disagree;
        return out;
    };

    const bool live_usable = live_counters_.live_ok && live_counters_.mirror_ok;
    result.aicpu_task =
        classify(snapshot.aicpu_task, live_usable, live_counters_.aicpu_task_total, live_counters_.aicpu_task_dropped);
    // The AICore class compares against the device figures
    // `reconcile_aicore_counters` summed from the same refreshed mirror.
    //
    // With no expected run identity there is nothing to attribute a record to,
    // so the class reports unknown outright rather than a coverage verdict: a
    // snapshot's entries cannot be called present-or-missing for a run this
    // collector was never armed for. An accounting whose population included
    // another run's records is likewise not a live side for this one.
    if (aicore_accounting_.identity_ok) {
        result.aicore_task = classify(
            snapshot.aicore_task, aicore_accounting_.known, aicore_accounting_.device_total,
            aicore_accounting_.device_dropped
        );
    }
    return result;
}

void ChipSwimlaneCollector::report_run_terminal_snapshot(uint32_t bank_index, uint64_t run_epoch) {
    if (shm_host_ == nullptr) return;

    RunTerminalSnapshot snapshot = read_run_terminal_snapshot(bank_index, run_epoch);
    // Retained before the readability branch: an unreadable bank is itself this
    // run's terminal fact, and the default verdict for it is Unknown.
    terminal_snapshot_ = snapshot;
    terminal_consistency_ = RunTerminalConsistency{};
    terminal_reported_ = true;

    // Built before the readability branch too, and for the same reason from the
    // other side: what the transport presented to this host is known whatever
    // the bank says, and it is exactly the run where the bank is unreadable
    // that a reader needs it. `build_handoff_report` keeps every class Unknown
    // when the snapshot did not arrive, so nothing device-side is invented.
    handoff_report_ = build_handoff_report(snapshot, 0);
    if (handoff_report_.unroutable_buffers > 0 || handoff_report_.transport_retired_buffers > 0) {
        LOG_WARN(
            "ChipSwimlane handoff: epoch %lu transport presented %lu buffer(s), %lu of an unroutable kind; a further "
            "%lu were retired by the drain path and never presented",
            static_cast<unsigned long>(run_epoch), static_cast<unsigned long>(handoff_report_.presented_buffers),
            static_cast<unsigned long>(handoff_report_.unroutable_buffers),
            static_cast<unsigned long>(handoff_report_.transport_retired_buffers)
        );
    }

    if (!snapshot.transport_ok) {
        LOG_INFO(
            "ChipSwimlane terminal: bank %u unreadable for epoch %lu — no snapshot and no consistency verdict",
            bank_index, static_cast<unsigned long>(run_epoch)
        );
        return;
    }

    auto log_class = [&](const char *kind, const RunTerminalClassSnapshot &cls) {
        if (cls.producers == 0) return;
        LOG_INFO(
            "ChipSwimlane terminal: epoch %lu %s retained total=%lu dropped=%lu across %d producer(s)",
            static_cast<unsigned long>(run_epoch), kind, static_cast<unsigned long>(cls.total),
            static_cast<unsigned long>(cls.dropped), cls.producers
        );
    };
    log_class("PERF", snapshot.aicpu_task);
    log_class("AICORE", snapshot.aicore_task);
    log_class("SCHED_PHASE", snapshot.sched_phase);
    log_class("ORCH_PHASE", snapshot.orch_phase);

    if (snapshot.foreign_entries > 0) {
        // Expected on a reused bank: entries the previous occupant closed that
        // this run's producers did not overwrite. Counted rather than summed,
        // because they belong to another run's accounting.
        LOG_INFO(
            "ChipSwimlane terminal: bank %u holds %d entr(ies) from an earlier run", bank_index,
            snapshot.foreign_entries
        );
    }

    // Snapshot-vs-live consistency for this run only. It says whether the
    // retained copy matches the live counters reconcile summed; it is not a
    // statement that the run's accounting is complete, that no records were
    // lost, or that the mechanism is sound under overlapping runs.
    RunTerminalConsistency consistency = run_terminal_consistency(snapshot);
    terminal_consistency_ = consistency;
    auto log_verdict = [&](const char *kind, const RunTerminalClassConsistency &c) {
        LOG_INFO(
            "ChipSwimlane terminal: epoch %lu %s snapshot-vs-live %s "
            "(expected=%d reported=%d missing=%d unexpected=%d)",
            static_cast<unsigned long>(run_epoch), kind, run_terminal_verdict_name(c.verdict), c.expected_count,
            c.reported_count, c.missing_count, c.unexpected_count
        );
    };
    log_verdict("PERF", consistency.aicpu_task);
    log_verdict("AICORE", consistency.aicore_task);

    auto log_handoff = [&](const char *kind, const ChipSwimlaneCollector::HandoffClassReport &r) {
        if (r.verdict == HandoffVerdict::NotApplicable) return;
        // A class with no entry, no denominator and no traffic has nothing to
        // state; anything else does, including an incomplete one.
        if (r.verdict == HandoffVerdict::Unknown && r.reported_producers == 0 && r.observed_buffers == 0) return;
        LOG_INFO(
            "ChipSwimlane handoff: epoch %lu %s %s (coverage %s expected=%d reported=%d missing=%d unexpected=%d; "
            "published buffers=%lu records=%lu; received buffers=%lu records=%lu %s; observed=%lu invalid_index=%lu "
            "foreign_epoch=%lu malformed=%lu live_at_close=%lu)",
            static_cast<unsigned long>(run_epoch), kind, handoff_verdict_name(r.verdict),
            handoff_coverage_name(r.coverage), r.expected_producers, r.reported_producers, r.missing_producers,
            r.unexpected_producers, static_cast<unsigned long>(r.published_buffers),
            static_cast<unsigned long>(r.published_records), static_cast<unsigned long>(r.received_buffers),
            static_cast<unsigned long>(r.received_records), r.records_trusted ? "trusted" : "untrusted",
            static_cast<unsigned long>(r.observed_buffers), static_cast<unsigned long>(r.invalid_index_buffers),
            static_cast<unsigned long>(r.foreign_epoch_buffers), static_cast<unsigned long>(r.malformed_count_buffers),
            static_cast<unsigned long>(r.live_at_close)
        );
        if (r.silent_loss_known && r.silent_loss > 0) {
            LOG_WARN(
                "ChipSwimlane handoff: epoch %lu %s %lu record(s) the device never handed over",
                static_cast<unsigned long>(run_epoch), kind, static_cast<unsigned long>(r.silent_loss)
            );
        }
    };
    log_handoff("PERF", handoff_report_.aicpu_task);
    log_handoff("AICORE", handoff_report_.aicore_task);
    log_handoff("SCHED_PHASE", handoff_report_.sched_phase);
    log_handoff("ORCH_PHASE", handoff_report_.orch_phase);
}

// Classify one class. Order matters: every state that makes the inputs
// something other than a closed set is decided before any comparison, so a
// bound is never reported as a count, an unfinished producer is never reported
// as loss, and a partial set of terminal entries is never summed as if it were
// the whole class.
ChipSwimlaneCollector::HandoffClassReport ChipSwimlaneCollector::classify_handoff(
    const RunTerminalClassSnapshot &cls, const HandoffReceipt &receipt, const TerminalIndexCoverage &coverage
) {
    HandoffClassReport out;
    out.published_buffers = cls.published_buffers;
    out.published_records = cls.published_records;
    out.received_buffers = receipt.received_buffers;
    out.received_records = receipt.received_records;
    out.observed_buffers = receipt.observed_buffers;
    out.invalid_index_buffers = receipt.invalid_index_buffers;
    out.foreign_epoch_buffers = receipt.foreign_epoch_buffers;
    out.malformed_count_buffers = receipt.malformed_count_buffers;
    out.live_at_close = cls.live_at_close;
    out.coverage = coverage.state;
    out.expected_producers = coverage.expected;
    out.reported_producers = coverage.reported;
    out.missing_producers = coverage.missing;
    out.unexpected_producers = coverage.unexpected;
    // Retained whatever the verdict turns out to be: record trust is a
    // property of the buffers the host received, not of the comparison.
    out.records_trusted = (receipt.malformed_count_buffers == 0);

    if (coverage.state == HandoffCoverage::NotApplicable) {
        out.verdict = HandoffVerdict::NotApplicable;
        return out;
    }
    if (coverage.state == HandoffCoverage::Unknown && cls.producers == 0) {
        // Absence without a denominator. It may be a class with no producer or
        // a class whose producers all failed to close, and nothing here can
        // tell those apart, so it is not reported as either.
        out.verdict = HandoffVerdict::Unknown;
        return out;
    }
    if (cls.saturated) {
        out.verdict = HandoffVerdict::Saturated;
        return out;
    }
    if (cls.live_at_close != 0) {
        out.verdict = HandoffVerdict::Unsettled;
        return out;
    }
    if (cls.dropped + cls.published_records > cls.total) {
        out.verdict = HandoffVerdict::Inconsistent;
        return out;
    }
    if (coverage.state == HandoffCoverage::Complete) {
        out.silent_loss = cls.total - cls.dropped - cls.published_records;
        out.silent_loss_known = true;
    }

    // A producer that published no terminal entry can only understate
    // `published_*`, so a receipt below it is a shortfall however incomplete
    // the coverage is. An entry at an unexpected index can overstate it, which
    // is why that case carries no comparison at all.
    if (coverage.unexpected == 0 && receipt.received_buffers < cls.published_buffers) {
        out.verdict = HandoffVerdict::Shortfall;
        return out;
    }
    if (coverage.state != HandoffCoverage::Complete) {
        // Equality and overrun are both reachable purely from the missing
        // entries, so neither is evidence of anything about this class.
        out.verdict =
            (coverage.state == HandoffCoverage::Incomplete) ? HandoffVerdict::Incomplete : HandoffVerdict::Unknown;
        return out;
    }
    if (receipt.received_buffers > cls.published_buffers) {
        out.verdict = HandoffVerdict::Overrun;
        return out;
    }
    if (!out.records_trusted) {
        // The buffers line up, but one of them carried an out-of-range count,
        // so `received_records` is not a figure the record comparison below
        // may be run on.
        out.verdict = HandoffVerdict::RecordsUntrusted;
        return out;
    }
    if (receipt.received_records != cls.published_records) {
        // Equal buffer totals say nothing about the records inside them.
        out.verdict = HandoffVerdict::RecordMismatch;
        return out;
    }
    out.verdict = HandoffVerdict::Match;
    return out;
}

ChipSwimlaneCollector::HandoffReport
ChipSwimlaneCollector::build_handoff_report(const RunTerminalSnapshot &snapshot, size_t slot) {
    merge_epoch_store(slot);
    HandoffReport report;
    report.presented_buffers = merged_presented_buffers_;
    report.unroutable_buffers = merged_unroutable_buffers_;
    report.transport_retired_buffers = transport_retired_buffers_;
    if (!snapshot.transport_ok) {
        return report;  // every class stays Unknown; no target to compare against
    }

    // The task classes are judged against the host's own index set; the phase
    // classes have no host-side denominator, so their coverage stays Unknown
    // rather than being invented from what the device happened to report.
    const auto task_coverage = [&](const RunTerminalClassSnapshot &cls) {
        return terminal_index_coverage(cls, num_aicore_);
    };
    const TerminalIndexCoverage phase_coverage{};

    report.aicpu_task = classify_handoff(
        snapshot.aicpu_task, merged_receipt_[static_cast<size_t>(ProfBufferType::AICPU_TASK)],
        task_coverage(snapshot.aicpu_task)
    );
    report.aicore_task = classify_handoff(
        snapshot.aicore_task, merged_receipt_[static_cast<size_t>(ProfBufferType::AICORE_TASK)],
        task_coverage(snapshot.aicore_task)
    );
    report.sched_phase = classify_handoff(
        snapshot.sched_phase, merged_receipt_[static_cast<size_t>(ProfBufferType::AICPU_SCHED_PHASE)], phase_coverage
    );
    report.orch_phase = classify_handoff(
        snapshot.orch_phase, merged_receipt_[static_cast<size_t>(ProfBufferType::AICPU_ORCH_PHASE)], phase_coverage
    );
    return report;
}

void ChipSwimlaneCollector::publish_run_config() {
    // Nothing to publish before the region exists; initialize() writes the level
    // from the member begin_run() just set.
    if (shm_host_ == nullptr) return;

    ChipSwimlaneDataHeader *header = get_chip_swimlane_header(shm_host_);
    header->chip_swimlane_level = static_cast<uint32_t>(chip_swimlane_level_);
    wmb();
    // One field, not the region: a bulk write-back would race the AICPU's own
    // header fields (phase thread counts, core_to_thread) — see
    // buffer_pool_manager.h's note on narrow write_range_to_device calls. On SVM
    // platforms copy_to_device is null and this is a no-op, because the store
    // above already landed in device-visible memory.
    publish_field(&header->chip_swimlane_level, sizeof(header->chip_swimlane_level), "chip_swimlane_level");

    // A level the device orch pool cannot serve produces an empty orch section
    // and nothing else: the run completes, no buffer is lost, and reconcile
    // balances, so there is no other signal that the level did not take effect.
    // initialize() builds the pool on demand for exactly this case, so reaching
    // here unstocked means a caller armed the run without it.
    if (chip_swimlane_level_ >= ChipSwimlaneLevel::ORCH_PHASES && !host_orchestrated_ &&
        get_orch_phase_buffer_state(shm_host_, 0)->free_queue.tail == 0) {
        LOG_ERROR(
            "ChipSwimlane: published level %d with no device orch-phase pool; the device will emit no orchestrator "
            "phases this run",
            static_cast<int>(chip_swimlane_level_)
        );
    }

    // The pools' record counters are producer-side and never reset by the
    // device, so they carry the previous run's totals into this run's reconcile
    // unless cleared here.
    //
    // The four counters are contiguous, so one narrow write covers all of them
    // and leaves the device-owned fields in the same cache line
    // (current_buf_ptr, current_buf_seq) untouched. `live` and `published` have
    // to be cleared with the other two: the accounting identity
    // `published + live + dropped == total` is per run, so clearing only part of
    // it would leave the next run comparing a fresh total against carried-over
    // published records.
    auto reset_head = [this](ChipSwimlaneActiveHead *head) {
        head->total_record_count = 0;
        head->dropped_record_count = 0;
        head->live_record_count = 0;
        head->published_record_count = 0;
        head->published_buffer_count = 0;
        wmb();
        // Contiguity is asserted where the struct is declared, next to the field
        // order it constrains.
        publish_field(&head->total_record_count, 5 * sizeof(uint32_t), "record counters");
    };

    // Every slot, not just this run's: the grid is dimensioned by the platform
    // maximum and a later run may use more cores than the one that dirtied them.
    for (int i = 0; i < PLATFORM_MAX_CORES; i++) {
        reset_head(&get_perf_buffer_state(shm_host_, i)->head);
        reset_head(&get_aicore_buffer_state(shm_host_, i)->head);
    }
    for (int t = 0; t < PLATFORM_MAX_AICPU_THREADS; t++) {
        reset_head(&get_sched_phase_buffer_state(shm_host_, t)->head);
        reset_head(&get_orch_phase_buffer_state(shm_host_, t)->head);
    }
}

void ChipSwimlaneCollector::read_phase_header_metadata() {
    if (shm_host_ == nullptr) {
        return;
    }
    merge_collector_shards();

    // First post-stop() reader of the device-written header (phase thread
    // counts + core_to_thread). Pull the shm region into the shadow so these
    // reads don't depend on the timing of mgmt's final mirror.
    if (manager_.shared_mem_dev() != nullptr && shm_size_ > 0) {
        profiling_copy_from_device(shm_host_, manager_.shared_mem_dev(), shm_size_);
    }
    rmb();

    ChipSwimlaneDataHeader *header = get_chip_swimlane_header(shm_host_);

    int num_sched = static_cast<int>(header->num_sched_phase_threads);
    int num_orch = static_cast<int>(header->num_orch_phase_threads);
    if (num_sched == 0 && num_orch == 0) {
        LOG_INFO("No phase profiling data found (sched/orch phase thread counts both 0; phase init never ran)");
        return;
    }
    if (num_sched > PLATFORM_MAX_AICPU_THREADS || num_orch > PLATFORM_MAX_AICPU_THREADS) {
        LOG_ERROR(
            "Invalid phase thread counts from shared memory (sched=%d, orch=%d, max=%d)", num_sched, num_orch,
            PLATFORM_MAX_AICPU_THREADS
        );
        return;
    }
    // Scheduler threads occupy AICPU threads [0, num_sched); the dedicated
    // orchestrator runs on the last AICPU thread (aicpu_thread_num_ - 1). The
    // orch-phase pool is a single instance, so its pool index does not encode
    // the AICPU thread — derive the thread number from aicpu_thread_num_.
    // aicpu_thread_num_ is >= 1 (device-runner enqueue validates
    // launch_aicpu_num in [1, PLATFORM_MAX_AICPU_THREADS] before initialize()),
    // so the subtraction can't go negative. This is a log-only display value,
    // never an index.
    const int orch_thread = aicpu_thread_num_ - 1;
    LOG_INFO("Collecting phase metadata: scheduler threads 0-%d, orchestrator thread %d", num_sched - 1, orch_thread);

    for (size_t t = 0; t < collected_sched_phase_records_.size(); t++) {
        if (!collected_sched_phase_records_[t].empty()) {
            LOG_INFO("  Sched thread %zu: %zu records", t, collected_sched_phase_records_[t].size());
        }
    }
    for (size_t t = 0; t < collected_orch_phase_records_.size(); t++) {
        if (!collected_orch_phase_records_[t].empty()) {
            LOG_INFO("  Orch thread %d: %zu records", orch_thread, collected_orch_phase_records_[t].size());
        }
    }

    // has_phase_data_ is set by copy_sched_phase_buffer / copy_orch_phase_buffer
    // during the drain — every push goes through those call sites and toggles
    // the flag. No re-scan needed here.

    // Core-to-thread mapping (header-resident; not buffered).
    int num_phase_cores = static_cast<int>(header->num_phase_cores);
    if (num_phase_cores > 0 && num_phase_cores <= PLATFORM_MAX_CORES) {
        core_to_thread_.assign(header->core_to_thread, header->core_to_thread + num_phase_cores);
        LOG_INFO("  Core-to-thread mapping: %d cores", num_phase_cores);
    }

    LOG_INFO("Phase metadata collection complete: has_phase_data=%s", has_phase_data_ ? "yes" : "no");
}

void ChipSwimlaneCollector::set_core_types(const CoreType *types, int n) {
    if (types == nullptr || n <= 0) {
        core_types_.clear();
        return;
    }
    core_types_.assign(types, types + n);
}

void ChipSwimlaneCollector::set_host_phase_records(
    std::vector<HostPhaseRecord> submit_records, std::vector<HostPhaseRecord> upload_records, uint64_t submitted_tasks,
    uint64_t total_records, uint64_t dropped_records
) {
    host_submit_records_ = std::move(submit_records);
    host_upload_records_ = std::move(upload_records);
    host_phase_submitted_tasks_ = submitted_tasks;
    host_phase_total_records_ = total_records;
    host_phase_dropped_records_ = dropped_records;
    host_phase_records_present_ = true;
}

ChipSwimlaneCollector::RunExport ChipSwimlaneCollector::seal_run_export() { return seal_epoch_store(0); }

ChipSwimlaneCollector::RunExport ChipSwimlaneCollector::seal_epoch_store(size_t slot) {
    RunExport data;

    merge_epoch_store(slot);

    data.output_prefix = output_prefix_;
    data.level = chip_swimlane_level_;
    data.json_extensions = json_extensions_;

    data.perf_records = std::move(collected_perf_records_);
    data.aicore_records = std::move(collected_aicore_records_);
    data.sched_phase_records = std::move(collected_sched_phase_records_);
    data.orch_phase_records = std::move(collected_orch_phase_records_);
    // A moved-from vector's state is valid but unspecified. Clearing makes the
    // collector's own view of this run's records definite: empty.
    collected_perf_records_.clear();
    collected_aicore_records_.clear();
    collected_sched_phase_records_.clear();
    collected_orch_phase_records_.clear();
    // `merge_record_shards` copies rather than moves, so the per-shard vectors
    // still hold a second copy of every record now owned by `data`. Releasing
    // them here is what makes "the collector holds none of this run's records"
    // true, and it needs no precondition the merge above did not already need:
    // both touch these vectors from the calling thread while the collector
    // threads are quiesced. The shard and instance extents stay, so the
    // per-shard accessors remain in range and simply report empty.
    auto release_shard_copies = [](auto &by_collector) {
        for (auto &shard : by_collector) {
            for (auto &instance : shard) {
                instance.clear();
                instance.shrink_to_fit();
            }
        }
    };
    release_shard_copies(store(slot).perf);
    release_shard_copies(store(slot).aicore);
    release_shard_copies(store(slot).sched_phase);
    release_shard_copies(store(slot).orch_phase);

    data.num_aicore = num_aicore_;
    data.core_types = core_types_;
    data.core_to_thread = core_to_thread_;

    data.host_orchestrated = host_orchestrated_;
    data.host_phase_records_present = host_phase_records_present_;
    data.host_submit_records = host_submit_records_;
    data.host_upload_records = host_upload_records_;
    data.host_phase_submitted_tasks = host_phase_submitted_tasks_;
    data.host_phase_total_records = host_phase_total_records_;
    data.host_phase_dropped_records = host_phase_dropped_records_;

    // The header fields the writer used to read mid-serialization. Reading them
    // here is what removes the writer's last dependency on the region.
    data.sched_phase_dropped_records.assign(data.sched_phase_records.size(), 0);
    if (shm_host_ != nullptr) {
        for (size_t t = 0; t < data.sched_phase_records.size(); t++) {
            data.sched_phase_dropped_records[t] =
                get_sched_phase_buffer_state(shm_host_, static_cast<int>(t))->head.dropped_record_count;
        }
        data.num_orch_phase_threads = get_chip_swimlane_header(shm_host_)->num_orch_phase_threads;
    }

    data.total_perf_collected = total_perf_collected_;
    data.total_sched_phase_collected = total_sched_phase_collected_;
    data.total_orch_phase_collected = total_orch_phase_collected_;
    data.total_aicore_collected = total_aicore_collected_;
    data.aicore_accounting = {aicore_accounting_.known,          aicore_accounting_.identity_ok,
                              aicore_accounting_.device_total,   aicore_accounting_.device_dropped,
                              aicore_accounting_.host_collected, aicore_accounting_.host_skipped,
                              aicore_skipped_unwritten_,         aicore_skipped_overflow_,
                              aicore_skipped_bad_core_,          aicore_foreign_identity_};
    data.has_phase_data = has_phase_data_;
    data.armed_run_epoch = armed_run_epoch_;
    data.terminal_reported = terminal_reported_;
    data.terminal_snapshot = terminal_snapshot_;
    data.terminal_consistency = terminal_consistency_;

    return data;
}

int ChipSwimlaneCollector::export_swimlane_json() {
    if (shm_host_ == nullptr) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    return write_swimlane_json(seal_run_export());
}

// JSON v2 emit: the host now dumps raw cycle-domain per-stream records plus
// metadata, and `swimlane_converter.py` performs the join (AICore↔Scheduler on
// reg_task_id, base_time normalization, cycles→µs conversion, sort, core_type
// lookup, func_id resolution against deps.json). Moving the join into Python
// makes the schema easy to evolve without round-tripping through C++ + a
// rebuild, and shrinks this file to a pure dump.
int ChipSwimlaneCollector::write_swimlane_json(const RunExport &data) {
    auto extension = [&data](ChipSwimlaneExtensionSection section) -> const std::string * {
        const std::string &value = data.json_extensions[static_cast<size_t>(section)];
        return value.empty() ? nullptr : &value;
    };
    const std::string *scheduler_extension = extension(ChipSwimlaneExtensionSection::SchedulerRecords);
    const std::string *aicore_tasks_extension = extension(ChipSwimlaneExtensionSection::AicoreTasks);
    const std::string *scheduler_tasks_extension = extension(ChipSwimlaneExtensionSection::SchedulerTasks);
    const std::string *aicpu_lifecycle_extension = extension(ChipSwimlaneExtensionSection::AicpuLifecycleRecords);

    // Every stream is independently useful for DFX. In particular, a legal
    // HBG can contain only host-side dummy/hidden-allocation records and no
    // AICore dispatch at all.
    bool has_any_records = !data.host_submit_records.empty() || !data.host_upload_records.empty() ||
                           std::any_of(data.json_extensions.begin(), data.json_extensions.end(), [](const auto &value) {
                               return !value.empty();
                           });
    for (const auto &core_records : data.perf_records) {
        if (!core_records.empty()) {
            has_any_records = true;
            break;
        }
    }
    if (!has_any_records) {
        for (const auto &ac_records : data.aicore_records) {
            if (!ac_records.empty()) {
                has_any_records = true;
                break;
            }
        }
    }
    auto any_phase_records = [](const auto &per_thread_records) {
        for (const auto &records : per_thread_records) {
            if (!records.empty()) return true;
        }
        return false;
    };
    const bool has_aicpu_orch_phases = any_phase_records(data.orch_phase_records);
    const bool has_aicpu_scheduler_records = any_phase_records(data.sched_phase_records);
    if (scheduler_extension != nullptr && has_aicpu_scheduler_records) {
        LOG_ERROR("Both runtime and AICPU scheduler records are present; refusing ambiguous export");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    const bool has_aicore_tasks = any_phase_records(data.aicore_records);
    const bool has_platform_scheduler_tasks = any_phase_records(data.perf_records);
    if ((aicore_tasks_extension != nullptr && has_aicore_tasks) ||
        (scheduler_tasks_extension != nullptr && has_platform_scheduler_tasks)) {
        LOG_ERROR("Both runtime and platform task records are present; refusing ambiguous export");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    has_any_records = has_any_records || any_phase_records(data.sched_phase_records) || has_aicpu_orch_phases;
    if (!has_any_records && !data.collection.present) {
        LOG_WARN("Warning: No performance data to export.");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (!has_any_records) {
        // A session has already promised this file exists, and its `collection`
        // object is what reports why the run carries no records — a refused
        // budget, an evicted epoch, a run that produced nothing. Refusing to
        // write would turn an accounted emptiness into a missing artifact.
        LOG_WARN(
            "ChipSwimlane session: epoch %lu has no records; publishing its verdict alone",
            static_cast<unsigned long>(data.collection.run_epoch)
        );
    }
    if (has_aicpu_orch_phases && data.host_orchestrated) {
        LOG_ERROR("Both host and AICPU orchestrator records are present; refusing mixed clock-domain export");
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    std::error_code ec;
    std::filesystem::create_directories(data.output_prefix, ec);
    if (ec) {
        LOG_ERROR("Error: Failed to create output directory %s: %s", data.output_prefix.c_str(), ec.message().c_str());
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    // A session names its own file inside the directory it reserved; with the
    // session off this is empty and the legacy name is what it always was.
    std::string filepath =
        data.artifact_path.empty() ? data.output_prefix + "/chip_swimlane_records.json" : data.artifact_path;
    std::ofstream outfile(filepath);
    if (!outfile.is_open()) {
        LOG_ERROR("Error: Failed to open file: %s", filepath.c_str());
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    int chip_swimlane_level = static_cast<int>(data.level);

    outfile << "{\n";
    outfile << "  \"chip_swimlane_level\": " << chip_swimlane_level << ",\n";

    // metadata: everything python needs that isn't in a per-record stream.
    // clock_freq_hz drives the cycles→µs conversion (a2a3 = 50 MHz, a5 =
    // 1 GHz — must come from the host, not be hardcoded in python).
    outfile << "  \"metadata\": {\n";
    // Which runtime minted the records. A task_id carries whichever TaskId layout its
    // runtime uses and nothing in the value says which, so a reader that decodes one
    // has to be told; the name is a compile-time property of this host_runtime.so.
    outfile << "    \"runtime\": \"" << SIMPLER_RUNTIME_NAME << "\",\n";
    if (data.collection.present) {
        // Additive: a reader that does not know this key is unaffected, and the
        // default path never emits it.
        outfile << "    \"collection\": {\n";
        outfile << "      \"run_epoch\": " << data.collection.run_epoch << ",\n";
        outfile << "      \"session_id\": " << data.collection.session_id << ",\n";
        outfile << "      \"processing_complete\": " << (data.collection.processing_complete ? "true" : "false")
                << ",\n";
        outfile << "      \"metadata_complete\": " << (data.collection.metadata_complete ? "true" : "false") << ",\n";
        outfile << "      \"not_received_buffers\": " << data.collection.not_received_buffers << ",\n";
        outfile << "      \"unpublished_loss\": " << data.collection.unpublished_loss << ",\n";
        outfile << "      \"transport_retired\": " << data.collection.transport_retired << ",\n";
        outfile << "      \"cut_failed_queues\": " << data.collection.cut_failed_queues << ",\n";
        outfile << "      \"verdict\": \"" << simpler::dfx::session::verdict_name(data.collection.verdict) << "\"\n";
        outfile << "    },\n";
    }
    outfile << "    \"clock_freq_hz\": " << PLATFORM_PROF_SYS_CNT_FREQ << ",\n";
    outfile << "    \"num_cores\": " << data.num_aicore << ",\n";
    outfile << "    \"core_types\": [";
    for (int i = 0; i < data.num_aicore; i++) {
        CoreType ct = (i < static_cast<int>(data.core_types.size())) ? data.core_types[i] : CoreType::AIV;
        if (i > 0) outfile << ", ";
        outfile << "\"" << ((ct == CoreType::AIC) ? "aic" : "aiv") << "\"";
    }
    outfile << "]";
    if (data.host_phase_records_present) {
        // Earliest of both projections: an upload segment can start before the
        // first submit, and a negative offset from the origin is not renderable.
        uint64_t host_origin_ns = 0;
        for (const auto *population : {&data.host_submit_records, &data.host_upload_records}) {
            for (const auto &record : *population) {
                if (host_origin_ns == 0 || record.start_ns < host_origin_ns) host_origin_ns = record.start_ns;
            }
        }
        outfile << ",\n    \"orchestrator_source\": \"host\"";
        outfile << ",\n    \"orchestrator_clock_domain\": \"host_monotonic_ns\"";
        outfile << ",\n    \"device_clock_domain\": \"device_syscnt_cycles\"";
        // The producer stamps records straight from the host monotonic clock, so
        // a record's resolution is the clock's nanosecond and nothing is
        // quantized away on top of it.
        outfile << ",\n    \"host_timestamp_resolution_ns\": 1";
        outfile << ",\n    \"host_timestamp_quantization_ns\": 0";
        outfile << ",\n    \"host_orchestration_origin_ns\": " << host_origin_ns;
        outfile << ",\n    \"timeline_relation\": \"host_orchestration_precedes_device\"";
        // Completeness is per kind, not per record: the pool holds every timed
        // host operation, of which the task-submitting kinds are the projection
        // this file carries. Comparing the pool's total against total_tasks would
        // count the sub-operations of a submit as if each were a submit.
        const bool host_record_count_matches = data.host_submit_records.size() == data.host_phase_submitted_tasks;
        const bool host_capture_complete = data.host_phase_dropped_records == 0 && host_record_count_matches;
        const char *host_capture_status = host_capture_complete               ? "complete" :
                                          data.host_phase_dropped_records > 0 ? "dropped" :
                                                                                "incomplete";
        outfile << ",\n    \"host_capture\": {\"status\": \"" << host_capture_status
                << "\", \"expected_records\": " << data.host_phase_submitted_tasks
                << ", \"recorded_records\": " << data.host_submit_records.size()
                << ", \"pool_records\": " << data.host_phase_total_records
                << ", \"dropped_records\": " << data.host_phase_dropped_records << ", \"error\": ";
        if (host_capture_complete) {
            outfile << "null}";
        } else if (data.host_phase_dropped_records > 0) {
            outfile << "\"pool_overflow\"}";
        } else {
            outfile << "\"record_count_mismatch\"}";
        }
    }
    if (data.host_phase_records_present) {
        const std::string host_clock_domain_id = linux_boot_clock_domain_id();
        if (!host_clock_domain_id.empty()) {
            outfile << ",\n    \"host_clock_domain_id\": \"" << host_clock_domain_id << "\"";
        }
    }
    if (!data.core_to_thread.empty()) {
        outfile << ",\n    \"core_to_thread\": [";
        for (size_t i = 0; i < data.core_to_thread.size(); i++) {
            if (i > 0) outfile << ", ";
            outfile << static_cast<int>(data.core_to_thread[i]);
        }
        outfile << "]";
    }
    outfile << "\n  },\n";

    // Per-stream raw records. Flat array of tuples — compact at scale (a real
    // PA trace has ~100K records, and per-field JSON keys would dominate the
    // file size). Column order is documented in the schema comment at the top
    // of swimlane_converter.py's v2 reader.
    //
    //   aicore_tasks: [core_id, task_token_raw, reg_task_id, start_cycles, end_cycles, receive_to_start_cycles,
    //                  run_epoch]
    //   scheduler_tasks.records: [core_id, reg_task_id, dispatch_cycles, finish_cycles, run_epoch]
    //
    // `run_epoch` is the trailing column on every per-task row and the join key's
    // first component: reg_task_id restarts at 0 each run, so (core_id,
    // reg_task_id) alone collides across runs sharing one file. A row whose epoch
    // is absent is a pre-identity capture, not epoch 0 — the reader distinguishes
    // by column count, never by value.
    {
        // copy_aicore_buffer already drops r.start_time == 0 slots when
        // collecting from the device side, so no defensive filter here.
        outfile << "  \"" << chip_swimlane_extension_section_name(ChipSwimlaneExtensionSection::AicoreTasks) << "\": ";
        if (aicore_tasks_extension != nullptr) {
            outfile << *aicore_tasks_extension;
        } else {
            outfile << "[";
            bool first = true;
            size_t total = 0;
            for (size_t core_idx = 0; core_idx < data.aicore_records.size(); core_idx++) {
                for (const auto &collected : data.aicore_records[core_idx]) {
                    const ChipSwimlaneAicoreTaskRecord &r = collected.record;
                    if (!first) outfile << ",";
                    outfile << "\n    [" << core_idx << ", " << r.task_token_raw << ", " << r.reg_task_id << ", "
                            << r.start_time << ", " << r.end_time << ", " << r.receive_to_start_cycles << ", "
                            << collected.run_epoch << "]";
                    first = false;
                    total++;
                }
            }
            if (!first) outfile << "\n  ";
            outfile << "]";
            LOG_INFO("  aicore_tasks: %zu records", total);
        }
    }
    if (data.level >= ChipSwimlaneLevel::SCHEDULE_TIMING) {
        outfile << ",\n  \"" << chip_swimlane_extension_section_name(ChipSwimlaneExtensionSection::SchedulerTasks)
                << "\": ";
        if (scheduler_tasks_extension != nullptr) {
            outfile << *scheduler_tasks_extension;
        } else {
            outfile << "{\n    \"producer\": \"aicpu\",\n    \"records\": [";
            bool first = true;
            size_t total = 0;
            for (size_t core_idx = 0; core_idx < data.perf_records.size(); core_idx++) {
                for (const auto &collected : data.perf_records[core_idx]) {
                    const ChipSwimlaneAicpuTaskRecord &r = collected.record;
                    if (!first) outfile << ",";
                    outfile << "\n    [" << core_idx << ", " << r.reg_task_id << ", " << r.dispatch_time << ", "
                            << r.finish_time << ", " << collected.run_epoch << "]";
                    first = false;
                    total++;
                }
            }
            if (!first) outfile << "\n    ";
            outfile << "]\n  }";
            LOG_INFO("  scheduler_tasks: %zu AICPU records", total);
        }
    }

    if (data.level >= ChipSwimlaneLevel::SCHED_PHASES) {
        outfile << ",\n  \"" << chip_swimlane_extension_section_name(ChipSwimlaneExtensionSection::SchedulerRecords)
                << "\": ";
        if (scheduler_extension != nullptr) {
            outfile << *scheduler_extension;
        } else {
            chip_swimlane_write_scheduler_records(outfile, data.sched_phase_records, data.sched_phase_dropped_records);
        }

        if (has_aicpu_orch_phases) {
            size_t orch_lanes = static_cast<size_t>(data.num_orch_phase_threads);
            if (orch_lanes == 0 || orch_lanes > data.orch_phase_records.size()) {
                orch_lanes = data.orch_phase_records.size();
            }
            outfile << ",\n  \"aicpu_orchestrator_phases\": [\n";
            for (size_t t = 0; t < orch_lanes; t++) {
                outfile << "    [";
                bool first = true;
                for (const auto &collected : data.orch_phase_records[t]) {
                    const ChipSwimlaneAicpuOrchPhaseRecord &pr = collected.record;
                    if (!first) outfile << ",";
                    outfile << "\n      {\"submit_idx\": " << pr.submit_idx << ", \"task_id\": " << pr.task_id
                            << ", \"start_cycles\": " << pr.start_time << ", \"end_cycles\": " << pr.end_time
                            << ", \"run_epoch\": " << collected.run_epoch << "}";
                    first = false;
                }
                if (!first) outfile << "\n    ";
                outfile << "]";
                if (t < orch_lanes - 1) outfile << ",";
                outfile << "\n";
            }
            outfile << "  ]";
        }
        if (!data.host_submit_records.empty()) {
            outfile << ",\n  \"host_orchestrator_phases\": [[";
            bool first = true;
            for (const auto &record : data.host_submit_records) {
                if (!first) outfile << ",";
                outfile << "\n      {\"submit_idx\": " << record.index << ", \"task_id\": " << record.payload
                        << ", \"start_host_ns\": " << record.start_ns << ", \"end_host_ns\": " << record.end_ns << "}";
                first = false;
            }
            if (!first) outfile << "\n    ";
            outfile << "]]";
        }
        if (!data.host_upload_records.empty()) {
            outfile << ",\n  \"host_device_uploads\": [";
            bool first = true;
            for (const auto &record : data.host_upload_records) {
                if (!first) outfile << ",";
                outfile << "\n      {\"phase\": \"" << host_phase_kind_name(static_cast<HostPhaseKind>(record.kind))
                        << "\", \"start_host_ns\": " << record.start_ns << ", \"end_host_ns\": " << record.end_ns
                        << ", \"detail\": " << record.payload << "}";
                first = false;
            }
            if (!first) outfile << "\n    ";
            outfile << "]";
        }
    }

    if (aicpu_lifecycle_extension != nullptr) {
        outfile << ",\n  \""
                << chip_swimlane_extension_section_name(ChipSwimlaneExtensionSection::AicpuLifecycleRecords)
                << "\": " << *aicpu_lifecycle_extension;
    }

    outfile << "\n}\n";
    outfile.close();

    if (!outfile) {
        LOG_ERROR("Failed to write JSON file (stream error): %s", filepath.c_str());
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    LOG_INFO("=== JSON Export Complete ===");
    LOG_INFO("File: %s", filepath.c_str());

    return 0;
}

int ChipSwimlaneCollector::finalize(
    ChipSwimlaneUnregisterCallback unregister_cb, const ChipSwimlaneFreeCallback &free_cb
) {
    if (shm_host_ == nullptr) {
        return 0;
    }

    // A session that is still open owns the publisher thread and may hold
    // epoch storage; closing it here keeps `finalize()` correct for a caller
    // that never called `close_diagnostics_session()`. Idempotent.
    session_close();

    // Stop mgmt + collector threads if the caller didn't already (idempotent).
    stop();

    // `stop()` has joined every drain owner and every collector shard, so
    // storage a close could not prove safe to touch is now unreachable by any
    // reader. This is the only point at which that is true: the production
    // close path runs before this, and joining the publisher says nothing about
    // the readers.
    session_release_deferred_storage();

    LOG_DEBUG("Cleaning up performance profiling resources");

    // Every release site below goes through release_one_buffer so an
    // optional halHostRegister unregister and the free stay an inseparable
    // pair — each dev_ptr a register_cb mapped is unregistered before its
    // device memory is freed. On non-SVM platforms register_cb is null, so the
    // unregister branch is a no-op and only the device free runs; the paired
    // host shadows are reclaimed separately by clear_mappings() below.
    // The pairing matters on a2a3, where leaking HAL registrations across
    // init_chip_swimlane() invocations makes back-to-back tests on a reused
    // Worker fail at rc=8 from halHostRegister.

    // Free standalone chip_swimlane_aicore_rotation_table table
    release_one_buffer(aicore_ring_addr_table_dev_, unregister_cb, free_cb);
    aicore_ring_addr_table_dev_ = nullptr;

    // Release framework-owned buffers (recycled pools, done_queue, ready_queue).
    manager_.release_owned_buffers([this, unregister_cb, free_cb](void *p) {
        release_one_buffer(p, unregister_cb, free_cb);
    });

    // Per-core: current buffer + free_queue slots — these were owned by
    // the AICPU side, not the framework. Same drain pattern for both the
    // ChipSwimlaneAicpuTaskBuffer pool and the ChipSwimlaneAicoreTaskBuffer pool.
    auto drain_free_queue = [&](ChipSwimlaneFreeQueue &fq) {
        rmb();
        uint32_t head = fq.head;
        uint32_t tail = fq.tail;
        uint32_t queued = tail - head;
        if (queued > PLATFORM_PROF_SLOT_COUNT) {
            queued = PLATFORM_PROF_SLOT_COUNT;
        }
        for (uint32_t k = 0; k < queued; k++) {
            uint32_t slot = (head + k) % PLATFORM_PROF_SLOT_COUNT;
            release_one_buffer(reinterpret_cast<void *>(fq.buffer_ptrs[slot]), unregister_cb, free_cb);
            fq.buffer_ptrs[slot] = 0;
        }
        fq.head = tail;
    };

    for (int i = 0; i < num_aicore_; i++) {
        ChipSwimlaneAicpuTaskPool *state = get_perf_buffer_state(shm_host_, i);
        release_one_buffer(reinterpret_cast<void *>(state->head.current_buf_ptr), unregister_cb, free_cb);
        state->head.current_buf_ptr = 0;
        drain_free_queue(state->free_queue);

        ChipSwimlaneAicoreTaskPool *ac_state = get_aicore_buffer_state(shm_host_, i);
        release_one_buffer(reinterpret_cast<void *>(ac_state->head.current_buf_ptr), unregister_cb, free_cb);
        ac_state->head.current_buf_ptr = 0;
        drain_free_queue(ac_state->free_queue);
    }

    auto release_phase_pool = [&](ChipSwimlaneAicpuTaskPool *state) {
        release_one_buffer(reinterpret_cast<void *>(state->head.current_buf_ptr), unregister_cb, free_cb);
        state->head.current_buf_ptr = 0;

        rmb();
        uint32_t head = state->free_queue.head;
        uint32_t tail = state->free_queue.tail;
        uint32_t queued = tail - head;
        if (queued > PLATFORM_PROF_SLOT_COUNT) {
            queued = PLATFORM_PROF_SLOT_COUNT;
        }
        for (uint32_t k = 0; k < queued; k++) {
            uint32_t slot = (head + k) % PLATFORM_PROF_SLOT_COUNT;
            release_one_buffer(reinterpret_cast<void *>(state->free_queue.buffer_ptrs[slot]), unregister_cb, free_cb);
            state->free_queue.buffer_ptrs[slot] = 0;
        }
        state->free_queue.head = tail;
    };
    int num_phase_threads = PLATFORM_MAX_AICPU_THREADS;
    for (int t = 0; t < num_phase_threads; t++) {
        release_phase_pool(get_sched_phase_buffer_state(shm_host_, t));
    }
    for (int t = 0; t < num_phase_threads; t++) {
        release_phase_pool(get_orch_phase_buffer_state(shm_host_, t));
    }

    // Main shm: unregister + free as a pair, same as every other buffer.
    // ProfilerBase's set_memory_context handed register_cb == nullptr iff the
    // caller doesn't intend to register, so checking unregister_cb inside
    // release_one_buffer is sufficient — no separate ``was_registered_`` flag.
    release_one_buffer(perf_shared_mem_dev_, unregister_cb, free_cb);
    LOG_DEBUG("Main shm released");

    perf_shared_mem_dev_ = nullptr;
    // Free any malloc'd host shadows still tracked in the manager's
    // malloc_shadows_ — the shm region, rotation table, and per-pool buffers
    // were freed above via release_one_buffer (device pointer only), so their
    // paired shadows (allocated by alloc_paired_buffer on the non-SVM path)
    // never went through release_owned_buffers. clear_mappings() std::free's
    // them. No-op on SVM (host_ptr == dev_ptr, nothing in malloc_shadows_).
    // Matches PMU / DepGen finalize.
    manager_.clear_mappings();
    // shm_host_ aliases freed device/host memory now; null it so is_initialized()
    // reports false, the dtor's "destroyed without finalize()" warning stays
    // quiet, and a re-entrant finalize() / re-init hits the early-out instead of
    // walking freed buffer state. Mirrors PMU/DepGen/ArgsDump collectors.
    shm_host_ = nullptr;
    collected_perf_records_.clear();
    collected_aicore_records_.clear();
    collected_sched_phase_records_.clear();
    collected_orch_phase_records_.clear();
    host_submit_records_.clear();
    host_upload_records_.clear();
    for (EpochStore &slot_store : epoch_stores_) {
        slot_store.perf.clear();
        slot_store.aicore.clear();
        slot_store.sched_phase.clear();
        slot_store.orch_phase.clear();
        slot_store.counters.clear();
    }
    core_to_thread_.clear();
    has_phase_data_ = false;
    total_perf_collected_ = 0;
    total_sched_phase_collected_ = 0;
    total_orch_phase_collected_ = 0;
    total_aicore_collected_ = 0;
    aicore_skipped_unwritten_ = 0;
    aicore_skipped_overflow_ = 0;
    aicore_skipped_bad_core_ = 0;
    aicore_foreign_identity_ = 0;
    // The region this identity was armed against is gone.
    armed_run_epoch_ = 0;
    // Read out of the same region, so it goes with it.
    terminal_reported_ = false;
    terminal_snapshot_ = RunTerminalSnapshot{};
    terminal_consistency_ = RunTerminalConsistency{};
    merged_slot_ = -1;
    host_orchestrated_ = false;
    host_phase_records_present_ = false;
    host_phase_total_records_ = 0;
    host_phase_dropped_records_ = 0;
    host_phase_submitted_tasks_ = 0;
    json_extensions_.fill({});
    clear_memory_context();

    LOG_DEBUG("Performance profiling cleanup complete");
    return 0;
}

// ---------------------------------------------------------------------------
// Continuous-collection session
// ---------------------------------------------------------------------------
//
// One run's receipt, sealing and file write continue on this host while the
// next run executes on the device. Device execution stays serial and the run
// boundary keeps its completion fence, its terminal read and its execution
// claim. The boundary drains no pipeline and wipes no record store, so a
// predecessor's buffers keep arriving into a store of its own.
//
// Threads: the runtime thread opens and closes epochs, the drain owners
// capture the per-queue cut, the collector shards copy into the epoch their
// buffer names, and one session thread seals and publishes. The session thread
// never waits on a caller, which is what keeps the capacity backstop free of a
// cycle.

namespace {

// Bounded, monotonic session identity. A process-local run epoch repeats
// across processes, so the directory reservation below is what actually makes
// an artifact path unique; this only labels the file's contents.
std::atomic<uint64_t> g_session_seq{0};

}  // namespace

size_t ChipSwimlaneCollector::session_epoch_fixed_bytes() const {
    // Everything an admitted epoch holds whose size a platform maximum already
    // bounds, plus an allowance for the two paths a run and its writer keep.
    // Reserved rather than charged, so an epoch that has been admitted can
    // never be refused the storage its own artifact needs — a refusal there
    // could only answer by omitting metadata the file is not readable without.
    //
    // The terminal snapshot's per-class index vectors are owned storage the
    // `RunExport` shell does not cover: the shell holds the vector objects, not
    // the blocks they allocate. One entry per producer that reported, so two
    // classes bounded by the core count and two by the thread count.
    const size_t terminal_indices =
        (2 * static_cast<size_t>(PLATFORM_MAX_CORES) + 2 * static_cast<size_t>(PLATFORM_MAX_AICPU_THREADS)) *
        sizeof(int);
    return sizeof(RunExport) + static_cast<size_t>(PLATFORM_MAX_CORES) * sizeof(int8_t) +
           static_cast<size_t>(PLATFORM_MAX_CORES) * sizeof(CoreType) +
           static_cast<size_t>(PLATFORM_MAX_AICPU_THREADS) * sizeof(uint32_t) + terminal_indices +
           2 * simpler::dfx::session::kPathAllowanceBytes;
}

size_t ChipSwimlaneCollector::session_fixed_overhead() const {
    const size_t shards = static_cast<size_t>(manager_.shard_count() > 0 ? manager_.shard_count() : 1);
    const size_t instances = static_cast<size_t>(num_aicore_) + static_cast<size_t>(PLATFORM_MAX_AICPU_THREADS);
    // Two record classes per instance group, one vector header each, in every
    // slot; plus the per-shard counters, the per-shard epoch views (a shard
    // holds its current view and, across one refresh, the new one), the
    // permanent error summary, each slot's bounded run metadata, and the
    // writer's scratch.
    const size_t skeleton =
        simpler::dfx::session::kMaxOpenEpochs * shards * 2 * instances * sizeof(std::vector<CollectedRecord<int>>);
    const size_t counters = simpler::dfx::session::kMaxOpenEpochs * shards * sizeof(CollectorShardCounters);
    const size_t views = (2 * shards + 1) * sizeof(ShardEpochView);
    const size_t buckets = simpler::dfx::session::kMaxOpenEpochs * sizeof(EpochBucket);
    const size_t tombstones = simpler::dfx::session::kMaxTombstones * sizeof(uint64_t);
    const size_t epoch_metadata = simpler::dfx::session::kMaxOpenEpochs * session_epoch_fixed_bytes();
    return skeleton + counters + views + buckets + tombstones + epoch_metadata + sizeof(session_errors_) +
           simpler::dfx::session::kWriterScratchBytes;
}

bool ChipSwimlaneCollector::session_reserve_directory(const std::string &output_root) {
    // Atomic reservation: `mkdir` fails EEXIST without a window, so two
    // processes sharing a prefix take different directories by construction and
    // no identity is derived from a pid, which repeats.
    std::error_code ec;
    std::filesystem::create_directories(output_root, ec);
    if (ec) {
        LOG_ERROR("ChipSwimlane session: cannot create output root %s: %s", output_root.c_str(), ec.message().c_str());
        return false;
    }
    for (int k = 0; k < 4096; k++) {
        std::string candidate = output_root + "/swimlane-" + std::to_string(k);
        if (::mkdir(candidate.c_str(), 0755) == 0) {
            session_dir_ = candidate;
            return true;
        }
        if (errno != EEXIST) {
            LOG_ERROR("ChipSwimlane session: cannot reserve %s: %s", candidate.c_str(), std::strerror(errno));
            return false;
        }
    }
    LOG_ERROR("ChipSwimlane session: no free session directory under %s", output_root.c_str());
    return false;
}

bool ChipSwimlaneCollector::session_open(const SessionOptions &options, const std::string &output_root) {
    if (!options.enabled) return false;
    if (session_active_.load(std::memory_order_acquire)) return true;
    if (shm_host_ == nullptr) {
        LOG_ERROR("ChipSwimlane session: collector is not initialized");
        return false;
    }
    // A session's device-side bound is a byte figure, and a release that did
    // not report success leaves bytes this pool may still hold. The release
    // surface reports a status per pointer and no size, so there is no honest
    // figure to carry forward — the session refuses instead of admitting
    // against an occupancy it cannot state.
    if (manager_.release_unproven()) {
        LOG_ERROR(
            "ChipSwimlane session: refused, an earlier buffer release did not report success and its paired "
            "occupancy cannot be established"
        );
        return false;
    }
    // The per-epoch reservation covers a path of at most this length, and an
    // artifact name is the reserved directory plus a bounded suffix. Refusing
    // here is what lets every later path be reserved rather than charged.
    if (output_root.size() + 64 > simpler::dfx::session::kPathAllowanceBytes) {
        LOG_ERROR(
            "ChipSwimlane session: output root of %zu bytes exceeds the %zu byte path allowance", output_root.size(),
            simpler::dfx::session::kPathAllowanceBytes
        );
        return false;
    }
    const size_t fixed = session_fixed_overhead();
    if (!session_budget_.open(options.budget_bytes, fixed)) return false;
    if (!session_reserve_directory(output_root)) {
        session_budget_.close();
        return false;
    }

    // Twice the paired bytes `init()` seeded, per kind: one replenishment
    // generation beyond the seed is admitted and the rest refused. The seed is
    // a figure growth does not enter, so a session that grew and a session
    // opened after it get the same cap; the live total that admissions are
    // compared against does include that growth, so a pool already holding
    // more than twice its seed admits nothing further.
    //
    // A kind whose seed is zero has no instance in this run's configuration and
    // so never asks for a block. It takes a small floor rather than zero, which
    // this manager reads as uncapped: the floor is unreachable for such a kind,
    // and "uncapped" would not be. The doubling is clamped, because a wrapped
    // cap would refuse every allocation rather than bound it.
    constexpr size_t kMinPairedCapBytes = sizeof(ChipSwimlaneAicoreTaskBuffer);
    for (int kind = 0; kind < ChipSwimlaneModule::kBufferKinds; kind++) {
        const size_t seeded = manager_.paired_initial(kind);
        const size_t baseline = seeded > kMinPairedCapBytes ? seeded : kMinPairedCapBytes;
        const size_t cap = baseline > SIZE_MAX / 2 ? SIZE_MAX : baseline * 2;
        manager_.set_paired_cap(kind, cap);
    }

    session_id_ = g_session_seq.fetch_add(1, std::memory_order_relaxed) + 1;
    session_fatal_.store(false, std::memory_order_release);
    session_fatal_reason_.clear();
    session_release_deferred_.store(false, std::memory_order_release);
    session_aicore_collected_.store(0, std::memory_order_relaxed);
    session_aicore_foreign_.store(0, std::memory_order_relaxed);
    session_close_watermark_.store(0, std::memory_order_release);
    for (auto &t : session_tombstones_)
        t.store(0, std::memory_order_relaxed);
    for (size_t slot = 0; slot < session_buckets_.size(); slot++) {
        session_buckets_[slot].state.store(static_cast<int>(EpochState::Free), std::memory_order_relaxed);
        session_buckets_[slot].epoch.store(0, std::memory_order_relaxed);
        session_buckets_[slot].charged_bytes.store(0, std::memory_order_relaxed);
        reset_epoch_store(slot, /*reset_merged_view=*/slot == 0);
    }
    set_drain_quantum(simpler::dfx::session::kDrainQuantum);
    // Armed before the first run is admitted, not at the first cut: a target is
    // captured from the same counters, so they have to have been counting for
    // the whole span the target covers.
    set_session_counters(true);
    session_active_.store(true, std::memory_order_release);
    session_thread_running_.store(true, std::memory_order_release);
    session_thread_ = std::thread(&ChipSwimlaneCollector::session_thread_main, this);
    LOG_INFO(
        "ChipSwimlane session %lu open: dir=%s budget=%zu B (fixed %zu B)", static_cast<unsigned long>(session_id_),
        session_dir_.c_str(), options.budget_bytes, fixed
    );
    return true;
}

void ChipSwimlaneCollector::session_close() {
    if (!session_active_.load(std::memory_order_acquire)) return;
    // Stop admitting first, then let the thread finish whatever it can.
    {
        std::lock_guard<std::mutex> lk(session_mu_);
        session_close_watermark_.store(UINT64_MAX, std::memory_order_release);
        session_progress_++;
    }
    session_cv_.notify_all();
    std::string ignored;
    (void)session_flush(simpler::dfx::session::kCutAckBudgetMs * 8, &ignored);

    {
        std::lock_guard<std::mutex> lk(session_mu_);
        session_thread_running_.store(false, std::memory_order_release);
        session_progress_++;
    }
    session_cv_.notify_all();
    if (session_thread_.joinable()) session_thread_.join();
    session_active_.store(false, std::memory_order_release);
    set_drain_quantum(0);
    set_session_counters(false);
    // The pools go back to uncapped, which is what a collector without a
    // session has.
    for (int kind = 0; kind < ChipSwimlaneModule::kBufferKinds; kind++) {
        manager_.set_paired_cap(kind, 0);
    }

    // Whatever is still occupied here is storage a *collector shard* may be
    // writing, and joining the publisher proves nothing about the readers —
    // the production close path runs before the collector's threads are joined
    // at all (`finalize_device` calls this, then `runner->finalize()`). So
    // nothing is reset or freed here: `session_release_deferred_storage()`,
    // which `finalize()` calls once `stop()` has joined every reader, is the
    // only place that may touch it.
    for (size_t slot = 0; slot < session_buckets_.size(); slot++) {
        if (session_buckets_[slot].state.load(std::memory_order_acquire) == static_cast<int>(EpochState::Free)) {
            continue;
        }
        session_release_deferred_.store(true, std::memory_order_release);
        LOG_WARN(
            "ChipSwimlane session %lu: epoch %lu storage is held until the collector threads are joined",
            static_cast<unsigned long>(session_id_),
            static_cast<unsigned long>(session_buckets_[slot].epoch.load(std::memory_order_acquire))
        );
    }
    session_budget_.close();
    LOG_INFO(
        "ChipSwimlane session %lu closed: %s", static_cast<unsigned long>(session_id_), session_errors_.report().c_str()
    );
}

void ChipSwimlaneCollector::session_release_deferred_storage() {
    if (!session_release_deferred_.exchange(false, std::memory_order_acq_rel)) return;
    for (size_t slot = 0; slot < session_buckets_.size(); slot++) {
        if (session_buckets_[slot].state.load(std::memory_order_acquire) == static_cast<int>(EpochState::Free)) {
            continue;
        }
        reset_epoch_store(slot, /*reset_merged_view=*/false);
        session_release_slot(slot);
    }
}

void ChipSwimlaneCollector::session_note_host_state_incomplete() {
    if (!session_active_.load(std::memory_order_acquire)) return;
    host_state_incomplete_ = true;
}

void ChipSwimlaneCollector::session_note_boundary_close_failed() {
    // A fixed reason, not a composed one: the summary copies it into its own
    // storage and nothing here has to build a string before the flag is set.
    static constexpr const char *kReason = "a run boundary could not close its epoch";
    bool first = false;
    {
        // Same mutex as session_set_fatal, and for the same reason: a capacity
        // or flush waiter holds it across its check and its wait.
        std::lock_guard<std::mutex> lk(session_mu_);
        if (!session_fatal_.load(std::memory_order_relaxed)) {
            session_fatal_.store(true, std::memory_order_release);
            first = true;
            // A display value, and the only allocation this function makes
            // before its state is complete. A reason that cannot be stored
            // leaves the fatal set and unnamed; the flag above, the count
            // below and the summary's own copy do not depend on it.
            try {
                session_fatal_reason_ = kReason;
            } catch (...) {}
        }
        session_progress_++;
    }
    // The permanent summary writes through snprintf into its own fixed buffer,
    // so it needs nothing this path may be out of.
    if (first) session_errors_.record_fatal(kReason);
    // Ahead of the log line, because the log line is not a non-throwing call:
    // with no writer bound the host logger takes its synchronous path, which
    // constructs the process file sink and assigns its directory string. Every
    // caller that could otherwise wait for a count no one will reach — a
    // capacity wait, a flush barrier — is released before that can matter, and
    // a diagnostic that throws cannot replace the failure that brought us
    // here.
    session_cv_.notify_all();
    if (first) {
        try {
            LOG_ERROR("ChipSwimlane session %lu fatal: %s", static_cast<unsigned long>(session_id_), kReason);
        } catch (...) {}
    }
}

void ChipSwimlaneCollector::session_set_fatal(const std::string &reason) {
    bool first = false;
    {
        // The flag is published under the same mutex a capacity or flush waiter
        // holds across its check and its wait, so a fatal cannot land in the
        // window between them and leave that waiter asleep for good.
        std::lock_guard<std::mutex> lk(session_mu_);
        if (!session_fatal_.load(std::memory_order_relaxed)) {
            session_fatal_.store(true, std::memory_order_release);
            session_fatal_reason_ = reason;
            first = true;
        }
        session_progress_++;
    }
    if (first) {
        // Recorded in the permanent summary, not only in the flag: a writer
        // that died before sealing anything leaves no epoch-scoped verdict, and
        // a flush that tested only those rows would call that a clean session.
        session_errors_.record_fatal(reason.c_str());
        LOG_ERROR("ChipSwimlane session %lu fatal: %s", static_cast<unsigned long>(session_id_), reason.c_str());
    }
    // Wake every caller that could otherwise wait for a count no one will
    // reach: a capacity wait, and a flush barrier.
    session_cv_.notify_all();
}

void ChipSwimlaneCollector::session_note_progress() {
    if (!session_active_.load(std::memory_order_acquire)) return;
    {
        std::lock_guard<std::mutex> lk(session_mu_);
        session_progress_++;
    }
    session_cv_.notify_all();
}

bool ChipSwimlaneCollector::session_charge(size_t slot, size_t bytes) {
    if (!session_active_.load(std::memory_order_acquire)) return true;
    if (bytes == 0) return true;
    if (slot >= session_buckets_.size()) return false;
    if (!session_budget_.charge(bytes)) {
        EpochBucket &bucket = session_buckets_[slot];
        // Withdraw retention for the rest of this epoch rather than blocking
        // this shard: a shard parked here is a shard that cannot acknowledge a
        // reference release, which is exactly what the publisher is waiting
        // for. Receipts keep counting and the verdict reports the loss.
        if (bucket.retain.exchange(false, std::memory_order_acq_rel)) {
            LOG_WARN(
                "ChipSwimlane session %lu: epoch %lu stops retaining records — %zu B would exceed the %zu B budget",
                static_cast<unsigned long>(session_id_),
                static_cast<unsigned long>(bucket.epoch.load(std::memory_order_acquire)), bytes, session_budget_.limit()
            );
        }
        return false;
    }
    session_buckets_[slot].charged_bytes.fetch_add(bytes, std::memory_order_acq_rel);
    return true;
}

void ChipSwimlaneCollector::session_credit(size_t slot, size_t bytes) {
    if (bytes == 0 || slot >= session_buckets_.size()) return;
    session_budget_.credit(bytes);
    EpochBucket &bucket = session_buckets_[slot];
    size_t held = bucket.charged_bytes.load(std::memory_order_acquire);
    while (true) {
        const size_t next = bytes > held ? 0 : held - bytes;
        if (bucket.charged_bytes.compare_exchange_weak(held, next, std::memory_order_acq_rel)) return;
    }
}

bool ChipSwimlaneCollector::session_admit_run_metadata(size_t slot) {
    // Measured at the sources, before any copy exists, and conservatively: the
    // charge counts what a copy can allocate, never only what it shows.
    size_t total = 0;
    auto add = [&total](size_t bytes) {
        if (bytes > SIZE_MAX - total) return false;
        total += bytes;
        return true;
    };

    // Per extension, the element object *and* the block a copy of its payload
    // allocates. `capacity() + 1` rather than `size()` because a string owns a
    // terminator and may hold more than it shows; the per-element term is
    // charged even though this container keeps its elements inline, so the
    // accounting does not silently depend on that staying true — many short or
    // empty extensions would otherwise charge nothing for storage that exists.
    for (const auto &ext : json_extensions_) {
        if (!add(sizeof(std::string)) || !add(ext.capacity()) || !add(1)) return false;
    }

    // Capacity and not size, for the same reason: charging the source's reserve
    // is the direction that cannot come out short.
    size_t submit_bytes = 0;
    size_t upload_bytes = 0;
    if (!simpler::dfx::session::checked_bytes(
            host_submit_records_.capacity(), sizeof(HostPhaseRecord), &submit_bytes
        )) {
        return false;
    }
    if (!simpler::dfx::session::checked_bytes(
            host_upload_records_.capacity(), sizeof(HostPhaseRecord), &upload_bytes
        )) {
        return false;
    }
    if (!add(submit_bytes) || !add(upload_bytes)) return false;
    return session_charge(slot, total);
}

bool ChipSwimlaneCollector::session_read_shm_field(const volatile void *host_field, void *dst, size_t size) {
    if (dst == nullptr || host_field == nullptr || shm_host_ == nullptr || size == 0) return false;
    const auto base = reinterpret_cast<uintptr_t>(shm_host_);
    const auto field = reinterpret_cast<uintptr_t>(host_field);
    if (field < base || size > shm_size_ || field - base > shm_size_ - size) {
        LOG_ERROR("ChipSwimlane session: a field of %zu bytes lies outside the shared region", size);
        return false;
    }
    void *dev = manager_.shared_mem_dev();
    if (dev == nullptr || dev == shm_host_) {
        // Host and device share the region, so the bytes are already in place
        // and a transfer would be the shadow write this exists to avoid.
        std::memcpy(dst, const_cast<const void *>(host_field), size);
        return true;
    }
    void *dev_field = static_cast<char *>(dev) + (field - base);
    return profiling_copy_from_device(dst, dev_field, size) == 0;
}

int ChipSwimlaneCollector::session_find_slot(uint64_t run_epoch) const {
    for (size_t slot = 0; slot < session_buckets_.size(); slot++) {
        if (session_buckets_[slot].state.load(std::memory_order_acquire) == static_cast<int>(EpochState::Free)) {
            continue;
        }
        if (session_buckets_[slot].epoch.load(std::memory_order_acquire) == run_epoch) return static_cast<int>(slot);
    }
    return -1;
}

bool ChipSwimlaneCollector::session_epoch_is_tombstoned(uint64_t run_epoch) const {
    for (const auto &t : session_tombstones_) {
        if (t.load(std::memory_order_relaxed) == run_epoch) return true;
    }
    return false;
}

void ChipSwimlaneCollector::session_bump_control_view() {
    // Publish the table, then make every shard adopt it before returning. A
    // shard reads the control epoch before refreshing, so an ack can never
    // describe a view taken before this change.
    if (!session_request_reference_release(simpler::dfx::session::kControlAckBudgetMs)) {
        session_set_fatal("a collector shard did not acknowledge the epoch table in time");
    }
}

bool ChipSwimlaneCollector::session_run_begin(
    uint64_t run_epoch, const std::string &output_prefix, ChipSwimlaneLevel level
) {
    if (!session_active_.load(std::memory_order_acquire)) return false;
    // Refused before a slot is claimed or a byte copied: the path an admitted
    // epoch retains is reserved, not charged, so a prefix the reservation
    // cannot hold has to be turned away here rather than discovered later by a
    // charge that would have to keep what it could not pay for.
    if (output_prefix.size() > simpler::dfx::session::kPathAllowanceBytes) {
        LOG_ERROR(
            "ChipSwimlane session: run %lu refused, its output prefix of %zu bytes exceeds the %zu byte allowance",
            static_cast<unsigned long>(run_epoch), output_prefix.size(), simpler::dfx::session::kPathAllowanceBytes
        );
        return false;
    }
    size_t slot = 0;
    {
        std::unique_lock<std::mutex> lk(session_mu_);
        // Capacity backstop. The session thread is what frees a slot and it
        // never waits on this thread, so this wait cannot close a cycle; it is
        // bounded by that thread's progress, which file I/O does not bound in
        // wall-clock terms. Both a slot release and a fatal publish under this
        // same mutex, so there is nothing here to poll for.
        while (true) {
            if (session_fatal_.load(std::memory_order_acquire)) {
                LOG_ERROR(
                    "ChipSwimlane session: run %lu refused, session is fatal", static_cast<unsigned long>(run_epoch)
                );
                return false;
            }
            bool found = false;
            for (size_t i = 0; i < session_buckets_.size(); i++) {
                if (session_buckets_[i].state.load(std::memory_order_acquire) == static_cast<int>(EpochState::Free)) {
                    slot = i;
                    found = true;
                    break;
                }
            }
            if (found) break;
            session_cv_.wait(lk);
        }

        EpochBucket &bucket = session_buckets_[slot];
        bucket.epoch.store(run_epoch, std::memory_order_relaxed);
        bucket.retain.store(true, std::memory_order_relaxed);
        bucket.target_installed = false;
        bucket.cut_slot = -1;
        bucket.cut_request = 0;
        bucket.pending = RunExport{};
        bucket.pending.output_prefix = output_prefix;
        bucket.pending.level = level;
        bucket.pending.armed_run_epoch = run_epoch;
        bucket.terminal_ok = false;
        bucket.transport_retired = 0;
        bucket.charged_bytes.store(0, std::memory_order_relaxed);
        bucket.verdict = simpler::dfx::session::CollectionVerdict{};
        bucket.state.store(static_cast<int>(EpochState::Admitting), std::memory_order_release);
        reset_epoch_store(slot, /*reset_merged_view=*/false);

        // The per-run collector state a producer and an exporter both read.
        // Deliberately not `begin_run`: that clears the one record store, which
        // is exactly what a predecessor's still-arriving buffers need to keep.
        output_prefix_ = output_prefix;
        chip_swimlane_level_ = level;
        json_extensions_.fill({});
        // Host-side per-run state, cleared for the same reason the extensions
        // are. `session_run_close` copies whatever the collector holds into
        // this epoch's metadata, and the collector holds one copy across every
        // run it serves, so a run that produces none of it carries none rather
        // than its predecessor's. The vectors keep their capacity: the
        // metadata charge is taken from capacity and is conservative by
        // design, so carrying it forward can only over-charge.
        host_submit_records_.clear();
        host_upload_records_.clear();
        host_phase_records_present_ = false;
        host_phase_submitted_tasks_ = 0;
        host_phase_total_records_ = 0;
        host_phase_dropped_records_ = 0;
        host_state_incomplete_ = false;
    }
    // Per-run device counter reset, unchanged from the legacy path.
    publish_run_config();
    // Every shard must see this epoch before the device can publish into it,
    // or its first buffers would be classified as belonging to no bucket.
    session_bump_control_view();
    return !session_fatal_.load(std::memory_order_acquire);
}

void ChipSwimlaneCollector::session_run_close(uint64_t run_epoch, uint32_t bank_index, bool device_execution_complete) {
    if (!session_active_.load(std::memory_order_acquire)) return;
    const int found = session_find_slot(run_epoch);
    if (found < 0) return;
    const size_t slot = static_cast<size_t>(found);
    EpochBucket &bucket = session_buckets_[slot];

    // Device-side capture, on the teardown thread, while this run still holds
    // its claim: the successor has not launched, so the ready-queue tails the
    // cut reads are stable and the terminal bank is this run's.
    //
    // Every read below lands in this bucket's own storage. Nothing is written
    // into the host shadow: the drain owners keep refreshing their queue
    // cursors and pool metadata there for as long as they run, and a bulk — or
    // even a narrow — write from this thread would put a second writer on words
    // that have exactly one.
    bucket.transport_retired = drain_dropped_buffers();
    report_drain_drops();
    bool reads_ok = true;
    uint64_t total_device = 0;
    uint64_t dropped_device = 0;
    for (int i = 0; i < num_aicore_; i++) {
        ChipSwimlaneActiveHead head{};
        ChipSwimlaneAicpuTaskPool *state = get_perf_buffer_state(shm_host_, i);
        if (!session_read_shm_field(&state->head, &head, sizeof(head))) {
            reads_ok = false;
            break;
        }
        total_device += head.total_record_count;
        dropped_device += head.dropped_record_count;
    }
    bucket.live.mirror_ok = reads_ok;
    bucket.live.aicpu_task_total = total_device;
    bucket.live.aicpu_task_dropped = dropped_device;
    bucket.live.live_ok = reads_ok;

    // Caller-sized metadata is admitted before a byte of it is copied. A
    // refusal after the copy could only answer by keeping storage the budget
    // said it could not pay for, or by freeing records whose readers have not
    // been released yet — neither of which the hard total bound survives.
    const bool metadata_admitted = session_admit_run_metadata(slot);
    // Both ways this epoch's metadata can fall short of its run: a budget that
    // could not admit the caller-sized part, and a host-side publication that
    // did not complete. They settle the same way — the artifact carries what
    // there is and says so — so a run whose publication failed can never be
    // read as one that had nothing to publish.
    bucket.verdict.metadata_complete = metadata_admitted && !host_state_incomplete_;

    ChipSwimlaneDataHeader *header = get_chip_swimlane_header(shm_host_);
    uint32_t num_orch_phase_threads = 0;
    if (session_read_shm_field(&header->num_orch_phase_threads, &num_orch_phase_threads, sizeof(uint32_t))) {
        bucket.pending.num_orch_phase_threads = num_orch_phase_threads;
    }
    std::array<int8_t, PLATFORM_MAX_CORES> core_to_thread{};
    if (session_read_shm_field(header->core_to_thread, core_to_thread.data(), core_to_thread.size())) {
        bucket.pending.core_to_thread.assign(core_to_thread.begin(), core_to_thread.end());
    } else {
        bucket.pending.core_to_thread.assign(static_cast<size_t>(PLATFORM_MAX_CORES), -1);
    }
    bucket.pending.num_aicore = num_aicore_;
    bucket.pending.core_types = core_types_;
    bucket.pending.host_orchestrated = host_orchestrated_;
    if (metadata_admitted) {
        bucket.pending.json_extensions = json_extensions_;
        bucket.pending.host_submit_records = host_submit_records_;
        bucket.pending.host_upload_records = host_upload_records_;
        // The publication's own figures travel with its records. The writer's
        // host-capture status compares the two, so deriving presence from an
        // empty vector and leaving the counts at zero would report every
        // session run's capture as a count mismatch.
        bucket.pending.host_phase_records_present = host_phase_records_present_;
        bucket.pending.host_phase_submitted_tasks = host_phase_submitted_tasks_;
        bucket.pending.host_phase_total_records = host_phase_total_records_;
        bucket.pending.host_phase_dropped_records = host_phase_dropped_records_;
    } else {
        // Reported in the artifact rather than left to be inferred from a
        // missing section. The records this epoch already holds are untouched:
        // nothing may be freed before its readers are proved released.
        LOG_ERROR(
            "ChipSwimlane session %lu: epoch %lu publishes without its caller-sized metadata — the %zu B budget "
            "cannot admit it",
            static_cast<unsigned long>(session_id_), static_cast<unsigned long>(run_epoch), session_budget_.limit()
        );
    }
    bucket.pending.sched_phase_dropped_records.assign(static_cast<size_t>(PLATFORM_MAX_AICPU_THREADS), 0);
    for (int t = 0; t < PLATFORM_MAX_AICPU_THREADS; t++) {
        ChipSwimlaneAicpuSchedPhasePool *state = get_sched_phase_buffer_state(shm_host_, t);
        uint32_t dropped = 0;
        if (!session_read_shm_field(&state->head.dropped_record_count, &dropped, sizeof(dropped))) {
            bucket.live.mirror_ok = false;
            break;
        }
        bucket.pending.sched_phase_dropped_records[static_cast<size_t>(t)] = dropped;
    }

    if (device_execution_complete) {
        bucket.terminal = read_run_terminal_snapshot(bank_index, run_epoch);
        bucket.terminal_ok = bucket.terminal.transport_ok;
    }

    // Arm the per-queue cut. Every buffer this run will ever publish is already
    // in a device ready queue at this instant, so each queue's target is finite
    // and a successor's later traffic cannot discharge it.
    uint64_t request = 0;
    bucket.cut_slot = cut_arm(&request);
    bucket.cut_request = request;
    if (bucket.cut_slot < 0) {
        LOG_WARN("ChipSwimlane session: no cut slot for epoch %lu", static_cast<unsigned long>(run_epoch));
    } else if (!cut_wait_for_ack(request, simpler::dfx::session::kCutAckBudgetMs)) {
        LOG_WARN(
            "ChipSwimlane session: epoch %lu cut capture did not complete in %d ms",
            static_cast<unsigned long>(run_epoch), simpler::dfx::session::kCutAckBudgetMs
        );
    }

    {
        std::lock_guard<std::mutex> lk(session_mu_);
        bucket.target_installed = true;
        bucket.closed_at = std::chrono::steady_clock::now();
        uint64_t watermark = session_close_watermark_.load(std::memory_order_acquire);
        if (watermark != UINT64_MAX && run_epoch > watermark) {
            session_close_watermark_.store(run_epoch, std::memory_order_release);
        }
        session_progress_++;
    }
    session_cv_.notify_all();
}

void ChipSwimlaneCollector::refresh_session_epoch_view(int collector_shard) {
    if (collector_shard < 0 || static_cast<size_t>(collector_shard) >= shard_views_.size()) return;
    ShardEpochView view;
    {
        std::lock_guard<std::mutex> lk(session_mu_);
        for (size_t slot = 0; slot < session_buckets_.size(); slot++) {
            const int state = session_buckets_[slot].state.load(std::memory_order_acquire);
            if (state != static_cast<int>(EpochState::Admitting)) continue;
            view.entries[view.count].epoch = session_buckets_[slot].epoch.load(std::memory_order_acquire);
            view.entries[view.count].slot = static_cast<int>(slot);
            view.entries[view.count].retain = session_buckets_[slot].retain.load(std::memory_order_acquire);
            view.count++;
        }
    }
    shard_views_[static_cast<size_t>(collector_shard)] = view;
}

void ChipSwimlaneCollector::session_release_slot(size_t slot) {
    EpochBucket &bucket = session_buckets_[slot];
    const uint64_t epoch = bucket.epoch.load(std::memory_order_acquire);
    const size_t cursor = session_tombstone_cursor_.fetch_add(1, std::memory_order_relaxed);
    session_tombstones_[cursor % session_tombstones_.size()].store(epoch, std::memory_order_relaxed);
    // Free the storage first, then give its bytes back: a credit ahead of the
    // release would let an admission see headroom that does not exist yet.
    bucket.pending = RunExport{};
    const size_t charged = bucket.charged_bytes.exchange(0, std::memory_order_acq_rel);
    if (charged > 0) session_budget_.credit(charged);
    // Published under `session_mu_` so an admission waiting for capacity cannot
    // miss the transition between its scan and its wait.
    {
        std::lock_guard<std::mutex> lk(session_mu_);
        bucket.state.store(static_cast<int>(EpochState::Free), std::memory_order_release);
        session_progress_++;
    }
    session_cv_.notify_all();
}

int ChipSwimlaneCollector::session_publish_file(RunExport &data, const std::string &path) {
    const std::string tmp = path + ".tmp";
    // Exclusive create: no other writer can share this temp file, and a stale
    // one from an earlier crash is an error rather than a silent reuse.
    int fd = ::open(tmp.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0644);
    if (fd < 0) {
        LOG_ERROR("ChipSwimlane session: cannot create %s: %s", tmp.c_str(), std::strerror(errno));
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    ::close(fd);
    // The writer fills the temp file, never the published path: the final name
    // must come into existence only through the link below, or a reader could
    // see a half-written artifact under the name that means "complete".
    data.artifact_path = tmp;
    const int rc = write_swimlane_json(data);
    if (rc != 0) {
        ::unlink(tmp.c_str());
        return rc;
    }
    // Publication that cannot replace: `link` fails EEXIST atomically, so a
    // second epoch targeting an occupied path reports a collision instead of
    // overwriting somebody's artifact. No fsync — this is crash atomicity
    // against process death, not durability.
    if (::link(tmp.c_str(), path.c_str()) != 0) {
        LOG_ERROR("ChipSwimlane session: cannot publish %s: %s", path.c_str(), std::strerror(errno));
        ::unlink(tmp.c_str());
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    ::unlink(tmp.c_str());
    return 0;
}

bool ChipSwimlaneCollector::session_seal_and_publish(size_t slot, simpler::dfx::session::Verdict verdict) {
    EpochBucket &bucket = session_buckets_[slot];
    const uint64_t epoch = bucket.epoch.load(std::memory_order_acquire);

    // Withdraw admission, then wait for every shard to drop its reference.
    // Nothing is moved or freed before the last ack; a timeout quarantines.
    bucket.state.store(static_cast<int>(EpochState::Closing), std::memory_order_release);
    if (!session_request_reference_release(simpler::dfx::session::kControlAckBudgetMs)) {
        bucket.state.store(static_cast<int>(EpochState::Quarantined), std::memory_order_release);
        session_finish_bucket(
            slot, simpler::dfx::session::Verdict::Quarantined, "a collector shard still holds a reference"
        );
        return false;
    }

    // The merged copy every record needs here was charged when the record was
    // retained, so this cannot run out of budget and cannot answer by dropping
    // records it has already accepted.
    merge_epoch_store(slot);
    RunExport &data = bucket.pending;
    data.perf_records = std::move(collected_perf_records_);
    data.aicore_records = std::move(collected_aicore_records_);
    data.sched_phase_records = std::move(collected_sched_phase_records_);
    data.orch_phase_records = std::move(collected_orch_phase_records_);
    collected_perf_records_.clear();
    collected_aicore_records_.clear();
    collected_sched_phase_records_.clear();
    collected_orch_phase_records_.clear();
    for (auto &shard : store(slot).perf)
        for (auto &instance : shard) {
            instance.clear();
            instance.shrink_to_fit();
        }
    for (auto &shard : store(slot).aicore)
        for (auto &instance : shard) {
            instance.clear();
            instance.shrink_to_fit();
        }
    for (auto &shard : store(slot).sched_phase)
        for (auto &instance : shard) {
            instance.clear();
            instance.shrink_to_fit();
        }
    for (auto &shard : store(slot).orch_phase)
        for (auto &instance : shard) {
            instance.clear();
            instance.shrink_to_fit();
        }
    data.total_perf_collected = total_perf_collected_;
    data.total_sched_phase_collected = total_sched_phase_collected_;
    data.total_orch_phase_collected = total_orch_phase_collected_;
    data.total_aicore_collected = total_aicore_collected_;
    // The merged view is this epoch's, so these are its figures. Accumulated
    // because the artifact carries the AICore rows and not the count behind
    // them, so this is where that count is observable.
    session_aicore_collected_.fetch_add(total_aicore_collected_, std::memory_order_relaxed);
    session_aicore_foreign_.fetch_add(aicore_foreign_identity_, std::memory_order_relaxed);
    data.has_phase_data = has_phase_data_;
    data.terminal_snapshot = bucket.terminal;
    data.terminal_reported = bucket.terminal_ok;
    const HandoffReport report = build_handoff_report(bucket.terminal, slot);
    // The merged view's records have been moved out, so it describes no slot.
    // The next epoch's merge tests this, and a stale slot number would make it
    // short-circuit and seal an empty artifact.
    merged_slot_ = -1;
    data.collection.present = true;
    data.collection.run_epoch = epoch;
    data.collection.session_id = session_id_;
    data.collection.processing_complete =
        verdict == simpler::dfx::session::Verdict::Published || verdict == simpler::dfx::session::Verdict::PartialSafe;
    data.collection.transport_retired = bucket.transport_retired;
    data.collection.metadata_complete = bucket.verdict.metadata_complete;
    // Decided when this epoch was serviced, against that cut's own request. A
    // second reading here could find a count the retirement below has already
    // made meaningless.
    data.collection.cut_failed_queues = bucket.verdict.cut_failed_queues;
    data.collection.unpublished_loss = report.aicore_task.silent_loss_known ? report.aicore_task.silent_loss : 0;
    data.collection.not_received_buffers =
        report.aicore_task.published_buffers > report.aicore_task.received_buffers ?
            report.aicore_task.published_buffers - report.aicore_task.received_buffers :
            0;
    if (verdict == simpler::dfx::session::Verdict::Published &&
        (data.collection.transport_retired != 0 || data.collection.not_received_buffers != 0 ||
         data.collection.unpublished_loss != 0 || !data.collection.metadata_complete ||
         !bucket.retain.load(std::memory_order_acquire))) {
        verdict = simpler::dfx::session::Verdict::PartialSafe;
    }
    data.collection.verdict = verdict;

    bucket.state.store(static_cast<int>(EpochState::Sealed), std::memory_order_release);

    // A verdict that leaves no artifact still has to come through here: the
    // records were only safe to move because the handshake above proved every
    // shard had dropped its reference, and the slot has to be handed back
    // either way.
    if (!simpler::dfx::session::verdict_publishes(verdict)) {
        session_finish_bucket(slot, verdict, "no artifact is written for this verdict");
        return false;
    }

    const std::string path = session_dir_ + "/records_e" + std::to_string(epoch) + ".json";
    const int rc = session_publish_file(data, path);
    if (rc != 0) {
        session_finish_bucket(slot, simpler::dfx::session::Verdict::WriteFailed, path.c_str());
        return false;
    }
    session_finish_bucket(slot, verdict, nullptr);
    return true;
}

void ChipSwimlaneCollector::session_finish_bucket(
    size_t slot, simpler::dfx::session::Verdict verdict, const char *detail
) {
    EpochBucket &bucket = session_buckets_[slot];
    const uint64_t epoch = bucket.epoch.load(std::memory_order_acquire);
    session_errors_.record(epoch, verdict, detail);
    if (bucket.cut_slot >= 0) {
        // Retiring waits for every drain owner to prove it is out of this
        // slot's arrays. A failure leaves the slot retired for good rather than
        // handing a live reader's memory to the next arm.
        if (!cut_release(bucket.cut_slot, simpler::dfx::session::kCutAckBudgetMs)) {
            session_set_fatal("cut slot " + std::to_string(bucket.cut_slot) + " was not retired by every drain owner");
        }
        bucket.cut_slot = -1;
        bucket.cut_request = 0;
    }
    if (verdict == simpler::dfx::session::Verdict::Quarantined) {
        // No release: the references were never proved gone. The slot stays
        // occupied, admission fails from here on, and only the reader-join
        // teardown in `session_release_deferred_storage()` may touch this
        // storage.
        session_set_fatal("epoch " + std::to_string(epoch) + " quarantined: " + (detail != nullptr ? detail : ""));
        return;
    }
    if (verdict == simpler::dfx::session::Verdict::WriteFailed ||
        verdict == simpler::dfx::session::Verdict::CounterExhausted) {
        // Sealed already moved the records out of the shards, so the memory is
        // this thread's to release even though no file exists.
        LOG_ERROR(
            "ChipSwimlane session: epoch %lu ended %s (%s)", static_cast<unsigned long>(epoch),
            simpler::dfx::session::verdict_name(verdict), detail != nullptr ? detail : ""
        );
    } else {
        LOG_INFO(
            "ChipSwimlane session: epoch %lu %s", static_cast<unsigned long>(epoch),
            simpler::dfx::session::verdict_name(verdict)
        );
    }
    reset_epoch_store(slot, /*reset_merged_view=*/false);
    session_release_slot(slot);
}

void ChipSwimlaneCollector::session_service_once() {
    for (size_t slot = 0; slot < session_buckets_.size(); slot++) {
        EpochBucket &bucket = session_buckets_[slot];
        if (bucket.state.load(std::memory_order_acquire) != static_cast<int>(EpochState::Admitting)) continue;
        bool ready = false;
        {
            std::lock_guard<std::mutex> lk(session_mu_);
            ready = bucket.target_installed;
        }
        if (!ready) continue;

        // Every terminal input acts, not only a pass: a cut that can never
        // settle still has to release its slot, or two failed epochs would
        // strand the capacity for good.
        simpler::dfx::session::Verdict verdict = simpler::dfx::session::Verdict::Published;
        int failed = 0;
        const bool cut_known = bucket.cut_slot >= 0 && cut_failed_queues(bucket.cut_slot, bucket.cut_request, &failed);
        bucket.verdict.cut_failed_queues = static_cast<uint64_t>(failed);
        if (cut_counters_exhausted()) {
            // A counter within its wrap margin makes every target comparison
            // meaningless, so no cut after it can be justified.
            verdict = simpler::dfx::session::Verdict::CounterExhausted;
        } else if (!cut_known || failed != 0) {
            verdict = simpler::dfx::session::Verdict::CutUnknown;
        } else if (!cut_stage2_done(bucket.cut_slot)) {
            const auto waited = std::chrono::steady_clock::now() - bucket.closed_at;
            if (waited < std::chrono::milliseconds(simpler::dfx::session::kCutAckBudgetMs * 4)) continue;
            verdict = simpler::dfx::session::Verdict::CutUnknown;
        }
        if (!bucket.terminal_ok && verdict == simpler::dfx::session::Verdict::Published) {
            verdict = simpler::dfx::session::Verdict::CutUnknown;
        }
        (void)session_seal_and_publish(slot, verdict);
    }
}

std::optional<std::chrono::steady_clock::time_point> ChipSwimlaneCollector::session_next_wakeup() const {
    // The only thing this thread ever needs a clock for: an epoch whose cut has
    // not settled gives up after a bounded wait, and that expiry is not an
    // event anybody can signal. Everything else — a target landing, stage 1
    // publishing, a shard reaching its watermark, a fatal, a close — bumps
    // `session_progress_` and wakes this thread on the spot.
    std::optional<std::chrono::steady_clock::time_point> earliest;
    for (const auto &bucket : session_buckets_) {
        if (bucket.state.load(std::memory_order_acquire) != static_cast<int>(EpochState::Admitting)) continue;
        if (!bucket.target_installed) continue;
        const auto expiry = bucket.closed_at + std::chrono::milliseconds(simpler::dfx::session::kCutAckBudgetMs * 4);
        if (!earliest.has_value() || expiry < earliest.value()) earliest = expiry;
    }
    return earliest;
}

void ChipSwimlaneCollector::session_thread_main() {
    while (session_thread_running_.load(std::memory_order_acquire)) {
        uint64_t seen = 0;
        {
            std::lock_guard<std::mutex> lk(session_mu_);
            seen = session_progress_;
        }
        try {
            session_service_once();
        } catch (const std::exception &e) {
            // A background writer that dies silently strands every waiter, so
            // the fatal is published and broadcast before anything else.
            session_set_fatal(std::string("session thread failed: ") + e.what());
        } catch (...) {
            session_set_fatal("session thread failed with an unknown exception");
        }
        std::unique_lock<std::mutex> lk(session_mu_);
        if (!session_thread_running_.load(std::memory_order_acquire)) break;
        // Progress that landed while this pass was running is not a missed
        // wakeup: the counter says so, and the next pass picks it up.
        if (session_progress_ != seen) continue;
        const auto deadline = session_next_wakeup();
        if (deadline.has_value()) {
            session_cv_.wait_until(lk, deadline.value());
        } else {
            session_cv_.wait(lk);
        }
    }
}

bool ChipSwimlaneCollector::session_flush(int timeout_ms, std::string *error) {
    if (!session_active_.load(std::memory_order_acquire)) return true;
    const uint64_t watermark = session_close_watermark_.load(std::memory_order_acquire);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (true) {
        bool pending = false;
        {
            std::unique_lock<std::mutex> lk(session_mu_);
            for (size_t slot = 0; slot < session_buckets_.size(); slot++) {
                const int state = session_buckets_[slot].state.load(std::memory_order_acquire);
                if (state == static_cast<int>(EpochState::Free)) continue;
                if (state == static_cast<int>(EpochState::Quarantined)) continue;  // terminal, reported below
                const uint64_t epoch = session_buckets_[slot].epoch.load(std::memory_order_acquire);
                if (epoch <= watermark && session_buckets_[slot].target_installed) pending = true;
            }
            if (!pending) break;
            if (session_fatal_.load(std::memory_order_acquire)) break;
            if (session_cv_.wait_until(lk, deadline) == std::cv_status::timeout &&
                std::chrono::steady_clock::now() >= deadline) {
                if (error != nullptr) {
                    *error = "chip swimlane flush timed out with epochs still unpublished; " + session_errors_.report();
                }
                return false;
            }
        }
    }
    // A fatal is a failure of the session itself and is reported as one even
    // when no epoch has a verdict yet: a writer that died before sealing
    // anything leaves no per-epoch row, and a promised file does not exist.
    if (session_fatal_.load(std::memory_order_acquire)) {
        if (error != nullptr) {
            std::lock_guard<std::mutex> lk(session_mu_);
            *error = "chip swimlane session is fatal: " + session_fatal_reason_ + "; " + session_errors_.report();
        }
        return false;
    }
    // A published partial is a verdict, not a failure; anything that left no
    // file is reported, and the permanent summary is what remembers a failure
    // from more than `kMaxTombstones` epochs ago.
    if (session_errors_.has_error()) {
        if (error != nullptr) *error = "chip swimlane session reported failures: " + session_errors_.report();
        return false;
    }
    return true;
}

ChipSwimlaneCollector::SessionStats ChipSwimlaneCollector::session_stats_for_test() const {
    SessionStats stats;
    stats.active = session_active_.load(std::memory_order_acquire);
    stats.fatal = session_fatal_.load(std::memory_order_acquire);
    stats.late_after_seal = session_late_after_seal_.load(std::memory_order_relaxed);
    stats.unknown_epoch = session_unknown_epoch_.load(std::memory_order_relaxed);
    stats.no_bucket = session_no_bucket_.load(std::memory_order_relaxed);
    stats.aicore_collected = session_aicore_collected_.load(std::memory_order_relaxed);
    stats.aicore_foreign = session_aicore_foreign_.load(std::memory_order_relaxed);
    stats.host_charged = session_budget_.charged();
    stats.budget_refusals = session_budget_.refusals();
    stats.release_deferred = session_release_deferred_.load(std::memory_order_acquire);
    for (const auto &bucket : session_buckets_) {
        if (bucket.state.load(std::memory_order_acquire) != static_cast<int>(EpochState::Free)) stats.open_slots++;
    }
    // One locked read, so the per-verdict rows and the aggregate cannot
    // disagree. `published` is every epoch that left a readable artifact:
    // settled, content-partial and cut-unknown together.
    const auto counts = session_errors_.counts();
    stats.partial = counts.partial;
    stats.cut_unknown = counts.cut_unknown;
    stats.write_failed = counts.write_failed;
    stats.quarantined = counts.quarantined;
    stats.counter_exhausted = counts.counter_exhausted;
    stats.published = counts.published + counts.partial + counts.cut_unknown;
    return stats;
}
