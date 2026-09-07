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
    int num_aicore, int aicpu_thread_num, int device_id, const ChipSwimlaneAllocCallback &alloc_cb,
    ChipSwimlaneRegisterCallback register_cb, const ChipSwimlaneFreeCallback &free_cb
) {
    if (shm_host_ != nullptr) {
        // Already holding this run's device resources. They are not per-run:
        // configuration arrives via begin_run() and the layout is fixed at
        // compile time, so there is nothing here left to re-apply.
        return 0;
    }
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
    has_phase_data_ = false;
    collector_shards_merged_ = false;
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
    const size_t shard_count = collector_counters_.size();
    const bool valid_shard = collector_shard >= 0 && static_cast<size_t>(collector_shard) < shard_count;
    if (!valid_shard) {
        assert(false && "collector_shard out of range");
        return shard_count;
    }
    return static_cast<size_t>(collector_shard);
}

void ChipSwimlaneCollector::reset_collector_shards() {
    const size_t shard_count = static_cast<size_t>(manager_.shard_count());

    collected_perf_records_.assign(num_aicore_, {});
    collected_aicore_records_.assign(num_aicore_, {});
    collected_sched_phase_records_.assign(PLATFORM_MAX_AICPU_THREADS, {});
    collected_orch_phase_records_.assign(PLATFORM_MAX_AICPU_THREADS, {});

    perf_records_by_collector_.assign(shard_count, {});
    aicore_records_by_collector_.assign(shard_count, {});
    sched_phase_records_by_collector_.assign(shard_count, {});
    orch_phase_records_by_collector_.assign(shard_count, {});
    for (size_t shard = 0; shard < shard_count; shard++) {
        perf_records_by_collector_[shard].assign(num_aicore_, {});
        aicore_records_by_collector_[shard].assign(num_aicore_, {});
        sched_phase_records_by_collector_[shard].assign(PLATFORM_MAX_AICPU_THREADS, {});
        orch_phase_records_by_collector_[shard].assign(PLATFORM_MAX_AICPU_THREADS, {});
    }
    collector_counters_.assign(shard_count, {});
    total_perf_collected_ = 0;
    total_sched_phase_collected_ = 0;
    total_orch_phase_collected_ = 0;
    has_phase_data_ = false;
    collector_shards_merged_ = false;
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

void ChipSwimlaneCollector::merge_collector_shards() {
    if (collector_shards_merged_) {
        return;
    }

    merge_record_shards(perf_records_by_collector_, collected_perf_records_, static_cast<size_t>(num_aicore_));
    merge_record_shards(aicore_records_by_collector_, collected_aicore_records_, static_cast<size_t>(num_aicore_));
    merge_record_shards(
        sched_phase_records_by_collector_, collected_sched_phase_records_,
        static_cast<size_t>(PLATFORM_MAX_AICPU_THREADS)
    );
    merge_record_shards(
        orch_phase_records_by_collector_, collected_orch_phase_records_, static_cast<size_t>(PLATFORM_MAX_AICPU_THREADS)
    );

    total_perf_collected_ = 0;
    total_sched_phase_collected_ = 0;
    total_orch_phase_collected_ = 0;
    has_phase_data_ = false;
    for (const auto &counter : collector_counters_) {
        total_perf_collected_ += counter.total_perf_collected;
        total_sched_phase_collected_ += counter.total_sched_phase_collected;
        total_orch_phase_collected_ += counter.total_orch_phase_collected;
        has_phase_data_ = has_phase_data_ || counter.has_phase_data;
    }
    collector_shards_merged_ = true;
}

void ChipSwimlaneCollector::copy_perf_buffer(const ReadyBufferInfo &info, int collector_shard) {
    ChipSwimlaneAicpuTaskBuffer *buf = reinterpret_cast<ChipSwimlaneAicpuTaskBuffer *>(info.host_buffer_ptr);
    rmb();
    uint32_t count = buf->count;
    if (count > PLATFORM_PROF_BUFFER_SIZE) {
        count = PLATFORM_PROF_BUFFER_SIZE;
    }
    uint32_t core_index = info.index;
    size_t shard = normalize_collector_shard(collector_shard);
    if (core_index < static_cast<uint32_t>(num_aicore_) && shard < perf_records_by_collector_.size()) {
        auto &dst = perf_records_by_collector_[shard][core_index];
        dst.reserve(dst.size() + count);
        for (uint32_t i = 0; i < count; i++) {
            dst.push_back(buf->records[i]);
        }
        collector_counters_[shard].total_perf_collected += count;
    }
}

void ChipSwimlaneCollector::copy_sched_phase_buffer(const ReadyBufferInfo &info, int collector_shard) {
    auto *buf = reinterpret_cast<ChipSwimlaneAicpuSchedPhaseBuffer *>(info.host_buffer_ptr);
    rmb();
    uint32_t count = buf->count;
    if (count > static_cast<uint32_t>(PLATFORM_PHASE_RECORDS_PER_THREAD)) {
        count = PLATFORM_PHASE_RECORDS_PER_THREAD;
    }
    uint32_t tidx = info.index;
    size_t shard = normalize_collector_shard(collector_shard);
    if (shard < sched_phase_records_by_collector_.size() && tidx < sched_phase_records_by_collector_[shard].size()) {
        auto &dst = sched_phase_records_by_collector_[shard][tidx];
        dst.reserve(dst.size() + count);
        for (uint32_t i = 0; i < count; i++) {
            dst.push_back(buf->records[i]);
        }
        collector_counters_[shard].total_sched_phase_collected += count;
        if (count > 0) {
            collector_counters_[shard].has_phase_data = true;
        }
    }
}

void ChipSwimlaneCollector::copy_orch_phase_buffer(const ReadyBufferInfo &info, int collector_shard) {
    auto *buf = reinterpret_cast<ChipSwimlaneAicpuOrchPhaseBuffer *>(info.host_buffer_ptr);
    rmb();
    uint32_t count = buf->count;
    if (count > static_cast<uint32_t>(PLATFORM_PHASE_RECORDS_PER_THREAD)) {
        count = PLATFORM_PHASE_RECORDS_PER_THREAD;
    }
    uint32_t tidx = info.index;
    size_t shard = normalize_collector_shard(collector_shard);
    if (shard < orch_phase_records_by_collector_.size() && tidx < orch_phase_records_by_collector_[shard].size()) {
        auto &dst = orch_phase_records_by_collector_[shard][tidx];
        dst.reserve(dst.size() + count);
        for (uint32_t i = 0; i < count; i++) {
            dst.push_back(buf->records[i]);
        }
        collector_counters_[shard].total_orch_phase_collected += count;
        if (count > 0) {
            collector_counters_[shard].has_phase_data = true;
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
void ChipSwimlaneCollector::copy_aicore_buffer(const ReadyBufferInfo &info, int collector_shard) {
    ChipSwimlaneAicoreTaskBuffer *buf = reinterpret_cast<ChipSwimlaneAicoreTaskBuffer *>(info.host_buffer_ptr);
    rmb();
    uint32_t core_index = info.index;
    if (core_index >= static_cast<uint32_t>(num_aicore_)) {
        return;
    }
    uint32_t count = buf->count;
    if (count > static_cast<uint32_t>(PLATFORM_AICORE_BUFFER_SIZE)) {
        count = PLATFORM_AICORE_BUFFER_SIZE;
    }
    uint32_t skipped = 0;
    size_t shard = normalize_collector_shard(collector_shard);
    if (shard < aicore_records_by_collector_.size()) {
        auto &dst = aicore_records_by_collector_[shard][core_index];
        dst.reserve(dst.size() + count);
        for (uint32_t i = 0; i < count; i++) {
            const ChipSwimlaneAicoreTaskRecord &r = buf->records[i];
            if (r.start_time == 0) {
                skipped++;
                continue;
            }
            dst.push_back(r);
        }
    }
    if (skipped > 0) {
        LOG_WARN(
            "Core %u: skipped %u AICore record slot(s) with start_time=0 (race-window write or "
            "recycled-buffer tail). buf seq=%u count=%u",
            core_index, skipped, info.buffer_seq, count
        );
    }
}

void ChipSwimlaneCollector::on_buffer_collected(const ReadyBufferInfo &info, int collector_shard) {
    switch (info.type) {
    case ProfBufferType::AICPU_TASK:
        copy_perf_buffer(info, collector_shard);
        break;
    case ProfBufferType::AICPU_SCHED_PHASE:
        copy_sched_phase_buffer(info, collector_shard);
        break;
    case ProfBufferType::AICPU_ORCH_PHASE:
        copy_orch_phase_buffer(info, collector_shard);
        break;
    case ProfBufferType::AICORE_TASK:
        copy_aicore_buffer(info, collector_shard);
        break;
    }
}

// ---------------------------------------------------------------------------
// reconcile_counters / read_phase_header_metadata
// ---------------------------------------------------------------------------
//
// Host never recovers records from device-side current_buf_ptr. Device flush
// is the only data path: a flush failure must bump dropped_record_count and
// clear current_buf_ptr on the device side. Host's job here is purely
// accounting + sanity check.

void ChipSwimlaneCollector::reconcile_counters() {
    if (shm_host_ == nullptr) {
        return;
    }
    merge_collector_shards();

    // Refresh the pool states (current_buf_ptr + total/dropped counters) from
    // device before the sanity loop so leftovers reflect post-stop() device
    // state. Per-buffer contents are pulled individually inside reconcile_one —
    // an un-flushed active buffer was never enqueued, so the mgmt loop's
    // process_entry never copied its contents into the shadow.
    if (manager_.shared_mem_dev() != nullptr && shm_size_ > 0) {
        profiling_copy_from_device(shm_host_, manager_.shared_mem_dev(), shm_size_);
    }
    rmb();

    // Two-bucket invariant (post-AICore-as-producer): every commit attempt
    // bumps total_record_count; capacity-driven drops (no free buffer /
    // queue full / flush failure) bump dropped_record_count.
    //   silent_loss = device_total - (collected + dropped)
    // and any non-zero silent loss flags an unaccounted gap on top of the
    // already-classified dropped losses.
    //
    // Sanity sub-check: after stop(), any active buffer with records must
    // have been flushed by AICPU (success → current_buf_ptr=0; failure →
    // bump dropped, clear count + current_buf_ptr). A non-zero pointer with
    // non-zero count means records AICPU neither delivered nor accounted
    // for — i.e. a device-side flush bug. Empty buffers (count=0, never
    // written) are fine; AICPU's flush legitimately skips them.
    auto reconcile_one = [&](const char *kind, const char *unit_name, int unit_count, auto get_state,
                             auto read_buf_count, size_t buf_size, uint64_t collected, bool optional) {
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
        sizeof(ChipSwimlaneAicpuTaskBuffer), total_perf_collected_, /*optional=*/false
    );

    reconcile_one(
        "SCHED_PHASE", "thread", PLATFORM_MAX_AICPU_THREADS,
        [this](int thread_index) {
            return get_sched_phase_buffer_state(shm_host_, thread_index);
        },
        [](void *host_ptr) {
            return reinterpret_cast<ChipSwimlaneAicpuSchedPhaseBuffer *>(host_ptr)->count;
        },
        sizeof(ChipSwimlaneAicpuSchedPhaseBuffer), total_sched_phase_collected_, /*optional=*/true
    );

    reconcile_one(
        "ORCH_PHASE", "thread", PLATFORM_MAX_AICPU_THREADS,
        [this](int thread_index) {
            return get_orch_phase_buffer_state(shm_host_, thread_index);
        },
        [](void *host_ptr) {
            return reinterpret_cast<ChipSwimlaneAicpuOrchPhaseBuffer *>(host_ptr)->count;
        },
        sizeof(ChipSwimlaneAicpuOrchPhaseBuffer), total_orch_phase_collected_, /*optional=*/true
    );
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
    (void)manager_.write_range_to_device(&header->chip_swimlane_level, sizeof(header->chip_swimlane_level));

    // The pools' record counters are producer-side and never reset by the
    // device, so they carry the previous run's totals into this run's reconcile
    // unless cleared here.
    //
    // total_record_count and dropped_record_count are adjacent, so one narrow
    // write covers both and leaves the device-owned fields in the same cache
    // line (current_buf_ptr, current_buf_seq) untouched.
    auto reset_head = [this](ChipSwimlaneActiveHead *head) {
        head->total_record_count = 0;
        head->dropped_record_count = 0;
        wmb();
        static_assert(
            offsetof(ChipSwimlaneActiveHead, dropped_record_count) ==
                offsetof(ChipSwimlaneActiveHead, total_record_count) + sizeof(uint32_t),
            "the two counters must stay adjacent for this single write-back to cover both"
        );
        (void)manager_.write_range_to_device(&head->total_record_count, 2 * sizeof(uint32_t));
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

void ChipSwimlaneCollector::begin_clock_correlation_session(
    const char *provider_name, const char *raw_device_timestamp_unit
) {
    clock_correlation_session_.begin(provider_name, raw_device_timestamp_unit);
}

void ChipSwimlaneCollector::record_clock_anchor_samples(std::vector<simpler::dfx::ClockAnchorSample> samples) {
    clock_correlation_session_.append(std::move(samples));
}

void ChipSwimlaneCollector::finish_clock_correlation_session() { clock_correlation_session_.finish(); }

// JSON v2 emit: the host now dumps raw cycle-domain per-stream records plus
// metadata, and `swimlane_converter.py` performs the join (AICore↔Scheduler on
// reg_task_id, base_time normalization, cycles→µs conversion, sort, core_type
// lookup, func_id resolution against deps.json). Moving the join into Python
// makes the schema easy to evolve without round-tripping through C++ + a
// rebuild, and shrinks this file to a pure dump.
int ChipSwimlaneCollector::export_swimlane_json() {
    if (shm_host_ == nullptr) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    merge_collector_shards();

    auto extension = [this](ChipSwimlaneExtensionSection section) -> const std::string * {
        const std::string &value = json_extensions_[static_cast<size_t>(section)];
        return value.empty() ? nullptr : &value;
    };
    const std::string *scheduler_extension = extension(ChipSwimlaneExtensionSection::SchedulerRecords);
    const std::string *aicore_tasks_extension = extension(ChipSwimlaneExtensionSection::AicoreTasks);
    const std::string *scheduler_tasks_extension = extension(ChipSwimlaneExtensionSection::SchedulerTasks);
    const std::string *aicpu_lifecycle_extension = extension(ChipSwimlaneExtensionSection::AicpuLifecycleRecords);

    // Every stream is independently useful for DFX. In particular, a legal
    // HBG can contain only host-side dummy/hidden-allocation records and no
    // AICore dispatch at all.
    bool has_any_records = !host_submit_records_.empty() || !host_upload_records_.empty() ||
                           clock_correlation_session_.started() ||
                           std::any_of(json_extensions_.begin(), json_extensions_.end(), [](const auto &value) {
                               return !value.empty();
                           });
    for (const auto &core_records : collected_perf_records_) {
        if (!core_records.empty()) {
            has_any_records = true;
            break;
        }
    }
    if (!has_any_records) {
        for (const auto &ac_records : collected_aicore_records_) {
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
    const bool has_aicpu_orch_phases = any_phase_records(collected_orch_phase_records_);
    const bool has_aicpu_scheduler_records = any_phase_records(collected_sched_phase_records_);
    if (scheduler_extension != nullptr && has_aicpu_scheduler_records) {
        LOG_ERROR("Both runtime and AICPU scheduler records are present; refusing ambiguous export");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    const bool has_aicore_tasks = any_phase_records(collected_aicore_records_);
    const bool has_platform_scheduler_tasks = any_phase_records(collected_perf_records_);
    if ((aicore_tasks_extension != nullptr && has_aicore_tasks) ||
        (scheduler_tasks_extension != nullptr && has_platform_scheduler_tasks)) {
        LOG_ERROR("Both runtime and platform task records are present; refusing ambiguous export");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    has_any_records = has_any_records || any_phase_records(collected_sched_phase_records_) || has_aicpu_orch_phases;
    if (!has_any_records) {
        LOG_WARN("Warning: No performance data to export.");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (has_aicpu_orch_phases && host_orchestrated_) {
        LOG_ERROR("Both host and AICPU orchestrator records are present; refusing mixed clock-domain export");
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    std::error_code ec;
    std::filesystem::create_directories(output_prefix_, ec);
    if (ec) {
        LOG_ERROR("Error: Failed to create output directory %s: %s", output_prefix_.c_str(), ec.message().c_str());
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    std::string filepath = output_prefix_ + "/chip_swimlane_records.json";
    std::ofstream outfile(filepath);
    if (!outfile.is_open()) {
        LOG_ERROR("Error: Failed to open file: %s", filepath.c_str());
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    int chip_swimlane_level = static_cast<int>(chip_swimlane_level_);

    outfile << "{\n";
    outfile << "  \"chip_swimlane_level\": " << chip_swimlane_level << ",\n";

    // metadata: everything python needs that isn't in a per-record stream.
    // clock_freq_hz drives the cycles→µs conversion (a2a3 = 50 MHz, a5 =
    // 1 GHz — must come from the host, not be hardcoded in python).
    outfile << "  \"metadata\": {\n";
    outfile << "    \"clock_freq_hz\": " << PLATFORM_PROF_SYS_CNT_FREQ << ",\n";
    outfile << "    \"num_cores\": " << num_aicore_ << ",\n";
    outfile << "    \"core_types\": [";
    for (int i = 0; i < num_aicore_; i++) {
        CoreType ct = (i < static_cast<int>(core_types_.size())) ? core_types_[i] : CoreType::AIV;
        if (i > 0) outfile << ", ";
        outfile << "\"" << ((ct == CoreType::AIC) ? "aic" : "aiv") << "\"";
    }
    outfile << "]";
    if (host_phase_records_present_) {
        // Earliest of both projections: an upload segment can start before the
        // first submit, and a negative offset from the origin is not renderable.
        uint64_t host_origin_ns = 0;
        for (const auto *population : {&host_submit_records_, &host_upload_records_}) {
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
        const bool host_record_count_matches = host_submit_records_.size() == host_phase_submitted_tasks_;
        const bool host_capture_complete = host_phase_dropped_records_ == 0 && host_record_count_matches;
        const char *host_capture_status = host_capture_complete           ? "complete" :
                                          host_phase_dropped_records_ > 0 ? "dropped" :
                                                                            "incomplete";
        outfile << ",\n    \"host_capture\": {\"status\": \"" << host_capture_status
                << "\", \"expected_records\": " << host_phase_submitted_tasks_
                << ", \"recorded_records\": " << host_submit_records_.size()
                << ", \"pool_records\": " << host_phase_total_records_
                << ", \"dropped_records\": " << host_phase_dropped_records_ << ", \"error\": ";
        if (host_capture_complete) {
            outfile << "null}";
        } else if (host_phase_dropped_records_ > 0) {
            outfile << "\"pool_overflow\"}";
        } else {
            outfile << "\"record_count_mismatch\"}";
        }
    }
    if (host_phase_records_present_ || clock_correlation_session_.started()) {
        const std::string host_clock_domain_id = linux_boot_clock_domain_id();
        if (!host_clock_domain_id.empty()) {
            outfile << ",\n    \"host_clock_domain_id\": \"" << host_clock_domain_id << "\"";
        }
    }
    if (clock_correlation_session_.started()) {
        uint64_t host_timeline_origin_ns = 0;
        for (const auto &sample : clock_correlation_session_.samples()) {
            if (sample.position != simpler::dfx::ClockAnchorPosition::HostOrchestrationBegin || !sample.valid()) {
                continue;
            }
            const uint64_t midpoint = sample.host_before_ns + (sample.host_after_ns - sample.host_before_ns) / 2;
            if (host_timeline_origin_ns == 0 || midpoint < host_timeline_origin_ns) {
                host_timeline_origin_ns = midpoint;
            }
        }
        if (host_timeline_origin_ns != 0) {
            outfile << ",\n    \"host_timeline_origin_ns\": " << host_timeline_origin_ns;
        }
        outfile << ",\n    \"clock_anchors\": {";
        outfile << "\n      \"provider\": \"" << clock_correlation_session_.provider_name() << "\",";
        outfile << "\n      \"device_timestamp_unit\": \"syscnt_cycles\",";
        outfile << "\n      \"raw_device_timestamp_unit\": \"" << clock_correlation_session_.raw_device_timestamp_unit()
                << "\",";
        outfile << "\n      \"samples_per_position\": " << simpler::dfx::kClockAnchorSamplesPerPosition << ",";
        outfile << "\n      \"samples\": [";
        bool first_anchor = true;
        for (const auto &sample : clock_correlation_session_.samples()) {
            if (!first_anchor) outfile << ",";
            const uint64_t rtt_ns =
                sample.host_after_ns >= sample.host_before_ns ? sample.host_after_ns - sample.host_before_ns : 0;
            outfile << "\n        {\"position\": \"" << simpler::dfx::clock_anchor_position_name(sample.position)
                    << "\", \"sample_idx\": " << sample.sample_idx << ", \"host_before_ns\": " << sample.host_before_ns
                    << ", \"raw_device_timestamp\": ";
            if (sample.raw_device_timestamp == 0) {
                outfile << "null";
            } else {
                outfile << sample.raw_device_timestamp;
            }
            outfile << ", \"device_cycles\": ";
            if (sample.device_cycles == 0) {
                outfile << "null";
            } else {
                outfile << sample.device_cycles;
            }
            outfile << ", \"host_after_ns\": " << sample.host_after_ns << ", \"rtt_ns\": " << rtt_ns
                    << ", \"uncertainty_ns\": " << (rtt_ns + 1) / 2 << ", \"error\": ";
            if (sample.error_stage == simpler::dfx::ClockAnchorErrorStage::None && sample.error_code == 0) {
                outfile << "null";
            } else {
                outfile << "{\"stage\": \"" << simpler::dfx::clock_anchor_error_stage_name(sample.error_stage)
                        << "\", \"code\": " << sample.error_code << "}";
            }
            outfile << "}";
            first_anchor = false;
        }
        if (!first_anchor) outfile << "\n      ";
        outfile << "]\n    }";
    }
    if (!core_to_thread_.empty()) {
        outfile << ",\n    \"core_to_thread\": [";
        for (size_t i = 0; i < core_to_thread_.size(); i++) {
            if (i > 0) outfile << ", ";
            outfile << static_cast<int>(core_to_thread_[i]);
        }
        outfile << "]";
    }
    outfile << "\n  },\n";

    // Per-stream raw records. Flat array of tuples — compact at scale (a real
    // PA trace has ~100K records, and per-field JSON keys would dominate the
    // file size). Column order is documented in the schema comment at the top
    // of swimlane_converter.py's v2 reader.
    //
    //   aicore_tasks: [core_id, task_token_raw, reg_task_id, start_cycles, end_cycles, receive_to_start_cycles]
    //   scheduler_tasks.records: [core_id, reg_task_id, dispatch_cycles, finish_cycles]
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
            for (size_t core_idx = 0; core_idx < collected_aicore_records_.size(); core_idx++) {
                for (const auto &r : collected_aicore_records_[core_idx]) {
                    if (!first) outfile << ",";
                    outfile << "\n    [" << core_idx << ", " << r.task_token_raw << ", " << r.reg_task_id << ", "
                            << r.start_time << ", " << r.end_time << ", " << r.receive_to_start_cycles << "]";
                    first = false;
                    total++;
                }
            }
            if (!first) outfile << "\n  ";
            outfile << "]";
            LOG_INFO("  aicore_tasks: %zu records", total);
        }
    }
    if (chip_swimlane_level_ >= ChipSwimlaneLevel::SCHEDULE_TIMING) {
        outfile << ",\n  \"" << chip_swimlane_extension_section_name(ChipSwimlaneExtensionSection::SchedulerTasks)
                << "\": ";
        if (scheduler_tasks_extension != nullptr) {
            outfile << *scheduler_tasks_extension;
        } else {
            outfile << "{\n    \"schema_version\": 1,\n    \"producer\": \"aicpu\",\n    \"records\": [";
            bool first = true;
            size_t total = 0;
            for (size_t core_idx = 0; core_idx < collected_perf_records_.size(); core_idx++) {
                for (const auto &r : collected_perf_records_[core_idx]) {
                    if (!first) outfile << ",";
                    outfile << "\n    [" << core_idx << ", " << r.reg_task_id << ", " << r.dispatch_time << ", "
                            << r.finish_time << "]";
                    first = false;
                    total++;
                }
            }
            if (!first) outfile << "\n    ";
            outfile << "]\n  }";
            LOG_INFO("  scheduler_tasks: %zu AICPU records", total);
        }
    }

    if (chip_swimlane_level_ >= ChipSwimlaneLevel::SCHED_PHASES) {
        outfile << ",\n  \"" << chip_swimlane_extension_section_name(ChipSwimlaneExtensionSection::SchedulerRecords)
                << "\": ";
        if (scheduler_extension != nullptr) {
            outfile << *scheduler_extension;
        } else {
            std::vector<uint32_t> dropped_records(collected_sched_phase_records_.size());
            for (size_t t = 0; t < collected_sched_phase_records_.size(); ++t) {
                const auto *pool = get_sched_phase_buffer_state(shm_host_, static_cast<int>(t));
                dropped_records[t] = pool->head.dropped_record_count;
            }
            chip_swimlane_write_scheduler_records(
                outfile, collected_sched_phase_records_, dropped_records, SIMPLER_RUNTIME_NAME
            );
        }

        if (has_aicpu_orch_phases) {
            size_t orch_lanes = static_cast<size_t>(get_chip_swimlane_header(shm_host_)->num_orch_phase_threads);
            if (orch_lanes == 0 || orch_lanes > collected_orch_phase_records_.size()) {
                orch_lanes = collected_orch_phase_records_.size();
            }
            outfile << ",\n  \"aicpu_orchestrator_phases\": [\n";
            for (size_t t = 0; t < orch_lanes; t++) {
                outfile << "    [";
                bool first = true;
                for (const auto &pr : collected_orch_phase_records_[t]) {
                    if (!first) outfile << ",";
                    outfile << "\n      {\"submit_idx\": " << pr.submit_idx << ", \"task_id\": " << pr.task_id
                            << ", \"start_cycles\": " << pr.start_time << ", \"end_cycles\": " << pr.end_time << "}";
                    first = false;
                }
                if (!first) outfile << "\n    ";
                outfile << "]";
                if (t < orch_lanes - 1) outfile << ",";
                outfile << "\n";
            }
            outfile << "  ]";
        }
        if (!host_submit_records_.empty()) {
            outfile << ",\n  \"host_orchestrator_phases\": [[";
            bool first = true;
            for (const auto &record : host_submit_records_) {
                if (!first) outfile << ",";
                outfile << "\n      {\"submit_idx\": " << record.index << ", \"task_id\": " << record.payload
                        << ", \"start_host_ns\": " << record.start_ns << ", \"end_host_ns\": " << record.end_ns << "}";
                first = false;
            }
            if (!first) outfile << "\n    ";
            outfile << "]]";
        }
        if (!host_upload_records_.empty()) {
            outfile << ",\n  \"host_device_uploads\": [";
            bool first = true;
            for (const auto &record : host_upload_records_) {
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

    // Stop mgmt + collector threads if the caller didn't already (idempotent).
    stop();

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
    clock_correlation_session_.reset();
    perf_records_by_collector_.clear();
    aicore_records_by_collector_.clear();
    sched_phase_records_by_collector_.clear();
    orch_phase_records_by_collector_.clear();
    collector_counters_.clear();
    core_to_thread_.clear();
    has_phase_data_ = false;
    total_perf_collected_ = 0;
    total_sched_phase_collected_ = 0;
    total_orch_phase_collected_ = 0;
    collector_shards_merged_ = false;
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
