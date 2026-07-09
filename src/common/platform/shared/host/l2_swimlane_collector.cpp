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
 * @file l2_swimlane_collector.cpp
 * @brief Performance data collector implementation. The mgmt-thread + buffer-pool
 *        machinery lives in profiling_common::BufferPoolManager parameterized by
 *        L2SwimlaneModule (host/l2_swimlane_collector.h); the poll loop lives in
 *        profiling_common::ProfilerBase. This file owns the per-buffer
 *        on_buffer_collected callback and the export logic.
 */

#include "host/l2_swimlane_collector.h"

#include <cassert>
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

// =============================================================================
// L2SwimlaneCollector Implementation
// =============================================================================

// Sched / orch phase records route through separate BufferKinds; no
// parse-time discriminator function is needed (the device-side type tag is
// the source of truth).

L2SwimlaneCollector::~L2SwimlaneCollector() {
    stop();
    if (shm_host_ != nullptr) {
        LOG_WARN("L2SwimlaneCollector destroyed without finalize()");
    }
}

int L2SwimlaneCollector::initialize(
    int num_aicore, int aicpu_thread_num, int device_id, L2SwimlaneLevel l2_swimlane_level,
    const L2SwimlaneAllocCallback &alloc_cb, L2SwimlaneRegisterCallback register_cb,
    const L2SwimlaneFreeCallback &free_cb, const std::string &output_prefix
) {
    if (shm_host_ != nullptr) {
        LOG_ERROR("L2SwimlaneCollector already initialized");
        return -1;
    }

    // register_cb may legitimately be null on simulation / non-SVM platforms;
    // alloc and free callbacks are mandatory. Matches dep_gen / pmu / scope_stats.
    if (alloc_cb == nullptr || free_cb == nullptr) {
        LOG_ERROR("L2SwimlaneCollector::initialize: alloc_cb/free_cb must be non-null");
        return -1;
    }

    LOG_INFO_V0("Initializing performance profiling");

    if (num_aicore <= 0 || num_aicore > PLATFORM_MAX_CORES) {
        LOG_ERROR("Invalid number of AICores: %d (max=%d)", num_aicore, PLATFORM_MAX_CORES);
        return -1;
    }

    num_aicore_ = num_aicore;
    aicpu_thread_num_ = aicpu_thread_num;
    l2_swimlane_level_ = l2_swimlane_level;
    output_prefix_ = output_prefix;
    total_perf_collected_ = 0;
    total_sched_phase_collected_ = 0;
    total_orch_phase_collected_ = 0;
    has_phase_data_ = false;
    collector_shards_merged_ = false;

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
    size_t total_size = calc_perf_data_size_with_phases(num_aicore, num_phase_threads, num_phase_threads);

    LOG_DEBUG("Shared memory allocation plan:");
    LOG_DEBUG("  Number of cores:      %d", num_aicore);
    LOG_DEBUG("  Header size:          %zu bytes", sizeof(L2SwimlaneDataHeader));
    LOG_DEBUG("  L2SwimlaneAicpuTaskPool size: %zu bytes each", sizeof(L2SwimlaneAicpuTaskPool));
    LOG_DEBUG("  L2SwimlaneAicpuSchedPhasePool size: %zu bytes each", sizeof(L2SwimlaneAicpuSchedPhasePool));
    LOG_DEBUG("  L2SwimlaneAicpuOrchPhasePool size:  %zu bytes each", sizeof(L2SwimlaneAicpuOrchPhasePool));
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
        return -1;
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
    L2SwimlaneDataHeader *header = get_l2_swimlane_header(perf_host_ptr);

    for (int t = 0; t < PLATFORM_MAX_AICPU_THREADS; t++) {
        memset(header->queues[t], 0, sizeof(header->queues[t]));
        header->queue_heads[t] = 0;
        header->queue_tails[t] = 0;
    }

    header->num_cores = num_aicore;
    header->l2_swimlane_level = static_cast<uint32_t>(l2_swimlane_level_);
    // Phase metadata: must be zero-initialized here. alloc_cb returns
    // uninitialized device memory; AICPU only writes these fields when
    // phase init runs (level >= SCHED_PHASES). Without zeroing, lower
    // levels (AICORE_TIMING / AICPU_TIMING) leave garbage that
    // for_each_instance iterates as `num_sched_phase_threads` /
    // `num_orch_phase_threads`, walking off the end of the allocated pool
    // array → segfault. The host-side reader (read_phase_header_metadata)
    // and BufferPoolManager replenish loop both gate on these counts being
    // sane values.
    header->num_sched_phase_threads = 0;
    header->num_orch_phase_threads = 0;
    header->num_phase_cores = 0;
    memset(header->core_to_thread, -1, sizeof(header->core_to_thread));

    LOG_DEBUG("Initialized L2SwimlaneDataHeader:");
    LOG_DEBUG("  num_cores:              %d", header->num_cores);
    LOG_DEBUG("  l2_swimlane_level: %u", header->l2_swimlane_level);
    LOG_DEBUG("  buffer_capacity:        %d", PLATFORM_PROF_BUFFER_SIZE);
    LOG_DEBUG("  queue capacity:         %d", PLATFORM_PROF_READYQUEUE_SIZE);

    // Step 5: Initialize L2SwimlaneAicpuTaskPools. Seed as many buffers as
    // the device-side free_queue can hold; any remaining buffers stay in the
    // host recycled pool.
    for (int i = 0; i < num_aicore; i++) {
        L2SwimlaneAicpuTaskPool *state = get_perf_buffer_state(perf_host_ptr, i);
        memset(state, 0, sizeof(L2SwimlaneAicpuTaskPool));

        state->free_queue.head = 0;
        state->free_queue.tail = 0;
        state->head.current_buf_ptr = 0;
        state->head.current_buf_seq = 0;

        const int initial_free_count = (PLATFORM_PROF_BUFFERS_PER_CORE < PLATFORM_PROF_SLOT_COUNT) ?
                                           PLATFORM_PROF_BUFFERS_PER_CORE :
                                           PLATFORM_PROF_SLOT_COUNT;
        for (int s = 0; s < PLATFORM_PROF_BUFFERS_PER_CORE; s++) {
            void *host_buf_ptr = nullptr;
            void *dev_buf_ptr = alloc_paired_buffer(sizeof(L2SwimlaneAicpuTaskBuffer), &host_buf_ptr);
            if (dev_buf_ptr == nullptr) {
                LOG_ERROR("Failed to allocate L2SwimlaneAicpuTaskBuffer for core %d, buffer %d", i, s);
                return -1;
            }
            L2SwimlaneAicpuTaskBuffer *buf = reinterpret_cast<L2SwimlaneAicpuTaskBuffer *>(host_buf_ptr);
            memset(buf, 0, sizeof(L2SwimlaneAicpuTaskBuffer));
            buf->count = 0;

            if (s < initial_free_count) {
                state->free_queue.buffer_ptrs[s] = reinterpret_cast<uint64_t>(dev_buf_ptr);
            } else {
                manager_.push_recycled(static_cast<int>(ProfBufferType::AICPU_TASK), dev_buf_ptr);
            }
        }
        wmb();
        state->free_queue.tail = static_cast<uint32_t>(initial_free_count);
        wmb();
    }

    // Step 5b: Initialize L2SwimlaneAicoreTaskPools — per-core AICore rotation
    // channel + buffer pool. Same SPSC pattern as the AICPU pool above.
    for (int i = 0; i < num_aicore; i++) {
        L2SwimlaneAicoreTaskPool *ac_state = get_aicore_buffer_state(perf_host_ptr, num_aicore, i);
        memset(ac_state, 0, sizeof(L2SwimlaneAicoreTaskPool));

        const int initial_free_count = (PLATFORM_AICORE_BUFFERS_PER_CORE < PLATFORM_PROF_SLOT_COUNT) ?
                                           PLATFORM_AICORE_BUFFERS_PER_CORE :
                                           PLATFORM_PROF_SLOT_COUNT;
        for (int s = 0; s < PLATFORM_AICORE_BUFFERS_PER_CORE; s++) {
            void *host_buf_ptr = nullptr;
            void *dev_buf_ptr = alloc_paired_buffer(sizeof(L2SwimlaneAicoreTaskBuffer), &host_buf_ptr);
            if (dev_buf_ptr == nullptr) {
                LOG_ERROR("Failed to allocate L2SwimlaneAicoreTaskBuffer for core %d, buffer %d", i, s);
                return -1;
            }
            L2SwimlaneAicoreTaskBuffer *buf = reinterpret_cast<L2SwimlaneAicoreTaskBuffer *>(host_buf_ptr);
            memset(buf, 0, sizeof(L2SwimlaneAicoreTaskBuffer));
            buf->count = 0;

            if (s < initial_free_count) {
                ac_state->free_queue.buffer_ptrs[s] = reinterpret_cast<uint64_t>(dev_buf_ptr);
            } else {
                manager_.push_recycled(static_cast<int>(ProfBufferType::AICORE_TASK), dev_buf_ptr);
            }
        }
        wmb();
        ac_state->free_queue.tail = static_cast<uint32_t>(initial_free_count);
        wmb();
    }
    LOG_DEBUG(
        "Initialized buffer pools: %d L2SwimlaneAicpuTaskBuffers/core + %d L2SwimlaneAicoreTaskBuffers/core "
        "(seeded up to PLATFORM_PROF_SLOT_COUNT free_queue slots, rest in recycled pool)",
        PLATFORM_PROF_BUFFERS_PER_CORE, PLATFORM_AICORE_BUFFERS_PER_CORE
    );

    // Step 5c: Standalone uint64_t[num_aicore] table that will hold per-core
    // L2SwimlaneActiveHead device addresses. Host only allocates the bytes and
    // hands the device pointer to AICPU via KernelArgs::l2_swimlane_aicore_rotation_table;
    // AICPU itself fills the entries inside `l2_swimlane_aicpu_init` (it has
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
            LOG_ERROR("Failed to allocate l2_swimlane_aicore_rotation_table (rotation) table (%zu bytes)", table_bytes);
            return -1;
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
    auto init_phase_pools = [&](auto *buffer_tag, L2SwimlaneAicpuTaskPool *(*get_state)(void *, int, int),
                                int state_count, int buffer_count, int buffers_per_thread, ProfBufferType recycle_kind,
                                const char *kind_label) -> int {
        using Buffer = std::remove_pointer_t<decltype(buffer_tag)>;
        constexpr size_t buffer_bytes = sizeof(Buffer);
        for (int t = 0; t < state_count; t++) {
            auto *state = get_state(perf_host_ptr, num_aicore, t);
            memset(state, 0, sizeof(L2SwimlaneAicpuTaskPool));
            if (t >= buffer_count) continue;  // zeroed state only; no buffers (unused slot)
            const int initial_free_count =
                (buffers_per_thread < PLATFORM_PROF_SLOT_COUNT) ? buffers_per_thread : PLATFORM_PROF_SLOT_COUNT;
            for (int s = 0; s < buffers_per_thread; s++) {
                void *host_buf_ptr = nullptr;
                void *dev_buf_ptr = alloc_paired_buffer(buffer_bytes, &host_buf_ptr);
                if (dev_buf_ptr == nullptr) {
                    LOG_ERROR("Failed to allocate %s phase buffer for thread %d, slot %d", kind_label, t, s);
                    return -1;
                }
                // Zero only the `count` word at the buffer's tail, using the
                // matching Buffer type. The records payload is overwritten by
                // AICPU on first use.
                reinterpret_cast<Buffer *>(host_buf_ptr)->count = 0;
                if (s < initial_free_count) {
                    state->free_queue.buffer_ptrs[s] = reinterpret_cast<uint64_t>(dev_buf_ptr);
                } else {
                    manager_.push_recycled(static_cast<int>(recycle_kind), dev_buf_ptr);
                }
            }
            wmb();
            state->free_queue.tail = static_cast<uint32_t>(initial_free_count);
            wmb();
        }
        return 0;
    };

    // Sched: actual scheduler-thread count is unknown at host-alloc time, so
    // size buffers to the platform max. Orch: a single instance (pool 0), so
    // allocate buffers for just one pool while still zeroing all MAX states.
    if (init_phase_pools(
            static_cast<L2SwimlaneAicpuSchedPhaseBuffer *>(nullptr), get_sched_phase_buffer_state,
            /*state_count=*/num_phase_threads, /*buffer_count=*/num_phase_threads,
            /*buffers_per_thread=*/PLATFORM_PROF_SCHED_BUFFERS_PER_THREAD, ProfBufferType::AICPU_SCHED_PHASE, "sched"
        ) != 0) {
        return -1;
    }
    auto orch_get_state = [](void *base, int n_cores, int t) {
        return get_orch_phase_buffer_state(base, n_cores, t);
    };
    if (init_phase_pools(
            static_cast<L2SwimlaneAicpuOrchPhaseBuffer *>(nullptr), orch_get_state,
            /*state_count=*/num_phase_threads, /*buffer_count=*/1,
            /*buffers_per_thread=*/PLATFORM_PROF_ORCH_BUFFERS_PER_THREAD, ProfBufferType::AICPU_ORCH_PHASE, "orch"
        ) != 0) {
        return -1;
    }
    LOG_DEBUG(
        "Initialized %d sched (%d buf/thread) + 1 orch (%d buf) PhaseBufferStates", num_phase_threads,
        PLATFORM_PROF_SCHED_BUFFERS_PER_THREAD, PLATFORM_PROF_ORCH_BUFFERS_PER_THREAD
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
    // kernel_args.l2_swimlane_data_base (read back via get_l2_swimlane_setup_device_ptr()).
    LOG_DEBUG("L2 swimlane device base = 0x%lx", reinterpret_cast<uint64_t>(perf_dev_ptr));

    // Reserve the per-core / per-thread record vectors while the rollback guard
    // is still armed, so a std::bad_alloc here unwinds through the guard and
    // frees every buffer. Publication of the device pointers and the memory
    // context is deferred to after commit (below): otherwise a throw here would
    // leave perf_shared_mem_dev_ dangling and shm_host_ non-null, which would
    // make is_initialized() report true and finalize() double-free.
    reset_collector_shards();

    LOG_INFO_V0("Performance profiling initialized (dynamic buffer mode)");
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

size_t L2SwimlaneCollector::normalize_collector_shard(int collector_shard) const {
    const size_t shard_count = collector_counters_.size();
    const bool valid_shard = collector_shard >= 0 && static_cast<size_t>(collector_shard) < shard_count;
    if (!valid_shard) {
        assert(false && "collector_shard out of range");
        return shard_count;
    }
    return static_cast<size_t>(collector_shard);
}

void L2SwimlaneCollector::reset_collector_shards() {
    const size_t shard_count = static_cast<size_t>(L2SwimlaneModule::kCollectorThreadCount);

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

void L2SwimlaneCollector::merge_collector_shards() {
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

void L2SwimlaneCollector::copy_perf_buffer(const ReadyBufferInfo &info, int collector_shard) {
    L2SwimlaneAicpuTaskBuffer *buf = reinterpret_cast<L2SwimlaneAicpuTaskBuffer *>(info.host_buffer_ptr);
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

void L2SwimlaneCollector::copy_sched_phase_buffer(const ReadyBufferInfo &info, int collector_shard) {
    auto *buf = reinterpret_cast<L2SwimlaneAicpuSchedPhaseBuffer *>(info.host_buffer_ptr);
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

void L2SwimlaneCollector::copy_orch_phase_buffer(const ReadyBufferInfo &info, int collector_shard) {
    auto *buf = reinterpret_cast<L2SwimlaneAicpuOrchPhaseBuffer *>(info.host_buffer_ptr);
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
void L2SwimlaneCollector::copy_aicore_buffer(const ReadyBufferInfo &info, int collector_shard) {
    L2SwimlaneAicoreTaskBuffer *buf = reinterpret_cast<L2SwimlaneAicoreTaskBuffer *>(info.host_buffer_ptr);
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
            const L2SwimlaneAicoreTaskRecord &r = buf->records[i];
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

void L2SwimlaneCollector::on_buffer_collected(const ReadyBufferInfo &info, int collector_shard) {
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

void L2SwimlaneCollector::reconcile_counters() {
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
            L2SwimlaneAicpuTaskPool *state = get_state(i);
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
                "L2Swimlane reconcile: %s %d has un-flushed %s buffer (current_buf_ptr=0x%lx, count=%u) "
                "after stop() — device flush failed",
                unit_name, i, kind, static_cast<unsigned long>(buf_ptr), count
            );
            leftover_active++;
        }

        uint64_t total_device = 0;
        uint64_t dropped_device = 0;
        for (int i = 0; i < unit_count; i++) {
            L2SwimlaneAicpuTaskPool *state = get_state(i);
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
                "L2Swimlane reconcile: %lu %s records dropped on device side.",
                static_cast<unsigned long>(dropped_device), kind
            );
        }
        uint64_t accounted = collected + dropped_device;
        if (accounted != total_device) {
            LOG_WARN(
                "L2Swimlane reconcile: %s count mismatch (collected=%lu + dropped=%lu != "
                "device_total=%lu, silent_loss=%ld)",
                kind, static_cast<unsigned long>(collected), static_cast<unsigned long>(dropped_device),
                static_cast<unsigned long>(total_device), static_cast<long>(total_device) - static_cast<long>(accounted)
            );
        } else {
            LOG_INFO_V0(
                "L2Swimlane reconcile: %s counts match (collected=%lu, dropped=%lu, device_total=%lu)", kind,
                static_cast<unsigned long>(collected), static_cast<unsigned long>(dropped_device),
                static_cast<unsigned long>(total_device)
            );
        }

        if (leftover_active > 0) {
            LOG_ERROR(
                "L2Swimlane reconcile: %d %s(s) had un-cleared %s current_buf_ptr — see prior errors", leftover_active,
                unit_name, kind
            );
        }
    };

    reconcile_one(
        "PERF", "core", num_aicore_,
        [this](int core_index) {
            return get_perf_buffer_state(shm_host_, core_index);
        },
        [](void *host_ptr) {
            return reinterpret_cast<L2SwimlaneAicpuTaskBuffer *>(host_ptr)->count;
        },
        sizeof(L2SwimlaneAicpuTaskBuffer), total_perf_collected_, /*optional=*/false
    );

    reconcile_one(
        "SCHED_PHASE", "thread", PLATFORM_MAX_AICPU_THREADS,
        [this](int thread_index) {
            return get_sched_phase_buffer_state(shm_host_, num_aicore_, thread_index);
        },
        [](void *host_ptr) {
            return reinterpret_cast<L2SwimlaneAicpuSchedPhaseBuffer *>(host_ptr)->count;
        },
        sizeof(L2SwimlaneAicpuSchedPhaseBuffer), total_sched_phase_collected_, /*optional=*/true
    );

    reconcile_one(
        "ORCH_PHASE", "thread", PLATFORM_MAX_AICPU_THREADS,
        [this](int thread_index) {
            return get_orch_phase_buffer_state(shm_host_, num_aicore_, thread_index);
        },
        [](void *host_ptr) {
            return reinterpret_cast<L2SwimlaneAicpuOrchPhaseBuffer *>(host_ptr)->count;
        },
        sizeof(L2SwimlaneAicpuOrchPhaseBuffer), total_orch_phase_collected_, /*optional=*/true
    );
}

void L2SwimlaneCollector::read_phase_header_metadata() {
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

    L2SwimlaneDataHeader *header = get_l2_swimlane_header(shm_host_);

    int num_sched = static_cast<int>(header->num_sched_phase_threads);
    int num_orch = static_cast<int>(header->num_orch_phase_threads);
    if (num_sched == 0 && num_orch == 0) {
        LOG_INFO_V0("No phase profiling data found (sched/orch phase thread counts both 0; phase init never ran)");
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
    // aicpu_thread_num_ is >= 1 (DeviceRunner::run validates launch_aicpu_num in
    // [1, PLATFORM_MAX_AICPU_THREADS] before initialize()), so the subtraction
    // can't go negative. This is a log-only display value, never an index.
    const int orch_thread = aicpu_thread_num_ - 1;
    LOG_INFO_V0(
        "Collecting phase metadata: scheduler threads 0-%d, orchestrator thread %d", num_sched - 1, orch_thread
    );

    for (size_t t = 0; t < collected_sched_phase_records_.size(); t++) {
        if (!collected_sched_phase_records_[t].empty()) {
            LOG_INFO_V0("  Sched thread %zu: %zu records", t, collected_sched_phase_records_[t].size());
        }
    }
    for (size_t t = 0; t < collected_orch_phase_records_.size(); t++) {
        if (!collected_orch_phase_records_[t].empty()) {
            LOG_INFO_V0("  Orch thread %d: %zu records", orch_thread, collected_orch_phase_records_[t].size());
        }
    }

    // has_phase_data_ is set by copy_sched_phase_buffer / copy_orch_phase_buffer
    // during the drain — every push goes through those call sites and toggles
    // the flag. No re-scan needed here.

    // Core-to-thread mapping (header-resident; not buffered).
    int num_phase_cores = static_cast<int>(header->num_phase_cores);
    if (num_phase_cores > 0 && num_phase_cores <= PLATFORM_MAX_CORES) {
        core_to_thread_.assign(header->core_to_thread, header->core_to_thread + num_phase_cores);
        LOG_INFO_V0("  Core-to-thread mapping: %d cores", num_phase_cores);
    }

    LOG_INFO_V0("Phase metadata collection complete: has_phase_data=%s", has_phase_data_ ? "yes" : "no");
}

void L2SwimlaneCollector::set_core_types(const CoreType *types, int n) {
    if (types == nullptr || n <= 0) {
        core_types_.clear();
        return;
    }
    core_types_.assign(types, types + n);
}

// JSON v2 emit: the host now dumps raw cycle-domain per-stream records plus
// metadata, and `swimlane_converter.py` performs the join (AICore↔AICPU on
// reg_task_id, base_time normalization, cycles→µs conversion, sort, core_type
// lookup, func_id resolution against deps.json). Moving the join into Python
// makes the schema easy to evolve without round-tripping through C++ + a
// rebuild, and shrinks this file to a pure dump.
int L2SwimlaneCollector::export_swimlane_json() {
    if (shm_host_ == nullptr) {
        return -1;
    }
    merge_collector_shards();

    // Empty-export guard: nothing useful on disk if every per-stream source is
    // empty. AICPU_TIMING+ relies on `collected_perf_records_`; AICORE_TIMING
    // (level=1) relies on `collected_aicore_records_` alone.
    bool has_any_records = false;
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
    if (!has_any_records) {
        LOG_WARN("Warning: No performance data to export.");
        return -1;
    }

    std::error_code ec;
    std::filesystem::create_directories(output_prefix_, ec);
    if (ec) {
        LOG_ERROR("Error: Failed to create output directory %s: %s", output_prefix_.c_str(), ec.message().c_str());
        return -1;
    }

    std::string filepath = output_prefix_ + "/l2_swimlane_records.json";
    std::ofstream outfile(filepath);
    if (!outfile.is_open()) {
        LOG_ERROR("Error: Failed to open file: %s", filepath.c_str());
        return -1;
    }

    int l2_swimlane_level = static_cast<int>(l2_swimlane_level_);

    outfile << "{\n";
    outfile << "  \"l2_swimlane_level\": " << l2_swimlane_level << ",\n";

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
    //   aicpu_tasks:  [core_id, reg_task_id, dispatch_cycles, finish_cycles]
    {
        // copy_aicore_buffer already drops r.start_time == 0 slots when
        // collecting from the device side, so no defensive filter here.
        outfile << "  \"aicore_tasks\": [";
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
        LOG_INFO_V0("  aicore_tasks: %zu records", total);
    }
    {
        outfile << ",\n  \"aicpu_tasks\": [";
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
        if (!first) outfile << "\n  ";
        outfile << "]";
        LOG_INFO_V0("  aicpu_tasks: %zu records", total);
    }

    // Phase records keep their per-thread sub-array shape so the python
    // consumer's existing iteration pattern (one thread per inner list) stays
    // unchanged; only the field names move from *_us to *_cycles.
    if (l2_swimlane_level_ >= L2SwimlaneLevel::SCHED_PHASES) {
        auto sched_phase_name = [](L2SwimlaneSchedPhaseKind kind) -> const char * {
            switch (kind) {
            case L2SwimlaneSchedPhaseKind::Complete:
                return "complete";
            case L2SwimlaneSchedPhaseKind::Dispatch:
                return "dispatch";
            case L2SwimlaneSchedPhaseKind::Release:
                return "release";
            case L2SwimlaneSchedPhaseKind::Dummy:
                return "dummy";
            case L2SwimlaneSchedPhaseKind::EarlyDispatch:
                return "early_dispatch";
            case L2SwimlaneSchedPhaseKind::Resolve:
                return "resolve";
            case L2SwimlaneSchedPhaseKind::DummyTask:
                return "dummy_task";
            }
            return "unknown";
        };

        auto emit_depth_array = [&outfile](const char *key, const int16_t arr[L2SWIMLANE_NUM_QUEUE_SHAPES]) {
            outfile << ", \"" << key << "\": [" << arr[0] << "," << arr[1] << "," << arr[2] << "]";
        };
        outfile << ",\n  \"aicpu_scheduler_phases\": [\n";
        for (size_t t = 0; t < collected_sched_phase_records_.size(); t++) {
            outfile << "    [";
            bool first = true;
            for (const auto &pr : collected_sched_phase_records_[t]) {
                if (!first) outfile << ",";
                outfile << "\n      {\"kind\": \"" << sched_phase_name(pr.kind) << "\""
                        << ", \"start_cycles\": " << pr.start_time << ", \"end_cycles\": " << pr.end_time
                        << ", \"loop_iter\": " << pr.loop_iter << ", \"tasks_processed\": " << pr.tasks_processed;
                if (pr.kind == L2SwimlaneSchedPhaseKind::Dispatch) {
                    outfile << ", \"pop_hit\": " << pr.pop_hit << ", \"pop_miss\": " << pr.pop_miss;
                }
                // Queue-depth snapshots — [AIC, AIV, MIX] per L2SwimlaneAicpuSchedPhaseRecord docstring.
                emit_depth_array("shared_at_start", pr.shared_depth_at_start);
                emit_depth_array("shared_at_end", pr.shared_depth_at_end);
                outfile << "}";
                first = false;
            }
            if (!first) outfile << "\n    ";
            outfile << "]";
            if (t < collected_sched_phase_records_.size() - 1) outfile << ",";
            outfile << "\n";
        }
        outfile << "  ]";

        bool has_orch_phases = false;
        if (l2_swimlane_level_ >= L2SwimlaneLevel::ORCH_PHASES) {
            for (const auto &v : collected_orch_phase_records_) {
                if (!v.empty()) {
                    has_orch_phases = true;
                    break;
                }
            }
        }
        if (has_orch_phases) {
            size_t orch_lanes = static_cast<size_t>(get_l2_swimlane_header(shm_host_)->num_orch_phase_threads);
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
    }

    outfile << "\n}\n";
    outfile.close();

    if (!outfile) {
        LOG_ERROR("Failed to write JSON file (stream error): %s", filepath.c_str());
        return -1;
    }

    LOG_INFO_V0("=== JSON Export Complete ===");
    LOG_INFO_V0("File: %s", filepath.c_str());

    return 0;
}

int L2SwimlaneCollector::finalize(L2SwimlaneUnregisterCallback unregister_cb, const L2SwimlaneFreeCallback &free_cb) {
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
    // init_l2_swimlane() invocations makes back-to-back tests on a reused
    // Worker fail at rc=8 from halHostRegister.

    // Free standalone l2_swimlane_aicore_rotation_table table
    release_one_buffer(aicore_ring_addr_table_dev_, unregister_cb, free_cb);
    aicore_ring_addr_table_dev_ = nullptr;

    // Release framework-owned buffers (recycled pools, done_queue, ready_queue).
    manager_.release_owned_buffers([this, unregister_cb, free_cb](void *p) {
        release_one_buffer(p, unregister_cb, free_cb);
    });

    // Per-core: current buffer + free_queue slots — these were owned by
    // the AICPU side, not the framework. Same drain pattern for both the
    // L2SwimlaneAicpuTaskBuffer pool and the L2SwimlaneAicoreTaskBuffer pool.
    auto drain_free_queue = [&](L2SwimlaneFreeQueue &fq) {
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
        L2SwimlaneAicpuTaskPool *state = get_perf_buffer_state(shm_host_, i);
        release_one_buffer(reinterpret_cast<void *>(state->head.current_buf_ptr), unregister_cb, free_cb);
        state->head.current_buf_ptr = 0;
        drain_free_queue(state->free_queue);

        L2SwimlaneAicoreTaskPool *ac_state = get_aicore_buffer_state(shm_host_, num_aicore_, i);
        release_one_buffer(reinterpret_cast<void *>(ac_state->head.current_buf_ptr), unregister_cb, free_cb);
        ac_state->head.current_buf_ptr = 0;
        drain_free_queue(ac_state->free_queue);
    }

    auto release_phase_pool = [&](L2SwimlaneAicpuTaskPool *state) {
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
        release_phase_pool(get_sched_phase_buffer_state(shm_host_, num_aicore_, t));
    }
    for (int t = 0; t < num_phase_threads; t++) {
        release_phase_pool(get_orch_phase_buffer_state(shm_host_, num_aicore_, t));
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
    // walking freed buffer state. Mirrors PMU/DepGen/TensorDump collectors.
    shm_host_ = nullptr;
    collected_perf_records_.clear();
    collected_aicore_records_.clear();
    collected_sched_phase_records_.clear();
    collected_orch_phase_records_.clear();
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
    clear_memory_context();

    LOG_DEBUG("Performance profiling cleanup complete");
    return 0;
}
