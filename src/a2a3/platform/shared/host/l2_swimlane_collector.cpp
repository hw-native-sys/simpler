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

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/memory_barrier.h"
#include "common/unified_log.h"

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

void *L2SwimlaneCollector::alloc_single_buffer(size_t size, void **host_ptr_out) {
    void *dev_ptr = alloc_cb_(size);
    if (dev_ptr == nullptr) {
        LOG_ERROR("Failed to allocate buffer (%zu bytes)", size);
        *host_ptr_out = nullptr;
        return nullptr;
    }

    if (register_cb_ != nullptr) {
        void *host_ptr = nullptr;
        int rc = register_cb_(dev_ptr, size, device_id_, &host_ptr);
        if (rc != 0 || host_ptr == nullptr) {
            LOG_ERROR("Buffer registration failed: %d", rc);
            *host_ptr_out = nullptr;
            return nullptr;
        }
        *host_ptr_out = host_ptr;
    } else {
        *host_ptr_out = dev_ptr;
    }

    // Register mapping so the BufferPoolManager can resolve dev→host
    manager_.register_mapping(dev_ptr, *host_ptr_out);
    return dev_ptr;
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

    // Stash the memory context on the base up-front so alloc_single_buffer
    // sees consistent values during init. shm_host_ stays nullptr until the
    // shm allocation succeeds — the nullptr guard makes a post-failure
    // start(tf) a no-op.
    set_memory_context(
        alloc_cb, register_cb, free_cb, /*copy_to=*/nullptr, /*copy_from=*/nullptr, /*shm_dev=*/nullptr,
        /*shm_host=*/nullptr, /*shm_size=*/0, device_id
    );

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

    // Step 2: Allocate shared memory for slot arrays
    void *perf_dev_ptr = alloc_cb(total_size);
    if (perf_dev_ptr == nullptr) {
        LOG_ERROR("Failed to allocate shared memory (%zu bytes)", total_size);
        return -1;
    }
    LOG_DEBUG("Allocated shared memory: %p", perf_dev_ptr);

    // Step 3: Register to host mapping (optional)
    void *perf_host_ptr = nullptr;
    if (register_cb != nullptr) {
        int rc = register_cb(perf_dev_ptr, total_size, device_id, &perf_host_ptr);
        if (rc != 0) {
            LOG_ERROR("Memory registration failed: %d", rc);
            return rc;
        }
        if (perf_host_ptr == nullptr) {
            LOG_ERROR("register_cb succeeded but returned null host_ptr");
            return -1;
        }
        LOG_DEBUG("Mapped to host memory: %p", perf_host_ptr);
    } else {
        perf_host_ptr = perf_dev_ptr;
        LOG_DEBUG("Simulation mode: host_ptr = dev_ptr = %p", perf_host_ptr);
    }

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

    // Step 5: Initialize L2SwimlaneAicpuTaskPools — 1 buffer per core in free_queue, rest to recycled pool
    for (int i = 0; i < num_aicore; i++) {
        L2SwimlaneAicpuTaskPool *state = get_perf_buffer_state(perf_host_ptr, i);
        memset(state, 0, sizeof(L2SwimlaneAicpuTaskPool));

        state->free_queue.head = 0;
        state->free_queue.tail = 0;
        state->head.current_buf_ptr = 0;
        state->head.current_buf_seq = 0;

        for (int s = 0; s < PLATFORM_PROF_BUFFERS_PER_CORE; s++) {
            void *host_buf_ptr = nullptr;
            void *dev_buf_ptr = alloc_single_buffer(sizeof(L2SwimlaneAicpuTaskBuffer), &host_buf_ptr);
            if (dev_buf_ptr == nullptr) {
                LOG_ERROR("Failed to allocate L2SwimlaneAicpuTaskBuffer for core %d, buffer %d", i, s);
                return -1;
            }
            L2SwimlaneAicpuTaskBuffer *buf = reinterpret_cast<L2SwimlaneAicpuTaskBuffer *>(host_buf_ptr);
            memset(buf, 0, sizeof(L2SwimlaneAicpuTaskBuffer));
            buf->count = 0;

            if (s == 0) {
                state->free_queue.buffer_ptrs[0] = reinterpret_cast<uint64_t>(dev_buf_ptr);
            } else {
                manager_.push_recycled(static_cast<int>(ProfBufferType::AICPU_TASK), dev_buf_ptr);
            }
        }
        wmb();
        state->free_queue.tail = 1;
        wmb();
    }

    // Step 5b: Initialize L2SwimlaneAicoreTaskPools — per-core AICore rotation
    // channel + buffer pool. Same SPSC pattern as the AICPU pool above.
    for (int i = 0; i < num_aicore; i++) {
        L2SwimlaneAicoreTaskPool *ac_state = get_aicore_buffer_state(perf_host_ptr, num_aicore, i);
        memset(ac_state, 0, sizeof(L2SwimlaneAicoreTaskPool));

        for (int s = 0; s < PLATFORM_AICORE_BUFFERS_PER_CORE; s++) {
            void *host_buf_ptr = nullptr;
            void *dev_buf_ptr = alloc_single_buffer(sizeof(L2SwimlaneAicoreTaskBuffer), &host_buf_ptr);
            if (dev_buf_ptr == nullptr) {
                LOG_ERROR("Failed to allocate L2SwimlaneAicoreTaskBuffer for core %d, buffer %d", i, s);
                return -1;
            }
            L2SwimlaneAicoreTaskBuffer *buf = reinterpret_cast<L2SwimlaneAicoreTaskBuffer *>(host_buf_ptr);
            memset(buf, 0, sizeof(L2SwimlaneAicoreTaskBuffer));
            buf->count = 0;

            if (s == 0) {
                ac_state->free_queue.buffer_ptrs[0] = reinterpret_cast<uint64_t>(dev_buf_ptr);
            } else {
                manager_.push_recycled(static_cast<int>(ProfBufferType::AICORE_TASK), dev_buf_ptr);
            }
        }
        wmb();
        ac_state->free_queue.tail = 1;
        wmb();
    }
    LOG_DEBUG(
        "Initialized buffer pools: %d L2SwimlaneAicpuTaskBuffers/core + %d L2SwimlaneAicoreTaskBuffers/core (1 in "
        "free_queue, "
        "rest in recycled pool)",
        PLATFORM_PROF_BUFFERS_PER_CORE, PLATFORM_AICORE_BUFFERS_PER_CORE
    );

    // Step 5c: Standalone uint64_t[num_aicore] table that will hold per-core
    // L2SwimlaneActiveHead device addresses. Host only allocates the bytes and
    // hands the device pointer to AICPU via KernelArgs::l2_swimlane_aicore_rotation_table;
    // AICPU itself fills the entries inside `l2_swimlane_aicpu_init` (it has
    // direct access to `&ac_state->head` device addresses, no
    // host-to-device translation needed). AICore reads
    // rotation_table[block_idx] at kernel entry.
    {
        size_t table_bytes = static_cast<size_t>(num_aicore) * sizeof(uint64_t);
        void *rotation_table_host = nullptr;
        void *rotation_table_dev = alloc_single_buffer(table_bytes, &rotation_table_host);
        if (rotation_table_dev == nullptr) {
            LOG_ERROR("Failed to allocate l2_swimlane_aicore_rotation_table (rotation) table (%zu bytes)", table_bytes);
            return -1;
        }
        aicore_ring_addr_table_dev_ = rotation_table_dev;
    }

    // Step 6: Initialize per-thread phase pools — both sched and orch. Each
    // pool is sized to PLATFORM_PROF_BUFFERS_PER_THREAD buffers (1 in
    // free_queue, rest in the recycled pool tagged by kind). Templated on the
    // concrete TypedBuffer so the `count` zero-store uses the matching layout
    // — sched and orch buffers have DIFFERENT sizes (40B vs 32B records),
    // so a single cast type for both would land the count store past the end
    // of the orch allocation and corrupt the heap.
    // state_count pool states are zeroed (so the host's [0, PLATFORM_MAX)
    // reconcile/iteration reads count=0 for unused slots); buffers are
    // allocated only for the first buffer_count pools. For sched the two are
    // equal; orch is a single instance (pool 0), so it zeroes all slots but
    // allocates buffers for just pool 0 — no buffers wasted on unused slots.
    auto init_phase_pools = [&](auto buffer_tag, L2SwimlaneAicpuTaskPool *(*get_state)(void *, int, int),
                                int state_count, int buffer_count, ProfBufferType recycle_kind,
                                const char *kind_label) -> int {
        using Buffer = typename decltype(buffer_tag)::type;
        constexpr size_t buffer_bytes = sizeof(Buffer);
        for (int t = 0; t < state_count; t++) {
            auto *state = get_state(perf_host_ptr, num_aicore, t);
            memset(state, 0, sizeof(L2SwimlaneAicpuTaskPool));
            if (t >= buffer_count) continue;  // zeroed state only; no buffers (unused slot)
            for (int s = 0; s < PLATFORM_PROF_BUFFERS_PER_THREAD; s++) {
                void *host_buf_ptr = nullptr;
                void *dev_buf_ptr = alloc_single_buffer(buffer_bytes, &host_buf_ptr);
                if (dev_buf_ptr == nullptr) {
                    LOG_ERROR("Failed to allocate %s phase buffer for thread %d, slot %d", kind_label, t, s);
                    return -1;
                }
                // Zero only the `count` word at the buffer's tail, using the
                // matching Buffer type. The records payload is overwritten by
                // AICPU on first use.
                reinterpret_cast<Buffer *>(host_buf_ptr)->count = 0;
                if (s == 0) {
                    state->free_queue.buffer_ptrs[0] = reinterpret_cast<uint64_t>(dev_buf_ptr);
                } else {
                    manager_.push_recycled(static_cast<int>(recycle_kind), dev_buf_ptr);
                }
            }
            wmb();
            state->free_queue.tail = 1;
            wmb();
        }
        return 0;
    };

    // Type tags so the templated lambda can deduce the buffer type without
    // having to spell out an explicit template argument (not portable on a
    // generic lambda before C++20 explicit template-parameter syntax).
    struct SchedTag {
        using type = L2SwimlaneAicpuSchedPhaseBuffer;
    };
    struct OrchTag {
        using type = L2SwimlaneAicpuOrchPhaseBuffer;
    };

    // Sched: actual scheduler-thread count is unknown at host-alloc time, so
    // size buffers to the platform max. Orch: a single instance (pool 0), so
    // allocate buffers for just one pool while still zeroing all MAX states.
    if (init_phase_pools(
            SchedTag{}, get_sched_phase_buffer_state, /*state_count=*/num_phase_threads,
            /*buffer_count=*/num_phase_threads, ProfBufferType::AICPU_SCHED_PHASE, "sched"
        ) != 0) {
        return -1;
    }
    auto orch_get_state = [](void *base, int n_cores, int t) {
        return get_orch_phase_buffer_state(base, n_cores, t);
    };
    if (init_phase_pools(
            OrchTag{}, orch_get_state, /*state_count=*/num_phase_threads, /*buffer_count=*/1,
            ProfBufferType::AICPU_ORCH_PHASE, "orch"
        ) != 0) {
        return -1;
    }
    LOG_DEBUG(
        "Initialized %d sched (+1 orch) PhaseBufferStates: 1 buffer/thread, %d in recycled pool each",
        num_phase_threads, PLATFORM_PROF_BUFFERS_PER_THREAD - 1
    );

    wmb();

    // Step 7: Stash device pointer for the caller to publish via
    // kernel_args.l2_swimlane_data_base (read back via get_l2_swimlane_setup_device_ptr()).
    LOG_DEBUG("L2 swimlane device base = 0x%lx", reinterpret_cast<uint64_t>(perf_dev_ptr));

    perf_shared_mem_dev_ = perf_dev_ptr;
    // Refresh memory context with the now-known SHM tuple. start(tf) (inherited)
    // gates on shm_host_, so this is the moment the collector becomes startable.
    set_memory_context(
        alloc_cb, register_cb, free_cb, /*copy_to=*/nullptr, /*copy_from=*/nullptr, perf_dev_ptr, perf_host_ptr,
        total_size, device_id
    );

    collected_perf_records_.assign(num_aicore_, {});
    collected_aicore_records_.assign(num_aicore_, {});
    collected_sched_phase_records_.assign(PLATFORM_MAX_AICPU_THREADS, {});
    collected_orch_phase_records_.assign(PLATFORM_MAX_AICPU_THREADS, {});

    LOG_INFO_V0("Performance profiling initialized (dynamic buffer mode)");
    return 0;
}

// ---------------------------------------------------------------------------
// ProfilerBase callbacks
// ---------------------------------------------------------------------------

void L2SwimlaneCollector::copy_perf_buffer(const ReadyBufferInfo &info) {
    L2SwimlaneAicpuTaskBuffer *buf = reinterpret_cast<L2SwimlaneAicpuTaskBuffer *>(info.host_buffer_ptr);
    rmb();
    uint32_t count = buf->count;
    if (count > PLATFORM_PROF_BUFFER_SIZE) {
        count = PLATFORM_PROF_BUFFER_SIZE;
    }
    uint32_t core_index = info.index;
    if (core_index < static_cast<uint32_t>(num_aicore_)) {
        for (uint32_t i = 0; i < count; i++) {
            collected_perf_records_[core_index].push_back(buf->records[i]);
        }
        total_perf_collected_ += count;
    }
}

void L2SwimlaneCollector::copy_sched_phase_buffer(const ReadyBufferInfo &info) {
    auto *buf = reinterpret_cast<L2SwimlaneAicpuSchedPhaseBuffer *>(info.host_buffer_ptr);
    rmb();
    uint32_t count = buf->count;
    if (count > static_cast<uint32_t>(PLATFORM_PHASE_RECORDS_PER_THREAD)) {
        count = PLATFORM_PHASE_RECORDS_PER_THREAD;
    }
    uint32_t tidx = info.index;
    if (tidx < collected_sched_phase_records_.size()) {
        for (uint32_t i = 0; i < count; i++) {
            collected_sched_phase_records_[tidx].push_back(buf->records[i]);
        }
        total_sched_phase_collected_ += count;
        if (count > 0) {
            has_phase_data_ = true;
        }
    }
}

void L2SwimlaneCollector::copy_orch_phase_buffer(const ReadyBufferInfo &info) {
    auto *buf = reinterpret_cast<L2SwimlaneAicpuOrchPhaseBuffer *>(info.host_buffer_ptr);
    rmb();
    uint32_t count = buf->count;
    if (count > static_cast<uint32_t>(PLATFORM_PHASE_RECORDS_PER_THREAD)) {
        count = PLATFORM_PHASE_RECORDS_PER_THREAD;
    }
    uint32_t tidx = info.index;
    if (tidx < collected_orch_phase_records_.size()) {
        for (uint32_t i = 0; i < count; i++) {
            collected_orch_phase_records_[tidx].push_back(buf->records[i]);
        }
        total_orch_phase_collected_ += count;
        if (count > 0) {
            has_phase_data_ = true;
        }
    }
}

// AICore record buffers arrive on the ready queue in per-core rotation order
// (AICPU enqueues them at PLATFORM_AICORE_BUFFER_SIZE dispatch boundaries +
// once at flush). Within a single buffer, AICore wrote records[0..buf->count)
// in the order tasks ran on that core (completion-before-dispatch invariant
// + AICPU stamps buf->count just before enqueue). Flattening in arrival
// order gives us the per-core task stream that join_aicore_records()
// indexes by reg_task_id.
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
void L2SwimlaneCollector::copy_aicore_buffer(const ReadyBufferInfo &info) {
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
    auto &dst = collected_aicore_records_[core_index];
    dst.reserve(dst.size() + count);
    uint32_t skipped = 0;
    for (uint32_t i = 0; i < count; i++) {
        const L2SwimlaneAicoreTaskRecord &r = buf->records[i];
        if (r.start_time == 0) {
            skipped++;
            continue;
        }
        dst.push_back(r);
    }
    if (skipped > 0) {
        LOG_WARN(
            "Core %u: skipped %u AICore record slot(s) with start_time=0 (race-window write or "
            "recycled-buffer tail). buf seq=%u count=%u",
            core_index, skipped, info.buffer_seq, count
        );
    }
}

void L2SwimlaneCollector::on_buffer_collected(const ReadyBufferInfo &info) {
    switch (info.type) {
    case ProfBufferType::AICPU_TASK:
        copy_perf_buffer(info);
        break;
    case ProfBufferType::AICPU_SCHED_PHASE:
        copy_sched_phase_buffer(info);
        break;
    case ProfBufferType::AICPU_ORCH_PHASE:
        copy_orch_phase_buffer(info);
        break;
    case ProfBufferType::AICORE_TASK:
        copy_aicore_buffer(info);
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
                             auto read_buf_count, uint64_t collected, bool optional) {
        int leftover_active = 0;
        for (int i = 0; i < unit_count; i++) {
            L2SwimlaneAicpuTaskPool *state = get_state(i);
            uint64_t buf_ptr = state->head.current_buf_ptr;
            if (buf_ptr == 0) continue;
            void *host_ptr = manager_.resolve_host_ptr(reinterpret_cast<void *>(buf_ptr));
            if (host_ptr == nullptr) continue;
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
                "L2Swimlane reconcile: %lu %s records dropped on device side (buffer full / "
                "ready_queue full).",
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
        total_perf_collected_, /*optional=*/false
    );

    reconcile_one(
        "SCHED_PHASE", "thread", PLATFORM_MAX_AICPU_THREADS,
        [this](int thread_index) {
            return get_sched_phase_buffer_state(shm_host_, num_aicore_, thread_index);
        },
        [](void *host_ptr) {
            return reinterpret_cast<L2SwimlaneAicpuSchedPhaseBuffer *>(host_ptr)->count;
        },
        total_sched_phase_collected_, /*optional=*/true
    );

    reconcile_one(
        "ORCH_PHASE", "thread", PLATFORM_MAX_AICPU_THREADS,
        [this](int thread_index) {
            return get_orch_phase_buffer_state(shm_host_, num_aicore_, thread_index);
        },
        [](void *host_ptr) {
            return reinterpret_cast<L2SwimlaneAicpuOrchPhaseBuffer *>(host_ptr)->count;
        },
        total_orch_phase_collected_, /*optional=*/true
    );
}

void L2SwimlaneCollector::read_phase_header_metadata() {
    if (shm_host_ == nullptr) {
        return;
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

// AICore-as-producer post-processing: build a per-core
// `reg_task_id → (task_token_raw, start, end)` lookup so the export pass can
// pull identity + AICore timing for each AICPU record by its join key.
//
// AICore rotation guarantees each per-core stream is a complete prefix of
// "all dispatched tasks on this core" with no wrap loss (the AICore buffer
// pool is recycled via free_queue while the session runs, so arbitrarily
// long sessions work). Map keying tolerates AICPU-side drops — a missing
// AICPU record just produces fewer emitted entries, not a stream desync.
//
// Why reg_task_id and not task_token_raw: SPMD with `block_num > num_cores`
// (e.g. q_proj in qwen3 decode has block_num=40 over 24 AIC cores)
// dispatches the same task_token_raw multiple times to the same core. Each
// dispatch produces its own AICore record (different start/end) and its own
// AICPU record, but all share the same task_token_raw. A task_token_raw-keyed
// map would collapse all dispatches into the last AICore record's timing,
// making every AICPU record on that core show identical (start, end) —
// visible as "duplicate" task spans in the viewer. reg_task_id is per-core
// monotonic per dispatch (scheduler_dispatch.cpp:140) — unique at dispatch
// granularity, which is what the join needs.
std::vector<L2SwimlaneCollector::AicoreLookup> L2SwimlaneCollector::build_aicore_lookup() const {
    std::vector<AicoreLookup> per_core(num_aicore_);
    for (int core_idx = 0; core_idx < num_aicore_; core_idx++) {
        const auto &ac_stream = collected_aicore_records_[core_idx];
        per_core[core_idx].reserve(ac_stream.size() * 2);
        for (const auto &r : ac_stream) {
            per_core[core_idx][r.reg_task_id] = {r.task_token_raw, r.start_time, r.end_time};
        }
    }
    return per_core;
}

void L2SwimlaneCollector::set_core_types(const CoreType *types, int n) {
    if (types == nullptr || n <= 0) {
        core_types_.clear();
        return;
    }
    core_types_.assign(types, types + n);
}

int L2SwimlaneCollector::export_swimlane_json() {
    // shm_host_ is read once (phase-header metadata) below; guard it up front
    // like the other collector methods so a never-initialized / post-finalize
    // call returns instead of dereferencing null.
    if (shm_host_ == nullptr) {
        return -1;
    }

    // Step 0: Build per-core reg_task_id → AICore-info lookup. Used by the
    // emit loop to pull identity (task_token_raw) and AICore timing
    // (start_time, end_time) for each AICPU record. AICore is the producer
    // of those fields; AICPU only writes (reg_task_id, dispatch_time,
    // finish_time).
    auto aicore_lookup = build_aicore_lookup();

    // Step 1: Validate collected data. AICPU records (collected_perf_records_)
    // are required at AICPU_TIMING+ but absent at AICORE_TIMING (level=1)
    // where complete_task is bypassed. At level=1 the AICore stream alone is
    // the source of truth; check it too before declaring nothing to export.
    bool has_any_records = false;
    for (const auto &core_records : collected_perf_records_) {
        if (!core_records.empty()) {
            has_any_records = true;
            break;
        }
    }
    if (!has_any_records && l2_swimlane_level_ == L2SwimlaneLevel::AICORE_TIMING) {
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

    // Step 2: Create output directory (recursively — parent `outputs/` may not
    // yet exist on a clean checkout / standalone run). `output_prefix_` was
    // captured at initialize() time.
    std::error_code ec;
    std::filesystem::create_directories(output_prefix_, ec);
    if (ec) {
        LOG_ERROR("Error: Failed to create output directory %s: %s", output_prefix_.c_str(), ec.message().c_str());
        return -1;
    }

    // Step 3: Flatten per-core AICPU records and resolve identity/timing
    // from the AICore lookup. Records whose reg_task_id has no matching
    // AICore entry are dropped here (AICore-side rotation dropped the
    // buffer; surfaced via dropped_record_count and the unmatched log below).
    struct TaggedRecord {
        const L2SwimlaneAicpuTaskRecord *record;
        const AicoreInfo *aicore;  // identity + AICore timing
        uint32_t core_id;
    };
    std::vector<TaggedRecord> tagged_records;
    size_t total_records = 0;
    for (const auto &core_records : collected_perf_records_) {
        total_records += core_records.size();
    }
    tagged_records.reserve(total_records);
    size_t total_unmatched = 0;
    for (size_t core_idx = 0; core_idx < collected_perf_records_.size(); core_idx++) {
        const auto &lookup = aicore_lookup[core_idx];
        size_t core_unmatched = 0;
        for (const auto &record : collected_perf_records_[core_idx]) {
            auto it = lookup.find(record.reg_task_id);
            if (it == lookup.end()) {
                core_unmatched++;
                continue;
            }
            tagged_records.push_back({&record, &it->second, static_cast<uint32_t>(core_idx)});
        }
        if (core_unmatched > 0) {
            LOG_WARN(
                "Core %zu: %zu L2SwimlaneAicpuTaskRecord(s) had no matching AICore entry "
                "(AICore buffer drops on rotation? PLATFORM_AICORE_BUFFERS_PER_CORE=%d may "
                "be undersized for host drain rate)",
                core_idx, core_unmatched, PLATFORM_AICORE_BUFFERS_PER_CORE
            );
        }
        total_unmatched += core_unmatched;
    }
    if (total_unmatched > 0) {
        LOG_WARN("Dropped %zu task record(s) with unmatched AICore timing from swimlane export", total_unmatched);
    }

    // Sort by canonical task_id (64-bit PTO2 raw) via the joined AICore entry.
    std::sort(tagged_records.begin(), tagged_records.end(), [](const TaggedRecord &a, const TaggedRecord &b) {
        return a.aicore->task_token_raw < b.aicore->task_token_raw;
    });

    // Step 4: Calculate base time (minimum timestamp across all records).
    uint64_t base_time_cycles = UINT64_MAX;
    for (const auto &tagged : tagged_records) {
        if (tagged.aicore->start_time > 0 && tagged.aicore->start_time < base_time_cycles) {
            base_time_cycles = tagged.aicore->start_time;
        }
        if (tagged.record->dispatch_time > 0 && tagged.record->dispatch_time < base_time_cycles) {
            base_time_cycles = tagged.record->dispatch_time;
        }
    }

    // Include phase record timestamps (sched + orch) in base_time calculation
    if (has_phase_data_) {
        for (const auto &thread_records : collected_sched_phase_records_) {
            for (const auto &pr : thread_records) {
                if (pr.start_time > 0 && pr.start_time < base_time_cycles) {
                    base_time_cycles = pr.start_time;
                }
            }
        }
        for (const auto &thread_records : collected_orch_phase_records_) {
            for (const auto &pr : thread_records) {
                if (pr.start_time > 0 && pr.start_time < base_time_cycles) {
                    base_time_cycles = pr.start_time;
                }
            }
        }
    }

    // AICORE_TIMING (level=1): tagged_records is empty (complete_task bypassed)
    // and phase records are also empty (level < SCHED_PHASES), so the two
    // loops above leave base_time_cycles == UINT64_MAX. The AICore-records-only
    // emit path below would then subtract UINT64_MAX from each timestamp and
    // produce decades-large microsecond values. Scan AICore records here as
    // the timestamp anchor in that mode.
    if (l2_swimlane_level_ == L2SwimlaneLevel::AICORE_TIMING) {
        for (const auto &ac_records : collected_aicore_records_) {
            for (const auto &r : ac_records) {
                if (r.start_time > 0 && r.start_time < base_time_cycles) {
                    base_time_cycles = r.start_time;
                }
            }
        }
    }

    // Step 5: Compose output path. Filename is fixed (no timestamp) — the
    // caller-provided directory is the per-task uniqueness boundary.
    std::string filepath = output_prefix_ + "/l2_swimlane_records.json";

    // Step 6: Open JSON file for writing
    std::ofstream outfile(filepath);
    if (!outfile.is_open()) {
        LOG_ERROR("Error: Failed to open file: %s", filepath.c_str());
        return -1;
    }

    // Step 7: Write JSON data
    // Fanout fields are emitted as empty/zero — the device-side hot path no
    // longer carries them. Downstream (swimlane_converter.py) joins fanout
    // from the sibling deps.json (dep_gen output).
    int l2_swimlane_level = static_cast<int>(l2_swimlane_level_);
    outfile << "{\n";
    outfile << "  \"l2_swimlane_level\": " << l2_swimlane_level << ",\n";
    outfile << "  \"tasks\": [\n";

    // tagged_records was already filtered to entries with a matching AICore
    // join target in Step 3; unmatched ones were logged and dropped there.
    // No further filter pass needed — emit straight from tagged_records.
    // core_type per record comes from the host-published core_types_ table
    // (set_core_types() at init); func_id is emitted as -1 and resolved
    // post-process by `swimlane_converter.py` from deps.json's `kernel_ids[]`
    // — same path AICORE_TIMING (level=1) uses, so all levels go through the
    // same identity-resolution machinery.
    for (size_t e = 0; e < tagged_records.size(); ++e) {
        const auto &tagged = tagged_records[e];
        const auto &record = *tagged.record;
        const auto &ac = *tagged.aicore;

        // Convert times to microseconds
        double start_us = cycles_to_us(ac.start_time - base_time_cycles);
        double end_us = cycles_to_us(ac.end_time - base_time_cycles);
        double duration_us = end_us - start_us;
        double dispatch_us = (record.dispatch_time > 0) ? cycles_to_us(record.dispatch_time - base_time_cycles) : 0.0;
        double finish_us = (record.finish_time > 0) ? cycles_to_us(record.finish_time - base_time_cycles) : 0.0;

        CoreType ct = (tagged.core_id < core_types_.size()) ? core_types_[tagged.core_id] : CoreType::AIV;
        const char *core_type_str = (ct == CoreType::AIC) ? "aic" : "aiv";

        outfile << "    {\n";
        outfile << "      \"task_id\": " << ac.task_token_raw << ",\n";
        outfile << "      \"func_id\": -1,\n";
        outfile << "      \"core_id\": " << tagged.core_id << ",\n";
        outfile << "      \"core_type\": \"" << core_type_str << "\",\n";
        outfile << "      \"ring_id\": " << static_cast<int>(ac.task_token_raw >> 32) << ",\n";
        outfile << "      \"start_time_us\": " << std::fixed << std::setprecision(3) << start_us << ",\n";
        outfile << "      \"end_time_us\": " << std::fixed << std::setprecision(3) << end_us << ",\n";
        outfile << "      \"duration_us\": " << std::fixed << std::setprecision(3) << duration_us << ",\n";
        outfile << "      \"dispatch_time_us\": " << std::fixed << std::setprecision(3) << dispatch_us << ",\n";
        outfile << "      \"finish_time_us\": " << std::fixed << std::setprecision(3) << finish_us << "\n";
        // Fanout is no longer carried on the device hot path — dep_gen replay
        // (deps.json) is the sole source of truth, joined in by tooling.
        outfile << "    }";
        if (e + 1 < tagged_records.size()) {
            outfile << ",";
        }
        outfile << "\n";
    }

    // AICORE_TIMING (level=1) fallback: at this level complete_task is
    // bypassed and tagged_records is empty. The AICore record stream alone
    // carries identity + timing, so synthesize one task[] entry per AICore
    // record. Same identity-resolution rules as the level≥2 emit above
    // (core_type from core_types_, func_id = -1 → python join).
    if (tagged_records.empty() && l2_swimlane_level_ == L2SwimlaneLevel::AICORE_TIMING) {
        size_t aicore_emitted = 0;
        for (int core_id = 0; core_id < num_aicore_; core_id++) {
            for (const auto &r : collected_aicore_records_[core_id]) {
                if (r.start_time == 0) continue;
                double start_us = cycles_to_us(r.start_time - base_time_cycles);
                double end_us = cycles_to_us(r.end_time - base_time_cycles);
                double duration_us = end_us - start_us;
                CoreType ct = (core_id < static_cast<int>(core_types_.size())) ? core_types_[core_id] : CoreType::AIV;
                const char *core_type_str = (ct == CoreType::AIC) ? "aic" : "aiv";
                if (aicore_emitted > 0) outfile << ",\n";
                outfile << "    {\n";
                outfile << "      \"task_id\": " << r.task_token_raw << ",\n";
                outfile << "      \"func_id\": -1,\n";
                outfile << "      \"core_id\": " << core_id << ",\n";
                outfile << "      \"core_type\": \"" << core_type_str << "\",\n";
                outfile << "      \"ring_id\": " << static_cast<int>(r.task_token_raw >> 32) << ",\n";
                outfile << "      \"start_time_us\": " << std::fixed << std::setprecision(3) << start_us << ",\n";
                outfile << "      \"end_time_us\": " << std::fixed << std::setprecision(3) << end_us << ",\n";
                outfile << "      \"duration_us\": " << std::fixed << std::setprecision(3) << duration_us << ",\n";
                outfile << "      \"dispatch_time_us\": 0.000,\n";
                outfile << "      \"finish_time_us\": 0.000\n";
                outfile << "    }";
                aicore_emitted++;
            }
        }
        if (aicore_emitted > 0) outfile << "\n";
    }
    outfile << "  ]";

    // Step 8: Write phase profiling data (level >= 3)
    if (l2_swimlane_level_ >= L2SwimlaneLevel::SCHED_PHASES) {
        auto sched_phase_name = [](L2SwimlaneSchedPhaseKind kind) -> const char * {
            switch (kind) {
            case L2SwimlaneSchedPhaseKind::Complete:
                return "complete";
            case L2SwimlaneSchedPhaseKind::Dispatch:
                return "dispatch";
            }
            return "unknown";
        };

        // AICPU scheduler phases — one nested array per sched-phase thread.
        outfile << ",\n  \"aicpu_scheduler_phases\": [\n";
        for (size_t t = 0; t < collected_sched_phase_records_.size(); t++) {
            outfile << "    [\n";
            bool first = true;
            for (const auto &pr : collected_sched_phase_records_[t]) {
                double start_us = cycles_to_us(pr.start_time - base_time_cycles);
                double end_us = cycles_to_us(pr.end_time - base_time_cycles);
                if (!first) outfile << ",\n";
                outfile << "      {\"start_time_us\": " << std::fixed << std::setprecision(3) << start_us
                        << ", \"end_time_us\": " << std::fixed << std::setprecision(3) << end_us << ", \"phase\": \""
                        << sched_phase_name(pr.kind) << "\""
                        << ", \"loop_iter\": " << pr.loop_iter << ", \"tasks_processed\": " << pr.tasks_processed;
                // Dispatch-only deltas. Complete records carry zeros — omit
                // them to keep the JSON terse per record.
                if (pr.kind == L2SwimlaneSchedPhaseKind::Dispatch) {
                    outfile << ", \"pop_hit\": " << pr.pop_hit << ", \"pop_miss\": " << pr.pop_miss;
                }
                outfile << "}";
                first = false;
            }
            if (!first) outfile << "\n";
            outfile << "    ]";
            if (t < collected_sched_phase_records_.size() - 1) outfile << ",";
            outfile << "\n";
        }
        outfile << "  ]";

        // Per-task orchestrator phase records (level >= 4).
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
            // Orch is a single instance (pool ordinal 0): emit only the actual
            // orch lane count (num_orch_phase_threads), not the full MAX-sized
            // vector, so the trace shows one orchestrator lane with no empties.
            size_t orch_lanes = static_cast<size_t>(get_l2_swimlane_header(shm_host_)->num_orch_phase_threads);
            if (orch_lanes == 0 || orch_lanes > collected_orch_phase_records_.size()) {
                orch_lanes = collected_orch_phase_records_.size();
            }
            outfile << ",\n  \"aicpu_orchestrator_phases\": [\n";
            for (size_t t = 0; t < orch_lanes; t++) {
                outfile << "    [\n";
                bool first = true;
                for (const auto &pr : collected_orch_phase_records_[t]) {
                    double start_us = cycles_to_us(pr.start_time - base_time_cycles);
                    double end_us = cycles_to_us(pr.end_time - base_time_cycles);
                    if (!first) outfile << ",\n";
                    // "phase" key is the only orch event today (ORCH_SUBMIT) —
                    // hard-coded since the record type tag is "orch".
                    outfile << "      {\"phase\": \"orch_submit\""
                            << ", \"start_time_us\": " << std::fixed << std::setprecision(3) << start_us
                            << ", \"end_time_us\": " << std::fixed << std::setprecision(3) << end_us
                            << ", \"submit_idx\": " << pr.submit_idx << ", \"task_id\": " << pr.task_id << "}";
                    first = false;
                }
                if (!first) outfile << "\n";
                outfile << "    ]";
                if (t < orch_lanes - 1) outfile << ",";
                outfile << "\n";
            }
            outfile << "  ]";
        }
    }

    // Core-to-thread mapping
    if (!core_to_thread_.empty()) {
        outfile << ",\n  \"core_to_thread\": [";
        for (size_t i = 0; i < core_to_thread_.size(); i++) {
            outfile << static_cast<int>(core_to_thread_[i]);
            if (i < core_to_thread_.size() - 1) outfile << ", ";
        }
        outfile << "]";
    }

    outfile << "\n}\n";

    // Step 9: Close file
    outfile.close();

    uint32_t record_count = static_cast<uint32_t>(tagged_records.size());
    LOG_INFO_V0("=== JSON Export Complete ===");
    LOG_INFO_V0("File: %s", filepath.c_str());
    LOG_INFO_V0("Records: %u", record_count);

    return 0;
}

int L2SwimlaneCollector::finalize(L2SwimlaneUnregisterCallback unregister_cb, const L2SwimlaneFreeCallback &free_cb) {
    if (shm_host_ == nullptr) {
        return 0;
    }

    // Stop mgmt + collector threads if the caller didn't already (idempotent).
    stop();

    LOG_DEBUG("Cleaning up performance profiling resources");

    // Every release site below goes through release_one_buffer so the
    // unregister and free are an inseparable pair — each dev_ptr that
    // alloc_single_buffer installed via halHostRegister is unregistered
    // before its device memory is freed. Without this the Ascend HAL's
    // per-device registration table accumulates leaked entries across
    // init_l2_swimlane() invocations and back-to-back l2_swimlane tests on
    // a reused Worker fail at rc=8 from halHostRegister.

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
    // shm_host_ aliases freed device/host memory now; null it so is_initialized()
    // reports false, the dtor's "destroyed without finalize()" warning stays
    // quiet, and a re-entrant finalize() / re-init hits the early-out instead of
    // walking freed buffer state. Mirrors PMU/DepGen/TensorDump collectors.
    shm_host_ = nullptr;
    collected_perf_records_.clear();
    collected_sched_phase_records_.clear();
    collected_orch_phase_records_.clear();
    core_to_thread_.clear();
    has_phase_data_ = false;
    total_perf_collected_ = 0;
    total_sched_phase_collected_ = 0;
    total_orch_phase_collected_ = 0;
    clear_memory_context();

    LOG_DEBUG("Performance profiling cleanup complete");
    return 0;
}
